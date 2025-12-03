import glob
import os
import random

import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, IntTensor, LongTensor
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,
                    BIO_NO_LBL, BIO_START_LBL, BIO_INTERM_LBL, 
                    TOK_START_CHAR, 
                    match_substrings, 
                    ASSISTANT_START_TOK,
                    )
from .vqa_dataset import VQADataset
from dataclasses import dataclass, asdict
from .device_shiftable import DeviceShiftable
from .explanatory_seg_dataset import ConversationType, AnswerType, ExplanatorySegInstance, ConversationQuestionType
from .question_type import QuestionType
from .explanatory_seg_datasets_adapter import ExplanatorySegDatasetsAdapter
from itertools import chain

@dataclass
class ExplanatorySegBatch(DeviceShiftable):
    img_paths: list[str] = None
    img_labels: list[str] = None

    sam_img_inputs: Tensor = None
    clip_img_inputs: Tensor = None

    input_ids: IntTensor = None # Shape (n_conversations_in_batch, seq_len)
    attention_masks: IntTensor = None # Shape (n_conversations_in_batch, seq_len)

    mask_dicts: list[dict[str, Tensor]] = None
    masks: list[Tensor] = None

    label_masks: list[IntTensor] = None
    resized_img_dims: list[tuple[int,int]] = None

    img_to_conversations_offsets: LongTensor = None # Start and end indices of conversations for each image index; (n_images_in_batch + 1,)
    is_inference: bool = None

    questions: list[list[str]] = None
    question_types: list[QuestionType] = None

    conversations: list[list[str]] = None # List of conversations for each image in the batch
    conversation_types: list[list[ConversationType]] = None
    conversation_question_types: list[list[ConversationQuestionType]] = None

    part_answer_choices: list[list[str]] = None
    part_answer_types: list[list[AnswerType]] = None
    answer_parts: list[list[list[str]]] = None # The parts in each answer choice

    object_answer_choices: list[list[str]] = None
    object_answer_types: list[list[AnswerType]] = None
    answer_objects: list[list[str]] = None # The class of the object in each answer choice
    
    per_token_labels: LongTensor = None
    mask_positions_in_input_ids: list[list[int]] = None

    def to_dict(self):
        return asdict(self)


def collate_fn(
    batch: list[ExplanatorySegInstance],
    tokenizer=None,
    conv_type="llava_v1",
    use_mm_start_end=True,
    local_rank=-1
):

    # --- Construct batch fields ---
    img_paths = [instance.img_path for instance in batch]
    img_labels = [instance.img_label for instance in batch]

    sam_img_inputs = [instance.sam_img_input for instance in batch]
    clip_img_inputs = [instance.clip_img_input for instance in batch]

    mask_dicts = [instance.mask_dicts for instance in batch]
    masks = [instance.masks for instance in batch]

    label_masks = [instance.label_mask for instance in batch]
    resized_img_sizes = [instance.resized_img_dims for instance in batch]

    inferences = [instance.is_inference for instance in batch]
    inference = inferences[0]

    questions = [instance.questions for instance in batch]
    question_types = [instance.question_type for instance in batch]

    conversations = list(chain.from_iterable([instance.conversations for instance in batch])) # This needs to be reconstructed into list[list[str]] form after processing
    conversation_types = [instance.conversation_types for instance in batch]
    conversation_question_types = [instance.conversation_question_types for instance in batch]
    offsets = torch.cumsum(torch.tensor([0] + [len(instance.conversations) for instance in batch], dtype=torch.long), dim=0)
    
    part_answer_choices = [instance.part_answer_choices for instance in batch]
    part_answer_types = [instance.part_answer_types for instance in batch]
    answer_parts = [instance.answer_parts for instance in batch]

    object_answer_choices = [instance.object_answer_choices for instance in batch]
    object_answer_types = [instance.object_answer_types for instance in batch]
    answer_objects = [instance.answer_objects for instance in batch]

    # --- Tokenize Conversations ---
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    
    # # NOTE: DEBUG - replace IMAGE_TOKEN and IGNORE_INDEX with the tokenizer's pad_token
    # input_ids_fixed = torch.where(
    #     (input_ids == IMAGE_TOKEN_INDEX) | (input_ids == IGNORE_INDEX),
    #     torch.tensor(tokenizer.pad_token_id, device=input_ids.device),
    #     input_ids
    # )
    # targets_fixed = torch.where(
    #     (targets == IMAGE_TOKEN_INDEX) | (targets == IGNORE_INDEX),
    #     torch.tensor(tokenizer.pad_token_id, device=targets.device),
    #     targets
    # )
    # ####### DEBUG #######
    
    # breakpoint() # XXX ####

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        if conv.sep2 not in conversation: # Intentional open-ended generation; no need to construct targets (whose code will break)
            continue

        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":  # Break if the round is empty (i.e., one user query and one assistant response)
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            try:
                assert cur_len == total_len
            except Exception as e:
                breakpoint()

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            
    # Reorganize conversations to match conversation_types format of list[list[str]] (from flattened list[str])
    # reorg_conversations = []
    # conv_offset = 0
    # for conv_type_list in conversation_types:
    #     n_convs_in_instance = len(conv_type_list)
    #     reorg_conversations.append(conversations[conv_offset:conv_offset+n_convs_in_instance])
    #     conv_offset += n_convs_in_instance

    # conversations = reorg_conversations

    # construct the BIO tag scheme label here following the input_ids construction
    per_token_labels_list = []   # will be a list of lists (one per conversation)
    mask_positions_in_input_ids = []  # list of starting token indices for each conversation
    # find the text span within the tokenized input texts that corresponds to the part_text span (this is for embedding sampling for SAM later on)
    BIO_NO_LBL = 0
    BIO_START_LBL = 1
    BIO_INTERM_LBL = 2
    
    # len(answer_parts[0]) == len(conversations)  # answer_parts has the part texts that can be used to index mask_dicts

    # construct 'per_token_labels_dict_list': a list of dictionaries, where each dictionary contains the part_text and the corresponding mask_dict
    # len(per_token_labels_dict_list) == len(conversations)
    per_token_label_dict_list = []
    for part_texts in answer_parts[0]:
        part_to_mask = {}
        for pt in part_texts:
            if pt in mask_dicts[0]:
                part_to_mask[pt] = mask_dicts[0][pt]
        per_token_label_dict_list.append(part_to_mask)
    
    for batch_idx in range(len(offsets) - 1):
        start = offsets[batch_idx]
        end = offsets[batch_idx + 1]
        for conv_idx in range(start, end):
            conversation = conversations[conv_idx]
            per_token_labels = [BIO_NO_LBL] * input_ids.size(1)
            mask_positions = set()

            # NOTE: Hacky fix to handle the case where the mask_dicts != conversations
            if len(per_token_label_dict_list) < len(conversations):  # fill it with a dummy dict
                for _ in range(len(conversations) - len(per_token_label_dict_list)):
                    per_token_label_dict_list.append({})
            # TODO: Resolve error later
            try:
                part_to_mask = per_token_label_dict_list[conv_idx]
            except Exception as e:
                part_to_mask = {}
                print("="*20)
                print(e)
                print(">> len(conversations): ", len(conversations))
                print(">> mask_dicts: ", len(mask_dicts))
                print(">> offsets: ", offsets)
                print(">> conv_idx: ", conv_idx)
                print(">> TERMINATING >>")

            ASSISTANT_START_TOK = 'ASSISTANT:'
            TOK_START_CHAR = '▁'
            if len(part_to_mask) != 0:
                conversation_input_ids = input_ids[conv_idx].clone()  # shape [seq_len]
                seq_len = conversation_input_ids.size(0)
                assistant_start_tok_ids = tokenizer(ASSISTANT_START_TOK)['input_ids'][1:]
                conversation_input_ids = torch.where(
                    (conversation_input_ids == IMAGE_TOKEN_INDEX) | (conversation_input_ids == IGNORE_INDEX),
                    torch.tensor(tokenizer.pad_token_id, device=conversation_input_ids.device),
                    conversation_input_ids,
                )
                input_ids_tokens = tokenizer.convert_ids_to_tokens(conversation_input_ids.tolist())
                # If part_text falls before "ASSISTANT:", then it's not a match
                for assistant_start_tok_idx in range(len(conversation_input_ids) - len(assistant_start_tok_ids) + 1):
                    sub_str_ids = conversation_input_ids[assistant_start_tok_idx : assistant_start_tok_idx + len(assistant_start_tok_ids)].tolist()
                    assistant_str_ids = assistant_start_tok_ids
                    if sub_str_ids == assistant_str_ids:
                        break
                    
                # for part_text in part_to_mask.keys():
                for part_text in sorted(part_to_mask, key=len, reverse=True):
                    part_token_ids = tokenizer(part_text)['input_ids'][1:]  # skip the <s> token
                    part_token_ids_tokens = tokenizer.convert_ids_to_tokens(part_token_ids)
                    substr_found = False
                    for j in range(len(input_ids_tokens) - len(part_token_ids_tokens) + 1):
                        sub_str = input_ids_tokens[j : j + len(part_token_ids_tokens)]
                        part_tok = part_token_ids_tokens
                        if sub_str == part_tok:
                            if j <= assistant_start_tok_idx:  # do not include any match that falls before <image> token
                                continue
                            elif TOK_START_CHAR not in input_ids_tokens[j + len(part_token_ids_tokens)] and \
                                input_ids_tokens[j + len(part_token_ids_tokens)][0].isalnum():  # check if part_text is a whole word and not a subtoken
                                continue
                            elif per_token_labels[j] != BIO_NO_LBL:  # skip if the part_text is already matched to a previous span
                                continue
                            per_token_labels[j] = BIO_START_LBL
                            for k in range(1, len(part_tok)):
                                per_token_labels[j + k] = BIO_INTERM_LBL
                            mask_positions.add(j)
                            substr_found = True
                            # print(f"(explanatory_dataset.py) >> {' '.join(input_ids_tokens[j : j + len(part_token_ids)])} | {per_token_labels[j : j + len(part_token_ids)]}")

                    if not substr_found:
                        pass
                        # print(f">> Warning: part_text '{part_text}' not found in the tokenized conversation.")
                        # print(f">> conversation: {conversation}")

            per_token_labels_list.append(torch.tensor(per_token_labels, dtype=torch.long))
            mask_positions_in_input_ids.append(sorted(list(mask_positions)))
    
    # just pad to the same dim as input_ids
    max_len = input_ids.size(1)
    padded_per_token_labels = []
    for labels in per_token_labels_list:
        if labels.size(0) < max_len:
            pad_size = max_len - labels.size(0)
            labels = torch.cat([labels, torch.zeros(pad_size, dtype=torch.long)], dim=0)
        else:
            labels = labels[:max_len]
        padded_per_token_labels.append(labels.unsqueeze(0))
    per_token_labels_tensor = torch.cat(padded_per_token_labels, dim=0)  # shape: [num_convs(across the batches), seq_len]
    
    # print("(dataset.py) >> per_token_labels_tensor: ", per_token_labels_tensor)
    # print("(dataset.py) >> mask_positions_in_input_ids: ", mask_positions_in_input_ids)
    # print("==" * 20)

    # Reorganize conversations to match conversation_types format of list[list[str]] (from flattened list[str])
    reorg_conversations = []
    conv_offset = 0
    for conv_type_list in conversation_types:
        n_convs_in_instance = len(conv_type_list)
        reorg_conversations.append(conversations[conv_offset:conv_offset+n_convs_in_instance])
        conv_offset += n_convs_in_instance

    conversations = reorg_conversations

    return ExplanatorySegBatch(
        img_paths=img_paths,
        img_labels=img_labels,
        sam_img_inputs=torch.stack(sam_img_inputs, dim=0),
        clip_img_inputs=torch.stack(clip_img_inputs, dim=0),
        input_ids=input_ids,
        attention_masks=attention_masks,
        mask_dicts=mask_dicts,
        masks=masks,
        label_masks=label_masks,
        resized_img_dims=resized_img_sizes,
        img_to_conversations_offsets=offsets,
        is_inference=inference,
        questions=questions,
        question_types=question_types,
        conversations=conversations,
        conversation_types=conversation_types,
        conversation_question_types=conversation_question_types,
        part_answer_choices=part_answer_choices,
        part_answer_types=part_answer_types,
        answer_parts=answer_parts,
        object_answer_choices=object_answer_choices,
        object_answer_types=object_answer_types,
        answer_objects=answer_objects,
        per_token_labels=per_token_labels_tensor,
        mask_positions_in_input_ids=mask_positions_in_input_ids
    )
    

class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg||explanatory_seg",
        sample_rate=[9, 3, 3, 1, 9],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory_seg_datasets: ExplanatorySegDatasetsAdapter = None,
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

        if explanatory_seg_datasets:
            self.all_datasets.append(explanatory_seg_datasets)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], " ")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], " ") # NOTE: Why does LISA pass [SEG] in as input during validation?
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True
        
        per_token_label_dict = [{k: v} for k, v in zip(sampled_sents, masks)]

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            per_token_label_dict,
            inference,
        )

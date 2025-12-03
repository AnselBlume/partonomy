import glob
import os
import random

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .explanatory_seg_datasets_adapter import ExplanatorySegDatasetsAdapter
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,
                    BIO_NO_LBL, BIO_START_LBL, BIO_INTERM_LBL, 
                    TOK_START_CHAR, 
                    match_substrings, 
                    ASSISTANT_START_TOK,
                    )
from .vqa_dataset import VQADataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    per_token_label_dict_list = []
    inferences = []
    dataset_names = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        per_token_label_dict,  # stores the concept label to mask (e.g., [{'airplane engine': mask, 'airplane wing': mask}])
        inference,
        dataset_name,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations) # NOTE: this will create sum(number of conversations over batches) - need 'offset_list' to keep track of the start index of each batch
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)  # e.g., len(conversations[0]) == 4, len(conversations[1]) == 2, offset_list = [0, 4, 6]
        per_token_label_dict_list.append(per_token_label_dict)  # len(per_token_label_dict_list) == bsz
        inferences.append(inference)
        dataset_names.append(dataset_name)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    # input_ids.shape = [num_conversations, max_len]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
        
    if (dataset_names[0] in ["textvqa", "gqa", "pope"] and inferences[0] == True):  # handle TextVQA validation case
        targets = sampled_classes_list[0]
    else:
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
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
                assert cur_len == total_len
    
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255  # hack for adding 255 image patch tokens (max_len = 512)

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            
    # construct the BIO tag scheme label here following the input_ids construction
    matched_substrings_list = []
    per_token_labels_list = []   # will be a list of lists (one per conversation)
    mask_positions_in_input_ids = []  # list of starting token indices for each conversation

    # find the text span within the tokenized input texts that corresponds to the part_text span (this is for embedding sampling for SAM later on)
    for batch_idx in range(len(offset_list) - 1):
        start = offset_list[batch_idx]
        end = offset_list[batch_idx + 1]
        # NOTE: len(part_to_mask_list) is the number of conversations
        # each dict (e.g., part_to_mask_list[i]) contains part_text as key and the corresponding mask as value
        # part_to_mask_list[i] can have multiple parts (for 'explanatory_seg'), or single part (for 'reason_seg', 'sem_seg', 'refer_seg')
        part_to_mask_list = per_token_label_dict_list[batch_idx]  # e.g., [{'engine': mask, 'wing': mask}]
        
        for conv_idx in range(start, end):
            part_mask_idx = conv_idx - start
            part_to_mask = part_to_mask_list[part_mask_idx]
            
            conversation = conversation_list[conv_idx]
            per_token_labels = [BIO_NO_LBL] * input_ids.size(1)
            mask_positions = set()
            
            if len(part_to_mask_list) != len(conversation_list[start:end]):
                print(f">> Warning: len(part_to_mask_list) != len(conversation_list[start:end])")
                print(f">> part_to_mask_list: {part_to_mask_list}")
                print(f">> conversation_list[start:end]: {conversation_list[start:end]}")
                exit()
            
            if len(part_to_mask) != 0:
                conversation_input_ids = input_ids[conv_idx].clone()  # shape [seq_len]
                conversation_input_ids = torch.where(
                    (conversation_input_ids == IMAGE_TOKEN_INDEX) | (conversation_input_ids == IGNORE_INDEX),
                    torch.tensor(tokenizer.pad_token_id, device=conversation_input_ids.device),
                    conversation_input_ids,
                )
                # Find the part_text spans in the conversation input_ids
                matched_substrings, per_token_labels, mask_positions = match_substrings(
                    conversation_input_ids,
                    [tokenizer(part_text)['input_ids'][1:] for part_text in list(part_to_mask.keys())],
                    tokenizer,
                    ASSISTANT_START_TOK
                )
            matched_substrings_list.append(matched_substrings)
            per_token_labels_list.append(torch.tensor(per_token_labels, dtype=torch.long))
            mask_positions_in_input_ids.append(list(mask_positions))
    
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
    
    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "per_token_labels": per_token_labels_tensor,  # BIO ids per token (same dim as input_ids)
        "mask_positions_in_input_ids": mask_positions_in_input_ids,  # list of lists of starting token indices per sample
    }


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
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 5, 5, 1],
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
        return len(self.all_datasets[0]) if self.dataset == 'explanatory_seg' else self.samples_per_epoch

    def __getitem__(self, idx):
        if len(self.all_datasets) == 1:
            ind = 0
        else:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        # return *data[0], inference, self.datasets[ind]
        return *data[idx], inference, self.datasets[ind]


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
            self.reformulated_text_path = os.path.join(base_image_dir, "reason_seg", ds, split, "val_reformulated_responses.json")
            with open(self.reformulated_text_path, "r") as f:
                self.reformulated_text_dict = json.load(f)
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
            self.reformulated_text_dict = None

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
            json_id = os.path.basename(json_path).replace(".json", "")
            response_texts = self.reformulated_text_dict[json_id]
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        chosen_response_texts = []
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            print(">> (ValDataset) sampled_sents: ", text)
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please explain why.".format(text),
                )
                # conv.append_message(conv.roles[1], " ")
                chosen_response = random.choice(response_texts)
                conv.append_message(conv.roles[1], chosen_response)
                chosen_response_texts.append(chosen_response)
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please explain why.".format(
                        text
                    ),
                )
                # conv.append_message(conv.roles[1], " ")
                chosen_response = random.choice(response_texts)
                conv.append_message(conv.roles[1], chosen_response)
                chosen_response_texts.append(chosen_response)
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
        
        sampled_sents = chosen_response_texts if self.data_type == "reason_seg" else sampled_sents
        
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
            'reason_seg'
        )

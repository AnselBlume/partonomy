# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import copy
from dataclasses import dataclass, field
import logging
import pathlib
from pprint import pformat
from typing import Dict, Optional, Sequence, List
import orjson

import cv2
import numpy as np
import torch
from tqdm import tqdm

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_IM_GEN_TOKEN,DEFAULT_AUDIO_GEN_TOKEN,DEFAULT_MSK_TOKEN,DEFAULT_IM_GEN_START_TOKEN,DEFAULT_AUDIO_GEN_START_TOKEN,DEFAULT_AUDIO_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import torch.distributed as dist
from llava.train.seg_register.register_dataset import Register as COCORegister
from explanatory_seg_dataset import ExplanatorySegDataset, ExplanatorySegInstance
from llava.train.train import *
from accelerate import Accelerator
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from question_type import QuestionType
from explanatory_seg_dataset_adapter import ExplanatorySegLazySupervisedDataset
from train_eseg import DataCollatorForSupervisedDataset
from evaluation.evaluators import (
    IoUEvaluator,
    IoUEvaluatorConfig,
    PartTextEvaluator,
    PartTextEvaluatorConfig,
    MCTextEvaluator,
)
from evaluation.evaluators.masks import Reduction
from evaluation.prediction import Prediction, Predictions
from evaluation.rle_dict import get_mask_rle_dicts

logger = logging.getLogger(__name__)


def _to_python_type(obj):
    if isinstance(obj, dict):
        return {k: _to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_type(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_python_type(v) for v in obj)
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


EXAMPLES = {
    'students.jpg' :
        "   Rnd 1: Segment the person wearing glasses." + "\n"
        "   Rnd 2: Segment the his hair.[REF:1]" + "\n"
        "   Rnd 3: Segment the person next to him.[REF:1] " + "\n"
        "   Rnd 4: Segment the bag that she is carrying.[REF:3]" + "\n\n",
    'john_mayer.jpg' :
        "   Rnd 1: Segment John Mayer." + "\n"
        "   Rnd 2: Segment the famous British singer." + "\n"
        "   Rnd 3: Segment the guitar played by instance 1.[REF:1] " + "\n"
        "   Rnd 4: Segment the guitar played by instance 2.[REF:2]" + "\n\n",
    'baseball.jpg' :
        "   Rnd 1: Segment the batter." + "\n"
        "   Rnd 2: Segment the catcher." + "\n"
        "   Rnd 3: Segment the helmet of instance 1.[REF:1]" + "\n"
        "   Rnd 3: Segment the helmet of instance 2.[REF:2]" + "\n\n",
    'wii.jpg' :
        "   Rnd 1: Segment the man." + "\n"
        "   Rnd 2: Segment the other person.[REF:1]" + "\n"
        "   Rnd 3: Segment the object that instance 2 is holding.[REF:2]" + "\n"
        "   Rnd 4: Segment the arm of instance 1.[REF:1]" + "\n\n",
    'cat.jpg' :
        "   Rnd 1: Segment the cat." + "\n"
        "   Rnd 2: Segment the object that instance 1 is standing on.[REF:1]" + "\n"
        "   Rnd 3: Segment the backpack behind instance 1.[REF:1]" + "\n\n",
    'frisbee.jpg' :
        "   Rnd 1: Can you segment the dog?" + "\n"
        "   Rnd 2: Can you segment the frisbee caught by instance 1 in the air?[REF:1]" + "\n\n",
}


def build_conversation(
        model,
        tokenizer,
        conv_dict,
        round_counter: int,
        training_args,
        data_args,
        input_image_path: str,
        input_text,
        user_selected_idx: str
    ) -> dict:
    '''
    Arguments:
        round_counter (int): >= 1
        user_selected_idx (str): of ints of form "idx1,idx2" 1-indexed to refer to previous rounds

    Returns:
        dict: A single batch returned by llava.train.train.DataCollatorForSupervisedDataset
    '''

    if user_selected_idx: # Which
        encode_indices_list = user_selected_idx.split(',')
        encode_indices_list = [int(x) for x in encode_indices_list]
    else:
        encode_indices_list = None

    # First round
    if round_counter-1 == 0:
        curr_round = [
            {
                "from": "human",
                "value": f"[IMAGE256:{input_image_path}] {input_text}"
            },
            {
                "from": "gpt",
                "value": f"[MASK-DECODE:{input_image_path}|INFERENCE|NULL]"
            },
        ]
    # Subsequent rounds
    else:
        if encode_indices_list: # If user referenced prior conversational rounds
            # print("User selected indices:", encode_indices_list)
            # if multiple mask encode selected by user, mulitple [MASK-ENCODE], [BOX-ENCODE] tokens will be generated
            curr_round = [
                {
                    "from": "human",
                    "value": f"[MASK-ENCODE:{input_image_path}|INFERENCE|NULL][BOX-ENCODE:{input_image_path}|INFERENCE|NULL]" * len(encode_indices_list) + input_text,
                    "ind" : [x for x in encode_indices_list]  # <--- NOTE: (expects 1 indexed)
                },
                {
                    "from": "gpt",
                    "value": f"[MASK-DECODE:{input_image_path}|INFERENCE|NULL]"
                },
            ]
        else:
            # print("User did not select.")
            curr_round = [
                {
                    "from": "human",
                    "value": f"{input_text}",
                    "ind" : [-1]
                },
                {
                    "from": "gpt",
                    "value": f"[MASK-DECODE:{input_image_path}|INFERENCE|NULL]"
                },
            ]

    # Grow the conversation by extending its turns/ round
    conv_dict['conversations'].extend(curr_round)

    # reset update flag
    # update_mask_encode_ref = False


    # Make data loader (which performs build_query)
    # Dict with keys train_dataset, eval_dataset, data_collator of types:
    #    LazySupervisedDataset, LazySupervisedDataset, DataCollatorForSupervisedDataset
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        is_inference=True,
        inference_conv=conv_dict
    )

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    pipe=None,
                    **data_module)
    train_dataloader = trainer.get_train_dataloader()


    # Get single conversation
    input_data = list(train_dataloader)
    # print("len input data:", len(input_data))
    inputs = input_data[0]

    return inputs


def inference(
    model,
    inputs: CollatedBatch,
    input_image_path: str,
    round_counter: int,
    training_args,
    all_mask_encode_torch,
    all_box_encode_torch,
    all_masked_image,
    all_masks_cropped,
    output_image_with_mask: bool = False,
    perform_segmentation: bool = True
):
    '''
    Arguments:
        inputs (dict): Output of build_conversation
    '''

    # hack
    device = training_args.device
    def move_device_list(l):
        for i,x in enumerate(l):
            if isinstance(x, Dict):
                l[i] = move_device_dict(x)
            elif isinstance(x, List):
                l[i] = move_device_list(x)
            elif isinstance(x, torch.Tensor):
                l[i] = x.to(device)
            else:
                pass
        return l

    def move_device_dict(d):
        for k, v in d.items():
            if isinstance(v, Dict):
                d[k] = move_device_dict(v)
            elif isinstance(v, List):
                d[k] = move_device_list(v)
            elif isinstance(v, torch.Tensor):
                d[k] = v.to(device)
            else:
                pass
        return d
    inputs = move_device_dict(inputs)

    # replace extra_replacement data with actual outputs from prev n-1 rounds
    num_rounds = round_counter
    mask_encode_ref = inputs['extra_replacement']['mask_encode_ref'][0]     # select batch 0
    mask_encode_ref_no_pad = [x for x in mask_encode_ref if x != -1]        # for turns without mask-encode, padded using -1
    # assert (num_rounds - 1) == len(mask_encode_ref)                       # with -1 padding, length should match (not the case for multi-instance encode)

    # print("Mask Encode Ref:", mask_encode_ref)

    mask_encode_count = 0
    bbox_encode_count = 0
    replacement_data = inputs['extra_replacement']['data'][0]      # [('image-encode', ...), ('mask-decode', ...), ('mask-encode', ...), ('bbox-encode', ...)]
    for idx, (task, data_tuple) in enumerate(replacement_data):                        # idx, (<task>, <data>)
        # task = task_data[0]
        # curr_data_tuple = task_data[1]
        curr_data = data_tuple[0]
        curr_mask_id = data_tuple[1]                                     # for inference, this is 'NULL'
        # breakpoint()
        if task == 'mask-encode':
            encode_ref = mask_encode_ref_no_pad[mask_encode_count]            # get encode_ref for this mask-encode
            prev_output_mask = all_mask_encode_torch[encode_ref]
            if prev_output_mask is not None:
                new_data = prev_output_mask.to(curr_data.device)
            else:
                new_data = torch.zeros_like(curr_data)                        # in case model predicts empty mask
            replacement_data[idx] = ['mask-encode', [new_data, curr_mask_id]] # replace mask-encode
            mask_encode_count += 1                                            # increment counter
        if task == 'bbox-encode':
            encode_ref = mask_encode_ref_no_pad[bbox_encode_count]                   # get encode_ref for this box-encode
            prev_output_bbox = all_box_encode_torch[encode_ref]
            if prev_output_bbox is not None:
                new_data = prev_output_bbox.to(curr_data.device)
            else:
                new_data = torch.zeros_like(curr_data)                        # in case model predicts empty mask
            replacement_data[idx] = ['bbox-encode', [new_data, curr_mask_id]] # replace bbox-encode
            bbox_encode_count += 1                                            # increment counter

    # Save inputs in current state (forward pass will mutate inputs dict)
    inputs_after_replacement = copy.deepcopy(inputs)

    # sanity check: length of mask-encode jobs should line up with none -1 mask-encode idx
    if mask_encode_count > 0:
        assert mask_encode_count == len(mask_encode_ref_no_pad)
    if bbox_encode_count > 0:
        assert bbox_encode_count == len(mask_encode_ref_no_pad)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)


    # processors
    mask_image_processor = model.get_segmentator().process_images      # resize original_image --> hipie decoder size
    clip_image_processor = model.get_vision_tower().image_processor    # resize original_image --> clip encoder size

    # resize image
    original_image = np.array(Image.open(input_image_path).convert('RGB'))
    fake_masks = torch.zeros(original_image.shape[:2]).float()
    resized_image = mask_image_processor(original_image,[fake_masks,fake_masks])['image'].permute(1,2,0).numpy()

    # Extract predicted masks, predicted boxes, gt masks
    predicted_masks = outputs['individual_losses']['segm_loss_mask']
    predicted_boxes = outputs['individual_losses']['segm_loss_boxes']
    for pred_mask, pred_box in zip(predicted_masks, predicted_boxes):
        (x0,y0,x1,y1) = np.clip(pred_box.astype(int),0,resized_image.shape[0])
        x1 = np.clip(x1,x0+2,resized_image.shape[0])
        y1 = np.clip(y1,y0+2,resized_image.shape[0])
        max_width = max(x1-x0,y1-y0)
        bbox_coords_sam = torch.tensor([y0,x0,y1,x1]) / 1024.0
        # gt_masks = metrics['mask_data']             # during inference, gt_mask will be np.one

        # sanity check
        # print("len predicted masks:", len(predicted_masks))
        # crop instance using predicted mask
        # if model_args.segmentator == 'sam':
        #     pred_mask = pred_mask.transpose(1,2,0)      # (1024 x 1024 x 1)
        image_masked = cv2.bitwise_and(resized_image, resized_image, mask=pred_mask.astype(np.uint8))
        image_masked_cropped = image_masked[y0:y1,x0:x1]
        image_masked_cropped_padded = np.zeros((max_width,max_width,image_masked.shape[-1]),dtype=image_masked.dtype)
        image_masked_cropped_padded[:image_masked_cropped.shape[0],:image_masked_cropped.shape[1]] = image_masked_cropped
        processed_mask_encode = clip_image_processor(Image.fromarray(image_masked_cropped_padded.astype(np.uint8)))
        processed_mask_encode = torch.tensor(processed_mask_encode.pixel_values[0])

        fake_mask_id = -1
        all_masks_cropped.append((image_masked_cropped, f'round {round_counter}'))
        all_mask_encode_torch.append(processed_mask_encode)
        all_box_encode_torch.append(bbox_coords_sam)

        # Display segmentation mask on top of image
        image_with_mask = None
        if output_image_with_mask:
            image_with_mask = resized_image.copy()

            pred_mask_expanded = pred_mask[:, :, None].astype(np.uint8)
            image_with_mask[pred_mask] = (resized_image * 0.5 +  pred_mask_expanded* np.array([255, 0, 0]) * 0.5)[pred_mask]

            # get resized image size
            mask_data_list = outputs['individual_losses']['mask_data']
            mask_data = mask_data_list[-1]
            (h,w) = mask_data['input_size']
            image_with_mask = image_with_mask[:h, :w, :]

            all_masked_image.append(
                (image_with_mask, f'round {round_counter}')
            )

        # Sanity check: Visualize mask encode data
        mask_encode_data = []
        box_encode_data = []
        replacement_data = inputs_after_replacement['extra_replacement']['data'][0]
        for (task, data_tuple) in replacement_data:
            data = data_tuple[0]
            mask_id = data_tuple[1]
            if task == 'mask-encode':
                data_np = data.permute(1,2,0).detach().cpu().numpy()       # tensor: CxHxW, gallery expects HxWxC
                data_np = np.clip(data_np, -1, 1)
                mask_encode_data.append(data_np)
            if task == 'bbox-encode':
                data_np = data.detach().cpu().numpy()
                box_encode_data.append(data_np)

    # print("Box Encode:")
    # print(box_encode_data)

    # breakpoint()

    return outputs, predicted_masks, all_masked_image, all_masks_cropped, mask_encode_data

    # all_masked_image_square = [resize_image(pad_to_square(img), 224) for (img, label) in all_masked_image]
    # return image_with_mask, all_masked_image_square, all_masks_cropped, mask_encode_data




def main(args: list[str] = None):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument('--limit_instances', type=int, default=None)
    parser.add_argument('--use_last_predicted_mask', action='store_true')
    parser.add_argument('--save_images_with_masks', action='store_true')
    parser.add_argument('--save_mask_overlays', action='store_true')

    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses(args)

    local_rank = training_args.local_rank
    model_name = model_args.load or model_args.model_name_or_path
    if len(model_name.split('/')) == 3:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/'.join(model_name.split('/')[:2]),
            subfolder=model_name.split('/')[-1],
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    # Load Eseg stuff
    datasets = []
    for question_type in QuestionType:
        eseg_dataset = ExplanatorySegDataset(
            dataset_path='/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partonomy/partonomy_qa_pairs_val.json',
            question_type=question_type,
        )

        datasets.append(eseg_dataset)

    segmentation_dataset_adapter = ExplanatorySegLazySupervisedDataset(
        datasets,
        tokenizer,
        data_args,
        only_return_correct_conversation=True, # For segmentation, only return the correct conversation
        is_inference=True
    )

    mc_dataset_adapter = ExplanatorySegLazySupervisedDataset(
        datasets,
        tokenizer,
        data_args,
        only_return_correct_conversation=False, # Need all conversations to choose best response
        is_inference=True
    )

    bnb_model_from_pretrained_args = {}

    # print("Loading checkpoint:", model_args.model_name_or_path)
    if len(model_name.split('/')) == 3:                             # Huggingface expects user_name/repo_name
        model = LlavaLlamaForCausalLM.from_pretrained(
            '/'.join(model_name.split('/')[:2]),
            subfolder=model_name.split('/')[-1],
            cache_dir=training_args.cache_dir,
            use_last_predicted_mask=extra_args.use_last_predicted_mask,
            **bnb_model_from_pretrained_args
        )
    elif len(model_name.split('/')) in [1,2]:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            use_last_predicted_mask=extra_args.use_last_predicted_mask,
            **bnb_model_from_pretrained_args
        )
    else:
        logger.info(f'Loading model from local path: {model_name}')

        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            use_last_predicted_mask=extra_args.use_last_predicted_mask,
            **bnb_model_from_pretrained_args
        )

    model.eval()
    model.initialize_vision_tokenizer(model_args,tokenizer)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    # fix for new dtype mismatch issue
    model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    # breakpoint()

    data_args.image_processor = vision_tower.image_processor
    data_args.mask_processor = model.get_segmentator().process_images
    data_args.is_multimodal = True
    if data_args.segmentation_config:
        data_args.register = COCORegister(data_args,is_eval=True)       # don't have to pass in annotations_config (during inference, not gt mask will be loaded)

    # clear screen
    # os.system('cls' if os.name == 'nt' else 'clear')

    seg_dl = DataLoader(
        segmentation_dataset_adapter,
        batch_size=1,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer, unpack_multiple_choice=False),
        shuffle=False
    )

    mc_dl = DataLoader(
        mc_dataset_adapter,
        batch_size=1,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer, unpack_multiple_choice=True),
        shuffle=False
    )

    def create_metric_bundle():
        part_text_config = PartTextEvaluatorConfig()
        return {
            'predictions': Predictions(),
            'evaluators': {
                'mask_micro': IoUEvaluator(
                    IoUEvaluatorConfig(reduction=Reduction.MICRO),
                    metric_group_name='mask_micro'
                ),
                'mask_macro': IoUEvaluator(
                    IoUEvaluatorConfig(reduction=Reduction.MACRO),
                    metric_group_name='mask_macro'
                ),
                'part_text': PartTextEvaluator(part_text_config),
                'mc_part_text': MCTextEvaluator(metric_group_name='mc_part_text'),
                'mc_object_text': MCTextEvaluator(metric_group_name='mc_object_text'),
            }
        }

    def get_evaluators(bundle: dict):
        evaluators = bundle['evaluators']
        return [
            evaluators['mask_micro'],
            evaluators['mask_macro'],
            evaluators['part_text'],
            evaluators['mc_part_text'],
            evaluators['mc_object_text'],
        ]

    question_type_results: dict[str, dict] = {}
    num_instances = 0

    total_instances = min(len(seg_dl), extra_args.limit_instances if extra_args.limit_instances is not None else len(seg_dl))
    prog_bar = tqdm(zip(seg_dl, mc_dl), total=total_instances)
    for i, (seg_batch, mc_batch) in enumerate(prog_bar):
        if extra_args.limit_instances is not None and i >= extra_args.limit_instances:
            break

        seg_inputs = seg_batch
        mc_inputs = mc_batch

        # Find an example with multiple mask-decodes for debugging
        # conversations = mc_inputs['conversations']
        # conversation = conversations[0]
        # assistant_turn = conversation[1]
        # if assistant_turn['value'].count('<seg>') <= 1:
        #     continue

        round_counter = 0
        conv_dict = {
            'task': 'segmentation',
            'base': '[null]',
            'conversations': []
        }

        all_masked_image = []           # image with mask
        all_masks_cropped = []          # cropped instance
        all_mask_encode_torch = []      # cropped instance preprocessed
        all_box_encode_torch = []       # bbox coords of cropped instance, preprocessed

        # while True:
        #     round_counter += 1          # 1-indexed

        #     # TODO construct input_text from question and answer
        #     input_text = 'fake user input'
        #     user_selected_idx = None # 1-indexed conversation round to refer to; None for no reference


        #     inputs = build_conversation(
        #         model,
        #         tokenizer,
        #         conv_dict,
        #         round_counter,          # 1-indexed
        #         training_args,
        #         data_args,
        #         input_image_path,       # relative to data_args.image_folder
        #         input_text,
        #         user_selected_idx
        #     )
        #     inputs = collator([instance_dict])[0]

        #     pred_mask, image_with_mask, _, _, _ = inference(
        #         model,
        #         inputs,
        #         input_image_path,       # absolute path
        #         round_counter,          # 0-indexed
        #         training_args,
        #         all_mask_encode_torch,
        #         all_box_encode_torch,
        #         all_masked_image,
        #         all_masks_cropped,
        #     )


        #     if image_with_mask is not None:
        #         name = image_file.replace('.jpg', '')
        #         out_dir = os.path.join(data_args.inference_output_dir, name)
        #         os.makedirs(out_dir, exist_ok=True)
        #         out_file = os.path.join(out_dir, f'round_{round_counter}_text.txt')
        #         with open(out_file, "w") as f:
        #             f.write(input_text + "\n")
        #         out_file = os.path.join(out_dir, f'round_{round_counter}_output.jpg')
        #         Image.fromarray(image_with_mask).save(out_file)

        # TODO loop over the replacement data, truncating it at the start and extending it on each inference round
        image_path = mc_inputs['image_paths'][0]

        # First forward pass to get the predicted ground truth mask
        _seg_outputs, predicted_masks, all_images_with_masks, _, _ = inference(
            model,
            seg_inputs,
            image_path,       # absolute path
            round_counter,          # 0-indexed
            training_args,
            all_mask_encode_torch,
            all_box_encode_torch,
            all_masked_image,
            all_masks_cropped,
            output_image_with_mask=extra_args.save_images_with_masks,
            perform_segmentation=True
        )

        all_images_with_masks: list[tuple[np.ndarray, str]]
        all_images_with_masks: list[np.ndarray] = [t[0] for t in all_images_with_masks]

        predicted_masks: list[np.ndarray] = list(predicted_masks)  # n_masks_in_instance x h x w

        mc_outputs, _, _, _, _ = inference(
            model,
            mc_inputs,
            image_path,       # absolute path
            round_counter,          # 0-indexed
            training_args,
            all_mask_encode_torch,
            all_box_encode_torch,
            all_masked_image,
            all_masks_cropped,
            output_image_with_mask=False,
            perform_segmentation=False
        )

        seq_logprobs = mc_outputs['individual_losses']['log_probs']  # (n_conversations,)
        if isinstance(seq_logprobs, torch.Tensor):
            if seq_logprobs.dim() != 1:
                raise ValueError(f'Expected 1D tensor, got {seq_logprobs.dim()}D tensor')
            seq_logprobs = seq_logprobs.detach().float().cpu().numpy()
        else:
            seq_logprobs = np.asarray(seq_logprobs)
        assert seq_logprobs.ndim == 1, f'Expected 1D array, got {seq_logprobs.ndim}D'

        dataset_index = i # Do NOT use seg_inputs['dataset_index'][0] as for some reason it doesn't match dataset index
        dataset_idx, local_idx = segmentation_dataset_adapter.convert_global_index(dataset_index)
        instance: ExplanatorySegInstance = segmentation_dataset_adapter.datasets[dataset_idx][local_idx]

        question_type = instance.question_type
        question_type_name = question_type.name if question_type else 'UNKNOWN'
        bundle = question_type_results.setdefault(question_type_name, create_metric_bundle())
        evaluators = bundle['evaluators']

        prediction_metrics = {
            evaluators['mask_micro'].metric_group_name: None,
            evaluators['mask_macro'].metric_group_name: None,
            evaluators['part_text'].metric_group_name: None,
            evaluators['mc_part_text'].metric_group_name: None,
            evaluators['mc_object_text'].metric_group_name: None,
        }

        part_indices = [
            idx for idx, cq_type in enumerate(instance.conversation_question_types)
            if cq_type == 'part_question'
        ]
        part_conversation_types = [instance.conversation_types[idx] for idx in part_indices]

        gt_parts = None
        predicted_parts = None
        gt_parts_answer = None
        predicted_parts_answer = None
        gt_parts_answer_index = None
        predicted_parts_answer_index = None

        # Compute multiple choice and part metrics
        part_log_probs = [seq_logprobs[idx] for idx in part_indices]
        gt_parts_answer_index = part_conversation_types.index('correct_answer')
        predicted_parts_answer_index = int(np.argmax(part_log_probs))

        mc_part_metrics = evaluators['mc_part_text'].update(predicted_parts_answer_index, gt_parts_answer_index)
        prediction_metrics[evaluators['mc_part_text'].metric_group_name] = mc_part_metrics

        answer_parts = instance.answer_parts
        gt_parts = answer_parts[gt_parts_answer_index]
        predicted_parts = answer_parts[predicted_parts_answer_index]

        part_text_metrics = evaluators['part_text'].update(predicted_parts, gt_parts)
        prediction_metrics[evaluators['part_text'].metric_group_name] = part_text_metrics

        gt_parts_answer = instance.part_answer_choices[gt_parts_answer_index]
        predicted_parts_answer = instance.part_answer_choices[predicted_parts_answer_index]

        resized_pred_masks = None
        gt_masks_np = None
        if predicted_masks and instance.masks is not None:
            gt_masks_tensor = instance.masks
            gt_masks_np = gt_masks_tensor.detach().cpu().numpy() > 0.5

            if len(predicted_masks) != gt_masks_np.shape[0]:
                logger.warning(
                    'Mismatch in number of predicted (%d) and ground truth (%d) masks for %s; skipping mask metrics.',
                    len(predicted_masks),
                    gt_masks_np.shape[0],
                    instance.img_path
                )
            else:
                # Determine the resized region used by HIPIE (before right/bottom padding)
                # Then crop predicted masks to that region before resizing back to original size.
                orig_h, orig_w = gt_masks_np.shape[1:]
                original_image_eval = np.array(Image.open(image_path).convert('RGB'))
                _fake_masks_eval = torch.zeros(original_image_eval.shape[:2]).float()
                proc_info_eval = model.get_segmentator().process_images(original_image_eval, [_fake_masks_eval, _fake_masks_eval])
                resized_h, resized_w = proc_info_eval['input_size']

                resized_pred_masks_list = []

                for mask in predicted_masks:
                    m = mask.astype(np.uint8)
                    # Remove right/bottom padding introduced by the HIPIE preprocessor
                    m = m[:resized_h, :resized_w]

                    if (resized_w, resized_h) != (orig_w, orig_h):
                        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                    resized_pred_masks_list.append(m.astype(bool))

                resized_pred_masks = np.array(resized_pred_masks_list)
                micro_metrics = evaluators['mask_micro'].update(resized_pred_masks, gt_masks_np)
                macro_metrics = evaluators['mask_macro'].update(resized_pred_masks, gt_masks_np)
                prediction_metrics[evaluators['mask_micro'].metric_group_name] = micro_metrics
                prediction_metrics[evaluators['mask_macro'].metric_group_name] = macro_metrics

                # Optional: save visual overlays for debugging
                if extra_args.save_mask_overlays:
                    try:
                        # Load original image
                        orig_img_bgr = cv2.imread(image_path)
                        if orig_img_bgr is None:
                            raise RuntimeError(f"Failed to read image at {image_path}")

                        # Convert to RGB for blending consistency, then back to BGR before writing
                        orig_img = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

                        # Per-mask overlays (pred in red, GT in green)
                        name = os.path.splitext(os.path.basename(image_path))[0]
                        out_dir = os.path.join(data_args.inference_output_dir, name)
                        os.makedirs(out_dir, exist_ok=True)

                        num_masks = resized_pred_masks.shape[0]
                        for k in range(num_masks):
                            pmask = resized_pred_masks[k]
                            gmask = gt_masks_np[k]

                            overlay_pred = np.zeros_like(orig_img, dtype=np.uint8)
                            overlay_gt = np.zeros_like(orig_img, dtype=np.uint8)

                            overlay_pred[pmask] = (255, 0, 0)
                            overlay_gt[gmask] = (0, 255, 0)

                            alpha = 0.8
                            blended = orig_img.copy()
                            blended = cv2.addWeighted(blended, 1.0, overlay_pred, alpha, 0)
                            blended = cv2.addWeighted(blended, 1.0, overlay_gt, alpha, 0)

                            # Optional: include part name if available
                            suffix = ''
                            if gt_parts and k < len(gt_parts):
                                # sanitize filename component
                                part_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', gt_parts[k])
                                suffix = f'_{part_name}'

                            out_file = os.path.join(out_dir, f'round_{round_counter}_overlay_{k}{suffix}.jpg')
                            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(out_file, blended_bgr)
                    except Exception as e:
                        logger.warning('Failed to save mask overlay(s) for %s: %s', image_path, e)

        elif predicted_masks:
            logger.warning('Ground truth masks missing for %s; skipping mask metrics.', instance.img_path)
        else:
            logger.warning('No predicted masks returned for %s; skipping mask metrics.', instance.img_path)

        gt_object_answer_index = None
        predicted_object_answer_index = None
        gt_object_answer = None
        predicted_object_answer = None
        gt_object = None
        predicted_object = None

        if instance.question_type in [QuestionType.WHOLE_TO_PART, QuestionType.PART_TO_WHOLE]:
            object_indices = [
                idx for idx, cq_type in enumerate(instance.conversation_question_types)
                if cq_type == 'object_question'
            ]
            if object_indices:
                object_conversation_types = [instance.conversation_types[idx] for idx in object_indices]
                object_log_probs = [seq_logprobs[idx] for idx in object_indices]

                gt_object_answer_index = object_conversation_types.index('correct_answer')
                predicted_object_answer_index = int(np.argmax(object_log_probs))

                mc_object_metrics = evaluators['mc_object_text'].update(
                    predicted_object_answer_index,
                    gt_object_answer_index
                )
                prediction_metrics[evaluators['mc_object_text'].metric_group_name] = mc_object_metrics

                gt_object_answer = instance.object_answer_choices[gt_object_answer_index]
                predicted_object_answer = instance.object_answer_choices[predicted_object_answer_index]
                gt_object = instance.answer_objects[gt_object_answer_index] if instance.answer_objects else None
                predicted_object = (
                    instance.answer_objects[predicted_object_answer_index]
                    if instance.answer_objects else None
                )

        # Store prediction
        if resized_pred_masks is not None and gt_masks_np is not None and gt_parts:
            gt_masks_rle = get_mask_rle_dicts(gt_masks_np.astype(np.uint8), list(gt_parts))
            predicted_masks_rle = get_mask_rle_dicts(resized_pred_masks.astype(np.uint8), list(gt_parts))
        else:
            gt_masks_rle = None
            predicted_masks_rle = None

        bundle['predictions'].add_prediction(
            Prediction(
                image_path=instance.img_path,
                question_type=question_type,
                questions=instance.questions,
                parts_answer_choices=instance.part_answer_choices,
                gt_parts_answer=gt_parts_answer,
                predicted_parts_answer=predicted_parts_answer,
                gt_parts=gt_parts,
                predicted_parts=predicted_parts,
                gt_masks=gt_masks_rle,
                predicted_masks=predicted_masks_rle,
                mask_confidences=None,
                object_answer_choices=instance.object_answer_choices,
                gt_object_answer=gt_object_answer,
                predicted_object_answer=predicted_object_answer,
                gt_object=gt_object,
                predicted_object=predicted_object,
                metrics=prediction_metrics
            )
        )

        num_instances += 1

        if all_images_with_masks:
            name = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.join(data_args.inference_output_dir, name)
            os.makedirs(out_dir, exist_ok=True)

            # Write image
            for i, image, in enumerate(all_images_with_masks):
                out_file = os.path.join(out_dir, f'round_{round_counter}_seg{i}.jpg')
                Image.fromarray(image).save(out_file)

                # Write text
                out_file = os.path.join(out_dir, f'round_{round_counter}_seg{i}_text.txt')
                with open(out_file, "w") as f:
                    f.write(seg_inputs['conversations'][0][1]['value']) # Output the GPT model response for the first conversation


    question_type_payload = {}
    for qtype_name, bundle in sorted(question_type_results.items()):
        qtype_dict = bundle['predictions'].to_dict(get_evaluators(bundle))
        qtype_dict['meta'] = {
            'num_instances': len(bundle['predictions'].predictions)
        }
        question_type_payload[qtype_name] = qtype_dict

    output_payload = {
        'question_types': question_type_payload,
        'meta': {
            'num_instances': num_instances
        }
    }
    output_payload = _to_python_type(output_payload)

    if data_args.inference_output_dir:
        os.makedirs(data_args.inference_output_dir, exist_ok=True)
        metrics_path = pathlib.Path(data_args.inference_output_dir) / 'metrics.json'

        with metrics_path.open('wb') as f:
            f.write(orjson.dumps(output_payload, option=orjson.OPT_INDENT_2))

        logger.info(f'Metrics saved to {metrics_path}')

    payload_to_log = {
        q_type : output_payload['question_types'][q_type]['metrics']
        for q_type in output_payload['question_types']
    }
    logger.info('Aggregated metrics by question type:\n%s', pformat(payload_to_log, indent=4))


if __name__ == '__main__':
    from pathlib import Path
    import coloredlogs

    coloredlogs.install(level='INFO')

    # Arguments taken from segllm/scripts/inference/launch_cli_demo.sh
    CHECKPOINT = 'Marlo-Z/SegLLM/all_data_checkpoint'
    # CHECKPOINT = '/scratch/blume5/checkpoints/segllm/checkpoint-400'

    # ROOT_DIR = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/src/models/segllm'
    ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
    args = [
        "--deepspeed", os.path.join(ROOT_DIR, "scripts/deepspeed_configs/zero2.json"),
        "--model_name_or_path", CHECKPOINT,
        "--load", CHECKPOINT,
        "--image_folder", os.path.join(ROOT_DIR, "inference_images"),
        "--mm_use_seg", "True",
        "--segmentator", "hipie",
        "--vision_tower", "openai/clip-vit-large-patch14",
        "--mm_projector_type", "mlp2x_gelu",
        "--tune_mm_mlp_adapter", "False",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_vision_select_feature", "patch",
        "--mm_use_im_patch_token", "False",
        "--bf16", "True",
        "--lora_enable", "False",
        "--split_loading", "False",
        "--version", "plain",
        "--mm_use_gen", "True",
        "--inference_output_dir", os.path.join(ROOT_DIR, "inference_results"),
        "--output_dir", "./out_dir",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--save_total_limit", "2",
        "--learning_rate", "2e-5",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "False",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "0",
        "--lazy_preprocess", "True",
        "--output_text",
        # '--limit_instances', '10', # DEBUG
        # '--use_last_predicted_mask',
        # '--save_images_with_masks',
        # '--save_mask_overlays',
    ]

    main(args)

    # main()

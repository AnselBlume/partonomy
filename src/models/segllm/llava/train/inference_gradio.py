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

import gradio as gr

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from tqdm.cli import tqdm
import torch

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
from llava.train.train import *
from accelerate import Accelerator 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


pipe = None
parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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

bnb_model_from_pretrained_args = {}

if model_args.dev == 'test3': # test 2 layer
    print("Debug mode...")
    cfg = LlavaConfig.from_pretrained(model_args.model_name_or_path)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 128
    model = LlavaLlamaForCausalLM._from_config(cfg)
else:
    model_name = model_args.load or model_args.model_name_or_path
    if len(model_name.split('/')) == 3:                             # Huggingface expects user_name/repo_name
        model = LlavaLlamaForCausalLM.from_pretrained(
            '/'.join(model_name.split('/')[:2]),
            subfolder=model_name.split('/')[-1],
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif len(model_name.split('/')) in [1,2]:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError()


model.eval()
model.initialize_vision_tokenizer(model_args,tokenizer)
vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

# fix for new dtype mismatch issue
model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
# breakpoint()

data_args.image_processor = vision_tower.image_processor
if model_args.segmentator:
    data_args.mask_processor = model.get_segmentator().process_images
data_args.is_multimodal = True
if data_args.segmentation_config:
    data_args.register = COCORegister(data_args,is_eval=True)       # don't have to pass in annotations_config (during inference, not gt mask will be loaded)


# Global variables 
    
global_conv_dict = {
        "task": "segmentation",
        "base": "[null]",
        "conversations": []             # each round will be appended
    }

all_masked_image = []           # image with mask
all_masks_cropped = []          # cropped instance
all_mask_encode_torch = []      # cropped instance preprocessed
all_box_encode_torch = []       # bbox coords of cropped instance, preprocessed
# user_selected_idx = -1
# update_mask_encode_ref = False  # If True: mask_encode_ref = user_selected_idx + 1, Else: = -1 

round_counter = 0

def inference(
        input_text,
        user_selected_idx,
        mask_color,
        input_image_path,
):
    
    # print()
    # print("Input text: ", input_text)
    # print("Input selected mask encode:", user_selected_idx)
    # print("Input Image Path: ", input_image_path)
    
    # global update_mask_encode_ref    
    global round_counter
    round_counter = round_counter + 1

    encode_indices_list = None
    if user_selected_idx:
        encode_indices_list = user_selected_idx.split(',')
        encode_indices_list = [int(x) for x in encode_indices_list]

    if mask_color:
        mask_color = np.array([int(c) for c in mask_color.split(',')])
        if len(mask_color) != 3:
            mask_color = np.array([255,0,0])
    else:
        mask_color = np.array([255,0,0])
    print("Mask color:", mask_color)

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
        if encode_indices_list:
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
            print("User did not select.")
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

    # Grow the conversation by extending it's turns/ round
    global_conv_dict['conversations'].extend(curr_round)

    # reset update flag
    # update_mask_encode_ref = False


    # Make data loader (which performs build_query)
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        is_inference=True,
        inference_conv=global_conv_dict
    )

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    pipe=pipe,
                    **data_module)
    train_dataloader = trainer.get_train_dataloader()
    model.eval()

    # Get single conversation
    input_data = list(train_dataloader)
    print("len input data:", len(input_data))
    inputs = input_data[0]

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

    print("Mask Encode Ref:", mask_encode_ref)

    mask_encode_count = 0
    bbox_encode_count = 0
    replacement_data = inputs['extra_replacement']['data'][0]      # [('image-encode', ...), ('mask-decode', ...), ('mask-encode', ...), ('mask-decode', ...)]
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
    pred_mask = predicted_masks[-1]
    pred_box = predicted_boxes[-1]
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
    image_with_mask = resized_image.copy()

    if model_args.segmentator == 'hipie':
        pred_mask_expanded = pred_mask[:, :, None].astype(np.uint8)
    else:
        pred_mask_expanded = pred_mask.astype(np.uint8)
    image_with_mask[pred_mask] = (resized_image * 0.5 +  pred_mask_expanded* mask_color * 0.5)[pred_mask]

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

    return image_with_mask, all_masked_image, all_masks_cropped, mask_encode_data

    # all_masked_image_square = [resize_image(pad_to_square(img), 224) for (img, label) in all_masked_image]
    # return image_with_mask, all_masked_image_square, all_masks_cropped, mask_encode_data

def clear_history():
    global global_conv_dict
    global all_masked_image
    global all_masks_cropped
    global all_mask_encode_torch
    global all_box_encode_torch
    global round_counter

    global_conv_dict = {
        "task": "segmentation",
        "base": "[null]",
        "conversations": []             # each round will be appended
    }

    all_masked_image = []
    all_masks_cropped = []
    all_mask_encode_torch = []
    all_box_encode_torch = []

    round_counter = 0

    outputs = [
        None,                           # segmentation output window
        all_masked_image,               # List[]: image with mask
        all_masks_cropped,              # List[]: cropped instance
        []                              # List[]: mask-encode data
    ] 

    return outputs


with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")

    gallery_num_cols = 5
    gallery_num_rows = 1

    with gr.Row():
        # left column
        with gr.Column():
            # 1)
            text_input = gr.Textbox(label="Input Prompt:", placeholder="Input text")
            # 2)
            with gr.Row():
                selected_index_text = gr.Textbox(label="Reference instance from round (1-indexed):", placeholder="None")       # will be 1-indexed
                # deselect_btn = gr.Button("Deselect")]
                mask_color = gr.Textbox(label="Mask color (RGB):", placeholder="255,0,0")
            # 3)
            image_input = gr.Image(
                type="filepath", label="Input Image"
            )
        # right column
        mask_output = gr.Image(
            type='numpy', label="Segmentation Output", elem_id='output_image'
        )
    
    # Seg output history
    mask_output_history = gr.Gallery(
        label="Segmentation History", show_label=True, elem_id="mask_output_gallery", 
        columns=gallery_num_cols, rows=gallery_num_rows, object_fit="fill", height=245
    )

    with gr.Row():
        run_btn = gr.Button("Run")
        clear_btn = gr.Button("Clear History")
        
    run_btn.click(
        fn=inference, 
        inputs=[
            text_input,
            selected_index_text,
            mask_color,
            image_input
        ], 
        outputs=[
            mask_output,                # segmentation output window
            mask_output_history,        # List[]: image with mask
        ]
    )

    clear_btn.click(
        fn=clear_history, 
        inputs=[], 
        outputs=[
            mask_output,                # segmentation output window
            mask_output_history,        # List[]: image with mask
        ]
    )


demo.queue()
demo.launch(share=True)


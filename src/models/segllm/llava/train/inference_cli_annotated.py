# NOTE: This is an annotated copy of ``inference_cli.py`` that explains the
# inference data flow for SegLLM. The goal is to highlight how multi-round
# feature passing is implemented today so that it can be adapted for
# multi-mask generation within a single response. The executable logic is
# intentionally kept identical to the source file while additional comments
# and docstrings clarify the intent of each block.
#
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
        round_counter,
        training_args,
        data_args,
        input_image_path,
        input_text,
        user_selected_idx
    ):
    """Build the tokenized conversation for the current round.

    Returns:
        Dict[str, torch.Tensor]: A single batch worth of tokenized inputs that
            can be fed directly to :func:`LlavaLlamaForCausalLM.forward`.

    The helper mirrors the training data preparation pipeline so that the
    inference CLI can re-use all multimodal preprocessing (image embeds,
    feature placeholders, etc.). Understanding how the ``conv_dict`` grows over
    time is essential when adapting the model for sequential mask generation
    inside one response: each ``[MASK-ENCODE]`` token in ``conv_dict`` will be
    filled with the features from previous rounds right before the forward
    pass in :func:`inference`.
    """

    # ``user_selected_idx`` encodes which previous masks should be available as
    # references. It is provided via ``[REF:x]`` markers. When adapting the
    # model to emit multiple masks per response, you would generate the
    # ``encode_indices_list`` programmatically after each mask is produced so
    # that subsequent masks can condition on earlier ones.

    if user_selected_idx:
        encode_indices_list = user_selected_idx.split(',')
        encode_indices_list = [int(x) for x in encode_indices_list]
    else:
        encode_indices_list = None

    # First round
    # -----------
    # The initial prompt always includes the ``[IMAGE256:...]`` and
    # ``[MASK-DECODE:...]`` tokens because there are no historical masks to
    # encode yet. Future multi-mask responses will still need this setup for
    # the first mask of the response.
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
    # -----------------
    # For later rounds we optionally inject ``[MASK-ENCODE]`` (and
    # ``[BOX-ENCODE]``) tokens which will be replaced with the dense features
    # of a previous prediction. This is the critical "feature passing"
    # mechanism that must be replicated when looping over masks within a
    # single response.
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

    # Grow the conversation by extending it's turns/ round
    # ``conv_dict`` accumulates the alternating human/GPT entries. During
    # training this mirrors the raw dataset, while at inference it allows us to
    # simulate multi-turn history for conversational segmentation.
    conv_dict['conversations'].extend(curr_round)

    # reset update flag
    # update_mask_encode_ref = False


    # Make data loader (which performs build_query)
    # -------------------------------------------------
    # ``make_supervised_data_module`` (defined in ``train.py``) tokenizes the
    # conversation, prepares image tensors, and returns a PyTorch ``Dataset``.
    # Re-using this helper guarantees that inference and training share the
    # same preprocessing semantics.
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
        inputs,
        input_image_path,
        round_counter,
        training_args,
        all_mask_encode_torch,
        all_box_encode_torch,
        all_masked_image,
        all_masks_cropped,
):
    """Run one forward pass and update feature caches for future rounds.

    Returns:
        Tuple[np.ndarray, list[tuple[np.ndarray, str]], list[tuple[np.ndarray, str]], list[np.ndarray]]:
            ``image_with_mask`` showing the current prediction, accumulated
            visualizations, cropped mask previews, and the tensors queued for
            future ``[MASK-ENCODE]`` substitutions.

    The sequence of operations in this function is crucial for multi-mask
    generation. Immediately before the forward pass we replace placeholder
    tensors in ``inputs['extra_replacement']`` with previously predicted mask
    embeddings. After the forward pass we extract the new mask features and
    append them to ``all_mask_encode_torch`` so that the next inference round –
    or the next mask in a single response – can condition on them.
    """

    # ``training_args.device`` controls where tensors are moved. In a
    # multi-mask scenario we will likely keep calling this helper inside a loop
    # without resetting the global history, so keeping the cache tensors on the
    # correct device prevents repeated host<->device transfers.

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
    # ----------------------------------------------------------------------
    # ``inputs['extra_replacement']`` comes from the dataset and contains dummy
    # tensors. Here we splice in the real mask / bbox encodings generated in
    # earlier rounds. For single-response multi-mask generation you would call
    # this function once per predicted mask, reusing and extending
    # ``all_mask_encode_torch`` between iterations.
    num_rounds = round_counter
    mask_encode_ref = inputs['extra_replacement']['mask_encode_ref'][0]     # select batch 0
    mask_encode_ref_no_pad = [x for x in mask_encode_ref if x != -1]        # for turns without mask-encode, padded using -1
    # assert (num_rounds - 1) == len(mask_encode_ref)                       # with -1 padding, length should match (not the case for multi-instance encode)

    # print("Mask Encode Ref:", mask_encode_ref)

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
    # -----------
    # The model consumes the fully materialized tensor tree and produces
    # losses + auxiliary data. Note that ``segm_loss_mask`` is used as the
    # predicted mask logits, so extracting the last entry gives the most recent
    # prediction.
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
    # -------------------------------------------------
    # ``predicted_masks`` and ``predicted_boxes`` are stacked across decoding
    # steps; the final element corresponds to the latest ``[MASK-DECODE]``.
    # This is where you would capture intermediate masks if you allow the
    # decoder to emit multiple masks within a single response.
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

    return image_with_mask, all_masked_image, all_masks_cropped, mask_encode_data

    # all_masked_image_square = [resize_image(pad_to_square(img), 224) for (img, label) in all_masked_image]
    # return image_with_mask, all_masked_image_square, all_masks_cropped, mask_encode_data




def main(args: list[str] = None):
    """Entry point for the CLI demo.

    Returns:
        None: The function runs an interactive REPL loop for segmentation
        queries until the user exits.

    Although this function drives an interactive UI, the initialization steps
    (tokenizer/model loading, processor setup, and cache priming) mirror what a
    scripted multi-mask inference routine would need. When adapting for batched
    inference, you can re-use the setup portion and replace the input loop with
    automated prompts plus the per-mask loop that repeatedly calls
    :func:`build_conversation` and :func:`inference`.
    """

    pipe = None
    parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
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

    # print("Loading checkpoint:", model_args.model_name_or_path)
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
    data_args.mask_processor = model.get_segmentator().process_images
    data_args.is_multimodal = True
    if data_args.segmentation_config:
        data_args.register = COCORegister(data_args,is_eval=True)       # don't have to pass in annotations_config (during inference, not gt mask will be loaded)

    # clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        image_file = input(
            "Choose an image in inference_images folder." + "\n"
            "Examples: students.jpg | john_mayer.jpg | wii.jpg | cat.jpg | baseball.jpg | frisbee.jpg" + "\n"
            "Input image: "
        )
        input_image_path = image_file
        while not os.path.exists(os.path.join(data_args.image_folder, input_image_path)):
            image_file = input(
                f"Image file does not exist: {input_image_path}. Choose another one." + "\n"
                "Input image: "
            )
            input_image_path = image_file



        # initialize round_counter and conversation history
        round_counter = 0
        conv_dict = {
            "task": "segmentation",
            "base": "[null]",
            "conversations": []             # each round will be appended
        }
        all_masked_image = []           # image with mask
        all_masks_cropped = []          # cropped instance
        all_mask_encode_torch = []      # cropped instance preprocessed
        all_box_encode_torch = []       # bbox coords of cropped instance, preprocessed

        while True:
            round_counter += 1          # 1-indexed

            # print("mask encode length:", len(all_mask_encode_torch))

            if image_file in EXAMPLES:
                user_inputs = input(
                    f"----------------- Round {round_counter} -------------------" + "\n"
                    "Enter a segmentation query." + "\n"
                    "Here is an example of a multi-round conversation for this image:" + "\n\n"
                    f"{EXAMPLES[image_file]}"
                    "(Optional) Use the [REF:X] to indicate which round's (1-indexed) output you would like to use as a reference object." + "\n"
                    "Enter 'exit' to input a different image." + "\n"
                    f"Round {round_counter} query: "
                )
            else:
                user_inputs = input(
                    f"----------------- Round {round_counter} -------------------" + "\n"
                    "Enter a segmentation query." + "\n"
                    "(Optional) Use the [REF:X] to indicate which round's (1-indexed) output you would like to use as a reference object." + "\n"
                    "Enter 'exit' to input a different image." + "\n"
                    f"Round {round_counter} query: "
                )


            # re-select image
            if user_inputs == 'exit':
                break

            if matches := re.findall(r'\[REF:(\d+)\]', user_inputs):
                user_selected_idx = matches[0]
            else:
                user_selected_idx = None

            input_text = re.sub(r'\[REF:\d+\]', '', user_inputs)

            # print("Input Text:", input_text)
            # print("Encode idx:", user_selected_idx)

            # reset
            if user_inputs == 'clear history':
                round_counter = 0
                conv_dict = {
                    "task": "segmentation",
                    "base": "[null]",
                    "conversations": []             # each round will be appended
                }
                all_masked_image = []
                all_masks_cropped = []
                all_mask_encode_torch = []
                all_box_encode_torch = []
                continue

            inputs = build_conversation(
                model,
                tokenizer,
                conv_dict,
                round_counter,          # 1-indexed
                training_args,
                data_args,
                input_image_path,       # relative to data_args.image_folder
                input_text,
                user_selected_idx
            )

            image_with_mask, _, _, _ = inference(
                model,
                inputs,
                os.path.join(data_args.image_folder, input_image_path),       # absolute path
                round_counter,          # 0-indexed
                training_args,
                all_mask_encode_torch,
                all_box_encode_torch,
                all_masked_image,
                all_masks_cropped,
            )

            name = image_file.replace('.jpg', '')
            out_dir = os.path.join('./inference_results', name)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f'round_{round_counter}_text.txt')
            with open(out_file, "w") as f:
                f.write(user_inputs + "\n")
            out_file = os.path.join(out_dir, f'round_{round_counter}_output.jpg')
            Image.fromarray(image_with_mask).save(out_file)





if __name__ == '__main__':
    # Arguments taken from segllm/scripts/inference/launch_cli_demo.sh
    CHECKPOINT = 'Marlo-Z/SegLLM/all_data_checkpoint'
    ROOT_DIR = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/src/models/segllm'
    args = [
        "--deepspeed", os.path.join(ROOT_DIR, "scripts/deepspeed_configs/zero2.json"),
        "--model_name_or_path", CHECKPOINT,
        "--load", CHECKPOINT,
        "--image_folder", os.path.join(ROOT_DIR, "./inference_images"),
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
        "--output_dir", os.path.join(ROOT_DIR, "./out_dir"),
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
    ]

    main(args)

    # main()

import jsonargparse as argparse
import json
import os
import sys
from functools import partial
from pprint import pformat

import torch
import tqdm
import transformers
from transformers import BitsAndBytesConfig
import numpy as np

from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib

sys.path.append(os.path.realpath(os.path.join(__file__, '../../..'))) # Add src directory to import eval tools
from models.PLUM.utils.vqa_dataset import TextVQADataset  # import from plum utils
from models.PLUM.utils.m4c_evaluator import TextVQAAccuracyEvaluator

from models.lisa.utils.dataset import collate_fn_vqa

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX
)
import logging
import coloredlogs

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate PixelLM on TextVQA Dataset")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    # parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument(
        "--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
        help="precision for inference"
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="textvqa|val", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument('--limit_batches', default=0, type=int)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--predictions_path', type=str, default='predictions.json')
    
    parser.add_argument("--exp_name", default="lisa-13b-vqa-eval", type=str)

    # PixelLM specific arguments
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--use_expand_question_list", action="store_true", default=False)
    parser.add_argument("--masks_process_with_clip", default=False, action="store_true")
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--pad_val_clip_images", action="store_true", default=False)

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)

    print(f"*(validate_vqa) >> Loading model checkpoints from [ {args.version} ]*")

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    if args.seg_token_num == 1:  # args.seg_token_num * args.image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    elif args.seg_token_num > 1:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]        
    else:
        raise ValueError(f"args.seg_token_num cannot be 0 or negative.")

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # build LISA model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_args = {
        "seg_token_idx": args.seg_token_idx,
        "vision_tower": args.vision_tower,
        "preprocessor_config": args.preprocessor_config,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_token_num": args.seg_token_num,
        "logger": logger,
        "tokenizer": tokenizer,
        "local_rank": args.local_rank,
        "pad_val_clip_images": args.pad_val_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,  # Flag for using PixelLM's lightweight decoder
        "separate_mm_projector": args.separate_mm_projector,
        "masks_process_with_clip": args.masks_process_with_clip,
        "image_feature_scale_num": args.image_feature_scale_num,
    }
    
    print(">> model_args: ", model_args)

    # Load model
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.half,
        "fp32": torch.float32,
    }[args.precision]
    
    model = PixelLMForCausalLM.from_pretrained(
        args.version, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **model_args
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # initialize vision modules in PixelLM
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_pixellm_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    if args.resize_vision_tower_size == 224:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.eval()
    model.to(args.device)

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]
    model.resize_token_embeddings(len(tokenizer))
    
    # PixelLM has L * N number of seg tokens 
    # - L: pre-defined number of intermediate layers (corresponds to 'args.image_feature_scale_num')
    # - N: pre-defined # of segmentation tokens in a codebook) (corresponds to 'args.seg_token_num')
    token_num = args.seg_token_num * args.image_feature_scale_num

    world_size = 1 # torch.cuda.device_count()  # NOTE: We keep the world size to 1 for experiment
    is_distributed = world_size > 1
    
    # build dataset for validation
    val_dataset = TextVQADataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower
    )

    # build DataLoader
    if val_dataset is not None:
        assert args.val_batch_size == 1, "Currently supports batch_size=1 for segmentation eval"
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(
                collate_fn_vqa,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    else:
        raise ValueError("No validation dataset provided. Exiting.")

    # run evaluation on the partonomy dataset
    validate(val_loader, model, tokenizer, args, parser)


def to_pixellm_inputs(batch: dict, tokenizer: transformers.AutoTokenizer):
    input_dict = {
        'images_clip': batch['images_clip'],
        'images': batch['images'],
        "input_ids": batch['input_ids'],
        'resize_list': batch['resize_list'],
        'clip_resize_list': batch['resize_list'],  # --- Account for the 'clip_resize_list' in PixelLM/utils/multi_reason_seg_val_dataset.py ---
        'original_size_list': batch['label_list'],
        'tokenizer': tokenizer
    }
    return input_dict


@torch.no_grad()
def validate(val_loader, model, tokenizer, args, parser):
    '''
    Validation function specialized for 'textvqa' QA pairs
    '''
    model.eval()

    skipped_instance_num = 0
    if args.limit_batches > 0:
        n_batches = min(args.limit_batches, len(val_loader))
    else:
        n_batches = len(val_loader)

    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(range(n_batches), desc="Validating Partonomy with PixelLM")

    total_accuracy = 0
    output_dicts = []
    for idx in pbar:
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
        else:
            to_dtype = lambda x: x.float()
            
        batch['images'] = to_dtype(batch['images'])
        batch['images_clip'] = to_dtype(batch['images_clip'])

        # Forward Pass
        output_ids, pred_masks, _, _ = model.evaluate(**to_pixellm_inputs(batch, tokenizer))  # multiple_choice == True only for Partonomy - this nees to be set to return token-level loss

        ground_truth_text = batch['sampled_classes_list'][0]
        
        # get the output_ids and output_text and evaluate against the ground truth answer texts
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        evaluator = TextVQAAccuracyEvaluator()
        accuracy = evaluator.eval_pred_list([{'pred_answer': output_text, 'gt_answers': ground_truth_text}])
        logger.info(f"==> Output Text: {output_text} | Ground Truth Text: {ground_truth_text} | Accuracy: {accuracy}")
        
        total_accuracy += accuracy
        
        output_dict = {
            "image_path": batch['image_paths'][0],
            "output_text": output_text,
            "ground_truth_text": ground_truth_text,
            "accuracy": accuracy
        }
        output_dicts.append(output_dict)
        
    if not os.path.exists(args.predictions_path):
        os.makedirs(args.predictions_path, exist_ok=True)

    pred_file_path = os.path.join(args.predictions_path, 'predictions.json')
    if os.path.exists(pred_file_path):
        # read and calculate the total_accuracy
        with open(pred_file_path, 'r') as f:
            existing_output_dicts = json.load(f)
        existing_output_dicts.extend(output_dicts)
        total_accuracy = sum([d['accuracy'] for d in existing_output_dicts]) / len(existing_output_dicts)
        print(f"==> Total Accuracy: {total_accuracy}")
    else:
        total_accuracy = total_accuracy / n_batches
        print(f"==> Total Accuracy: {total_accuracy}")
        with open(pred_file_path, 'w') as f:
            json.dump(output_dicts, f)
        

    logger.info(f">> [validate_vqa] Skipped instances: {skipped_instance_num}")


if __name__ == "__main__":
    coloredlogs.install(level='DEBUG')

    main(sys.argv[1:])

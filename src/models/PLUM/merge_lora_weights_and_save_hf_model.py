import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.PLUM import PLUMForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    
    # PLUM-specific arguments
    parser.add_argument("--use_teacher_ref", action="store_true", default=False)
    parser.add_argument("--use_bidir_bio", action="store_true", default=False)
    parser.add_argument("--use_feedback_loop", action="store_true", default=False)
    parser.add_argument("--use_crf_bio", action="store_true", default=False)
    parser.add_argument("--use_hinge_loss", action="store_true", default=False)
    parser.add_argument("--pred_binary_span", action="store_true", default=False)
    parser.add_argument("--use_cross_attn_bio", action="store_true", default=False)
    
    # Training arguments that affect model structure
    parser.add_argument("--bidir_nhead", default=8, type=int)
    parser.add_argument("--bidir_dim_feedforward", default=2048, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_type", default="focal_tversky", type=str)
    parser.add_argument("--dice_loss_weight", default=8, type=float)
    parser.add_argument("--dice_scale_factor", default=1000.0, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--seg_cls_loss_weight", default=8, type=float)
    parser.add_argument("--seg_cls_loss_per_cls_weight", default=[0.1, 1.0, 1.0], type=list)
    parser.add_argument("--kld_loss_weight", default=0.1, type=float)
    parser.add_argument("--kld_sigma", default=1.0, type=float)
    parser.add_argument("--focal_tversky_alpha", default=0.7, type=float)
    parser.add_argument("--focal_tversky_beta", default=0.3, type=float)
    
    # LoRA arguments
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--train_mask_prompt_encoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./plum_model_ckpt", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # complete model arguments matching training script
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_type": args.dice_type,
        "dice_loss_weight": args.dice_loss_weight,
        "dice_scale_factor": args.dice_scale_factor,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_cls_loss_weight": args.seg_cls_loss_weight,
        "seg_cls_loss_per_cls_weight": args.seg_cls_loss_per_cls_weight,
        "kld_loss_weight": args.kld_loss_weight,
        "kld_sigma": args.kld_sigma,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "use_teacher_ref": args.use_teacher_ref,
        "use_bidir_bio": args.use_bidir_bio,
        "use_crf_bio": args.use_crf_bio,
        "use_hinge_loss": args.use_hinge_loss,
        "use_feedback_loop": args.use_feedback_loop,
        "pred_binary_span": args.pred_binary_span,
        "use_cross_attn_bio": args.use_cross_attn_bio,
        "train_mask_prompt_encoder": args.train_mask_prompt_encoder,
        "focal_tversky_alpha": args.focal_tversky_alpha,
        "focal_tversky_beta": args.focal_tversky_beta,
        "bidir_nhead": args.bidir_nhead,
        "bidir_dim_feedforward": args.bidir_dim_feedforward,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    print("Creating PLUM model with complete arguments...")
    model = PLUMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
    )
    
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize vision modules
    print("Initializing vision modules...")
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    
    # Initialize PLUM modules
    print("Initializing PLUM modules...")
    model.get_model().initialize_plum_modules(model.get_model().config, **model_args)

    # Create mask_pooler structure if needed
    if model.use_feedback_loop:
        print("Creating mask pooler structure...")
        model.initialize_mask_pooler()

    # Set up LoRA
    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower", 
                                "mm_projector",
                                "text_hidden_fcs",
                                "token_to_mask_fcs"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        print(f"LoRA target modules: {lora_target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # load the trained weights - overwrite any random initialization
    print(">> loading weights from: ", args.weight)
    state_dict = torch.load(args.weight, map_location="cpu")
    
    # remove teacher_llm weights if present (not needed for inference)
    state_dict = {k: v for k, v in state_dict.items() if "teacher_llm" not in k}
    
    # load with strict=False to handle any missing keys gracefully
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print("Missing keys:", missing_keys[:10])  # Show first 10
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys[:10])  # Show first 10
    
    print(">> loaded weights from: ", args.weight)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    # save the merged model
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    
    print(f"Saving merged model to: {args.save_path}")
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)
    
    print("Merge completed successfully!")


if __name__ == "__main__":
    main(sys.argv[1:]) 
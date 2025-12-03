#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <7B|13B>"
    exit 1
fi

# assign PixelLM model size to MODEL_SIZE
MODEL_SIZE=$1

if [ "$MODEL_SIZE" != "7B" ] && [ "$MODEL_SIZE" != "13B" ]; then
    echo "Error: Invalid model size. Please enter either '7B' or '13B'."
    exit 1
fi

# Check if GPU ids are passed as an argument; if not, ask for it.
if [ -z "$2" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$2
fi

CUDA_VISIBLE_DEVICES=${gpu_ids} python validate_partonomy_pixellm.py \
          --version="./runs/PixelLM-${MODEL_SIZE}/hf_model" \
          --dataset_dir="dataset/partonomy_descriptors/" \
          --exp_name="pixellm-${MODEL_SIZE}" \
          --eval_only \
          --pad_val_clip_images \
          --preprocessor_config='./configs/preprocessor_448.json' \
          --resize_vision_tower \
          --resize_vision_tower_size=448 \
          --vision_tower_for_mask \
          --seg_token_num=3 \
          --image_feature_scale_num=2 \
          --separate_mm_projector

	# --vision-tower='openai/clip-vit-large-patch14-336'
	# vision_tower_for_mask - flag for building and using its own mask decoder vs. SAM

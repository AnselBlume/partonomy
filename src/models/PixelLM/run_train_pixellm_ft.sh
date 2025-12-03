#!/bin/bash

# 1) gpu_ids
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

# 2) BATCH_SIZE
if [ -z "$2" ]; then
    read -p "Enter per device batch size: " BATCH_SIZE
else
    BATCH_SIZE=$2
fi

# 3) partonomy_dataset_split_name
if [ -z "${3}" ]; then
    read -p "Enter partonomy_dataset_split_name (partimagenet|paco_lvis|pascal_part|partonomy): " partonomy_dataset_split_name
    PARTONOMY_DATASET_SPLIT_NAME=${partonomy_dataset_split_name:-"partimagenet"}
else
    PARTONOMY_DATASET_SPLIT_NAME=${3}
fi

LLAVA_MODEL_PATH="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
PIXELLM_MODEL_PATH="./runs/PixelLM-13B/hf_model"

# Find a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} train_ds.py \
  --version=${PIXELLM_MODEL_PATH} \
  --dataset_dir='./dataset' \
  --dataset="explanatory_seg" \
  --partonomy_dataset_split=$PARTONOMY_DATASET_SPLIT_NAME \
  --sample_rates="1" \
  --exp_name="pixellm-13b" \
  --explanatory_seg_data \
  --vision-tower='openai/clip-vit-large-patch14-336' \
  --seg_token_num=1 \
  --num_classes_per_question=3 \
  --batch_size=${BATCH_SIZE} \
  --grad_accumulation_steps 10 \
  --pad_train_clip_images \
  --preprocessor_config='./configs/preprocessor_448.json' \
  --resize_vision_tower \
  --epochs 3 \
  --train_mask_decoder \
  --resize_vision_tower_size=448 \
  --vision_tower_for_mask \
  --use_expand_question_list \
  --image_feature_scale_num=1 \
  --separate_mm_projector
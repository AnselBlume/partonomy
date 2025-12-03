#!/bin/bash

# USAGE: CUDA_VISIBLE_DEVICES=0 ./scripts/inference/launch_gradio_demo.sh

CHECKPOINT=Marlo-Z/SegLLM/all_data_checkpoint

python3 llava/train/inference_gradio.py \
    --deepspeed ./scripts/deepspeed_configs/zero2.json \
    --model_name_or_path $CHECKPOINT \
    --load $CHECKPOINT \
    --image_folder ./images_folder \
    --mm_use_seg True \
    --segmentator hipie \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_vision_select_feature patch \
    --mm_use_im_patch_token False \
    --bf16 True \
    --lora_enable False \
    --split_loading False \
    --version plain \
    --mm_use_gen True \
    --output_dir ./out_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --output_text \
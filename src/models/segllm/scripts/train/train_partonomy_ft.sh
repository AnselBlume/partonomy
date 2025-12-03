#!/bin/bash

CHECKPOINT=Marlo-Z/SegLLM/all_data_checkpoint

echo "OUTPUT_DIR=${OUTPUT_DIR}"

deepspeed \
    --include "localhost:${LOCAL_HOST:-3,4,5,6}" \
    --master_port "${MASTER_PORT:-12345}" \
    llava/train/train_mem_ft.py \
    --deepspeed ./scripts/deepspeed_configs/zero2.json \
    --model_name_or_path "${CHECKPOINT}" \
    --image_folder ./images_folder \
    --annotation_folder ./annotations_folder \
    --conversation_folder ./conversations_folder/all_data_mix_train \
    --val_dataset reason_seg_test.json \
    --segmentation_config ./scripts/annotation_configs/train/all_data_mix_train.yaml \
    --conversation_config ./scripts/conversation_configs/train/all_data_mix_train.yaml \
    --output_dir "${OUTPUT_DIR:-trained_checkpoints}" \
    --eval_use_gt_mask_encode True \
    --limit_rounds 1 \
    --mm_use_bbox_encode True \
    --lora_enable False \
    --split_loading False \
    --version plain \
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
    --fp16 False \
    --tf32 False \
    --mm_use_gen True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --max_steps 1000 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --eval_steps 500 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb ${@:1} \
    --output_text \
    --do_eval \

# --num_train_epochs 2 \
# --save_total_limit 2 \
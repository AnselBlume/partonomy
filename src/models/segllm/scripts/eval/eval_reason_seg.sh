#!/bin/bash

CHECKPOINT=Marlo-Z/SegLLM/all_data_checkpoint
CONV_DIR=all_data_mix_val
RESULTS_PATH=./val_results/reason_seg_eval_results.txt

SPLITS=("val" "test")
for split in "${SPLITS[@]}"
do
    SEG_CONFIG_FILE=reason_seg_${split}.yaml
    VAL_DATA=reason_seg_${split}.json
    deepspeed \
        --include "localhost:${LOCAL_HOST}" \
        --master_port 12342 \
        llava/train/train_mem.py \
        --deepspeed ./scripts/deepspeed_configs/zero2.json \
        --model_name_or_path liuhaotian/llava-v1.5-7b \
        --load $CHECKPOINT \
        --image_folder ./images_folder \
        --annotation_folder ./annotations_folder \
        --conversation_folder ./conversations_folder/${CONV_DIR} \
        --segmentation_config ./scripts/annotation_configs/val/${SEG_CONFIG_FILE} \
        --val_dataset $VAL_DATA \
        --val_results_save_file $RESULTS_PATH \
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
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --eval_steps 1000 \
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
        --eval_only \
        --output_dir ./val_output \
        --eval_use_gt_mask_encode True \
        --per_device_eval_batch_size 1
done
#!/bin/bash

LLAVA_MODEL_PATH=liuhaotian/llava-llama-2-13b-chat-lightning-preview
DATASET_DIR=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset
VISION_MODEL_PATH=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/weights/sam_vit_h_4b8939.pth
# CKPT_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft
# CKPT_PATH=plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft_bidirbio_2048_maxlen512_epochs20_bidir_bio_feedback_loop_exp_seg_train_prompt_enc_exp_seg_val_partonomy_partimagenet_paco_lvis_pascal_part_ckpt_model

CKPT_TYPE=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg
CKPT_PATH=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg_bidirbio_2048_maxlen512_epochs25_bidir_bio_feedback_loop_train_prompt_enc_srates_9_5_5_1_ckpt_model
# CKPT_PATH=plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg_maxlen512_epochs25_kld_loss_0_dice_loss_8_feedback_loop_train_prompt_enc_srates_9_5_5_1_ckpt_model  # No bidirection layer

# CKPT_TYPE=plum-13b_kld_0.1_dice_8_v1_0shot_w_reasonseg
# CKPT_PATH=plum-13b_kld_0.1_dice_8_v1_0shot_w_reasonseg_bidirbio_2048_maxlen512_epochs25_bidir_bio_feedback_loop_train_prompt_enc_focal_tversky_04_06_srates_9_5_5_1_ckpt_model

# Convert DeepSpeed checkpoint to PyTorch format
echo "Converting DeepSpeed checkpoint to PyTorch format..."
cd ./runs/${CKPT_TYPE}/${CKPT_PATH} && python zero_to_fp32.py . ../pytorch_model.bin
cd ../../..

## --use_bidir_bio \ <- TODO: Must revive this for the final model!

echo "Merging LoRA weights..."
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version=$LLAVA_MODEL_PATH \
  --weight="./runs/${CKPT_TYPE}/pytorch_model.bin" \
  --save_path="./runs/${CKPT_TYPE}/merged_model_9551" \
  --vision_pretrained=$VISION_MODEL_PATH \
  --vision-tower="openai/clip-vit-large-patch14" \
  --use_bidir_bio \
  --use_feedback_loop \
  --train_mask_prompt_encoder \
  --train_mask_decoder \
  --bidir_nhead=8 \
  --bidir_dim_feedforward=2048 \
  --ce_loss_weight=1.0 \
  --dice_type="focal_tversky" \
  --dice_loss_weight=8 \
  --dice_scale_factor=1000.0 \
  --bce_loss_weight=2.0 \
  --seg_cls_loss_weight=8 \
  --kld_loss_weight=0.1 \
  --kld_sigma=1.0 \
  --focal_tversky_alpha=0.7 \
  --focal_tversky_beta=0.3 \
  --out_dim=256 \
  --model_max_length=512 \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --lora_target_modules="q_proj,v_proj" \
  --use_mm_start_end

echo "Merge completed! Now you can compare the fixed merged model." 
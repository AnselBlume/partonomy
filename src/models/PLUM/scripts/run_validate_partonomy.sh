#!/bin/bash

# Script for validating PLUM on the Partonomy eval dataset

if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi
# /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partonomy
MODEL_TYPE="plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft"
MODEL_CKPT_PATH="/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/${MODEL_TYPE}/merged_model_joint_fixed"
# MODEL_CKPT_PATH=/shared/nas2/jk100/partonomy_private/src/models/PLUM/runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot/merged_model
VISION_MODEL_PATH=/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/weights/sam_vit_h_4b8939.pth

# CUDA_VISIBLE_DEVICES=$gpu_ids python validate_partonomy.py \
CUDA_VISIBLE_DEVICES=$gpu_ids python validate_partonomy.py \
	  --version=$MODEL_CKPT_PATH \
	  --dataset_path='/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset/partonomy/partonomy_qa_pairs.json' \
	  --vision_pretrained=$VISION_MODEL_PATH \
	  --exp_name="${MODEL_TYPE}" \
	  --precision='bf16' \
	  --use_bidir_bio \
	  --use_feedback_loop \
	  --eval_only
# Script for chatting with PLUM
# E.g., 
# input text: What parts does the commercial airplane have in common with a military fighter jet?
# input text: What parts do you see in this airplane?
# input image: ROOT_PATH/dataset/vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages/2007_000033.jpg


# check for GPU ids
if [ -z "$1" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$1
fi

# MODEL_CKPT_PATH=runs/plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft/merged_model_joint_fixed
MODEL_CKPT_PATH=runs/plum-13b_kld_0.1_focal_tversky_8_v1_0shot_w_reasonseg/merged_model_9551

CUDA_VISIBLE_DEVICES=${gpu_ids} python chat.py \
	  --version=$MODEL_CKPT_PATH \
	  --precision='bf16' \
      --use_bidir_bio \
      --use_feedback_loop
    #   --prompt_user_input
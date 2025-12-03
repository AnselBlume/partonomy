LLAVA_MODEL_PATH=liuhaotian/llava-llama-2-13b-chat-lightning-preview

CKPT_TYPE=pixellm-13b
CKPT_PATH=pixellm-13b-ft-partimagenet-paco_lvis-pascal_part_ckpt

# cd ./runs/${CKPT_TYPE}/${CKPT_PATH} && python zero_to_fp32.py . ../pytorch_model.bin

CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version=${LLAVA_MODEL_PATH} \
  --weight="./runs/${CKPT_TYPE}/pytorch_model.bin" \
  --save_path="./runs/${CKPT_TYPE}/merged_model_joint" \
  --seg_token_num=3 \
  --image_feature_scale_num=2 \
  --preprocessor_config="" \
  --resize_vision_tower \
  --resize_vision_tower_size=224
  
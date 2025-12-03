python3 ./chat.py --version="./runs/PixelLM-13B/hf_model" --precision='bf16' --seg_token_num=3  \
	--pad_train_clip_images \
	--preprocessor_config='./configs/preprocessor_448.json' \
	 --resize_vision_tower \
	 --resize_vision_tower_size=448 \
	 --vision-tower='openai/clip-vit-large-patch14-336' \
	 --vision_tower_for_mask \
	 --image_feature_scale_num=2 \
	 --separate_mm_projector

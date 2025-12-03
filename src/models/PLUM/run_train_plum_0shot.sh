#!/bin/bash

# Script for pre-training PLUM

# fnd a random free port
FREE_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# 1) Model size
if [ -z "$1" ]; then
    read -p "Enter PLUM model size (13B or 7B): " model_size
else
    model_size=$1
fi

# 2) kld_loss_weight
if [ -z "$2" ]; then
    read -p "Enter kld_loss_weight: " kld_loss_weight
else
    kld_loss_weight=$2
fi

# 3) dice_loss_weight
if [ -z "$3" ]; then
    read -p "Enter dice_loss_weight: " dice_loss_weight
else
    dice_loss_weight=$3
fi

# 4) seg_cls_loss_weight
if [ -z "$4" ]; then
    read -p "Enter seg_cls_loss_weight: " seg_cls_loss_weight
else
    seg_cls_loss_weight=$4
fi

# 5) Teacher reference input
if [ -z "$5" ]; then
    read -p "Enable teacher reference? (true/false): " teacher_ref_input
else
    teacher_ref_input=$5
fi

# 6) dice_type
if [ -z "$6" ]; then
    read -p "Use Focal-Tversky Loss? (true/false): " dice_type
else
    dice_type=$6
fi

# 7) BATCH_SIZE
if [ -z "$7" ]; then
    read -p "Enter per device batch size: " BATCH_SIZE
else
    BATCH_SIZE=$7
fi

# 8) precision
if [ -z "$8" ]; then
    read -p "Enable precision? (fp32/bf16/fp16): " precision
else
    precision=$8
fi

# 9) gpu_ids
if [ -z "$9" ]; then
    read -p "Enter GPU IDs (comma-separated for multiple GPUs): " gpu_ids
else
    gpu_ids=$9
fi

if [ "$dice_type" == "true" ]; then
    DICE_TYPE="focal_tversky"
    
    # 10) FOCAL_ALPHA
    if [ -z "${10}" ]; then
        read -p "Enter FOCAL_ALPHA (default 0.7): " input_focal_alpha
        FOCAL_ALPHA=${input_focal_alpha:-0.7}
    else
        FOCAL_ALPHA=${10}
    fi

    # 11) FOCAL_BETA
    if [ -z "${11}" ]; then
        read -p "Enter FOCAL_BETA (default 0.3): " input_focal_beta
        FOCAL_BETA=${input_focal_beta:-0.3}
    else
        FOCAL_BETA=${11}
    fi

else
    DICE_TYPE="dice"
    FOCAL_ALPHA=0.3
    FOCAL_BETA=0.7
fi

# Teacher reference flag
if [ "$teacher_ref_input" == "true" ]; then
    TEACHER_REF="--use_teacher_ref"
else
    TEACHER_REF=""
fi

# Convert model_size to lower case
model_size=$(echo "$model_size" | tr '[:upper:]' '[:lower:]')

LLAVA_MODEL_PATH=liuhaotian/llava-llama-2-13b-chat-lightning-preview

if [ "$model_size" = "13b" ]; then
    LLAVA_MODEL_PATH="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
elif [ "$model_size" = "7b" ]; then
    LLAVA_MODEL_PATH="weights/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview"
else
    echo "Invalid model size. Please use 13B or 7B."
    exit 1
fi

ROOT_PATH=""  # TODO: Set the root path that contains both the weights and the dataset
VISION_MODEL_PATH=${ROOT_PATH}/weights/sam_vit_h_4b8939.pth
DATASET_DIR=${ROOT_PATH}/dataset

# NOTE: 0shot means no finetuning on Partonomy dataset
EXP_NAME="plum-${model_size}_kld_${kld_loss_weight}_${DICE_TYPE}_${dice_loss_weight}_v1_0shot_w_reasonseg"

echo "Running experiment: $EXP_NAME"
echo " - kld_loss_weight=${kld_loss_weight}"
echo " - dice_loss_weight=${dice_loss_weight}"
echo " - teacher_ref_input=${teacher_ref_input}"
echo " - DICE_TYPE=${DICE_TYPE}"
echo " - FOCAL_ALPHA=${FOCAL_ALPHA}"
echo " - FOCAL_BETA=${FOCAL_BETA}"
echo " - Using GPUs: ${gpu_ids}"


# training
deepspeed --include localhost:${gpu_ids} --master_port=${FREE_PORT} plum_train_ds.py \
    --version=$LLAVA_MODEL_PATH \
    --dataset_dir=$DATASET_DIR \
    --vision_pretrained=$VISION_MODEL_PATH \
    --dataset="sem_seg||refer_seg||vqa||reason_seg" \
    --val_dataset="reason_seg|val" \
    --sample_rates="9,5,5,1" \
    --batch_size=$BATCH_SIZE \
    --grad_accumulation_steps 10 \
    --use_bidir_bio \
    --use_feedback_loop \
    --ce_loss_weight=1.0 \
    --dice_loss_weight=${dice_loss_weight} \
    --bce_loss_weight=2.0 \
    --kld_loss_weight=${kld_loss_weight} \
    --seg_cls_loss_weight=${seg_cls_loss_weight} \
    --exp_name="$EXP_NAME" \
    $TEACHER_REF \
    --precision="$precision" \
	--model_max_length 512 \
    --dice_scale_factor 1000.0 \
    --epochs 25 \
    --train_mask_prompt_encoder \
    --focal_tversky_alpha=$FOCAL_ALPHA \
    --focal_tversky_beta=$FOCAL_BETA \
    --bidir_dim_feedforward 2048 \
    --auto_resume
import argparse
import os
import json
import types
import shutil
import sys
import wandb
import time
from functools import partial
from tqdm import tqdm
from itertools import islice, chain
from jsonargparse import Namespace, ArgumentParser

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

try:
    from deepspeed.utils.zero_to_fp32 import GatheredParameters
except ImportError:
    # allows the code to run when DeepSpeed isn't installed
    from contextlib import contextmanager
    @contextmanager
    def GatheredParameters(params, modifier_rank=None):
        yield

from model.PLUM import PLUMForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.explanatory_seg_dataset import ExplanatorySegDataset
# NOTE: ExplanatorySegDatasetsAdapter turns ExplanatorySegInstance to a tuple
# NOTE: Make sure you use it correctly.
from utils.explanatory_seg_datasets_adapter import ExplanatorySegDatasetsAdapter

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.question_type import QuestionType

# add IoUEvaluator evaluation functionality
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from evaluation.prediction import Prediction, Predictions
from evaluation.rle_dict import get_mask_rle_dicts
from evaluation.evaluators import (
    IoUEvaluator, IoUEvaluatorConfig, MaskMatchingStrategy,
    PartTextEvaluator, PartTextEvaluatorConfig,
    MCTextEvaluator, Reduction
)


def parse_args(args):
    parser = ArgumentParser(description="PLUM Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--zero_shot_ckpt_path", default="", type=str)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1536, type=int)  # Originally, 512
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument(
        "--explanatory_seg_data", action="store_true"
    )
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project-name", default="plum-training")
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)  # llava_instruct_150k ; llava_v1_5_mix665k
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    
    # Partonomy args
    parser.add_argument("--partonomy_train_dataset_path", default="/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/{dataset_split}/{dataset_split}_qa_pairs_train.json", type=str)
    parser.add_argument("--partonomy_val_dataset_path", default="/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/{dataset_split}/{dataset_split}_qa_pairs_val.json", type=str)
    parser.add_argument("--partonomy_dataset_split", default="", type=str)  # e.g., "partimagenet|pascal_part|paco_lvis|partonomy"
    parser.add_argument("--partonomy_question_type", default="identification_with_label", type=str)
    parser.add_argument("--sample_one_question_per_image", action="store_true")
    parser.add_argument("--random_seed", default=42, type=int)
    
    # Evaluators
    parser.add_argument('--metrics.iou_evaluator_config', default=IoUEvaluatorConfig(), type=IoUEvaluatorConfig) # NOTE: Change to either reduction='macro' or reduction='micro'
    parser.add_argument('--metrics.part_text_evaluator_config', default=PartTextEvaluatorConfig(), type=PartTextEvaluatorConfig)
    parser.add_argument('--predictions_path', default='/shared/nas2/jk100/partonomy_private/results/predictions.json', type=str)
    parser.add_argument('--output_predictions', action='store_true', default=False)
    
    # Training args
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="plum", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_type", default="dice", type=str)  # [dice, focal_tversky]
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--dice_scale_factor", default=1000.0, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)  # 2.0 in LISA
    parser.add_argument("--kld_loss_weight", default=0.5, type=float)
    parser.add_argument("--kld_sigma", default=1.0, type=float)
    parser.add_argument("--seg_cls_loss_weight", default=0.5, type=float)
    parser.add_argument("--seg_cls_loss_per_cls_weight", default=[0.1, 1.0, 1.0], type=list, help="per class weight for seg_cls_loss")
    parser.add_argument("--use_teacher_ref", action="store_true", default=False)
    parser.add_argument("--use_bidir_bio", action="store_true", default=False)
    parser.add_argument("--use_cross_attn_bio", action="store_true", default=False)
    parser.add_argument("--use_hinge_loss", action="store_true", default=False)
    parser.add_argument("--use_crf_bio", action="store_true", default=False)
    parser.add_argument("--pred_binary_span", action="store_true", default=False)
    parser.add_argument("--focal_tversky_alpha", default=0.7, type=float)
    parser.add_argument("--focal_tversky_beta", default=0.3, type=float)
    parser.add_argument("--limit_batches", default=None, type=int)
    parser.add_argument("--use_feedback_loop", action="store_true", default=False)
    
    # bidirectionalencoderblock params
    parser.add_argument("--bidir_nhead", default=8, type=int)
    parser.add_argument("--bidir_dim_feedforward", default=2048, type=int)
    
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=5, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--train_mask_prompt_encoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--log_dir", default="./runs", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    
    parser.add_argument("--debug", action="store_true", default=False)
    # Add distributed argument to prevent jsonargparse validation error
    parser.add_argument("--distributed", action="store_true", default=False, help="whether training is distributed across multiple GPUs")

    return parser.parse_args(args), parser


def main(args):
    args, parser = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        
        run_name = (f"{args.exp_name}_maxlen{args.model_max_length}"
                      f"_epochs{args.epochs}_bsz{args.batch_size}"
                      f"_bce_loss_{args.bce_loss_weight}_kld_loss_{args.kld_loss_weight}"
                      f"_{args.dice_type}_loss_{args.dice_loss_weight}_{args.dice_scale_factor}_lr{args.lr}"
                    )
        run_ckpt_path = (f"{args.exp_name}_maxlen{args.model_max_length}"
                            f"_epochs{args.epochs}"
                            f"_kld_loss_{int(args.kld_loss_weight)}"
                            f"_{args.dice_type}_loss_{int(args.dice_loss_weight)}"
                        )
        if args.use_teacher_ref:
            run_name += "_teacher"
            run_ckpt_path += "_teacher"  # e.g., plum-13b_kld_0_dice_4_accum_10_maxlen512_epochs50_segloss_2_bce_loss_2_kld_loss_0_dice_loss_4_teacher
        if args.use_bidir_bio:
            run_name = (f"{args.exp_name}_bidirbio_{args.bidir_dim_feedforward}_maxlen{args.model_max_length}"
                        f"_epochs{args.epochs}_bsz{args.batch_size}"
                        f"_lr{args.lr}"
                        )
            run_ckpt_path = (f"{args.exp_name}_bidirbio_{args.bidir_dim_feedforward}_maxlen{args.model_max_length}"
                             f"_epochs{args.epochs}"
                            )
            run_name += "_bidir_bio"
            run_ckpt_path += "_bidir_bio"
        if args.use_crf_bio:
            run_name += "_crf_bio"
            run_ckpt_path += "_crf_bio"
        if args.use_hinge_loss:
            run_name += "_hinge_loss"
            run_ckpt_path += "_hinge_loss"
        if args.use_feedback_loop:
            run_name += "_feedback_loop"
            run_ckpt_path += "_feedback_loop"
        if args.use_cross_attn_bio:
            run_name += "_cross_attn_bio"
            run_ckpt_path += "_cross_attn_bio"
        if args.explanatory_seg_data:
            run_name += "_exp_seg"
            run_ckpt_path += "_exp_seg"
        if args.train_mask_prompt_encoder:
            run_name += "_train_prompt_enc"
            run_ckpt_path += "_train_prompt_enc"
        if args.pred_binary_span:
            if len(args.seg_cls_loss_per_cls_weight) > 2:
                args.seg_cls_loss_per_cls_weight = [0.1, 1.0]
            run_name += "_binary_span"
            run_ckpt_path += "_binary_span"
        if args.vision_tower == "naclip":
            run_name += "_naclip"
            run_ckpt_path += "_naclip"
        if args.explanatory_seg_data:
            run_name += "_exp_seg_val"
            run_ckpt_path += "_exp_seg_val"
        if args.focal_tversky_alpha != 0.7 and args.focal_tversky_beta != 0.3:
            run_name += f"_focal_tversky_{str(args.focal_tversky_alpha).replace('.', '')}_{str(args.focal_tversky_beta) .replace('.', '')}"
            run_ckpt_path += f"_focal_tversky_{str(args.focal_tversky_alpha).replace('.', '')}_{str(args.focal_tversky_beta).replace('.', '')}"
        if args.partonomy_dataset_split:
            run_name += f"_partonomy_{'_'.join(args.partonomy_dataset_split.split('|'))}"
            run_ckpt_path += f"_partonomy_{'_'.join(args.partonomy_dataset_split.split('|'))}"
        
        sample_rates = args.sample_rates.split(",")
        run_name += f"_srates_{'_'.join(sample_rates)}"
        run_ckpt_path += f"_srates_{'_'.join(sample_rates)}"
        
        print(">> (train) run_name: ", run_name)
        print(">> (train) run_ckpt_path: ", run_ckpt_path)
        
        # wandb logging
        import dataclasses
        from enum import Enum
        
        config_for_wandb = {}
        for arg_key, arg_value in vars(args).items():
            if dataclasses.is_dataclass(arg_value) and not isinstance(arg_value, type):
                # Convert dataclass instance to dict and sanitize enums
                try:
                    dict_representation = dataclasses.asdict(arg_value)
                    sanitized_dict = {}
                    for k_inner, v_inner in dict_representation.items():
                        if isinstance(v_inner, Enum):
                            sanitized_dict[k_inner] = v_inner.name
                        # elif dataclasses.is_dataclass(v_inner) and not isinstance(v_inner, type): # Optional: handle nested dataclasses
                        #     # This part might need more robust handling if deeply nested structures with enums exist
                        #     nested_dc_dict = dataclasses.asdict(v_inner)
                        #     sanitized_nested_dict = {nk: nv.name if isinstance(nv, Enum) else nv for nk, nv in nested_dc_dict.items()}
                        #     sanitized_dict[k_inner] = sanitized_nested_dict
                        else:
                            sanitized_dict[k_inner] = v_inner
                    config_for_wandb[arg_key] = sanitized_dict
                except Exception as e:
                    print(f"Warning: Could not fully serialize dataclass arg {arg_key} to dict for wandb: {e}. Using str() as fallback.")
                    config_for_wandb[arg_key] = str(arg_value)
            elif isinstance(arg_value, Enum):
                 config_for_wandb[arg_key] = arg_value.name
            else:
                config_for_wandb[arg_key] = arg_value
                
        wandb.init(
            project=args.wandb_project_name, 
            name=run_name,
            config=config_for_wandb # Use the sanitized config
        )

        epoch_metrics = {}  # dict to save metric outputs
        metrics_path = os.path.join(args.log_dir, f"{run_name}.json")
        latest_epoch = 0
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                epoch_metrics = json.load(f)
                latest_epoch = max(map(int, list(epoch_metrics.keys())))

    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_type": args.dice_type,
        "dice_loss_weight": args.dice_loss_weight,
        "dice_scale_factor": args.dice_scale_factor,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_cls_loss_weight": args.seg_cls_loss_weight,
        "seg_cls_loss_per_cls_weight": args.seg_cls_loss_per_cls_weight,
        "kld_loss_weight": args.kld_loss_weight,
        "kld_sigma": args.kld_sigma,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "use_teacher_ref": args.use_teacher_ref,
        "use_bidir_bio": args.use_bidir_bio,
        "use_crf_bio": args.use_crf_bio,
        "use_hinge_loss": args.use_hinge_loss,
        "use_feedback_loop": args.use_feedback_loop,
        "pred_binary_span": args.pred_binary_span,
        "use_cross_attn_bio": args.use_cross_attn_bio,
        "train_mask_prompt_encoder": args.train_mask_prompt_encoder,
        "focal_tversky_alpha": args.focal_tversky_alpha,
        "focal_tversky_beta": args.focal_tversky_beta,
        "bidir_nhead": args.bidir_nhead,
        "bidir_dim_feedforward": args.bidir_dim_feedforward,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    # model initialization
    model = PLUMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=torch.device("cuda", args.local_rank))
    model.get_model().initialize_plum_modules(model.get_model().config, **model_args)

    # Freeze the vision tower and mm_projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "token_to_mask_fcs"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    
    params_to_train = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "token_to_mask_fcs"]
    print(">> model.config.train_mask_prompt_encoder: ", model.config.train_mask_prompt_encoder)
    if model.config.train_mask_prompt_encoder:
        params_to_train.append("prompt_encoder")
    if model.use_bidir_bio:
        params_to_train.append("bio_encoder")
    if model.use_crf_bio:
        params_to_train.append("crf")
    if model.use_cross_attn_bio:
        params_to_train.append("bio_cross_attn")
    if model.use_feedback_loop:
        params_to_train.append("mask_pooler")

    # make text_hidden_fcs, token_to_mask_fcs mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in params_to_train
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    if args.use_bidir_bio:
        model.bio_encoder._init_weights()
        if args.use_crf_bio:
            model.crf.reset_parameters()

    if args.use_cross_attn_bio:
        model.bio_cross_attn.initialize_weights()

    if model.use_teacher_ref and not args.eval_only:  # initialize the frozen teacher_llm
        model.initialize_teacher_llm()

    if model.use_feedback_loop:
        model.initialize_mask_pooler()

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    if args.vision_tower == "naclip":
        vision_tower_for_image_preprocessing = "openai/clip-vit-large-patch14"
    else:
        vision_tower_for_image_preprocessing = args.vision_tower
    
    # setup partonomy dataset for training
    explanatory_seg_dataset_adapter = None
    if args.explanatory_seg_data and args.zero_shot_ckpt_path:
        print(">> (plum_train_ds) loading ExplanatorySegDataset...")
        train_question_types = [  # NOTE: We only train on with label case
            QuestionType.IDENTIFICATION_WITH_LABEL,
            QuestionType.POSITIVE_WITH_LABEL,
            QuestionType.NEGATIVE_WITH_LABEL,
            # QuestionType.DIFFERENCE,  # NOTE: We don't train on difference, part2whole, whole2part
            # QuestionType.DIFFERENCE_WITH_LABEL
            # QuestionType.PART_TO_WHOLE,
            # QuestionType.WHOLE_TO_PART,
        ]
        dataset_splits = args.partonomy_dataset_split.split("|")  # 'partimagenet|pascal_part|paco_lvis|partonomy'
        
        for dataset_split in dataset_splits:
            if dataset_split == 'partonomy':
                train_paths = []
            else:
                train_paths = [args.partonomy_train_dataset_path.format(dataset_split=dataset_split)]
                
        explanatory_seg_datasets = []
        for question_type in train_question_types:
            print(">> (plum_train_ds) train_question_types: ", question_type)
            explanatory_seg_datasets.extend(
                [
                    ExplanatorySegDataset(
                        path, 
                        tokenizer, 
                        vision_tower_for_image_preprocessing, 
                        question_type=question_type, 
                        sample_one_question_per_image=args.sample_one_question_per_image, 
                        random_seed=args.random_seed
                    ) 
                    for path in train_paths
                ]
            )
        explanatory_seg_dataset_adapter = ExplanatorySegDatasetsAdapter(explanatory_seg_datasets, indexing_strategy='concatenate')
        args.sample_rates = '1'  # e.g., ['explanatory_seg'] with a sample_rate = 1

    # load the training dataset
    warmup_num_steps = 100
    total_num_steps = args.epochs * args.steps_per_epoch
    if args.eval_only or args.partonomy_dataset_split == 'partonomy':
        train_dataset = None
    else:
        samples_per_epoch = args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size
        total_num_steps = args.epochs * args.steps_per_epoch
        warmup_num_steps = 100
            
        train_dataset = HybridDataset(
                args.dataset_dir,
                tokenizer,
                vision_tower_for_image_preprocessing,
                samples_per_epoch=samples_per_epoch,
                precision=args.precision,
                image_size=args.image_size,
                num_classes_per_sample=args.num_classes_per_sample,
                exclude_val=args.exclude_val,
                dataset=args.dataset,
                sample_rate=[float(x) for x in args.sample_rates.split(",")],
                sem_seg_data=args.sem_seg_data,
                refer_seg_data=args.refer_seg_data,
                vqa_data=args.vqa_data,
                reason_seg_data=args.reason_seg_data,
                explanatory_seg_datasets=explanatory_seg_dataset_adapter,
                explanatory=args.explanatory,
        )
        print(
            f"Training with {len(train_dataset)} examples."
        )

    if args.no_eval == False:
        if args.explanatory_seg_data and args.zero_shot_ckpt_path:
            dataset_splits = args.partonomy_dataset_split.split("|")  # 'partimagenet|pascal_part|paco_lvis|partonomy'
            #dataset_splits = ['partimagenet']
            explanatory_seg_val_datasets = []
            for dname in dataset_splits:
                explanatory_seg_val_dataset = ExplanatorySegDataset(
                    dataset_path=args.partonomy_val_dataset_path.format(dataset_split=dname),
                    tokenizer=tokenizer,
                    vision_tower=args.vision_tower,
                    question_type=QuestionType.IDENTIFICATION_WITH_LABEL,  # XXX
                    image_size=args.image_size,
                    model_str='plum'
                )
                explanatory_seg_val_datasets.append(explanatory_seg_val_dataset)
            val_dataset = ExplanatorySegDatasetsAdapter(explanatory_seg_val_datasets, indexing_strategy='concatenate', inference=True)
            print(f">> ExplanatorySegDataset ({args.partonomy_question_type}): Validating with {len(val_dataset)} examples.")
        else:
            val_dataset = ValDataset(
                args.dataset_dir,
                tokenizer,
                vision_tower_for_image_preprocessing,
                args.val_dataset,
                args.image_size,
            )
            print(
                f"Validating with {len(val_dataset)} examples."
            )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_num_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": warmup_num_steps,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    
    if args.debug:
        def _engine_evaluate(self, **kwargs):
            """
            This runs on the DeepSpeedEngine.  
            It gathers the ZeRO shards, then delegates to the plain model.
            """
            with GatheredParameters(list(self.module.parameters()), modifier_rank=0):
                # self.module is the underlying PLUMForCausalLM
                return self.module.evaluate(**kwargs)

        # Bind it as a method of the engine (self will be the engine):
        model_engine.evaluate = types.MethodType(_engine_evaluate, model_engine)

    print("(train) >> AFTER DEEPSPEED")    
    
    if args.vision_tower == "naclip":
        for mod in vision_tower.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.data = mod.weight.data.float()  # keep LayerNorm in float32 for NACLIP
                if mod.bias is not None:
                    mod.bias.data = mod.bias.data.float()
                mod.weight.requires_grad = False
                if mod.bias is not None:
                    mod.bias.requires_grad = False
                
    for p in vision_tower.parameters():
        p.requires_grad = False

    # resume deepspeed checkpoint or 0-shot checkpoints for Partonomy fine-tuning
    if (args.auto_resume and len(args.resume) == 0) or args.zero_shot_ckpt_path:
        if args.zero_shot_ckpt_path and args.explanatory_seg_data:
            resume = args.zero_shot_ckpt_path
            print(">> (train) Train on Partonomy dataset from: ", resume)
            print(">> (train) resume exists: ", os.path.exists(resume))
        else:
            resume = os.path.join(args.log_dir, f"{run_ckpt_path}_ckpt_model")
            print(">> (train) Auto-resume from: ", resume)
            print(">> (train) resume exists: ", os.path.exists(resume))
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        if ckpt_dir is None:
            raise ValueError("No checkpoint found")
        
        print(">> (train) Loading checkpoint from: ", ckpt_dir)
        print(">> list dir of ckpt_dir: ", os.listdir(os.path.join(args.resume, ckpt_dir)))
        # load model checkpoint
        if args.zero_shot_ckpt_path and args.explanatory_seg_data:
            load_lr_scheduler_states = False
            load_optimizer_states = False
        else:
            load_lr_scheduler_states = True
            load_optimizer_states = True
            
        load_path, client_state = model_engine.load_checkpoint(
                                        args.resume, load_module_strict=False, 
                                        load_optimizer_states=load_optimizer_states, 
                                        load_lr_scheduler_states=load_lr_scheduler_states
                                    )
        
        scheduler = model_engine.lr_scheduler
        if args.zero_shot_ckpt_path and args.explanatory_seg_data:
            args.start_epoch = latest_epoch
        else:
            # args.start_epoch = (int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch) + latest_epoch
            args.start_epoch = latest_epoch + 1
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1, "Currently supports batch_size=1 for segmentation eval"
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
        
    if args.eval_only:
        if args.explanatory_seg_data and args.zero_shot_ckpt_path:
            print(f"QuestionType.IDENTIFICATION_WITH_LABEL")
            giou, ciou, macro_gt_mask_metrics, micro_gt_mask_metrics, bio_per_cls_acc = validate_partonomy(val_loader, model_engine, 0, writer, args, parser, tokenizer)
        else:
            giou, ciou, bio_per_cls_acc = validate(val_loader, model_engine, 0, writer, args)
        exit()

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args
        )

        if args.no_eval == False:
            if args.explanatory_seg_data and args.zero_shot_ckpt_path:
                giou, ciou, macro_gt_mask_metrics, micro_gt_mask_metrics, bio_per_cls_acc = validate_partonomy(val_loader, model_engine, epoch, writer, args, parser)
                
                # XXX: Change this per task eval
                print(f"> giou: {giou:.4f}, ciou: {ciou:.4f}")
                print(f"> macro_gt_mask_metrics: {macro_gt_mask_metrics}")
                print(f"> micro_gt_mask_metrics: {micro_gt_mask_metrics}")
                
                exit() # XXX
                
            else:
                giou, ciou, bio_per_cls_acc = validate(val_loader, model_engine, epoch, writer, args)
                macro_gt_mask_metrics = {'gmIoU-paired': 0.0}
                micro_gt_mask_metrics = {'gmIoU-paired': 0.0}
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou
            
            if args.local_rank == 0:
                writer.add_scalar("val/giou", giou, epoch)
                writer.add_scalar("val/ciou", ciou, epoch)
                # wandb logging
                wandb.log({
                    "val/giou": giou,
                    "val/ciou": ciou,
                    "val/macro_gIoU": macro_gt_mask_metrics['gmIoU-paired'],
                    "val/micro_gIoU": micro_gt_mask_metrics['gmIoU-paired'],
                    "val/b_acc": bio_per_cls_acc[1],
                    "val/i_acc": bio_per_cls_acc[2],
                    "val/o_acc": bio_per_cls_acc[0]
                })
                epoch_metrics[epoch] = {
                    'giou': float(giou), 
                    'ciou': float(ciou), 
                    'macro_gIoU': float(macro_gt_mask_metrics['gmIoU-paired']),
                    'micro_gIoU': float(micro_gt_mask_metrics['gmIoU-paired']),
                    'b_acc': float(bio_per_cls_acc[1]),
                    'i_acc': float(bio_per_cls_acc[2]),
                    'o_acc': float(bio_per_cls_acc[0])
                    }

            with open(metrics_path, "w") as f:
                json.dump(epoch_metrics, f, indent=4)

        if is_best or args.no_eval:
            save_dir = os.path.join(args.log_dir,  f"{run_ckpt_path}_ckpt_model")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    seg_cls_losses = AverageMeter("SegCLSLoss", ":.4f")
    kl_losses = AverageMeter("KLLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            seg_cls_losses,
            kl_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
  
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            seg_cls_loss = output_dict["seg_cls_loss"]
            kl_loss = output_dict["kl_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            
            loss /= args.grad_accumulation_steps
            ce_loss /= args.grad_accumulation_steps
            seg_cls_loss /= args.grad_accumulation_steps
            kl_loss /= args.grad_accumulation_steps
            mask_bce_loss /= args.grad_accumulation_steps
            mask_dice_loss /= args.grad_accumulation_steps
            mask_loss /= args.grad_accumulation_steps

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            seg_cls_losses.update(seg_cls_loss.item(), input_dict["images"].size(0))
            kl_losses.update(kl_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            
            if args.zero_shot_ckpt_path and args.explanatory_seg_data:
                model.step()
        
        if not (args.zero_shot_ckpt_path and args.explanatory_seg_data):
            model.step()  # during 0-shot training, we take model.step() every args.gradient_accumulation_steps for additional gradient accumulation

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            total_step = epoch * args.steps_per_epoch + global_step
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                seg_cls_losses.all_reduce()
                kl_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(total_step + 1)
                writer.add_scalar("train/loss", losses.avg, total_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, total_step)
                writer.add_scalar(
                    "train/seg_cls_loss", seg_cls_losses.avg, total_step
                )
                writer.add_scalar(
                    "train/kl_loss", kl_losses.avg, total_step
                )
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, total_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, total_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, total_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, total_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, total_step
                )
                
                # wandb logging
                wandb.log({
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/seg_cls_loss": seg_cls_losses.avg,
                    "train/kl_loss": kl_losses.avg,
                    "train/mask_bce_loss": mask_bce_losses.avg,
                    "train/mask_dice_loss": mask_dice_losses.avg,
                    "train/mask_loss": mask_losses.avg,
                    "metrics/total_secs_per_batch": batch_time.avg,
                    "metrics/data_secs_per_batch": data_time.avg
                }, step=total_step)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            seg_cls_losses.reset()
            kl_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], total_step)
                wandb.log({"train/lr": curr_lr[0]}, step=total_step)

    return train_iter


@torch.no_grad()
def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    
    bio_o_meter = AverageMeter("bio_O_acc", ":6.3f")
    bio_b_meter = AverageMeter("bio_B_acc", ":6.3f")
    bio_i_meter = AverageMeter("bio_I_acc", ":6.3f")

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()  # TODO: Remove if not necessary

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int() if len(output_dict["gt_masks"]) > 0 else None
        bio_per_cls_counts_dict = output_dict["bio_per_cls_counts_dict"]
        correct_0 = bio_per_cls_counts_dict['correct_0']
        correct_1 = bio_per_cls_counts_dict['correct_1']
        correct_2 = bio_per_cls_counts_dict['correct_2']
        total_0 = bio_per_cls_counts_dict['total_0'] if bio_per_cls_counts_dict['total_0'] > 0 else 1  # prevent ZeroDivisionError
        total_1 = bio_per_cls_counts_dict['total_1'] if bio_per_cls_counts_dict['total_1'] > 0 else 1
        total_2 = bio_per_cls_counts_dict['total_2'] if bio_per_cls_counts_dict['total_2'] > 0 else 1
        if len(pred_masks) > 0:
            output_list = (pred_masks[0] > 0).int()
        else:
            output_list = None
        
        assert len(pred_masks) == 1, f"len(pred_masks) = {len(pred_masks)}"
        
        bio_o_meter.update(correct_0 / total_0, total_0)
        bio_b_meter.update(correct_1 / total_1, total_1)
        bio_i_meter.update(correct_2 / total_2, total_2)
        
        if masks_list is None or output_list is None:
            continue

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    
    bio_o_meter.all_reduce()
    bio_b_meter.all_reduce()
    bio_i_meter.all_reduce()
    
    o_acc = bio_o_meter.avg  # correct_O / total_O
    b_acc = bio_b_meter.avg  # correct_B / total_B
    i_acc = bio_i_meter.avg  # correct_I / total_I

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/o_acc", o_acc, epoch)
        writer.add_scalar("val/b_acc", b_acc, epoch)
        writer.add_scalar("val/i_acc", i_acc, epoch)

        print(f"giou: {giou:.4f}, ciou: {ciou:.4f} | BIO per cls acc: O={o_acc:.4f}, B={b_acc:.4f}, I={i_acc:.4f}")
        
    return giou, ciou, (o_acc, b_acc, i_acc) # bio_per_cls_acc


@torch.no_grad()
def validate_partonomy(
            val_loader,
            model_engine, 
            epoch, 
            writer, 
            args,
            parser,
            tokenizer=None
        ):
    """
    Validate the model on the Partonomy dataset using the validate function from validate_partonomy.py
    """
    print(f">> Validating on Partonomy dataset ({args.partonomy_question_type})")
    # Configure your evaluators
    macro_config = Namespace(args.metrics.iou_evaluator_config)
    macro_config.reduction = Reduction.MACRO
    macro_iou_evaluator = IoUEvaluator(macro_config, metric_group_name='mask_macro')
    
    micro_config = Namespace(args.metrics.iou_evaluator_config)
    micro_config.reduction = Reduction.MICRO
    micro_iou_evaluator = IoUEvaluator(micro_config, metric_group_name='mask_micro')
    
    part_text_evaluator = PartTextEvaluator(args.metrics.part_text_evaluator_config)
    mc_part_text_evaluator = MCTextEvaluator(metric_group_name='mc_part_text')
    mc_object_text_evaluator = MCTextEvaluator(metric_group_name='mc_object_text')
    
    predictions = Predictions()
    
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    
    bio_o_meter = AverageMeter("bio_O_acc", ":6.3f")
    bio_b_meter = AverageMeter("bio_B_acc", ":6.3f")
    bio_i_meter = AverageMeter("bio_I_acc", ":6.3f")
    
    model_engine.eval()
    
    n_batches = args.limit_batches if args.limit_batches is not None else len(val_loader)
    val_loader = islice(val_loader, n_batches)
    
    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader, total=n_batches, desc="Validating Partonomy with PLUM")):
        
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        
        with torch.no_grad():
            output_dict = model_engine(**input_dict)
            
        if args.debug:  # use model_engine.evaluate() and model_engine() to compare outputs
            evaluate_input_dict = {
                'images_clip': input_dict['images_clip'],
                'images': input_dict['images'],
                'input_ids': input_dict['input_ids'],
                'resize_list': input_dict['resize_list'],
                'label_list': input_dict['label_list'],
                'attention_masks': input_dict['attention_masks'],
                'gt_bio_span': input_dict['per_token_labels'],
                'max_new_tokens': args.model_max_length,
                'tokenizer': tokenizer,
                'multiple_choice': True,
                'prompt_user_input': False,
            }
            output_dict_eval = model_engine.evaluate(**evaluate_input_dict)
            output_ids_eval = output_dict_eval.get("output_ids", None)
            spans_dicts_eval = output_dict_eval.get("spans_dicts", None)
            pred_masks_eval = output_dict_eval.get("pred_masks", None)  # list[list[torch.Tensor]] - e.g., len(pred_masks) == 5 since there are 5 answer choices and len(pred_masks[0]) == 3 if there are 3 parts
            ce_losses_eval = output_dict_eval.get("ce_losses", None)
            pred_masks_eval = torch.cat([(pred_mask > 0).int() for pred_mask in pred_masks_eval[0]], dim=0)

        pred_masks = output_dict["pred_masks"]
        if len(pred_masks) > 0:
            output_list = torch.cat([(pred_mask > 0).int() for pred_mask in pred_masks], dim=0)
        else:
            output_list = None
            
        if len(output_dict["gt_masks"]) > 0:
            masks_list = torch.cat([gt_mask.int() for gt_mask in output_dict["gt_masks"]], dim=0)
        else:
            print(f"No ground truth masks found")
            continue
        
        bio_per_cls_counts_dict = output_dict["bio_per_cls_counts_dict"]
        correct_0 = bio_per_cls_counts_dict['correct_0']
        correct_1 = bio_per_cls_counts_dict['correct_1']
        correct_2 = bio_per_cls_counts_dict['correct_2']
        total_0 = bio_per_cls_counts_dict['total_0'] if bio_per_cls_counts_dict['total_0'] > 0 else 1  # prevent ZeroDivisionError
        total_1 = bio_per_cls_counts_dict['total_1'] if bio_per_cls_counts_dict['total_1'] > 0 else 1
        total_2 = bio_per_cls_counts_dict['total_2'] if bio_per_cls_counts_dict['total_2'] > 0 else 1
        
        # NOTE: this assert only applies for model.evaluate() for partonomy dataset
        # assert len(pred_masks) == 1, f"len(pred_masks) = {len(pred_masks)}"
        bio_o_meter.update(correct_0 / total_0, total_0)
        bio_b_meter.update(correct_1 / total_1, total_1)
        bio_i_meter.update(correct_2 / total_2, total_2)

        if masks_list is None or output_list is None:
            continue

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
        
        macro_giou_metrics = macro_iou_evaluator.update(output_list.cpu().numpy(), masks_list.cpu().numpy())
        micro_giou_metrics = micro_iou_evaluator.update(output_list.cpu().numpy(), masks_list.cpu().numpy())
        
        print(f">>> macro_giou_metrics: {macro_giou_metrics}")
        print(f">>> micro_giou_metrics: {micro_giou_metrics}")
        
        # Evaluator Predictions
        metrics = {
            micro_iou_evaluator.metric_group_name: micro_giou_metrics,
            macro_iou_evaluator.metric_group_name: macro_giou_metrics,
        }
        predictions.add_prediction(
            Prediction(
                image_path=input_dict["image_paths"][0],

                question_type=QuestionType.IDENTIFICATION_WITH_LABEL,
                questions=input_dict["questions_list"][0],

                # Parts question outputs
                # parts_answer_choices=input_dict["part_answer_choices"][0],

                # gt_parts_answer=input_dict["part_answer_choices"][0][gt_parts_answer_index],
                # predicted_parts_answer=input_dict["part_answer_choices"][0][predicted_parts_answer_index],

                # gt_parts=gt_parts,
                # predicted_parts=predicted_parts,

                gt_masks=get_mask_rle_dicts(masks_list.cpu().numpy(), ['no_lbl'] * masks_list.shape[0]),
                predicted_masks=get_mask_rle_dicts(output_list.cpu().numpy(), ['no_lbl'] * output_list.shape[0]),  # gt_predicted_masks or pred_predicted_masks

                metrics=metrics
            )
        )


    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    
    bio_o_meter.all_reduce()
    bio_b_meter.all_reduce()
    bio_i_meter.all_reduce()
    
    o_acc = bio_o_meter.avg  # correct_O / total_O
    b_acc = bio_b_meter.avg  # correct_B / total_B
    i_acc = bio_i_meter.avg  # correct_I / total_I

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        # Output predictions
        if args.output_predictions:
            output_dir = os.path.dirname(args.predictions_path)
            os.makedirs(output_dir, exist_ok=True)

            # Dump config
            config_path = os.path.join(output_dir, 'config.yaml')
            if not os.path.exists(config_path):
                parser.save(args, config_path) # This fails if the file already exists

            # Dump predictions
            import json

            predictions_dict = predictions.to_dict([
                micro_iou_evaluator,
                macro_iou_evaluator,
                part_text_evaluator,
                mc_part_text_evaluator,
                mc_object_text_evaluator
            ])

            with open(args.predictions_path, 'w') as f:
                json.dump(predictions_dict, f, indent=4)

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/o_acc", o_acc, epoch)
        writer.add_scalar("val/b_acc", b_acc, epoch)
        writer.add_scalar("val/i_acc", i_acc, epoch)

        print(f"giou: {giou:.4f}, ciou: {ciou:.4f} | BIO per cls acc: O={o_acc:.4f}, B={b_acc:.4f}, I={i_acc:.4f}")
        
    return giou, ciou, macro_giou_metrics, micro_giou_metrics, (o_acc, b_acc, i_acc) # bio_per_cls_acc


if __name__ == "__main__":
    main(sys.argv[1:])

import jsonargparse as argparse
import json
import os
import sys
import numpy as np

import torch
import torch.distributed as dist

import tqdm
import transformers

from functools import partial
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler

from typing import Sequence, Iterable, Any, Callable, Mapping

from model.PLUM import PLUMForCausalLM
from model.llava import conversation as conversation_lib

from utils.dataset import ValDataset

from utils.dataset import collate_fn

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter, ProgressMeter, Summary, dict_to_cuda,
    intersectionAndUnionGPU
)
from utils.question_type import QuestionType

sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from evaluation.rle_dict import get_mask_rle_dicts
from evaluation.evaluators import (
    IoUEvaluator, IoUEvaluatorConfig, MaskMatchingStrategy,
    PartTextEvaluator, PartTextEvaluatorConfig,
    MCTextEvaluator, Reduction
)
from evaluation.prediction import Prediction, Predictions
from root_utils import get_timestr
from itertools import islice, chain
from bisect import bisect_right
import logging
import coloredlogs

from jsonargparse import Namespace, ArgumentParser


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="Validate PLUM on Partonomy Dataset")
    
    # Model
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--backbone", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview", type=str)
    parser.add_argument("--version", default=None)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
        help="precision for inference"
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    
    # Dataset
    parser.add_argument("--dataset_path", default="/path/to/partonomy_qa_pairs.json", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="plum", type=str)
    parser.add_argument("--val_dataset", default="ade20k|validation", type=str)  # "ade20k|validation", "refcoco|unc|testA", "refcoco|unc|testB"
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)  # --workers=1 after debugging
    
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    
    # bidirectionalencoderblock params
    parser.add_argument("--bidir_nhead", default=8, type=int)
    parser.add_argument("--bidir_dim_feedforward", default=2048, type=int)
    
    parser.add_argument("--use_bidir_bio", action="store_true", default=False)
    parser.add_argument("--use_feedback_loop", action="store_true", default=False)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--limit_batches', default=None, type=int)
    parser.add_argument('--output_generation_prompt_in_dataset', type=bool, default=False)
    parser.add_argument('--question_type', default=QuestionType.POSITIVE, type=QuestionType)
    parser.add_argument('--test_with_gt_object', type=bool, default=False,
                        help='Prompts the model with the GT object label for testing; GT masks are set to the full image')

    parser.add_argument('--output_predictions', action='store_true')
    parser.add_argument('--predictions_path', type=str, default='predictions.json')

    # Evaluators
    parser.add_argument('--metrics.iou_evaluator_config', default=IoUEvaluatorConfig(), type=IoUEvaluatorConfig) # NOTE: Change to either reduction='macro' or reduction='micro'
    parser.add_argument('--metrics.part_text_evaluator_config', default=PartTextEvaluatorConfig(), type=PartTextEvaluatorConfig)

    parser.add_argument("--debug", action="store_true", help="Debug mode. If set load only the dataloader not the model.")

    args = parser.parse_string(config_str) if config_str else parser.parse_args(args)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)
    log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    print(f"*(validate_partonomy) >> Loading model checkpoints from [ {args.version} ]*")

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.backbone,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    # build PLUM model        
    model_args = {
        "out_dim": args.out_dim,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "use_bidir_bio": args.use_bidir_bio,
        "use_feedback_loop": args.use_feedback_loop,
        "bidir_nhead": args.bidir_nhead,
        "bidir_dim_feedforward": args.bidir_dim_feedforward,
    }
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        
    if args.load_in_4bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        model_args.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = PLUMForCausalLM.from_pretrained(
        args.version,
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=False,
        **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # initialize vision modules in PLUM
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=torch.device("cuda", args.local_rank))

    # device setup
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()
            
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    val_dataset = ValDataset(
        args.dataset_path,
        tokenizer,
        args.vision_tower,
        args.val_dataset,
        args.image_size,
    )
    print(
        f"Validating with {len(val_dataset)} examples."
    )
    
    # build dataloader
    if val_dataset is not None:
        assert args.val_batch_size == 1, "Currently supports batch_size=1 for segmentation eval"
        val_sampler = SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size, 
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler if args.local_rank >= 0 else None,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    else:
        raise ValueError("No validation dataset provided")

    # run evaluation on the partonomy dataset
    validate(val_loader, model, tokenizer, 0, writer, args, parser)


def to_plum_inputs(args, batch, tokenizer: transformers.AutoTokenizer, set_mc_choices=False):
    input_dict = {
        'images': batch['images'],
        'images_clip': batch['images_clip'],
        "input_ids": batch['input_ids'],
        'labels': batch['labels'],
        'attention_masks': batch['attention_masks'],
        'offset': batch['offset'],
        'questions_list': batch['questions_list'],
        'sampled_classes_list': batch['sampled_classes_list'],
        'conversation_list': batch['conversation_list'],
        'masks_list': batch['masks_list'],
        'label_list': batch['label_list'],
        'resize_list': batch['resize_list'],
        'per_token_labels': batch['per_token_labels'],
        'mask_positions_in_input_ids': batch['mask_positions_in_input_ids'],
        'inference': batch['inference'],
        'max_new_tokens': args.model_max_length,
        'tokenizer': tokenizer,
    }
    
    return input_dict


@torch.no_grad()
def validate(val_loader, model_engine, tokenizer, epoch, writer, args, parser):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    
    bio_o_meter = AverageMeter("bio_O_acc", ":6.3f")
    bio_b_meter = AverageMeter("bio_B_acc", ":6.3f")
    bio_i_meter = AverageMeter("bio_I_acc", ":6.3f")
    
    # micro-, micro-gIoU evaluator initialization
    iou_evaluator = IoUEvaluator(args.metrics.iou_evaluator_config)
    micro_config = Namespace(args.metrics.iou_evaluator_config)
    micro_config.reduction = Reduction.MICRO
    iou_evaluator_micro = IoUEvaluator(micro_config, metric_group_name='mask_micro')
    macro_config = Namespace(args.metrics.iou_evaluator_config)
    macro_config.reduction = Reduction.MACRO
    iou_evaluator_macro = IoUEvaluator(macro_config, metric_group_name='mask_macro')

    model_engine.eval()
    
    n_batches = args.limit_batches if args.limit_batches is not None else len(val_loader)
    val_loader = islice(val_loader, n_batches)
    
    predictions = []
    for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        input_dict = to_plum_inputs(args, input_dict, tokenizer)
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

        pred_masks = torch.cat([(pred_mask > 0).int() for pred_mask in output_dict["pred_masks"]]) if len(output_dict["pred_masks"]) > 0 else None
        masks_list = torch.cat([gt_mask.int() for gt_mask in output_dict["gt_masks"]]) if len(output_dict["gt_masks"]) > 0 else None
        masks_classes = input_dict['sampled_classes_list'][0] if len(input_dict['sampled_classes_list'][0]) > 0 else None
        
        assert len(pred_masks) == len(masks_list) == len(masks_classes), f"len(pred_masks) = {len(pred_masks)} | len(masks_list) = {len(masks_list)} | len(masks_classes) = {len(masks_classes)}"
        
        bio_per_cls_counts_dict = output_dict["bio_per_cls_counts_dict"]
        correct_0 = bio_per_cls_counts_dict['correct_0']
        correct_1 = bio_per_cls_counts_dict['correct_1']
        correct_2 = bio_per_cls_counts_dict['correct_2']
        total_0 = bio_per_cls_counts_dict['total_0'] if bio_per_cls_counts_dict['total_0'] > 0 else 1  # prevent ZeroDivisionError
        total_1 = bio_per_cls_counts_dict['total_1'] if bio_per_cls_counts_dict['total_1'] > 0 else 1
        total_2 = bio_per_cls_counts_dict['total_2'] if bio_per_cls_counts_dict['total_2'] > 0 else 1
    
        bio_o_meter.update(correct_0 / total_0, total_0)
        bio_b_meter.update(correct_1 / total_1, total_1)
        bio_i_meter.update(correct_2 / total_2, total_2)
        
        if masks_list is None or pred_masks is None:
            continue

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        
        for mask_i, pred_i in zip(masks_list, pred_masks):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
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
        
        # Compute micro and macro gIoU
        # Convert tensors to CPU and then to numpy arrays for the evaluator
        pred_masks_cpu = pred_masks.cpu().numpy().astype(np.uint8)
        masks_list_cpu = masks_list.cpu().numpy().astype(np.uint8)
        
        micro_giou_metrics = iou_evaluator_micro.update(pred_masks_cpu, masks_list_cpu)
        macro_giou_metrics = iou_evaluator_macro.update(pred_masks_cpu, masks_list_cpu)
        
        print("=" * 20)
        print(">>> micro_giou_metrics: ", micro_giou_metrics)
        print(">>> macro_giou_metrics: ", macro_giou_metrics)
        print("=" * 20)
        
        # Create prediction entries for each mask
        mask_preds ={
            'sample_idx': sample_idx,
            'image_path': input_dict.get('image_path', f'sample_{sample_idx}'),
            'question': input_dict['questions_list'][0] if len(input_dict['questions_list']) > 0 else "",
            'pred_class': masks_classes if len(masks_classes) > 0 else ['no_label'],
            'pred_mask': get_mask_rle_dicts(pred_masks_cpu, masks_classes if len(masks_classes) > 0 else ['no_label']),
            'gt_mask': get_mask_rle_dicts(masks_list_cpu, masks_classes if len(masks_classes) > 0 else ['no_label']),
            'micro_giou': float(micro_giou_metrics.get(iou_evaluator_micro.metric_group_name, {}).get('gmIoU-paired', 0.0)),
            'macro_giou': float(macro_giou_metrics.get(iou_evaluator_macro.metric_group_name, {}).get('gmIoU-paired', 0.0)),
            'giou': float(acc_iou_meter.avg[1]),
            'ciou': float(intersection_meter.avg[1] / (union_meter.avg[1] + 1e-10)),
        }
        predictions.append(mask_preds)
    
    if dist.is_available() and dist.is_initialized():
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

    # log the predictions
    if args.output_predictions:
        # check if args.predictions_path directory exists
        predictions_dir = os.path.dirname(args.predictions_path)
        os.makedirs(predictions_dir, exist_ok=True)
        
        # get the predictions from predictions.json and then append MODEL_TYPE and args.val_dataset to the file name
        predictions_fname_prefix = args.predictions_path.split('/')[-1].split('.')[0]
        model_type = args.version.split('/')[-2]
        ds_sampling_ratio = args.version.split('/')[-1].split('_')[-1]
        val_dataset_suffix = "_".join(args.val_dataset.split('|'))
        predictions_path = f"{predictions_fname_prefix}_{model_type}_{ds_sampling_ratio}_{val_dataset_suffix}.json"
        predictions_path = os.path.join(predictions_dir, predictions_path)
        predictions_dict = {
            "giou": float(giou),
            "ciou": float(ciou),
            "o_acc": float(o_acc),
            "b_acc": float(b_acc),
            "i_acc": float(i_acc),
            "predictions": predictions,
        }
        with open(predictions_path, 'w') as f:
            json.dump(predictions_dict, f)
        
    return giou, ciou, (o_acc, b_acc, i_acc) # bio_per_cls_acc



if __name__ == "__main__":
    coloredlogs.install(level='DEBUG')

    main(sys.argv[1:])
    
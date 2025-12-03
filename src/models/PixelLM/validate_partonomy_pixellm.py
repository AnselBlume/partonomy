import os
import sys
import json
import torch
import tqdm
import numpy as np
import jsonargparse as argparse
from functools import partial
from pprint import pformat
from itertools import islice, chain
from bisect import bisect_right

import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from typing import Sequence, Iterable, Any, Callable, Mapping

from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib
from utils.multi_reason_seg_val_dataset import MultiReasonSegValDataset
# from utils.data_processing import preprocess

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))  # NOTE: sys.path setting needs to fixed later
from models.lisa.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_device,
                         intersectionAndUnionGPU)
from models.lisa.utils.question_type import QuestionType
from models.lisa.utils.explanatory_dataset import ExplanatorySegBatch, collate_fn
from models.lisa.utils.explanatory_seg_dataset import ExplanatorySegDataset

from evaluation.rle_dict import get_mask_rle_dicts
from evaluation.evaluators import (
    IoUEvaluator, IoUEvaluatorConfig, MaskMatchingStrategy,
    PartTextEvaluator, PartTextEvaluatorConfig,
    MCTextEvaluator, Reduction
)
from models.lisa.utils.explanatory_seg_dataset import ConversationType, ConversationQuestionType
from evaluation.prediction import Prediction, Predictions
from root_utils import get_timestr
from itertools import islice, chain
from bisect import bisect_right

from jsonargparse import Namespace, ArgumentParser

import logging
import coloredlogs

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(args: list[str] = None, config_str: str = None):
    parser = argparse.ArgumentParser(description="PixelLM Model Validation")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--val_dataset", default="ExplanatorySeg", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"], type=str)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)

    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--dataset_dir", default="dataset/partonomy_descriptors", type=str)
    parser.add_argument("--dataset_path", default="partonomy_qa_pairs_val.json", type=str)
    parser.add_argument('--output_generation_prompt_in_dataset', action="store_true", default=False)
    
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="pixellm", type=str)

    # model args
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--use_expand_question_list", action="store_true", default=False)
    parser.add_argument("--masks_process_with_clip", default=False, action="store_true")
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--pad_val_clip_images", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument('--limit_batches', default=None, type=int)
    parser.add_argument('--question_type', default=QuestionType.POSITIVE, type=QuestionType)
    parser.add_argument('--test_with_gt_object', type=bool, default=False,
                        help='Prompts the model with the GT object label for testing; GT masks are set to the full image')
    parser.add_argument('--output_predictions', type=bool, default=False)
    parser.add_argument('--predictions_path', type=str, default='predictions.json')

    parser.add_argument('--metrics.iou_evaluator_config', default=IoUEvaluatorConfig(), type=IoUEvaluatorConfig)
    parser.add_argument('--metrics.part_text_evaluator_config', default=PartTextEvaluatorConfig(), type=PartTextEvaluatorConfig)

    args = parser.parse_args(args) if not config_str else parser.parse_string(config_str)
    return args, parser


def main(args: list[str] = None, config_str: str = None):
    args, parser = parse_args(args, config_str)
    log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        timestr = get_timestr()
        log_filename = os.path.join(log_dir, 'meta.log')
        i = 1
        while os.path.exists(log_filename):
            log_filename = os.path.join(log_dir, f'meta_{timestr}.log')
            i += 1
        logger = logging.getLogger('pixellm_logger')
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(args)
    else:
        writer = None

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")

    if args.seg_token_num == 1:  # args.seg_token_num * args.image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    elif args.seg_token_num > 1:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]        
    else:
        raise ValueError(f"args.seg_token_num cannot be 0 or negative.")

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "preprocessor_config": args.preprocessor_config,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_token_num": args.seg_token_num,
        "logger": logger,
        "tokenizer": tokenizer,
        "local_rank": args.local_rank,
        "pad_val_clip_images": args.pad_val_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,  # Flag for using PixelLM's lightweight decoder
        "separate_mm_projector": args.separate_mm_projector,
        "masks_process_with_clip": args.masks_process_with_clip,
        "image_feature_scale_num": args.image_feature_scale_num,
    }
    print(">> model_args: ", model_args)

    # Load model
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.half,
        "fp32": torch.float32,
    }[args.precision]

    model = PixelLMForCausalLM.from_pretrained(
        args.version, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **model_args
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_pixellm_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    if args.resize_vision_tower_size == 224:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.eval()
    model.to(args.device)

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]
    model.resize_token_embeddings(len(tokenizer))

    # PixelLM has L * N number of seg tokens 
    # - L: pre-defined number of intermediate layers (corresponds to 'args.image_feature_scale_num')
    # - N: pre-defined # of segmentation tokens in a codebook) (corresponds to 'args.seg_token_num')
    token_num = args.seg_token_num * args.image_feature_scale_num

    world_size = 1 # torch.cuda.device_count()  # NOTE: We keep the world size to 1 for experiment
    is_distributed = world_size > 1

    if args.val_dataset == 'ExplanatorySeg':
        # Load dataset
        val_dataset = ExplanatorySegDataset(
            dataset_path=os.path.join(args.dataset_dir, args.dataset_path),
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            question_type=args.question_type,
            image_size=224,
            transform=None,
            model_str='pixellm',  # e.g., ['lisa', 'glamm']
            test_with_gt_object=args.test_with_gt_object,
            output_question_prompt_for_generation=args.output_generation_prompt_in_dataset,
            preprocessor_config=args.preprocessor_config,
            pad_val_clip_images=args.pad_val_clip_images,
        )

    elif args.val_dataset == 'MultiReasonSeg':
        ValDataset_type = MultiReasonSegValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
            seg_token_num=token_num,
            pad_val_clip_images=args.pad_val_clip_images,
            masks_process_with_clip=args.masks_process_with_clip,
            preprocessor_config=args.preprocessor_config,
        )

    if 'cuda' in args.device:
        model_engine = model.bfloat16().cuda()
    else:
        model_engine = model.cpu()

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        if is_distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
        else:
            val_sampler = None

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

    validate(val_loader, model_engine, tokenizer, args, parser)


def to_pixellm_inputs(batch: ExplanatorySegBatch):
    input_dict = {
        'images': batch.sam_img_inputs,
        'images_clip': batch.clip_img_inputs,
        "input_ids": batch.input_ids,
        'attention_masks': batch.attention_masks,
        'offset': batch.img_to_conversations_offsets,
        'masks_list': batch.masks,
        'label_list': batch.label_masks,
        'labels': torch.zeros_like(batch.input_ids),  # NOTE: TypeError: PixelLMForCausalLM.model_forward() missing 1 required positional argument: 'labels'
        'resize_list': batch.resized_img_dims,
        'clip_resize_list': batch.resized_img_dims,  # --- Account for the 'clip_resize_list' in PixelLM/utils/multi_reason_seg_val_dataset.py ---
        'inference': batch.is_inference
    }
    return input_dict

def get_subarray(src: Sequence, indices: Iterable[int]) -> list:
    if isinstance(src, np.ndarray) or isinstance(src, torch.Tensor):
        return src[indices]
    elif hasattr(src, '__getitem__'): # Supports indexing
        return [src[i] for i in indices]
    else:
        raise ValueError(f"Unsupported type: {type(src)}")

def indices_of_cq_type(cq_types: list[ConversationQuestionType], target_cq_type: ConversationQuestionType):
    '''
    Returns the indices of the elements in the list of ConversationQuestionTypes that match the target ConversationQuestionType.
    '''
    return [i for i, cq_type in enumerate(cq_types) if cq_type == target_cq_type]


@torch.no_grad()
def validate(val_loader, model_engine, tokenizer, args, parser):
    iou_evaluator = IoUEvaluator(args.metrics.iou_evaluator_config)
    
    micro_config = Namespace(args.metrics.iou_evaluator_config)
    micro_config.reduction = Reduction.MICRO
    iou_evaluator_micro = IoUEvaluator(micro_config, metric_group_name='mask_micro')

    macro_config = Namespace(args.metrics.iou_evaluator_config)
    macro_config.reduction = Reduction.MACRO
    iou_evaluator_macro = IoUEvaluator(macro_config, metric_group_name='mask_macro')
    
    part_text_evaluator = PartTextEvaluator(args.metrics.part_text_evaluator_config)
    mc_part_text_evaluator = MCTextEvaluator(metric_group_name='mc_part_text')
    mc_object_text_evaluator = MCTextEvaluator(metric_group_name='mc_object_text')
    
    random_part_text_evaluator = PartTextEvaluator(args.metrics.part_text_evaluator_config, metric_group_name='random_part_text')
    
    trackers = { # LISA metrics
        "intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
        "union": AverageMeter("Union", ":.4f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)
    }
        
    rng = np.random.default_rng(42)

    predictions = Predictions()

    model_engine.eval()

    skipped_instance_num = 0
    n_batches = args.limit_batches if args.limit_batches is not None else len(val_loader)
    val_loader = islice(val_loader, n_batches)
    val_iter = iter(val_loader)
    
    for idx, batch in enumerate(tqdm.tqdm(val_loader, total=n_batches)):
        batch: ExplanatorySegBatch
        batch.to(args.device)
        
        if args.precision == "fp16":
            to_dtype = lambda x: x.half()
        elif args.precision == "bf16":
            to_dtype = lambda x: x.bfloat16()
        else:
            to_dtype = lambda x: x.float()

        batch.sam_img_inputs = to_dtype(batch.sam_img_inputs)
        batch.clip_img_inputs = to_dtype(batch.clip_img_inputs)

        conversation_types = list(chain.from_iterable(batch.conversation_types)) # (n_conversations_in_batch,)
        conversation_question_types = list(chain.from_iterable(batch.conversation_question_types)) # (n_conversations_in_batch,)

        # Potentially filter out non-multiple choice conversation prompts
        # Capture only the multiple choice questions to feed to LISA (if there is also a generation prompt)
        if args.output_generation_prompt_in_dataset: # Filtering if we are outputting a currently unused generation prompt
            orig_img_to_conversations_offsets = batch.img_to_conversations_offsets.cpu().tolist()
            filtered_img_to_conversations_offsets = list(orig_img_to_conversations_offsets)

            filtered_input_ids = []
            filtered_attention_masks = []

            for i, (input_ids, attention_mask, conversation_type) in enumerate(zip(batch.input_ids, batch.attention_masks, conversation_types)):
                if conversation_type in ['correct_answer', 'incorrect_answer']:
                    filtered_input_ids.append(input_ids)
                    filtered_attention_masks.append(attention_mask)

                # We're skipping one conversation (with type 'question'), so need to adjust offset used to map image (batch) index to
                # its corresponding conversation boundaries (conversations for img i are offsets[i]:offsets[i+1])
                else:
                    # Find the smallest conversation boundary index > i
                    mod_start_idx = bisect_right(orig_img_to_conversations_offsets, i)

                    # Decrement all the offsets starting from the index of the image that has the conversation index i
                    for j in range(mod_start_idx, len(filtered_img_to_conversations_offsets)):
                        filtered_img_to_conversations_offsets[j] -= 1

            batch.input_ids = torch.stack(filtered_input_ids)
            batch.attention_masks = torch.stack(filtered_attention_masks)
            batch.img_to_conversations_offsets = torch.tensor(filtered_img_to_conversations_offsets).to(batch.img_to_conversations_offsets)
            conversation_types = [ctype for ctype in conversation_types if ctype != 'question']

        # Determine indices of predicted masks corresponding to parts of ground truth answer
        pred_masks_for_gt_parts_start_idx = 0 # Index in the output masks for the predictions corresponding to the ground truth parts
        pred_masks_for_gt_parts_end_idx = None

        assert len(batch.input_ids) == len(conversation_types) == len(conversation_question_types)
        for input_ids, conversation_type, conversation_question_type in zip(batch.input_ids, conversation_types, conversation_question_types):
            if conversation_type in ['correct_answer', 'incorrect_answer']:
                n_mask_outputs_for_conversation = (input_ids == tokenizer.added_tokens_encoder['[SEG]']).sum()

                # Compute offset of ground truth answer masks in set of output masks
                if conversation_type == 'correct_answer' and conversation_question_type == 'part_question':
                    pred_masks_for_gt_parts_end_idx = pred_masks_for_gt_parts_start_idx + n_mask_outputs_for_conversation.item()
                    break
                else:
                    pred_masks_for_gt_parts_start_idx += n_mask_outputs_for_conversation

        # --- Forward Pass ---
        output_dict = model_engine(**to_pixellm_inputs(batch))  # PixelForCauslLM

        pred_masks_l: list[Tensor] = output_dict['pred_masks']
        assert len(pred_masks_l) == 1 # Ensure batch size 1

        pred_masks = (pred_masks_l[0] > 0).int() # (n_masks_for_instance, h, w)
        pred_masks = pred_masks[pred_masks_for_gt_parts_start_idx:pred_masks_for_gt_parts_end_idx] # Extract masks for GT parts

        ###################

        if output_dict is None:
            skipped_instance_num += 1
            continue
        
        # XXX Evaluation code assumes a batch size of one
        assert len(batch.img_paths) == 1
        
        # Compute indices of part and object questions in batch's conversations (all corresponding to the same image)
        part_indices = indices_of_cq_type(conversation_question_types, 'part_question')
        
        # Compute multiple choice accuracy
        gt_parts_answer_index = get_subarray(conversation_types, part_indices).index('correct_answer')
        predicted_parts_answer_index = get_subarray(output_dict['ce_losses'][0], part_indices).argmin().item()

        mc_part_text_metrics = mc_part_text_evaluator.update(predicted_parts_answer_index, gt_parts_answer_index)
        
        has_object_question = batch.question_types[0] in [QuestionType.WHOLE_TO_PART, QuestionType.PART_TO_WHOLE]
        if has_object_question:
            object_indices = indices_of_cq_type(conversation_question_types, 'object_question')

            gt_object_answer_index = get_subarray(conversation_types, object_indices).index('correct_answer')
            predicted_object_answer_index = get_subarray(output_dict['ce_losses'][0], object_indices).argmin().item()

            mc_object_text_metrics = mc_object_text_evaluator.update(predicted_object_answer_index, gt_object_answer_index)

        # Compute part recall, precision, f1
        answer_parts = batch.answer_parts[0] # list[list[str]]: outer list has length n_part_questions, and inner lists have length n_parts_in_answer
        gt_parts = answer_parts[gt_parts_answer_index]
        predicted_parts = answer_parts[predicted_parts_answer_index]

        part_text_metrics = part_text_evaluator.update(predicted_parts, gt_parts)

        # Compute for random answer selection
        random_index = rng.integers(len(answer_parts))
        random_answer_parts = answer_parts[random_index]
        random_part_text_metrics = random_part_text_evaluator.update(random_answer_parts, gt_parts)

        # Compute gmIoU
        assert len(pred_masks) == len(gt_parts)
        pred_masks = np.stack([m.cpu().numpy() for m in pred_masks])
        logger.debug(f'Number of predicted pixels: {pred_masks[0].sum()}')

        gt_masks = [batch.mask_dicts[0][gt_part] for gt_part in gt_parts]
        gt_masks = np.stack([m.cpu().numpy() for m in gt_masks])

        macro_giou_metrics = iou_evaluator_macro.update(pred_masks, gt_masks)
        micro_giou_metrics = iou_evaluator_micro.update(pred_masks, gt_masks)

        # Output predictions
        if args.output_predictions:
            bsize = len(batch.img_paths)
            assert bsize == 1 # Prediction construction assumes batch size 1

            metrics = {
                iou_evaluator_micro.metric_group_name: micro_giou_metrics,
                iou_evaluator_macro.metric_group_name: macro_giou_metrics,
                part_text_evaluator.metric_group_name: part_text_metrics,
                random_part_text_evaluator.metric_group_name: random_part_text_metrics,
                mc_part_text_evaluator.metric_group_name: mc_part_text_metrics,
                mc_object_text_evaluator.metric_group_name: mc_object_text_metrics if has_object_question else None
            }

            predictions.add_prediction(
                Prediction(
                    image_path=batch.img_paths[0],

                    question_type=batch.question_types[0],
                    questions=batch.questions[0],

                    # Parts question outputs
                    parts_answer_choices=batch.part_answer_choices[0],

                    gt_parts_answer=batch.part_answer_choices[0][gt_parts_answer_index],
                    predicted_parts_answer=batch.part_answer_choices[0][predicted_parts_answer_index],

                    gt_parts=gt_parts,
                    predicted_parts=predicted_parts,

                    gt_masks=get_mask_rle_dicts(gt_masks, gt_parts),
                    predicted_masks=get_mask_rle_dicts(pred_masks, gt_parts),

                    # Object question outputs
                    object_answer_choices=batch.object_answer_choices[0],

                    gt_object_answer=batch.object_answer_choices[0][gt_object_answer_index] if has_object_question else None, # Prevent indexing into None
                    predicted_object_answer=batch.object_answer_choices[0][predicted_object_answer_index] if has_object_question else None,

                    gt_object=batch.answer_objects[0][gt_object_answer_index] if has_object_question else None, # Prevent indexing into None
                    predicted_object=batch.answer_objects[0][predicted_object_answer_index] if has_object_question else None,

                    metrics=metrics
                )
            )
    
    if args.local_rank == 0:
        # Output predictions
        if args.output_predictions:
            output_dir = os.path.dirname(args.predictions_path)
            os.makedirs(output_dir, exist_ok=True)

            # Dump config
            config_path = os.path.join(output_dir, 'config.yaml')
            if not os.path.exists(config_path):
                if "seg_token_idx" in args:  # 'args.seg_token_idx' was dynamically added to args following tokenizer - will cause error during save
                    del args.seg_token_idx
                parser.save(args, config_path) # This fails if the file already exists

            # Dump predictions
            import json

            predictions_dict = predictions.to_dict([
                iou_evaluator_micro,
                iou_evaluator_macro,
                part_text_evaluator,
                random_part_text_evaluator,
                mc_part_text_evaluator,
                mc_object_text_evaluator
            ])

            with open(args.predictions_path, 'w') as f:
                logger.info(f"Predictions saved in {args.predictions_path} for QuestionType={args.question_type.value}")
                json.dump(predictions_dict, f, indent=4)

        # Log metrics
        logger.info(f'Metrics:\n{pformat(predictions_dict["metrics"], indent=4)}')

    logger.info(f">> skipped_instances: {skipped_instance_num}")
    
    logger.info(f'PixelLM Validation on {args.val_dataset} Completed.')


if __name__ == "__main__":
    coloredlogs.install(level='DEBUG')
    # main(sys.argv[1:])

    # Ansel Config
    question_types = [
        # QuestionType.IDENTIFICATION,
        QuestionType.IDENTIFICATION_WITH_LABEL,
        # QuestionType.POSITIVE,
        QuestionType.POSITIVE_WITH_LABEL,
        # QuestionType.NEGATIVE,
        QuestionType.NEGATIVE_WITH_LABEL,
        # QuestionType.DIFFERENCE,
        # QuestionType.DIFFERENCE_WITH_LABEL
        QuestionType.PART_TO_WHOLE,
        QuestionType.WHOLE_TO_PART,
    ]

    timestr = get_timestr()
    
    seg_token_num = 1
    image_feature_scale_num = 2
    model_size = '13B'  # '13B'
    
    for dataset_split in ['partonomy']:   # ['partimagenet', 'paco_lvis', 'pascal_part']:
        for question_type in question_types:
            logger.info('=' * 10 + f'Validating question type: {question_type.name}' + '=' * 10)

            main(config_str=f'''
                # version: ./runs/PixelLM-{model_size}/hf_model
                version: ./runs/pixellm-13b/merged_model_joint
                
                dataset_dir: dataset/partonomy_descriptors
                dataset_path: {dataset_split}/{dataset_split}_qa_pairs_val.json
                
                vision_pretrained: weights/sam_vit_h_4b8939.pth
                exp_name: pixellm-13b-joint-ft-{dataset_split}
                
                eval_only: true
                pad_val_clip_images: true
                
                preprocessor_config: ./configs/preprocessor_448.json

                resize_vision_tower: true
                resize_vision_tower_size: 448
                vision_tower_for_mask: true
                image_feature_scale_num: {image_feature_scale_num}
                separate_mm_projector: true
                seg_token_num: {seg_token_num}
                

                # precision: fp32
                # limit_batches: 200
                device: cuda
                question_type: {question_type.name}

                # test_with_gt_object: true
                output_predictions: true
                predictions_path: /shared/nas2/jk100/partonomy_private/results/{exp_name}/{dataset_split}/{timestr}-predictions/{question_type.value}_seg_token_num_{seg_token_num}_feature_scale_num_{image_feature_scale_num}.json
                # predictions_path: /shared/nas2/blume5/sp25/partonomy/results/{timestr}-predictions/{question_type.value}.json

                metrics:
                    iou_evaluator_config:
                        matching_strategy: ALL
            ''')

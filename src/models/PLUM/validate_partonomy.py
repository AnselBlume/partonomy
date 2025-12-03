import jsonargparse as argparse
import os
import sys
from functools import partial
from pprint import pformat

import torch
import tqdm
import transformers
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Sequence, Iterable, Any, Callable, Mapping

from model.PLUM import PLUMForCausalLM
from model.llava import conversation as conversation_lib

from utils.dataset import HybridDataset, ValDataset

from utils.explanatory_dataset import collate_fn, ExplanatorySegBatch
from utils.explanatory_seg_dataset import ExplanatorySegDataset

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter, ProgressMeter, Summary, dict_to_device,
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
from utils.explanatory_seg_dataset import ConversationType, ConversationQuestionType
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

    parser.add_argument('--output_predictions', type=bool, default=False)
    parser.add_argument('--predictions_path', type=str, default='predictions.json')

    # Evaluators
    parser.add_argument('--metrics.iou_evaluator_config', default=IoUEvaluatorConfig(), type=IoUEvaluatorConfig) # NOTE: Change to either reduction='macro' or reduction='micro'
    parser.add_argument('--metrics.part_text_evaluator_config', default=PartTextEvaluatorConfig(), type=PartTextEvaluatorConfig)

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

    # build dataset for validation
    val_dataset = ExplanatorySegDataset(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        question_type=args.question_type,
        image_size=args.image_size,
        model_str='plum',
        test_with_gt_object=args.test_with_gt_object,
        output_question_prompt_for_generation=args.output_generation_prompt_in_dataset
    )
    
    # sanity check the loaded dataset
    print(f"* (validate_partonomy) >> Loaded {len(val_dataset)} samples from the validation dataset *")

    # build dataloader
    if val_dataset is not None:
        assert args.val_batch_size == 1, "Currently supports batch_size=1 for segmentation eval"
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
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


def to_plum_inputs(args, batch: ExplanatorySegBatch, tokenizer: transformers.AutoTokenizer, set_mc_choices=False):
    input_dict = {
        'images': batch.sam_img_inputs,
        'images_clip': batch.clip_img_inputs,
        "input_ids": batch.input_ids,
        'resize_list': batch.resized_img_dims,
        'label_list': batch.label_masks,
        'attention_masks': batch.attention_masks,
        'gt_bio_span': batch.per_token_labels,
        'max_new_tokens': args.model_max_length,
        'tokenizer': tokenizer,
        'multiple_choice': set_mc_choices,
        'questions`: batch.questions,'
        'question_types': batch.question_types,
        'conversations': batch.conversations,
        'conversation_types': batch.conversation_types,
        'conversation_question_types': batch.conversation_question_types,
        'part_answer_choices': batch.part_answer_choices,
        'part_answer_types': batch.part_answer_types,
        'answer_parts': batch.answer_parts,
        'object_answer_choices': batch.object_answer_choices,
        'object_answer_types': batch.object_answer_types,
        'answer_objects': batch.answer_objects,
        'per_token_labels': batch.per_token_labels,
        'mask_positions_in_input_ids': batch.mask_positions_in_input_ids,
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
def validate(val_loader, model_engine, tokenizer, epoch, writer, args, parser):
    '''
    Validation function specialized for 'partonomy' QA pairs, using PLUM's .evaluate() method.
    
    '''
    # Configure your evaluators
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

    # If you want a random baseline for part text
    random_part_text_evaluator = PartTextEvaluator(args.metrics.part_text_evaluator_config,
                                                   metric_group_name='random_part_text')
    
    rng = np.random.default_rng(42)

    predictions = Predictions()
    
    model_engine.eval()

    skipped_instance_num = 0
    n_batches = args.limit_batches if args.limit_batches is not None else len(val_loader)
    val_loader = islice(val_loader, n_batches)

    val_iter = iter(val_loader)

    for idx, batch in enumerate(tqdm.tqdm(val_loader, total=n_batches, desc="Validating Partonomy with PLUM")):
        batch: ExplanatorySegBatch
        
        if 'cuda' in args.device:
            torch.cuda.empty_cache() # Maybe we can remove this?

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
        # Capture only the multiple choice questions to feed to PLUM (if there is also a generation prompt)
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
        
        for i, (input_ids, conversation_type, conversation_question_type) in enumerate(zip(batch.input_ids, conversation_types, conversation_question_types)):
            if conversation_type in ['correct_answer', 'incorrect_answer']:
                # n_mask_outputs_for_conversation = (input_ids == tokenizer.added_tokens_encoder['[SEG]']).sum() # - XXX: This is for the original LISA code where [SEG] corresponds to a part
                if conversation_question_type == 'part_question' and i >= len(batch.answer_parts[0]):
                    n_mask_outputs_for_conversation = len(batch.answer_parts[0][i - len(batch.answer_parts[0])])
                
                    # Compute offset of ground truth answer masks in set of output masks
                    if conversation_type == 'correct_answer':
                        pred_masks_for_gt_parts_end_idx = pred_masks_for_gt_parts_start_idx + n_mask_outputs_for_conversation
                        break
                    elif pred_masks_for_gt_parts_end_idx is None:
                        pred_masks_for_gt_parts_start_idx += n_mask_outputs_for_conversation

        # Forward Pass
        output_dict = model_engine.evaluate(**to_plum_inputs(args, batch, tokenizer, set_mc_choices=True))  # multiple_choice == True only for Partonomy - this nees to be set to return conversation-level loss

        if output_dict is None:
            skipped_instance_num += 1
            continue

        output_ids = output_dict.get("output_ids", None)
        spans_dicts = output_dict.get("spans_dicts", None)
        pred_masks = output_dict.get("pred_masks", None)  # list[list[torch.Tensor]] - e.g., len(pred_masks) == 5 since there are 5 answer choices and len(pred_masks[0]) == 3 if there are 3 parts
        ce_losses = output_dict.get("ce_losses", None)
        
        # Convert pred_masks into a tensor of shape (num_total_parts, H, W)
        no_pred_masks_cnt = 0
        gt_parts_answer_index = conversation_types[0].index('correct_answer')
        try:
            pred_masks = torch.cat([(pred_mask > 0).int() for pred_mask in pred_masks[gt_parts_answer_index]], dim=0)
            # pred_masks = torch.stack(list(chain.from_iterable(pred_masks[gt_parts_answer_index]))).squeeze(1)
        except Exception as e:
            print(f"Error in converting pred_masks to tensor: {e}")
            pred_masks = None
            no_pred_masks_cnt += 1
        
        mismatch_spans_dicts_cnt = 0
        try:
            assert len(spans_dicts) == len(batch.input_ids), \
                f"len(spans_dicts): {len(spans_dicts)} || input_ids.size(0): {input_ids.size(0)}"  # i.e., number of answer choices
        except Exception as e:
            print(f"Error in checking spans_dicts: {e}")
            mismatch_spans_dicts_cnt += 1
            continue

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
        answer_parts = batch.answer_parts[0]
        gt_parts = answer_parts[gt_parts_answer_index]
        predicted_parts = answer_parts[predicted_parts_answer_index]
        
        part_text_metrics = part_text_evaluator.update(predicted_parts, gt_parts)
        
        # Computer for random answer selection
        random_index = rng.integers(len(answer_parts))
        random_answer_parts = answer_parts[random_index]
        random_part_text_metrics = random_part_text_evaluator.update(random_answer_parts, gt_parts)
        
        # TODO Compute AP50
        
        # Compute gmIoU
        mask_dict = batch.mask_dicts[0]  # ground-truth masks
        gt_masks = np.stack([mask_dict[gt_part].cpu().numpy() for gt_part in gt_parts])
        
        if pred_masks is None:
            logger.warning("No predicted masks found from PLUM evaluate. Skipping.")
            skipped_instance_num += 1
            continue
        
        gt_choice_masks = pred_masks[pred_masks_for_gt_parts_start_idx:pred_masks_for_gt_parts_end_idx] # pred_masks[gt_parts_answer_index]
        predicted_choice_masks = pred_masks[pred_masks_for_gt_parts_start_idx:pred_masks_for_gt_parts_end_idx]  # TODO: Later assign a separate pred_masks_for_pred_parts_start_idx and pred_masks_for_pred_parts_end_idx
        
        gt_predicted_masks = gt_choice_masks
        pred_predicted_masks = predicted_choice_masks
        gt_predicted_masks = np.stack([m.cpu().numpy() for m in gt_predicted_masks])  # predicted masks using gt text span embeddings)

        if len(pred_predicted_masks) == 0:
            pred_predicted_masks = gt_predicted_masks
        else:
            pred_predicted_masks = np.stack([m.cpu().numpy() for m in pred_predicted_masks])
        
        logger.debug(f'Number of predicted pixels (for gt span): {gt_predicted_masks[0].sum()}')
        logger.debug(f'Number of predicted pixels (for pred span): {pred_predicted_masks[0].sum()}')
        
        gt_mask_metrics = iou_evaluator.update(gt_predicted_masks, gt_masks)
        # pred_mask_metrics = iou_evaluator.update(pred_predicted_masks, gt_masks)
        macro_giou_metrics = iou_evaluator_macro.update(gt_predicted_masks, gt_masks)
        micro_giou_metrics = iou_evaluator_micro.update(gt_predicted_masks, gt_masks)
        
        print("=" * 20)
        print(">>> macro_giou_metrics: ", macro_giou_metrics)
        print(">>> micro_giou_metrics: ", micro_giou_metrics)
        print("=" * 20)
        
        mismatch_masks_cnt = 0
        try:
            assert len(gt_masks) == len(gt_parts) == len(gt_predicted_masks), f"len(gt_masks): {len(gt_masks)} || len(gt_parts): {len(gt_parts)} || len(gt_predicted_masks): {len(gt_predicted_masks)}"
        except Exception as e:
            print(f"len(gt_masks): {len(gt_masks)} || len(gt_parts): {len(gt_parts)} || len(gt_predicted_masks): {len(gt_predicted_masks)}")
            mismatch_masks_cnt += 1
            continue
            
        # Output predictions
        if args.output_predictions:
            bsize = len(batch.img_paths)
            assert bsize == 1  # Prediction construction assumes batch size 1
            
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
                    predicted_masks=get_mask_rle_dicts(gt_predicted_masks, gt_parts),  # gt_predicted_masks or pred_predicted_masks

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
                json.dump(predictions_dict, f, indent=4)

        # Log metrics
        logger.info(f'Metrics:\n{pformat(predictions_dict["metrics"], indent=4)}')
        
    logger.info(f">> [validate_partonomy] Skipped instances: {skipped_instance_num} | No pred masks: {no_pred_masks_cnt} | Mismatch spans dicts: {mismatch_spans_dicts_cnt}")


# if __name__ == "__main__":
#     coloredlogs.install(level='DEBUG')

#     main(sys.argv[1:])
    
    
if __name__ == "__main__":
    
    coloredlogs.install(level='DEBUG')
    # main(sys.argv[1:])

    # Ansel Config
    
    question_types = [
        QuestionType.IDENTIFICATION,
        QuestionType.IDENTIFICATION_WITH_LABEL,
        QuestionType.POSITIVE,
        QuestionType.POSITIVE_WITH_LABEL,
        QuestionType.NEGATIVE,
        QuestionType.NEGATIVE_WITH_LABEL,
        QuestionType.DIFFERENCE,
        QuestionType.DIFFERENCE_WITH_LABEL,
        QuestionType.PART_TO_WHOLE,
        QuestionType.WHOLE_TO_PART,
    ]

    timestr = get_timestr()
    ROOT_PATH = ""  # TODO: Set the root path that contains both the weights and the dataset
    LLAVA_MODEL_PATH = "liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    exp_name = "plum-13b_kld_0.1_focal_tversky_8_v1_partonomy_ft"
    exp_ckpt_name = "merged_model"
    dataset_split = "partonomy"  # 'partimagenet', 'paco_lvis', 'pascal_part', 'joint', 'partonomy' (NOTE: 'partonomy' is the partonomy-core split)
    
    for question_type in question_types:
        logger.info('=' * 10 + f'Validating question type: {question_type.name}' + '=' * 10)

        main(config_str=f'''
            backbone: {LLAVA_MODEL_PATH}
            version: ./runs/{exp_name}/{exp_ckpt_name}
            dataset_path: {ROOT_PATH}/dataset/{dataset_split}/{dataset_split}_qa_pairs_val.json
            vision_pretrained: {ROOT_PATH}/weights/sam_vit_h_4b8939.pth
            exp_name: {exp_name}_{dataset_split}

            eval_only: true
            use_bidir_bio: true
            use_feedback_loop: true

            # precision: fp32
            # limit_batches: 25
            device: cuda
            question_type: {question_type.name}

            # test_with_gt_object: true
            output_predictions: true
            predictions_path: {ROOT_PATH}/results/{exp_name}/{dataset_split}/{timestr}-predictions/{question_type.value}.json

            metrics:
                iou_evaluator_config:
                    matching_strategy: ALL
        ''')


from pycocotools import mask as maskutils
from dataclasses import dataclass
from collections import defaultdict
import logging
import numpy as np
from enum import Enum
from scipy.optimize import linear_sum_assignment as hungarian_alg
from .evaluator import Evaluator
from typing import Literal

logger = logging.getLogger(__name__)

@dataclass
class PairedIoU:
    mask1_index: int
    mask2_index: int
    iou: float

class MatchingStrategy(Enum):
    PAIRED = 'paired'
    GREEDY = 'greedy'
    HUNGARIAN = 'hungarian'
    UNION = 'union'
    ALL = 'all'
    
class Reduction(Enum):
    MACRO = 'macro'
    MICRO = 'micro'

class Reduction(Enum):
    MACRO = 'macro'   # one value per image, then average across images
    MICRO = 'micro'   # one value per mask, then average cross images

@dataclass
class IoUEvaluatorConfig:
    matching_strategy: MatchingStrategy = MatchingStrategy.PAIRED
    reduction: Reduction = Reduction.MACRO

class IoUEvaluator(Evaluator):
    DEFAULT_METRIC_GROUP_NAME = 'mask'

    def __init__(self, config: IoUEvaluatorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.ious: dict[str, list[float]] = defaultdict(list)

    def update(self, predictions: list, targets: list, return_paired_ious: bool = False):
        matching_strategies = (
            [self.config.matching_strategy] if self.config.matching_strategy != MatchingStrategy.ALL
            else [strategy for strategy in MatchingStrategy if strategy != MatchingStrategy.ALL]
        )

        iou_dict = {}
        for matching_strategy in matching_strategies:
            paired_ious, mean_iou = gmIoU(predictions, targets, matching_strategy)

            metric_name = self._metric_name(matching_strategy)
            iou_dict[metric_name] = mean_iou # Return the per-image mean IoU

            if self.config.reduction is Reduction.MACRO: # compute per-image mean IoU then average across all images
                self.ious[metric_name].append(mean_iou)
            else:  # Reduction.MICRO; compute per-mask IoU then average across all masks
                self.ious[metric_name].extend(pi.iou for pi in paired_ious)

            if return_paired_ious:
                iou_dict[f'{metric_name}-pairs'] = paired_ious

        return iou_dict

    def summarize(self):
        return {
            metric_name : float(np.mean(l))
            for metric_name, l in self.ious.items()
        }

    def _metric_name(self, matching_strategy: MatchingStrategy) -> str:
        return f'gmIoU-{matching_strategy.value}'
    

EmptyPolicy = Literal['skip', 'zero', 'one']


EmptyPolicy = Literal['skip', 'zero', 'one']

def gmIoU(predictions: list, targets: list, matching_strategy: MatchingStrategy) -> tuple[list[PairedIoU], float]:
    '''
    Computes the grounded mean IoU as implemented by GLaMM: calculates the best matching between predicted and target masks,
    calculates the IoU for each pair, then averages the IoU values.

    Args:
    - predictions (list): list of predicted masks (either as COCO RLE dict or numpy binary masks)
    - targets (list): list of target masks (either as COCO RLE dict or numpy binary masks)
    - matching_strategy (MatchingStrategy): strategy to use for matching the predicted and target masks

    Returns:
    - ious (list[PairedIoU]): list of paired IoU values
    - mean_iou (float): mean IoU value for all masks in this image
    '''
    if isinstance(predictions[0], dict):
        predictions = np.array([maskutils.decode(prediction) for prediction in predictions])
    if isinstance(targets[0], dict):
        targets = np.array([maskutils.decode(target) for target in targets])

    if matching_strategy == MatchingStrategy.PAIRED:
        ious = [
            PairedIoU(
                mask1_index=i,
                mask2_index=i,
                iou=compute_iou(pred, target)
            )
            for i, (pred, target) in enumerate(zip(predictions, targets))
        ]

    elif matching_strategy == MatchingStrategy.GREEDY:
        iou_matrix = compute_iou_matrix(predictions, targets)
        ious = compute_greedy_ious(iou_matrix)

    elif matching_strategy == MatchingStrategy.HUNGARIAN:
        iou_matrix = compute_iou_matrix(predictions, targets)
        if np.isnan(iou_matrix).any():
            print("Warning: NaN values found in IoU matrix. Replacing with -1.")
        iou_matrix = np.nan_to_num(iou_matrix, nan=-1.0)
        row_inds, col_inds = hungarian_alg(iou_matrix, maximize=True)
        ious = [
            PairedIoU(
                mask1_index=row_inds[i],
                mask2_index=col_inds[i],
                iou=iou_matrix[row_inds[i], col_inds[i]]
            )
            for i in range(len(row_inds))
        ]

    elif matching_strategy == MatchingStrategy.UNION:
        # Take union over all masks
        predictions = np.any(predictions, axis=0)
        targets = np.any(targets, axis=0)

        iou = compute_iou(predictions, targets)
        ious = [PairedIoU(mask1_index=0, mask2_index=0, iou=iou)]

    else:
        raise RuntimeError(f'Invalid matching strategy: {matching_strategy}')


    for piou in ious:
        piou.iou = float(piou.iou)

    return ious, float(np.mean([iou.iou for iou in ious]))

def compute_greedy_ious(iou_matrix) -> list[PairedIoU]:
    '''
    Based on https://github.com/mbzuai-oryx/groundingLMM/blob/4073365f652f9ea27ee53daabb417cb4da8361de/eval/gcg/evaluate.py#L42
    but doesn't delete rows and columns to maintain indices and potentially be faster.
    '''
    candidate_dict = {
        (i, j): iou_matrix[i, j]
        for i in range(iou_matrix.shape[0])
        for j in range(iou_matrix.shape[1])
    }

    # Sort by IoU in descending order
    sorted_candidates = sorted(candidate_dict.items(), key=lambda item: item[1], reverse=True)

    paired_ious = []
    used_rows = set()
    used_cols = set()

    # Greedily select pairs; each row and column should only be used once
    for (row, col), iou in sorted_candidates:
        if row in used_rows or col in used_cols:
            continue

        paired_ious.append(PairedIoU(mask1_index=row, mask2_index=col, iou=iou))
        used_rows.add(row)
        used_cols.add(col)

    return paired_ious

def compute_iou_matrix(pred_masks, gt_masks):
    '''
    From: https://github.com/mbzuai-oryx/groundingLMM/blob/4073365f652f9ea27ee53daabb417cb4da8361de/eval/gcg/evaluate.py#L88
    '''
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    return np.array(iou_matrix)

def compute_iou(mask1, mask2, empty_policy: EmptyPolicy = 'one', warn_on_empty: bool = True):
    '''
    From: https://github.com/mbzuai-oryx/groundingLMM/blob/4073365f652f9ea27ee53daabb417cb4da8361de/eval/utils.py#L86
    '''
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    union_sum = np.sum(union)
    if union_sum == 0:
        if warn_on_empty:
            logger.warning(f'Union is empty; following empty_policy {empty_policy}')

        if empty_policy == 'skip':
            return np.nan
        elif empty_policy == 'zero':
            return 0.0
        elif empty_policy == 'one':
            return 1.0
        else:
            raise ValueError(f"Invalid empty_policy: {empty_policy}")
    return iou
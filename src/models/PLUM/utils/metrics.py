import torch
from collections import Counter

def is_subsequence_in_window(window_tokens: list, gt_tokens: list) -> bool:
    '''
    Checks if the ground-truth token sequence, 'gt_tokens', appears as a contiguous subsequence in the list window_tokens.
    '''

    n = len(window_tokens)
    m = len(gt_tokens)
    if m == 0 or m > n:
        return False

    for i in range(n - m + 1):
        if window_tokens[i:i + m] == gt_tokens:
            return True
    return False


def standard_metrics(generated_ids: torch.Tensor, ground_truth_ids: torch.Tensor):
    '''
    Compute regular precision, recall, and F1 score from generate LLM outputs ('generated_ids').
    
    Args:
      - generated_ids: torch.Tensor of token IDs from the generated text.
      - ground_truth_ids: torch.Tensor of token IDs representing the ground-truth answer.
    
    Return:
      - A tuple (precision, recall, f1) computed over the entire generated sequence.
    '''

    pred_tokens = generated_ids.tolist()
    gt_tokens = ground_truth_ids.tolist()    
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def relaxed_metrics(generated_ids: torch.Tensor, ground_truth_ids: torch.Tensor, m: int):
    '''
    Compute the relaxed Precision, Recall, and F1 score based on token IDs.
    
    Ags:
      - generated_ids: torch.Tensor of token IDs from the generated text.
      - ground_truth_ids: torch.Tensor of token IDs representing the ground-truth answer.
      - m: The number of tokens to consider as the window.
      
    Return:
      - (precision, recall, f1)
    '''

    generated_list = generated_ids.tolist()
    gt_list = ground_truth_ids.tolist()
    window_tokens = generated_list[:m]  # if the first 'm' tokens contain the ground-truth, consider it correct
    
    if is_subsequence_in_window(window_tokens, gt_list):
        return 1.0, 1.0, 1.0
    else:
        return standard_metrics(generated_ids, ground_truth_ids)
    
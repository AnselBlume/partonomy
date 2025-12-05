from typing import TypedDict
import numpy as np
from einops import rearrange
import pycocotools.mask as mask_utils

class RLEDict(TypedDict):
    size: list[int] = None
    counts: str = None
    bbox: list[int] = None
    label: str = None

def get_mask_rle_dicts(masks: np.ndarray, labels: list[str]) -> RLEDict:
    '''
        masks: (n_masks, h, w)
        labels: (n_masks,)
    '''
    assert len(masks) == len(labels)

    masks = rearrange(masks, 'n h w -> h w n').astype(np.uint8) # encode expects (h, w, n)
    rle_dicts = mask_utils.encode(np.asfortranarray(masks))
    
    if len(labels) < len(rle_dicts):
        labels += ['no_label'] * (len(rle_dicts) - len(labels))

    for rle_dict, label in zip(rle_dicts, labels):
        rle_dict['label'] = label
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8') # Convert bytes to str for JSON serializability

    return rle_dicts
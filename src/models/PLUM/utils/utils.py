from enum import Enum

import os
import yaml
import json
import cv2
import pycocotools.mask as mask_utils
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as PILImage


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

TOK_START_CHAR = '▁'  # token start character
BIO_NO_LBL = 0
BIO_START_LBL = 1
BIO_INTERM_LBL = 2
ASSISTANT_START_TOK = "ASSISTANT:"
IMAGE_TOKENS_OFFSET = 255  # 255 is the number of image tokens in the tokenizer


SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please elaborate your answer and explain why.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please give some explanation.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you isolate the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Would you please extract the {class_name} from the image below?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {class_name} from the provided image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Display a segmentation mask for the {class_name} shown in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please highlight the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you identify and segment out the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please point out the {class_name} in this picture.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please elaborate your answer and explain why.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent}. Please give some explanation.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

PLUM_REASONSEG_EXPLANATORY_QUESTION_LIST = [
    "Please explain why.",
    "Please elaborate your answer and explain why.",
    "Please give some explanation.",
    "Feel free to explain your answer.",
    "Please provide some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

PLUM_REASONSEG_ANSWER_LIST = [
    "{class_name}"
]

PLUM_ANSWER_LIST = [
    "Here is the {class_name} you asked about.",
    "Sure — this shows the {class_name}.",
    "The {class_name} is presented here.",
    "Below you can see the {class_name}.",
    "{class_name}.",
    "This view focuses on the {class_name}.",
    "I've marked the {class_name} for you.",
    "Here's where the {class_name} appears in the image.",
    "Take a look at the {class_name} here.",
    "The region corresponding to the {class_name} is shown.",
    "Result: {class_name}.",
    "You can see the {class_name} in this frame.",
    "The {class_name} portion is displayed below.",
    "Here's the area for {class_name}.",
    "This is the selected {class_name}.",
    "Displayed here is the {class_name}."
]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    return dict_to_device(input_dict, torch.device("cuda"))

def dict_to_device(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(device, non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(device, non_blocking=True) for ele in v]
    return input_dict


def load_yaml(path: str, encoding='utf-8'):
    if os.path.exists(path):
        print(f"loading the yaml file from : {path}")
        with open(path, 'r') as f:
            graph_yaml = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"[ {path} ] does not exist in the file system.")
    return graph_yaml


def load_json(path: str):
    if os.path.exists(path):
        with open(path, 'r') as f:
            json_dict = json.load(f)
    else:
        raise FileNotFoundError(f"[ {path} ] does not exist in the file system.")
    return json_dict


def load_rle(rle_dict: dict):
    # open the JSON dictionary with "size" and "counts"
    dense_mask = mask_utils.decode(rle_dict)
    return dense_mask


def load_polygon(segmentation: list, height: int, width: int):
    '''
    Converts a polygon segmentation into a binary mask using OpenCV.

    Args:
        - segmentation (list of float): List of polygon coordinates [x1, y1, x2, y2, ..., xn, yn]
        - height (int): height of the mask
        - width (int): width of the mask
    '''
    polygon = np.array(segmentation).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask

def open_image(path: str) -> PILImage:
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def is_alphanumeric_or_underscore(char: str) -> bool:
    '''
    Returns True if 'char' is alphanumeric or underscore,
    used to check if token boundaries are within a larger word.
    '''
    return char.isalnum() or char == '_'


def find_sublist_index(full_list, sublist):
    '''
    Find the starting index of sublist in full_list
    - full_list: list of token ids for the full text
    - sublist: list of token ids for the substrings to match
    '''
    if not isinstance(full_list, list):
        full_list = full_list.tolist()
    if not sublist:
        return 0
    start_indices = []
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i:i+len(sublist)] == sublist:
            start_indices.append(i)
    return start_indices


def match_substrings(
    full_text_ids: list[int],
    substring_ids_list: list[list[int]],
    tokenizer,  # transformers AutoTokenizer
    anchor_token: str = "ASSISTANT:",
):
    '''
    - full_text_ids: list of token ids for the full text
    - substring_ids_list: list of lists of token ids for the substrings to match
    - tokenizer: transformers AutoTokenizer
    - anchor_token: the token to anchor the match to

    Returns:
    - matched_substrings: list of substrings that fall within the full text (after the 'anchor_token')
    - per_token_labels: list of BIO labels for each token in the full text
    - mask_positions: list of indices of the tokens that match the substrings
    '''
    matched_substrings = []
    per_token_labels = [BIO_NO_LBL] * len(full_text_ids)
    mask_positions = set()
    anchor_ids = tokenizer(anchor_token)['input_ids'][1:]
    anchor_idx = find_sublist_index(full_text_ids, anchor_ids)

    assert len(anchor_idx) == 1, f"Anchor token '{anchor_token}' found multiple times in the full text | anchor_idx: {anchor_idx} | full_text_ids: {full_text_ids} | anchor_text: {tokenizer.convert_ids_to_tokens(anchor_ids)} | full_text: {tokenizer.convert_ids_to_tokens(full_text_ids)}"
    anchor_idx = anchor_idx[0]
    
    # NOTE: multiple matches are possible for the same substring
    # substring match is only valid after the anchor_token
    # - e.g., "tail" in "red tail, curvy tail"
    for substring_ids in substring_ids_list:
        start_indices = find_sublist_index(full_text_ids[anchor_idx:], substring_ids)
        start_indices = [idx + anchor_idx for idx in start_indices]
        matched_substrings.extend([tokenizer.decode(substring_ids) for _ in start_indices])
        for substring_idx in start_indices:
            per_token_labels[substring_idx] = BIO_START_LBL
            for i in range(1, len(substring_ids)):
                per_token_labels[substring_idx + i] = BIO_INTERM_LBL
            mask_positions.add(substring_idx)
    
    return matched_substrings, per_token_labels, mask_positions

def _with_patch_offset(token_idx: int):
    return token_idx + IMAGE_TOKENS_OFFSET

def _imread_rgb_sans_icc(path: str) -> np.ndarray:
    '''
    Safely reads both png and jpg images, ignoring the ICC profile that causes IO cluttering.
    '''
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".png":
        # PIL will automatically load and convert to sRGB
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img)  # H x W x 3 uint8 (RGB)
    else:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
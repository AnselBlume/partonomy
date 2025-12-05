import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from PIL.Image import Image
import torch.nn as nn
from torchvision import transforms
from typing import Sequence, Optional
import torch
import math
import itertools
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, reduce, einsum
from typing import Union
from functools import partial
import logging, coloredlogs
logger = logging.getLogger(__file__)

#############################
# DINOv2 Model Construction #
#############################

DEFAULT_DINO_MODEL = 'dinov2_vitg14_reg'

def build_dino(model_name: str = DEFAULT_DINO_MODEL, device: str = 'cuda'):
    return torch.hub.load('facebookresearch/dinov2', model_name).to(device)

############################
# DINOv2 Feature Utilities #
############################

# Transforms copied from
# https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py
# and
# https://github.com/michalsr/dino_sam/blob/0742c580bcb1fb24ad2c22bb3b629f35dabd9345/extract_features.py#L96
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.no_grad()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

class _MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

def _make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def _compute_resized_output_size(
    image_size: tuple[int, int], size: list[int], max_size: Optional[int] = None
) -> list[int]:
    '''
        Method to compute the output size for the resize operation.
        Copied from https://pytorch.org/vision/0.15/_modules/torchvision/transforms/functional.html
        since the PyTorch version used in desco environment doesn't have this method.
    '''
    if len(size) == 1:  # specified size only for the smallest edge
        h, w = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]

DEFAULT_RESIZE_SIZE = 256
DEFAULT_RESIZE_MAX_SIZE = 800
DEFAULT_RESIZE_INTERPOLATION = transforms.InterpolationMode.BICUBIC
DEFAULT_CROP_SIZE = 224

def get_dino_transform(
    crop_img: bool,
    *,
    padding_multiple: int = 14, # aka DINOv2 model patch size
    resize_img: bool = True,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    resize_max_size: int = DEFAULT_RESIZE_MAX_SIZE,
    interpolation = DEFAULT_RESIZE_INTERPOLATION,
    crop_size: int = DEFAULT_CROP_SIZE,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    '''
        If crop_img is True, will automatically set resize_img to True.
    '''
    # With the default parameters, this is the transform used for DINO classification
    if crop_img:
        transforms_list = [
            # DINO's orig transform doesn't have the max_size set, but we set it here to match
            # the behavior of our resize without cropping
            transforms.Resize(resize_size, interpolation=interpolation, max_size=resize_max_size),
            transforms.CenterCrop(crop_size),
            _MaybeToTensor(),
            _make_normalize_transform(mean=mean, std=std),
        ]

    # Transform used for DINO segmentation in Region-Based Representations revisited and at
    # https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb
    else:
        transforms_list = []

        if resize_img:
            transforms_list.append(
                transforms.Resize(resize_size, interpolation=interpolation, max_size=resize_max_size)
            )

        transforms_list.extend([
            transforms.ToTensor(),
            lambda x: x.unsqueeze(0),
            CenterPadding(multiple=padding_multiple),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transforms.Compose(transforms_list)

class DinoFeatureExtractor(nn.Module):
    def __init__(self, dino: nn.Module, resize_images: bool = True, crop_images: bool = False):
        super().__init__()

        self.model = dino.eval()
        self.resize_images = resize_images
        self.crop_images = crop_images
        self.transform = get_dino_transform(crop_images, resize_img=resize_images)

    @property
    def device(self):
        return self.model.cls_token.device

    def forward(self, images: list[Image]):
        '''
            image: list[PIL.Image.Image]
            See https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L44
            for model forward details.
        '''
        # Prepare inputs
        inputs = [self.transform(img).to(self.device) for img in images]

        if self.crop_images:
            inputs = torch.stack(inputs) # (n_imgs, 3, 224, 224)
            outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

            cls_tokens = outputs['x_norm_clstoken'] # (n_imgs, n_features)
            patch_tokens = outputs['x_norm_patchtokens'] # (n_imgs, n_patches, n_features)

            # Rearrange patch tokens
            n_patches_h, n_patches_w = torch.tensor(inputs.shape[-2:]) // self.model.patch_size
            patch_tokens = rearrange(patch_tokens, 'n (h w) d -> n h w d', h=n_patches_h, w=n_patches_w) # (n_imgs, n_patches_h, n_patches_w, n_features)

        else: # Padding to multiple of patch_size; need to run forward separately
            cls_tokens_l = []
            patch_tokens_l = []

            for img_t in inputs:
                outputs = self.model(img_t, is_training=True) # Set is_training=True to return all outputs

                cls_tokens = outputs['x_norm_clstoken'] # (1, n_features)
                patch_tokens = outputs['x_norm_patchtokens'] # (1, n_patches, n_features)

                # Rearrange patch tokens
                n_patches_h, n_patches_w = torch.tensor(img_t.shape[-2:]) // self.model.patch_size
                patch_tokens = rearrange(patch_tokens, '1 (h w) d -> h w d', h=n_patches_h, w=n_patches_w)

                cls_tokens_l.append(cls_tokens)
                patch_tokens_l.append(patch_tokens)

            cls_tokens = torch.cat(cls_tokens_l, dim=0) # (n_imgs, n_features)
            patch_tokens = patch_tokens_l # list[(n_patches_h, n_patches_w, n_features)]

        return cls_tokens, patch_tokens

    def forward_from_tensor(self, image: torch.Tensor):
        # Normalize & crop according to DINOv2 settings for ImageNet
        inputs = image.to(self.device)
        outputs = self.model(inputs, is_training=True) # Set is_training=True to return all outputs

        cls_token = outputs['x_norm_clstoken']
        patch_tokens = outputs['x_norm_patchtokens']

        return cls_token, patch_tokens

def rescale_features(
    features: torch.Tensor,
    img: Image = None,
    height: int = None,
    width: int = None,
    do_resize: bool = False,
    resize_size: Union[int, tuple[int,int]] = DEFAULT_RESIZE_SIZE
):
    '''
        Returns the features rescaled to the size of the image.

        features: (n, h_patch, w_patch, d) or (h_patch, w_patch, d)

        Returns: Interpolated features to the size of the image.
    '''
    if bool(img) + bool(width and height) + bool(do_resize) != 1:
        raise ValueError('Exactly one of img, (width and height), or do_resize must be provided')

    has_batch_dim = features.dim() > 3
    if not has_batch_dim: # Create batch dimension for interpolate
        features = features.unsqueeze(0)

    features = rearrange(features, 'n h w d -> n d h w').contiguous()

    # Resize based on min dimension or interpolate to specified dimensions
    if do_resize:
        features = TF.resize(features, resize_size, interpolation=DEFAULT_RESIZE_INTERPOLATION)

    else:
        if img:
            width, height = img.size
        features = F.interpolate(features, size=(height, width), mode='bilinear')

    features = rearrange(features, 'n d h w -> n h w d')

    if not has_batch_dim: # Squeeze the batch dimension we created to interpolate
        features = features.squeeze(0)

    return features

def get_rescaled_features(
    feature_extractor: DinoFeatureExtractor,
    images: list[Image],
    patch_size: int = 14,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    resize_max_size: int = DEFAULT_RESIZE_MAX_SIZE,
    crop_height: int = DEFAULT_CROP_SIZE,
    crop_width: int = DEFAULT_CROP_SIZE,
    interpolate_on_cpu: bool = False,
    fall_back_to_cpu: bool = False,
    return_on_cpu: bool = False
) -> tuple[torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]]:
    '''
        Extracts features from the image and rescales them to the size of the image.

        patch_size: The patch size of the Dino model used in the DinoFeatureExtractor.
            Accessible by feature_extractor.model.patch_size.
        crop_height: The height of the cropped image, if cropping is used in the DinoFeatureExtractor.
        crop_width: The width of the cropped image, if cropping is used in the DinoFeatureExtractor.
        interpolate_on_cpu: If True, interpolates on CPU to avoid CUDA OOM errors.
        fall_back_to_cpu: If True, falls back to CPU if CUDA OOM error is caught.
        return_on_cpu: If True, returns the features on CPU, helping to prevent out of memory errors when storing patch features
            generated one-by-one when not resizing multiple images.

        Returns: shapes (1, d), (1, h, w, d) or list[(h, w, d) torch.Tensor]
    '''

    with torch.no_grad():
        cls_feats, patch_feats = feature_extractor(images)

    are_images_cropped = feature_extractor.crop_images
    are_images_resized = feature_extractor.resize_images

    if return_on_cpu:
        cls_feats = cls_feats.cpu()

    def patch_feats_to_cpu(patch_feats):
        if isinstance(patch_feats, torch.Tensor):
            return patch_feats.cpu()

        else:
            assert isinstance(patch_feats, list)
            assert all(isinstance(patch_feat, torch.Tensor) for patch_feat in patch_feats)

            return [
                patch_feat.cpu()
                for patch_feat in patch_feats
            ]

    def try_rescale(rescale_func, patch_feats):
        try:
            return rescale_func(patch_feats)

        except RuntimeError as e:
            if fall_back_to_cpu:
                logger.info(f'Caught out of memory error; falling back to CPU for rescaling.')
                patch_feats = patch_feats_to_cpu(patch_feats)
                return rescale_func(patch_feats)

            else:
                raise e

    # Avoid CUDA oom errors by interpolating on CPU
    if interpolate_on_cpu:
        patch_feats = patch_feats_to_cpu(patch_feats)

    # Rescale patch features
    if are_images_cropped: # All images are the same size
        rescale_func = partial(rescale_features, height=crop_height, width=crop_width)
        patch_feats = try_rescale(rescale_func, patch_feats)

        if return_on_cpu:
            patch_feats = patch_feats.cpu()

    else: # Images aren't cropped, so each patch feature has a different dimension and comes in a list
        # Rescale to padded size
        rescaled_patch_feats = []

        for patch_feat, img in zip(patch_feats, images):
            if are_images_resized: # Interpolate to padded resized size
                # Compute resized dimensions used in resize method
                height, width = _compute_resized_output_size(img.size[::-1], [resize_size], max_size=resize_max_size)

                # Interpolate to padded resized size
                padded_resize_size = math.ceil(resize_size / patch_size) * patch_size
                rescale_func = partial(rescale_features, do_resize=True, resize_size=padded_resize_size)

            else: # Interpolate to full, padded image size
                width, height = img.size
                padded_height = math.ceil(height / patch_size) * patch_size
                padded_width = math.ceil(width / patch_size) * patch_size
                rescale_func = partial(rescale_features, height=padded_height, width=padded_width)

            rescaled = try_rescale(rescale_func, patch_feat)

            # Remove padding from upscaled features
            rescaled = rearrange(rescaled, 'h w d -> d h w')
            rescaled = TF.center_crop(rescaled, (height, width))
            rescaled = rearrange(rescaled, 'd h w -> h w d')

            if return_on_cpu:
                rescaled = rescaled.cpu()

            rescaled_patch_feats.append(rescaled)

        patch_feats = rescaled_patch_feats

    return cls_feats, patch_feats

def region_pool(masks: torch.BoolTensor, features: torch.Tensor, allow_empty_masks: bool = False):
    '''
        Computes the mean of the features within each mask.

        masks: (n, h, w) torch.BoolTensor
        features: (n, h, w, d) torch.Tensor

        Returns: (n, d) torch.Tensor
    '''
    assert masks.shape[-2:] == features.shape[-3:-1]
    feature_sums = einsum(masks.float(), features, 'n h w, n h w d -> n d') # Einsum needs floats
    n_pixels_per_mask = reduce(masks, 'n h w -> n', 'sum').unsqueeze(-1)

    empty_masks = n_pixels_per_mask == 0
    if empty_masks.any():
        if not allow_empty_masks:
            raise RuntimeError('Some masks have no pixels; cannot divide by zero')

        # Set empty masks to 1 to avoid division by zero
        logger.warning('Some masks have no pixels; setting number of pixels to 1 to avoid division by zero. This outputs a zero vector of features for empty masks.')
        n_pixels_per_mask[empty_masks] = 1

    # Divide by number of elements in each mask
    region_feats = feature_sums / n_pixels_per_mask

    return region_feats

def interpolate_masks(
    masks: torch.BoolTensor,
    do_resize: bool = False,
    do_crop: bool = False,
    resize_size: Union[int, tuple[int, int]] = DEFAULT_RESIZE_SIZE,
    resize_max_size: int = DEFAULT_RESIZE_MAX_SIZE,
    resize_interpolation = DEFAULT_RESIZE_INTERPOLATION,
    crop_size: Union[int, tuple[int, int]] = DEFAULT_CROP_SIZE
):
    '''
        Interpolates masks to the same size as the potentially resized image.

        masks: (n, h, w) torch.BoolTensor
    '''
    masks = masks.float()

    if do_resize:
        masks = TF.resize(masks, resize_size, resize_interpolation, max_size=resize_max_size)

    if do_crop:
        masks = TF.center_crop(masks, crop_size)

    return masks.round().bool() # Nop if not resized or cropped

if __name__ == '__main__':
    coloredlogs.install(logger=logger, level='DEBUG')

    dino = build_dino(device='cpu')
    dino_fe = DinoFeatureExtractor(dino, resize_images=True, crop_images=False)
    heatmap_gen = HeatmapGenerator(dino_fe, attention_pool_examples=False)

    # Load images
    input_image = open_image('input.jpg')
    positive_images = [open_image('positive1.jpg'), open_image('positive2.jpg')]
    negative_images = [open_image('negative1.jpg'), open_image('negative2.jpg')]
    # negative_images = [] # Without negatives

    # Generate heatmap
    heatmap = heatmap_gen.generate_heatmap(input_image, positive_images, negative_images)
    image = heatmap_gen.image_from_heatmap(
        heatmap,
        input_image,
        use_relative_heatmap=False,
        heatmap_blend_ratio=0.5,
        clamp_max=.25
    )
    image.save('heatmap.jpg')
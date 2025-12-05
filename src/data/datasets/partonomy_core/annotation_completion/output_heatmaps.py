import os
import pycocotools.mask as mask_utils
from tqdm import tqdm
from typing import Union
import json
import torch
from torch import Tensor
import numpy as np
from jsonargparse import ArgumentParser, Namespace
from data.utils import open_image
from sklearn.linear_model import LogisticRegression
from .data_collector import TrainDataCollectorConfig, TrainDataCollector
from .annotations import get_object_prefix, get_part_suffix, is_part_name
from .embed import ImageEmbedder, ImageEmbedderConfig
from einops import rearrange
from .heatmap import HeatmapGenerator, HeatmapGeneratorConfig
import logging, coloredlogs

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import numpy as np

class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, sklearn_lr, device='cpu'):
        """
        Initialize the torch wrapper by extracting weights and intercept
        from the sklearn LogisticRegression model.
        """
        # sklearn_lr.coef_ is of shape (n_classes, n_features)
        # sklearn_lr.intercept_ is of shape (n_classes,)
        self.device = device
        self.weight = torch.tensor(sklearn_lr.coef_, dtype=torch.float32, device=device)
        self.bias = torch.tensor(sklearn_lr.intercept_, dtype=torch.float32, device=device)
        self.classes_ = sklearn_lr.classes_

    @torch.inference_mode()
    def decision_function(self, X, return_as_numpy: bool = True) -> Union[np.ndarray, Tensor]:
        '''
        Compute the decision function (raw logits) for input data.
        X can be a numpy array or a torch tensor.
        '''
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.weight)

        # Compute logits: X @ W^T + b
        logits = X.matmul(self.weight.t()) + self.bias  # shape: (n_samples, n_classes)

        # For binary classification, sklearn returns a 1d array
        if len(self.classes_) == 2:
            return logits.squeeze(1)

        if return_as_numpy:
            logits = logits.cpu().numpy()

        return logits

    @torch.inference_mode()
    def predict_proba(self, X, return_as_numpy: bool = True) -> np.ndarray:
        '''
        Compute class probabilities for input data.
        For binary classification: probabilities for both classes are returned.
        For multi-class: softmax is applied.
        '''
        logits = self.decision_function(X, return_as_numpy=False)

        if len(self.classes_) == 2:
            # For binary classification, compute probability for the positive class
            prob_positive = torch.sigmoid(logits)
            prob_negative = 1 - prob_positive
            # Stack to shape (n_samples, 2)
            probs = torch.stack([prob_negative, prob_positive], dim=1)
        else:
            # For multi-class, use softmax along the class dimension.
            probs =  F.softmax(logits, dim=1)

        if return_as_numpy:
            probs = probs.cpu().numpy()

        return probs

def parse_args(cl_args: list[str] = None, config_str: str = None):
    parser = ArgumentParser()

    parser.add_argument('--mask_output_dir', required=True)

    parser.add_argument('--output_heatmap_images', type=bool, default=False)
    parser.add_argument('--image_output_dir')

    parser.add_argument('--data_collector_config', type=TrainDataCollectorConfig, required=True)
    parser.add_argument('--heatmap_generator_config', type=HeatmapGeneratorConfig, default=HeatmapGeneratorConfig())

    parser.add_argument('--restrict_heatmap_to_foreground', type=bool, default=False)
    parser.add_argument('--use_logits', type=bool, default=False)

    args = parser.parse_args(cl_args) if cl_args else parser.parse_string(config_str)

    return args, parser

def get_heatmap(img_path: str, embedder: ImageEmbedder, detector, output_logits: bool) -> Tensor:
    patch_embeds = embedder.get_patch_embeddings([img_path])[0]

    h, w = patch_embeds.shape[:2]
    patch_embeds = rearrange(patch_embeds, 'h w d -> (h w) d').cpu().numpy()

    if output_logits:
        heatmap = detector.decision_function(patch_embeds).reshape(h, w)
    else:
        heatmap = detector.predict_proba(patch_embeds)[:, 1].reshape(h, w)

    return torch.from_numpy(heatmap)

def get_heatmap_image_output_path(img_path: str, part_label: str, object_name: str, output_dir: str):
    part_name = get_part_suffix(part_label)
    img_basename = os.path.basename(img_path)
    img_basename = os.path.splitext(img_basename)[0]

    return os.path.join(output_dir, object_name, f'{part_name}-{img_basename}.jpg')

def get_rle_output_path(part_label: str, image_path: str, output_dir: str):
    image_basename = os.path.splitext(os.path.basename(image_path))[0] + '.json'
    return os.path.join(output_dir, part_label, image_basename)

def get_rle_dict(heatmap: Tensor, image_path: str) -> dict:
    rle_dict = mask_utils.encode(np.asfortranarray(heatmap.numpy()).astype(np.uint8))
    rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    rle_dict['image_path'] = image_path

    return rle_dict

def output_heatmaps(
    label: str,
    embedder: ImageEmbedder,
    collector: TrainDataCollector,
    heatmap_generator: HeatmapGenerator,
    args: Namespace
):
    logger.info(f'Building dataset for label: {label}')

    reenable_object_flags = False
    if not is_part_name(label) and (collector.config.use_objects_as_negatives or collector.config.use_category_parts_as_negatives):
        logger.warning(f'{label} is not a part name but '
                       f'use_objects_as_negatives={collector.config.use_objects_as_negatives} '
                       f'and use_category_parts_as_negatives={collector.config.use_category_parts_as_negatives}. Disabling...')

        prev_use_objects_as_negatives = collector.config.use_objects_as_negatives
        prev_use_category_parts_as_negatives = collector.config.use_category_parts_as_negatives

        collector.config.use_objects_as_negatives = False
        collector.config.use_category_parts_as_negatives = False

        reenable_object_flags = True

    embeds, labels = collector.build_dataset(label) # Build dataset

    if reenable_object_flags: # Reset flags
        collector.config.use_objects_as_negatives = prev_use_objects_as_negatives
        collector.config.use_category_parts_as_negatives = prev_use_category_parts_as_negatives

    # Train model
    logger.info(f'Training detector for label: {label}')
    embeds = [e.numpy() for e in embeds]
    labels = np.array(labels)

    detector = LogisticRegression(max_iter=1000)
    detector.fit(embeds, labels)

    # For some reason the sklearn LogisticRegression model hangs on large input, but torch does not
    detector = TorchLogisticRegression(detector)

    # Perform inference on parent images
    object_name = get_object_prefix(label)
    object_img_paths = collector.ds_metadata.img_paths_by_label[object_name]
    for object_img_path in object_img_paths:
        rle_path = get_rle_output_path(label, object_img_path, args.mask_output_dir)

        if os.path.exists(rle_path):
            continue

        heatmap = get_heatmap(object_img_path, embedder, detector, output_logits=args.use_logits)
        object_img = open_image(object_img_path)

        if args.restrict_heatmap_to_foreground:
            heatmap_generator.restrict_heatmap_to_foreground(heatmap, object_img)

        heatmap = heatmap_generator.postprocess_heatmap(heatmap)

        # Output RLE to file
        rle_dict = get_rle_dict(heatmap, object_img_path)

        os.makedirs(os.path.dirname(rle_path), exist_ok=True)
        with open(rle_path, 'w') as f:
            json.dump(rle_dict, f, indent=4)

        # Save image
        if args.output_heatmap_images:
            assert args.image_output_dir is not None, 'Must provide image_output_dir if output_heatmap_images is True'
            heatmap_img = heatmap_generator.image_from_heatmap(heatmap, object_img)
            heatmap_path = get_heatmap_image_output_path(object_img_path, label, object_name, args.image_output_dir)

            os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
            try:
                heatmap_img.save(heatmap_path)
            except OSError:
                logger.error(f'Failed to save image: {heatmap_path}')
                # Try with shorter filename
                basename = os.path.basename(heatmap_path)
                basename = basename[:100] # Limit to 100 characters
                heatmap_path = os.path.join(os.path.dirname(heatmap_path), basename)
                heatmap_img.save(heatmap_path)

def main(cl_args: list[str] = None, config_str: str = None):
    args, parser = parse_args(cl_args, config_str)

    embedder_config = ImageEmbedderConfig(device='cuda:0', resize_images=True)
    embedder = ImageEmbedder(config=embedder_config)

    collector = TrainDataCollector(embedder, args.data_collector_config)
    heatmap_generator = HeatmapGenerator(embedder, config=args.heatmap_generator_config)

    for part_label in tqdm(collector.ds_metadata.part_labels):
        output_heatmaps(part_label, embedder, collector, heatmap_generator, args)

if __name__ == '__main__':
    coloredlogs.install(level='DEBUG')
    logging.getLogger('PIL').setLevel(logging.WARNING) # Suppress PIL logging

    main(config_str='''
        data_collector_config:
            partonomy_root_dir: /shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-11-18/annotations/merged_annotations
            cache_dir: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_cleaning_cache
            use_objects_as_negatives: true
            use_category_parts_as_negatives: true
            use_imagenet_negatives: true

        restrict_heatmap_to_foreground: false
        heatmap_generator_config:
            use_relative_scaling: false
            use_absolute_scaling: false

            threshold_heatmap: true
            heatmap_threshold: 0.5

        mask_output_dir: /shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_part_heatmaps-.5

        output_heatmap_images: true
        image_output_dir:
            /shared/nas2/blume5/sp25/partonomy/results/heatmap_predictions-proba-.5
    ''')
import os
from torch import Tensor
import torch
from tqdm import tqdm
from dataclasses import dataclass
from .embed import get_rescaled_features, DinoFeatureExtractor, build_dino, rescale_features
from data.utils import open_image
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImageEmbedderConfig:
    cache_dir: str = None # Directory to cache image embeddings
    image_dir: str = None # Used for caching purposes to extract relative paths

    try_load_from_cache: bool = False
    save_to_cache: bool = False

    rescale_features_to_full_image_size: bool = True # Otherwise, use the resized size
    resize_images: bool = True
    crop_images: bool = False

    model_name: str = 'dinov2_vitl14_reg'
    device: str = 'cuda'

class ImageEmbedder:
    def __init__(self, config: ImageEmbedderConfig = ImageEmbedderConfig()):
        self.config = config

        # Build feature extractor
        dino = build_dino(model_name=config.model_name, device=config.device)
        self.fe = DinoFeatureExtractor(
            dino,
            resize_images=config.resize_images,
            crop_images=config.crop_images
        )

    def get_patch_embeddings(self, image_paths: list[str], save_to_cache: bool = None, use_tqdm: bool = False) -> list[Tensor]:
        '''
        Gets the image embeddings for the given image paths.

        Returns: List of image embeddings of shape (h, w, d).
        '''
        if save_to_cache is not None:
            save_to_cache = self.config.save_to_cache

        embeds_l = []
        for image_path in tqdm(image_paths, desc='Loading image embeds', disable=not use_tqdm):
            if self.config.try_load_from_cache and os.path.exists(cache_path):
                cache_path = self.image_path_to_cache_path(image_path)
                logger.debug(f'Loading image embeds from cache: {cache_path}')
                patch_embeds = torch.load(cache_path)

            else: # Compute image embedding
                image = open_image(image_path)
                logger.debug(f'Computing image embeds for: {image_path}')
                _, patch_feats = get_rescaled_features(
                    self.fe,
                    [image],
                )
                patch_embeds = patch_feats[0].cpu() # Extract from list

                if self.config.rescale_features_to_full_image_size:
                    patch_embeds = rescale_features(patch_embeds, image)

            embeds_l.append(patch_embeds)

            if self.config.save_to_cache:
                cache_path = self.image_path_to_cache_path(image_path)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(patch_embeds.cpu(), cache_path)

        return embeds_l

    def cache_patch_embeddings(self, image_paths: list[str]):
        '''
        Caches the image embeddings for the given image paths.
        '''
        self.get_patch_embeddings(image_paths, save_to_cache=True)

    def image_path_to_cache_path(self, image_path: str) -> str:
        '''
        Converts an image path to a cache path.
        '''
        rel_path = os.path.relpath(image_path, self.config.image_dir)
        cache_path = os.path.join(self.config.cache_dir, rel_path)
        cache_path = os.path.splitext(cache_path)[0] + '.pt'

        return cache_path

if __name__ == '__main__':
    from data.partonomy_core.annotation_completion.annotations import list_paths

    image_dir = '/shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-11-18/annotations/merged_annotations/images'
    cache_dir = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_embeds'

    cacher = ImageEmbedder(
        ImageEmbedderConfig(
            image_dir=image_dir,
            cache_dir=cache_dir
        )
    )

    image_paths = list_paths(image_dir)
    cacher.get_patch_embeddings(image_paths)
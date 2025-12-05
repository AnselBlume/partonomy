import torch
import numpy as np
import json
import pycocotools.mask as mask_utils
import os
from dataclasses import dataclass
from .annotations import (
    collect_annotations,
    RLEAnnotation,
    get_category_name,
    get_object_prefix,
    get_part_suffix,
    is_part_name
)
from data.utils import list_paths, open_image
from tqdm import tqdm
from .embed import ImageEmbedder, region_pool
from rembg import remove, new_session
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainDataCollectorConfig:
    partonomy_root_dir: str
    imagenet_imgs_dir: str = '/shared/nas2/blume5/fa23/ecole/data/imagenet/negatives_rand_1k'

    cache_dir: str = None

    foreground_pixel_min: int = 100 # Minimum number of foreground pixels for bg removal before whole image is used

    use_objects_as_negatives: bool = True
    use_category_parts_as_negatives: bool = True
    use_imagenet_negatives: bool = True

class TrainDataCollector:
    def __init__(self, embedder: ImageEmbedder, config: TrainDataCollectorConfig):
        self.embedder = embedder
        self.config = config
        self.rembg_session = new_session()

        # Load dataset metadata
        ds_metadata_cache_path = os.path.join(config.cache_dir, 'ds_metadata.pt')
        self.ds_metadata = self._try_load_from_cache(ds_metadata_cache_path)

        if self.ds_metadata is None: # Collect metadata if not cached
            self.ds_metadata = collect_annotations(
                img_dir = os.path.join(config.partonomy_root_dir, 'images'),
                mask_dir = os.path.join(config.partonomy_root_dir, 'masks')
            )

            self._save_to_cache(self.ds_metadata, ds_metadata_cache_path)

    def build_dataset(self, label_name: str) -> tuple[list[torch.Tensor], list[int]]:
        image_embeds = []
        labels = []

        # Collect negative image embeddings from imagenet
        if self.config.use_imagenet_negatives:
            logger.info(f'Collecting negative image embeddings from imagenet for: {label_name}')
            imagenet_embeds = self._collect_imagenet_embeds()
            image_embeds.extend(imagenet_embeds)
            labels.extend([0] * len(imagenet_embeds))

        # TODO reorder these properly again

        # Collect positive image embeddings
        logger.info(f'Collecting positive image embeddings for: {label_name}')
        rle_embeds = self._collect_rle_embeds(label_name)
        image_embeds.extend(rle_embeds)
        labels.extend([1] * len(rle_embeds))

        # Collect negative image embeddings from objects
        if self.config.use_objects_as_negatives:
            if not is_part_name(label_name):
                logger.warning(f'Label is not a part name, skipping object negatives: {label_name}')

            logger.info(f'Collecting negative image embeddings from objects for: {label_name}')
            object_name = get_object_prefix(label_name)
            object_embeds = self._collect_rle_embeds(object_name)
            image_embeds.extend(object_embeds)
            labels.extend([0] * len(object_embeds))

        # Collect negative image embeddings from category parts
        if self.config.use_category_parts_as_negatives:
            if not is_part_name(label_name):
                logger.warning(f'Label is not a part name, skipping category part negatives: {label_name}')

            logger.info(f'Collecting negative image embeddings from category parts for: {label_name}')
            category_name = get_category_name(label_name)
            category_parts = [label for label in self.ds_metadata.part_labels if get_category_name(label) == category_name]
            for part_label in category_parts:
                if get_part_suffix(part_label) == get_part_suffix(label_name):
                    continue

                part_embeds = self._collect_rle_embeds(part_label)
                image_embeds.extend(part_embeds)
                labels.extend([0] * len(part_embeds))

        return image_embeds, labels

    def _collect_rle_embeds(self, label_name: str):
        embeds = []

        rle_paths = self.ds_metadata.rle_paths_by_label[label_name]
        prev_image_path = None
        for rle_path in tqdm(rle_paths, desc=f'RLE Embeds: {label_name}'):
            # Load from cache if available
            cache_path = self._cache_path(rle_path, self.config.partonomy_root_dir, 'partonomy_regions')
            embed = self._try_load_from_cache(cache_path)

            if embed is not None:
                embeds.append(embed)
                continue

            # Compute region embedding
            with open(rle_path, 'r') as f:
                rle_dict: RLEAnnotation = json.load(f)

            image_path = rle_dict['image_path']

            if image_path != prev_image_path: # Use previous image embeds if available
                patch_embeds = self.embedder.get_patch_embeddings([image_path])[0]

            mask = torch.from_numpy(mask_utils.decode(rle_dict)).bool().to(patch_embeds)
            if mask.sum() == 0:
                logger.warning(f'Empty mask, skipping: {rle_path}')
                continue

            region_embed = region_pool(mask.unsqueeze(0), patch_embeds.unsqueeze(0)).squeeze(0).cpu()

            embeds.append(region_embed)
            self._save_to_cache(region_embed, cache_path)
            prev_image_path = image_path

        return embeds

    def _collect_imagenet_embeds(self):
        embeds = []

        image_paths = list_paths(self.config.imagenet_imgs_dir)

        # Change image and cache dirs so that the embedder can cache the imagenet embeddings
        # separately
        if self.embedder.config.cache_dir:
            old_cache_dir = self.embedder.config.cache_dir
            old_image_dir = self.embedder.config.image_dir

            self.embedder.config.cache_dir = old_cache_dir + '_imagenet'
            self.embedder.config.image_dir = self.config.imagenet_imgs_dir

        for image_path in tqdm(image_paths, desc='ImageNet Embeds'):
            # Load from cache if available
            cache_path = self._cache_path(image_path, self.config.imagenet_imgs_dir, 'imagenet_negatives')
            embed = self._try_load_from_cache(cache_path)
            if embed is not None:
                embeds.append(embed)
                continue

            # Compute image embedding
            patch_embeds = self.embedder.get_patch_embeddings([image_path])[0]

            # Compute mask
            image = open_image(image_path)
            mask = np.array(remove(image, session=self.rembg_session, only_mask=True, post_process_mask=True))

            if mask.sum() < self.config.foreground_pixel_min:
                logger.warning(f'Foreground pixel count below threshold, using whole image: {image_path}')
                mask = np.ones_like(mask)

            mask = torch.from_numpy(mask).bool().to(patch_embeds)

            # Region pool
            region_embed = region_pool(mask.unsqueeze(0), patch_embeds.unsqueeze(0)).squeeze(0).cpu()

            # Save
            embeds.append(region_embed)
            self._save_to_cache(region_embed, cache_path)

        # Reset image_dir for caching of partonomy images
        if self.embedder.config.cache_dir:
            self.embedder.config.cache_dir = old_cache_dir
            self.embedder.config.image_dir = old_image_dir

        return embeds

    def _cache_path(self, path: str, base_dir: str, cache_subdir: str = ''):
        rel_path = os.path.relpath(path, base_dir)
        output_dir = self.config.cache_dir if not cache_subdir else os.path.join(self.config.cache_dir, cache_subdir)

        cache_path = os.path.join(output_dir, rel_path)
        cache_path = os.path.splitext(cache_path)[0] + '.pt'

        return cache_path

    def _try_load_from_cache(self, cache_path: str):
        if os.path.exists(cache_path):
            logger.debug(f'Loading image embeds from cache: {cache_path}')
            embed = torch.load(cache_path)
        else:
            embed = None

        return embed

    def _save_to_cache(self, embed: torch.Tensor, cache_path: str):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embed, cache_path)

if __name__ == '__main__':
    import coloredlogs
    from .embed import ImageEmbedderConfig

    coloredlogs.install(level='DEBUG')
    logging.getLogger('PIL').setLevel(logging.WARNING) # Suppress PIL logging

    embedder_config = ImageEmbedderConfig(device='cuda:0', resize_images=True)
    embedder = ImageEmbedder(config=embedder_config)

    data_collector_config = TrainDataCollectorConfig(
        partonomy_root_dir='/shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-11-18/annotations/merged_annotations',
        cache_dir='/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_cleaning_cache',
        use_objects_as_negatives=True,
        use_category_parts_as_negatives=True,
        use_imagenet_negatives=True
    )
    collector = TrainDataCollector(embedder, data_collector_config)

    collector.build_dataset('airplanes--agricultural--part:wings')
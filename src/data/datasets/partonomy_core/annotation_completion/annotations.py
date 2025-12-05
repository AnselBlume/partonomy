# %%
'''
    Adapted from: https://github.com/AnselBlume/ecole_mo9_demo/blob/main/src/kb_ops/build_kb.py
'''
import json
from dataclasses import dataclass
from collections import defaultdict
from typing import TypedDict, Callable
import os
from tqdm import tqdm
import logging, coloredlogs
from data.utils import open_image, list_paths
from data.datasets.build_dataset_descriptors.partonomy import label_from_directory

# Reexported for convenience
from data.part_dataset_descriptor import is_part_name, get_part_suffix, get_object_prefix, get_category_name
from functools import partial
get_part_suffix = partial(get_part_suffix, safe=True)

logger = logging.getLogger(__file__)

class RLEAnnotation(TypedDict):
    size: list[int]
    counts: str
    image_path: str
    is_root_concept: bool

class RLEAnnotationWithMaskPath(RLEAnnotation):
    mask_path: str

@dataclass
class DatasetMetadata:
    img_dir: str
    mask_dir: str

    object_labels: list[str]
    part_labels: list[str]

    img_paths_by_label: dict[str,str]
    rle_paths_by_label: dict[str,str]

    img_paths_to_rle_dicts: dict[str, dict[str, RLEAnnotationWithMaskPath]]

def collect_annotations(
    img_dir: str,
    mask_dir: str,
    label_from_path_fn: Callable[[str],str] = label_from_directory,
    image_exts: list[str] = ['.jpg', '.jpeg', '.webp', '.png'],
    follow_links: bool = True
) -> DatasetMetadata:
    '''
        Constructs a concept knowledge base from images in a directory.

        Arguments:
            img_dir (str): Directory containing images.

        Returns: ConceptKB
    '''
    def validate_rle_dict(rle_dict: dict):
        '''
            Validates an RLE dict by checking for the presence of required keys.
        '''
        keys = sorted(list(rle_dict.keys()))

        # Check for required keys
        if set(keys) != {'counts', 'size', 'image_path', 'is_root_concept'}:
            err_str = f'Invalid RLE dict. Expected keys: {sorted(["counts", "size", "image_path", "is_root_concept"])}. Got: {keys}'
            logger.error(err_str)
            raise ValueError(err_str)

        rle_dict: RLEAnnotation

        # Check that image exists and has the correct dimensions
        try:
            corres_img = open_image(rle_dict['image_path'])
        except FileNotFoundError as e:
            err_str = f'Image not found: {rle_dict["image_path"]}'
            logger.warning(err_str)
            raise ValueError(err_str)

        w, h = corres_img.size
        if rle_dict['size'] != [h, w]:
            err_str = f'Image size mismatch: {rle_dict["size"]} vs. {h, w}'
            logger.warning(err_str)
            raise ValueError(err_str)

    object_labels = set()
    part_labels = set()

    img_paths_by_label = defaultdict(set)
    rle_paths_by_label = defaultdict(list)
    img_paths_to_rle_dicts = defaultdict(dict)

    # Collect mask annotations
    logger.info('Collecting masks...')
    mask_paths = list_paths(mask_dir, exts=['.json'], follow_links=follow_links)
    for mask_path in tqdm(mask_paths):
        with open(mask_path, 'r') as f:
            rle_dict: RLEAnnotation = json.load(f)

        try:
            validate_rle_dict(rle_dict)
        except ValueError as e:
            continue

        label = label_from_path_fn(mask_path)
        if is_part_name(label):
            part_labels.add(label)
        else:
            object_labels.add(label)

        image_path = rle_dict['image_path']
        img_paths_by_label[label].add(image_path)
        rle_paths_by_label[label].append(mask_path)

        rle_dict['mask_path'] = mask_path
        img_paths_to_rle_dicts[image_path][label] = rle_dict

    # Collect image paths
    logger.info('Collecting images...')
    img_paths = list_paths(img_dir, exts=image_exts, follow_links=follow_links)
    for img_path in tqdm(img_paths):
        label = label_from_path_fn(img_path)
        img_paths_by_label[label].add(img_path) # Some images may not have masks

    img_paths_by_label = {k : sorted(img_paths_by_label[k]) for k in sorted(img_paths_by_label)}
    rle_paths_by_label = {k : sorted(rle_paths_by_label[k]) for k in sorted(rle_paths_by_label)}

    return DatasetMetadata(
        img_dir=img_dir,
        mask_dir=mask_dir,
        object_labels=sorted(object_labels),
        part_labels=sorted(part_labels),
        img_paths_by_label=img_paths_by_label,
        rle_paths_by_label=rle_paths_by_label,
        img_paths_to_rle_dicts=img_paths_to_rle_dicts
    )

# %%
if __name__ == '__main__':
    coloredlogs.install(level='WARNING')
    annotations_root = '/shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-11-18/annotations/merged_annotations'
    # annotations_root = '/Users/Ansel/Desktop/24-11-18/annotations/merged_annotations'

    ds_metadata = collect_annotations(
        img_dir = os.path.join(annotations_root, 'images'),
        mask_dir = os.path.join(annotations_root, 'masks')
    )
    breakpoint()
# %%

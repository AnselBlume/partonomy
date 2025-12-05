'''
    Splits qa_pairs file into splits of specified sizes by image.
'''
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))) # Add src directory
import orjson
from typing import Any
import numpy as np
import jsonargparse
from itertools import chain
from qa_generation import QAPair
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def split_qa_pairs(
    qa_pairs: list[QAPair],
    split_props: list[float] = [.8, .2],
    seed: int = 42
) -> list[list[QAPair]]:
    '''
        Splits QAPairs into splits of specified proportions by image.
    '''

    pairs_by_img_path = _group_by_image_path(qa_pairs)
    split_sizes = _get_split_sizes(len(pairs_by_img_path), split_props)
    pair_lists: list[list[QAPair]] = list(pairs_by_img_path.values())

    rng = np.random.default_rng(seed)
    rng.shuffle(pair_lists)

    splits: list[list[list[QAPair]]] = _split(pair_lists, split_sizes)

    return [list(chain.from_iterable(split)) for split in splits]

def split_qa_pairs_stratified(
    qa_pairs: list[QAPair],
    split_props: list[float] = [.8, .2],
    seed: int = 42
) -> list[list[QAPair]]:
    '''
        Approximately splits QAPairs into splits of specified proportions by image class.

        As one image may correspond to multiple classes, we randomly assign one of the image's classes
        to it before performing per-class splitting.
    '''
    pairs_by_img_path = _group_by_image_path(qa_pairs)
    rng = np.random.default_rng(seed)

    img_paths_to_label = {
        img_path : rng.choice(list({pair.image_label for pair in pairs_by_img_path[img_path]}))
        for img_path in pairs_by_img_path
    }
    label_to_img_paths: dict[str, list[str]] = defaultdict(list)
    for img_path, label in img_paths_to_label.items():
        label_to_img_paths[label].append(img_path)

    # Split each label's QAPairs into splits of specified proportions
    splits = [[] for _ in split_props]
    for label, img_paths in label_to_img_paths.items():
        split_sizes = _get_split_sizes(len(img_paths), split_props)
        rng.shuffle(img_paths)
        label_splits: list[list[str]] = _split(img_paths, split_sizes)

        for label_split, target_split in zip(label_splits, splits):
            for img_path in label_split:
                target_split.extend(pairs_by_img_path[img_path])

    return splits

def _group_by_image_path(qa_pairs: list[QAPair]) -> dict[str, list[QAPair]]:
    pairs_by_img = defaultdict(list)
    for qa_pair in qa_pairs:
        pairs_by_img[qa_pair.image_path].append(qa_pair)

    return pairs_by_img

def _split(l: list[Any], split_sizes: list[int]) -> list[list[Any]]:
    inds = [0] + np.cumsum(split_sizes).tolist()
    return [l[inds[i]:inds[i+1]] for i in range(len(inds)-1)]

def _get_split_sizes(l_len: int, split_props: list[float] = [.8, .2]) -> list[int]:
    split_props = np.array(split_props)
    prop_sum = split_props.sum()
    if not np.isclose(prop_sum, 1.0):
        raise ValueError(f'Split proportions must sum to 1.0, got {prop_sum}')

    split_sizes = np.zeros(len(split_props), dtype=int)
    split_sizes[:-1] = (split_props[:-1] * l_len).astype(int)
    split_sizes[-1] = l_len - split_sizes.sum() # Last split gets the remainder

    return split_sizes

def parse_args(cl_args: list[str] = None, config_str: str = None):
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--qa_pairs_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split_props', type=list[float], default=[.8, .2])
    parser.add_argument('--split_names', type=list[str], default=['train', 'val'])
    parser.add_argument('--stratify', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    if config_str is not None:
        args = parser.parse_string(config_str)
    else:
        args = parser.parse_args(cl_args)

    return args

def main(cl_args: list[str] = None, config_str: str = None):
    args = parse_args(cl_args, config_str)

    # Load QAPairs
    logger.info(f'Loading QAPairs from {args.qa_pairs_path}')
    with open(args.qa_pairs_path, 'r') as f:
        qa_pairs = orjson.loads(f.read())

    qa_pairs = [QAPair.from_dict(d) for d in qa_pairs]

    # Split QAPairs
    split_fn = split_qa_pairs_stratified if args.stratify else split_qa_pairs
    logger.info(f'Splitting QAPairs with {"stratified" if args.stratify else "random"} split')
    splits = split_fn(qa_pairs, args.split_props, args.seed)

    # Set split names if not provided
    if args.split_names is None:
        args.split_names = [str(f) for f in args.split_props]
    else:
        assert len(args.split_names) == len(args.split_props)

    # Save splits
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, split in zip(args.split_names, splits):
        basename = os.path.splitext(os.path.basename(args.qa_pairs_path))[0]

        logger.info(f'Saving {len(split)} QAPairs to {os.path.join(args.output_dir, f"{basename}_{split_name}.json")}')
        with open(os.path.join(args.output_dir, f'{basename}_{split_name}.json'), 'wb') as f:
            f.write(orjson.dumps([qa_pair.to_dict() for qa_pair in split]))

if __name__ == '__main__':
    import coloredlogs
    coloredlogs.install(level='DEBUG')

    qa_pair_paths = [
        '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/paco_lvis/paco_lvis_qa_pairs.json',
        '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/pascal_part/pascal_part_qa_pairs.json',
        '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partimagenet/partimagenet_qa_pairs.json',
        # '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partonomy/partonomy_qa_pairs.json',
    ]

    for qa_pair_path in qa_pair_paths:
        main(config_str=f'''
            qa_pairs_path: {qa_pair_path}
            output_dir: {os.path.dirname(qa_pair_path)}
            split_props: [.8, .2]
            split_names: ['train', 'val']
            stratify: True
            seed: 42
        ''')
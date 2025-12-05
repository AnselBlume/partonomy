import os
from pycocotools.coco import COCO
import yaml
import glob
from typing import Literal
from tqdm import tqdm
from collections import defaultdict

BASE_IMAGE_DIR = '/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset'

PART_IMAGE_NET_IMG_DIR = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/dataset/partimagenet/PartImageNet/images'

def init_paco_lvis(
    base_image_dir = BASE_IMAGE_DIR,
    split: Literal['train', 'val', 'test'] = 'train'
):
    annotations_file = os.path.join(base_image_dir, "vlpart", "paco", "annotations", f"paco_lvis_v1_{split}.json")
    img_dir = os.path.join(base_image_dir, 'coco')

    coco_api_paco_lvis = COCO(annotations_file)
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, img_dir, coco_api_paco_lvis


def paco_lvis_to_concept_graph(
    base_image_dir: str = BASE_IMAGE_DIR,
    class_map: dict = None,
    dump_file: bool = False
):
    '''
    Convert category names in 'class_map' into graph.yaml structure
    '''
    out_path = os.path.join(base_image_dir, "vlpart", "paco", "graph.yaml")
    concept_graph = {
        'instance_graph': {},
        'component_graph': {},
        'id2cat_map': {},
        'cat2id_map': {}
        }
    # since there is no 'superordinate' in paco as in partonomy, we use 'object' as
    # a surrogate superordinate category
    superordinate = 'object'
    if superordinate not in concept_graph['instance_graph']:
        concept_graph['instance_graph'][superordinate] = []

    for cat_id, cat_name in class_map.items():
        # 'cat_name' can be passed in two forms : (i) 'object' (ii) (object, part)
        full_cat_name = superordinate + '--' + cat_name[0] if isinstance(cat_name, tuple) else superordinate + '--' + cat_name
        if full_cat_name not in concept_graph['instance_graph'][superordinate]:
            concept_graph['instance_graph'][superordinate].append(full_cat_name)
        if full_cat_name not in concept_graph['component_graph']:
            concept_graph['component_graph'][full_cat_name] = []

        if isinstance(cat_name, tuple) and len(cat_name) > 1:  # e.g., (object, part) - ('guitar', 'neck')
            part_cat_name = '--part:'.join(list(cat_name)).strip().lower()
            concept_graph['component_graph'][full_cat_name].append(part_cat_name)

        cat_lbl = full_cat_name if not isinstance(cat_name, tuple) else part_cat_name

        concept_graph['id2cat_map'][cat_id] = cat_lbl
        concept_graph['cat2id_map'][cat_lbl] = cat_id

    if dump_file:
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(concept_graph, f, default_flow_style=False, sort_keys=False)

    return concept_graph


def init_pascal_part(
    base_image_dir = BASE_IMAGE_DIR,
    split: Literal['train'] = 'train'
):
    annotation_file = os.path.join(base_image_dir, "vlpart", "pascal_part", f"{split}.json")
    img_dir = os.path.join(base_image_dir, 'vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages')

    coco_api_pascal_part = COCO(annotation_file)
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, img_dir, coco_api_pascal_part


def pascal_part_to_concept_graph(
    base_image_dir: str = BASE_IMAGE_DIR,
    class_map: dict = None,
    dump_file: bool = False
):
    '''
    Convert category names in 'class_map' into graph.yaml structure
    '''
    out_path = os.path.join(base_image_dir, "vlpart", "pascal_part", "graph.yaml")
    concept_graph = {
        'instance_graph': {},
        'component_graph': {},
        'id2cat_map': {},
        'cat2id_map': {}
        }
    # since there is no 'superordinate' in pascal_part as in partonomy, we use 'object' as
    # a surrogate superordinate category
    superordinate = 'object'
    if superordinate not in concept_graph['instance_graph']:
        concept_graph['instance_graph'][superordinate] = []

    for cat_id, cat_name in class_map.items():
        # 'cat_name' can be passed in two forms : (i) 'object' (ii) (object, part)
        full_cat_name = superordinate + '--' + cat_name[0] if isinstance(cat_name, tuple) else superordinate + '--' + cat_name
        if full_cat_name not in concept_graph['instance_graph'][superordinate]:
            concept_graph['instance_graph'][superordinate].append(full_cat_name)

        if isinstance(cat_name, tuple) and len(cat_name) > 1:  # e.g., (object, part) - ('guitar', 'neck')
            part_cat_name = '--part:'.join(list(cat_name)).strip().lower()
            if full_cat_name not in concept_graph['component_graph']:
                concept_graph['component_graph'][full_cat_name] = []
            concept_graph['component_graph'][full_cat_name].append(part_cat_name)

        concept_graph['id2cat_map'][cat_id] = full_cat_name
        concept_graph['cat2id_map'][full_cat_name] = cat_id

    if dump_file:
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(concept_graph, f, default_flow_style=False, sort_keys=False)

    return concept_graph


def init_partimagenet(
    base_image_dir = BASE_IMAGE_DIR,
    split: Literal['train', 'train_whole', 'val', 'val_whole', 'test', 'test_whole'] = 'train'
):

    no_suffix_split = split.split('_')[0]
    annotation_file = os.path.join(base_image_dir, "partimagenet", "PartImageNet", "annotations", split, f"{no_suffix_split}.json")
    img_dir = os.path.join(PART_IMAGE_NET_IMG_DIR, split)

    # load the coco-style annotations
    coco_api_partimagenet = COCO(annotation_file)

    """
    In PartImageNet/annotations/train/train.json
    "categories": [
        {
        "id": 0,
        "name": "Quadruped Head",
        "supercategory": "Quadruped"
        },
        {
        "id": 1,
        "name": "Quadruped Body",
        "supercategory": "Quadruped"
        }, ... ]
    """

    all_classes = coco_api_partimagenet.loadCats(coco_api_partimagenet.getCatIds())

    class_map_partimagenet = {}
    for cat in all_classes:
        class_main = cat["supercategory"].strip().lower()
        # PartImageNet part names by default are in the format "Quadruped Head", but we want
        # them to just be "head"
        class_part = cat["name"].lower().removeprefix(class_main).strip()
        name = (class_main, class_part)
        class_map_partimagenet[cat["id"]] = name

    # Extract image IDs
    img_ids = coco_api_partimagenet.getImgIds()

    print(f"PartImageNet: {len(img_ids)} images loaded.")
    return class_map_partimagenet, img_ids, img_dir, coco_api_partimagenet


def partimagenet_to_concept_graph(
    base_image_dir: str = BASE_IMAGE_DIR,
    class_map: dict = None,
    dump_file: bool = False
):
    '''
    Convert category names in 'class_map' into graph.yaml structure
    '''
    out_path = os.path.join(base_image_dir, "partimagenet", "graph.yaml")
    concept_graph = {
        'instance_graph': {},
        'component_graph': {},
        'id2cat_map': {},
        'cat2id_map': {}
        }
    # since there is no 'superordinate' in pascal_part as in partonomy, we use 'object' as
    # a surrogate superordinate category
    superordinate = 'object'
    if superordinate not in concept_graph['instance_graph']:
        concept_graph['instance_graph'][superordinate] = []

    for cat_id, cat_name in class_map.items():
        # 'cat_name' can be passed in two forms : (i) 'object' (ii) (object, part)
        full_cat_name = superordinate + '--' + cat_name[0] if isinstance(cat_name, tuple) else superordinate + '--' + cat_name
        if full_cat_name not in concept_graph['instance_graph'][superordinate]:
            concept_graph['instance_graph'][superordinate].append(full_cat_name)

        if isinstance(cat_name, tuple) and len(cat_name) > 1:  # e.g., (object, part) - ('guitar', 'neck')
            part_cat_name = '--part:'.join(list(cat_name)).strip().lower()
            if full_cat_name not in concept_graph['component_graph']:
                concept_graph['component_graph'][full_cat_name] = []
            concept_graph['component_graph'][full_cat_name].append(part_cat_name)

        concept_graph['id2cat_map'][cat_id] = full_cat_name
        concept_graph['cat2id_map'][full_cat_name] = cat_id

    if dump_file:
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(concept_graph, f, default_flow_style=False, sort_keys=False)

    return concept_graph


def partonomy_to_concept_graph(base_image_dir = BASE_IMAGE_DIR):
    concept_graph = os.path.join(base_image_dir, "partonomy", "graph.yaml")
    if os.path.exists(concept_graph):
        with open(concept_graph, "r") as fp:
            concept_graph = yaml.safe_load(fp)
    else:
        raise FileNotFoundError(f"'concept_graph' has no file: {concept_graph} ")

    return concept_graph
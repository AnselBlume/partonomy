
import os
"""Annotated dataset registry utilities for SegLLM.

This file mirrors ``register_dataset.py`` but adds explanatory comments about
how segmentation annotations are materialized into bitmasks and bounding boxes
at runtime. Understanding these steps is essential for adapting the training
and inference loops to support multiple masks per response.
"""

import json
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.mask import decode, frPyObjects, merge

import yaml
import numpy as np
import cv2
from PIL import Image



def get_mask_from_json(json_path, image_dim):
    """Load polygon annotations exported from labelme-style JSON files.

    Returns:
        np.ndarray: A ``(H, W)`` binary mask where ``1`` denotes the annotated
        region.

    The helper is primarily used for ReasonSeg datasets that store polygon
    vertices instead of COCO RLE. The polygons are sorted by area before being
    rasterized so that larger regions occlude smaller ones as expected.
    """
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    is_sentence = anno["is_sentence"]

    height, width = image_dim

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
            continue
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask


class Register:

    def __init__(self, data_args, is_eval: bool = False) -> None:
        """Parse the segmentation registry configuration.

        Returns:
            None: The constructor populates lookup tables for each dataset.

        The configuration file (``data_args.segmentation_config``) lists all
        datasets and their annotation files. During inference we typically pass
        ``is_eval=True`` so that evaluation-only annotation files are also
        registered. This registry is later used inside ``make_supervised_data_module``
        to fetch ground-truth masks or previously predicted masks that should be
        injected as ``[MASK-ENCODE]`` placeholders.
        """
        with open(data_args.segmentation_config) as f:
            config = yaml.safe_load(f.read())
            # print("Segmentation Config File")
            # print(config)
        self.config = config
        self.data = {}
        self.json_cache = {}
        self.data_args = data_args
        self.ds_with_eval = set()
        for dataset in self.config['datasets']:
            ds_name = dataset['name']
            ds_path = dataset['path']
            # if is_eval and 'path_eval' in dataset:
            #     ds_path = dataset['path_eval']
            ''' 
            Previously, train and eval has a separate segmentation register, 
            but now eval gt mask register is combined with train gt mask register.
            (e.g. Refcoco: both train, eval gt masks in same annotation file,
                  PACO:    have separate files for train, eval gt masks)
            '''
            self.add_dataset(ds_name, ds_path)      # register default (train seg anno file)
            if 'path_eval' in dataset:              # if ds has another anno file for eval, register that as well
                ds_name_eval = f'{ds_name}_eval'
                ds_path_eval = dataset['path_eval']
                self.add_dataset(ds_name_eval, ds_path_eval)
                self.ds_with_eval.add(ds_name)      # remember which ds has a separate eval anno file

    def add_dataset(self, ds_name, ds_path):
        """Register a dataset and eagerly load annotation metadata.

        Returns:
            None: Updates ``self.data`` with dataset-specific loaders.

        Each dataset can store annotations in slightly different formats
        (COCO JSON, raw bitmasks, per-image JSON, etc.), so we normalize them
        here into objects that expose the same downstream API.
        """
        print(f"------------------- Loading {ds_path} annotations ------------------")
        ds_path = os.path.join(self.data_args.annotation_folder, ds_path)
        if ds_name in ["reason_seg", "reason_seg_eval", "ade20k", "cocostuff"]:
            self.data[ds_name] = ds_path
        elif ds_name in ['visual_genome',"description_based_coco","pascal"]:
            self.data[ds_name] = self.load_json(ds_path)
        else:
            self.data[ds_name] = COCO(ds_path)

    def load_json(self, json_path):
        """Read annotation JSON with caching.

        Returns:
            dict: The parsed JSON structure describing masks.

        Some datasets are stored as massive JSON files (e.g. Visual Genome
        region descriptions). Caching avoids repeatedly parsing them when the
        data loader iterates over the dataset.
        """
        if json_path in self.json_cache:
            return self.json_cache[json_path]
        else:
            try:
                with open(json_path, "r") as r:
                    data = json.load(r)
            except UnicodeDecodeError:
                with open(json_path, "r", encoding="cp1252") as r:
                    data = json.load(r)
            self.json_cache[json_path] = data
            return data
    
    def get_bitmask(
        self,
        dataset,
        idx,
        is_eval=False,
        image_file=None,
        image_dim=None
    ):
        """Return a binary mask for the requested instance.

        Returns:
            np.ndarray: A ``(H, W, 1)`` array with ``{0, 1}`` labels indicating
            the target region.

        Args:
            dataset: Dataset name from the configuration file.
            idx: Dataset-specific mask identifier (object id or class id).
            is_eval: Whether evaluation annotations should be used when
                available.
            image_file: File name of the image associated with the mask.
            image_dim: Image height/width, required for polygon-based datasets.

        The branching logic mirrors the heterogeneous annotation formats used
        across SegLLM training corpora. When adapting the model to output
        multiple masks per response, remember that each mask can independently
        request reference masks from *any* dataset handled here, so the feature
        passing pipeline must be compatible with all formats.
        """
        if is_eval and (dataset in self.ds_with_eval):
            dataset = f'{dataset}_eval'

        if dataset == "reason_seg" or dataset == "reason_seg_eval":
            json_dir = self.data[dataset]
            json_file = image_file.replace(".jpg", ".json")
            json_path = os.path.join(json_dir, json_file)
            mask = get_mask_from_json(json_path, image_dim)
            mask = mask.reshape(*mask.shape, 1) # H, W, 1

        elif dataset == "ade20k":
            anns_dir = self.data["ade20k"]
            anns_file = image_file.replace(".jpg", ".png")
            anns_path = os.path.join(anns_dir, anns_file)
            anns_img = np.array(Image.open(anns_path))     # anns_img[x][y] = class_id

            anns_img[anns_img == 0] = 255
            anns_img -= 1
            anns_img[anns_img == 254] = 255

            class_id = idx                                 # for ade20k (semantic), mask_id will be class_id
            binary_mask = (anns_img == class_id).astype(np.uint8)

            mask = binary_mask.reshape(*binary_mask.shape, 1)
        elif dataset == "cocostuff":
            anns_dir = self.data["cocostuff"]
            anns_file = image_file.replace(".jpg", ".png")
            anns_path = os.path.join(anns_dir, anns_file)
            anns_img = np.array(Image.open(anns_path))    
            class_id = idx                                 
            binary_mask = (anns_img == class_id).astype(np.uint8)
            mask = binary_mask.reshape(*binary_mask.shape, 1)
        # COCO format
        elif dataset == "visual_genome":  
            anno = self.data['visual_genome']   
            mask = decode(anno[idx])            # idx = mask_id
            mask = mask.reshape(*mask.shape, 1)
        elif dataset == "pascal":  
            anno = self.data['pascal']   
            mask = decode(anno[idx])            # idx = mask_id
            mask = mask.reshape(*mask.shape, 1)
        elif dataset == "description_based_coco":
            anno = self.data["description_based_coco"][idx]
            seg=anno["segmentation"]
            if type(seg) == list:
                rles=frPyObjects(seg,anno["image_dim"][0],anno["image_dim"][1])
                rle=merge(rles)
            elif type(seg['counts']) == list:
                rle = frPyObjects(seg,anno["image_dim"][0],anno["image_dim"][1])
            else:
                rle=seg
            mask=decode(rle)
            mask = mask.reshape(*mask.shape,1)
        else:
            coco = self.data[dataset]
            ann = coco.loadAnns(ids=[idx])
            mask = coco.annToMask(ann[0])
            mask = mask.reshape(*mask.shape,1) # H W 1
        return mask
    
    def get_bbox(
        self,
        dataset,
        idx,
        is_eval=False,
        image_file=None,
        image_dim=None,
        mask=None
    ):
        """Compute a tight bounding box for the mask requested by ``idx``.

        Returns:
            list[int]: ``[row0, col0, row1, col1]`` style bounding box
            coordinates in image pixel space.

        Bounding boxes are derived lazily because some datasets only provide
        masks. The bbox encodings are passed alongside mask features whenever
        ``[BOX-ENCODE]`` tokens appear in the prompt, which is especially useful
        when cascading multiple mask predictions in a single response.
        """
        if mask is None:                                                # for debugging, when want to pass a specific dummy mask (normally, mask is None)
            mask = self.get_bitmask(
                dataset,
                idx,
                is_eval=is_eval,
                image_file=image_file,
                image_dim=image_dim
            )
            mask = mask[:,:,0] > 0 # H, W
        h, w = mask.shape[:2]

        x = mask.any(1).nonzero()[0]
        y = mask.any(0).nonzero()[0]
        box = [x[0], y[0], x[-1] + 1, y[-1] + 1]  # x0 y0 x1 y1

        return box
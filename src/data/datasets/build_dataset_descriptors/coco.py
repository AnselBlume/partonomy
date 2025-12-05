from PIL import Image
from PIL.Image import Image as PILImage
from data.part_dataset_descriptor import PartDatasetDescriptor, PartDatasetInstance
from collections import defaultdict
from typing import Literal
from tqdm import tqdm
import os
import shutil
from pycocotools.coco import COCO
from pprint import pformat
from .match_objects_and_parts import ObjectPartMatcher
import torch
from torchvision.ops import masks_to_boxes, box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import logging

logger = logging.getLogger(__name__)

def generate_coco_part_dataset_descriptor(
    dataset_name: str,
    img_ids: list,
    img_dir: str,
    coco_api: COCO,
    matcher: ObjectPartMatcher,
    modified_image_output_dir: str = None,
    copy_method: Literal['only_modified', 'copy', 'link'] = 'only_modified'
) -> PartDatasetDescriptor:
    '''
        Generates a PartDatasetDescriptor from a COCO dataset.

        modified_image_output_dir: If provided, the images will be copied to this directory.
        copy_method: If modified_image_output_dir is provided, this argument controls how the images are copied.
            - 'only_modified': Only the modified images are saved to the output directory.
            - 'copy': All images (with their modified replacements) are saved to the output directory.
            - 'link': All images (with their modified replacements) are linked to the output directory.
    '''

    assert dataset_name in ['paco_lvis', 'pascal_part', 'partimagenet']
    has_warned_about_no_modified_images = False

    part_dataset_descriptor = PartDatasetDescriptor()
    part_dataset_descriptor.dataset_name = dataset_name

    n_failures = 0
    for img_id in tqdm(img_ids, desc=f'Generating PartDatasetDescriptor for {dataset_name}'):
        # Compute image path
        image_info = coco_api.loadImgs([img_id])[0]
        orig_img_path = os.path.join(img_dir, image_info['file_name'])
        output_img_path = orig_img_path

        # Compute pairing between objects and parts
        annIds = coco_api.getAnnIds(imgIds=image_info['id'])  # contains the annotation Ids within the image
        anns = coco_api.loadAnns(annIds)
        matches = matcher.match_objects_and_parts(anns)

        # Organize annotations by class
        class_to_matches = defaultdict(list)
        for match in matches:
            class_to_matches[matcher.get_object_class(match)].append(match)

        # For each class, sample annotation with the most parts (or this being equal, the one largest one)
        sampled_matches = []
        for cls, matches in class_to_matches.items():
            # Sort by number of parts, then by area
            matches.sort(
                key=lambda m: (len(m.part_annotations), matcher.get_object_area(m)),
                reverse=True
            )

            # Sample the first annotation
            sampled_match = matches[0]
            sampled_matches.append(sampled_match)

        modified_img_idx = 1
        for match in sampled_matches: # There will be one match per class
            segmentations = defaultdict(list)  # Maps labels to lists of segmentations
            obj_class = matcher.get_object_class(match)

            # Generate segmentation for object
            object_label = f'object--{obj_class}' # Used for image label even if segmentation doesn't exist
            try:
                if match.object_annotation is not None:
                    segmentations[object_label].append(_ann_to_rle_dict(match.object_annotation, coco_api))

                # Generate segmentation for parts
                for part_annot in match.part_annotations:
                    obj_name, part_name = matcher.get_ann_class(part_annot)
                    assert obj_name == obj_class

                    part_label = f'{obj_name}--part:{part_name}'
                    segmentations[part_label].append(_ann_to_rle_dict(part_annot, coco_api))
            except Exception as e:
                logger.debug(
                    f'Caught error for match {pformat(match)}: {e}\n\n'
                    f'This is likely because the segmentation is interpreted as a bounding box instead of a polygon. See:\n'
                    f'https://github.com/cocodataset/cocoapi/issues/139\n'
                    f'Skipping this match.'
                )
                n_failures += 1
                continue

            # Annotate image with bounding box if there is more than one object annotation for the class
            if modified_image_output_dir is not None:
                modified_img_path = os.path.join(modified_image_output_dir, os.path.relpath(orig_img_path, img_dir))
                os.makedirs(os.path.dirname(modified_img_path), exist_ok=True)

                n_obj_annots_for_class = len(class_to_matches[obj_class])
                if n_obj_annots_for_class > 1: # Need to specify which object in the image to use with bounding box
                    # Generate image with bounding box for selected object annotation
                    assert match.object_annotation is not None
                    annotated_image = _annotate_bounding_box(orig_img_path, match.object_annotation, coco_api)

                    base, ext = os.path.splitext(modified_img_path)
                    modified_img_path = f'{base}-{modified_img_idx}{ext}' # Annotate match index in filename as there may be multiple classes per image
                    annotated_image.save(modified_img_path)

                    modified_img_idx += 1
                    output_img_path = modified_img_path

                else: # Only one object annotation for class; no modification needed
                    output_img_path = _copy_image(orig_img_path, modified_img_path, copy_method)
            else:
                if not has_warned_about_no_modified_images:
                    logger.warning(f'Warning: No modified images will be saved because modified_image_output_dir is not provided.')
                    has_warned_about_no_modified_images = True

            part_dataset_instance = PartDatasetInstance(
                image_path=output_img_path,
                image_label=object_label,
                segmentations=segmentations
            )
            part_dataset_descriptor.instances.append(part_dataset_instance)

    logger.info(f'{n_failures} matches skipped compared to {len(part_dataset_descriptor.instances)} matches output due to errors in the segmentation format.')

    return part_dataset_descriptor

def _annotate_bounding_box(image_path: str, ann: dict, coco_api: COCO, relax_width: bool = 5) -> PILImage:
    # Load image
    image = Image.open(image_path)
    image_t = pil_to_tensor(image)

    # mask = coco_api.annToMask(ann)
    # bboxes = masks_to_boxes(torch.from_numpy(mask)[None, ...]) # [1, 4]
    bbox = ann['bbox'] # [x, y, width, height]
    bboxes = box_convert(torch.tensor([bbox]), in_fmt='xywh', out_fmt='xyxy') # (1, 4)

    if relax_width:
        bboxes[:, :2] = (bboxes[:, :2] - relax_width).clamp(min=0)
        bboxes[:, 2] = (bboxes[:, 2] + relax_width).clamp(max=image.width)
        bboxes[:, 3] = (bboxes[:, 3] + relax_width).clamp(max=image.height)

    annotated_image = draw_bounding_boxes(image_t, bboxes, colors='red', width=3)

    return to_pil_image(annotated_image)

def _copy_image(image_path: str, output_path: str, copy_method: Literal['only_modified', 'copy', 'link']):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if copy_method == 'only_modified':
        return image_path
    elif copy_method == 'copy':
        shutil.copy(image_path, output_path)
    elif copy_method == 'link':
        os.symlink(image_path, output_path)
    else:
        raise ValueError(f'Invalid copy method: {copy_method}')

    return output_path

def _ann_to_rle_dict(ann: dict, coco_api: COCO) -> dict:
    rle_dict = coco_api.annToRLE(ann) # Use the original function
    # rle_dict = annToRLE(ann, coco_api) # Use the modified function
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8') # Convert to string for JSON serialization

    return rle_dict

# '''
#     NOTE The following code is a modified version of the `annToRLE` function in the `pycocotools.coco.COCO` class.
#     We modify it to handle the case where the segmentation is a polygon of four points, where the original function
#     treats it as a bounding box, throwing an error.
# '''

# from pycocotools import mask as maskUtils
# from pycocotools._mask import frBbox, frPoly, frUncompressedRLE
# import numpy as np

# def annToRLE(ann: dict, coco_api: COCO):
#     """
#     Modified version of the `annToRLE` function in the `pycocotools.coco.COCO` class
#     in order to handle the case where the segmentation is a polygon of four points.
#     """
#     t = coco_api.imgs[ann['image_id']]
#     h, w = t['height'], t['width']
#     segm = ann['segmentation']
#     if type(segm) == list:
#         # polygon -- a single object might consist of multiple parts
#         # we merge all parts into one mask rle code
#         # XXX We modify this to handle the case where the segmentation is a polygon of four points
#         # rles = maskUtils.frPyObjects(segm, h, w)
#         rles = _frPyObjectsOverride(segm, h, w)

#         rle = maskUtils.merge(rles)
#     elif type(segm['counts']) == list:
#         # uncompressed RLE
#         rle = maskUtils.frPyObjects(segm, h, w)
#     else:
#         # rle
#         rle = ann['segmentation']
#     return rle

# def _frPyObjectsOverride(pyobj, h, w):
#     # encode rle from a list of python objects
#     if type(pyobj) == np.ndarray:
#         objs = frBbox(pyobj, h, w)
#     # elif type(pyobj) == list and len(pyobj[0]) == 4:
#     #     objs = frBbox(pyobj, h, w)
#     # elif type(pyobj) == list and len(pyobj[0]) >= 4:
#     #     objs = frPoly(pyobj, h, w)
#     elif type(pyobj) == list and type(pyobj[0]) != dict:
#         objs = frPoly(pyobj, h, w)
#     elif type(pyobj) == list and type(pyobj[0]) == dict \
#         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
#         objs = frUncompressedRLE(pyobj, h, w)
#     # encode rle from single python object
#     # elif type(pyobj) == list and len(pyobj) == 4:
#     #     objs = frBbox([pyobj], h, w)[0]
#     # elif type(pyobj) == list and len(pyobj) > 4:
#     #     objs = frPoly([pyobj], h, w)[0]
#     elif type(pyobj) == list:
#         objs = frPoly([pyobj], h, w)[0]
#     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
#         objs = frUncompressedRLE([pyobj], h, w)[0]
#     else:
#         raise Exception('input type is not supported.')
#     return objs


# def _frPyObjectsOrig(pyobj, h, w):
#     # See https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/_mask.pyx#L292
#     # encode rle from a list of python objects
#     if type(pyobj) == np.ndarray:
#         objs = frBbox(pyobj, h, w)
#     elif type(pyobj) == list and len(pyobj[0]) == 4:
#         objs = frBbox(pyobj, h, w)
#     elif type(pyobj) == list and len(pyobj[0]) > 4:
#         objs = frPoly(pyobj, h, w)
#     elif type(pyobj) == list and type(pyobj[0]) == dict \
#         and 'counts' in pyobj[0] and 'size' in pyobj[0]:
#         objs = frUncompressedRLE(pyobj, h, w)
#     # encode rle from single python object
#     elif type(pyobj) == list and len(pyobj) == 4:
#         objs = frBbox([pyobj], h, w)[0]
#     elif type(pyobj) == list and len(pyobj) > 4:
#         objs = frPoly([pyobj], h, w)[0]
#     elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
#         objs = frUncompressedRLE([pyobj], h, w)[0]
#     else:
#         raise Exception('input type is not supported.')
#     return objs
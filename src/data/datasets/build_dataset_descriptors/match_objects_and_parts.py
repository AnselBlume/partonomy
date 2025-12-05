# %%
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from einops import einsum, reduce
from pycocotools.coco import COCO
import logging

logger = logging.getLogger(__name__)

@dataclass
class ObjectPartMatch:
    object_annotation: dict = None
    part_annotations: list[dict] = field(default_factory=list)
    match_scores: list[float] = field(default_factory=list)
    is_default_match: bool = False

@dataclass
class ObjectPartMatcherConfig:
    object_part_match_threshold: float = field(
        default=0.1,
        metadata={
            'help': 'Minimum overlap between an object and a part to consider them a match.'
                    ' If 0, we will match all parts to the object with the greatest overlap, regardless of overlap value.'
        }
    )

class ObjectPartMatcher:
    def __init__(
        self,
        coco_api: COCO = None,
        class_map: dict[int, tuple[str, str] | str] = None,
        config: ObjectPartMatcherConfig = ObjectPartMatcherConfig()
    ):

        self.coco_api = coco_api
        self.class_map = class_map
        self.config = config

    def _remove_object_annotations_with_no_parts(anns: list, class_map: dict):
        '''
        Filters out object annotations that do not have any parts from a COCO annotation list

        class_map is a dictionary that maps category ids to (object, part) tuple[str, str] or an object string.
        '''
        filtered_anns = []
        classes_with_parts = set()
        for ann in anns:
            ann_class = class_map[ann['category_id']]
            if isinstance(ann_class, tuple): # This is (object, part)
                classes_with_parts.add(ann_class)
            else:
                filtered_anns.append(ann)

        # Remove object annotations that are not in classes_with_parts
        filtered_anns = [ann for ann in filtered_anns if ann['category_id'] in classes_with_parts]

        return filtered_anns, classes_with_parts

    def match_objects_and_parts(self, img_annotations: list[dict]) -> list[ObjectPartMatch]:
        '''
        Matches object annotations to part annotations in an image.

        img_annotations is a list of COCO annotations for a single image, obtained by:
            annIds = coco_api.getAnnIds(imgIds=img_id)
            img_annotations = coco_api.loadAnns(annIds)
        '''
        object_annotations = defaultdict(list)
        object_class_to_part_annotations = defaultdict(list)

        for ann in img_annotations:
            ann_class: tuple[str, str] | str = self.get_ann_class(ann)

            if self.is_object_annotation(ann_class):
                object_class = ann_class
                object_annotations[object_class].append(ann)
            else:
                object_class, part_class = ann_class
                object_class_to_part_annotations[object_class].append(ann)

        matches = []
        for object_class, part_annotations in object_class_to_part_annotations.items():
            if object_class not in object_annotations:
                logger.debug(f'Object class {object_class} not found in object_annotations; constructing full image match')
                matches.append(ObjectPartMatch(
                    object_annotation=None,
                    part_annotations=part_annotations,
                    match_scores=[1.] * len(part_annotations),
                    is_default_match=True
                ))
                continue

            elif len(object_annotations[object_class]) == 1:
                logger.debug(f'Object class {object_class} has only one annotation; matching all parts to this object')
                matches.append(ObjectPartMatch(
                    object_annotation=object_annotations[object_class][0],
                    part_annotations=part_annotations,
                    match_scores=[1.] * len(part_annotations),
                    is_default_match=True
                ))
                continue

            # Object class has multiple annotations in this image; match each part to the object with the greatest overlap
            part_masks = np.stack([self.coco_api.annToMask(ann) for ann in part_annotations]) # (num_parts, h, w)

            object_annotations = object_annotations[object_class]
            object_masks = np.stack([self.coco_api.annToMask(ann) for ann in object_annotations]) # (n_objects, h, w)

            # Assign each part annotation to the object annotation with the highest IoU
            scores = einsum(object_masks, part_masks, 'n_objects h w, n_parts h w -> n_objects n_parts')
            pixels_per_part = reduce(part_masks, 'n_parts h w -> n_parts', reduction='sum')
            scores = scores / pixels_per_part

            # Find the object annotation with the highest score for each part annotation
            best_obj_annotation_per_part = np.argmax(scores, axis=0) # (n_parts,)
            best_obj_annotation_per_part_scores = np.max(scores, axis=0) # (n_parts,)

            # Create ObjectPartMatch objects for all reasonable matches
            obj_ind_to_match = defaultdict(ObjectPartMatch)
            for part_ind, (matched_obj_ind, matched_obj_score) in enumerate(zip(best_obj_annotation_per_part, best_obj_annotation_per_part_scores)):
                object_class, part_class = self.get_ann_class(part_annotations[part_ind])
                object_class = self.get_ann_class(object_annotations[matched_obj_ind])

                if matched_obj_score < self.config.object_part_match_threshold:
                    part_class = self.class_map[part_annotations[part_ind]['category_id']]
                    object_class = self.class_map[object_annotations[matched_obj_ind]['category_id']]
                    logger.debug(f'Part {part_class} has overlap {matched_obj_score} with {object_class} < {self.config.object_part_match_threshold}; skipping')
                    continue

                match = obj_ind_to_match[matched_obj_ind]
                if match.object_annotation is None:
                    match.object_annotation = object_annotations[matched_obj_ind]

                match.part_annotations.append(part_annotations[part_ind])
                match.match_scores.append(matched_obj_score)

            matches.extend(obj_ind_to_match.values())

        return matches

    def get_ann_class(self, ann: dict) -> str | tuple[str, str]:
        return self.class_map[ann['category_id']]

    def get_object_class(self, match: ObjectPartMatch) -> str:
        if match.object_annotation is None:
            return self.get_ann_class(match.part_annotations[0])[0]
        else:
            return self.get_ann_class(match.object_annotation)

    @staticmethod
    def get_object_area(match: ObjectPartMatch) -> float:
        if match.object_annotation is None:
            return 0.
        else:
            return match.object_annotation['area']

    @staticmethod
    def is_object_annotation(ann_class: str | tuple[str, str]) -> bool:
        return isinstance(ann_class, str)

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..')))
    import numpy as np
    from data.datasets import init_paco_lvis, init_pascal_part, init_partimagenet
    from data.utils import mask_vis as vis
    from data.utils import open_image
    import coloredlogs

    coloredlogs.install(level='DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    dataset_name = 'paco_lvis'
    # output_dir = f'{dataset_name}-small_part_examples' # None to not save images
    output_dir = None
    random_seed = 40
    n_examples = 50

    BASE_DATASET_DIR = '/shared/nas/data/m1/jk100/code/OpenAttrLibrary/LISA/dataset'

    if dataset_name == 'paco_lvis':
        init_fn = init_paco_lvis
    elif dataset_name == 'pascal_part':
        init_fn = init_pascal_part
    elif dataset_name == 'partimagenet':
        init_fn = init_partimagenet

    class_map, img_ids, img_dir, coco_api = init_fn(BASE_DATASET_DIR)
    object_part_matcher = ObjectPartMatcher(class_map=class_map, coco_api=coco_api)

    np.random.default_rng(random_seed).shuffle(img_ids)
    image_from_masks = lambda mask, image: vis.image_from_masks(
        mask[None, ...],
        combine_as_binary_mask=True,
        combine_color='aqua',
        superimpose_on_image=image,
        superimpose_alpha=.8
    )

    for img_id in img_ids[:n_examples]:
        annIds = coco_api.getAnnIds(imgIds=img_id)
        anns = coco_api.loadAnns(annIds)
        matches = object_part_matcher.match_objects_and_parts(anns)

        # Draw the annotations on the image
        image_info = coco_api.loadImgs([img_id])[0]
        image_path = os.path.join(img_dir, image_info['file_name'])
        image = open_image(image_path)

        # Show the object annotation (if it exists), then each part annotation independently
        object_class_counts = defaultdict(int)
        for match in matches:
            if match.object_annotation is not None:
                mask = coco_api.annToMask(match.object_annotation).astype(bool)
                image_with_masks = image_from_masks(mask, image)
            else:
                logger.debug(f'Object annotation not found for match {match}; showing full mask')
                mask = np.full((image.height, image.width), True)
                image_with_masks = image_from_masks(mask, image)

            # Show the object annotation
            object_class = object_part_matcher.get_ann_class(match.object_annotation) if match.object_annotation is not None else 'object'
            fig, ax = vis.show(image_with_masks, title=f'Object annotation: {object_class}')
            fig.show()

            object_class_counts[object_class] += 1
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, f'{img_id}_object-{object_class}-{object_class_counts[object_class]}.png'))

            # Now, just show the part annotations
            part_class_counts = defaultdict(int)
            for i, part_annotation in enumerate(match.part_annotations, start=1):
                mask = coco_api.annToMask(part_annotation).astype(bool)
                image_with_masks = image_from_masks(mask, image)
                _, part_class = object_part_matcher.get_ann_class(part_annotation)
                fig, ax = vis.show(image_with_masks, title=f'Part annotation: {part_class}')
                fig.show()

                part_class_counts[part_class] += 1
                if output_dir is not None:
                    fig.savefig(os.path.join(output_dir, f'{img_id}_part-{object_class}-{object_class_counts[object_class]}-{part_class}-{part_class_counts[part_class]}.png'))

# %%

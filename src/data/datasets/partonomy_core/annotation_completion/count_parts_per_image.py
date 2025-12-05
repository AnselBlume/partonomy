# %%
import sys
sys.path.append('/shared/nas2/blume5/sp25/partonomy/partonomy_private/src')
from dataclasses import dataclass
from data.utils import open_image
from tqdm import tqdm
import json
import os
from collections import defaultdict
from data.partonomy_core.annotation_completion.annotations import collect_annotations, RLEAnnotation

@dataclass
class ImagePartCount:
    image_path: str
    part_label: str
    part_count: int

if __name__ == '__main__':
    partonomy_root_dir = '/shared/nas2/blume5/fa24/concept_downloading/data/image_annotations/24-11-18/annotations/merged_annotations'
    image_dir = os.path.join(partonomy_root_dir, 'images')
    mask_dir = os.path.join(partonomy_root_dir, 'masks')

    metadata = collect_annotations(image_dir, mask_dir)

    part_label_to_counts = defaultdict(list) # Mapping from part label to part count for each image

    for part_label in tqdm(metadata.part_labels):
        part_count_per_image = defaultdict(int)
        part_rle_paths = metadata.rle_paths_by_label[part_label]

        for part_rle_path in part_rle_paths:
            with open(part_rle_path, 'r') as f:
                rle_dict: RLEAnnotation = json.load(f)

            image_path = rle_dict['image_path']
            part_count_per_image[image_path] += 1

        for image_path, part_count in part_count_per_image.items():
            part_label_to_counts[part_label].append(ImagePartCount(image_path, part_label, part_count))

    # %% Number of parts with more than one part annotation in at least one image
    part_labels_with_multiple_annots = [
        part_label for part_label, image_part_counts in part_label_to_counts.items()
        if any([image_part_count.part_count > 1 for image_part_count in image_part_counts])
    ]

    print(f'Number of parts with more than one part annotation in at least one image: {len(part_labels_with_multiple_annots)}')

    # %% Get images of example parts with multiple part annotations
    example_part_label = part_labels_with_multiple_annots[0]
    print(f'Example part label with multiple part annotations: {example_part_label}')

    example_image_part_counts = part_label_to_counts[example_part_label]

    example_image_paths = [image_part_count.image_path for image_part_count in example_image_part_counts if image_part_count.part_count > 1]
    image_path_index = 0
    image_path = example_image_paths[image_path_index]

    print(f'Example image path: {image_path}')
    open_image(image_path)

    # %% Out of these, what is the distribution of the number of part annotations per image?
    for part_label in part_labels_with_multiple_annots:
        image_part_counts = part_label_to_counts[part_label]

        part_count_to_image_count = defaultdict(int)
        for image_part_count in image_part_counts:
            part_count_to_image_count[image_part_count.part_count] += 1

        part_count_to_image_count = {pc : ic for pc, ic in sorted(part_count_to_image_count.items(), key=lambda x: x[0])}

        print(f'Part label: {part_label}')
        print(f'Part count to image count: {part_count_to_image_count}')
        print()
# %%

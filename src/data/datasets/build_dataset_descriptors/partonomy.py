from data.part_dataset_descriptor import PartDatasetDescriptor, PartDatasetInstance, get_object_prefix
from collections import defaultdict
from tqdm import tqdm
import os
import glob
from data.utils import load_json

def generate_partonomy_dataset_descriptor(base_dataset_dir: str, dataset_name: str, concept_graph: dict) -> PartDatasetDescriptor:
    '''
    Generate a PartDatasetDescriptor for the Partonomy dataset.

    Args:
        base_dataset_dir: The base directory of the Partonomy dataset.
        dataset_name: The name of the dataset.
        concept_graph: The concept graph of the Partonomy dataset, returned by partonomy_to_concept_graph
    '''

    masks_dir = os.path.join(base_dataset_dir, 'partonomy', 'masks')

    # Directories with concept or part names
    concept_masks_dirs = sorted(os.listdir(masks_dir)) # e.g. ['airplanes--agricultural', 'airplanes--agricultural--part:wing', ...]

    path_to_instance = defaultdict(PartDatasetInstance)
    for concept_mask_dir in tqdm(concept_masks_dirs, desc=f"Constructing PartDatasetDescriptor for {dataset_name}"):
        concept_mask_dir_path = os.path.join(masks_dir, concept_mask_dir)
        concept_name = label_from_directory(concept_mask_dir_path) # Can be an object or part

        # Get all mask annotations for the concept or part
        json_paths = glob.glob(os.path.join(concept_mask_dir_path, '*.json'))

        for json_path in json_paths:
            data = load_json(json_path)

            image_path = data['image_path']
            if image_path not in path_to_instance:
                path_to_instance[image_path] = PartDatasetInstance(
                    image_path=image_path,
                    image_label=get_object_prefix(concept_mask_dir), # Images are labeled with the object name
                    segmentations=defaultdict(list)
                )

            path_to_instance[image_path].segmentations[concept_name].append(data)

    part_dataset_descriptor = PartDatasetDescriptor(
        dataset_name=dataset_name,
        instances=list(path_to_instance.values()),
        part_graph=concept_graph['component_graph'],
        instance_graph=concept_graph['instance_graph']
    )

    return part_dataset_descriptor

def label_from_directory(path: str):
    if not os.path.isdir(path):
        path = os.path.dirname(path)

    return os.path.basename(path).lower()
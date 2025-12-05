from __future__ import annotations
from typing import Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class PartDatasetInstance:
    image_path: str = ''
    image_label: str = ''
    segmentations: dict[str, list[Any]] = field(default_factory=dict) # Mapping from labels to list of segmentations

    @property
    def segmentation_labels(self) -> list[str]:
        # For backwards compatibility
        return sorted(self.segmentations)

    def to_dict(self) -> dict[str, Any]:
        return {
            'image_path': self.image_path,
            'image_label': self.image_label,
            'segmentations': self.segmentations,
            'segmentation_labels': self.segmentation_labels,
        }

    def __hash__(self):
        return hash((self.image_path, self.image_label))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PartDatasetInstance:
        return PartDatasetInstance(
            image_path=d['image_path'],
            image_label=d['image_label'],
            segmentations=d['segmentations']
        )

@dataclass
class PartDatasetDescriptor:
    dataset_name: str = ''
    instances: list[PartDatasetInstance] = field(default_factory=list)
    part_graph: dict[str, list[str]] = field(default_factory=dict) # Adjacency list of objects --> their parts (possibly empty list)
    instance_graph: dict[str, list[str]] = field(default_factory=dict) # Adjacency list of objects --> their subobjects (possibly empty list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'dataset_name': self.dataset_name,
            'instances': [inst.to_dict() for inst in self.instances],
            'part_graph': self.part_graph,
            'instance_graph': self.instance_graph,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PartDatasetDescriptor:
        return PartDatasetDescriptor(
            dataset_name=d['dataset_name'],
            instances=[PartDatasetInstance.from_dict(inst) for inst in d['instances']],
            part_graph=d['part_graph'],
            instance_graph=d['instance_graph'],
        )

###################
# Utility Methods #
###################

PART_SEP = '--part:'

def clean_dataset_descriptor(
    dataset_descriptor: PartDatasetDescriptor,
    remove_non_part_segmentations: bool = True,
    object_name_map: Callable[[str], str] = None,
    part_name_map: Callable[[str], str] = None,
    strip_concept_specific_part_prefix: bool = True,
    strip_object_literal_prefix: bool = True,
    deduce_part_graph: bool = False
) -> PartDatasetDescriptor:

    # Remove non-part segmentations, then filtering out instances with no parts
    if remove_non_part_segmentations:
        logger.info(f'Removing non-part segmentations from dataset instances...')

        filtered_instances = []
        for instance in dataset_descriptor.instances:
            instance.segmentations = {label : segmentation for label, segmentation in instance.segmentations.items() if is_part_name(label)}
            if len(instance.segmentations) > 0:
                filtered_instances.append(instance)

        dataset_descriptor.instances = filtered_instances

    # Deduce part graph from instances, replacing the existing part graph
    if deduce_part_graph:
        logger.info(f'Deducing part graph from dataset instances...')
        part_graph = defaultdict(set)

        for instance in dataset_descriptor.instances:
            object_name = get_object_prefix(instance.image_label)

            for label in instance.segmentations:
                if is_part_name(label):
                    part_graph[object_name].add(label)

        part_graph: dict[str, list[str]] = {
            object_name : sorted(part_graph[object_name])
            for object_name in sorted(part_graph)
        }

        dataset_descriptor.part_graph = part_graph

    ############################
    # Start Name Modifications #
    ############################
    if object_name_map is not None or part_name_map is not None:
        logger.info(f'Mapping names based on provided object and part name maps...')

        if object_name_map is None:
            object_name_map = lambda x: x

        if part_name_map is None:
            part_name_map = lambda x: x

        def map_name(label: str) -> str:
            if is_part_name(label): # Has PART_SEP in it
                return join_object_and_part(
                    object_name_map(get_object_prefix(label)),
                    part_name_map(get_part_suffix(label))
                )

            return object_name_map(label)

        for instance in dataset_descriptor.instances:
            instance.image_label = map_name(instance.image_label)

            instance.segmentations = {
                map_name(label) : segmentation
                for label, segmentation in instance.segmentations.items()
            }

        if dataset_descriptor.part_graph is not None:
            dataset_descriptor.part_graph = {
                map_name(label) : [map_name(part) for part in parts]
                for label, parts in dataset_descriptor.part_graph.items()
            }

        if dataset_descriptor.instance_graph is not None:
            dataset_descriptor.instance_graph = {
                map_name(label) : [map_name(descendant) for descendant in descendants]
                for label, descendants in dataset_descriptor.instance_graph.items()
            }

    # Strip concept-specific part prefixes
    if strip_concept_specific_part_prefix:
        logger.info(f'Stripping concept-specific part prefixes from segmentation labels...')

        for instance in dataset_descriptor.instances:
            instance.segmentations = {get_part_suffix(label) : segmentation for label, segmentation in instance.segmentations.items()}

        if dataset_descriptor.part_graph is not None:
            dataset_descriptor.part_graph = {
                get_part_suffix(label) : [get_part_suffix(part) for part in parts]
                for label, parts in dataset_descriptor.part_graph.items()
            }

        if dataset_descriptor.instance_graph is not None:
            dataset_descriptor.instance_graph = {
                get_part_suffix(label) : [get_part_suffix(descendant) for descendant in descendants]
                for label, descendants in dataset_descriptor.instance_graph.items()
            }

    if strip_object_literal_prefix:
        logger.info(f'Stripping "object--" prefixes from segmentation labels...')

        def strip_object_literal_prefix(label: str) -> str:
            return label.removeprefix('object--')

        for instance in dataset_descriptor.instances:
            instance.image_label = strip_object_literal_prefix(instance.image_label)
            instance.segmentations = {strip_object_literal_prefix(label) : segmentation for label, segmentation in instance.segmentations.items()}

        if dataset_descriptor.part_graph is not None:
            dataset_descriptor.part_graph = {
                strip_object_literal_prefix(label) : [strip_object_literal_prefix(part) for part in parts]
                for label, parts in dataset_descriptor.part_graph.items()
            }

        if dataset_descriptor.instance_graph is not None:
            dataset_descriptor.instance_graph = {
                strip_object_literal_prefix(label) : [strip_object_literal_prefix(descendant) for descendant in descendants]
                for label, descendants in dataset_descriptor.instance_graph.items()
            }

    # TODO validate that the part graph and instance graph are consistent with the instances

    return dataset_descriptor

def is_part_name(label: str) -> bool:
    return PART_SEP in label

def get_part_suffix(label: str, safe: bool = False) -> str:
    if safe and not is_part_name(label):
        raise ValueError(f'Label is not a part name: {label}')
    return label.split(PART_SEP)[-1]

def get_object_prefix(label: str) -> str:
    return label.split(PART_SEP)[0]

def get_category_name(label: str):
    return label.split('--')[0]

def join_object_and_part(object_name: str, part_name: str) -> str:
    return f'{object_name}{PART_SEP}{part_name}'
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from data.part_dataset_descriptor import PartDatasetInstance
from sentence_transformers import SentenceTransformer
from typing import Literal
import logging

logger = logging.getLogger(__name__)

@dataclass
class ObjectSamplerConfig:
    sampling_strategy: Literal['uniform', 'weighted'] = 'weighted'

    restrict_objects_to_dataset: bool = True # Otherwise, can use all objects in instance_graph
    restrict_samples_to_siblings: bool = True

    sentence_transformer_model_name: str = 'all-mpnet-base-v2'
    random_seed: int = 42

class ObjectSampler:
    def __init__(
        self,
        instances: list[PartDatasetInstance],
        instance_graph: dict[str, list[str]] = None,
        config: ObjectSamplerConfig = ObjectSamplerConfig()
    ):

        self.instances = instances
        self.instance_graph = instance_graph
        self.config = config

        self.rng = np.random.default_rng(self.config.random_seed)

        self.objects_from_graph = self._get_objects_from_graph()
        self.objects_from_dataset = self._get_objects_from_dataset()
        self.objects_set = set(self.objects)

        if self.config.restrict_samples_to_siblings:
            assert self.instance_graph is not None, 'Instance graph is required when restricting samples to siblings'
            self.object_to_parent = {obj : parent for parent, children in instance_graph.items() for obj in children}

        self.sentence_transformer = SentenceTransformer(self.config.sentence_transformer_model_name)

        self.object_embeds: Tensor = self.sentence_transformer.encode(self.objects, convert_to_tensor=True) # (n_objects, embedding_dim)
        self.object_to_idx = {obj : i for i, obj in enumerate(self.objects)}

        self.has_warned_about_restriction_failure = False

    @property
    def objects(self):
        return self.objects_from_dataset if self.config.restrict_objects_to_dataset else self.objects_from_graph

    def sample_similar_objects(
        self,
        n_samples: int,
        current_object: str,
        with_replacement: bool = False
    ) -> list[str]:

        restriction_failed = False
        if self.config.restrict_samples_to_siblings:
            if current_object not in self.object_to_parent:
                raise ValueError(f'Current object {current_object} is not in the instance graph')

            sibling_names = self._get_siblings(current_object)
            object_embeds = self.object_embeds[[self.object_to_idx[s_name] for s_name in sibling_names]] # (n_siblings, embedding_dim)
            object_labels = sibling_names

            restriction_failed = len(object_labels) < n_samples # If we don't have enough siblings, we can't sample

            if restriction_failed and not self.has_warned_about_restriction_failure:
                self.has_warned_about_restriction_failure = True
                logger.warning(
                    f'Only {len(object_labels)} siblings found for object {current_object}, '
                    f'falling back to all objects. No further warnings will be logged.'
                )

        if not self.config.restrict_samples_to_siblings or restriction_failed:
            object_embeds = torch.cat([ # Skip current object
                self.object_embeds[:self.object_to_idx[current_object]],
                self.object_embeds[self.object_to_idx[current_object] + 1:]
            ]) # (n_objects - 1, embedding_dim)

            object_labels = [
                obj for obj in self.objects
                if obj != current_object
            ]

        if self.config.sampling_strategy == 'uniform':
            return self.rng.choice(object_labels, size=n_samples, replace=with_replacement)

        elif self.config.sampling_strategy == 'weighted':
            # Compute similarity between current object and all other objects
            current_object_embed = self.object_embeds[self.object_to_idx[current_object]] # (embedding_dim,)
            similarities = current_object_embed @ object_embeds.T # (n,)

            # Normalize similarities to get a probability distribution
            probs = similarities.softmax(dim=0).cpu().numpy()

            # Sample objects from most similar to least similar
            return self.rng.choice(object_labels, size=n_samples, replace=with_replacement, p=probs)

        else:
            raise ValueError(f'Invalid sampling strategy: {self.config.sampling_strategy}')

    def _get_objects_from_graph(self):
        return sorted(self.instance_graph) if self.instance_graph else None

    def _get_objects_from_dataset(self):
        return sorted(set(instance.image_label for instance in self.instances))

    def _get_siblings(self, current_object: str) -> list[str]:
        return [
            child for child in self.instance_graph[self.object_to_parent[current_object]]
            if child != current_object and child in self.objects_set
        ]
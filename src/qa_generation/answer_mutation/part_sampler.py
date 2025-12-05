import numpy as np
from dataclasses import dataclass
from data.part_dataset_descriptor import PartDatasetInstance
from sklearn.linear_model import LogisticRegression
from itertools import chain
import logging
from tqdm import tqdm
from typing import Union, Iterable

logger = logging.getLogger(__name__)

@dataclass
class PartSamplerConfig:
    predict_closest_parts: bool = True
    predicted_parts_dist_weight: float = 0.1 # Combined with provided related parts distribution
    predicted_parts_dist_temperature: float = 1.5 # Temperature to smooth the predicted parts distribution

    restrict_parts_to_dataset: bool = True # Otherwise, uses all parts in part_graph

    predict_constant_for_single_label_parts: float = 0.

    random_seed: int = 42

class PartSampler:
    def __init__(
        self,
        instances: list[PartDatasetInstance],
        part_graph: dict[str, list[str]],
        config: PartSamplerConfig = PartSamplerConfig()
    ):

        self.instances = instances
        self.part_graph = part_graph
        self.config = config

        self.parts_from_graph = self._get_parts_from_graph(part_graph)
        self.parts_from_dataset = self._get_parts_from_dataset(include_obj_lbls=True)
        self.part_to_idx = self._get_part_to_idx_map(self.parts)

        self.rng = np.random.default_rng(self.config.random_seed)

        # NOTE The set of part predictors is a subset of all parts, as only parts which appear
        # in the dataset instances can be used to train predictors
        self.part_predictors: dict[str, LogisticRegression] = {}
        if self.config.predict_closest_parts:
            self._train_closest_part_predictors()

    @property
    def parts(self):
        return self.parts_from_dataset if self.config.restrict_parts_to_dataset else self.parts_from_graph

    def sample_part(
        self,
        existing_parts: Iterable[str],
        related_parts: Union[dict[str,float], Iterable[str]] = None,
        excluded_parts: Iterable[str] = set()
    ) -> str:
        '''
        Samples a new part using existing parts, a related parts distribution, and predictor outputs based on the current configuration.

        Sampling behavior depends on whether part predictions are enabled (i.e. `self.config.predict_closest_parts` is True) and on
        the availability of a related parts distribution after filtering out excluded parts.

        When part predictions are enabled (`predict_closest_parts` is True):
            1) If a related parts distribution is available:
                - If one or more part predictors contribute, their predicted distribution is computed (via softmax over logits)
                and merged with the related parts distribution using the weight `self.config.predicted_parts_dist_weight`.
                - If no predictors are included, the method falls back to using the full related parts distribution.
            2) If no related parts remain after exclusion:
                - If predictors are available, the predicted parts distribution is used exclusively.
                - Otherwise, a uniform distribution over all parts (excluding those in `excluded_parts`) is constructed;
                if this uniform distribution is empty, a RuntimeError is raised.

        When part predictions are disabled (`predict_closest_parts` is False):
            1) If a related parts distribution is available, it is used directly for sampling.
            2) If no related parts remain after exclusion, a uniform distribution over all parts (excluding those in `excluded_parts`)
            is used; if this uniform distribution is empty, a RuntimeError is raised.

        Args:
            existing_parts (Iterable[str]): The parts that are already present.
            related_parts (Union[dict[str, float], Iterable[str]], optional):
                Either a dictionary mapping parts to their sampling probabilities or an iterable of parts
                (in which case a uniform distribution is assumed). If not provided, all parts in the dataset are considered.
            excluded_parts (Iterable[str], optional):
                Parts that must be excluded from sampling.

        Returns:
            str: The sampled part.
        '''
        # Construct known parts distribution
        excluded_parts = set(excluded_parts)

        if not related_parts: # No parts provided; use all parts
            related_parts = self.parts

        if not isinstance(related_parts, dict): # Iterable of part names
            related_parts = list(filter(lambda part: part not in excluded_parts, related_parts))
            related_parts = {part : 1 / len(related_parts) for part in related_parts}

        else: # dict with probabilities provided
            assert not any(part in excluded_parts for part in related_parts), 'Excluded parts should not be in related parts distribution'

        # Handle case where there are no related parts after exclusion
        # If predict_closest_parts, use the predicted parts distribution. Otherwise, try uniform distribution over all parts
        has_related_parts = bool(related_parts)

        if not has_related_parts:
            if self.config.predict_closest_parts: # Use predicted parts distribution
                logger.debug('No related parts after exclusion; falling back to predicted parts distribution')

            else: # Set to uniform distribution over all parts not in excluded_parts
                logger.debug('No related parts after exclusion; falling back to uniform distribution over all parts')
                related_parts = self._uniform_parts_distribution(excluded_parts)

                if len(related_parts) == 0:
                    raise RuntimeError('Failed to construct related_parts distribution as all parts in dataset were excluded')
                else:
                    has_related_parts = True

        related_parts_dist = self._dict_to_vector(related_parts)

        # Construct final distribution, possibly incorporating predictions from part predictors
        if self.config.predict_closest_parts:
            # Predict the distribution of closest parts given the existing parts
            predicted_parts_logits = np.full(len(self.parts), -np.inf) # Initialize to -inf
            input_vector = self._binary_vector_from_parts(existing_parts)[None, :] # (1, n_parts)

            has_included_part_predictor = False
            for part, predictor in self.part_predictors.items():
                if part in excluded_parts: # Leave logit as -inf
                    continue

                part_idx = self.part_to_idx[part]
                part_input_vector = np.concatenate([ # Remove the part's column; (1, n_parts - 1)
                    input_vector[:, :part_idx],
                    input_vector[:, part_idx+1:]
                ], axis=1)
                logit = predictor.decision_function(part_input_vector)[0]
                predicted_parts_logits[part_idx] = logit
                has_included_part_predictor = True

            if not has_included_part_predictor:
                if has_related_parts: # Fall back to related parts distribution
                    logger.warning('No part predictors were included; falling back to related parts distribution')
                    final_dist = related_parts_dist
                else: # Fall back to uniform distribution
                    logger.warning('No part predictors or related parts were included; using uniform distribution over all parts')
                    final_dist = self._uniform_parts_distribution(excluded_parts)
                    final_dist = self._dict_to_vector(final_dist)

                    if len(final_dist) == 0: # Nothing else to try
                        raise RuntimeError('Failed to construct final distribution as all parts in dataset were excluded')
            else:
                # Compute predicted distribution via softmax
                predicted_parts_logits = predicted_parts_logits / self.config.predicted_parts_dist_temperature # Temperature
                predicted_parts_logits_exp = np.exp(predicted_parts_logits)
                predicted_parts_dist = predicted_parts_logits_exp / np.sum(predicted_parts_logits_exp) # (n_parts,)

                # Combine predicted and related parts distributions
                predicted_parts_dist_weight = self.config.predicted_parts_dist_weight if has_related_parts else 1.
                final_dist = (1 - predicted_parts_dist_weight) * related_parts_dist + \
                            predicted_parts_dist_weight * predicted_parts_dist

        else:
            final_dist = related_parts_dist

        return str(self.rng.choice(self.parts, p=final_dist)) # Convert back to string since choice returns numpy string by default

    def _uniform_parts_distribution(self, excluded_parts: Iterable[str] = set()) -> dict[str, float]:
        parts = [part for part in self.parts if part not in excluded_parts]
        return {part : 1 / len(parts) for part in parts}

    def _binary_vector_from_parts(self, parts: Iterable[str]):
        binary_vector = np.zeros(len(self.parts))
        binary_vector[[self.part_to_idx[part] for part in parts]] = 1

        return binary_vector

    def _get_part_to_idx_map(self, parts: Iterable[str]):
        return {part : i for i, part in enumerate(parts)}

    def _get_parts_from_graph(self, part_graph: dict[str, list[str]]) -> list[str]:
        return list(dict.fromkeys(chain.from_iterable(part_graph.values()))) # Flatten part graph and remove duplicates

    def _dict_to_vector(self, part_probs: dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.parts))
        for part, p in part_probs.items():
            vec[self.part_to_idx[part]] = p

        return vec

    def _get_parts_from_dataset(self, include_obj_lbls=False) -> list[str]:
        all_parts = set(self.parts_from_graph)
        encountered_parts = {}  # Parts which appear in dataset, which may be subset of all parts
        for instance in self.instances:
            segmentation_labels = set(instance.segmentation_labels)
            if include_obj_lbls:
                part_labels = list(segmentation_labels)
            else:
                part_labels = list(segmentation_labels & all_parts) # (n_parts,); ensure only parts are included (not object labels)
            encountered_parts.update({part : None for part in part_labels})

        return list(encountered_parts)

    def _train_closest_part_predictors(self):
        '''
        For each part, trains a model to predict the presence of its part based on the presence of all other parts
        '''
        # Construct encoded dataset
        parts_from_dataset = set(self.parts_from_dataset)
        X = np.zeros((len(self.instances), len(self.parts_from_dataset)))

        for i, instance in enumerate(self.instances):
            segmentation_labels = set(instance.segmentation_labels)
            part_labels = list(segmentation_labels & parts_from_dataset) # (n_parts,); ensure only parts are included (not object labels)

            for part_label in part_labels:
                part_idx = self.part_to_idx[part_label]
                X[i, part_idx] = 1

        # Train a model for each part
        logger.info('Training part predictors')
        for part in tqdm(parts_from_dataset):
            X_part = np.delete(X, self.part_to_idx[part], axis=1) # Remove the part's column; (n_instances, n_parts - 1)
            y = X[:, self.part_to_idx[part]] # (n_instances,)

            if len(np.unique(y)) == 1:
                logger.warning(f'Part {part} appears in all or no data instances; will predict constant logit for this part')
                predictor = ConstantPredictor(self.config.predict_constant_for_single_label_parts)
            else:
                predictor = LogisticRegression()
                predictor.fit(X_part, y)

            self.part_predictors[part] = predictor

class ConstantPredictor:
    def __init__(self, constant: float = 0.):
        self.constant = constant

    def decision_function(self, X):
        return np.full(X.shape[0], self.constant)
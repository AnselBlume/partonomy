# %%
import os
import sys
sys.path = [os.path.realpath(f'{__file__}/../..')] + sys.path
from dataclasses import dataclass
from data.part_dataset_descriptor import PartDatasetInstance
from itertools import chain
from typing import Iterable
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class InstanceSamplerConfig:
    sample_singletons: bool = False

    sample_pairs: bool = False
    allow_same_image_in_pairs: bool = False
    allow_same_image_label_in_pairs: bool = True
    treat_order_as_distinct: bool = True

    # Note: when generating, this yields the proportion of samples to keep in expectation,
    # but the actual number of samples may be different. To get the exact number of samples
    # denoted by this proportion, call sample_exact_proportion, which samples after generation.
    proportion_to_yield: float = 1.0
    random_seed: int = 42

    def __post_init__(self):
        if sum([self.sample_singletons, self.sample_pairs]) != 1:
            raise ValueError('Exactly one of sample_singletons and sample_pairs must be True.')

        if self.allow_same_image_in_pairs and not self.allow_same_image_label_in_pairs:
            raise ValueError('allow_same_image_in_pairs can only be True if allow_same_image_label_in_pairs is True.')

        if not (0 <= self.proportion_to_yield <= 1):
            raise ValueError('Sampling proportion must be between 0 and 1.')

class InstanceSampler:
    def __init__(self, instances_by_image_label: dict[str, list[PartDatasetInstance]], config: InstanceSamplerConfig):
        self.instances_by_image_label = instances_by_image_label
        self.config = config
        self.rng = np.random.default_rng(self.config.random_seed)

    def __len__(self):
        return self._compute_approx_length()

    def __iter__(self) -> Iterable[PartDatasetInstance]:
        # TODO implement filtering
        if self.config.sample_singletons:
            for instances in self.instances_by_image_label.values():
                for instance in instances:
                    yield from self._conditional_yield(instance)

        elif self.config.sample_pairs:
            if self.config.allow_same_image_label_in_pairs:
                all_instances = list(chain.from_iterable(self.instances_by_image_label.values()))

                if self.config.treat_order_as_distinct: # Ordered pairs
                    for instance1 in all_instances:
                        for instance2 in all_instances:
                            # Handle same image case
                            if self.config.allow_same_image_in_pairs or instance1 != instance2:
                                yield from self._conditional_yield((instance1, instance2))

                else: # Unordered pairs
                    # Start from i + 1 if not allowing same image in pairs
                    offset = not self.config.allow_same_image_in_pairs

                    for i in range(len(all_instances)):
                        for j in range(i + offset, len(all_instances)):
                            yield from self._conditional_yield((all_instances[i], all_instances[j]))

            else: # Not allowing same image label in pairs
                labels = list(self.instances_by_image_label)

                for label1_index in range(len(self.instances_by_image_label)):
                    # Start second label index from zero if treating order as distinct, else from label1_index + 1
                    offset = -label1_index if self.config.treat_order_as_distinct else 1

                    for label2_index in range(label1_index + offset, len(self.instances_by_image_label)):
                        if label1_index == label2_index: # Same label
                            continue

                        label1 = labels[label1_index]
                        label2 = labels[label2_index]

                        for label1_instance in self.instances_by_image_label[label1]:
                            for label2_instance in self.instances_by_image_label[label2]:
                                yield from self._conditional_yield((label1_instance, label2_instance))
        else:
            raise ValueError('Invalid configuration.')

    def sample_exact_proportion(self) -> list[PartDatasetInstance]:
        # Get all samples
        old_proportion = self.config.proportion_to_yield
        self.config.proportion_to_yield = 1.0
        all_samples = list(self)
        self.config.proportion_to_yield = old_proportion

        # Sample from all samples
        n_samples = int(self.config.proportion_to_yield * len(all_samples))
        return self.rng.choice(all_samples, n_samples, replace=False).tolist()

    def _conditional_yield(self, item):
        if self.rng.random() < self.config.proportion_to_yield:
            yield item

    def _compute_approx_length(self):
        '''
            Returns the length if not rejecting any samples; else returns approximate length (not accounting for
            runtime filtering).
        '''
        n_singletons = sum([len(instances) for instances in self.instances_by_image_label.values()])

        if self.config.sample_singletons:
            return n_singletons

        elif self.config.sample_pairs:
            if self.config.allow_same_image_in_pairs: # Sampling from all with replacement
                n_ordered_pairs = n_singletons ** 2
                if self.config.treat_order_as_distinct:
                    return n_ordered_pairs
                else: # Subtract out pairs of the same singleton, divide result by two, add back pairs of same singleton
                    return (n_ordered_pairs - n_singletons) // 2 + n_singletons

            elif self.config.allow_same_image_label_in_pairs: # Sampling from all without replacement
                return n_singletons * (n_singletons - 1) // (1 + (not self.config.treat_order_as_distinct))

            else: # Pairs must come from different image labels
                if len(self.instances_by_image_label) == 1:
                    raise ValueError('Cannot sample pairs from a single image label when allow_same_image_label_in_pairs is False.')

                n_pairs = 0
                n_images_per_label = [len(instances) for instances in self.instances_by_image_label.values()]
                instances_remaining = sum(n_images_per_label) # Num remaining instances of other labels to sample from

                for n_images in n_images_per_label:
                    n_options = instances_remaining - n_images # Exclude instances from the same label
                    n_pairs += n_images * n_options

                    if not self.config.treat_order_as_distinct: # Exclude pairs of reverse order
                        instances_remaining = n_options

                return n_pairs

if __name__ == '__main__':
    from pprint import pformat
    import coloredlogs
    coloredlogs.install(level='DEBUG')

    label_to_instance = {
        'label1': ['1_1'],
        'label2': ['2_1', '2_2'],
        'label3': ['3_1', '3_2', '3_3']
    }

    # %% Test singletons
    logger.info('Testing singletons')

    config = InstanceSamplerConfig(sample_singletons=True)
    sampler = InstanceSampler(label_to_instance, config)
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    # %% Test pairs
    # Allow same image in pairs
    logger.info('\nTesting pairs with same image in pairs')

    config = InstanceSamplerConfig(
        sample_pairs=True,
        allow_same_image_in_pairs=True,
        allow_same_image_label_in_pairs=True,
        treat_order_as_distinct=True
    )
    sampler = InstanceSampler(label_to_instance, config)
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    logger.info(f'\nTesting same as above, but unordered')

    sampler.config.treat_order_as_distinct = False
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    # %% Disallow same image in pairs
    logger.info(f'\nTesting pairs without same image in pairs')

    config = InstanceSamplerConfig(
        sample_pairs=True,
        allow_same_image_in_pairs=False,
        allow_same_image_label_in_pairs=True,
        treat_order_as_distinct=True
    )
    sampler = InstanceSampler(label_to_instance, config)
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    logger.info(f'\nTesting same as above, but unordered')

    sampler.config.treat_order_as_distinct = False
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    # %% Disallow same label in pairs
    logger.info(f'\nTesting pairs without same label in pairs')

    config = InstanceSamplerConfig(
        sample_pairs=True,
        allow_same_image_in_pairs=False,
        allow_same_image_label_in_pairs=False,
        treat_order_as_distinct=True
    )
    sampler = InstanceSampler(label_to_instance, config)
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    logger.info(f'\nTesting same as above, but unordered')

    sampler.config.treat_order_as_distinct = False
    sampled = list(sampler)

    logger.info(f'Expected length: {len(sampler)}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

    # %% Test proportional sampling
    logger.info('\nTesting proportional sampling')

    config = InstanceSamplerConfig(
        sample_singletons=True,
        proportion_to_yield=0.5
    )
    sampler = InstanceSampler(label_to_instance, config)
    n_tests = 1000
    lens = []
    for _ in range(n_tests):
        lens.append(len(list(sampler)))

    logger.info(f'Expected length: {len(sampler) / 2}; Average length: {np.mean(lens)}')
    logger.info(pformat(sampled))

    # %% Test exact proportion sampling
    logger.info('\nTesting exact proportion sampling')

    config = InstanceSamplerConfig(
        sample_singletons=True,
        proportion_to_yield=0.5
    )
    sampler = InstanceSampler(label_to_instance, config)
    n_tests = 1000
    actual_len = len(sampler.sample_exact_proportion())
    for _ in range(n_tests):
        sampled = sampler.sample_exact_proportion()
        assert len(sampled) == actual_len

    logger.info(f'Expected length: {int(0.5 * len(sampler))}; Actual length: {len(sampled)}')
    logger.info(pformat(sampled))

# %%

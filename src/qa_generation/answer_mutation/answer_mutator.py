import numpy as np
from dataclasses import dataclass, field
from qa_generation.answer import AnswerOperation, AnswerOperationName
from typing import Iterable, Any
from .part_sampler import PartSampler
from .object_sampler import ObjectSampler
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnswerMutatorConfig:
    # Part mutation parameters
    min_mutations: int = 1
    max_mutations: int = 3

    answer_operation_probs: dict[AnswerOperationName, float] = field(default_factory=lambda: {
        AnswerOperationName.ADD: 1/3,
        AnswerOperationName.DELETE: 1/3,
        AnswerOperationName.REPLACE: 1/3
    })

    random_seed: int = 42

    def __post_init__(self):
        assert np.isclose(np.sum(list(self.answer_operation_probs.values())), 1), 'Answer operation probabilities must sum to 1'

class AnswerMutator:
    def __init__(
        self,
        part_sampler: PartSampler,
        object_sampler: ObjectSampler,
        config: AnswerMutatorConfig = AnswerMutatorConfig()
    ):

        self.part_sampler = part_sampler
        self.object_sampler = object_sampler
        self.config = config
        self.rng = np.random.default_rng(self.config.random_seed)

    def mutate_object(self, current_object: str, n_samples: int = 1) -> list[str]:
        '''
        Mutate an object by sampling similar objects.

        Args:
            current_object: The object to mutate.
            n_samples: The number of similar objects to sample.

        Returns:
            A list of sampled objects which are not equal to the current object.
        '''
        return self.object_sampler.sample_similar_objects(n_samples=n_samples, current_object=current_object)

    def mutate_parts(
        self,
        parts: Iterable[str],
        related_parts: Iterable[str],
        possible_operations: Iterable[AnswerOperationName] = None,
        operation_probs: Iterable[float] = None,
        parts_to_exclude_for_addition: Iterable[str] = []
    ) -> tuple[list[str], list[AnswerOperation]]:
        '''
        Mutate a list of parts by applying a random sequence of operations.

        Args:
            parts: The list of parts to mutate.
            related_parts: The list of related parts to the parts.
            possible_operations: The list of possible operations to apply.
            operation_probs: The list of probabilities for each operation.
        '''

        parts = list(parts)

        if possible_operations is None:
            possible_operations = list(self.config.answer_operation_probs)
            operation_probs = list(self.config.answer_operation_probs.values())

        if operation_probs is None:
            operation_probs = np.full(len(possible_operations), 1/len(possible_operations))
        else:
            assert len(possible_operations) == len(operation_probs), 'Length of operation_probs must match length of possible_operations'
            assert np.isclose(np.sum(operation_probs), 1), 'operation_probs must sum to 1'

        n_operations = self.rng.integers(self.config.min_mutations, self.config.max_mutations) # Number of mutations to apply
        operation_names = self.rng.choice(possible_operations, n_operations, replace=True, p=operation_probs) # Which mutations to apply

        # Keep track of parts involved in past operations so they are not involved in future operations
        parts_from_past_ops = set()
        parts_to_exclude_for_addition = set(parts_to_exclude_for_addition)

        operations = []
        for operation_name in operation_names:
            # If there are no parts, only add operations are possible
            if not len(parts) and operation_name != AnswerOperationName.ADD:
                operation_name = AnswerOperationName.ADD

            try:
                operation = self._apply_operation(operation_name, parts, related_parts, parts_from_past_ops, parts_to_exclude_for_addition)
                parts_from_past_ops.update(operation.values)
                operations.append(operation)

            except RuntimeError as e:
                logger.debug(f'Failed to apply operation {operation_name}: {e}')

        return parts, operations

    def _apply_operation(
        self,
        operation_name: AnswerOperationName,
        parts: list[str],
        related_parts: Iterable[str],
        parts_from_past_ops: set[str] = set(),
        parts_to_exclude_for_addition: set[str] = set()
    ) -> AnswerOperation:
        '''
        Apply an operation to a list of parts.

        Args:
            operation_name: The name of the operation to apply.
            parts: The list of parts to apply the operation to.
            related_parts: The list of related parts to the parts.
            parts_from_past_ops: The list of parts that have been involved in past operations which will not be either added or deleted.
            parts_to_exclude_for_addition: The list of parts that should not be added.
        '''

        if operation_name == AnswerOperationName.ADD:
            parts_to_exclude = parts_from_past_ops.union(parts).union(parts_to_exclude_for_addition) # Don't add a part that's already present
            operation = self._add(parts, related_parts, parts_to_exclude)

        elif operation_name == AnswerOperationName.DELETE:
            operation = self._delete(parts, parts_from_past_ops)

        elif operation_name == AnswerOperationName.REPLACE:
            parts_to_exclude_for_deletion = parts_from_past_ops
            parts_to_exclude_for_addition = parts_from_past_ops.union(parts).union(parts_to_exclude_for_addition) # Don't add a part that's already present
            operation = self._replace(parts, related_parts, parts_to_exclude_for_deletion, parts_to_exclude_for_addition)

        return operation

    def _add(self, parts: list[str], related_parts: Iterable[str], parts_to_exclude: set[str]) -> AnswerOperation:
        part = self.part_sampler.sample_part(parts, related_parts, parts_to_exclude)
        insert_index = self._random_insert(part, parts)

        return AnswerOperation(AnswerOperationName.ADD, part, insert_index)

    def _delete(self, parts: list[str], parts_to_exclude: set[str]) -> AnswerOperation:
        valid_deletion_inds = [i for i, part in enumerate(parts) if part not in parts_to_exclude]

        if not valid_deletion_inds:
            raise RuntimeError(f'No valid parts to delete in {parts} with parts_to_exclude {parts_to_exclude}')

        deletion_ind = self.rng.choice(valid_deletion_inds)
        part = parts.pop(deletion_ind)

        return AnswerOperation(AnswerOperationName.DELETE, part)

    def _replace(
        self,
        parts: list[str],
        related_parts: Iterable[str],
        parts_to_exclude_for_deletion: set[str],
        parts_to_exclude_for_addition: set[str] = set()
    ) -> AnswerOperation:
        valid_replacement_inds = [i for i, part in enumerate(parts) if part not in parts_to_exclude_for_deletion]
        if not valid_replacement_inds:
            raise RuntimeError(f'No valid parts to replace in {parts} with parts_to_exclude {parts_to_exclude_for_deletion}')

        replacement_ind = self.rng.choice(valid_replacement_inds) # Index of the part to replace
        replaced_part = parts[replacement_ind]

        parts_to_exclude_for_addition = parts_to_exclude_for_addition.union([replaced_part]) # Shouldn't replace a part with itself
        replacement_part = self.part_sampler.sample_part(parts, related_parts, parts_to_exclude_for_addition)

        parts[replacement_ind] = replacement_part

        return AnswerOperation(AnswerOperationName.REPLACE, replaced_part, replacement_part)

    def _random_insert(self, elem_to_insert: Any, to_insert_into: list) -> int:
        insert_index = self.rng.integers(0, len(to_insert_into)) if len(to_insert_into) else 0
        to_insert_into.insert(insert_index, elem_to_insert)

        return int(insert_index)
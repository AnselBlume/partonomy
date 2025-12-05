import orjson
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from .question_type import QuestionType
from .qa_pair import QAPair, AnswerType
from .answer_mutation import AnswerMutator
from .answer import AnswerOperation
from .concept_graph import ConceptGraph
from .part_comparison import PartComparator, PartSetComparison
from data.part_dataset_descriptor import PartDatasetDescriptor, PartDatasetInstance
from itertools import chain, islice
from typing import Iterable
import inflect
import logging
import os
import openai

logger = logging.getLogger(__name__)

@dataclass
class QAGeneratorConfig:
    dataset_path: str = None

    # Wrong answer generation
    n_wrong_answers: int = 4
    allow_empty_answers: bool = False
    n_tries_to_generate: int = 20

    # Question generation
    max_questions_per_question_type_per_instance: int = 1
    negative_questions_require_nonempty_intersection: bool = True

    # Part outputs
    sort_parts: bool = True
    shuffle_parts: bool = False
    comparator_batch_size: int = 100

    random_seed: int = 42

class QAGenerator:
    def __init__(
        self,
        concept_graph: ConceptGraph,
        dataset_descriptor: PartDatasetDescriptor,
        answer_mutator: AnswerMutator,
        part_comparator: PartComparator,
        config: QAGeneratorConfig = QAGeneratorConfig()
    ):
        self.concept_graph = concept_graph
        self.dataset_descriptor = dataset_descriptor
        self.answer_mutator = answer_mutator
        self.part_comparator = part_comparator
        self.config = config

        self.instances_by_image_label: dict[str,list[PartDatasetInstance]] = None
        self.inflect = inflect.engine()
        self.rng = np.random.default_rng(self.config.random_seed)

        self._init_data_structures()

    def generate_qa_pairs(
        self,
        question_types: Iterable[QuestionType] = None,
        save_intermediate: bool = False,
        json_save_path: str = None
    ) -> list[QAPair]:
        '''
        Generates QA pairs for the concept graph.

        If `json_save_path` is provided, saves the QA pairs to json_save_path.
        If `save_intermediate` is True, the QA pairs are saved to `json_save_path` after every question type.
        '''
        if question_types is None:
            question_types = list(QuestionType)

        question_types = list(dict.fromkeys(question_types))

        def save_qa_pairs(qa_pairs: list[QAPair]):
            if json_save_path is None:
                raise ValueError('json_save_path must be provided if save_intermediate is True')

            logger.info(f'Saving QA pairs to {json_save_path}...')

            os.makedirs(os.path.dirname(json_save_path) or '.', exist_ok=True)
            with open(json_save_path, 'wb') as f:
                f.write(orjson.dumps([p.to_dict() for p in qa_pairs], option=orjson.OPT_INDENT_2))

        qa_pairs: list[QAPair] = []
        for question_type in question_types:
            logger.info(f'Generating QA pairs for question type {question_type}...')
            question_qa_pairs = self._generate_qa_pairs_for_question_type(question_type)
            # TODO balance empty correct answers with non-empty correct answers

            qa_pairs.extend(question_qa_pairs)

            if save_intermediate:
                save_qa_pairs(qa_pairs)

        if json_save_path and not save_intermediate: # If save_intermediate, we've already saved the QA pairs
            save_qa_pairs(qa_pairs)

        return qa_pairs

    def _generate_variations(self, text: str = None, n: int = 10, model='gpt-4') -> list[str]:
        '''
        Use LLM (e.g., 'model') to generate 'n' different variations of a given 'text'.
        '''
        variations = []
        if text is None or text == "":
            raise ValueError(f"text cannot be a NoneType.")

        openai.api_key = os.environ.get("OPENAI_API_KEY")  # NOTE: If you don't have the OPENAI_API_KEY in the os.environ, then need to set it up
        for _ in range(n):
            response = openai.ChatCompletion.create(
                model=model,
                messages = [
                    {
                    "role": "system",
                    "content": (
                        "You are a helpful rewriting assistant. Your task is to rephrase the user's text "
                        "in a different style while preserving the original meaning."
                    )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Rewrite the text in a different style but keep the same meaning:\n\n{text}"
                        )
                    }
                ],
                temperature=0.7,
                max_tokens=256  # NOTE: You can fix this to the maximum possible length
            )
            var_text = response['choices'][0]['message']['content'].strip()
            variations.append(var_text)

        return variations

    def _generate_qa_pairs_for_question_type(self, question_type: QuestionType):
        if question_type == QuestionType.IDENTIFICATION:
            return self._generate_identification_qa_pairs(include_label=False)
        elif question_type == QuestionType.IDENTIFICATION_WITH_LABEL:
            return self._generate_identification_qa_pairs(include_label=True)
        elif question_type == QuestionType.POSITIVE:
            # return self._generate_positive_qa_pairs(include_label=False)
            return self._generate_positive_qa_pairs_batched(include_label=False)
        elif question_type == QuestionType.POSITIVE_WITH_LABEL:
            # return self._generate_positive_qa_pairs(include_label=True)
            return self._generate_positive_qa_pairs_batched(include_label=True)
        elif question_type == QuestionType.NEGATIVE:
            # return self._generate_negative_qa_pairs(include_label=False)
            return self._generate_negative_qa_pairs_batched(include_label=False)
        elif question_type == QuestionType.NEGATIVE_WITH_LABEL:
            # return self._generate_negative_qa_pairs(include_label=True)
            return self._generate_negative_qa_pairs_batched(include_label=True)
        elif question_type == QuestionType.DIFFERENCE:
            return self._generate_difference_qa_pairs(include_label=False)
        elif question_type == QuestionType.DIFFERENCE_WITH_LABEL:
            return self._generate_difference_qa_pairs(include_label=True)
        elif question_type == QuestionType.WHOLE_TO_PART:
            return self._generate_whole_to_part_qa_pairs()
        elif question_type == QuestionType.PART_TO_WHOLE:
            return self._generate_part_to_whole_qa_pairs()
        else:
            raise ValueError(f'Invalid question type {question_type}.')

    def _build_part_str(self, parts: list[str]) -> str:
        ret_str = ''
        for i, part in enumerate(parts):
            if i == 0:
                ret_str += f'{part}'
            elif i == len(parts) - 1:
                ret_str += f', and {part}'
            else:
                ret_str += f', {part}'

        return ret_str

    def _generate_identification_qa_pairs(self, include_label: bool = False):
        '''
        '''
        def build_question(instance_label: str):
            return (
                f'What visible parts does the {self._get_object_name(instance_label, include_label)} in the image have?'
            )

        def build_answer(parts: list[str], instance_label: str):
            return (
                f'The {self._get_object_name(instance_label, include_label)} in the image has the following visible parts: {self._build_part_str(parts)}.'
            )

        question_type = QuestionType.IDENTIFICATION_WITH_LABEL if include_label else QuestionType.IDENTIFICATION

        # Collect all instances
        all_instances = list(chain.from_iterable(self.instances_by_image_label.values()))
        pairs = []

        for instance in tqdm(all_instances, desc='Generating Identification QA Pairs'):
            instance_label = instance.image_label
            instance_parts = self._get_instance_parts(instance)

            if not instance_parts:
                logger.debug(f'No parts found for instance {instance.image_path} with label {instance_label}. Skipping...')
                continue

            # Generate Q&A
            question = build_question(instance.image_label)
            answer = build_answer(instance_parts, instance.image_label)

            # Generate wrong answers
            related_parts = set(self._get_label_parts(instance_label)) # All parts of instance's label

            try:
                wrong_answers_parts, answers_operations = self._generate_wrong_answers(instance_parts, related_parts)
            except RuntimeError as e:
                logger.warning(e)
                continue  # Failed to generate wrong answers; skip instance

            # Construct final answers and types
            wrong_answers = [build_answer(parts, instance.image_label) for parts in wrong_answers_parts]
            answers = [answer] + wrong_answers
            answer_types = [AnswerType.CORRECT] + [AnswerType.INCORRECT] * len(wrong_answers)
            answer_parts = [instance_parts] + wrong_answers_parts # Use object parts as keys for segmentation masks

            pair = QAPair(
                image_path=instance.image_path,
                image_label=instance.image_label,
                question=question,
                question_type=question_type,
                answer_choices=answers,
                answer_types=answer_types,
                answer_parts=answer_parts,
                answer_operations=answers_operations,
                segmentations=instance.segmentations
            )

            pairs.append(pair)

        return pairs

    def _generate_positive_qa_pairs(self, include_label: bool = False):
        '''
        1. Sample two object labels (possibly equal), `label1` and `label2`
        2. Sample image from `label1`
        3. Set `label := label2`
        4. Answer: intersection of parts(label1) and parts(label2)
        '''
        def build_question(affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'What parts does the {self._get_object_name(instance_label, include_label)} in the image have '
                + f'in common with {self.inflect.a(affirmative_label)}?'
            )

        def build_answer(parts: list[str], affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'The {self._get_object_name(instance_label, include_label)} in the image has the following parts in common with '
                + f'{self.inflect.a(affirmative_label)}: {self._build_part_str(parts)}.'
            )

        question_type = QuestionType.POSITIVE_WITH_LABEL if include_label else QuestionType.POSITIVE

        pairs = []
        for instance in tqdm(list(chain.from_iterable(self.instances_by_image_label.values())), desc=f'Positive QA Pairs'):
            instance_label = instance.image_label

            possible_conditioning_labels = list(self.labels_with_nonempty_intersection[instance_label]) # Copy for shuffling

            if not possible_conditioning_labels:
                logger.warning(f'No possible conditioning labels for instance {instance.image_path} with label {instance_label}. Skipping...')
                continue

            self.rng.shuffle(possible_conditioning_labels)
            n_generated = 0
            curr_index = 0
            while n_generated < self.config.max_questions_per_question_type_per_instance and curr_index < len(possible_conditioning_labels):
                conditioned_label = possible_conditioning_labels[curr_index]
                curr_index += 1

                # Get gt parts
                image_parts = self._get_instance_parts(instance)
                label_parts = self._get_label_parts(conditioned_label)

                comparison = self._split_part_union((image_parts, label_parts))[0]
                gt_parts = [part.part1 for part in comparison.intersection]

                if not gt_parts and not self.config.allow_empty_answers:
                    continue

                # Build question and answer
                question = build_question(conditioned_label, None, instance.image_label)
                answer = build_answer(gt_parts, conditioned_label, None, instance.image_label)

                # Generate wrong answers
                related_parts = set(self._get_label_parts(instance_label)).union(label_parts) # All parts of instance's label and conditioned label

                try:
                    wrong_answers_parts, answers_operations = self._generate_wrong_answers(gt_parts, related_parts)
                except RuntimeError as e:
                    logger.warning(e)
                    continue

                # Construct final answers and types
                wrong_answers = [build_answer(parts, conditioned_label, None, instance.image_label) for parts in wrong_answers_parts]

                answers = [answer] + wrong_answers
                answer_types = [AnswerType.CORRECT] + [AnswerType.INCORRECT] * len(wrong_answers)
                answer_parts = [gt_parts] + wrong_answers_parts

                pair = QAPair(
                    image_path=instance.image_path,
                    image_label=instance_label,
                    question=question,
                    question_type=question_type,
                    answer_choices=answers,
                    answer_types=answer_types,
                    answer_parts=answer_parts,
                    answer_operations=answers_operations,
                    segmentations=instance.segmentations
                )

                pairs.append(pair)
                n_generated += 1

        return pairs

    def _generate_positive_qa_pairs_batched(self, include_label: bool = False):
        '''
        1. Sample two object labels (possibly equal), `label1` and `label2`
        2. Sample image from `label1`
        3. Set `label := label2`
        4. Answer: intersection of parts(label1) and parts(label2)
        '''
        def build_question(affirmative_label: str, instance_label: str):
            return (
                f'What visible parts does the {self._get_object_name(instance_label, include_label)} in the image have '
                f'in common with {self.inflect.a(affirmative_label)}?'
            )

        def build_answer(parts: list[str], affirmative_label: str, instance_label: str):
            return (
                f'The {self._get_object_name(instance_label, include_label)} in the image has the following visible parts in common with '
                + f'{self.inflect.a(affirmative_label)}: {self._build_part_str(parts)}.'
            )

        question_type = QuestionType.POSITIVE_WITH_LABEL if include_label else QuestionType.POSITIVE

        # Maintain instance order with an OrderedDict
        instance_pairs = defaultdict(list)  # Maps instances to their QAPairs list

        # Track which instances still need more questions
        remaining_instances = {}

        # Collect all instances
        all_instances = list(chain.from_iterable(self.instances_by_image_label.values()))

        # Initialize retry tracking
        logger.info(f'Initializing all instances for batching...')
        for instance in all_instances:
            instance_label = instance.image_label
            possible_labels = list(self.labels_with_nonempty_intersection[instance_label])
            if possible_labels:
                self.rng.shuffle(possible_labels)
                remaining_instances[instance] = possible_labels  # Store available labels

        prog_bar = tqdm(total=len(remaining_instances), desc='Generating Positive QA Pairs')
        while remaining_instances:  # Keep going until all instances are satisfied
            batch_instances = list(islice(remaining_instances, self.config.comparator_batch_size))

            # Prepare batched inputs
            batch_part_pairs = []
            batch_labels = []
            batch_instances_filtered = []  # Instances with conditioning labels to try remaining

            for instance in batch_instances:
                possible_labels = remaining_instances[instance]

                if not possible_labels:  # No more options for this instance
                    del remaining_instances[instance]
                    prog_bar.update(1)
                    continue

                # Pick the next label for this instance
                conditioned_label = possible_labels.pop(0)

                batch_part_pairs.append((self._get_instance_parts(instance), self._get_label_parts(conditioned_label)))
                batch_labels.append(conditioned_label)
                batch_instances_filtered.append(instance)  # Store for processing

            if not batch_part_pairs:
                continue  # Skip empty batch

            # Call `_split_part_union` in batch
            batch_comparisons = self._split_part_union(list_of_part_lists=batch_part_pairs)

            # Iterate over batch results and create Q&A
            for instance, (instance_parts, label_parts), conditioned_label, comparison in zip(
                batch_instances_filtered,
                batch_part_pairs,
                batch_labels,
                batch_comparisons
            ):
                # Extract parts from comparison
                gt_parts = comparison.intersection

                if not gt_parts and not self.config.allow_empty_answers:
                    continue  # Skip, but retry with another label

                head_gt_parts = []
                object_gt_parts = []
                concept_gt_parts = []

                for part in gt_parts:
                    head_gt_parts.append(part.head)
                    object_gt_parts.append(part.part1)
                    concept_gt_parts.append(part.part2)

                # Generate Q&A
                question = build_question(conditioned_label, instance.image_label)
                answer = build_answer(head_gt_parts, conditioned_label, instance.image_label)

                # Generate wrong answers
                related_parts = set(self._get_label_parts(instance.image_label)).union(label_parts) # All parts of instance's label and conditioned label

                try:
                    # Use object parts to generate wrong answers, as head may be synonym from other concept's parts
                    # Don't add either the object's or concept's GT parts which may collide with the head parts
                    parts_to_exclude_for_addition = set(object_gt_parts).union(concept_gt_parts)
                    wrong_answers_parts, answers_operations = self._generate_wrong_answers(head_gt_parts, related_parts, parts_to_exclude_for_addition)
                except RuntimeError as e:
                    logger.warning(e)
                    continue  # Retry with another label

                # Construct final answers and types
                wrong_answers = [build_answer(parts, conditioned_label, instance.image_label) for parts in wrong_answers_parts]
                answers = [answer] + wrong_answers
                answer_types = [AnswerType.CORRECT] + [AnswerType.INCORRECT] * len(wrong_answers)
                answer_parts = [[part.part1 for part in gt_parts], *wrong_answers_parts] # Use object parts as keys for segmentation masks

                pair = QAPair(
                    image_path=instance.image_path,
                    image_label=instance.image_label,
                    question=question,
                    question_type=question_type,
                    answer_choices=answers,
                    answer_types=answer_types,
                    answer_parts=answer_parts,
                    answer_operations=answers_operations,
                    segmentations=instance.segmentations
                )

                instance_pairs[instance].append(pair)  # Store pair for this instance

                # If this instance has generated enough questions, remove it
                if len(instance_pairs[instance]) >= self.config.max_questions_per_question_type_per_instance:
                    del remaining_instances[instance]
                    prog_bar.update(1)

        prog_bar.close()
        # Flatten results while preserving original instance order
        pairs = chain.from_iterable(instance_pairs.values())

        return pairs

    def _generate_negative_qa_pairs(self, include_label: bool = False):
        '''
        1. Sample two different object labels, `label1` and `label2`
        2. Sample image from `label1`
        3. Set `alternative := label2`
        4. Answer: parts(label1) \ parts(label2)
        '''
        def build_question(affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'What visible parts does the {self._get_object_name(instance_label, include_label)} in the image have '
                + f'which {self.inflect.a(contrastive_label)} does not have?'
            )

        def build_answer(parts: list[str], affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'The {self._get_object_name(instance_label, include_label)} in the image has the following visible parts which '
                + f'{self.inflect.a(contrastive_label)} does not: {self._build_part_str(parts)}.'
            )

        question_type = QuestionType.NEGATIVE_WITH_LABEL if include_label else QuestionType.NEGATIVE

        pairs = []
        for instance in tqdm(list(chain.from_iterable(self.instances_by_image_label.values())), desc='Negative QA Pairs'):
            instance_label = instance.image_label

            # The set of possible conditioning labels is that for which parts(instance_label) \ parts(conditioned_label) is nonempty
            possible_conditioning_labels = list(self.minuends_to_subtrahends_with_nonempty_difference[instance_label]) # Copy for shuffling

            if not possible_conditioning_labels:
                logger.warning(f'No possible conditioning labels for instance {instance.image_path} with label {instance_label}. Skipping...')
                continue

            self.rng.shuffle(possible_conditioning_labels)
            n_generated = 0
            curr_index = 0
            while n_generated < self.config.max_questions_per_question_type_per_instance and curr_index < len(possible_conditioning_labels):
                conditioned_label = possible_conditioning_labels[curr_index]
                curr_index += 1

                # Get gt parts
                image_parts = self._get_instance_parts(instance)
                label_parts = self._get_label_parts(conditioned_label)

                comparison = self._split_part_union((image_parts, label_parts))[0]
                gt_parts = comparison.one_minus_two

                if not gt_parts and not self.config.allow_empty_answers:
                    continue

                if self.config.negative_questions_require_nonempty_intersection and not comparison.intersection:
                    continue

                # Build question and answer
                question = build_question(None, conditioned_label, instance.image_label)
                answer = build_answer(gt_parts, None, conditioned_label, instance.image_label)

                # Generate wrong answers
                related_parts = set(self._get_label_parts(instance_label)).union(label_parts) # All parts of instance's label and conditioned label

                try:
                    wrong_answers_parts, answers_operations = self._generate_wrong_answers(gt_parts, related_parts)
                except RuntimeError as e:
                    logger.warning(e)
                    continue

                # Construct final answers and types
                wrong_answers = [build_answer(parts, None, conditioned_label, instance.image_label) for parts in wrong_answers_parts]

                answers = [answer] + wrong_answers
                answer_types = [AnswerType.CORRECT] + [AnswerType.INCORRECT] * len(wrong_answers)
                answer_parts = [gt_parts] + wrong_answers_parts

                pair = QAPair(
                    image_path=instance.image_path,
                    image_label=instance_label,
                    question=question,
                    question_type=question_type,
                    answer_choices=answers,
                    answer_types=answer_types,
                    answer_parts=answer_parts,
                    answer_operations=answers_operations,
                    segmentations=instance.segmentations
                )

                pairs.append(pair)
                n_generated += 1

        return pairs

    def _generate_negative_qa_pairs_batched(self, include_label: bool = False):
        '''
        1. Sample two different object labels, `label1` and `label2`
        2. Sample image from `label1`
        3. Set `alternative := label2`
        4. Answer: parts(label1) \ parts(label2)
        '''

        def build_question(affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'What visible parts does the {self._get_object_name(instance_label, include_label)} in the image have '
                + f'which {self.inflect.a(contrastive_label)} does not have?'
            )

        def build_answer(parts: list[str], affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'The {self._get_object_name(instance_label, include_label)} in the image has the following visible parts which '
                + f'{self.inflect.a(contrastive_label)} does not: {self._build_part_str(parts)}.'
            )

        question_type = QuestionType.NEGATIVE_WITH_LABEL if include_label else QuestionType.NEGATIVE

        # Maintain instance order with an OrderedDict
        instance_pairs = defaultdict(list)  # Maps instances to their QAPairs list

        # Track which instances still need more questions
        remaining_instances = {}

        # Collect all instances
        all_instances = list(chain.from_iterable(self.instances_by_image_label.values()))

        # Initialize retry tracking
        logger.info(f'Initializing all instances for batching...')
        for instance in all_instances:
            instance_label = instance.image_label
            possible_labels = list(self.minuends_to_subtrahends_with_nonempty_difference[instance_label])
            if possible_labels:
                self.rng.shuffle(possible_labels)
                remaining_instances[instance] = possible_labels  # Store available labels

        prog_bar = tqdm(total=len(remaining_instances), desc='Generating Negative QA Pairs')
        while remaining_instances:  # Keep going until all instances are satisfied
            batch_instances = list(islice(remaining_instances, self.config.comparator_batch_size))

            # Prepare batched inputs
            batch_part_pairs = []
            batch_labels = []
            batch_instances_filtered = []  # Instances with conditioning labels to try remaining

            for instance in batch_instances:
                possible_labels = remaining_instances[instance]

                if not possible_labels:  # No more options for this instance
                    del remaining_instances[instance]
                    prog_bar.update(1)
                    continue

                # Pick the next label for this instance
                conditioned_label = possible_labels.pop(0)

                batch_part_pairs.append((self._get_instance_parts(instance), self._get_label_parts(conditioned_label)))
                batch_labels.append(conditioned_label)
                batch_instances_filtered.append(instance)  # Store for processing

            if not batch_part_pairs:
                continue  # Skip empty batch

            # Call `_split_part_union` in batch
            batch_comparisons = self._split_part_union(list_of_part_lists=batch_part_pairs)

            # Iterate over batch results and create Q&A
            for instance, (instance_parts, label_parts), conditioned_label, comparison in zip(
                batch_instances_filtered,
                batch_part_pairs,
                batch_labels,
                batch_comparisons
            ):
                # Extract parts from comparison
                gt_parts = comparison.one_minus_two

                if not gt_parts and not self.config.allow_empty_answers:
                    continue  # Skip, but retry with another label

                if self.config.negative_questions_require_nonempty_intersection and not comparison.intersection:
                    continue  # Skip, but retry with another label

                head_gt_parts = gt_parts
                object_gt_parts = gt_parts
                concept_gt_parts = gt_parts

                # Generate Q&A
                question = build_question(None, conditioned_label, instance.image_label)
                answer = build_answer(head_gt_parts, None, conditioned_label, instance.image_label)

                # Generate wrong answers
                related_parts = set(self._get_label_parts(instance.image_label)).union(label_parts) # All parts of instance's label and conditioned label

                try:
                    # Use object parts to generate wrong answers, as head may be synonym from other concept's parts
                    # Don't add either the object's or concept's GT parts which may collide with the head parts
                    parts_to_exclude_for_addition = set(object_gt_parts).union(concept_gt_parts)
                    wrong_answers_parts, answers_operations = self._generate_wrong_answers(head_gt_parts, related_parts, parts_to_exclude_for_addition)
                except RuntimeError as e:
                    logger.warning(e)
                    continue  # Retry with another label

                # Construct final answers and types
                wrong_answers = [build_answer(parts, None, conditioned_label, instance.image_label) for parts in wrong_answers_parts]
                answers = [answer] + wrong_answers
                answer_types = [AnswerType.CORRECT] + [AnswerType.INCORRECT] * len(wrong_answers)
                answer_parts = [gt_parts, *wrong_answers_parts] # Use object parts as keys for segmentation masks

                pair = QAPair(
                    image_path=instance.image_path,
                    image_label=instance.image_label,
                    question=question,
                    question_type=question_type,
                    answer_choices=answers,
                    answer_types=answer_types,
                    answer_parts=answer_parts,
                    answer_operations=answers_operations,
                    segmentations=instance.segmentations
                )

                instance_pairs[instance].append(pair)  # Store pair for this instance

                # If this instance has generated enough questions, remove it
                if len(instance_pairs[instance]) >= self.config.max_questions_per_question_type_per_instance:
                    del remaining_instances[instance]
                    prog_bar.update(1)

        prog_bar.close()
        # Flatten results while preserving original instance order
        pairs = chain.from_iterable(instance_pairs.values())

        return pairs

    def _generate_difference_qa_pairs(self, include_label: bool = False):
        '''
        1. Sample three labels such that
            - parts(label1) intersect parts(label2) is nonempty
            - parts(label1) \ parts(label3) is nonempty
        2. Select an image of label1
        3. Set `concept := label2`, `ancestors/alternative := label3`
        '''
        def build_question(affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'What visible parts does the {self._get_object_name(instance_label, include_label)} in the image have '
                + f'in common with {self.inflect.a(affirmative_label)} '
                + f'but not with {self.inflect.a(contrastive_label)}?'
            )

        def build_answer(parts: list[str], affirmative_label: str, contrastive_label: str, instance_label: str):
            return (
                f'The {self._get_object_name(instance_label, include_label)} in the image has the following visible parts '
                + f'in common with {self.inflect.a(affirmative_label)} '
                + f'but not with {self.inflect.a(contrastive_label)}: {self._build_part_str(parts)}.'
            )

        question_type = QuestionType.DIFFERENCE_WITH_LABEL if include_label else QuestionType.DIFFERENCE

        # Construct mapping from label to valid doubles (label2, label3) where parts(label1) & parts(label2) and parts(label1) \ parts(label3) are nonempty
        label_to_valid_doubles: defaultdict[str,list[tuple[str,str]]] = defaultdict(list)

        for label1, labels_with_nonempty_intersection_with_label1 in self.labels_with_nonempty_intersection.items():
            for label2 in labels_with_nonempty_intersection_with_label1:
                for label3 in self.minuends_to_subtrahends_with_nonempty_difference[label1]:
                    if label2 != label3: # Don't want to ask "what parts does this object have in common with X but not with X?" where X = label2 = label3
                        label_to_valid_doubles[label1].append((label2, label3))

        pairs = []
        for instance in tqdm(list(chain.from_iterable(self.instances_by_image_label.values())), desc='Difference QA Pairs'):
            instance_label = instance.image_label

            # Construct QAPairs
            label1 = instance_label
            valid_doubles = list(label_to_valid_doubles[label1]) # Copy for shuffling

            if not valid_doubles:
                logger.warning(f'No valid doubles for instance {instance.image_path} with label {instance_label}. Skipping...')
                continue

            self.rng.shuffle(valid_doubles)
            n_generated = 0
            curr_index = 0
            while n_generated < self.config.max_questions_per_question_type_per_instance and curr_index < len(valid_doubles):
                label2, label3 = valid_doubles[curr_index]
                curr_index += 1

                # Get GT parts: (parts(image) & parts(label2)) - parts(label3)
                image_parts = self._get_instance_parts(instance)
                label2_parts = self._get_label_parts(label2)
                label3_parts = self._get_label_parts(label3)

                comparison = self._split_part_union((image_parts, label2_parts))[0]
                comparison = self._split_part_union(([part.part1 for part in comparison.intersection], label3_parts))[0]
                gt_parts = comparison.one_minus_two

                if not gt_parts and not self.config.allow_empty_answers:
                    continue

                # Build question and answer
                question = build_question(label2, label3, instance.image_label)
                answer = build_answer(gt_parts, label2, label3, instance.image_label)

                # Generate wrong answers
                related_parts = set(self._get_label_parts(instance_label)).union(label2_parts).union(label3_parts) # All parts of instance's label and conditioned label

                try:
                    wrong_answers_parts, answers_operations = self._generate_wrong_answers(gt_parts, related_parts)
                except RuntimeError as e:
                    logger.warning(e)
                    continue

                # Construct final answers and types
                wrong_answers = [build_answer(parts, label2, label3, instance.image_label) for parts in wrong_answers_parts]

                answers = [answer] + wrong_answers
                answer_types = [AnswerType.CORRECT] + [AnswerType.INCORRECT] * len(wrong_answers)
                answer_parts = [gt_parts] + wrong_answers_parts

                pair = QAPair(
                    image_path=instance.image_path,
                    image_label=instance_label,
                    question=question,
                    question_type=question_type,
                    answer_choices=answers,
                    answer_types=answer_types,
                    answer_parts=answer_parts,
                    answer_operations=answers_operations,
                    segmentations=instance.segmentations
                )

                pairs.append(pair)
                n_generated += 1

        return pairs

    def _generate_whole_to_part_qa_pairs(self):
        return self._generate_w2p_or_p2w_qa_pairs(QuestionType.WHOLE_TO_PART, include_label_in_parts_question=True)

    def _generate_part_to_whole_qa_pairs(self):
        return self._generate_w2p_or_p2w_qa_pairs(QuestionType.PART_TO_WHOLE, include_label_in_parts_question=False)

    def _generate_w2p_or_p2w_qa_pairs(self, question_type: QuestionType, include_label_in_parts_question: bool = False):
        def build_class_question() -> str:
            return 'What is the object in the image?'

        def build_class_answer(instance_label: str) -> str:
            return f'It is {self.inflect.a(instance_label)}.'

        def build_part_question(instance_label: str) -> str:
            return f'What visible parts does the {self._get_object_name(instance_label, include_label=include_label_in_parts_question)} in the image have?'

        def build_part_answer(parts: list[str], instance_label: str) -> str:
            return f'The {self._get_object_name(instance_label, include_label=include_label_in_parts_question)} in the image has the following visible parts: {self._build_part_str(parts)}.'

        assert question_type in [QuestionType.WHOLE_TO_PART, QuestionType.PART_TO_WHOLE]

        pairs = []
        for instance in tqdm(list(chain.from_iterable(self.instances_by_image_label.values())), desc='Whole to Part QA Pairs'):
            instance_label = instance.image_label

            # Get GT parts
            gt_parts = self._get_instance_parts(instance)
            if not gt_parts and not self.config.allow_empty_answers:
                continue

            # Sample wrong object classes
            wrong_object_labels = self.answer_mutator.mutate_object(instance_label, n_samples=self.config.n_wrong_answers)

            # Add object class questions and answers
            object_question = build_class_question()
            object_answer_choices = [build_class_answer(instance_label)]
            object_answer_types = [AnswerType.CORRECT]
            object_answer_classes = [instance_label]

            for wrong_object_label in wrong_object_labels:
                object_answer_choices.append(build_class_answer(wrong_object_label))
                object_answer_types.append(AnswerType.INCORRECT)
                object_answer_classes.append(wrong_object_label)

            # Add part questions and answers
            part_question = build_part_question(instance_label)
            part_answer_choices = [build_part_answer(gt_parts, instance_label)]
            part_answer_types = [AnswerType.CORRECT]
            answer_parts = [gt_parts]

            try:
                wrong_answers_parts, answers_operations = self._generate_wrong_answers(gt_parts, gt_parts)
            except RuntimeError as e:
                logger.warning(e)
                continue

            part_answer_choices.extend([
                build_part_answer(parts, instance_label)
                for parts in wrong_answers_parts
            ])
            part_answer_types.extend([AnswerType.INCORRECT] * len(wrong_answers_parts))
            answer_parts.extend(wrong_answers_parts)

            pair = QAPair(
                image_path=instance.image_path,
                image_label=instance_label,
                question=part_question,
                question_type=question_type,
                answer_choices=part_answer_choices,
                answer_types=part_answer_types,
                answer_parts=answer_parts,
                answer_operations=answers_operations,
                segmentations=instance.segmentations,
                object_question=object_question,
                object_answer_choices=object_answer_choices,
                object_answer_types=object_answer_types,
                object_answer_classes=object_answer_classes
            )

            pairs.append(pair)

        return pairs

    def _generate_wrong_answers(
        self,
        gt_parts: list[str],
        related_parts: list[str],
        parts_to_exclude_for_addition: list[str] = []
    ) -> tuple[list[list[str]], list[list[AnswerOperation]]]:
        '''
        Generates wrong answers for a given set of ground truth parts and related parts through mutations.
        '''
        answers_parts, answers_operations = [], []
        answers_parts_set = {tuple(sorted(gt_parts))} # Don't generate the correct answer again

        for _ in range(self.config.n_wrong_answers):
            n_tries = 0
            while n_tries < self.config.n_tries_to_generate:
                parts, operations = self.answer_mutator.mutate_parts(gt_parts, related_parts, parts_to_exclude_for_addition=parts_to_exclude_for_addition)
                parts_tuple = tuple(sorted(parts)) # Canonicalize parts for set comparison

                # Ensure that the generated answer is unique and nonempty, if required
                if parts_tuple not in answers_parts_set and (parts or self.config.allow_empty_answers):
                    answers_parts_set.add(parts_tuple)
                    break

                n_tries += 1
                logger.debug(f'Failed to generate answer after {n_tries} tries; {"retrying" if n_tries < self.config.n_tries_to_generate else "giving up"}.')

            if n_tries == self.config.n_tries_to_generate: # Failed to generate answer
                raise RuntimeError(f'Failed to generate answers for gt_parts {gt_parts} and related_parts {related_parts}.')

            if self.config.sort_parts:
                parts = sorted(parts)

            if self.config.shuffle_parts:
                parts = self.rng.shuffle(parts)

            answers_parts.append(parts)
            answers_operations.append(operations)

        return answers_parts, answers_operations

    def _get_object_name(self, label: str, include_label: bool = False):
        return label if include_label else 'object'

    def _split_part_union(
        self,
        part_lists: tuple[list[str], list[str]] = None,
        list_of_part_lists: list[tuple[list[str], list[str]]] = None
    ) -> list[PartSetComparison]:
        '''
        Splits the union of two sets of parts into their differences and intersection
        '''
        return self.part_comparator.compare(part_lists=part_lists, list_of_part_lists=list_of_part_lists)

    def _get_instance_parts(self, instance: PartDatasetInstance) -> list[str]:
        # Should either get the parts of the image label, or get the parts present as indicated by the segmentations
        '''
        Gets the parts present in the image as indicated by the segmentations.
        '''
        return sorted(set(label for label in instance.segmentation_labels))

    def _get_label_parts(self, label: str):
        '''
        Gets the parts corresponding to a label.
        '''
        return self.concept_graph.part_graph[label]

    def _generate_parts(self, concept_name: str) -> list[str]:
        '''
        Generates the parts for a concept (e.g. if it doesn't have any parts).

        Tentatively: provide an LLM with the concept as input, and generate parts.
        '''
        pass

    def _get_siblings(self, concept_name: str) -> list[str]:
        '''
        Gets the siblings of a concept in the concept graph.
        '''
        pass

    def _get_concept_parts(self, concept_name: str) -> list[str]:
        '''
        Gets the parts of a concept in the concept graph.
        '''
        return self.concept_graph.part_graph.get(concept_name, [])

    def _init_data_structures(self):
        '''
        Initializes data structures for the QA generator.
        '''
        # Reorganize dataset instances by label
        self.instances_by_image_label = {}

        logger.info(f'Organizing instances by image label...')
        for instance in self.dataset_descriptor.instances:
            self.instances_by_image_label.setdefault(instance.image_label, []).append(instance)

        logger.info(f'Caching part comparisons...')

        # TODO remove me when not testing
        # from qa_generation.part_comparison.part_comparison import ComparisonStrategy
        # old_strategy = self.part_comparator.config.strategy
        # self.part_comparator.config.strategy = ComparisonStrategy.EXACT

        self._cache_part_comparisons()

        # self.part_comparator.config.strategy = old_strategy

    def _cache_part_comparisons(self):
        '''
        Caches the comparisons between parts of concepts, computing their differences and intersections
        for all pairs of concepts.
        '''
        labels_with_nonempty_intersection = defaultdict(list)
        subtrahends_to_minuends_with_nonempty_difference = defaultdict(list)
        minuends_to_subtrahends_with_nonempty_difference = defaultdict(list)

        sorted_labels = sorted(self.instances_by_image_label)  # Lexicographic order since intersection is symmetric

        total_comparisons = len(sorted_labels) * (len(sorted_labels) - 1) // 2
        prog_bar = tqdm(total=total_comparisons, desc='Caching Part Comparisons')

        for i, label1 in enumerate(sorted_labels):
            labels_with_nonempty_intersection[label1].append(label1)  # Include self-intersection

            batch_pairs = []
            batch_labels = []

            for j in range(i + 1, len(sorted_labels)):
                label2 = sorted_labels[j]
                batch_pairs.append((self._get_label_parts(label1), self._get_label_parts(label2)))
                batch_labels.append((label1, label2))

                # Process in batches
                if len(batch_pairs) >= self.config.comparator_batch_size or j == len(sorted_labels) - 1:
                    batch_comparisons = self._split_part_union(list_of_part_lists=batch_pairs)

                    # Iterate over results and store accordingly
                    for (label1, label2), comparison in zip(batch_labels, batch_comparisons):
                        if comparison.intersection:
                            labels_with_nonempty_intersection[label1].append(label2)

                        if comparison.one_minus_two:
                            subtrahends_to_minuends_with_nonempty_difference[label2].append(label1)
                            minuends_to_subtrahends_with_nonempty_difference[label1].append(label2)

                        if comparison.two_minus_one:
                            subtrahends_to_minuends_with_nonempty_difference[label1].append(label2)
                            minuends_to_subtrahends_with_nonempty_difference[label2].append(label1)

                        prog_bar.update(1)

                    # Reset batch buffers
                    batch_pairs.clear()
                    batch_labels.clear()

        prog_bar.close()

        # Store results
        # labels_with_nonempty_intersection[label1] is a list of labels with nonempty intersection with label1
        self.labels_with_nonempty_intersection = labels_with_nonempty_intersection

        # label_minuends_with_nonempty_difference[label1] is a list of labels label_i such that parts(label_i) \ parts(label1) is nonempty
        # i.e. gets the minuends for which label1 is a subtrahend and the difference is nonempty
        self.subtrahends_to_minuends_with_nonempty_difference = subtrahends_to_minuends_with_nonempty_difference

        # label_subtrahends_with_nonempty_difference[label1] is a list of labels label_i such that parts(label1) \ parts(label_i) is nonempty
        # i.e. gets the subtrahends for which label1 is a minuend and the difference is nonempty
        self.minuends_to_subtrahends_with_nonempty_difference = minuends_to_subtrahends_with_nonempty_difference
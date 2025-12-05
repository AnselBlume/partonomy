from __future__ import annotations
import torch
import json
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict
import logging
from .prompts import LLM_COMPARISON_PROMPT, LLM_EXCLUSION_PROMPT

logger = logging.getLogger(__name__)

class ComparisonStrategy(Enum):
    EXACT = 'exact'
    LLM = 'llm'
    LEMMA = 'lemma'

@dataclass
class PartIntersection:
    head: str
    part1: str
    part2: str

    def to_dict(self) -> dict:
        return {
            'head': self.head,
            'part1': self.part1,
            'part2': self.part2
        }

@dataclass
class PartSetComparison:
    one_minus_two: list[str]
    intersection: list[PartIntersection]
    two_minus_one: list[str]

    def to_dict(self) -> dict:
        return {
            'one_minus_two': self.one_minus_two,
            'intersection': [p.to_dict() for p in self.intersection],
            'two_minus_one': self.two_minus_one
        }

    @staticmethod
    def from_dict(d: dict) -> PartSetComparison:
        return PartSetComparison(
            one_minus_two=d['one_minus_two'],
            intersection=[PartIntersection(**i) for i in d['intersection']],
            two_minus_one=d['two_minus_one']
        )

class PartComparatorCache(TypedDict):
    comparisons: dict[str, PartSetComparison]
    exclusions: dict[str, list[str]]

@dataclass
class PartComparatorConfig:
    strategy: ComparisonStrategy = ComparisonStrategy.EXACT

    cache_comparisons: bool = True

    # LLM comparison config
    llm_model: str = 'microsoft/Phi-4-mini-instruct'
    # llm_model: str = 'microsoft/phi-4'

    llm_temperature: float = .3
    max_tokens: int = 1000
    vllm_gpu_memory_utilization: float = 1.
    vllm_num_gpus: int = 1
    vllm_n_retries: int = 5

class PartComparator:
    def __init__(self, config: PartComparatorConfig = PartComparatorConfig()):
        self.config = config
        self.comparisons_cache = {}
        self.exclusions_cache = {}

        if self.config.strategy == ComparisonStrategy.LLM:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=self.config.llm_model,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                tensor_parallel_size=self.config.vllm_num_gpus
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens
            )

        if self.config.strategy == ComparisonStrategy.LEMMA:
            import spacy

            self.nlp = spacy.load('en_core_web_sm')

    def save_cache(self, cache_file: str):
        comparison_cache_dict = {
            k : c.to_dict()
            for k, c in self.comparisons_cache.items()
        }

        exclusions_cache_dict = {
            k : v
            for k, v in self.exclusions_cache.items()
        }

        cache_dict = {
            'comparisons': comparison_cache_dict,
            'exclusions': exclusions_cache_dict
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_dict, f, indent=4)

    def load_cache(self, cache_file: str):
        self.comparisons_cache = {}
        self.exclusions_cache = {}

        with open(cache_file, 'r') as f:
            cache_dict: PartComparatorCache = json.load(f)

        for k, v in cache_dict['comparisons'].items():
            self.comparisons_cache[k] = PartSetComparison.from_dict(v)

        for k, v in cache_dict['exclusions'].items():
            self.exclusions_cache[k] = v

    def get_exclusions(
        self,
        potentially_excluded_parts: list[str],
        concept_name: str,
        concept_parts: list[str]
    ) -> list[str]:

        pass

    def compare(
        self,
        part_lists: tuple[list[str], list[str]] = None,
        list_of_part_lists: list[tuple[list[str], list[str]]] = None
    ) -> list[PartSetComparison]:
        assert (part_lists is not None) ^ (list_of_part_lists is not None)

        if part_lists:
            list_of_part_lists = [part_lists]

        if self.config.cache_comparisons:
            # Use cache for every pair we have, then batch the rest, compute, and redistribute into the answer
            cached_comparisons = [] # List of comparisons for the pairs we have cached
            uncached_part_pairs = [] # List of pairs we don't have cached
            for p1, p2 in list_of_part_lists:
                part_pair_key = self._get_comparisons_key(p1, p2)

                if part_pair_key in self.comparisons_cache:
                    cached_comparisons.append(self.comparisons_cache[part_pair_key])
                else:
                    cached_comparisons.append(None)
                    uncached_part_pairs.append((p1, p2))

            if uncached_part_pairs:
                list_of_part_lists = uncached_part_pairs # Only compare the pairs we don't have cached
            else:
                return cached_comparisons

        if list_of_part_lists:
            if self.config.strategy == ComparisonStrategy.EXACT:
                comparisons = self._compare_exact(list_of_part_lists)

            elif self.config.strategy == ComparisonStrategy.LLM:
                n_tries = 0
                while True:
                    try:
                        comparisons = self._compare_llm(list_of_part_lists)
                        break
                    except Exception as e:
                        n_tries += 1
                        if n_tries > self.config.vllm_n_retries:
                            logger.warning(f'Caught exception "{e}" after {self.config.vllm_n_retries} retries; falling back to exact comparison')
                            comparisons = self._compare_exact(list_of_part_lists)
                            break
                        else:
                            logger.debug(f'Caught exception "{e}"; retrying {self.config.vllm_n_retries - n_tries}...')
                            torch.cuda.empty_cache()

            elif self.config.strategy == ComparisonStrategy.LEMMA:
                comparisons = self._compare_lemmatized(list_of_part_lists)

            else:
                raise ValueError(f'Invalid comparison strategy: {self.config.strategy}')

        else:
            comparisons = []

        if self.config.cache_comparisons:
            # Fill in the cache for the pairs we computed
            for comparison, part_pair in zip(comparisons, list_of_part_lists):
                self.comparisons_cache[self._get_comparisons_key(*part_pair)] = comparison

            if uncached_part_pairs: # Fill in the comparisons for the pairs we didn't originally have cached
                comparisons_ind = 0
                for i in range(len(cached_comparisons)):
                    if cached_comparisons[i] is None:
                        cached_comparisons[i] = comparisons[comparisons_ind]
                        comparisons_ind += 1

                comparisons = cached_comparisons

        return comparisons

    def _get_comparisons_key(self, parts1: list[str], parts2: list[str]) -> str:
        parts1 = sorted(parts1)
        parts2 = sorted(parts2)

        return f'{parts1}-{parts2}'

    def _get_exclusions_key(self, potentially_excluded_parts: list[str], concept_name: str, concept_parts: list[str]) -> str:
        potentially_excluded_parts = sorted(potentially_excluded_parts)
        concept_parts = sorted(concept_parts)

        return f'{potentially_excluded_parts}-{concept_name}-{concept_parts}'

    def _compare_exact(
        self,
        list_of_part_lists: list[tuple[list[str], list[str]]]
    ) -> list[PartSetComparison]:

        comparisons = []
        for p1, p2 in list_of_part_lists:
            if not isinstance(p1, set) or not isinstance(p2, set):
                p1, p2 = set(p1), set(p2)

            unique_parts1 = sorted(p1 - p2)
            intersection = [PartIntersection(head=p, part1=p, part2=p) for p in sorted(p1 & p2)]
            unique_parts2 = sorted(p2 - p1)

            comparisons.append(PartSetComparison(
                one_minus_two=unique_parts1,
                intersection=intersection,
                two_minus_one=unique_parts2
            ))

        return comparisons

    def _compare_lemmatized(
        self,
        list_of_part_lists: list[tuple[list[str], list[str]]]
    ) -> list[PartSetComparison]:
        comparisons = []

        for parts1, parts2 in list_of_part_lists:
            # Lemmatize all parts
            doc1 = list(self.nlp.pipe(parts1))
            doc2 = list(self.nlp.pipe(parts2))

            lemmas1_map = {p.text: " ".join([t.lemma_ for t in p]) for p in doc1}
            lemmas2_map = {p.text: " ".join([t.lemma_ for t in p]) for p in doc2}

            # Reverse mapping from lemma to original parts (handle possible collisions)
            lemma_to_original1 = {}
            for original, lemma in lemmas1_map.items():
                lemma_to_original1.setdefault(lemma, []).append(original)

            lemma_to_original2 = {}
            for original, lemma in lemmas2_map.items():
                lemma_to_original2.setdefault(lemma, []).append(original)

            # Sets of lemmas
            lemmas1 = set(lemmas1_map.values())
            lemmas2 = set(lemmas2_map.values())

            # Compute intersections and differences
            intersection = []
            for common_lemma in sorted(lemmas1 & lemmas2):
                for orig1 in lemma_to_original1[common_lemma]:
                    for orig2 in lemma_to_original2[common_lemma]:
                        intersection.append(PartIntersection(
                            head=common_lemma,
                            part1=orig1,
                            part2=orig2
                        ))

            unique_lemmas1 = lemmas1 - lemmas2
            unique_lemmas2 = lemmas2 - lemmas1

            one_minus_two = sorted({orig for l in unique_lemmas1 for orig in lemma_to_original1[l]})
            two_minus_one = sorted({orig for l in unique_lemmas2 for orig in lemma_to_original2[l]})

            comparisons.append(PartSetComparison(
                one_minus_two=one_minus_two,
                intersection=sorted(intersection, key=lambda x: x.part1),
                two_minus_one=two_minus_one
            ))

        return comparisons

    def _compare_llm(
        self,
        list_of_part_lists: list[tuple[list[str], list[str]]]
    ) -> list[PartSetComparison]:

        prompts = [
            LLM_COMPARISON_PROMPT.format(
                list1=self._get_parts_str(p1),
                list2=self._get_parts_str(p2)
            )
            for p1, p2 in list_of_part_lists
        ]

        conversations = [
            self._format_llm_conversation(prompt)
            for prompt in prompts
        ]

        outputs = self.llm.chat(conversations, self.sampling_params, use_tqdm=False)
        output_strs = [output.outputs[0].text.strip() for output in outputs]

        comparisons = []
        for part_lists, output_str in zip(list_of_part_lists, output_strs):
            comparisons.append(self._extract_part_comparison_from_llm_output(output_str, part_lists))

        return comparisons

    def _extract_part_comparison_from_llm_output(self, output: str, part_lists: tuple[list[str], list[str]]) -> PartSetComparison:
        unique_parts1: list[str] = []
        has_unique_parts1 = False

        intersection: list[PartIntersection] = []
        has_intersection = False
        is_in_intersection = False

        unique_parts2: list[str] = []
        has_unique_parts2 = False

        curr_list: list[str] | list[PartIntersection] = None

        has_output_started = False
        for line in output.split('\n'):
            if line.startswith('OUTPUT: Difference (parts1 - parts2):'):
                curr_list = unique_parts1
                has_unique_parts1 = True
                is_in_intersection = False
                has_output_started = True
                continue
            elif line.startswith('OUTPUT: Intersection (parts1 & parts2):'):
                curr_list = intersection
                has_intersection = True
                is_in_intersection = True
                has_output_started = True
                continue
            elif line.startswith('OUTPUT: Difference (parts2 - parts1):'):
                curr_list = unique_parts2
                has_unique_parts2 = True
                is_in_intersection = False
                has_output_started = True
                continue
            elif line := line.strip().lower():
                if not has_output_started or line.startswith('none'):
                    continue

                if line.startswith('-'):
                    result = line.split('-')[1].strip() # '- Part1' -> 'part1'

                    if result == 'none': # Catch '- None' case
                        continue

                    if is_in_intersection: # Extract head and parts and construct PartIntersection
                        result = self._extract_part_intersection_from_llm_line(result)

                    curr_list.append(result)

        if not has_unique_parts1 or not has_unique_parts2 or not has_intersection:
            raise RuntimeError(f'Missing output from LLM: {output}')

        # Validate that the parts are present in the part lists
        for part in unique_parts1:
            if part not in part_lists[0]:
                raise ValueError(f'Part {part} not found in part list 1')

        for part in unique_parts2:
            if part not in part_lists[1]:
                raise ValueError(f'Part {part} not found in part list 2')

        for part_intersection in intersection:
            if part_intersection.head not in part_lists[0] and part_intersection.head not in part_lists[1]:
                raise ValueError(f'Head "{part_intersection.head}" not found in corresponding part lists')

            if part_intersection.part1 not in part_lists[0] or part_intersection.part2 not in part_lists[1]:
                # Try swapping and check again
                part_intersection.part1, part_intersection.part2 = part_intersection.part2, part_intersection.part1

                if part_intersection.part1 not in part_lists[0] or part_intersection.part2 not in part_lists[1]:
                    raise ValueError(f'Part "{part_intersection.part1}" or "{part_intersection.part2}" not found in corresponding part list')

        return PartSetComparison(
            one_minus_two=sorted(unique_parts1),
            intersection=sorted(intersection, key=lambda x: x.part1),
            two_minus_one=sorted(unique_parts2)
        )

    def _extract_part_intersection_from_llm_line(self, intersection_line: str) -> PartIntersection:
        '''
        Expected format (lowered):
            HEAD: part1_or_part2 /// PART1: part1 /// PART2: part2
        '''
        head, part1, part2 = intersection_line.split('///')

        head = head.split('head:')[1].strip()
        part1 = part1.split('part1:')[1].strip()
        part2 = part2.split('part2:')[1].strip()

        return PartIntersection(head=head, part1=part1, part2=part2)

    def _get_parts_str(self, parts_list: list[str]) -> str:
        return '\n'.join(f'- {part}' for part in parts_list) if parts_list else 'None'

    def _format_llm_conversation(self, user_prompt: str) -> list[dict]:
        return [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
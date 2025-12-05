'''
This script computes the overlap between correct and incorrect answer choices in QA pairs, using both token-based and part-based strategies.
It calculates the mean, max, and min overlap between the correct answer and distractors, and logs the results for analysis.
Intended for evaluating answer similarity in partonomy QA datasets.
'''
import orjson
import logging
import numpy as np
import coloredlogs
from typing import Literal

logger = logging.getLogger(__name__)


log_every = 100


def token_seq_similarity(ts1: list[str], ts2: str) -> float:
    ts1 = [t.lower() for t in ts1]
    ts2 = [t.lower() for t in ts2]

    return len(set(ts1) & set(ts2)) / len(set(ts1) | set(ts2))

def parse_tokens(answer_choice: str) -> list[str]:
    answer_choice = answer_choice.lower()
    tokens = answer_choice.split()
    tokens = [t.strip(':., ') for t in tokens]

    return tokens

def compute_token_overlap(qa_pairs: list[dict]) -> dict:
    overlaps = []
    max_overlaps = []
    min_overlaps = []
    for i, qa_pair in enumerate(qa_pairs):
        answer_choices = qa_pair['answer_choices']
        correct_answer_tokens = parse_tokens(answer_choices[0])

        if i % log_every == 0:
            logger.info(f'Token Overlap {i} / {len(qa_pairs)}: {answer_choices[0]} -> {correct_answer_tokens}')

        max_overlap = 0
        min_overlap = 1
        for wrong_answer in answer_choices[1:]:
            wrong_answer_tokens = parse_tokens(wrong_answer)
            similarity = token_seq_similarity(correct_answer_tokens, wrong_answer_tokens)
            overlaps.append(similarity)
            max_overlap = max(max_overlap, similarity)
            min_overlap = min(min_overlap, similarity)

        max_overlaps.append(max_overlap)
        min_overlaps.append(min_overlap)

    return np.mean(max_overlaps), np.mean(min_overlaps), np.mean(overlaps)


def compute_part_overlap(qa_pairs: list[dict], parse_strategy: Literal['token', 'part'] = 'token') -> dict:
    def parse_parts(answer_choice: str) -> list[str]:
        # This version of parse_parts keeps each part as a separate string
        answer_choice = answer_choice.lower()
        parts = answer_choice.split(',')
        parts = [p.strip().removeprefix('and ').strip('., ') for p in parts]

        return parts

    parser = parse_parts if parse_strategy == 'part' else parse_tokens

    overlaps = []
    max_overlaps = []
    min_overlaps = []
    for i, qa_pair in enumerate(qa_pairs):
        answer_choices = qa_pair['answer_choices']
        correct_answer = answer_choices[0]
        part_substr = correct_answer.split(':')[1]
        part_tokens = parser(part_substr)

        if i % log_every == 0:
            logger.info(f'Part Tokens {i} / {len(qa_pairs)}: {correct_answer} -> {part_tokens}')

        max_overlap = 0
        min_overlap = 1
        for wrong_answer in answer_choices[1:]:
            wrong_answer_parts = parser(wrong_answer)
            similarity = token_seq_similarity(part_tokens, wrong_answer_parts)
            overlaps.append(similarity)
            max_overlap = max(max_overlap, similarity)
            min_overlap = min(min_overlap, similarity)

        max_overlaps.append(max_overlap)
        min_overlaps.append(min_overlap)

    return np.mean(max_overlaps), np.mean(min_overlaps), np.mean(overlaps)

if __name__ == '__main__':
    coloredlogs.install(level='INFO')

    qa_pairs_path = '/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partonomy/partonomy_qa_pairs_val.json'

    with open(qa_pairs_path, 'r') as f:
        qa_pairs = orjson.loads(f.read())

    token_max_overlap, token_min_overlap, token_overlap = compute_token_overlap(qa_pairs)
    part_max_overlap, part_min_overlap, part_overlap = compute_part_overlap(qa_pairs, parse_strategy='part')
    part_max_overlap_token, part_min_overlap_token, part_overlap_token = compute_part_overlap(qa_pairs, parse_strategy='token')

    logger.info(f'Token Overlap: {token_overlap}')
    logger.info(f'Token Max Overlap: {token_max_overlap}')
    logger.info(f'Token Min Overlap: {token_min_overlap}')
    print('')

    logger.info(f'Part Overlap (part): {part_overlap}')
    logger.info(f'Part Max Overlap (part): {part_max_overlap}')
    logger.info(f'Part Min Overlap (part): {part_min_overlap}')
    print('')

    logger.info(f'Part Overlap (token): {part_overlap_token}')
    logger.info(f'Part Max Overlap (token): {part_max_overlap_token}')
    logger.info(f'Part Min Overlap (token): {part_min_overlap_token}')
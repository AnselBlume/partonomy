from dataclasses import dataclass
from typing import Tuple, Union
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment  # for Hungarian matching
from typing import Literal
from enum import Enum
from .evaluator import Evaluator
import logging, coloredlogs

logger = logging.getLogger(__name__)

class MatchingStrategy(Enum):
    EXACT = 'exact'
    EMBEDDING = 'embedding'
    LLM = 'llm'

LLM_PROMPT = (
    'Is the phrase "{predicted}" synonymous with or an instance of any of the following phrases?\n'
    '{gt_phrases}\n'
    'Answer "yes" if the predicted phrase is present or synonymous with any of the reference phrases. Otherwise, just answer "no".\n'
    'Do not say anything else other than "yes" or "no".'
)
@dataclass
class PartTextEvaluatorConfig:
    # Match method to use
    match_method: MatchingStrategy = MatchingStrategy.EXACT

    # Embedding match config
    embedding_model: str = 'google-bert/bert-base-uncased'
    cosine_threshold: float = 0.75
    embedding_device: Union[str, torch.device] = 'cuda'
    embedding_matching_algorithm: Literal['greedy', 'hungarian'] = 'hungarian'

    # LLM match config
    llm_model: str = 'microsoft/Phi-4-mini-instruct'
    llm_temperature: float = 0
    max_tokens: int = 100
    vllm_gpu_memory_utilization: float = 1.

class PartTextEvaluator(Evaluator):
    DEFAULT_METRIC_GROUP_NAME = 'part_text'

    def __init__(self, config: PartTextEvaluatorConfig = PartTextEvaluatorConfig(), **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.true_positives = 0
        self.recall_denominator = 0
        self.precision_denominator = 0

        # For embedding or LLM matching, load the tokenizer and embedding model.
        if self.config.match_method == MatchingStrategy.EMBEDDING:
            from transformers import AutoModel, AutoTokenizer # Import here to lazy load/install

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)
            self.embedding_model = AutoModel.from_pretrained(self.config.embedding_model).to(self.config.embedding_device)

        # For LLM-based matching, initialize the LLM.
        if self.config.match_method == MatchingStrategy.LLM:
            from vllm import LLM, SamplingParams # Import here to lazy load/install

            self.llm = LLM(model=self.config.llm_model, gpu_memory_utilization=self.config.vllm_gpu_memory_utilization)
            self.sampling_params = SamplingParams(
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens
            )

    def _compute_embeddings(self, strs: list[str]) -> torch.Tensor:
        '''
        Compute embeddings for a list of length n of strings

        Args:
            texts: List of strings to embed.

        Returns:
            torch.Tensor: Embeddings of the input texts of shape (n, d).
        '''
        tokens = self.tokenizer(
            strs,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.config.embedding_device)

        with torch.no_grad():
            output = self.embedding_model(**tokens)

        return output.pooler_output # (n, d)

    def _greedy_match(self, sim_matrix: np.ndarray, threshold: float) -> int:
        '''
        Greedily pair entries in the similarity matrix (each prediction and reference can be used only once)
        and count pairs whose similarity exceeds the threshold.
        '''
        candidate_dict = {
            (i, j): sim_matrix[i, j]
            for i in range(sim_matrix.shape[0])
            for j in range(sim_matrix.shape[1])
        }
        sorted_candidates = sorted(candidate_dict.items(), key=lambda item: item[1], reverse=True)
        used_rows = set()
        used_cols = set()
        match_count = 0

        for (i, j), sim in sorted_candidates:
            if i in used_rows or j in used_cols:
                continue
            if sim >= threshold:
                match_count += 1
                used_rows.add(i)
                used_cols.add(j)

        return match_count

    def evaluate_exact(self, predictions: list[str], references: list[str]) -> Tuple[float, float, float]:
        '''
        Computes precision, recall, and F1 score by performing a normalized set membership check.
        '''
        normalized_refs = {ref.strip().lower() for ref in references}
        correct = sum(1 for pred in predictions if pred.strip().lower() in normalized_refs)
        precision = correct / len(predictions) if predictions else 0
        recall = correct / len(references) if references else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return correct, precision, recall, f1

    @torch.inference_mode()
    def evaluate_embedding(self, predictions: list[str], references: list[str]) -> Tuple[float, float, float]:
        '''
        Computes precision, recall, and F1 score based on pairwise cosine similarity between
        embeddings of predictions and ground truths.
        '''
        # Embed
        pred_embs = self._compute_embeddings(predictions)
        ref_embs = self._compute_embeddings(references)

        # Compute similarity matrix
        pred_embs_norm = pred_embs / pred_embs.norm(dim=1, keepdim=True)
        ref_embs_norm = ref_embs / ref_embs.norm(dim=1, keepdim=True)
        sim_matrix = (pred_embs_norm @ ref_embs_norm.T).cpu().numpy()

        # Match
        if self.config.embedding_matching_algorithm == 'hungarian':
            row_inds, col_inds = linear_sum_assignment(sim_matrix, maximize=True)
            matches = sum(
                1 for i, j in zip(row_inds, col_inds)
                if sim_matrix[i, j] >= self.config.cosine_threshold
            )
        else:
            matches = self._greedy_match(sim_matrix, self.config.cosine_threshold)

        precision = matches / len(predictions) if predictions else 0
        recall = matches / len(references) if references else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return matches, precision, recall, f1

    @torch.inference_mode()
    def evaluate_llm(self, predictions: list[str], references: list[str]) -> Tuple[float, float, float]:
        '''
        Computes precision, recall, and F1 score using an LLM.
        Each predicted phrase is sent as a separate prompt to the LLM, and the answer is processed
        to determine if the prediction matches any of the ground truth phrases.
        '''
        # Construct prompts
        gt_str = '\n'.join(f'- {gt}' for gt in references)

        prompts = [
            LLM_PROMPT.format(predicted=pred, gt_phrases=gt_str)
            for pred in predictions
        ]

        conversations = [
            self._format_llm_conversation(prompt)
            for prompt in prompts
        ]

        outputs = self.llm.chat(conversations, self.sampling_params)

        # Count matches
        matches = 0
        for output in outputs:
            response = output.outputs[0].text.strip().lower()
            matches += 1 if 'yes' in response else 0

        precision = matches / len(predictions) if predictions else 0
        recall = matches / len(references) if references else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return matches, precision, recall, f1

    def update(self, predictions: list[str], references: list[str]) -> Tuple[float, float, float]:
        '''
        A single entry point for evaluation that delegates to the appropriate method.
        '''
        method = self.config.match_method

        if method == MatchingStrategy.EXACT:
            matches, precision, recall, f1 = self.evaluate_exact(predictions, references)
        elif method == MatchingStrategy.EMBEDDING:
            matches, precision, recall, f1 = self.evaluate_embedding(predictions, references)
        elif method == MatchingStrategy.LLM:
            matches, precision, recall, f1 = self.evaluate_llm(predictions, references)
        else:
            raise NotImplementedError(f'Method "{method}" not implemented.')

        self.true_positives += matches
        self.precision_denominator += len(predictions)
        self.recall_denominator += len(references)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def summarize(self) -> dict[str, float]:
        '''
        Summarizes the evaluation results.
        '''
        precision = self.true_positives / self.precision_denominator if self.precision_denominator > 0 else 0
        recall = self.true_positives / self.recall_denominator if self.recall_denominator > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

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

# Example usage:
if __name__ == '__main__':
    coloredlogs.install(level='INFO')

    preds_and_refs = (
        (
            ['cat on mat', 'dog in yard', 'bird flying'],
            ['cat on mat', 'dog in yard', 'bird flying'],
        ),
        (
            ['cat on mat', 'dog in yard', 'bird flying'],
            ['cat sitting on mat', 'dog playing in the yard', 'a bird is flying']
        ),
        (
            ['cat on mat', 'dog in yard', 'bird flying'],
            ['cat in washer', 'dog in dryer', 'bird sitting']
        ),
    )
    for predictions, references in preds_and_refs:
        logger.info(f'Predictions: {predictions}')
        logger.info(f'References: {references}')

        # Exact matching evaluation.
        config = PartTextEvaluatorConfig(match_method=MatchingStrategy.EXACT)
        evaluator = PartTextEvaluator(config)
        precision, recall, f1 = evaluator.update(predictions, references)
        logger.info('Exact match -> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(precision, recall, f1))

        # Embedding-based matching evaluation (using greedy matching here).
        config.match_method = MatchingStrategy.EMBEDDING
        config.embedding_matching_algorithm = 'greedy'
        evaluator = PartTextEvaluator(config)
        precision, recall, f1 = evaluator.update(predictions, references)
        logger.info('Embedding (greedy) -> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(precision, recall, f1))

        # LLM-based matching evaluation.
        config.match_method = MatchingStrategy.LLM
        evaluator = PartTextEvaluator(config)
        precision, recall, f1 = evaluator.update(predictions, references)
        logger.info('LLM match -> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(precision, recall, f1))
from dataclasses import dataclass, asdict
from typing import Any
from .evaluators import Evaluator
from .rle_dict import RLEDict
from qa_generation import QuestionType

@dataclass
class Prediction:
    image_path: str = None

    question_type: QuestionType = None
    questions: list[str] = None

    parts_answer_choices: list[str] = None

    gt_parts_answer: str = None
    predicted_parts_answer: str = None

    gt_parts: list[str] = None
    predicted_parts: list[str] = None

    gt_masks: list[RLEDict] = None
    predicted_masks: list[RLEDict] = None
    mask_confidences: list[float] = None

    # Object question; only used in WHOLE_TO_PART and PART_TO_WHOLE questions
    object_answer_choices: list[str] = None

    gt_object_answer: str = None # The answer choice itself
    predicted_object_answer: str = None

    gt_object: str = None # Class of the object
    predicted_object: str = None

    metrics: dict[str, Any] = None

    def to_dict(self):
        d = asdict(self)
        if self.question_type is not None:
            d['question_type'] = self.question_type.value
        return d

    @staticmethod
    def from_dict(data: dict):
        if data['question_type'] is not None:
            data['question_type'] = QuestionType(data['question_type'])

        return Prediction(**data)

class Predictions:
    def __init__(self):
        self.predictions: list[Prediction] = []

    def add_prediction(self, prediction: Prediction):
        self.predictions.append(prediction)

    def summarize_evaluators(self, evaluators: list[Evaluator]):
        metrics = {}

        for evaluator in evaluators:
            metric_group_name = evaluator.metric_group_name

            if metric_group_name in metrics:
                raise ValueError(f'Duplicate metric group name: {metric_group_name}')

            metrics[metric_group_name] = evaluator.summarize()

        return metrics

    def to_dict(self, evaluators: list[Evaluator] = None):

        return {
            'metrics': self.summarize_evaluators(evaluators),
            'predictions': [prediction.to_dict() for prediction in self.predictions]
        }
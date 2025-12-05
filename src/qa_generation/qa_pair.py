from dataclasses import dataclass
from typing import Any, Iterable
from .answer import AnswerOperation, AnswerType
from qa_generation.question_type import QuestionType

@dataclass
class QAPair:
    image_path: str = None
    image_label: str = None
    question: str = None # Part-based question
    question_type: QuestionType = None
    answer_choices: list[str] = None
    answer_types: list[AnswerType] = None
    answer_parts: list[list[str]] = None
    answer_operations: list[list[AnswerOperation]] = None
    segmentations: list[Any] = None

    # This block is populated only for whole-to-part and part-to-whole questions
    object_question: str = None
    object_answer_choices: list[str] = None
    object_answer_types: list[AnswerType] = None
    object_answer_classes: list[str] = None

    def to_dict(self, exclude_keys: Iterable[str] = ['answer_operations']):
        answer_operations = [
            [answer_operation.to_dict() for answer_operation in answer_operations]
            for answer_operations in self.answer_operations
        ] if self.answer_operations else None

        ret_dict = {
            'image_path': self.image_path,
            'image_label': self.image_label,
            'question': self.question,
            'question_type': self.question_type.value,
            'answer_choices': self.answer_choices,
            'answer_types': [answer_type.value for answer_type in self.answer_types],
            'answer_parts': self.answer_parts,
            'answer_operations': answer_operations,
            'segmentations': self.segmentations,
            'object_question': self.object_question,
            'object_answer_choices': self.object_answer_choices,
            'object_answer_types': [answer_type.value for answer_type in self.object_answer_types] if self.object_answer_types else None,
            'object_answer_classes': self.object_answer_classes
        }

        for key in exclude_keys:
            ret_dict.pop(key, None)

        return ret_dict

    @staticmethod
    def from_dict(d: dict) -> 'QAPair':
        d['question_type'] = QuestionType(d['question_type'])
        d['answer_types'] = [AnswerType(answer_type) for answer_type in d['answer_types']]

        if d.get('object_answer_types', None):
            d['object_answer_types'] = [
                AnswerType(answer_type) for answer_type in d['object_answer_types']
            ]

        if d.get('answer_operations', None):
            d['answer_operations'] = [
                [AnswerOperation.from_dict(answer_operation) for answer_operation in answer_operations]
                for answer_operations in d['answer_operations']
            ]

        return QAPair(**d)
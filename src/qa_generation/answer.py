from enum import Enum

class AnswerType(Enum):
    CORRECT = 'correct'
    INCORRECT = 'incorrect'

class AnswerOperationName(Enum):
    ADD = 'add'
    DELETE = 'delete'
    REPLACE = 'replace'

class AnswerOperation:
    def __init__(self, operation_name: AnswerOperationName, *values):
        self.operation = operation_name
        self.values = list(values) # From a tuple

    def to_dict(self):
        return {
            'operation': self.operation.value,
            'values': self.values
        }

    @staticmethod
    def from_dict(d: dict):
        return AnswerOperation(AnswerOperationName(d['operation']), *d['values'])
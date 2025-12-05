from .evaluator import Evaluator

class MCTextEvaluator(Evaluator):
    DEFAULT_METRIC_GROUP_NAME = 'mc_text'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.correct = 0
        self.total = 0

    @property
    def accuracy(self):
        return self.correct / self.total if self.total > 0 else 0

    def update(self, predicted_ind: int, target_ind: int):
        self.total += 1

        is_correct = predicted_ind == target_ind
        self.correct += int(is_correct)

        return {
            'is_correct': is_correct
        }

    def summarize(self):
        return {
            'accuracy': self.accuracy,
            'correct': self.correct,
            'total': self.total
        }
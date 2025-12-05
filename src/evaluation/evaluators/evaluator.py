class Evaluator:
    DEFAULT_METRIC_GROUP_NAME: str = None

    def __init__(self, metric_group_name: str = None, **kwargs):
        self.metric_group_name = metric_group_name if metric_group_name else self.DEFAULT_METRIC_GROUP_NAME

        if not self.metric_group_name:
            raise ValueError('Evaluator subclasses must define a DEFAULT_METRIC_GROUP_NAME')

    def update(self, prediction, target):
        raise NotImplementedError('Evaluator subclasses must implement the update method')

    def summarize(self) -> dict:
        raise NotImplementedError('Evaluator subclasses must implement the summarize method')
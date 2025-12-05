from enum import Enum

class QuestionType(Enum):
    IDENTIFICATION = 'identification'
    IDENTIFICATION_WITH_LABEL = 'identification_with_label'

    POSITIVE = 'positive'
    POSITIVE_WITH_LABEL = 'positive_with_label'

    NEGATIVE = 'negative'
    NEGATIVE_WITH_LABEL = 'negative_with_label'

    DIFFERENCE = 'difference'
    DIFFERENCE_WITH_LABEL = 'difference_with_label'

    WHOLE_TO_PART = 'whole_to_part'
    PART_TO_WHOLE = 'part_to_whole'

    # COUNTERFACTUAL = 'counterfactual'
    # COMPARATIVE = 'comparative'
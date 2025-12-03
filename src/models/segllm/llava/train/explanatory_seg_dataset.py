import orjson
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, IntTensor
from collections import defaultdict
from copy import deepcopy

from question_type import QuestionType
from typing import Literal
from dataclasses import dataclass
import pycocotools.mask as mask_utils
import logging

logger = logging.getLogger(__name__)

ConversationType = Literal['correct_answer', 'incorrect_answer']
ConversationQuestionType = Literal['part_question', 'object_question']
AnswerType = Literal ['correct', 'incorrect']
from PIL import Image

@dataclass
class ExplanatorySegInstance:
    img_path: str = None
    img_label: str = None

    mask_dicts: dict[str,Tensor] = None # Dict with seg_labels as keys and mask_tensors with shape (height, width) as values
    masks: Tensor = None # Stacked segmentation masks of shape (num_parts, height, width)

    label_mask: IntTensor = None # Tensor representing a ground truth mask with dims (height, width); used to extract dimensions
    resized_img_dims: torch.Size[2] = None # (height, width) of the resized image in SAM.

    is_inference: bool = None

    questions: list[str] = None # Sequence of questions, e.g. [part_question], [part_question, object_question], or [object_question, part_question]
    question_type: QuestionType = None

    conversations: list[list[dict]] = None # List of conversation prompts. Contains part question choices and potentially object question choices
    conversation_types: list[ConversationType] = None # Whether conversation is a question, correct answer, or incorrect answer
    conversation_question_types: list[ConversationQuestionType] = None # Whether conversation is a part question or object question
    train_conversation: str = None # The conversation to use for autoregressive training

    part_answer_choices: list[str] = None
    part_answer_types: list[AnswerType] = None
    answer_parts: list[list[str]] = None # Same length as number of answer choices with ConversationQuestionType == 'part_question'

    object_answer_choices: list[str] = None
    object_answer_types: list[AnswerType] = None
    answer_objects: list[str] = None # Same length as number of answer choices with ConversationQuestionType == 'object_question

    def __post_init__(self):
        if self.answer_objects is not None: # Correct object answer is the image label
            assert self.answer_objects[self.object_answer_types.index('correct')] == self.img_label

class ExplanatorySegDataset(torch.utils.data.Dataset):
    '''
    A dataset returning ExplanatorySegInstances for a given dataset (e.g. Partonomy-{Core, PACO-LVIS, Pascal-Part, PartImagenet})
    and question type (e.g. positive, negative, difference, whole_to_part, part_to_whole).
    '''
    def __init__(
        self,
        dataset_path: str,
        question_type: QuestionType,
        test_with_gt_object: bool = False,
        output_question_prompt_for_generation: bool = False,
        sample_one_question_per_image: bool = False,
        random_seed: int = 42
    ):
        '''
        Args:
            dataset_path (str): Path to the JSON file containing the dataset instances.
            tokenizer: Tokenizer to process text input.
            vision_tower (str): Identifier (e.g., model name) for the CLIP image processor.
            image_size (int): Size for resizing images for the SAM branch.
            transform: Optional; a transformation with an `apply_image` method. Defaults to ResizeLongestSide.
            question_type (QuestionType): The type of question to filter the dataset by.
            sample_one_question_per_image (bool): Whether to sample one question per image to balance the dataset.
            random_seed (int): The random seed to use for sampling.
        '''

        with open(dataset_path, "rb") as f:
            temp_dataset = orjson.loads(f.read())
            # list of QAPair instances
            self.dataset = [instance for instance in temp_dataset if instance['question_type'] == question_type.value]

        self.rng = np.random.RandomState(random_seed)
        self.question_type = question_type
        self.test_with_gt_object = test_with_gt_object
        self.sample_one_question_per_image = sample_one_question_per_image

        if self.sample_one_question_per_image:
            self._restrict_dataset_to_one_question_per_image()

    def _restrict_dataset_to_one_question_per_image(self):
        qa_pairs_by_image = defaultdict(list)

        for instance in self.dataset:
            qa_pairs_by_image[instance['image_path']].append(instance)

        filtered_dataset = []
        for image_path, qa_pairs in qa_pairs_by_image.items():
            sampled_qa_pair = self.rng.choice(qa_pairs)
            filtered_dataset.append(sampled_qa_pair)

        self.dataset = filtered_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ExplanatorySegInstance:
        instance = self.dataset[idx]
        image_path = instance['image_path']
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size

        # text input construction
        image_label: str = instance['image_label']
        answer_types: list[str] = instance['answer_types']
        question: str = instance['question']
        answer_choices: list[str] = instance['answer_choices']
        answer_parts: list[list[str]] = instance['answer_parts']

        if len(answer_parts) == 0:
            raise ValueError("Ground-truth parts ('gt_parts') not found in the instance.")

        gt_idx = answer_types.index('correct') # index of the ground-truth answer choice

        conversations = []
        conversation_types = []
        conversation_question_types = []

        if self.test_with_gt_object:
            question = 'What is this object?'
            questions = [question]

            answer = f'It is a {image_label} [SEG]'
            conversation_types.append('correct_answer')
            conversation_question_types.append('object_question')

            answer_parts = [['full_image']]
            answer_types = ['correct']
            answer_objects = [image_label]

            # No need to set answer_choices since they're not used here (conversations are constructed manually)

        elif self.question_type.value in [
            'identification',
            'identification_with_label',
            'positive',
            'positive_with_label',
            'negative',
            'negative_with_label',
            'difference',
            'difference_with_label']:  # multiple choice question

            # Generate question
            questions = [question]

            # construct a response format equivalent to one in `all_data_mix_train/mr_paco_train.json`

            # Generate answer variants from answer_parts
            for i, (_, parts) in enumerate(zip(answer_choices, answer_parts)): # Iterating over conversations
                gt_part_str = "The object in the image has the following visible parts: "

                for part_idx, part_lbl in enumerate(parts): # Over the coversation's parts
                    if part_idx == 0:
                        gt_part_str += f"{part_lbl} [MASK-DECODE:none:{part_lbl}]"
                    else:
                        gt_part_str += f", {part_lbl} [MASK-DECODE:{parts[part_idx - 1]}:{part_lbl}]"

                conversation_types.append('incorrect_answer' if i != gt_idx else 'correct_answer')
                conversation_question_types.append('part_question')
                conversations.append(
                    [
                        {"from": "human", "value": f"[IMAGE256:{image_path}] {questions[0]}"},
                        {"from": "gpt", "value": gt_part_str + "."},
                    ]
                )

            # Question has no object fields
            object_answer_choices = None
            object_answer_types = None
            answer_objects = None

        elif self.question_type.value == "whole_to_part":  # no ground truth provided - open-ended generation
            '''
            'whole_to_part':
                1. Model predicts the object label and then the parts.
                - e.g., i) airplanes-attack, ii) airplanes--agriculture (gt) ....
                - e.g., i) parts_attack, ii) parts_agriculture (gt) ...
                2. Model is given the object label and then predicts the parts <- 'question' will contain the object label
                3. (tentative) free-form generation of parts and object label.

            'part_to_whole'
                1. Model predicts the parts and then the object label.
                2. Model is given the parts and then predicts the object label.
            '''

            # Generate object question/answers
            object_question: str = instance['object_question']
            object_answer_choices: list[str] = instance['object_answer_choices']
            object_answer_types: list[AnswerType] = instance['object_answer_types']
            answer_objects: list[str] = instance['object_answer_classes']

            questions = [object_question] # The object question

            gt_obj_idx = object_answer_types.index('correct')

            for i, answer in enumerate(object_answer_choices):
                conversations.append(
                    [
                        {"from": "human", "value": f"[IMAGE256:{image_path}] {questions[-1]}"},
                        {"from": "gpt", "value": answer}
                    ]
                )
                conversation_types.append('incorrect_answer' if i != gt_obj_idx else 'correct_answer')
                conversation_question_types.append('object_question')

            # Generate part question/answers
            questions.append(question)

            # construct a response format equivalent to one in `all_data_mix_train/mr_paco_train.json`

            base_conversation: list[dict] = deepcopy(conversations[conversation_types.index('correct_answer')])
            # Generate answer variants from answer_parts
            for i, (_, parts) in enumerate(zip(answer_choices, answer_parts)):
                conversation = deepcopy(base_conversation)

                gt_part_str = "The object in the image has the following visible parts: "

                for part_idx, part_lbl in enumerate(parts):
                    if part_idx == 0:
                        gt_part_str += f"{part_lbl} [MASK-DECODE:none:{part_lbl}]"
                    else:
                        gt_part_str += f", {part_lbl} [MASK-DECODE:{parts[part_idx - 1]}:{part_lbl}]"

                conversation.extend([
                    {'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': gt_part_str + '.'}
                ])
                conversations.append(conversation)

                conversation_types.append('incorrect_answer' if i != gt_idx else 'correct_answer')
                conversation_question_types.append('part_question')

        elif self.question_type.value == "part_to_whole":  # no ground truth provided - open-ended generation

            # Generate part question/answers
            questions: str = [question]

            # construct a response format equivalent to one in `all_data_mix_train/mr_paco_train.json`

            # e.g., "The object in the image has the following visible parts: part1 [SEG], part2 [SEG], part3 [SEG]."

            # Generate answer variants from answer_parts
            for i, (_, parts) in enumerate(zip(answer_choices, answer_parts)):
                gt_part_str = "The object in the image has the following visible parts: "

                for part_idx, part_lbl in enumerate(parts):
                    if part_idx == 0:
                        gt_part_str += f"{part_lbl} [MASK-DECODE:none:{part_lbl}]"
                    else:
                        gt_part_str += f", {part_lbl} [MASK-DECODE:{parts[part_idx - 1]}:{part_lbl}]"

                conversation_types.append('incorrect_answer' if i != gt_idx else 'correct_answer')
                conversation_question_types.append('part_question')

                conversations.append([
                    {"from": "human", "value": f"[IMAGE256:{image_path}] {questions[-1]}"},
                    {"from": "gpt", "value": gt_part_str + "."}
                ])

            # Generate object question/answers
            object_question: str = instance['object_question']
            object_answer_choices: list[str] = instance['object_answer_choices']
            object_answer_types: list[AnswerType] = instance['object_answer_types']
            answer_objects: list[str] = instance['object_answer_classes']

            questions.append(object_question) # The object question

            gt_obj_idx = object_answer_types.index('correct')

            base_conversation: list[dict] = deepcopy(conversations[conversation_types.index('correct_answer')])
            for i, answer in enumerate(object_answer_choices):
                conversation = deepcopy(base_conversation)
                conversation.extend([
                    {'from': 'human', 'value': object_question},
                    {'from': 'gpt', 'value': answer}
                ])

                conversations.append(conversation)
                conversation_types.append('incorrect_answer' if i != gt_obj_idx else 'correct_answer')
                conversation_question_types.append('object_question')
        else:
            raise ValueError(f"Invalid question type: {self.question_type}")

        # TODO: Map each mask to their corresponding part label for "correct" answer parts

        # Handle segmentations
        masks = defaultdict(list)  # {seg_label: [mask1, mask2, ...]}
        mask_list = []

        if self.test_with_gt_object:
            # Use a placeholder mask of all ones
            mask = np.ones((img_h, img_w))
            masks["full_image"].append(mask)
            mask_list.append(mask)

        else:
            segmentations = instance.get("segmentations", {})

            # load the segmentation masks along with images and
            # make sure the masks (e.g., polygon or RLE) are according to the same format as in SemSegDataset
            for seg_label, seg_anns in segmentations.items():
                for ann in seg_anns:
                    if "size" not in ann:
                        raise ValueError("Size not found in the segmentation annotation. Both RLE and Polygon loader requires image dim. (height and width).")
                    if "counts" in ann:
                        mask = mask_utils.decode(ann)

                    mask_h, mask_w = ann['size']

                    # TODO: Skip mismatching masks in case mask_dims != img_dims

                    if (mask_h, mask_w) != (img_h, img_w):
                        logger.warning(f'Mask dimensions {(mask_h, mask_w)} differ from images dimensions {(img_h, img_w)} with path {image_path}')

                        if mask_h < img_h or mask_w < img_w:  # nearest-neighbor interpolation to match the mask dim. to image dim.
                            '''
                                https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
                                cv2.INTER_LINEAR to enlarge
                                cv2.INTER_AREA to shrink the image
                            '''
                            interpolation_type = cv2.INTER_LINEAR
                        else:
                            interpolation_type = cv2.INTER_AREA  # or use cv2.INTER_NEAREST
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=interpolation_type)

                    masks[seg_label].append(mask)
                    mask_list.append(mask)

        # Union masks under the same part name
        masks_dict = {
            seg_label : torch.from_numpy(np.stack(seg_masks, axis=0).any(axis=0).astype(np.float32))
            for seg_label, seg_masks in masks.items()
        }

        # masks_stacked - (num_parts, height, width)
        gt_answer_parts = answer_parts[answer_types.index('correct')]
        masks_stacked = np.stack([masks_dict[p] for p in gt_answer_parts]) # In the same order as the labels        masks_tensor = torch.from_numpy(masks_stacked)
        masks_tensor = torch.from_numpy(masks_stacked)

        # NOTE: only for paco_lvis and pascal_part - label isn't needed for partonomy (it's only needs a binary 'mask')
        # label = torch.ones(masks_tensor.shape[1], masks_tensor.shape[2]) * self.ignore_label

        # inference = True

        # Check number of seg tokens matches number of ground truth masks
        # if instance['question_type'] == QuestionType.PART_TO_WHOLE.value:
        #     gt_cqtype = 'object_question'
        #     gt_answer_types = object_answer_types
        # else:
        #     gt_cqtype = 'part_question'
        #     gt_answer_types = answer_types

        # conv_subtype = [
        #     c
        #     for c, cqtype in zip(conversations, conversation_question_types)
        #     if cqtype == gt_cqtype
        # ]

        # train_conversation = conv_subtype[gt_answer_types.index('correct')]

        # assert len(masks_tensor) == train_conversation.count('[SEG]')

        return ExplanatorySegInstance(
            img_path=image_path,
            img_label=image_label,
            conversations=conversations,            # list of conversation prompts.
            mask_dicts=masks_dict,  # dict with seg_labels as keys and mask_tensors as values
            masks=masks_tensor,          # segmentation masks
            label_mask=None,
            resized_img_dims=None,              # (height, width) of the resized image in SAM.
            questions=questions,
            question_type=self.question_type,
            part_answer_choices=answer_choices,
            part_answer_types=answer_types,
            answer_parts=answer_parts,
            object_answer_choices=object_answer_choices,
            object_answer_types=object_answer_types,
            answer_objects=answer_objects,
            conversation_types=conversation_types,
            conversation_question_types=conversation_question_types,
            train_conversation=None
        )

if __name__ == "__main__":
    # test the dataset with a random instance
    dataset = ExplanatorySegDataset(
        dataset_path='/shared/nas2/blume5/sp25/partonomy/partonomy_private/data/partonomy_descriptors/partimagenet/partimagenet_qa_pairs_train.json',
        question_type=QuestionType.WHOLE_TO_PART,  # IDENTIFICATION_WITH_LABEL,
        random_seed=42
    )
    instance = dataset[0]
    print(instance)
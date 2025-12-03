import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, IntTensor
from collections import defaultdict
from transformers import CLIPImageProcessor

from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST, load_rle, load_polygon, open_image
from utils.question_type import QuestionType
from typing import Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

ConversationType = Literal['question', 'correct_answer', 'incorrect_answer']
ConversationQuestionType = Literal['part_question', 'object_question']
AnswerType = Literal ['correct', 'incorrect']

@dataclass
class ExplanatorySegInstance:
    img_path: str = None
    img_label: str = None

    sam_img_input: Tensor = None # Image tensor for SAM
    clip_img_input: Tensor = None # Image preprocessed for CLIP.

    mask_dicts: dict[str,Tensor] = None # Dict with seg_labels as keys and mask_tensors with shape (height, width) as values
    masks: Tensor = None # Stacked segmentation masks of shape (num_parts, height, width)

    label_mask: IntTensor = None # Tensor representing a ground truth mask with dims (height, width); used to extract dimensions
    resized_img_dims: torch.Size[2] = None # (height, width) of the resized image in SAM.

    is_inference: bool = None

    questions: list[str] = None
    question_type: QuestionType = None

    conversations: list[str] = None # List of conversation prompts. Contains part question choices and potentially object question choices
    conversation_types: list[ConversationType] = None # Whether conversation is a question, correct answer, or incorrect answer
    conversation_question_types: list[ConversationQuestionType] = None # Whether conversation is a part question or object question

    part_answer_choices: list[str] = None
    part_answer_types: list[AnswerType] = None
    answer_parts: list[list[str]] = None # Same length as number of answer choices with ConversationQuestionType == 'part_question'

    object_answer_choices: list[str] = None
    object_answer_types: list[AnswerType] = None
    answer_objects: list[str] = None # Same length as number of answer choices with ConversationQuestionType == 'object_question
    
    def __post_init__(self):
        if self.answer_objects is not None: # Correct object answer is the image label
            assert self.answer_objects[self.object_answer_types.index('correct')] == self.img_label


class PartSemSegDataset(torch.utils.data.Dataset):
    '''
    Training dataset for the Explanatory Segmentation Task.

    PartSemSegDataset consists of:
        - paco_lvis
        - pascal_part
        - partimagenet
        - partonomy
    '''
    pass


class ExplanatorySegDataset(torch.utils.data.Dataset):
    '''
    A dataset returning ExplanatorySegInstances for a given dataset (e.g. Partonomy-{Core, PACO-LVIS, Pascal-Part, PartImagenet})
    and question type (e.g. positive, negative, difference, whole_to_part, part_to_whole).
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    ignore_label = 255

    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        vision_tower: str,
        question_type: QuestionType,
        image_size=1024,
        transform=None,
        model_str: str = "plum",
        conv_type: str = 'llava_v1',
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

        with open(dataset_path, "r", encoding="utf-8") as f:
            print(f"(explanatory_seg_dataset.py) >>> Loading dataset from: {dataset_path}", dataset_path)
            temp_dataset = json.load(f)
            # list of QAPair instances
            self.dataset = [instance for instance in temp_dataset if instance['question_type'] == question_type.value]

        self.rng = np.random.default_rng(random_seed)
        self.tokenizer = tokenizer
        self.question_type = question_type
        self.image_size = image_size
        self.transform = transform if transform is not None else ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.model_str = model_str
        self.conv_type = conv_type
        self.test_with_gt_object = test_with_gt_object
        self.output_question_prompt_for_generation = output_question_prompt_for_generation
        self.sample_one_question_per_image = sample_one_question_per_image
        
        if self.sample_one_question_per_image:
            self._restrict_dataset_to_one_question_per_image()
        
        if self.model_str not in ["plum", "lisa", "glamm", "pixellm", "psalm"]:
            raise ValueError(f"Invalid model string: {self.model_str}")
        
    def _restrict_dataset_to_one_question_per_image(self):
        '''
        Restrict the dataset to only include one question per image.
        '''
        qa_pairs_by_image = defaultdict(list)
        
        for instance in self.dataset:
            qa_pairs_by_image[instance['image_path']].append(instance)
        
        filtered_dataset = []
        for image_path, qa_pairs in qa_pairs_by_image.items():
            sampled_qa_pair = self.rng.choice(qa_pairs)
            filtered_dataset.append(sampled_qa_pair)
        
        self.dataset = filtered_dataset
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Normalize pixel values and pad to a square input.
        '''
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ExplanatorySegInstance:
        instance = self.dataset[idx]
        image_path = instance.get("image_path")
        image = open_image(image_path)
        img_w, img_h = image.size
        image = np.array(image) # (height, width, channels)

        # preprocessed image for CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # preprocessed image for SAM
        image_resized = self.transform.apply_image(image)
        resize = image_resized.shape[:2]
        image_tensor = self.preprocess(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous())

        # text input construction
        image_label = instance.get("image_label", "")
        answer_types = instance.get("answer_types", [])
        question = instance.get("question", "")
        answer_choices = instance.get("answer_choices", "")
        answer_parts = instance.get("answer_parts", [[]])

        if len(answer_parts) == 0:
            raise ValueError("Ground-truth parts ('gt_parts') not found in the instance.")

        gt_idx = answer_types.index('correct') # index of the ground-truth answer choice

        conversations = []
        conversation_types = []
        conversation_question_types = []
        conv = conversation_lib.conv_templates[self.conv_type].copy()

        '''
        positive - "in common with" (gt_concept vs. counter_concept) - i.e., "Interserction Segmentation"
            * Q1: "What parts does this {gt_concept} have in common with [counter_concept]?
            * A1: "It has a tail, wings, cockpit, and a fuselage."

        negative - "different from" (gt_concept vs. counter_concept) (gt_concept should never be in the counter_concept place) - i.e., "Difference Segmentation"
            * Q1: "What parts does this {gt_concept} not have in common with [counter_concept]?
            * A1: "It has a tail, wings, cockpit, and a fuselage."

        difference - mixture of the above two
            * Q1: "What parts does this object have that {concept_1} has but {concept_2} does NOT have?"

        whole_to_part - Two sets of multiple choice answers (two question-answer pairs)
            * Q1: "What is the name of this object?"
            * A1: "airplanes--attack"
            * Q2: "Okay, what parts make this look like airplanes--attack?"
            * A2: "It has a tail wings cockpit fuselage

            E.g. “This is a [concept], and it has parts: [part1], [part2], …”

            Feed two sets of multiple choice answers: one to determine the concept P(object),
            then the second set conditioned on the first P(parts | object)

        part_to_whole - Two sets of multiple choice answers (two question-answer pairs)
            * Q1: "What parts of this object highlight its most distinctive parts?"
            * A1: "It has a tail wings cockpit fuselage"  # <- may have to divide this into multiple qa pairs for models like LISA
            * Q2: "Okay, based on the parts you observed, what is the name of this object?"
            * A2: "It's an airplanes--attack"

            E.g. “This object has [part1], [part2], … so it is a [concept]”

            Feed two sets of multiple choice answers: one to determine the parts P(parts | superordinate), then the second set conditioned on the first P(object | parts, superordinate)
        '''
        if self.test_with_gt_object:
            question = 'What is this object?'
            questions = [question]
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)

            answer = f'It is a {image_label}'
            conv.append_message(conv.roles[1], answer)
            conversations.append(conv.get_prompt())
            conversation_types.append('correct_answer')

            answer_parts = [['full_image']]
            answer_types = ['correct']
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

            '''
                During validate():

                Suppose LISA picks answer choice with parts [A, B, C]. Then you would pass in:
                "What parts does this object have in common with a [concept_name]? Assistant: It has an A, B, and C."
                "What parts does this object have in common with a [concept_name]? Assistant: It has an A, D, and K."
                "What parts does this object have in common with a [concept_name]? Assistant: It has an Q, J, and F."  # these are just example parts
                
                And we would take the argmin over the per sequence losses (i.e., the likelihood of the correct answer) to select the model-selected answer
            '''
            # Generate question
            questions = [question]
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
            question_conv = conv.copy()

            # Generate prompt with just the question with Assistant prompt
            if self.output_question_prompt_for_generation:
                # append_message adds sep2 if " " passed in, but doesn't add a space if empty string passed. Pass empty string then add space to prompt
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt() + " "

                conversations.append(prompt)
                conversation_types.append("question")
                conversation_question_types.append('part_question')

            '''
            Questions:
            You are a nice assistant. [USER]: What parts does this object have in common with a [concept_name]?

            MC Answers:
            You are a nice assistant. [USER]: What parts does this object have in common with a [concept_name]?
                [Assistant]: It has a part1 and part2

            You are a nice assistant. [USER]: What parts does this object have in common with a [concept_name]?
                [Assistant]: It has a part1 and part3

            You are a nice assistant. [USER]: What parts does this object have in common with a [concept_name]?
                [Assistant]: It has a part4 and part5
            '''
            # Generate answer variants:
            for i, answer in enumerate(answer_choices):
                conv = question_conv.copy()
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())
                conversation_types.append("incorrect_answer" if i != gt_idx else "correct_answer")
                conversation_question_types.append('part_question')

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
            
            questions = [object_question]
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + object_question)
            question_conv = conv.copy()
            
            gt_conv = None
            gt_object_idx = object_answer_types.index('correct')
            for i, answer in enumerate(object_answer_choices):
                conv = question_conv.copy()
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())
                conversation_types.append("incorrect_answer" if i != gt_object_idx else "correct_answer")
                conversation_question_types.append('object_question')
                
                if i == gt_object_idx:
                    gt_conv = conv

            # Generate part question/answers
            questions.append(question)
            question_conv = gt_conv
            question_conv.append_message(question_conv.roles[0], question)
            
            for i, answer in enumerate(answer_choices):
                conv = question_conv.copy()
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())
                conversation_types.append("incorrect_answer" if i != gt_idx else "correct_answer")
                conversation_question_types.append('part_question')

        elif self.question_type.value == "part_to_whole":  # no ground truth provided - open-ended generation
            # Generate part question/answers
            questions = [question]  # The part question
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
            question_conv = conv.copy()
            
            gt_conv = None
            for i, answer in enumerate(answer_choices):
                conv = question_conv.copy()
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())
                conversation_types.append('incorrect_answer' if i != gt_idx else 'correct_answer')
                conversation_question_types.append('part_question')

                if i == gt_idx:
                    gt_conv = conv
                    
            # Generate object questions/answers appended to gt part question/answer
            object_question: str = instance['object_question']
            object_answer_choices: list[str] = instance['object_answer_choices']
            object_answer_types: list[AnswerType] = instance['object_answer_types']
            answer_objects: list[str] = instance['object_answer_classes']
            
            questions.append(object_question)
            question_conv = gt_conv
            question_conv.append_message(question_conv.roles[0], object_question)
            
            object_gt_idx = object_answer_types.index('correct')
            for i, answer in enumerate(object_answer_choices):
                conv = question_conv.copy()
                conv.append_message(conv.roles[1], answer)
                conversations.append(conv.get_prompt())
                conversation_types.append('incorrect_answer' if i != object_gt_idx else 'correct_answer')
                conversation_question_types.append('object_question')
                
        else:
            raise ValueError(f"Invalid question type: {self.question_type}")

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
                    if "segmentation" in ann:
                        ann = ann['segmentation']
                    if "size" not in ann:
                        raise ValueError("Size not found in the segmentation annotation. Both RLE and Polygon loader requires image dim. (height and width).")
                    if "counts" in ann:
                        mask = load_rle(ann)
                    elif "polygon" in ann:  # polygon mask
                        mask = load_polygon(ann['polygon'], height=img_h, width=img_w)

                    mask_h, mask_w = ann['size']

                    # TODO: Skip mismatching masks in case mask_dims != img_dims - Look into the mask annotation (visualize)

                    if (mask_h, mask_w) != (img_h, img_w):
                        logger.warning(f'Mask dimensions {(mask_h, mask_w)} differ from images dimensions {(img_h, img_w)} - image_path: {image_path}')

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

                if 'bbox' in seg_anns:
                    raise NotImplementedError("Bounding box annotations are not supported yet.")

        # Union masks under the same part name
        masks_dict = {
            seg_label : torch.from_numpy(np.stack(seg_masks, axis=0).any(axis=0).astype(int))
            for seg_label, seg_masks in masks.items()
        }

        # masks_stacked - (num_parts, height, width)
        gt_answer_parts = answer_parts[answer_types.index('correct')]
        masks_stacked = np.stack([masks_dict[p] for p in gt_answer_parts])
        masks_tensor = torch.from_numpy(masks_stacked)

        # NOTE: only for paco_lvis and pascal_part - label isn't needed for partonomy (it's only needs a binary 'mask')
        label = torch.ones(masks_tensor.shape[1], masks_tensor.shape[2]) * self.ignore_label

        inference = True

        return ExplanatorySegInstance(
            img_path=image_path,
            img_label=image_label,
            sam_img_input=image_tensor,         # image tensor for SAM
            clip_img_input=image_clip,        # image preprocessed for CLIP.
            conversations=conversations,            # list of conversation prompts.
            mask_dicts=masks_dict,  # dict with seg_labels as keys and mask_tensors as values
            masks=masks_tensor,          # segmentation masks
            label_mask=label,
            resized_img_dims=resize,              # (height, width) of the resized image in SAM.
            questions=questions,
            question_type=self.question_type,
            part_answer_choices=answer_choices,
            part_answer_types=answer_types,
            answer_parts=answer_parts,
            object_answer_choices=object_answer_choices,
            object_answer_types=object_answer_types,
            answer_objects=answer_objects,
            is_inference=inference,
            conversation_types=conversation_types,
            conversation_question_types=conversation_question_types,
        )
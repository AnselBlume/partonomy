# %%
import torch
import numpy as np
from bisect import bisect_right
from typing import Literal
import logging

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

from .explanatory_seg_dataset import ExplanatorySegDataset, ExplanatorySegInstance

logger = logging.getLogger(__name__)

class ExplanatorySegDatasetsAdapter(torch.utils.data.Dataset):
    '''
    A dataset that concatenates or samples from a list of ExplanatorySegDatasets.
    Returns ExplanatorySegInstances in the format expected by dataset.py's collate_fn.

    Args:
        datasets (list[ExplanatorySegDataset]): A list of ExplanatorySegDatasets to provide instances from.

        indexing_strategy (Literal['concatenate', 'sample']): The strategy to use for indexing the datasets.
            - 'concatenate': Concatenates the datasets. The dataset length is the sum of the lengths of the input datasets.
            - 'sample': Samples from the datasets with replacement. Input indices are ignored. Each individual dataset is fully iterated through
                before repeating its instances.

        dataset_sampling_dist (list[float]): The sampling distribution to use for the datasets if indexing_strategy == 'sample'.
            - If None, will sample uniformly from the datasets.
            - If provided, must be of the same length as datasets.

        random_seed (int): The random seed to use for the datasets.

        verbose (bool): Whether to log debug information.
    '''
    def __init__(
        self,
        datasets: list[ExplanatorySegDataset],
        return_incorrect_answer_parts: bool = False,
        indexing_strategy: Literal['concatenate', 'sample'] = 'concatenate',
        dataset_sampling_dist: list[float] = None,
        random_seed: int = 42,
        verbose: bool = True,
        inference: bool = False
    ):
        self.datasets = datasets
        self.return_incorrect_answer_parts = return_incorrect_answer_parts
        self.indexing_strategy = indexing_strategy
        self.dataset_sampling_dist = dataset_sampling_dist
        self.verbose = verbose
        self.inference = inference
        
        # Initialize indexing strategy
        if indexing_strategy == 'concatenate':
            self.dataset_bounds = np.cumsum([0] + [len(dataset) for dataset in self.datasets])
            # e.g., 

        elif indexing_strategy == 'sample':
            if dataset_sampling_dist:
                assert len(dataset_sampling_dist) == len(datasets)
            else:
                self.dataset_sampling_dist = [1 / len(datasets) for _ in range(len(datasets))]

            self.rng = np.random.default_rng(random_seed)
            self.index_generators = [
                UniqueRangeGenerator(0, len(dataset), np.random.default_rng(random_seed + i))
                for i, dataset in enumerate(self.datasets)
            ]

        else:
            raise ValueError(f"Invalid indexing strategy: {indexing_strategy}")

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> tuple:
        self._log_debug(f'Getting item {idx}/{len(self) - 1}')

        if self.indexing_strategy == 'concatenate':
            if idx >= len(self):
                raise IndexError(f'Index {idx} is out of bounds for the dataset of length {len(self)}')

            dataset_idx = bisect_right(self.dataset_bounds, idx) - 1
            dataset_start_idx = self.dataset_bounds[dataset_idx]
            in_dataset_idx = idx - dataset_start_idx

            self._log_debug(f'Dataset idx: {dataset_idx}, dataset start idx: {dataset_start_idx}, in-dataset idx: {in_dataset_idx}')

            instance = self.datasets[dataset_idx][in_dataset_idx]

        elif self.indexing_strategy == 'sample':
            # Randomly sample a dataset, ignoring the provided idx
            dataset_idx = self.rng.choice(range(len(self.datasets)), p=self.dataset_sampling_dist)
            in_dataset_idx = self.index_generators[dataset_idx].next_int()

            self._log_debug(f'Dataset idx: {dataset_idx}, in-dataset idx: {in_dataset_idx}')

            instance = self.datasets[dataset_idx][in_dataset_idx]

        else:
            raise ValueError(f'Invalid indexing strategy: {self.indexing_strategy}')

        return self.instance_to_train_format(instance)

    def instance_to_train_format(self, instance: ExplanatorySegInstance) -> tuple:
        '''
        Convert the ExplanatorySegInstance to a tuple expected by dataset.py's collate_fn.

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            per_token_label_dict
        )

        based on SemSegDataset (of sem_seg_dataset.py)
        '''
        
        # Construct per_token_label_dict here and make necessary changes for collate_fn during training here
        GT_IDX = 0
        conversations = [instance.conversations[GT_IDX]]
        answer_parts = instance.answer_parts[GT_IDX]  # e.g., ['body', 'head', 'tail']
        per_token_label_dict_list = []
        masks = []
        part_to_mask = {}
        for part_text in answer_parts:
            if part_text in instance.mask_dicts:
                part_to_mask[part_text] = instance.mask_dicts[part_text]
                masks.append(instance.mask_dicts[part_text])
        per_token_label_dict_list.append(part_to_mask)
        
        # ###
        # if self.only_return_correct_conversation:
        #     part_convs = [
        #         c
        #         for c, c_type in zip(instance.conversations, instance.conversation_question_types)
        #         if c_type == 'part_question'
        #     ]

        #     conversations = [part_convs[instance.part_answer_types.index('correct')]]

        # else:
        #     conversations = instance.conversations
        # ###

        masks_stacked = np.stack(masks) # In the same order as the labels
        masks_tensor = torch.from_numpy(masks_stacked)  # masks_tensor.shape =   # (# of parts in the gt-conversation, height, width)
        
        if self.inference:
            return (
                instance.img_path,
                instance.sam_img_input,
                instance.clip_img_input,
                conversations,
                masks_tensor,
                instance.label_mask,
                instance.resized_img_dims,
                instance.questions,
                answer_parts,
                per_token_label_dict_list,
                self.inference,
                'explanatory_seg_dataset'
            )
        else:
            return (
                instance.img_path,
                instance.sam_img_input,
                instance.clip_img_input,
                conversations,
                masks_tensor,
                instance.label_mask,
                instance.resized_img_dims,
                instance.questions,
                answer_parts,
                per_token_label_dict_list,
            )

    def _log_debug(self, msg: str):
        if self.verbose:
            logger.debug(msg)

class UniqueRangeGenerator:
    '''
    Generate unique random integers between start (inclusive) and end (exclusive).

    When all numbers have been used up, will raise an error if cyclic is False. Otherwise, will restart from the beginning.
    '''
    def __init__(self, start: int, end: int, rng: np.random.Generator, cyclic: bool = True):
        self.start = start
        self.end = end
        self.rng = rng
        self.cyclic = cyclic
        self.numbers = list(range(self.start, self.end))

        self._reset_state()

    def has_next_int(self) -> bool:
        return self.tail_idx >= 0 or self.cyclic

    def next_int(self) -> int:
        if self.tail_idx < 0:
            if self.cyclic:
                self._reset_state()
            else:
                raise ValueError('All numbers have been used up!')

        num = self.numbers[self.tail_idx]
        self.tail_idx -= 1

        return num

    def _reset_state(self):
        self.rng.shuffle(self.numbers)
        self.tail_idx = len(self.numbers) - 1

if __name__ == '__main__':
    import coloredlogs
    from transformers import AutoTokenizer
    from question_type import QuestionType

    coloredlogs.install(level='DEBUG')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    ROOT_PATH = ""  # TODO: Set the root path that contains both the weights and the dataset

    paths = [
        f'{ROOT_PATH}/dataset/partonomy/partonomy_qa_pairs.json',
        f'{ROOT_PATH}/dataset/pascal_part/pascal_part_qa_pairs.json',
        f'{ROOT_PATH}/dataset/partimagenet/partimagenet_qa_pairs.json',
        f'{ROOT_PATH}/dataset/paco_lvis/paco_lvis_qa_pairs.json'
    ]

    # Defaults taken from train_ds.py
    tokenizer = AutoTokenizer.from_pretrained('liuhaotian/llava-llama-2-13b-chat-lightning-preview')
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens('[SEG]')

    vision_tower = 'openai/clip-vit-large-patch14'
    question_type = QuestionType.POSITIVE

    datasets = [ExplanatorySegDataset(path, tokenizer, vision_tower, question_type=question_type) for path in paths]

    # %% Test concatenate indexing strategy
    train_adapter = ExplanatorySegDatasetsAdapter(datasets, indexing_strategy='concatenate')
    print(train_adapter[0])
    print(train_adapter[len(train_adapter) - 1])

    # %% Test sample indexing strategy
    train_adapter = ExplanatorySegDatasetsAdapter(datasets, indexing_strategy='sample')
    print(train_adapter[0])

    # %%
from torch.utils.data import Dataset
import transformers
import yaml
import os
import json
import numpy as np
import torch
import cv2
import re
from PIL import Image
from torch import Tensor
from typing import Dict, Literal
from bisect import bisect_right
from llava.train.train import (
    DataArguments,
    rank0_print,
    preprocess_multimodal,
    clean,
    remove_prefix,
    preprocess,
    find_brackets,
    get_replacement_len
)
from llava.constants import DEFAULT_VIDEO_TOKEN,REPLACEMENT_TYPE, DEFAULT_SEGMENTATION_TOKEN
import copy
import random
from torchvision.ops import masks_to_boxes, box_convert
from explanatory_seg_dataset import ExplanatorySegDataset, ExplanatorySegInstance
import logging

logger = logging.getLogger(__name__)

def mask_to_bbox(mask: np.ndarray):
    '''
        Arguments:
            mask: np.ndarray, shape (h, w)
    '''
    h, w = mask.shape[:2]

    x = mask.any(1).nonzero()[0]
    y = mask.any(0).nonzero()[0]
    box = [x[0], y[0], x[-1] + 1, y[-1] + 1]  # x0 y0 x1 y1

    return box

class ExplanatorySegLazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
        datasets: list[ExplanatorySegDataset],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        is_eval=False,
        is_inference=False,
        inference_conv=None,
        debug_mode=False,
        only_return_correct_conversation: bool = True
    ):
        super().__init__()

        self.datasets = datasets

        is_training = not (is_eval or is_inference)
        self.is_inference = is_inference
        self.is_eval = is_eval
        self.eval_use_gt_mask_encode = data_args.eval_use_gt_mask_encode
        self.debug_mode = debug_mode
        self.only_return_correct_conversation = only_return_correct_conversation

        self.list_data_dict = []
        self.dataset_lengths = []
        self.dataset_bounds = np.cumsum([0] + [len(dataset) for dataset in self.datasets])

        # if is_training:
        #     with open(data_args.conversation_config) as config_file:
        #         conv_config = yaml.safe_load(config_file.read())
        #     for dataset in conv_config['datasets']:
        #         file_name = dataset['name']
        #         file_path = os.path.join(data_args.conversation_folder, file_name)
        #         print(f" ---- Training: Loading {file_path.split('/')[-1]} conversations ----")
        #         with open(file_path, "r") as f:
        #             if k := data_args.load_data:
        #                 to_extend = json.load(f)[:k]
        #             else:
        #                 to_extend = json.load(f)
        #             self.list_data_dict.extend(to_extend)
        #             self.dataset_lengths.append(len(to_extend))
        # elif is_eval:                                     # only load val dataset specified by data_args.val_dataset
        #     file_name = data_args.val_dataset
        #     conv_dir = data_args.conversation_folder
        #     if "all_data_mix_train" in conv_dir:
        #         conv_dir = conv_dir.replace("all_data_mix_train", "all_data_mix_val")
        #     file_path = os.path.join(conv_dir, file_name)
        #     print(f" ---- Validation: Loading {file_path.split('/')[-1]} conversations ----")
        #     with open(file_path, "r") as f:
        #         to_extend = json.load(f)
        #         self.list_data_dict.extend(to_extend)
        #         self.dataset_lengths.append(len(to_extend))
        # if is_inference:
        #     assert inference_conv is not None
        #     self.list_data_dict = [inference_conv]
        #     self.dataset_lengths.append(1)       # during inference, everything in one conversation
        # else:
        #     self.list_data_dict = None # TODO
        #     self.dataset_lengths = [len(d) for d in self.datasets]

        self.dataset_lengths = [len(d) for d in self.datasets]

        # FOR DEBUGGIN:
        # if num_rounds := data_args.limit_rounds:
        #     for entry in self.list_data_dict:
        #         entry['conversations'] = entry['conversations'][:2*num_rounds]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.TXT2TENSOR = dict()
        # print('--------------------Dataset Lengths -----------------------')
        print(self.dataset_lengths)

    def get_tensors_from_str(self,x):
        x = x.replace('[','',).replace(']','',)
        if x not in  self.TXT2TENSOR:
            print(x)
            return torch.zeros((1,1024))
        assert x in self.TXT2TENSOR,repr(x)
        z = self.TXT2TENSOR[x]
        # if 'any2any' in z['fpath']:
        #     print(x)
        #     return torch.zeros((1,1024))
        data = np.load(os.path.join(self.data_args.image_folder,z['fpath']))
        assert str(z['key']) in data,data.keys()
        res = torch.tensor(data[str(z['key'])]).view(1,1024)
        res = res / (res.norm()+1e-9) * 20
        return res

    def get_dataset_indices(self):
        return list([
            list(range(x)) for x in self.dataset_lengths
        ])

    def get_dataset_weight(self):
        # placeholder
        return [1] * len(self.dataset_lengths)

    # helper function for build_query (handles mask-encode, bbox-encode, not mask-decode)
    # def get_bitmask_bbox_encode(self, image_file_lst: list[str]):
    def get_bitmask_bbox_encode(self, image_path: str, gt_mask: np.ndarray):
        """Materialize encoder features referenced in the prompt.

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[int]]: Masked image
            crop, SAM-space bounding box, and the mask identifier.
        """
        # image_file,dataset_name,mask_id = image_file_lst[0].split('|')

        # Eval or Inference case (use dummy mask on 1st forward, otherwise load gt for mask-encode)
        # if (self.is_eval and not self.eval_use_gt_mask_encode) or dataset_name == 'INFERENCE':
        if self.is_eval and not self.eval_use_gt_mask_encode:
            # image_folder = self.data_args.image_folder
            # image_path = os.path.join(image_folder, image_file)
            processor = self.data_args.image_processor              # CLIP processor
            inputs = processor(Image.open(image_path))              # input image
            inputs = inputs.pixel_values[0]
            masked_instance_processed = torch.tensor(inputs)        # dummy mask-encode, just encode the image without masking or cropping
            bbox_coords_sam = torch.zeros(4)                        # dummy box-encode, all zeros
            # return masked_instance_processed, bbox_coords_sam, mask_id
            return masked_instance_processed, bbox_coords_sam, None # TODO

        # temp handle edge cases
        # if mask_id == '' or mask_id == "'":         # this is the case for reason_seg sentences
        #     mask_id = None
        # elif "_" in mask_id or "-" in mask_id:
        #     mask_id=mask_id
        # else:
        #     mask_id = int(mask_id)

        # image_folder = self.data_args.image_folder
        # image_path = os.path.join(image_folder, image_file)
        # image_path = image_path.replace('val2014', 'train2014')
        # image_path = image_path.replace('new_', '')
        # if 'VG_100K' in image_path:
        #     image_path = image_path.replace('./images_folder', './images_folder/vg')

        # if not os.path.exists(image_path):
        #     image_path = image_path.replace('val2017', 'train2017')     # new edge case for lvis (val)
        #     image_path = image_path.replace('images/', 'object365/')
        assert os.path.exists(image_path)

        # Mask image with GT mask
        image = Image.open(image_path)
        (w, h) = image.size
        image = np.array(image.convert('RGB'))
        # gt_mask = self.data_args.register.get_bitmask(
        #     dataset_name,
        #     mask_id,
        #     is_eval=self.is_eval,
        #     image_file=image_file.split("/")[-1],
        #     image_dim=(h,w)
        # )


        image_masked = cv2.bitwise_and(image, image, mask=gt_mask)

        # Crop image with bbox and pad
        # (x0,y0,x1,y1) = self.data_args.register.get_bbox(
        #     dataset_name,
        #     mask_id,
        #     is_eval=self.is_eval,
        #     image_file=image_file.split("/")[-1],
        #     image_dim=(h,w)
        # )
        x0, y0, x1, y1 = mask_to_bbox(gt_mask)
        # mask = masks
        x1 = max(x1,x0+1)
        y1 = max(y1,y0+1)
        max_width = max(x1-x0,y1-y0)
        image_masked_cropped = image_masked[x0:x1,y0:y1] # cropped  # H_C, W_C,1
        image_masked_cropped_padded = np.zeros((max_width,max_width,image_masked.shape[-1]),dtype=image_masked.dtype)
        image_masked_cropped_padded[:image_masked_cropped.shape[0],:image_masked_cropped.shape[1]] = image_masked_cropped

        # preprocess for CLIP
        processor = self.data_args.image_processor
        inputs = processor(Image.fromarray(image_masked_cropped_padded))
        inputs =inputs.pixel_values[0] # C H W, np.npndarr
        inputs = torch.tensor(inputs)#.permute(1,2,0)
        masked_instance_processed = inputs

        # bbox coords in SAM dimension
        processor = self.data_args.mask_processor
        # data_mask = processor(np.array(Image.open(image_path).convert('RGB')),
        #                     masks=[gt_mask,gt_mask])
        data_mask = processor(image, masks=[gt_mask,gt_mask])
        mask_bin = data_mask['mask']
        y0 = torch.where(mask_bin.sum((0,1)))[0].min()
        y1 = torch.where(mask_bin.sum((0,1)))[0].max()
        x0 = torch.where(mask_bin.sum((0,2)))[0].min()
        x1 = torch.where(mask_bin.sum((0,2)))[0].max()
        bbox_coords_sam = torch.tensor([x0,y0,x1,y1]) / 1024.0 # 1 1024

        # return masked_instance_processed, bbox_coords_sam,mask_id
        return masked_instance_processed, bbox_coords_sam, None # TODO

    # def get_bitmask_decode(self, image_file_lst: list[str]) -> tuple[dict, int]:
    def get_bitmask_decode(self, image_path: str, tgt_mask: np.ndarray, ref_mask: np.ndarray = None) -> tuple[dict, int]:
        """Decodes special [MASK-DECODE...] encodings in the prompt

        Returns:
            tuple[dict, int]: Processor-ready mask tensors and the
            resolved mask identifier.
        """
        #image_file,dataset_name,mask_id = image_file_lst[0].split('|')
        # if ':' in image_file_lst[0]:
        #     # (reference mask decoding format)
        #     if len(re.findall(':', image_file_lst[0])) == 4:
        #         # 1 reference mask
        #         task_type,ref_mask_id,tgt_mask_id,image_file,dataset_name = image_file_lst[0].split(':')
        #     elif len(re.findall(':', image_file_lst[0])) == 5:
        #         # 2 reference masks
        #         task_type,ref_mask_id,ref_mask_id_2,tgt_mask_id,image_file,dataset_name = image_file_lst[0].split(':')
        #     else:
        #         raise ValueError("Base ref-mask decode format:", image_file_lst[0])
        # elif '|' in image_file_lst[0]:
        #     # (no reference mask decoding format)
        #     image_file,dataset_name,tgt_mask_id = image_file_lst[0].split('|')
        #     task_type = 'none'
        #     ref_mask_id = None
        # else:
        #     raise ValueError("Base decode format:", image_file_lst[0])

        # Inference case:
        # if dataset_name == 'INFERENCE':
        if self.is_inference:
            # raise NotImplementedError("Obselete, do not use!!!")
            #TODO: @shaolun_zhang, please fix inference
            # image_folder = self.data_args.image_folder
            # image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            (w, h) = image.size
            mask = np.ones((h, w, 1), dtype=np.uint8)                   # dummy mask as gt mask for inference
            processor = self.data_args.mask_processor                   # processor --> decoder dimension
            data = processor(np.array(image.convert('RGB')), masks=[mask,mask]) # expect list of masks [ref, gt]
            data['image_path'] = image_path
            data['task_type'] = 'segmentation'
            # return data, tgt_mask_id # 1 1024     # Question: use mask_id_0 or mask_i_1
            return data, None # 1 1024     # Question: use mask_id_0 or mask_i_1

        # def process_mask_id(mask_id):
        #     if mask_id == '' or mask_id == "'" or mask_id == 'none' or mask_id == None:         # this is the case for reason_seg sentences
        #         mask_id = None
        #     elif "_" in mask_id or "-" in mask_id:
        #         mask_id=mask_id
        #     else:
        #         mask_id = int(mask_id)
        #     return mask_id

        # TODO
        # ref_mask_id = process_mask_id(ref_mask_id)
        # tgt_mask_id = process_mask_id(tgt_mask_id)

        # image_folder = self.data_args.image_folder
        # image_path = os.path.join(image_folder, image_file)

        # Edge case handling
        # image_path = image_path.replace('val2014', 'train2014')
        # image_path = image_path.replace('new_', '')
        # if 'VG_100K' in image_path:
        #     image_path = image_path.replace('./images_folder', './images_folder/vg')

        # if not os.path.exists(image_path):
        #     image_path = image_path.replace('val2017', 'train2017')     # new edge case for lvis (val)
        #     image_path = image_path.replace('images/', 'object365/')
        assert os.path.exists(image_path)

        image = Image.open(image_path)
        (w, h) = image.size
        image = np.array(image.convert('RGB'))
        # tgt_mask = self.data_args.register.get_bitmask(
        #     dataset_name,
        #     tgt_mask_id,
        #     is_eval=self.is_eval,                   # pass in is_eval flag, signals which seg anno file to use (train vs. eval)
        #     image_file=image_file.split("/")[-1],
        #     image_dim=(h,w)
        # )
        # if ref_mask_id:
        #     ref_mask = self.data_args.register.get_bitmask(
        #         dataset_name,
        #         ref_mask_id,
        #         is_eval=self.is_eval,
        #         image_file=image_file.split("/")[-1],
        #         image_dim=(h,w)
        #     )                                       # load mask from seg register
        # else:
        #     ref_mask = np.zeros_like(tgt_mask)
        if ref_mask is None:
            ref_mask = np.zeros_like(tgt_mask)

        masks = [ref_mask,tgt_mask]
        processor = self.data_args.mask_processor
        data = processor(image, masks=masks)

        data['image_path'] = image_path
        data['task_type'] = 'segmentation'
        # return data, tgt_mask_id
        return data, None # TODO

    # def build_query(self, x: str, image_path: str, gt_mask: np.ndarray, ref_mask: np.ndarray = None) -> \
    def build_query(self, x: str, instance: ExplanatorySegInstance, allow_mask_keyerr: bool = False) -> \
        tuple[str, Tensor] \
        | tuple[str, tuple[Tensor, int]] \
        | tuple[str, tuple[Tensor, int, int]] \
        | tuple[str, tuple[dict, str]] \
    :
        '''
        Dispatches serialized placeholders to modality-specific loaders.

        Arguments:
            x (str): The serialized placeholder.
            gt_mask (np.ndarray): The ground truth mask, or the reference mask provided by the user.
            ref_mask (np.ndarray): Reference mask only provided to the decoder during decoding of the gt_mask.
                NOTE This should be EQUAL to the gt_mask provided by the user in the same round.
                For reference, see shared/datasets/segllm/conversations_folder/all_data_mix_val/mr_paco_val.json:79
                which has the same ref_mask_id (733) as the previous human MASK-ENCODE's mask_id
            allow_mask_keyerr (bool): Allow the instance to not have a corresponding mask for non-GT conversations

        Returns:
            tuple[str, Any]: The modality key and associated payload.
            e.g. ('image-encode', data)
            e.g. ('mask-encode', (masked_instance_processed, mask_id))
            e.g. ('bbox-encode', (bbox_coords_sam, mask_id))
            e.g. ('mask-decode', (data, tgt_mask_id))

        '''

        data = torch.zeros(1,3,224,224)
        image_path = instance.img_path

        # if image_file_lst := re.compile('IMAGE256:(.*)$').findall(x):
        if re.match('IMAGE256:(.*)$', x):
            # image_file = image_file_lst[0]
            # image_folder = self.data_args.image_folder
            # image_path = os.path.join(image_folder, image_file)
            # image_path = image_file_lst[0]

            # if not os.path.exists(image_path):
            #     image_path = image_path.replace('val2017', 'train2017')     # new edge case for lvis (val)
            #     image_path = image_path.replace('images/', 'object365/')
            # if not os.path.exists(image_path):
            #     print('image file', image_file)
            #     print('image_path', image_path)
            assert os.path.exists(image_path)

            inputs = Image.open(image_path).convert('RGB')
            processor = self.data_args.image_processor
            inputs = processor(inputs)
            inputs =inputs.pixel_values[0] # C H W, np.npndarr
            inputs = torch.tensor(inputs)#.permute(1,2,0)
            data = inputs
            return 'image-encode',data

        # Extract gt_mask and possibly ref_mask from instance
        # Format: ENCODING:ref_part_name:gt_part_name:
        segments = x.split(':')
        ref_part_name, gt_part_name = segments[1], segments[2]

        try:
            gt_mask = instance.mask_dicts[gt_part_name].numpy().astype(np.uint8)
            ref_mask = instance.mask_dicts[ref_part_name].numpy().astype(np.uint8) if ref_part_name != 'none' else None
        except KeyError as e:
            if allow_mask_keyerr:
                gt_mask = np.zeros_like(instance.masks[0])
                ref_mask = np.zeros_like(instance.masks[0])
            else:
                raise e

        # if image_file_lst := re.compile('MASK-ENCODE:(.*)$').findall(x):
        if re.match('MASK-ENCODE:(.*)$', x):
            # masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_file_lst)
            masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_path, gt_mask)
            return 'mask-encode', (masked_instance_processed,mask_id)
        # elif image_file_lst := re.compile('BOX-ENCODE:(.*)$').findall(x):
        elif re.match('BOX-ENCODE:(.*)$', x):
            # masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_file_lst)
            masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_path, gt_mask)
            return 'bbox-encode', (bbox_coords_sam,mask_id)
        # elif image_file_lst := re.compile('MASK-DECODE:(.*)$').findall(x):
        elif re.match('MASK-DECODE:(.*)$', x):
            # data, tgt_mask_id = self.get_bitmask_decode(image_file_lst)
            data, tgt_mask_id = self.get_bitmask_decode(image_path, gt_mask, ref_mask)
            return 'mask-decode',(data,tgt_mask_id) # 1 1024
        else:
            raise NotImplementedError(x)

    def __len__(self):
        # return len(self.list_data_dict)
        return sum(len(dataset) for dataset in self.datasets)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def dataset_index(self,i):
        dataset_lengths = self.dataset_lengths
        total_length = sum(dataset_lengths)

        if i < 0 or i >= total_length:
            raise IndexError("Out of bound")

        cumulative_length = 0
        for index, length in enumerate(dataset_lengths):
            cumulative_length += length
            if i < cumulative_length:
                return index

    def convert_global_index(self, i: int) -> int:
        if i < 0 or i >= len(self):
            raise IndexError("Out of bound")
        dataset_idx = bisect_right(self.dataset_bounds, i) - 1
        local_idx = i - self.dataset_bounds[dataset_idx]

        return dataset_idx, local_idx


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # _dataset_idx = self.dataset_index(i) # The dataset the index corresponds to
        # TODO Extract/build conversations from ExplanatorySegInstance
        # sources = self.datasets[_dataset_idx]
        ds_idx, local_idx = self.convert_global_index(i)
        instance: ExplanatorySegInstance = self.datasets[ds_idx][local_idx]

        conversations = copy.deepcopy(instance.conversations)

        if self.only_return_correct_conversation: # Extract only correct answer part question for training
            found = False
            for i in range(len(conversations)):
                cq_type = instance.conversation_question_types[i]
                c_type = instance.conversation_types[i]

                if c_type == 'correct_answer' and cq_type == 'part_question':
                    found = True
                    break

            if not found:
                raise RuntimeError('No correct answer part question found in the instance')
            indices = [i]
        else:
            indices = list(range(len(conversations)))

        ret_dicts = [ # For incorrect answers, allow the instance to not have a corresponding mask
            self._build_conversation_dict(instance, index, i, allow_mask_keyerr=index != 0)
            for index in indices
        ]

        return ret_dicts


    def _build_conversation_dict(
        self,
        instance: ExplanatorySegInstance,
        conversation_idx: int,
        global_idx: int,
        allow_mask_keyerr: bool = False # Allow instance to not have a corresponding mask for non-GT conversations
    ) -> dict:

        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"
        extra_inputs = []
        conversation = instance.conversations[conversation_idx]

        # if 'image' in sources[0]:
        #     assert False, 'Should not be in this path'

            # image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            # image_path = os.path.join(image_folder, image_file)
            # inputs = {
            #     "image": [image_path,]
            # }
            # processor = self.data_args.image_processor
            # inputs = processor(inputs)
            # inputs['image']['pixel_values'] = inputs['image']['pixel_values'].squeeze(0) # hack
            # sources_p = preprocess_multimodal(
            #     copy.deepcopy([e["conversations"] for e in sources]),
            #     self.data_args) # move <image> to the start of all data
        # elif 'multimodal_input' in sources[0]:
        #     assert False, 'Should not be in this path'

            # multi_input = sources[0]['multimodal_input']
            # modality = multi_input['type']
            # assert modality!= 'image'
            # if modality == 'audio':
            #     raise NotImplemented
            # sources_p = copy.deepcopy([e["conversations"] for e in sources])
            # sources_p[0][0]['value'] = sources_p[0][0]['value'].replace(DEFAULT_AUDIO_TOKEN,DEFAULT_AUDIO_TOKEN*8)
            # # if DEFAULT_AUDIO_TOKEN in sources_p[0][0]['value'] :
            # #     raise AssertionError(sources_p[0][0]['value'])
        # else:
        #     sources_p = copy.deepcopy([e["conversations"] for e in sources])

        # do_generation=False
        info = {}
        # image_folder = self.data_args.image_folder
        # if sources[0].get('task') == 'generation':
        #     do_generation = True
        #     raise NotImplemented

        # Modification
        mask_encode_ref = []                 # a list of indices to keep track which round's mask output is THIS mask-encode referring to

        extra_replacement_mask = []
        delayed_process = False
        # if sources[0].get('task') == 'any2any':
        #     assert False, 'Should not be in this path'

            # info['generation_seq_len'] = 1
            # replacement = []
            # replacement_mask = [] # loss mask
            # base = sources[0]['base']
            # info['generation'] = True
            # drop_base = random.random() < 0.2
            # all_tgts = {x[1]:x for x in (sources[0]['added'] if sources[0]['added'] else [])}
            # adds = []
            # raw_val = []
            # for turn in sources_p[0]:
            #     src = turn['from']
            #     val = turn['value']
            #     if drop_base:
            #         val = val.replace('<base>','<base_null>')
            #     if src == 'human':
            #         matches = find_brackets(val)
            #         for prompt in matches: # list of str wit '[]'
            #             if prompt in all_tgts:
            #                 set_instance = True
            #             else:
            #                 set_instance = False
            #             prompt_clean = prompt[1:-1]
            #             if clean(prompt_clean) not in  self.TXT2TENSOR:
            #                 print(prompt_clean)
            #                 val = val.replace(prompt,remove_prefix(prompt_clean),1)
            #                 continue
            #             if prompt == base:
            #                 if drop_base:
            #                     val = val.replace(prompt,remove_prefix(prompt_clean),1)
            #                 else:
            #                     val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
            #                     replacement.append(prompt_clean)
            #                     replacement_mask.append(REPLACEMENT_TYPE.INPUT)
            #                     raw_val.append(prompt)
            #                     # if set_instance:
            #                     #     adds.append((all_tgts[prompt][0],prompt_clean))
            #             elif random.random() < 0.2:
            #                 val = val.replace(prompt,remove_prefix(prompt_clean),1)
            #             else:
            #                 val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
            #                 replacement.append(prompt_clean)
            #                 replacement_mask.append(REPLACEMENT_TYPE.INPUT)
            #                 raw_val.append(prompt)
            #                 if set_instance:
            #                     adds.append((all_tgts[prompt][0],prompt_clean))
            #         raw_val.append(val)
            #     elif src == 'gpt':
            #         matches = find_brackets(val)
            #         for prompt in matches: # list of str wit '[]'
            #             prompt_clean = prompt[1:-1]
            #             seen = 0
            #             if prompt == base and (drop_base or prompt_clean not in self.TXT2TENSOR):
            #                 val = val.replace(prompt,'',1)
            #                 val = val.replace('<base>','<base_null>')
            #             elif prompt == base:
            #                 val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
            #                 replacement.append(prompt_clean)
            #                 replacement_mask.append(REPLACEMENT_TYPE.BASE)
            #             else:
            #                 assert seen == 0, "Only one outout per instructions!!!"
            #                 seen =1
            #                 if self.data_args.output_text:
            #                     val = val.replace(prompt,prompt+DEFAULT_VIDEO_TOKEN,1)
            #                 else:
            #                     val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
            #                 replacement.append(prompt_clean)
            #                 replacement_mask.append(REPLACEMENT_TYPE.GEN)
            #         # if (not adds) and all_tgts:
            #         #     raise ValueError(f'{adds},{val},{all_tgts},{raw_val}')
            #         if adds:
            #             val += 'additions:'
            #             for addition_src,addition_caption in adds:
            #                 val += f'{addition_src}:{DEFAULT_VIDEO_TOKEN}.'
            #                 replacement.append(addition_caption)
            #                 replacement_mask.append(REPLACEMENT_TYPE.GEN)
            #             # raise ValueError(val)
            #             # print("MODIFIED:")
            #     else:
            #         raise NotImplemented
            #     turn['value']= val
            #     assert len(replacement_mask) == len(replacement)
            # if len(replacement):
            #     extra_replacement = torch.cat([self.get_tensors_from_str(clean(x)) for x in replacement])
            # extra_replacement_mask = replacement_mask

        # if sources[0].get('task') == 'segmentation':
        if True:
            delayed_process = True
            info['generation_seq_len'] = 1
            replacement = [] # List of special strings (e.g. "IMAGE256:students.jpg", "MASK-DECODE:...", "MASK-ENCODE:...", "BOX-ENCODE:...") that were replaced with special tokens (e.g. '<video>' * 256)
            replacement_mask = [] # loss mask
            # base = sources[0]['base']
            info['generation'] = True
            # Each pair of human/assistant turns can contain several encode and
            # decode placeholders. That structure enables sequential mask
            # decoding within one round: encodes (image/mask/box) appear first
            # so that subsequent ``MASK-DECODE`` tokens can pull features from
            # ``extra_replacement`` below.
            # for turn in sources_p[0]:
            for turn in conversation:
                src = turn['from']
                val = turn['value']
                if src == 'human':
                    matches = find_brackets(val)
                    contains_mask_encode = False
                    contains_box_encode = False
                    contains_image_encode = False
                    for prompt in matches: # list of str wit '[]'
                        prompt_clean = prompt[1:-1] # e.g. "IMAGE256"
                        val = val.replace(prompt,DEFAULT_VIDEO_TOKEN*(rl:=get_replacement_len(prompt_clean)),1) # Replaces prompt with specialized tokens (e.g. "IMAGE256" -> "<video>" * 256)
                        replacement.append(prompt_clean)
                        replacement_mask.extend([REPLACEMENT_TYPE.INPUT]*rl)

                        if self.is_inference:
                            if ('MASK-ENCODE' in prompt):
                                contains_mask_encode = True
                            if ('BOX-ENCODE' in prompt):
                                contains_box_encode = True
                            if 'IMAGE' in prompt:
                                contains_image_encode = True

                    if self.is_inference:
                        if (not contains_mask_encode) and (not contains_box_encode) and (not contains_image_encode):
                            mask_encode_ref.append(-1)                        # indicate this turn (round >= 1) does not have mask/box encode, round 0 expected to not have 'ind'
                        elif contains_mask_encode or contains_box_encode:
                            assert 'ind' in turn                              # turn has fields 'from', 'value', 'ind'
                            mask_encode_idx = [int(x) - 1 for x in turn['ind']]
                            mask_encode_ref.extend(mask_encode_idx)           # mask-encode, box-encode share the same mask_encode_ref



                elif src == 'gpt':
                    matches = find_brackets(val)
                    for prompt in matches: # list of str wit '[]'
                        prompt_clean = prompt[1:-1] # "e.g. MASK-DECODE:..."
                        val = val.replace(prompt,DEFAULT_SEGMENTATION_TOKEN*(rl:=get_replacement_len(prompt_clean)),1) # Replaces prompt (e.g. "[MASK-DECODE:...]" -> "<seg>")
                        replacement.append(prompt_clean)
                        replacement_mask.extend([REPLACEMENT_TYPE.SEG]*rl)
                else:
                    raise NotImplemented
                turn['value']= val
                #assert len(replacement_mask) == len(replacement)

            # For debug tokenizer
            if self.debug_mode:
                replacement = []

            if len(replacement): # Actually load the image assets corresponding to the special [...] strings
                extra_replacement = list([self.build_query(x, instance, allow_mask_keyerr=allow_mask_keyerr) for x in replacement]) # list[tuple[str, Any]]:  e.g. ('image-encode, data)

            extra_replacement_mask = replacement_mask

        info['output_text'] = self.data_args.output_text
        data_dict = preprocess(
            [conversation],
            self.tokenizer,
            # has_image=('image' in self.list_data_dict[i]),info=info)
            has_image=False,
            info=info)
        # if isinstance(i, int):
        if isinstance(global_idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # if do_generation:
        if False:
            data_dict['generation_target'] = generation_target
        else:
            data_dict['generation_target'] = None
        # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     assert False, 'Should not be in this path'
            # data_dict['image'] = inputs # hack, actually multimodal

        if self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = dict( # Placeholder image to be replaced
                image=dict(
                    pixel_values=torch.zeros(3, crop_size['height'], crop_size['width'])
                )
            )

        data_dict['conversation'] = conversation
        data_dict['image_path'] = instance.img_path
        data_dict['conv_id'] = str(global_idx)
        data_dict['mask_encode_ref'] = mask_encode_ref
        data_dict['extra_inputs']=extra_inputs
        data_dict['extra_replacement']=extra_replacement
        data_dict['extra_replacement_mask']=extra_replacement_mask
        data_dict['delayed_process']=delayed_process
        # data_dict['dataset_index'] = ds_idx
        data_dict['dataset_index'] = global_idx
        data_dict['mask'] = instance.masks
        data_dict['mask_dicts'] = instance.mask_dicts
        assert len(extra_replacement) == len(extra_replacement_mask) or delayed_process
        return data_dict
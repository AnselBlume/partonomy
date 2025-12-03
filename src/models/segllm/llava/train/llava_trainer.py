import os
import torch
import re
import json
from matplotlib import pyplot as plt

from torch.utils.data import Sampler,BatchSampler
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional, Iterable
from torch import nn
import torch.distributed as dist
from typing import Dict,Union,Any
from transformers.utils import is_sagemaker_mp_enabled
from packaging import  version
import wandb
import numpy as np
import copy
from PIL import Image
import cv2
from llava.train.log_callbacks import wandb_dump_images
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

from transformers.modeling_utils import unwrap_model
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

class MultiDatasetBatchSampler(BatchSampler):
    def __init__(self, datasets, weights, batch_size,shuffle=True,local_rank=0,world_size=1):
        self.datasets_length = np.array([len(x) for x in datasets])
        self.datasets_start_index = np.cumsum( self.datasets_length)
        self.datasets_start_index = np.concatenate([[0],self.datasets_start_index])
        self.datasets_start_index,self.length = self.datasets_start_index[:-1],self.datasets_start_index[-1]
        self.dataset_weight = torch.tensor(weights).float()
        self.batch_size = batch_size
        self.local_rank = local_rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = 0
        self.generator = torch.Generator()


    def __iter__(self):
        self.generator.manual_seed(self.seed)
        self.seed += 1
        batch = []
        # n_batches = self.length // (self.batch_size * self.world_size)
        n_batches = len(self) # ANSEL
        for dataset_idx in range(n_batches):
            # make sure all gpu see same dataset
            select_dataset = torch.multinomial(self.dataset_weight,1,replacement=True,generator=self.generator)[0].item() # sample one data
            # we don't actually care what happens here, so long as we do not repeat batch per gpu
            selected_index = np.random.randint(0,self.datasets_length[select_dataset],self.batch_size* self.world_size)
            selected_index = selected_index + self.datasets_start_index[select_dataset]
            selected_index = selected_index[self.local_rank *self.batch_size:(self.local_rank *self.batch_size+self.batch_size) ]
            #print(f"RANK:{self.local_rank }",selected_index)
            yield selected_index.tolist()

            # for debugging stuck: (hardcode which rank gets which data)
            # if self.local_rank == 0:
            #     yield [0,1]
            # if self.local_rank == 1:
            #     yield [2,3]
            # if self.local_rank == 2:
            #     yield [4,5]
            # if self.local_rank == 3:
            #     yield [6,7]

    def __len__(self):
        # return self.length // (self.batch_size * self.world_size)
        # ANSEL
        if self.length <= self.batch_size * self.world_size: # Only one batch
            return 1
        return self.length // (self.batch_size * self.world_size) # Implicitly drops last batch


class LLaVATrainer(Trainer):

    def __init__(self,model_args=None, data_args=None,pipe=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._move_model_to_device(self.model, self.args.device)
        assert pipe is None, "This code do not support image generation"
        self.model_args=model_args
        self.data_args=data_args
        self.pipe = pipe

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        batch_sampler = MultiDatasetBatchSampler(
            train_dataset.get_dataset_indices(),
            weights=train_dataset.get_dataset_weight(),
            batch_size=self._train_batch_size,
            shuffle=True,
            local_rank=self.accelerator.process_index,world_size=self.accelerator.num_processes
        )

        dataloader = DataLoader(
            train_dataset,
            #batch_size=args.train_batch_size,
            #shuffle=True,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            batch_sampler=batch_sampler
        )

        return dataloader #self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False) and False: # alwats load full
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            model.generation_config.do_sample = True
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # if getattr(self.args, 'tune_mm_mlp_adapter', False):
        #     pass
        # else:
            self.model.generation_config.do_sample = True
            super(LLaVATrainer, self)._save(output_dir, state_dict)


    def group_by_round_helper(self, img_indices, metrics_per_mask):
        '''
        img_indices = [0,0,1,1,1]
        img_indices[i] = the index of image/conv that mask_i belongs to
        len(img_indices) == len(metrics_per_mask) == N (total num mask decodes)
        round_counters[0]: pointer iterating over [0,0] (conv_1)
        round_counters[1]: pointer iterating over [1,1,1] (conv_2)
        '''
        N_convs = len(set(img_indices))         # number of conversations = number of distinct integers in img_indices
        round_counters = [0] * N_convs          # create a round counter for each conversation
        max_num_rounds = np.unique(img_indices,return_counts=True)[1].max() #Counter(img_indices).most_common(1)[0][1]  # get num_rounds for conv with longest num rounds
        metric_per_round = [list() for _ in range(max_num_rounds)]  # metric_per_round[i] = [round_i_metric for each conv]

        for img_idx, metric in zip(img_indices, metrics_per_mask):
            round_idx = round_counters[img_idx]
            metric_per_round[round_idx].append(metric)
            round_counters[img_idx] += 1

        return metric_per_round

    def group_by_round(
        self,
        img_indices,
        inter_per_mask,
        union_per_mask,
        iou_per_mask,
        iou_per_box,
        debug_mode=False
    ):
        '''
        SUPPOSE:
            conv 1: [round 1, round 2, round 3]
            conv 2: [round 1, round 2]
            conv 3: [round 1, round 2, round 3, round 4]
        OUTPUT:
            <metric>_per_round[i] = [round i <metric> for each conv]
            len(<metric>_per_round) = max([num_rounds(conv) for conv in batch])
        '''

        inter_per_round = self.group_by_round_helper(img_indices, inter_per_mask)
        union_per_round = self.group_by_round_helper(img_indices, union_per_mask)
        mask_iou_per_round = self.group_by_round_helper(img_indices, iou_per_mask)
        box_iou_per_round = self.group_by_round_helper(img_indices, iou_per_box)

        # CHECK:
        if debug_mode:
            print('IMG INDICES:')
            print(img_indices)
            print()
            print('INTERECTION:')
            print(inter_per_mask)
            for i, round_i_metrics in enumerate(inter_per_round):
                print(f'round {i}:', round_i_metrics)
            print()
            print('UNION:')
            print(union_per_mask)
            for i, round_i_metrics in enumerate(union_per_round):
                print(f'round {i}:', round_i_metrics)
            print()
            print('MASK IOU:')
            print(iou_per_mask)
            for i, round_i_metrics in enumerate(mask_iou_per_round):
                print(f'round {i}:', round_i_metrics)
            print()
            print('BOX IOU:')
            print(iou_per_box)
            for i, round_i_metrics in enumerate(box_iou_per_round):
                print(f'round {i}:', round_i_metrics)
            print()

        return dict(
            inter_per_round=inter_per_round,
            union_per_round=union_per_round,
            mask_iou_per_round=mask_iou_per_round,
            box_iou_per_round=box_iou_per_round,
        )

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
            ar_decoding = False
        ):
        print('eval mode: ' + ('ar eval' if ar_decoding else 'gt forcing') + '\n\n')
        eval_loader = self.get_eval_dataloader()

        metric_dict_prototype = lambda : dict(
            all_ious=0,
            true_pos=0,
            all_int=0,
            all_union=0,
            n_boxes=0,
            n_masks=0,
        )
        metric_per_round = {}
        self.model.eval()
        self.model.to(torch.bfloat16)
        n_eval = len(eval_loader)
        idx = 0
        input_counter = 0
        for i in range(12):
            metric_per_round[i] = metric_dict_prototype()
        for batch in eval_loader:
            idx += 1
            # if idx > 100:
            #     break # debug loop

            with torch.no_grad():

                if not ar_decoding:
                    outputs = self.model(**batch)
                else:
                    original_inputs = copy.deepcopy(batch)
                    replace_dict = {}
                    all_mask_encode = []
                    all_mask_decode = []
                    all_box_encode = []
                    all_mask_encode_images = []
                    all_box_encode_images = []
                    all_mask_encode_id_to_data = {}
                    all_mask_decode_id_to_data = {}
                    all_box_encode_id_to_data = {}
                    for i, row in enumerate(original_inputs['extra_replacement']['data']): # loop over img
                        for j,col in enumerate(row): # loop over col
                            if type(col) in [list,tuple]:
                                if col[0] == 'mask-encode':
                                    all_mask_encode.append(((i,j),col))
                                    all_mask_encode_id_to_data[(i,j)] =col[1][1]
                                elif col[0] == 'bbox-encode':
                                    all_box_encode.append(((i,j),col))
                                    all_box_encode_id_to_data[(i,j)] = col[1][1]
                                elif col[0] == 'mask-decode':
                                    all_mask_decode.append(((i,j),col))
                                    all_mask_decode_id_to_data[(i,j)] = col[1][1]
                    max_round = 100
                    for round_i in range(max_round):
                        # Jacobi fixed-point decoding
                        if round_i >= max_round:
                            break
                        curr_round_inputs = copy.deepcopy(original_inputs)
                        for (a,b),c in replace_dict.items():
                            curr_round_inputs['extra_replacement']['data'][a][b] = c
                        outputs = self.model(**curr_round_inputs)
                        gt_masks = outputs['individual_losses']['mask_data']
                        curr_masks = outputs['individual_losses']['segm_loss_mask'] # N X H X W
                        curr_boxes = outputs['individual_losses']['segm_loss_boxes'] # N X 4
                        curr_indices = outputs['individual_losses']['segm_loss_img_indices']
                        curr_counts ={}
                        curr_indices_counts = []
                        for i in curr_indices:
                            curr_counts[i] = curr_counts.get(i,0)
                            curr_indices_counts.append((i,curr_counts[i]))
                            curr_counts[i] += 1
                        max_round = max(curr_counts.values())
                        output_mask_ids = [(x[0][0],x[1][1][1]) for x in all_mask_decode] # list int
                        assert len(curr_masks) == len(output_mask_ids)
                        if len(all_mask_encode) > 0 or len(all_box_encode) > 0:
                            mask_image_processor = self.model.get_segmentator().process_images      # resize original_image --> sam / hipie decoder size
                            clip_image_processor = self.model.get_vision_tower().image_processor    # resize original_image --> clip encoder size
                            ref = original_inputs['images']['image']['pixel_values']
                            device = ref.device
                            dtype = ref.dtype
                            # NOTE:
                            # - curr_masks (hipie decoder): num_masks x 1024 x 1024        where num_masks = B*L = sum([n_masks for conv in batch])
                            # - curr_masks (sam decoder)  : num_masks x 1 x 1024 x 1024
                            for i in range(len(curr_masks)):
                                curr_mask = curr_masks[i]                                                                                   # 1024 x 1024 (hipie) (sam: 1 x 1024 x 1024)
                                mask_image = np.array(Image.open(all_mask_decode[i][1][1][0]['image_path']).convert('RGB'))                 # h x w x 3          (h,w original image size)
                                fake_masks = torch.zeros(mask_image.shape[:2]).float()                                                      # h x w
                                resized_image = mask_image_processor(mask_image,[fake_masks,fake_masks])['image'].permute(1,2,0).numpy()    # 1024 x 1024 x 3

                                # NOTE: SAM decoder's pre-processor performs normalize + resize to 1024x1024
                                if len(curr_mask.shape) == 3:                                                                               # detect that sam decoder is used
                                    curr_mask = curr_mask[0]                                                                                # 1024 x 1024       (remove singleton dimension from sam's mask output)
                                    resized_image = resized_image.transpose(2,0,1)                                                          # HWC --> CHW
                                    resized_image = torch.tensor(resized_image) * mask_image_processor.pixel_std + mask_image_processor.pixel_mean      # undo processor's normalization, pixel mean, std has shape 3x1x1 (matches resize_image shape CHW)
                                    resized_image = resized_image.numpy().astype(np.uint8)                                                  # CHW
                                    resized_image = resized_image.transpose(1,2,0)                                                          # CHW --> HWC

                                image_masked = cv2.bitwise_and(resized_image, resized_image, mask=curr_mask.astype(np.uint8))               # 1024 x 1024 x 3
                                assert resized_image.shape[0] == resized_image.shape[1]
                                (x0,y0,x1,y1) = np.clip(curr_boxes[i].astype(int),0,resized_image.shape[0])
                                x1 = np.clip(x1,x0+2,resized_image.shape[0])
                                y1 = np.clip(y1,y0+2,resized_image.shape[0])

                                image_masked_cropped = image_masked[y0:y1,x0:x1]
                                max_width = max(x1-x0,y1-y0)
                                image_masked_cropped_padded = np.zeros((max_width,max_width,image_masked.shape[-1]),dtype=image_masked.dtype)
                                image_masked_cropped_padded[:image_masked_cropped.shape[0],:image_masked_cropped.shape[1]] = image_masked_cropped
                                #Image.fromarray(image_masked_cropped_padded).save('test.jpg')
                                mask_encode_inputs = clip_image_processor(Image.fromarray(image_masked_cropped_padded))
                                mask_encode_inputs = torch.tensor(mask_encode_inputs.pixel_values[0]).to(device=device,dtype=dtype)
                                bbox_coords_sam = torch.tensor([y0,x0,y1,x1]) / 1024.0 # 1 1024

                                _d,_e = output_mask_ids[i]
                                for (a,b),c in all_mask_encode_id_to_data.items():
                                    if a == _d and c == _e:
                                        payload = ('mask-encode',[
                                            mask_encode_inputs,
                                            c
                                        ])
                                        replace_dict[(a,b)] = payload
                                for (a,b),c in all_box_encode_id_to_data.items():
                                    if a == _d and c == _e:
                                        payload = ('bbox-encode',[
                                            bbox_coords_sam,
                                            c
                                        ])
                                        replace_dict[(a,b)] = payload


                img_indices = outputs['individual_losses']['segm_loss_img_indices']         # N
                inter_per_mask = outputs['individual_losses']['segm_loss_inter_per_mask']   # N
                union_per_mask = outputs['individual_losses']['segm_loss_union_per_mask']   # N
                iou_per_mask = outputs['individual_losses']['segm_loss_iou_per_mask']       # N
                iou_per_box = outputs['individual_losses']['segm_loss_box_ious']            # N

                # len(iou_per_mask) = batch_size * num_rounds (N)
                # set eval batch size is to 1, so len(iou_per_mask) = num_rounds
                if (visualizations_dir := self.data_args.val_results_visualizations_dir) and np.mean(iou_per_mask)>= 0.7:
                    os.makedirs(visualizations_dir, exist_ok=True)
                    assert self.args.per_device_eval_batch_size == 1
                    input_counter += 1

                    ### Save conv ###
                    # get mask encode ref
                    mask_encode_ref = batch['extra_replacement']['mask_encode_ref'][0]  # select 0th instance in batch
                    # mask_encode_ref = [-1] + mask_encode_ref    # pad -1 for 1st round (no need for padding for v3 data, need padding for older data)

                    # get conversation id (entry index in conv json file)
                    conv_id = batch['extra_replacement']['conv_ids'][0]     # select 0th instance in batch

                    # save entire conversation
                    conv_txt = self.tokenizer.batch_decode(batch['input_ids'])[0]
                    conv_txt = conv_txt.replace("<video>", "")
                    conv_txt = conv_txt.replace("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", "")
                    curr_input_conversation = {}
                    for round_idx, round_txt in enumerate(re.findall(r'USER:(.*?).ASSISTANT', conv_txt)):
                        curr_input_conversation[f'round_{round_idx}'] = round_txt
                    image_path = outputs['individual_losses']['mask_data'][0]['image_path']
                    debug_payload = {
                        'conv_id' : conv_id,
                        'image_path' : image_path,
                        'conversation' : curr_input_conversation,
                        'encode_ref' : mask_encode_ref,
                        'mask_iou' : iou_per_mask.tolist()
                    }
                    conv_len = len(curr_input_conversation)
                    sub_dir = f'{conv_len}_rounds_conv'
                    if not os.path.exists(os.path.join(visualizations_dir, sub_dir)):
                        os.makedirs(os.path.join(visualizations_dir, sub_dir))
                    json.dump(debug_payload, open(os.path.join(visualizations_dir, sub_dir, f'{input_counter}.json'), "w"), indent=4)

                    ### Save masks ###
                    predicted_masks = outputs['individual_losses']['segm_loss_mask']
                    predicted_boxes = outputs['individual_losses']['segm_loss_boxes']
                    mask_data_list = outputs['individual_losses']['mask_data']

                    # iterate over each round's predicted masks in current input conversation
                    for round_idx, (pred_mask, pred_box, mask_data) in enumerate(
                        zip(predicted_masks, predicted_boxes, mask_data_list)
                    ):
                        image_path = mask_data['image_path']
                        (h, w) = mask_data['input_size']

                        gt_mask_uint8 = mask_data['mask'].reshape(1024,1024).detach().cpu().numpy()
                        gt_mask_bool = gt_mask_uint8.astype(bool)

                        # processors
                        mask_image_processor = self.model.get_segmentator().process_images      # resize original_image --> hipie decoder size
                        clip_image_processor = self.model.get_vision_tower().image_processor    # resize original_image --> clip encoder size

                        # resize image
                        original_image = np.array(Image.open(image_path).convert('RGB'))
                        fake_masks = torch.zeros(original_image.shape[:2]).float()
                        resized_image = mask_image_processor(original_image,[fake_masks,fake_masks])['image'].permute(1,2,0).numpy()

                        # Display segmentation mask on top of image
                        image_with_pred_mask = resized_image.copy()
                        image_with_gt_mask = resized_image.copy()

                        image_with_pred_mask[pred_mask] = (
                            resized_image * 0.5
                            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                        )[pred_mask]

                        image_with_gt_mask[gt_mask_bool] = (
                            resized_image * 0.5
                            + gt_mask_uint8[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                        )[gt_mask_bool]

                        # save predicted mask of current round
                        # im = Image.fromarray(image_with_mask[:h, :w, :])
                        # im.save(os.path.join(visualizations_dir, sub_dir, f'{input_counter}_round_{round_idx}.jpg'))

                        pred_im = image_with_pred_mask[:h, :w, :]
                        gt_im = image_with_gt_mask[:h, :w, :]


                        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                        axs[0].imshow(pred_im)
                        axs[0].set_title('Predicted Mask')
                        axs[0].axis('off')
                        axs[1].imshow(gt_im)
                        axs[1].set_title('GT Mask')
                        axs[1].axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(visualizations_dir, sub_dir, f'{input_counter}_round_{round_idx}.jpg'))
                        plt.close()

                # Edge case (when eval batch size = 1)
                if not isinstance(img_indices, Iterable):
                    img_indices = [img_indices]
                if not isinstance(inter_per_mask, Iterable):
                    inter_per_mask = [inter_per_mask]
                if not isinstance(union_per_mask, Iterable):
                    union_per_mask = [union_per_mask]
                if not isinstance(iou_per_mask, Iterable):
                    iou_per_mask = [iou_per_mask]
                if not isinstance(iou_per_box, Iterable):
                    iou_per_box = [iou_per_box]
                groupped_by_round = self.group_by_round(
                    img_indices,
                    inter_per_mask,
                    union_per_mask,
                    iou_per_mask,
                    iou_per_box
                )

                n_rounds = len(groupped_by_round['inter_per_round'])
                # Single-round
                if len(groupped_by_round['inter_per_round']) == 1:
                    # print("SINGLE ROUND")
                    select_idx = 0
                    assert np.all(inter_per_mask == np.array(groupped_by_round['inter_per_round'][0]))
                    assert np.all(union_per_mask == np.array(groupped_by_round['union_per_round'][0]))
                    assert np.all(iou_per_mask == np.array(groupped_by_round['mask_iou_per_round'][0]))
                    assert np.all(iou_per_box == np.array(groupped_by_round['box_iou_per_round'][0]))
                # Multi-round: only visualize 2nd round metric while training

                for i in range(n_rounds):
                    select_idx = i
                    if i not in metric_per_round:
                        metric_per_round[i] = metric_dict_prototype()
                    curr_round_metric = metric_per_round[i]
                    inter_selected=groupped_by_round['inter_per_round'][select_idx]
                    union_selected=groupped_by_round['union_per_round'][select_idx]
                    mask_ious_selected=groupped_by_round['mask_iou_per_round'][select_idx]
                    box_ious_selected=groupped_by_round['box_iou_per_round'][select_idx]

                    # aggregate metrics using only those from selected rounds
                    curr_round_metric['all_int'] += np.sum(inter_selected) / (1024*1024)
                    curr_round_metric['all_union'] += np.sum(union_selected) / (1024*1024)
                    curr_round_metric['true_pos'] += (np.array(box_ious_selected) > 0.5).sum()
                    curr_round_metric['all_ious'] += np.sum(mask_ious_selected)
                    curr_round_metric['n_masks'] += len(mask_ious_selected)
                    curr_round_metric['n_boxes'] += len(box_ious_selected)
                    assert curr_round_metric['n_masks'] == curr_round_metric['n_boxes']
                    payload = {
                        f"eval/P@0.5_round{i}": curr_round_metric['true_pos'] / (1e-10+curr_round_metric['n_boxes']),
                        f"eval/mIoU_round{i}": curr_round_metric['all_ious'] / (1e-10+curr_round_metric['n_masks']),
                        f"eval/oIoU_round{i}": curr_round_metric['all_int'] / (1e-10+curr_round_metric['all_union']),
                    }
                    str_x = f"Eval {idx}: / {n_eval}: Round {i}/{n_rounds} {payload}"
                    print(str_x)
                ''' Old impl
                replace:
                    box_ious --> box_ious_selected
                    per_mask_ious --> mask_ious_selected
                    inter (all masks) --> inter_selected
                    union (all masks) --> union_selected

                (old) inter, union, ious sum over all masks:
                    conv_1 masks: [round_1, round_2, round_3]
                    conv_2 masks: [round_1, round_2]
                    conv_3 masks: [round_1, round_2, round_3, round_4]
                (new) only sum over round_2 masks (2nd vertical column)
                '''


        loaded_checkpoint =  self.model_args.load if self.model_args.load else 'None'
        final_results_str = 'val data: ' + self.data_args.val_dataset + '\n'
        final_results_str += 'checkpoint: ' + loaded_checkpoint + '\n'
        final_results_str += 'eval mode: ' + ('ar eval' if ar_decoding else 'gt forcing') + '\n\n'
        for i,curr_round_metric in metric_per_round.items():
            state = torch.tensor([curr_round_metric[k] for k in ['true_pos','n_boxes','all_ious','n_masks','all_int','all_union']]).cuda()
            reduced_state = self.accelerator.reduce(state,reduction='sum')
            reduced_state = reduced_state.detach().cpu()
            true_pos,n_boxes,all_ious,n_masks,all_int,all_union =reduced_state.cpu().numpy()
            dist.barrier()
            metrics = {
                        f"eval/P@0.5_round{i}": true_pos / (1e-10+n_boxes),
                        f"eval/mIoU_round{i}": all_ious / (1e-10+n_masks),
                        f"eval/oIoU_round{i}": all_int / (1e-10+all_union),
                    }
            payload = dict(
                    global_step=self.state.global_step,
                    **metrics
                )
            str_x = f"**GATHERED** Eval Result for Round {i}: (Step {self.state.global_step}): {payload}"
            if (self.args.local_rank == 0 or self.args.local_rank == -1 ):
                if wandb.run is not None:
                    wandb.log(payload)
                print(str_x)
            final_results_str += str_x + '\n'
        final_results_str += '\n'
        dist.barrier()
        print(f"Eval Finished: (Step {self.state.global_step}")

        # write resuslt string to file
        if out_file := self.data_args.val_results_save_file:
            if (self.args.local_rank == 0 or self.args.local_rank == -1 ):
                out_dir = os.path.dirname(out_file)
                os.makedirs(out_dir, exist_ok=True)
                with open(out_file, "a") as out_file:
                    out_file.write(final_results_str)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # For debugging stuck:
        # conv_ids = inputs['extra_replacement']['conv_ids']
        # print("Iter:", self.state.global_step, "Rank:", self.args.local_rank, "Conv ids:", conv_ids)

        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        log_output = False
        outputs = None
        if self.state.global_step % 5 == 0:
            inputs['return_generations'] = True
            log_output = True
        # print(f"-------RANK {self.accelerator.process_index} Dataset_IDX {inputs['dataset_index']}--------")
        if log_output:
            with self.compute_loss_context_manager():
                loss,outputs = self.compute_loss(model, inputs,return_outputs=True)
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        if outputs is not None:
            payload = dict(
                global_step=self.state.global_step,
                **outputs['individual_losses']
            )

            # If dtype is bfloat16, convert to float32
            def bfloat_to_float32(d: dict) -> dict:
                for k, v in d.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                        d[k] = v.float()
                    elif isinstance(v, dict):
                        d[k] = bfloat_to_float32(v)
                    elif isinstance(v, list):
                        d[k] = [bfloat_to_float32(item) for item in v]
                    elif isinstance(v, tuple):
                        d[k] = tuple(bfloat_to_float32(item) for item in v)
                return d

            payload = bfloat_to_float32(payload)

            if (self.args.local_rank == 0 or self.args.local_rank == -1 ) and outputs is not None:
                if wandb.run is not None:
                    wandb.log(payload)
                print(payload)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                5.0,
                            )


        return loss.detach() / self.args.gradient_accumulation_steps

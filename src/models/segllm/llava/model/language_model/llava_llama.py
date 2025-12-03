#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union,Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast,ModelOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX
from transformers.modeling_outputs import BaseModelOutputWithPast
import logging
import numpy as np
from transformers.models.llama.modeling_llama import LlamaAttention
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
    def _prepare_decoder_attention_mask():
        return None

from llava.constants import REPLACEMENT_TYPE

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    base_model_prefix = "model"
    config_class = LlavaConfig

    def __init__(self, config, use_last_predicted_mask: bool = True):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.lm_head_img = nn.Linear(3, config.vocab_size, bias=False) # FIXME: Add config
        # Initialize weights and apply final processing
        self.post_init()
        self.warned_mask_targets_aux_inference = False
        self.use_last_predicted_mask = use_last_predicted_mask

    def get_model(self):
        return self.model

    def process_extra_replacement_data(self,data,ref):
        '''
        List[Union[torch.Tensor,Query]]
        '''
        REMAIN_LIST = ['mask-decode']
        final_tensors = []
        all_job_list = {}
        all_job_result = {}
        remaining_jobs = {}
        for sample_idx,row in enumerate(data):
            if isinstance(row,torch.Tensor):
                final_tensors.append(row)
            elif isinstance(row,list):
                for b in  row:
                    query,args = b
                    if query in REMAIN_LIST:
                        if query not in remaining_jobs:
                            remaining_jobs[query] = []
                        if query == 'mask-decode':
                            args[0]['sample_idx'] = sample_idx
                        remaining_jobs[query].append((query,args))
                        final_tensors.append(torch.zeros(1,self.config.hidden_size).to(ref))
                        continue
                    if query not in all_job_list:
                        all_job_list[query] = []
                    idx = len(all_job_list[query])
                    all_job_list[query].append(args)
                    final_tensors.append((query,idx))
            else:
                raise NotImplemented
        ## process
        for q,v in all_job_list.items():
            with torch.autocast(device_type = ref.device.type):
                if q == 'image-encode':
                    all_job_result[q] = self.encode_images(torch.stack(v).to(ref)) # N L D
                elif q == 'mask-encode':
                    v = [x[0] for x in v]
                    all_job_result[q] = self.encode_images(torch.stack(v).to(ref),features='cls') # N L D
                    if self.training:
                        dropout = torch.rand(all_job_result[q].shape[0],1,1) > 0.5
                        dropout = dropout.bool().int().to(all_job_result[q])
                        all_job_result[q] =  all_job_result[q] * dropout
                    else:
                        all_job_result[q] =  all_job_result[q] * 0.5


                elif q == 'bbox-encode':
                    v = [x[0] for x in v]
                    v_enc =  self.get_model().mask_enc_head(torch.stack(v).to(ref))     # project (N, 4) --> (N, D)
                    if self.training:
                        dropout = torch.rand(v_enc.shape[0],1) > 0.5
                        dropout = dropout.bool().int().to(v_enc)
                        v_enc = v_enc * dropout
                    else:
                        v_enc = v_enc * 0.5
                    all_job_result[q] = v_enc.unsqueeze(1)                              # (N, D) --> (N, 1, D)       N L D
                else:
                    raise NotImplemented
        for i in range(len(final_tensors)):
            if isinstance(final_tensors[i],tuple):
                query,idx = final_tensors[i]
                final_tensors[i] = all_job_result[query][idx]
                if len(final_tensors[i].shape) == 1:
                    final_tensors[i] = final_tensors[i][None]
        return torch.cat(final_tensors),remaining_jobs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Dict] = None,
        return_dict: Optional[bool] = None,
        generation_target: Optional[Dict] = None,
        return_generations=True,
        extra_inputs=None,
        extra_replacement=None,
        dataset_index=None, # hack, do not delete
        perform_segmentation: bool = True,
        **kwargs # Hack to catch any extra dataset keys without erroring
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # encode vae for
        replacement_mask = torch.zeros_like(input_ids,dtype=bool).to(input_ids.device)
        raw_input_ids = input_ids
        if labels is not None:
            raw_labels = labels # N L
        else:
            raw_labels = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images,novision=True)
        remaining_jobs = {} # list of additional targets
        if extra_replacement is not None:
            if not isinstance(extra_replacement['data'],torch.Tensor):
                lazy_encode = False
                extra_replacement['data'],remaining_jobs= self.process_extra_replacement_data(extra_replacement['data'],ref=torch.zeros(1).to(inputs_embeds))
            else:
                lazy_encode = True
            if self.training or labels is not None:
                extra_replacement_mask = (raw_input_ids == self.DEFAULT_VIDEO_TOKEN_IDX ) | ((raw_input_ids ==self.DEFAULT_SEGMENTATION_TOKEN_IDX )& (raw_labels == self.DEFAULT_SEGMENTATION_TOKEN_IDX) ) # | (
                if extra_replacement['mask'].shape[0] != inputs_embeds[extra_replacement_mask].shape[0]:
                    print("SKIPPED",extra_replacement['mask'].shape[0], inputs_embeds[extra_replacement_mask].shape[0])
                    # commnet this line for image generation where num tokens is expected to be larger
                    raise ValueError("Mismatch replacement shape, this should not happen for segmentation data. Plesae double check number of embeddings matches number of tokens to replace.")
                    extra_replacement['mask'] = torch.zeros(inputs_embeds[extra_replacement_mask].shape[0]).to(extra_replacement['mask'])
                z = torch.zeros_like(inputs_embeds)
                if lazy_encode:
                    z2 = self.get_model().mm_projector(
                        extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                else:
                    z2 = extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]
                a,b = torch.where(extra_replacement_mask)
                z[a[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT],b[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]] += z2
                inputs_embeds[extra_replacement_mask][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = 0.0
                z = z + inputs_embeds
                inputs_embeds = z
                # print("Replaced:",len(extra_replacement['mask']==REPLACEMENT_TYPE.INPUT),len(extra_replacement['mask']==REPLACEMENT_TYPE.GEN))
                extra_tgt_mask = (extra_replacement['mask']==REPLACEMENT_TYPE.BASE )| (extra_replacement['mask']==REPLACEMENT_TYPE.GEN)
                extra_replacement_gt = extra_replacement['data'][extra_tgt_mask]
                loss_fn_extra = nn.L1Loss()
                if extra_replacement_gt.shape[0]==0:
                    loss_fn_extra = None
            else:
                assert labels is None
                extra_replacement_mask = (raw_input_ids == self.DEFAULT_VIDEO_TOKEN_IDX )
                print(len(extra_replacement['mask']==REPLACEMENT_TYPE.INPUT))

                z = torch.zeros_like(inputs_embeds)
                if lazy_encode:
                    z2 = self.get_model().mm_projector(
                        extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                else:
                    z2 = extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]
                print("z2",z2)
                a,b = torch.where(extra_replacement_mask)
                a = a[:extra_replacement['mask'].shape[0]]
                b = b[:extra_replacement['mask'].shape[0]]
                z[a[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT],b[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]] += z2
                inputs_embeds[extra_replacement_mask][:extra_replacement['mask'].shape[0]][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = 0.0
                z = z + inputs_embeds
                inputs_embeds = z
                print("HERE")
                #print(inputs_embeds[extra_replacement_mask][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])

                # inputs_embeds[extra_replacement_mask][:extra_replacement['mask'].shape[0]][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = self.get_model().vae_projector_image(
                #     extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT].to(inputs_embeds))
            # z.sum().backward()



        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
           # replacement_mask=replacement_mask, # allow looking into future for images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        img_decode = None
        aud_decode=None
        individual_losses = {}
        extra_gen = None
        extra_gen_idx = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            hidden_states.shape

            loss_fct = CrossEntropyLoss()

                #prediction = logits_img.argmax(-1)

            # Flatten the tokens

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss_lang = loss.detach().item()
            individual_losses['loss_lang'] = loss_lang


            # Compute per-sequence log-probabilities for multiple-choice evaluation
            with torch.no_grad():
                bsize, seq_len = logits.shape[0], logits.shape[1] - 1 # - 1 for shift
                log_probs = -F.cross_entropy(shift_logits, shift_labels, reduction='none')
                log_probs = log_probs.reshape(bsize, seq_len)

                n_tokens = (labels[..., 1:] != -100).int().sum(dim=1) # (bsize,)
                log_probs = log_probs.sum(dim=1) / n_tokens # (bsize,)

                individual_losses['log_probs'] = log_probs.cpu()

            if remaining_jobs: # Mask decoding requires hidden states of language models, so is deferred until after the forward pass
                for job,job_tgt in remaining_jobs.items():
                    job_tgt: list[tuple[str, tuple[dict, None]]] # e.g. [('mask-decode', (dict, None))]

                    if job == 'mask-decode':
                        if not perform_segmentation:
                            individual_losses = None
                            continue

                        seg_mask_idx =  ((raw_input_ids ==self.DEFAULT_SEGMENTATION_TOKEN_IDX )& (raw_labels == self.DEFAULT_SEGMENTATION_TOKEN_IDX))
                        seg_tokens_per_conversation = seg_mask_idx.sum(dim=1) # B

                        # ANSEL Here, L is the number of <seg> tokens in the batch
                        mask_preds = self.get_model().segmentator_predictor(hidden_states[:,:-1][seg_mask_idx[:,1:]].contiguous()) # L X D (project [SEG] hidden states transformer dim --> SAM prompt dim)
                        job_tgt = list([x[1][0] for x in job_tgt])
                        all_seg_images = np.unique([x['image_path'] for x in job_tgt]) # unique_image_path

                        all_seg_images_lookup = {v:idx for idx,v in enumerate(all_seg_images)}
                        image_indices = np.array([all_seg_images_lookup[x['image_path']] for x in job_tgt]) # len(image_indices) = L
                        sample_indices = np.array([x['sample_idx'] for x in job_tgt])
                        job_tgt = np.array(job_tgt)
                        mask_inputs = dict(
                            images=[],          # input images          B x 3 x 1024 x 1024 (B = num convs = num images)
                            prompts=[],         # [SEG] hidden states   B x num_masks x 512 (prompt_dim * num_tokens)
                            targets=[],         # GT masks              B x num_masks x 1 x 1024 x 1024
                            prev_masks=[],       # GT REF masks          B x num_masks x 1 x 1024 x 1024,
                            sample_indices=sample_indices,
                        )

                        # Original code
                        # mask_data_all = []
                        # for idx,v in enumerate(all_seg_images):           # interate through each conv in batch, correspond to different image
                        #     mask_prompts = mask_preds[image_indices==idx] # L X D (for the current image)
                        #     mask_data = job_tgt[image_indices==idx]
                        #     mask_targets = torch.stack([x['mask'] for x in mask_data]) # L x 1 x H X W
                        #     # mask_targets_aux = torch.stack([x['aux_mask'] for x in mask_data]) # L x H X W

                        #     # Prepare mask_targets_aux as predictions (for inference) or ground truth (teacher forcing for training)
                        #     mask_targets_aux = torch.cat([
                        #         mask_targets.new_zeros(1, 1, mask_targets.shape[-2], mask_targets.shape[-1]),
                        #         mask_targets[:-1, ...]
                        #     ], dim=0)

                        #     # mask_inputs['images'].append(mask_data[0]['image'])     # 3 x 1024 x 1024
                        #     mask_inputs['images'].append(mask_data[0]['image'].cuda())     # 3 x 1024 x 1024
                        #     mask_inputs['prompts'].append(mask_prompts)             # L x 512 (or 256)      (each image/conv has L masks)
                        #     # mask_inputs['targets'].append(mask_targets)             # L x 1 x 1024 x 1024   GT tgt masks
                        #     mask_inputs['targets'].append(mask_targets.cuda())             # L x 1 x 1024 x 1024   GT tgt masks
                        #     # mask_inputs['prev_masks'].append(mask_targets_aux)      # L x 1 x 1024 x 1024   GT ref masks
                        #     mask_inputs['prev_masks'].append(mask_targets_aux.cuda())      # L x 1 x 1024 x 1024   GT ref masks

                        #     mask_data_all.extend(mask_data.tolist())

                        # with torch.autocast(device_type=hidden_states.device.type):
                        #     _, mask_loss = self.get_segmentator()(**mask_inputs)
                        #     if self.training:
                        #         for k,v in mask_loss.items():
                        #             individual_losses[f'segm_loss_{k}']=v.item()
                        #     else:
                        #         for k,v in mask_loss.items():                   # eval, there are additional metrics
                        #             if k in [
                        #                 'mask',
                        #                 'iou_per_mask',
                        #                 'inter_per_mask',
                        #                 'union_per_mask'                        # these are tensors, cannot take .item()
                        #             ]:
                        #                 individual_losses[f'segm_loss_{k}']=v
                        #             else:
                        #                 try:
                        #                     individual_losses[f'segm_loss_{k}']=v.item()
                        #                 except:
                        #                     individual_losses[f'segm_loss_{k}']=v

                        #         individual_losses['mask_data'] = mask_data_all        # mask_data, not from SAM output (maybe GT masks)

                        # loss += mask_loss['total_loss']

                        mask_data_all = []
                        if self.training:
                            for idx,v in enumerate(all_seg_images):           # interate through each conv in batch, correspond to different image
                                mask_prompts = mask_preds[image_indices==idx] # L X D (for the current image)
                                mask_data = job_tgt[image_indices==idx]
                                mask_targets = torch.stack([x['mask'] for x in mask_data]) # L x H X W
                                # mask_targets_aux = torch.stack([x['aux_mask'] for x in mask_data]) # L x H X W

                                # Prepare mask_targets_aux as predictions (for inference) or ground truth (teacher forcing for training)
                                mask_targets_aux = torch.cat([
                                    mask_targets.new_zeros(1, 1, mask_targets.shape[-2], mask_targets.shape[-1]),
                                    mask_targets[:-1, ...]
                                ], dim=0)

                                # mask_inputs['images'].append(mask_data[0]['image'])     # 3 x 1024 x 1024
                                mask_inputs['images'].append(mask_data[0]['image'].cuda())     # 3 x 1024 x 1024
                                mask_inputs['prompts'].append(mask_prompts)             # L x 512 (or 256)      (each image/conv has L masks)
                                # mask_inputs['targets'].append(mask_targets)             # L x 1 x 1024 x 1024   GT tgt masks
                                mask_inputs['targets'].append(mask_targets.cuda())             # L x 1 x 1024 x 1024   GT tgt masks
                                # mask_inputs['prev_masks'].append(mask_targets_aux)      # L x 1 x 1024 x 1024   GT ref masks
                                mask_inputs['prev_masks'].append(mask_targets_aux.cuda())      # L x 1 x 1024 x 1024   GT ref masks

                                mask_data_all.extend(mask_data.tolist())

                            with torch.autocast(device_type=hidden_states.device.type):
                                _, mask_loss = self.get_segmentator()(**mask_inputs)

                            for k,v in mask_loss.items():
                                individual_losses[f'segm_loss_{k}']=v.item()

                            loss += mask_loss['total_loss']

                        else:
                            # assert len(all_seg_images) == len(raw_input_ids), 'Need to assume that each image only appears once per batch for easy sequential decoding'

                            keys_to_average = set()
                            keys_to_concat = set()
                            for idx,v in enumerate(all_seg_images):           # interate through each conv in batch, correspond to different image
                                image_seg_tokens_mask = image_indices == idx

                                mask_prompts = mask_preds[image_seg_tokens_mask] # L X D (for the current image)
                                mask_data = job_tgt[image_seg_tokens_mask]

                                mask_targets = torch.stack([x['mask'] for x in mask_data]).cuda() # L x H X W
                                # mask_targets_aux = torch.stack([x['aux_mask'] for x in mask_data]) # L x H X W

                                # mask_inputs['images'].append(mask_data[0]['image'].cuda())     # 3 x 1024 x 1024
                                images = [mask_data[0]['image'].cuda()]

                                # mask_inputs['prompts'].append(mask_prompts)             # L x 512 (or 256)      (each image/conv has L masks)
                                # mask_inputs['targets'].append(mask_targets)             # L x 1 x 1024 x 1024   GT tgt masks
                                # mask_inputs['prev_masks'].append(mask_targets_aux.cuda())      # L x 1 x 1024 x 1024   GT ref masks

                                mask_data_all.extend(mask_data.tolist())

                                # Provide last predicted mask to mask decoder
                                # NOTE This logic allows us to extract the conversations corresponding to each image and correctly reset the
                                # last predicted mask on new conversations, but the HIPIESementator asserts that the number of sample indices should match
                                # the number of indices, making this a moot point (since it requires one conversation per image)
                                last_predicted_mask = mask_targets.new_zeros(1, 1, mask_targets.shape[-2], mask_targets.shape[-1])

                                image_conversation_indices = torch.tensor([ # Indices of conversations with the current image
                                    j['sample_idx'] for j in job_tgt if j['image_path'] == v
                                ]).unique()

                                seg_token_boundaries = seg_tokens_per_conversation[image_conversation_indices].cumsum(dim=0) # Num total [SEG] tokens as we move along conversations
                                conversation_offset = 0 # Track which conversation we are in

                                for i in range(len(mask_prompts)): # Number of [SEG] tokens in the current conversation/image
                                    prompts = [mask_prompts[i:i+1]]
                                    targets = [mask_targets[i:i+1]]
                                    local_sample_indices = sample_indices[i:i+1]
                                    prev_masks = [last_predicted_mask]

                                    mask_inputs = {
                                        'images': images,
                                        'prompts': prompts,
                                        'targets': targets,
                                        'prev_masks': prev_masks,
                                        'sample_indices': local_sample_indices
                                    }

                                    with torch.autocast(device_type=hidden_states.device.type):
                                        _, mask_loss = self.get_segmentator()(**mask_inputs)

                                    if self.use_last_predicted_mask:
                                        if i == seg_token_boundaries[conversation_offset] - 1: # New conversation; reset last predicted mask
                                            conversation_offset += 1
                                            last_predicted_mask = mask_targets.new_zeros(1, 1, mask_targets.shape[-2], mask_targets.shape[-1])
                                        else: # Same conversation; update last predicted mask
                                            last_predicted_mask = torch.from_numpy(mask_loss['mask']).unsqueeze(0).to(last_predicted_mask.device) # (1, 1, h, w)

                                    # TODO determine the shape of all of these tensors for storage
                                    for k, v in mask_loss.items():
                                        try:
                                            item = v.item()
                                            individual_losses.setdefault(f'segm_loss_{k}', []).append(item)
                                            keys_to_average.add(k)
                                        except:
                                            individual_losses.setdefault(f'segm_loss_{k}', []).append(v)
                                            keys_to_concat.add(k)

                                    # for k,v in mask_loss.items():                   # eval, there are additional metrics
                                    #     if k in [
                                    #         'mask',
                                    #         'iou_per_mask',
                                    #         'inter_per_mask',
                                    #         'union_per_mask'                        # these are tensors, cannot take .item()
                                    #     ]:
                                    #         individual_losses[f'segm_loss_{k}'] = v
                                    #     else:
                                    #         try:
                                    #             individual_losses[f'segm_loss_{k}'] = v.item()
                                    #         except:
                                    #             individual_losses[f'segm_loss_{k}'] = v

                                    loss += mask_loss['total_loss'] / len(mask_preds) # Weigh each mask loss equally

                            individual_losses['mask_data'] = mask_data_all        # mask_data, not from SAM output (maybe GT masks)

                            for k in keys_to_average:
                                individual_losses[f'segm_loss_{k}'] = np.mean(individual_losses[f'segm_loss_{k}'])

                            for k in keys_to_concat:
                                try:
                                    individual_losses[f'segm_loss_{k}'] = np.concatenate(individual_losses[f'segm_loss_{k}'])
                                except:
                                    individual_losses[f'segm_loss_{k}'] = np.stack(individual_losses[f'segm_loss_{k}'])



                    else:
                        raise ValueError
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            extra_gen=extra_gen,
            extra_gen_idx=extra_gen_idx,
            attentions=outputs.attentions,
            img_decode=img_decode,
            aud_decode=aud_decode,
            individual_losses=individual_losses,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "extra_replacement": kwargs.get("extra_replacement", None),
            }
        )
        return model_inputs

# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

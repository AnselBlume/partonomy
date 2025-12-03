# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
from torch import Tensor
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, TypedDict
import cv2
import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_IM_GEN_TOKEN,DEFAULT_AUDIO_GEN_TOKEN,DEFAULT_MSK_TOKEN,DEFAULT_IM_GEN_START_TOKEN,DEFAULT_AUDIO_GEN_START_TOKEN,DEFAULT_AUDIO_TOKEN,FAKE_SEGM_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import torch.distributed as dist
from llava.train.seg_register.register_dataset import Register as COCORegister
local_rank = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_gen: bool = field(default=False)
    mm_use_seg: bool = field(default=False)
    mm_use_bbox_encode: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="projection")
    segmentator: Optional[str] = field(default=None)
    from_huggingface: bool = field(default=False)
    dev: Optional[str] = field(default=None)
    load: Optional[str] = None



@dataclass
class DataArguments:
    image_folder: str = './images_folder'
    conversation_folder: str = './conversations_folder'
    annotation_folder: str = './annotations_folder'
    conversation_config: str = ''
    segmentation_config: str = ''
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    eval_num_forward: Optional[int] = field(default=None)
    image_grid_pinpoints: Optional[str] = field(default=None)
    output_text: Optional[bool] = False
    val_results_dir: Optional[str] = field(default=None)
    visualizations_dir: Optional[str] = field(default=None)
    load_data: Optional[int] = None
    limit_rounds: Optional[int] = None
    eval_use_gt_mask_encode: Optional[bool] = False
    inference_output_dir: Optional[str] = './inference_results'
    val_dataset:  Optional[str] = None
    val_results_save_file: Optional[str] = None
    val_results_visualizations_dir: Optional[str] = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    split_loading: bool = False
    eval_only: bool = False
    ar_eval: bool = False # only works for eval_only


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class Constant:
    MASK_ENCODE_LEN = 1
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   force_save=False):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False) and False: # Always save all
        # Only save Adapter
        keys_to_match = ['mm_projector']
        keys_to_match.extend(['embed_tokens', 'embed_in'])
        # if getattr(trainer.args, "use_im_start_end", False):
        #     keys_to_match.extend(['embed_tokens', 'embed_in'])
        keys_to_match.extend(['lm_head',])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save or force_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        #assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        if DEFAULT_IMAGE_TOKEN in source[0]['value']:
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_plain_gen(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image:bool,
    info:Dict,
) -> Dict:
    # add end signal and concatenate together
    #conv = conversation_lib.default_conversation.copy()
    conv = conversation_lib.conv_templates["vicuna_v1"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    conversations_raw = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            val = sentence['value']
            val = val.replace(DEFAULT_IM_GEN_TOKEN,DEFAULT_IM_GEN_START_TOKEN + DEFAULT_IM_GEN_TOKEN*info['generation_seq_len'])
            val = val.replace(DEFAULT_AUDIO_GEN_TOKEN,DEFAULT_AUDIO_GEN_START_TOKEN+DEFAULT_AUDIO_GEN_TOKEN*info['generation_seq_len'])

            conv.append_message(role, val)
        assert conv.sep is not None and conv.sep2 is not None
        conversations.append(conv.get_prompt().replace(FAKE_SEGM_TOKEN,DEFAULT_SEGMENTATION_TOKEN))
        conversations_raw.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations_raw, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        begin = [1]
        ll_ids = []
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            ll_ids.extend(tokenizer(rou).input_ids[1:])
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            rou = rou.replace(FAKE_SEGM_TOKEN,DEFAULT_SEGMENTATION_TOKEN)
            ignore_target = FAKE_SEGM_TOKEN in parts[1]
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) -1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if ignore_target:
                # print('--debug--, will not apply loss to wrong')
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
            else:
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX# target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        cur_len += 1
        target[cur_len:] = IGNORE_INDEX
        ll_ids.append(2)
        assert cur_len < tokenizer.model_max_length,"Out of Bounds!!!!!!!!!!!!!!!!!!!"
        if cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored) {len(rounds)}"
            )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    info={},
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # hack
    if info.get('generation'):
        return preprocess_plain_gen(sources, tokenizer,has_image,info)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

import re
import random

all_bracket_types = ['IMAGE256', 'MASK-ENCODE', 'BOX-ENCODE', 'MASK-DECODE']
def find_brackets(x):
    matches =  re.compile('\[[^\]]+\]').findall(x)
    # filter out any bad matches
    filtered_matches = []
    for bracket in matches:
        if any([x in bracket for x in all_bracket_types]):
            filtered_matches.append(bracket)
        else:
            print("Ignoring bad bracket:", bracket)
    return filtered_matches


def remove_prefix(x):
    # prefix = ['an image of','a photo of','a painting of','']
    # for p in prefix:
    #     x = x.replace(p,'')
    return x
from llava.constants import DEFAULT_VIDEO_TOKEN,REPLACEMENT_TYPE,DEFAULT_SEGMENTATION_INPUT_TOKEN,DEFAULT_SEGMENTATION_TOKEN
import numpy as np
def get_replacement_len(s):
    if 'IMAGE256' in s:
        return 256
    # elif 'MASK-ENCODE' in s:
    #     return Constant.MASK_ENCODE_LEN
    else:
        return 1
    # MASK-ENCODE and BBOX-ENCODE are separate replacement jobs, each gets replaced by len=1 token


import yaml
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 is_eval=False,
                 is_inference=False,
                 inference_conv=None,
                 debug_mode=False):
        super(LazySupervisedDataset, self).__init__()

        is_training = not (is_eval or is_inference)
        self.is_inference = is_inference
        self.is_eval = is_eval
        self.eval_use_gt_mask_encode = data_args.eval_use_gt_mask_encode
        self.debug_mode = debug_mode

        self.list_data_dict = []
        self.dataset_lengths = []

        if is_training:
            with open(data_args.conversation_config) as config_file:
                conv_config = yaml.safe_load(config_file.read())
            for dataset in conv_config['datasets']:
                file_name = dataset['name']
                file_path = os.path.join(data_args.conversation_folder, file_name)
                print(f" ---- Training: Loading {file_path.split('/')[-1]} conversations ----")
                with open(file_path, "r") as f:
                    if k := data_args.load_data:
                        to_extend = json.load(f)[:k]
                    else:
                        to_extend = json.load(f)
                    self.list_data_dict.extend(to_extend)
                    self.dataset_lengths.append(len(to_extend))
        elif is_eval:                                     # only load val dataset specified by data_args.val_dataset
            file_name = data_args.val_dataset
            conv_dir = data_args.conversation_folder
            if "all_data_mix_train" in conv_dir:
                conv_dir = conv_dir.replace("all_data_mix_train", "all_data_mix_val")
            file_path = os.path.join(conv_dir, file_name)
            print(f" ---- Validation: Loading {file_path.split('/')[-1]} conversations ----")
            with open(file_path, "r") as f:
                to_extend = json.load(f)
                self.list_data_dict.extend(to_extend)
                self.dataset_lengths.append(len(to_extend))
        elif is_inference:
            assert inference_conv is not None
            self.list_data_dict = [inference_conv]
            self.dataset_lengths.append(1)       # during inference, everything in one conversation

        # FOR DEBUGGIN:
        if num_rounds := data_args.limit_rounds:
            for entry in self.list_data_dict:
                entry['conversations'] = entry['conversations'][:2*num_rounds]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.TXT2TENSOR = dict()
        print('--------------------Dataset Lengths -----------------------')
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
    def get_bitmask_bbox_encode(self, image_file_lst):
        """Materialize encoder features referenced in the prompt.

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[int]]: Masked image
            crop, SAM-space bounding box, and the mask identifier.
        """
        image_file,dataset_name,mask_id = image_file_lst[0].split('|')

        # Eval or Inference case (use dummy mask on 1st forward, otherwise load gt for mask-encode)
        if (self.is_eval and not self.eval_use_gt_mask_encode) or dataset_name == 'INFERENCE':
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            processor = self.data_args.image_processor              # CLIP processor
            inputs = processor(Image.open(image_path))              # input image
            inputs = inputs.pixel_values[0]
            masked_instance_processed = torch.tensor(inputs)        # dummy mask-encode, just encode the image without masking or cropping
            bbox_coords_sam = torch.zeros(4)                        # dummy box-encode, all zeros
            return masked_instance_processed, bbox_coords_sam, mask_id

        # TODO: temp handle edge cases
        if mask_id == '' or mask_id == "'":         # this is the case for reason_seg sentences
            mask_id = None
        elif "_" in mask_id or "-" in mask_id:
            mask_id=mask_id
        else:
            mask_id = int(mask_id)
        image_folder = self.data_args.image_folder
        image_path = os.path.join(image_folder, image_file)
        image_path = image_path.replace('val2014', 'train2014')
        image_path = image_path.replace('new_', '')
        if 'VG_100K' in image_path:
            image_path = image_path.replace('./images_folder', './images_folder/vg')

        if not os.path.exists(image_path):
            image_path = image_path.replace('val2017', 'train2017')     # new edge case for lvis (val)
            image_path = image_path.replace('images/', 'object365/')
        assert os.path.exists(image_path)

        # Mask image with GT mask
        image = Image.open(image_path)
        (w, h) = image.size
        image = np.array(image.convert('RGB'))
        gt_mask = self.data_args.register.get_bitmask(
            dataset_name,
            mask_id,
            is_eval=self.is_eval,
            image_file=image_file.split("/")[-1],
            image_dim=(h,w)
        )
        image_masked = cv2.bitwise_and(image, image, mask=gt_mask)

        # Crop image with bbox and pad
        (x0,y0,x1,y1) = self.data_args.register.get_bbox(
            dataset_name,
            mask_id,
            is_eval=self.is_eval,
            image_file=image_file.split("/")[-1],
            image_dim=(h,w)
        )
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
        data_mask = processor(np.array(Image.open(image_path).convert('RGB')),
                            masks=[gt_mask,gt_mask])
        mask_bin = data_mask['mask']
        y0 = torch.where(mask_bin.sum((0,1)))[0].min()
        y1 = torch.where(mask_bin.sum((0,1)))[0].max()
        x0 = torch.where(mask_bin.sum((0,2)))[0].min()
        x1 = torch.where(mask_bin.sum((0,2)))[0].max()
        bbox_coords_sam = torch.tensor([x0,y0,x1,y1]) / 1024.0 # 1 1024

        return masked_instance_processed, bbox_coords_sam,mask_id

    def get_bitmask_decode(self, image_file_lst) -> tuple[dict, int]:
        """Decodes special [MASK-DECODE...] encodings in the prompt

        Returns:
            tuple[dict, int]: Processor-ready mask tensors and the
            resolved mask identifier.
        """
        #image_file,dataset_name,mask_id = image_file_lst[0].split('|')
        if ':' in image_file_lst[0]:
            # (reference mask decoding format)
            if len(re.findall(':', image_file_lst[0])) == 4:
                # 1 reference mask
                task_type,ref_mask_id,tgt_mask_id,image_file,dataset_name = image_file_lst[0].split(':')
            elif len(re.findall(':', image_file_lst[0])) == 5:
                # 2 reference masks
                task_type,ref_mask_id,ref_mask_id_2,tgt_mask_id,image_file,dataset_name = image_file_lst[0].split(':')
            else:
                raise ValueError("Base ref-mask decode format:", image_file_lst[0])
        elif '|' in image_file_lst[0]:
            # (no reference mask decoding format)
            image_file,dataset_name,tgt_mask_id = image_file_lst[0].split('|')
            task_type = 'none'
            ref_mask_id = None
        else:
            raise ValueError("Base decode format:", image_file_lst[0])
        # Inference case:
        if dataset_name == 'INFERENCE':
            # raise NotImplementedError("Obselete, do not use!!!")
            #TODO: @shaolun_zhang, please fix inference
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            (w, h) = image.size
            mask = np.ones((h, w, 1), dtype=np.uint8)                   # dummy mask as gt mask for inference
            processor = self.data_args.mask_processor                   # processor --> decoder dimension
            data = processor(np.array(image.convert('RGB')), masks=[mask,mask]) # expect list of masks [ref, gt]
            data['image_path'] = image_path
            data['task_type'] = task_type
            return data, tgt_mask_id # 1 1024     # Question: use mask_id_0 or mask_i_1

        # TODO: temp fix
        def process_mask_id(mask_id):
            if mask_id == '' or mask_id == "'" or mask_id == 'none' or mask_id == None:         # this is the case for reason_seg sentences
                mask_id = None
            elif "_" in mask_id or "-" in mask_id:
                mask_id=mask_id
            else:
                mask_id = int(mask_id)
            return mask_id
        ref_mask_id = process_mask_id(ref_mask_id)
        tgt_mask_id = process_mask_id(tgt_mask_id)

        image_folder = self.data_args.image_folder
        image_path = os.path.join(image_folder, image_file)

        # Edge case handling
        image_path = image_path.replace('val2014', 'train2014')
        image_path = image_path.replace('new_', '')
        if 'VG_100K' in image_path:
            image_path = image_path.replace('./images_folder', './images_folder/vg')

        if not os.path.exists(image_path):
            image_path = image_path.replace('val2017', 'train2017')     # new edge case for lvis (val)
            image_path = image_path.replace('images/', 'object365/')
        assert os.path.exists(image_path)

        image = Image.open(image_path)
        (w, h) = image.size
        image = np.array(image.convert('RGB'))
        tgt_mask = self.data_args.register.get_bitmask(
            dataset_name,
            tgt_mask_id,
            is_eval=self.is_eval,                   # pass in is_eval flag, signals which seg anno file to use (train vs. eval)
            image_file=image_file.split("/")[-1],
            image_dim=(h,w)
        )
        if ref_mask_id:
            ref_mask = self.data_args.register.get_bitmask(
                dataset_name,
                ref_mask_id,
                is_eval=self.is_eval,
                image_file=image_file.split("/")[-1],
                image_dim=(h,w)
            )                                       # load mask from seg register
        else:
            ref_mask = np.zeros_like(tgt_mask)
        masks = [ref_mask,tgt_mask]
        processor = self.data_args.mask_processor
        data = processor(np.array(Image.open(image_path).convert('RGB')),
                            masks=masks)
        data['image_path'] = image_path
        data['task_type'] = task_type
        return data, tgt_mask_id

    def build_query(self, x: str) -> \
        tuple[str, Tensor] \
        | tuple[str, tuple[Tensor, int]] \
        | tuple[str, tuple[Tensor, int, int]] \
        | tuple[str, tuple[dict, str]] \
    :
        '''
        Dispatches serialized placeholders to modality-specific loaders.

        Arguments:
            x (str): The serialized placeholder.

        Returns:
            tuple[str, Any]: The modality key and associated payload.
            e.g. ('image-encode', data)
            e.g. ('mask-encode', (masked_instance_processed, mask_id))
            e.g. ('bbox-encode', (bbox_coords_sam, mask_id))
            e.g. ('mask-decode', (data, tgt_mask_id))

        '''

        data = torch.zeros(1,3,224,224)
        if image_file_lst := re.compile('IMAGE256:(.*)$').findall(x):
            image_file = image_file_lst[0]
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)

            if not os.path.exists(image_path):
                image_path = image_path.replace('val2017', 'train2017')     # new edge case for lvis (val)
                image_path = image_path.replace('images/', 'object365/')
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
        elif image_file_lst := re.compile('MASK-ENCODE:(.*)$').findall(x):
            masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_file_lst)
            return 'mask-encode', (masked_instance_processed,mask_id)
        elif image_file_lst := re.compile('BOX-ENCODE:(.*)$').findall(x):
            masked_instance_processed, bbox_coords_sam,mask_id = self.get_bitmask_bbox_encode(image_file_lst)
            return 'bbox-encode', (bbox_coords_sam,mask_id)
        elif image_file_lst := re.compile('MASK-DECODE:(.*)$').findall(x):
            data, tgt_mask_id = self.get_bitmask_decode(image_file_lst)
            return 'mask-decode',(data,tgt_mask_id) # 1 1024
        else:
            raise NotImplementedError(x)
    def __len__(self):
        return len(self.list_data_dict)

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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        _dataset_idx = self.dataset_index(i)
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        inputs = None
        extra_inputs = []
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            inputs = {
                "image": [image_path,]
            }
            processor = self.data_args.image_processor
            inputs = processor(inputs)
            inputs['image']['pixel_values'] = inputs['image']['pixel_values'].squeeze(0) # hack
            sources_p = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args) # move <image> to the start of all data
        elif 'multimodal_input' in sources[0]:
            multi_input = sources[0]['multimodal_input']
            modality = multi_input['type']
            assert modality!= 'image'
            if modality == 'audio':
                raise NotImplemented
            sources_p = copy.deepcopy([e["conversations"] for e in sources])
            sources_p[0][0]['value'] = sources_p[0][0]['value'].replace(DEFAULT_AUDIO_TOKEN,DEFAULT_AUDIO_TOKEN*8)
            # if DEFAULT_AUDIO_TOKEN in sources_p[0][0]['value'] :
            #     raise AssertionError(sources_p[0][0]['value'])
        else:
            sources_p = copy.deepcopy([e["conversations"] for e in sources])
        do_generation=False
        info = {}
        image_folder = self.data_args.image_folder
        if sources[0].get('task') == 'generation':
            do_generation = True
            raise NotImplemented

        # Modification
        mask_encode_ref = []                 # a list of indices to keep track which round's mask output is THIS mask-encode referring to

        extra_replacement_mask = []
        delayed_process = False
        if sources[0].get('task') == 'any2any':
            info['generation_seq_len'] = 1
            replacement = []
            replacement_mask = [] # loss mask
            base = sources[0]['base']
            info['generation'] = True
            drop_base = random.random() < 0.2
            all_tgts = {x[1]:x for x in (sources[0]['added'] if sources[0]['added'] else [])}
            adds = []
            raw_val = []
            for turn in sources_p[0]:
                src = turn['from']
                val = turn['value']
                if drop_base:
                    val = val.replace('<base>','<base_null>')
                if src == 'human':
                    matches = find_brackets(val)
                    for prompt in matches: # list of str wit '[]'
                        if prompt in all_tgts:
                            set_instance = True
                        else:
                            set_instance = False
                        prompt_clean = prompt[1:-1]
                        if clean(prompt_clean) not in  self.TXT2TENSOR:
                            print(prompt_clean)
                            val = val.replace(prompt,remove_prefix(prompt_clean),1)
                            continue
                        if prompt == base:
                            if drop_base:
                                val = val.replace(prompt,remove_prefix(prompt_clean),1)
                            else:
                                val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                                replacement.append(prompt_clean)
                                replacement_mask.append(REPLACEMENT_TYPE.INPUT)
                                raw_val.append(prompt)
                                # if set_instance:
                                #     adds.append((all_tgts[prompt][0],prompt_clean))
                        elif random.random() < 0.2:
                            val = val.replace(prompt,remove_prefix(prompt_clean),1)
                        else:
                            val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                            replacement.append(prompt_clean)
                            replacement_mask.append(REPLACEMENT_TYPE.INPUT)
                            raw_val.append(prompt)
                            if set_instance:
                                adds.append((all_tgts[prompt][0],prompt_clean))
                    raw_val.append(val)
                elif src == 'gpt':
                    matches = find_brackets(val)
                    for prompt in matches: # list of str wit '[]'
                        prompt_clean = prompt[1:-1]
                        seen = 0
                        if prompt == base and (drop_base or prompt_clean not in self.TXT2TENSOR):
                            val = val.replace(prompt,'',1)
                            val = val.replace('<base>','<base_null>')
                        elif prompt == base:
                            val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                            replacement.append(prompt_clean)
                            replacement_mask.append(REPLACEMENT_TYPE.BASE)
                        else:
                            assert seen == 0, "Only one outout per instructions!!!"
                            seen =1
                            if self.data_args.output_text:
                                val = val.replace(prompt,prompt+DEFAULT_VIDEO_TOKEN,1)
                            else:
                                val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                            replacement.append(prompt_clean)
                            replacement_mask.append(REPLACEMENT_TYPE.GEN)
                    # if (not adds) and all_tgts:
                    #     raise ValueError(f'{adds},{val},{all_tgts},{raw_val}')
                    if adds:
                        val += 'additions:'
                        for addition_src,addition_caption in adds:
                            val += f'{addition_src}:{DEFAULT_VIDEO_TOKEN}.'
                            replacement.append(addition_caption)
                            replacement_mask.append(REPLACEMENT_TYPE.GEN)
                        # raise ValueError(val)
                        # print("MODIFIED:")
                else:
                    raise NotImplemented
                turn['value']= val
                assert len(replacement_mask) == len(replacement)
            if len(replacement):
                extra_replacement = torch.cat([self.get_tensors_from_str(clean(x)) for x in replacement])
            extra_replacement_mask = replacement_mask

        if sources[0].get('task') == 'segmentation':
            delayed_process = True
            info['generation_seq_len'] = 1
            replacement = [] # List of special strings (e.g. "IMAGE256:students.jpg", "MASK-DECODE:...", "MASK-ENCODE:...", "BOX-ENCODE:...") that were replaced with special tokens (e.g. '<video>' * 256)
            replacement_mask = [] # loss mask
            base = sources[0]['base']
            info['generation'] = True
            assert len(sources_p[0]) %2 == 0
            half_len = len(sources_p[0]) // 2
            start = np.random.choice(range(half_len))
            start = 2 * (start-1)
            # sources_p[0] = sources_p[0][start:start+2]
            # Each pair of human/assistant turns can contain several encode and
            # decode placeholders. That structure enables sequential mask
            # decoding within one round: encodes (image/mask/box) appear first
            # so that subsequent ``MASK-DECODE`` tokens can pull features from
            # ``extra_replacement`` below.
            for turn in sources_p[0]:
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
                        seen = 0
                else:
                    raise NotImplemented
                turn['value']= val
                #assert len(replacement_mask) == len(replacement)

            # For debug tokenizer
            if self.debug_mode:
                replacement = []

            if len(replacement): # Actually load the image assets corresponding to the special [...] strings
                extra_replacement = list([self.build_query(x) for x in replacement]) # list[tuple[str, Any]]:  e.g. ('image-encode, data)

            extra_replacement_mask = replacement_mask

        info['output_text'] = self.data_args.output_text
        data_dict = preprocess(
            sources_p,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),info=info)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        if do_generation:
            data_dict['generation_target'] = generation_target
        else:
            data_dict['generation_target'] = None
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = inputs # hack, actually multimodal
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = dict( # Placeholder image to be replaced
                image=dict(
                    pixel_values=torch.zeros(3, crop_size['height'], crop_size['width'])
                )
            )

        data_dict['conv_id'] = sources[0].get('conv_id', None)
        data_dict['mask_encode_ref'] = mask_encode_ref
        data_dict['extra_inputs']=extra_inputs
        data_dict['extra_replacement']=extra_replacement
        data_dict['extra_replacement_mask']=extra_replacement_mask
        data_dict['delayed_process']=delayed_process
        data_dict['dataset_index'] = _dataset_idx
        assert len(extra_replacement) == len(extra_replacement_mask) or delayed_process
        return data_dict

from torch.utils.data import default_collate
def gather_by_key(data):
    gathered_inputs = {}
    info = []
    for idx_b,b in enumerate(data):
        for row in b:
            modality,data = row['type'],row['data']
            if modality not in gathered_inputs:
                gathered_inputs[modality] = []
            gathered_inputs[modality].append(data)
            info.append(dict(bn=idx_b,modality=modality,idx=len(gathered_inputs[modality])-1))
    for modality in gathered_inputs.keys():
        gathered_inputs[modality] = default_collate(gathered_inputs[modality])
    gathered_inputs['info'] = info
    return gathered_inputs

def clean(x):
    x = x.lower().strip()
    x = x.replace('.','')
    return x

class CollatedBatch(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    generation_target: dict
    extra_replacement: dict
    images: Optional[dict]
    extra_inputs: dict
    dataset_index: list[int]

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> CollatedBatch:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        generation_target=list([x['generation_target'] for x in instances])
        gathered_generation_target= {}
        info = []
        for idx,tgt in enumerate(generation_target):
            if tgt is None:
                continue
            modality = tgt['type']
            data = tgt['data']
            if modality not in gathered_generation_target:
                    gathered_generation_target[modality] = []
            gathered_generation_target[modality].append(data)
            info.append(dict(batch=idx,modality=modality,idx=len(gathered_generation_target[modality])-1))
        for modality in gathered_generation_target.keys():
                gathered_generation_target[modality] = default_collate(gathered_generation_target[modality])
        gathered_generation_target['info'] = info
        batch['generation_target']=gathered_generation_target

        # instances = multiple data samples from __getitem__() of LazySupervisedDataset, chosen by data sampler
        # here, data collator groups each field from sampled instances into batches
        batched_mask_encode_ref = [x['mask_encode_ref'] for x in instances]

        extra_replacement=list([x['extra_replacement'] for x in instances if len(x['extra_replacement'])])


        extra_replacement_idx = list([torch.tensor([idx,] * len(instances[idx]['extra_replacement_mask']),dtype=torch.long)
                                     for idx in range(len(instances))
                                     ])
        if all(x['delayed_process'] is False for x in instances):
            extra_replacement = torch.cat(extra_replacement) # N D
            assert len(extra_replacement) == len(extra_replacement_idx)
        extra_replacement_idx = torch.cat(extra_replacement_idx) # N

        batch['extra_replacement'] = dict(
            mask_encode_ref = batched_mask_encode_ref,          # data collator, return additional field under 'extra_replacement'
            data=extra_replacement,
            idx=extra_replacement_idx,
            mask=torch.cat([torch.tensor(x['extra_replacement_mask'],dtype=torch.long) for x in instances]),
            conv_ids = [x.get('conv_id', None) for x in instances]        # add additional field (index of entry in conv json file)
        )


        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            gathered_inputs = {}
            info = []
            for image in images:
                assert len(image)==1
                modality,data = list(image.items())[0]
                if modality not in gathered_inputs:
                    gathered_inputs[modality] = []
                gathered_inputs[modality].append(data)
                info.append(dict(modality=modality,idx=len(gathered_inputs[modality])-1))
            for modality in gathered_inputs.keys():
                gathered_inputs[modality] = default_collate(gathered_inputs[modality])
            gathered_inputs['info'] = info
            batch['images']=gathered_inputs
        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        batch['extra_inputs']=gather_by_key([instance['extra_inputs'] for instance in instances])

        batch['dataset_index'] = [x['dataset_index'] for x in instances]
        assert len(np.unique( batch['dataset_index']))==1,f"ALl samples in the same batch need to come from same data, but got { batch['dataset_index']}"
        return batch

class SupervisedDataModule(TypedDict):
    train_dataset: LazySupervisedDataset
    eval_dataset: Optional[LazySupervisedDataset]
    data_collator: DataCollatorForSupervisedDataset

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                is_eval: bool = False,
                                is_inference: bool = False,
                                inference_conv: Optional[Dict] = None) -> SupervisedDataModule:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Inference (no load train data)
    if is_inference:
        train_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            is_inference=True,
            inference_conv=inference_conv,
        )
        modules = dict(
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        return modules

    # Eval (no laod train data)
    if is_eval:
        eval_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            is_eval=True
        )
        modules = dict(
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        return modules

    # Train
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        is_eval=True,
    )
    modules = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    return modules

def train():
    global local_rank
    pipe = None
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # if model_args.mm_use_pos_encode:
    #     Constant.MASK_ENCODE_LEN = 2
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    bnb_model_from_pretrained_args = {}
    if training_args.split_loading:
        ranks_LST = [ [0,1,2,3],
            [4,5,6,7],
            [8,9,10,11],]
        # ranks_LST = [ [0,1,],[2,3],
        #     [4,5,],[6,7],
        #     [8,9],[10,11],]
        bnb_model_from_pretrained_args = dict(low_cpu_mem_usage=True,)
    else:
        ranks_LST = [list(range(20)),]
    for ranks in ranks_LST:
        dist.barrier()
        if local_rank in ranks:
            if training_args.bits in [4, 8]:
                from transformers import BitsAndBytesConfig
                bnb_model_from_pretrained_args.update(dict(
                    device_map={"": training_args.device},
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=training_args.bits == 4,
                        load_in_8bit=training_args.bits == 8,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=training_args.double_quant,
                        bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                    )
                ))

            if model_args.vision_tower is not None:
                if 'mpt' in model_args.model_name_or_path:
                    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                    config.attn_config['attn_impl'] = training_args.mpt_attn_impl
                    model = LlavaMPTForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        cache_dir=training_args.cache_dir,
                        **bnb_model_from_pretrained_args
                    )
                else:
                    if model_args.dev == 'test': # test full size no load
                        cfg = LlavaConfig.from_pretrained(model_args.model_name_or_path)
                        model = LlavaLlamaForCausalLM._from_config(cfg)
                    elif model_args.dev == 'test2': # test 2 layer
                        cfg = LlavaConfig.from_pretrained(model_args.model_name_or_path)
                        cfg.num_hidden_layers = 2
                        model = LlavaLlamaForCausalLM._from_config(cfg)
                    elif model_args.dev == 'test3': # test 2 layer
                        cfg = LlavaConfig.from_pretrained(model_args.model_name_or_path)
                        cfg.num_hidden_layers = 2
                        cfg.hidden_size = 128
                        model = LlavaLlamaForCausalLM._from_config(cfg)
                    else:
                        model_name = model_args.load or model_args.model_name_or_path
                        if model_args.from_huggingface and len(model_name.split('/')) == 3:                             # Huggingface expects user_name/repo_name
                            model = LlavaLlamaForCausalLM.from_pretrained(
                                '/'.join(model_name.split('/')[:2]),
                                subfolder=model_name.split('/')[-1],
                                cache_dir=training_args.cache_dir,
                                **bnb_model_from_pretrained_args
                            )
                        else:
                            model = LlavaLlamaForCausalLM.from_pretrained(
                                model_name,
                                cache_dir=training_args.cache_dir,
                                **bnb_model_from_pretrained_args
                            )
            else:
                model = transformers.LlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            model.config.use_cache = False

            if model_args.freeze_backbone:
                model.model.requires_grad_(False)

            if training_args.bits in [4, 8]:
                from peft import prepare_model_for_kbit_training
                model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

            if training_args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if 'mpt' in model_args.model_name_or_path:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    model_max_length=training_args.model_max_length,
                    padding_side="right"
                )
            else:
                model_name = model_args.load or model_args.model_name_or_path
                if model_args.from_huggingface and len(model_name.split('/')) == 3:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        '/'.join(model_name.split('/')[:2]),
                        subfolder=model_name.split('/')[-1],
                        cache_dir=training_args.cache_dir,
                        model_max_length=training_args.model_max_length,
                        padding_side="right",
                        use_fast=True,
                    )
                else:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=training_args.cache_dir,
                        model_max_length=training_args.model_max_length,
                        padding_side="right",
                        use_fast=True,
                    )

            if model_args.version == "v0":
                if tokenizer.pad_token is None:
                    smart_tokenizer_and_embedding_resize(
                        special_tokens_dict=dict(pad_token="[PAD]"),
                        tokenizer=tokenizer,
                        model=model,
                    )
            elif model_args.version == "v0.5":
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.unk_token
                if model_args.version in conversation_lib.conv_templates:
                    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
                else:
                    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

            if model_args.vision_tower is not None:
                model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
            if training_args.lora_enable:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=find_all_linear_names(model),
                    modules_to_save=['lm_head'],
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                )
                if training_args.bits == 16:
                    if training_args.bf16:
                        model.to(torch.bfloat16)
                    if training_args.fp16:
                        model.to(torch.float16)
                rank0_print("Adding LoRA adapters...")
                model = get_peft_model(model, lora_config)

            if model_args.vision_tower is not None:
                if not model_args.load:
                    model.get_model().initialize_vision_modules(
                        model_args=model_args,
                        fsdp=training_args.fsdp
                    )

                vision_tower = model.get_vision_tower()
                vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

                data_args.image_processor = vision_tower.image_processor
                if model_args.segmentator:
                    data_args.mask_processor = model.get_segmentator().process_images
                data_args.is_multimodal = True
                if data_args.segmentation_config:
                    data_args.register = COCORegister(data_args)
                model.config.image_aspect_ratio = data_args.image_aspect_ratio
                model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

                model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
                if model_args.tune_mm_mlp_adapter:
                    model.requires_grad_(False)
                # HACK
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
                for p in model.lm_head.parameters():
                    p.requires_grad = True
                for p in model.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in model.get_output_embeddings().parameters():
                    p.requires_grad = False

                model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
                if training_args.freeze_mm_mlp_adapter:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = False

                if training_args.bits in [4, 8]:
                    model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

                model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
                training_args.use_im_start_end = model_args.mm_use_im_start_end
                model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
                #model.model_args, tokenizer=tokenizer)

            if training_args.bits in [4, 8]:
                from peft.tuners.lora import LoraLayer
                for name, module in model.named_modules():
                    if isinstance(module, LoraLayer):
                        if training_args.bf16:
                            module = module.to(torch.bfloat16)
                    if 'norm' in name:
                        module = module.to(torch.float32)
                    if 'lm_head' in name or 'embed_tokens' in name:
                        if hasattr(module, 'weight'):
                            if training_args.bf16 and module.weight.dtype == torch.float32:
                                module = module.to(torch.bfloat16)
            data_module = make_supervised_data_module(
                tokenizer=tokenizer,
                data_args=data_args,
                is_eval=training_args.eval_only
            )
            # if model_args.load:
            #     model.load_state_dict(torch.load(model_args.load))
            def compute_metrics(eval_preds):
                breakpoint()
            trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    model_args=model_args,
                    data_args=data_args,
                    pipe=pipe,
                    compute_metrics=compute_metrics,
                    **data_module)
        else:
            pass

    if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(os.path.join(training_args.output_dir,'tokenizer'))
    if training_args.eval_only:
        trainer.evaluate(ar_decoding=training_args.ar_eval)
        return
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        model.config.save_pretrained(training_args.output_dir)
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,force_save=True)


if __name__ == "__main__":
    train()

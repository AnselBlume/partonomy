import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from transformers import BitsAndBytesConfig, CLIPVisionModel
# from torchcrf import CRF

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX,
                         DEFAULT_IMAGE_PATCH_TOKEN, BIO_START_LBL, BIO_INTERM_LBL, BIO_NO_LBL, _with_patch_offset)

from .mask_feedback import TemporalMaskPooler, FilmModulator

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

try:
    from deepspeed.utils.zero_to_fp32 import GatheredParameters
except ImportError:
    # allows the code to run when DeepSpeed isn't installed
    from contextlib import contextmanager
    @contextmanager
    def GatheredParameters(params, modifier_rank=None):
        yield


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1,  # 100000.0, - original setting was 1000
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def focal_tversky_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,  # 3/4 (or 4/3 for 1/gamma as in the paper)
    eps: float = 1e-6,
):
    '''
    Compute the Focal Tversky Loss (FTL) for segmentation.
    This extends the Tversky index (a generalization of Dice) by adding a focal exponent 'gamma'.

    Args:
        inputs: A float tensor of shape [B, H, W] or [B, 1, H, W].
                These are raw logits from the model (will be passed through sigmoid).
        targets: A float tensor of the same shape as inputs (excluding channel if present). 
                 0/1 binary ground truth mask.
        num_masks: Normalizing factor used just like in dice_loss, typically 1 per mask.
        alpha: Weighting factor for FN in the Tversky index.
        beta: Weighting factor for FP in the Tversky index. (the sum of alpha and beta should be 1)
        gamma: Focal exponent. Raises (1 - TverskyIndex) to this power to focus more on hard examples.
        eps: Small constant to avoid division by zero.

    Returns:
        A scalar tensor representing the Focal Tversky Loss, averaged over 'num_masks'.
    '''
    probs = torch.sigmoid(inputs)  # shape: [B, H, W] or [B, 1, H, W]
    
    if probs.dim() == 4 and probs.size(1) == 1:  # if shape is [B, 1, H, W], remove channel dim
        probs = probs.squeeze(1)
    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    
    # flatten: [B, H, W] -> [B, H*W]
    probs_flat = probs.flatten(1)
    targets_flat = targets.flatten(1)
    
    #    TP = sum(p*g), FP = sum(p*(1-g)), FN = sum((1-p)*g)
    TP = (probs_flat * targets_flat).sum(dim=1)
    FP = (probs_flat * (1 - targets_flat)).sum(dim=1)
    FN = ((1 - probs_flat) * targets_flat).sum(dim=1)
    
    # 4) Compute the Tversky index for each sample in the batch
    tversky_index = (TP + eps) / (TP + alpha*FN + beta*FP + eps)  # shape: [B]
    
    # 5) Compute Focal Tversky Loss: (1 - TverskyIndex)^gamma
    focal_tversky = (1.0 - tversky_index) ** gamma  # shape: [B]
    
    # 6) Average over the batch, then divide by num_masks (like your dice_loss)
    loss = focal_tversky.sum() / (num_masks + 1e-8)
    
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def kl_divergence_loss(t_ref, t_proj, sigma=1.0):
    '''
    Args:
        t_ref: nn.Embedding parameters at t=0 (frozen parameters) used as reference distribution
        t_proj: nn.Embedding parameters that are being updated via gradients from SAM decoder
        
        This KL divergence loss is for two Gaussians with the same covariance.
    '''
    return 0.5 / (sigma ** 2) * F.mse_loss(t_proj, t_ref, reduction='mean')


def hinge_loss(log_likelihoods, margin=1.0):
    """
    log_likelihoods: Tensor of shape [batch_size], where
        log_likelihoods[0] is the correct answer,
        log_likelihoods[1:] are incorrect answers.
    """
    correct_ll = log_likelihoods[0]
    incorrect_lls = log_likelihoods[1:]
    # Hinge loss: max(0, margin - (correct - incorrect))
    losses = F.relu(margin - (correct_ll - incorrect_lls))
    return losses.mean()


def compute_bio_per_class_counts(
        pred_per_token_labels: torch.Tensor, 
        gt_per_token_labels: torch.Tensor,
        num_classes: int = 3,
    ):
    '''
    Returns the number of correct predictions and the total number of tokens
    for each of the 3 BIO classes.
    
    Args:
        pred_per_token_labels: [N]  (0: O, 1: B, 2: I)
        gt_per_token_labels:   [N]
        num_classes: [2, 3] - 2 if self.pred_binary_span == True
    
    Returns:
        correct_0, total_0,
        correct_1, total_1,
        correct_2, total_2
    '''
    correct_mask = (pred_per_token_labels == gt_per_token_labels)
    
    # class 0: 'O'
    mask_0 = (gt_per_token_labels == 0)
    correct_0 = (correct_mask & mask_0).sum().item()
    total_0 = mask_0.sum().item()
    
    # class 1: 'B'
    mask_1 = (gt_per_token_labels == 1)
    correct_1 = (correct_mask & mask_1).sum().item()
    total_1 = mask_1.sum().item()

    if num_classes == 3:
        # class 2: 'I'
        mask_2 = (gt_per_token_labels == 2)
        correct_2 = (correct_mask & mask_2).sum().item()
        total_2 = mask_2.sum().item()
    else:
        mask_2 = -1
        correct_2 = -1
        total_2 = -1
    
    return {
        'correct_0': correct_0, 'total_0': total_0,
        'correct_1': correct_1, 'total_1': total_1,
        'correct_2': correct_2, 'total_2': total_2
    }


def check_nan_hook(module, input, output):
    if torch.isnan(output).any():
        print(f">> NaN in {module}")
    

class BidirectionalEncoderBlock(nn.Module):
    '''
    Bidirectional Encoder Block for BIO classification
    '''
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, init_method="xavier"):
        super(BidirectionalEncoderBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model,
                            nhead,
                            layer_norm_eps=1e-5,
                            # dim_feedforward,
                            # dropout,
                            # activation='relu'
                        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.init_method = init_method

    def _init_weights(self):
        for name, param in self.encoder.named_parameters():
            if param.dim() > 1:
                if self.init_method == "he":
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif self.init_method == "xavier":
                    nn.init.xavier_uniform_(param)
                elif self.init_method == "normal":
                    nn.init.normal_(param, mean=0.0, std=0.01)
            else:
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'norm' in name:
                    nn.init.constant_(param, 1.0)
                else:
                    nn.init.constant_(param, 0.0)

    def forward(self, inp: torch.Tensor, attention_mask: torch.Tensor=None):
        '''
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len]
        '''
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        inp = inp.transpose(0, 1) #  transformer encoder expects seq_len first dim as default
        out = self.encoder(inp, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1)
        return out


class BioVisualFusionBlock(nn.Module):
    '''
    Using BIO span embeddings as queries to sample from input image embeddings
    '''
    def __init__(self, d_model, nhead, dropout=0.1):
        super(BioVisualFusionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        # self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def forward(self, query, key_value):
        # query: [seq_len, bsz, d_model] (from BIO spans)
        # key_value: [num_img_tokens, bsz, d_model] (from image features)
        attn_output, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm_attn(query + attn_output)
        ffn_output = self.ffn(query)
        query = self.norm_ffn(query + ffn_output)
        return query
    


class PlumMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PlumMetaModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        if not hasattr(self.config, "train_mask_decoder"):  # args for fine-tuning SAM decoder
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
        if not hasattr(self.config, "train_mask_prompt_encoder"):  # args for fine-tuning SAM prompt encoder
            self.config.train_mask_prompt_encoder = kwargs["train_mask_prompt_encoder"]
        
        self.initialize_plum_modules(self.config, **kwargs)

    def initialize_plum_modules(self, config, **kwargs):
        '''
        Initialize the text_hidden_fcs and tok2mask_fc layers for token embeddings to SAM query transformation
        '''
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)

        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        if config.train_mask_prompt_encoder:
            self.visual_model.prompt_encoder.train()
            for param in self.visual_model.prompt_encoder.parameters():
                param.requires_grad = True

        # Projection layer - converts text hidden_states into queries (input to SAM)
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
            
        # initialize the per-token classifier to determine which hidden_state should be fed into SAM
        self.pred_binary_span = kwargs.pop("pred_binary_span", False)
        cls_out_dim = 1 if self.pred_binary_span else 3  # using BIO tagging scheme to handle the sub-token aggregation
        tok2mask_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, cls_out_dim)
        ]
        self.token_to_mask_fcs = nn.ModuleList([nn.Sequential(*tok2mask_fc)])
        self.token_to_mask_fcs.train()
        for param in self.token_to_mask_fcs.parameters():
            param.requires_grad = True


class PlumModel(PlumMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PlumModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class PLUMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_type = kwargs.pop("dice_type", "dice")
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.dice_scale_factor = kwargs.pop("dice_scale_factor", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.kld_loss_weight = kwargs.pop("kld_loss_weight", None)
            self.kld_sigma = kwargs.pop("kld_sigma", None)
            self.seg_cls_loss_weight = kwargs.pop("seg_cls_loss_weight", None)
            self.seg_cls_loss_per_cls_weight = kwargs.pop("seg_cls_loss_per_cls_weight", None)
            self.focal_tversky_alpha = kwargs.pop("focal_tversky_alpha", 0.7)
            self.focal_tversky_beta = kwargs.pop("focal_tversky_beta", 0.3)
            self.use_hinge_loss = kwargs.pop("use_hinge_loss", False)
            self.use_feedback_loop = kwargs.pop("use_feedback_loop", False)
        else:
            config.mm_vision_tower = config.vision_tower
            self.dice_type = 'dice'
            self.dice_scale_factor = 0.0
            self.ce_loss_weight = 0.0
            self.bce_loss_weight = 0.0
            self.kld_loss_weight = 0.0
            self.kld_sigma = 0.0
            self.seg_cls_loss_weight = 0.0
            self.focal_tversky_alpha = 0.0
            self.focal_tversky_beta = 0.0
            self.seg_cls_loss_per_cls_weight = None
            self.use_hinge_loss = False
            self.use_feedback_loop = False
        
        super().__init__(config)

        self.model = PlumModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_pooler = None
        
        # initialize weights and apply final processing
        self.post_init()
        
        self.use_teacher_ref = kwargs.pop("use_teacher_ref", False)
        self.teacher_llm = None  # postpone teacher_llm initialization until after pretrained weights are loaded.
        self.register_buffer('seg_h_ref', None)  # t=0 copy placeholder for seg_h_ref
        self.ema_update_seg_h_ref = False
        
        # use bidirectional encoder for BIO tagging or not
        self.use_bidir_bio = kwargs.pop("use_bidir_bio", False)
        self.use_crf_bio = kwargs.pop("use_crf_bio", False)
        self.pred_binary_span = kwargs.pop("pred_binary_span", False)
        self.num_token_classes = 2 if self.pred_binary_span else 3
        if self.use_bidir_bio:
            self.bio_encoder = BidirectionalEncoderBlock(
                d_model=config.hidden_size,
                nhead=kwargs.pop("bidir_nhead", 8),
                dim_feedforward=kwargs.pop("bidir_dim_feedforward", 2048),
                dropout=0.1
            )
            if self.use_crf_bio:
                pass
                # self.crf = CRF(num_tags=3, batch_first=True)  # 3 -> 0: O, 1: B, 2: I
        else:
            self.bio_encoder = None
            
        if self.use_feedback_loop:
            self.initialize_mask_pooler()

        # Cross attention block for BIO spans to attend to the image patch embeddings
        self.use_cross_attn_bio = kwargs.pop("use_cross_attn_bio", False)
        if self.use_cross_attn_bio:
            self.bio_cross_attn = BioVisualFusionBlock(
                d_model=config.hidden_size,
                nhead=8,
                dropout=0.1
            )
        else:
            self.bio_cross_attn = None


    def _touch_trainable_modules(self) -> torch.Tensor:
        '''
        Ensure every trainable parameter in the relevant submodules participates
        in the graph each step. This prevents per-rank bucket divergence leading to
        WatdogError during multi-gpu training.
        '''
        device = next(self.parameters()).device
        z = torch.zeros((), device=device, dtype=torch.float32)

        def touch_all_params(mod):
            nonlocal z
            if mod is None:
                return
            # accumulate a zero-weight scalar that depends on every trainable param
            for p in mod.parameters():
                if p.requires_grad:
                    z = z + p.float().sum() * 0.0  # 0 grad

        touch_all_params(self.lm_head)
        touch_all_params(self.model.text_hidden_fcs)
        touch_all_params(self.model.token_to_mask_fcs)
        touch_all_params(getattr(self.model.visual_model, "mask_decoder", None))

        if getattr(self.model.config, "train_mask_prompt_encoder", False):
            touch_all_params(self.model.visual_model.prompt_encoder)
        if getattr(self, "bio_encoder", None) is not None:
            touch_all_params(self.bio_encoder)
        if getattr(self, "mask_pooler", None) is not None:
            touch_all_params(self.mask_pooler)
        if getattr(self, "bio_cross_attn", None) is not None:
            touch_all_params(self.bio_cross_attn)

        return z


    def initialize_teacher_llm(self):
        print(">> (PLUM.py) Initializing teacher LLM...")
        # Deep copy the underlying LlavaLlamaModel from the fully loaded model
        self.teacher_llm = copy.deepcopy(self.get_model()).float()
        for param in self.teacher_llm.parameters():
            param.requires_grad = False
        self.teacher_llm.eval()
        print(">> (PLUM.py) Teacher LLM initialized.")

        
    def initialize_mask_pooler(self):
        prompt_encoder = self.model.visual_model.prompt_encoder
        decoder_embed_dim = self.model.visual_model.prompt_encoder.embed_dim
        conv_layers = [m for m in prompt_encoder.mask_downscaling if isinstance(m, nn.Conv2d)]
        mask_modulator = FilmModulator(
            span_embed_dim=decoder_embed_dim,
            mask_feat_dims=[conv_layer.out_channels for conv_layer in conv_layers]
        )
        self.mask_pooler = TemporalMaskPooler(
            prompt_encoder=prompt_encoder,
            mask_modulator=mask_modulator
        )
        

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],  # list of gt segmentation masks (len == batch size)
        label_list: List[torch.Tensor],  # auxiliary info for postprocessing (e.g., original sizes)
        resize_list: List[tuple],  # list of target size for masks
        per_token_labels: torch.LongTensor,  # [# of conv, seq_len] with BIO labels: 0 (O), 1 (B), 2 (I)
        mask_positions_in_input_ids: list,  # list of lists of starting token indices per sample
        inference: bool = False,
        **kwargs,
    ):
        im_start_id = 32000  # tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        im_end_id   = 32001  # tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        has_image_tokens = (input_ids == im_start_id).any()
        if has_image_tokens:
            image_embeddings = self.get_visual_embs(images)
        else:
            image_embeddings = None
        batch_size = image_embeddings.shape[0] if image_embeddings is not None else len(offset) - 1
        assert batch_size == len(offset) - 1
        
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            
            output_hidden_states = []
            ce_losses = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                outputs = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    labels=input_ids[start_i:end_i],
                    output_hidden_states=True,
                    output_per_instance_loss=False  # NOTE: 'output_per_instance_loss' needs to be set to True for multiple-choice case during eval
                )
                output_hidden_states.append(outputs.hidden_states)
                ce_losses.append(outputs.loss)

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            
            output_per_instance_loss = False
            if self.use_hinge_loss:
                output_per_instance_loss = True

            outputs = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                output_per_instance_loss=output_per_instance_loss
            )
            output_hidden_states = outputs.hidden_states
            logits = outputs.logits
            ce_loss = outputs.loss  # the lm loss

        IMAGE_TOKENS_OFFSET = 255 if has_image_tokens else 0
        im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True) if has_image_tokens else (torch.tensor([], device=input_ids.device, dtype=torch.long), torch.tensor([], device=input_ids.device, dtype=torch.long))
        im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True) if has_image_tokens else (torch.tensor([], device=input_ids.device, dtype=torch.long), torch.tensor([], device=input_ids.device, dtype=torch.long))
        im_start_pos_list = list(zip(im_start_positions[0].tolist(), im_start_positions[1].tolist())) if has_image_tokens else []
        im_end_pos_list = list(zip(im_end_positions[0].tolist(), im_end_positions[1].tolist())) if has_image_tokens else []
        
        last_hidden_states = output_hidden_states[-1]
        # self.use_bidir_bio = False
        if self.use_bidir_bio:
            # pad attention_masks with 255 1's to match the input_ids
            img_tok_offset_attention_masks = torch.cat([torch.ones((attention_masks.shape[0], IMAGE_TOKENS_OFFSET), device=attention_masks.device), attention_masks], dim=1)
            bidir_last_hidden_states = self.bio_encoder(last_hidden_states, attention_mask=img_tok_offset_attention_masks)
        else:
            bidir_last_hidden_states = last_hidden_states

        # NOTE: Hacky fix for IMAGE_TOKEN_INDEX (supposing there is only one image; 256 patch tokens are added in between <im_start> and <im_end>)
        per_token_labels = torch.cat(
            [
                torch.zeros((per_token_labels.shape[0], IMAGE_TOKENS_OFFSET), dtype=per_token_labels.dtype, device=last_hidden_states.device), 
                per_token_labels
            ], dim=1
        )
        for b, gt_positions in enumerate(mask_positions_in_input_ids):
            mask_positions_in_input_ids[b] = [pos + IMAGE_TOKENS_OFFSET for pos in gt_positions]

        sam_proj_hidden_states = self.model.text_hidden_fcs[0](last_hidden_states) # [256, num_tokens]
        # sam_proj_hidden_states = self.model.text_hidden_fcs[0](bidir_last_hidden_states)  # NOTE: so far, 'last_hidden_states' gave best performance
        seg_logits = self.model.token_to_mask_fcs[0](bidir_last_hidden_states)
        
        # NOTE: Avoiding seg_cls_loss for VQADataset instances to alleviate 'O'-class bias
        has_io_lbls = per_token_labels.sum(dim=-1).bool()
        if has_io_lbls.sum() > 0:
            if self.seg_cls_loss_per_cls_weight is not None and sum(self.seg_cls_loss_per_cls_weight) > 0:
                cls_weights = torch.tensor(self.seg_cls_loss_per_cls_weight, device=seg_logits.device).to(seg_logits.dtype)
            else:
                cls_weights = None
            
            if self.use_crf_bio:  # seg_logits - emissions ; per_token_labels - provide targets as well as the transition probability
                seg_cls_loss = -self.crf(seg_logits, per_token_labels, mask=img_tok_offset_attention_masks.bool(), reduction='mean')
            elif self.pred_binary_span:
                per_token_labels[per_token_labels == 2] = 1  # convert 2's in per_token_labels to 1's since it's binary case
                logits_bin = seg_logits[has_io_lbls].view(-1)
                labels_bin = per_token_labels[has_io_lbls].float().view(-1)
                pos_weight_val = cls_weights[1] / (cls_weights[0] + 1e-8)  # weight for the positive class (segment-able span)
                seg_cls_loss = F.binary_cross_entropy_with_logits(
                    logits_bin,
                    labels_bin,
                    weight=torch.tensor(pos_weight_val, device=logits_bin.device),
                    reduction='mean'
                )
            else:
                seg_cls_loss = F.cross_entropy(
                    seg_logits[has_io_lbls].view(-1, 3), 
                    per_token_labels[has_io_lbls].view(-1), 
                    weight=cls_weights
                )  # calculate the bce between the seg_scores and gt_parts
        else:
            seg_cls_loss = torch.tensor(0.0, device=seg_logits.device)
        
        if self.use_bidir_bio and self.use_crf_bio:
            # convert img_tok_offset_attention_masks to torch bool type
            pred_seg_label_list = self.crf.decode(seg_logits, mask=img_tok_offset_attention_masks.bool())
            pred_seg_labels = []
            for decoded_lbl in pred_seg_label_list:
                if len(decoded_lbl) < seg_logits.shape[1]:
                    decoded_lbl.extend([0] * (seg_logits.shape[1] - len(decoded_lbl)))
                pred_seg_labels.append(decoded_lbl)
            pred_seg_labels = torch.tensor(pred_seg_labels, device=seg_logits.device, dtype=seg_logits.dtype)
        else:
            pred_seg_labels = torch.argmax(seg_logits, dim=-1)

        bio_per_cls_counts_dict = compute_bio_per_class_counts(pred_seg_labels, per_token_labels, self.num_token_classes)
        
        # print("(PLUM.py) >> pred_seg_labels:\n", pred_seg_labels)
        # print("\n(PLUM.py) >> per_token_labels:\n", per_token_labels)
        
        if self.use_teacher_ref or self.use_cross_attn_bio:
            (
                t_input_ids,
                t_attention_mask,
                t_past_key_values,
                t_inputs_embeds,
                t_labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    attention_masks,
                    past_key_values=None,  # or the appropriate value
                    labels=labels,
                    images=images_clip
                )

        # output frozen teacher representations
        if self.use_teacher_ref:
            return_dict = None
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            with torch.no_grad():
                self.teacher_llm.train(False)
                with torch.cuda.amp.autocast(enabled=False):  # NOTE: enforce float32 computation
                    teacher_outputs = self.teacher_llm(
                        attention_mask=t_attention_mask,  # input_ids=t_input_ids,
                        past_key_values=t_past_key_values,
                        inputs_embeds=t_inputs_embeds,
                        output_hidden_states=True,
                        return_dict=return_dict
                    )                        
                teacher_last_hidden_states = teacher_outputs.hidden_states[-1]
                assert teacher_last_hidden_states.shape == last_hidden_states.shape
        else:
            if self.seg_h_ref is None:
                self.seg_h_ref = last_hidden_states.detach().clone()
            elif self.ema_update_seg_h_ref:
                self.seg_h_ref = self.momentum * self.seg_h_ref + (1 - self.momentum) * last_hidden_states.detach()

        batch_kl_loss = 0.0
        batch_seg_loss = 0.0
        batch_bce_loss = 0.0
        group_count = 0  # for each sample, use 'mask_positions_in_input_ids' to group tokens

        pred_masks = []
        gt_masks = []
        BIO_INTERM_LBL = 2 if not self.pred_binary_span else 1
        
        for batch_idx in range(len(offset) - 1):  # batch-level iteration
            # e.g., offset = [0, 7, 8] - 0-6 for conversations in batch 0, 7-8 for conversations in batch 1
            start = offset[batch_idx]
            end = offset[batch_idx + 1]
            is_mask_per_conv = ((end - start) == masks_list[batch_idx].size(0))   # for refer_seg, sem_seg, reason_seg datasets (and explanatory_seg in case there is only one mask per conv)
            
            for conv_idx in range(start, end):  # conv-level interation
                token_labels_b = per_token_labels[conv_idx]
                gt_positions = mask_positions_in_input_ids[conv_idx]
                
                if masks_list[batch_idx].shape[0] == 0:  # VQADataset does not have masks
                    continue
                elif is_mask_per_conv:  # a mask for each conversation (refer_seg, sem_seg, reason_seg)
                    gt_mask = masks_list[batch_idx][conv_idx - start].unsqueeze(0)  # [1, H, W]
                
                span_embeds_l = []
                masks_l = []
                # Group over BIO spans
                for mask_span_idx, pos in enumerate(gt_positions):
                    assert_msg = f">> original pos: {pos - IMAGE_TOKENS_OFFSET}\n==\nper_token_labels (len={len(token_labels_b[IMAGE_TOKENS_OFFSET:])}): {token_labels_b[IMAGE_TOKENS_OFFSET:]}\n==\ninput_ids (len={len(input_ids)}): {input_ids[conv_idx]}"
                    try:
                        assert token_labels_b[pos].item() == BIO_START_LBL, assert_msg
                    except AssertionError as e:
                        print(e)
                        breakpoint()
                    
                    indices = [pos]
                    j = pos + 1
                    while j < token_labels_b.size(0) and token_labels_b[j].item() == BIO_INTERM_LBL:
                        indices.append(j)
                        j += 1
                    
                    current_group_hidden = last_hidden_states[conv_idx, indices, :]
                    if self.use_teacher_ref:
                        cached_group_hidden = teacher_last_hidden_states[conv_idx, indices, :]
                        group_kl_loss = kl_divergence_loss(cached_group_hidden, current_group_hidden, sigma=1.0)
                    else:
                        group_kl_loss = torch.tensor(0.0, device=last_hidden_states.device)

                    # input layer proj. image embeddings are the key and values and query is the bio spans
                    if self.use_cross_attn_bio:
                        im_start_pos = im_start_pos_list[conv_idx][1]
                        im_end_pos = im_end_pos_list[conv_idx][1]                            
                        # key_value = last_hidden_states[conv_idx][im_start_pos + 1 : im_end_pos + IMAGE_TOKENS_OFFSET, :].unsqueeze(1)  # sample last_hidden_states that correspond to image patches
                        key_value = t_inputs_embeds[conv_idx][im_start_pos + 1 : im_end_pos + IMAGE_TOKENS_OFFSET, :].unsqueeze(1)  # image embeddings  # XXX If we want to use pre-LLM image embeds
                        query = current_group_hidden.unsqueeze(1)
                        cross_attn_bio_embeds = self.bio_cross_attn(query, key_value)
                        cross_attn_group_proj = self.model.text_hidden_fcs[0](cross_attn_bio_embeds.squeeze(1))

                    batch_kl_loss += group_kl_loss
                    
                    if self.use_cross_attn_bio:
                        group_proj = cross_attn_group_proj.mean(dim=0)
                    else:
                        group_proj = sam_proj_hidden_states[conv_idx, indices, :].mean(dim=0)  # text embeddings for BIO spans
                    
                    if not is_mask_per_conv:  # for explanatory_seg dataset where there are multiple masks per conversation
                        if mask_span_idx < masks_list[batch_idx].shape[0]:
                            gt_mask = masks_list[batch_idx][mask_span_idx].unsqueeze(0)  # [1, H, W]
                        else:
                            try:
                                raise ValueError(f"Mask index out of bounds: gt_mask (shape): {gt_mask.shape} | masks_list[batch_idx] (shape): {masks_list[batch_idx].shape}")
                            except ValueError as e:
                                print(e)
                                breakpoint()

                    image_emb = image_embeddings[batch_idx].unsqueeze(0)
                    
                    # run the SAM prompt encoder and mask decoder with these inputs - NOTE: Why don't we try SAM2 to see if the IoU improves?
                    sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                        points=None, boxes=None, masks=None, text_embeds=group_proj.unsqueeze(0).unsqueeze(0)
                    )
                    if self.use_feedback_loop:
                        dense_embeddings = self.mask_pooler(span_embeds_l, masks_l)

                    sparse_embeddings = sparse_embeddings.to(group_proj.dtype)
                    low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_emb,
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[batch_idx],
                        original_size=label_list[batch_idx].shape,
                    )

                    # Continue computing dice loss and BCE loss etc.
                    if self.dice_type == "dice":
                        group_seg_loss = dice_loss(pred_mask[:, 0], gt_mask.float(), num_masks=1, scale=self.dice_scale_factor)
                    elif self.dice_type == "focal_tversky":
                        group_seg_loss = focal_tversky_loss(pred_mask[:, 0], gt_mask.float(), num_masks=1, alpha=self.focal_tversky_alpha, beta=self.focal_tversky_beta)
                    group_bce_loss = sigmoid_ce_loss(pred_mask[:, 0], gt_mask.float(), num_masks=1)
                    batch_seg_loss += group_seg_loss
                    batch_bce_loss += group_bce_loss
                    group_count += 1

                    if self.use_feedback_loop:
                        span_embeds_l.append(group_proj.squeeze())
                        masks_l.append(low_res_masks.squeeze())

                    pred_masks.append(pred_mask[:, 0])
                    gt_masks.append(gt_mask)
                    
        # no-op pass through SAM when no spans this step - prevent multi-gpu hanging
        if has_image_tokens and group_count == 0:
            # fabricate a single dummy query + single image emb to run prompt+decoder
            # (all multiplied by 0 later; this only anchors the graph for grad compute)
            sam_dtype = next(self.model.visual_model.mask_decoder.parameters()).dtype
            with torch.no_grad():
                image_emb = self.get_visual_embs(images[:1]).to(sam_dtype)
            dummy_query = self.model.text_hidden_fcs[0](last_hidden_states[:1, :1, :]).mean(dim=1)  # [1, out_dim]
            dummy_query = dummy_query.to(sam_dtype)
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=dummy_query.unsqueeze(0)
            )
            pe = self.model.visual_model.prompt_encoder
            sparse_embeddings, dense_embeddings = pe(
                points=None, boxes=None, masks=None,
                text_embeds=dummy_query.unsqueeze(0)
            )
            sparse_embeddings = sparse_embeddings.to(sam_dtype)
            dense_embeddings  = dense_embeddings.to(sam_dtype)

            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_emb,
                image_pe=pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # consume in a zero-weight way so it participates in autograd graph
            _ = (low_res_masks.sum() * 0.0)
                    
        if group_count > 0:
            avg_kl_loss = batch_kl_loss / group_count
            avg_seg_loss = batch_seg_loss / group_count
            avg_bce_loss = batch_bce_loss / group_count
        else:
            avg_kl_loss = torch.tensor(0.0, device=last_hidden_states.device)
            avg_seg_loss = torch.tensor(0.0, device=last_hidden_states.device)
            avg_bce_loss = torch.tensor(0.0, device=last_hidden_states.device)
            
        if inference:
            final_loss = None
        else:
            final_loss = ce_loss \
                        + self.seg_cls_loss_weight * seg_cls_loss \
                        + self.kld_loss_weight * avg_kl_loss \
                        + self.dice_loss_weight * avg_seg_loss \
                        + self.bce_loss_weight * avg_bce_loss
                        
        # call self._touch_trainable_modules() to prevent hanging in multi-gpu settings
        if self.training:
            final_loss = final_loss + self._touch_trainable_modules()

        if inference:            
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "ce_losses": ce_losses,
                "bio_per_cls_counts_dict": bio_per_cls_counts_dict
            }

        # return dict during training
        return {
            "loss": final_loss,
            "ce_loss": ce_loss,  # LLM loss
            "seg_cls_loss": seg_cls_loss,
            "kl_loss": avg_kl_loss,
            "mask_dice_loss": avg_seg_loss,
            "mask_bce_loss": avg_bce_loss,
            "mask_loss": avg_bce_loss + avg_seg_loss,
            "logits": logits if not inference else None,
            "hidden_states": output_hidden_states,
        }


    def evaluate(
        self,
        images_clip: torch.FloatTensor,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        resize_list: list,
        label_list: list,
        attention_masks: torch.LongTensor = None,
        gt_bio_span: bool = None,
        max_new_tokens: int = 512,
        tokenizer: str = None,
        multiple_choice: bool = False,
        prompt_user_input: bool = False,
        **kwargs,
    ):
        questions = kwargs.pop("questions", None)
        question_types = kwargs.pop("question_types", None)
        conversations = kwargs.pop("conversations", None)
        conversation_types = kwargs.pop("conversation_types", None)
        conversation_question_types = kwargs.pop("conversation_question_types", None)
        part_answer_choices = kwargs.pop("part_answer_choices", None)
        part_answer_types = kwargs.pop("part_answer_types", None)
        answer_parts = kwargs.pop("answer_parts", None)
        object_answer_choices = kwargs.pop("object_answer_choices", None)
        object_answer_types = kwargs.pop("object_answer_types", None)
        answer_objects = kwargs.pop("answer_objects", None)
        per_token_labels = kwargs.pop("per_token_labels", None)
        mask_positions_in_input_ids = kwargs.pop("mask_positions_in_input_ids", None)
        
        '''
        Evaluate the model in inference mode.
        
        This function performs the following steps:
        1. Generates output sequences with hidden states.
        2. Computes segmentation logits for each token.
        3. Derives predicted segmentation labels (using a BIO-like scheme where:
            1 = Begin (B), 2 = Inside (I), 0 = Outside (O)).
        4. Uses the projected hidden states (via text_hidden_fcs) to extract token spans for which the segmentation label indicates a mask query.
            For each span, aggregates the corresponding token embeddings (mean pooling).
        5. For each aggregated query, runs the SAM prompt encoder and mask decoder
            (using the image embeddings) to predict a segmentation mask.
        
        Returns:
        A dictionary with:
            - "output_ids": the generated token sequences.
            - "batch_spans": a list (per sample) of dictionaries, each with keys:
                "spans": a list of (start, end) token index tuples for each mask query,
                "embeddings": the aggregated embedding for that span.
            - "pred_masks": a list (per sample) of predicted masks (one per detected span).
            ...
            - "prompt_user_input": a boolean indicating if the user input is required (in case pred_seg_labels is all 0's).
        '''
        num_generated_tokens = 0
        im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        im_end_id   = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        has_image_tokens = (input_ids == im_start_id).any()
        IMAGE_TOKENS_OFFSET = 255 if has_image_tokens else 0
        
        # NOTE: The way the baseline code did it:
        # "USER: <img_start> <image> <img_end> What is this?"
        #  0       1           2        3
        #  "USER: <img_start> t t t ... t <img_end> What is this?"
        #  0       1          2 3 4 ... 257 258           257
        with torch.no_grad():
            ce_losses = []
            if multiple_choice:  # only for multiple_choice==True case, do we need the seq-level ce_loss for answer selection
                images_clip_extend = images_clip.expand(input_ids.size(0), -1, -1, -1).contiguous()
                labels = input_ids.clone()
                labels[labels == self.config.pad_token_id] = -100  # NOTE: or, tokenizer.pad_token_id would do the same thing
                outputs = super().forward(
                    images=images_clip_extend,
                    attention_mask=attention_masks,
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    output_per_instance_loss=True
                )
                output_hidden_states = outputs.hidden_states  # i.e., 'last_hidden_states'
                input_ids_fixed = input_ids.clone()
                input_ids_fixed[input_ids_fixed == IMAGE_TOKEN_INDEX] = tokenizer.unk_token_id
                output_ids_trunc = input_ids_fixed
                output_hidden_states_trunc = output_hidden_states
                ce_losses.append(outputs.loss)
                # print("input_ids: ", input_ids)
                # print("input_ids (decoded): ", tokenizer.batch_decode(input_ids_fixed, skip_special_tokens=False))
            else:
                outputs = self.generate(
                    images=images_clip,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                output_ids = outputs.sequences
                output_ids_fixed = output_ids.clone()
                output_ids_fixed[output_ids_fixed == IMAGE_TOKEN_INDEX] = tokenizer.unk_token_id
                output_hidden_states = outputs.hidden_states[-1]  # [bsz, seq_len, d_model] - 'seq_len' = 'len(input_ids)' + IMAGE_TOKENS_OFFSET + 'num_generated_tokens' - 1 (-1 because early break occurs when </s> is generated)
                output_ids_trunc = output_ids_fixed[0][input_ids.size(1):].unsqueeze(0)
                num_generated_tokens = output_ids_trunc.size(1)
                
            if self.use_bidir_bio:
                # pad attention_masks with 255 1's to match the input_ids
                attention_mask_offset = IMAGE_TOKENS_OFFSET if multiple_choice else IMAGE_TOKENS_OFFSET + num_generated_tokens - 1
                img_tok_offset_attention_masks = torch.cat(
                    [
                        torch.ones((attention_masks.shape[0], attention_mask_offset), device=attention_masks.device), attention_masks
                    ], dim=1)
                bidir_last_hidden_states = self.bio_encoder(output_hidden_states, attention_mask=img_tok_offset_attention_masks)
            else:
                bidir_last_hidden_states = output_hidden_states
            
            sam_proj_hidden_states = self.model.text_hidden_fcs[0](output_hidden_states)
            seg_logits = self.model.token_to_mask_fcs[0](bidir_last_hidden_states)

            pred_seg_labels = torch.argmax(seg_logits, dim=-1)  # [bsz, seq_len]
            
            if gt_bio_span is None:
                bio_labels = pred_seg_labels
                # NOTE: Receive user input and parse last_hidden_states (used as input to the mask decoder)
                # Use for case demo on "arbitrary" text span groundability on images
                if prompt_user_input:  # bio_labels.sum(dim=-1) == 0 and 
                    model_output_txt = tokenizer.decode(output_ids_trunc[0], skip_special_tokens=False)
                    user_spans = input(f"[ PLUM Output ]\n>>\n{model_output_txt}\n\n>>\n>> Please input the text span you want to see highlighted (e.g., wings, cockpit): ")
                    user_span_txts = [s.strip() for s in user_spans.split(',')]
                    user_span_pos = []
                    for s in user_span_txts:
                        tok_span_ids = torch.tensor(tokenizer.encode(s, add_special_tokens=False), dtype=output_ids_trunc.dtype).to(device=output_ids_trunc.device)
                        for i in range(len(output_ids_trunc[0]) - len(tok_span_ids) + 1):
                            if torch.equal(output_ids_trunc[0][i : i + len(tok_span_ids)], tok_span_ids):
                                bio_labels[0][i] = BIO_START_LBL
                                for j in range(i + 1, i + len(tok_span_ids)):
                                    bio_labels[0][j] = BIO_INTERM_LBL
                                user_span_pos.append((i, i + len(tok_span_ids)))                      
            else:
                bio_labels = gt_bio_span
                # pad bio_labels with 0's to match dim=1 of output_hidden_states
                bio_labels = torch.cat(
                    [
                        torch.zeros((bio_labels.shape[0], IMAGE_TOKENS_OFFSET), device=output_hidden_states.device), bio_labels
                    ], dim=1
                )

            # for each sample, extract token spans based on predicted BIO labels.
            def aggregate_bio_spans(
                        batch_idx: int,
                        pred_lbls: list, 
                        seq_len: int = -1,
                        tokenizer = None,
                        aggregate_interm_lbl: bool = False,  # NOTE: True only when we want to aggregate faulty span of only 2's
                    ) -> dict:
                spans = [] # list of (start, end) indices for each detected span.
                indices = []  # list of BIO span token-level indices
                span_texts = []
                span_embeddings = []  # aggregated embeddings (mean-pooled over the span).
                bio_start_flag = False
                
                if seq_len != -1:
                    assert pred_lbls.size(0) == seq_len, \
                        f"pred_lbls.size(0)={pred_lbls.size(0)} != seq_len={seq_len}"

                def decode_span_text(start_idx, end_idx):
                    """Helper function to decode span text with proper index mapping"""
                    if multiple_choice:
                        # For multiple choice, output_ids_trunc contains the full input sequence
                        span_ids = output_ids_trunc[batch_idx, start_idx:end_idx]
                        return tokenizer.decode(span_ids, skip_special_tokens=False)
                    else:
                        # For generation, output_ids_trunc only contains generated tokens
                        # Map from full sequence indices to generated token indices
                        input_seq_len = input_ids.size(1) + IMAGE_TOKENS_OFFSET
                        
                        # Check if span is within the generated tokens
                        if start_idx >= input_seq_len and end_idx <= input_seq_len + output_ids_trunc.size(1):
                            span_start_gen = start_idx - input_seq_len
                            span_end_gen = end_idx - input_seq_len
                            span_ids = output_ids_trunc[batch_idx, span_start_gen:span_end_gen]
                            return tokenizer.decode(span_ids, skip_special_tokens=False)
                        else:
                            return ""  # Span is not in generated tokens
                        
                log_texts = []

                for tok_idx, bio_lbl in enumerate(pred_lbls.tolist()):
                    if bio_lbl == BIO_START_LBL:
                        if bio_start_flag:  # close any existing span before starting a new one
                            end = tok_idx
                            spans.append((start, end))
                            bio_start_flag = False
                            span_emb = sam_proj_hidden_states[batch_idx, indices, :].mean(dim=0)
                            span_embeddings.append(span_emb)
                            span_text = decode_span_text(start, end)
                            log_texts.append(f"start: {start}, end: {end}, span_text: {span_text}")
                            span_texts.append(span_text)
                        # start a new span    
                        start = tok_idx
                        indices = [tok_idx]
                        bio_start_flag = True
                        
                    elif bio_lbl == BIO_INTERM_LBL:
                        if bio_start_flag:
                            indices.append(tok_idx)
                        elif aggregate_interm_lbl:
                            start = tok_idx
                            indices = [tok_idx]
                            bio_start_flag = True
                            
                    elif bio_lbl == BIO_NO_LBL:
                        if bio_start_flag:
                            end = tok_idx
                            spans.append((start, end))
                            bio_start_flag = False
                            span_emb = sam_proj_hidden_states[batch_idx, indices, :].mean(dim=0)
                            span_embeddings.append(span_emb)
                            span_text = decode_span_text(start, end)
                            log_texts.append(f"start: {start}, end: {end}, span_text: {span_text}")
                            span_texts.append(span_text)

                print(f">> log_texts: {log_texts}")
                            
                # Handle case where sequence ends with a span (bio_start_flag is still True)
                if bio_start_flag:
                    end = len(pred_lbls.tolist())
                    spans.append((start, end))
                    span_emb = sam_proj_hidden_states[batch_idx, indices, :].mean(dim=0)
                    span_embeddings.append(span_emb)
                    span_text = decode_span_text(start, end)
                    span_texts.append(span_text)
                            
                return {
                    'spans': spans,
                    'span_texts': span_texts,
                    'embeddings': span_embeddings
                }
            
            output_dicts = []
            for batch_idx in range(output_hidden_states.size(0)):
                output_dict = aggregate_bio_spans(
                    batch_idx,
                    bio_labels[batch_idx], 
                    seq_len=output_hidden_states.size(1), 
                    tokenizer=tokenizer,
                    aggregate_interm_lbl=False  # NOTE: True only when we want to aggregate faulty span of only 2's
                )
                output_dicts.append(output_dict)
            # mask prediction block
            batch_pred_masks = []
            batch_iou_predictions = []

            if has_image_tokens:
                image_embeddings = self.get_visual_embs(images)  # [bsz, c, H, W]
                image_embeddings = image_embeddings.expand(input_ids.size(0), -1, -1, -1).contiguous()  # during inference, we deal with a single image with multiple answer choices

                for batch_idx, output_dict in enumerate(output_dicts):
                    iou_predictions_list = []
                    pred_masks = []
                    span_embeds_l = []
                    masks_l = []
                    for span_emb in output_dict["embeddings"]:  # len(span_emb) = # of detected spans
                        image_emb = image_embeddings[batch_idx].unsqueeze(0)
                        text_embed = span_emb.unsqueeze(0).unsqueeze(0)
                        # Use the SAM prompt encoder to obtain sparse and dense embeddings.
                        sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                            points=None, boxes=None, masks=None, text_embeds=text_embed
                        )
                        if self.use_feedback_loop:
                            dense_embeddings = self.mask_pooler(span_embeds_l, masks_l)

                        sparse_embeddings = sparse_embeddings.to(span_emb.dtype)
                        # Use the mask decoder with the image embeddings.
                        low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                            image_embeddings=image_emb,
                            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        # Postprocess the low-resolution masks to the desired size.
                        try:
                            resize_dim = resize_list[0]
                            original_size = label_list[0] if isinstance(label_list[0], tuple) else tuple(label_list[0].shape)
                        except IndexError as e:
                            print(f"IndexError: {e}")
                            raise RuntimeError(f"[PIXAR: evaluate()] {e}")

                        pred_mask = self.model.visual_model.postprocess_masks(
                            low_res_masks,
                            input_size=resize_dim,
                            original_size=original_size,
                        )
                        if self.use_feedback_loop:
                            span_embeds_l.append(span_emb.squeeze())
                            masks_l.append(low_res_masks.squeeze())

                        pred_masks.append(pred_mask[:, 0])
                        iou_predictions_list.append(iou_predictions.item())

                    batch_pred_masks.append(pred_masks)
                    batch_iou_predictions.append(iou_predictions_list)

            else:  # text-only case (no image in the input; no mask to predict)
                n = output_hidden_states.size(0)
                batch_pred_masks = [[] for _ in range(n)]
                batch_iou_predictions = [[] for _ in range(n)]

        return {
            "output_ids": output_ids_trunc if not multiple_choice else None,
            "spans_dicts": output_dicts,
            "pred_masks": batch_pred_masks,
            "ce_losses": ce_losses if multiple_choice else None,
            "iou_predictions": batch_iou_predictions
        }

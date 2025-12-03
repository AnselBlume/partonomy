from __future__ import annotations
import torch
from torch import Tensor
from einops import rearrange
import numpy as np
import torch.nn as nn
import abc
from model.segment_anything.modeling import PromptEncoder
from dataclasses import dataclass

@dataclass
class TemporalMaskPoolerConfig:
    freeze_mask_encoder: bool = False

# https://github.com/dvlab-research/LISA/blob/3cb2d4301f1af4691bd4f3938335ef06e76f155a/model/LISA.py#L275
class TemporalMaskPooler(nn.Module):
    def __init__(
        self,
        prompt_encoder: PromptEncoder,
        mask_modulator: MaskModulator,
        config: TemporalMaskPoolerConfig = TemporalMaskPoolerConfig()
    ):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_modulator = mask_modulator
        self.config = config

        mask_downscaling = self.prompt_encoder.mask_downscaling

        # Downscaling block 1
        self.block1 = nn.Sequential(mask_downscaling[0:2])
        self.activation1 = mask_downscaling[2]

        # Downsampling block 2
        self.block2 = nn.Sequential(mask_downscaling[3:5])
        self.activation2 = mask_downscaling[5]

        # 1x1 convolution to output dimension
        self.final_conv = mask_downscaling[6]

        # Attention pool
        self.queries = nn.Parameter(
            torch.randn(np.prod(self.prompt_encoder.image_embedding_size), 1, self.embed_dim)
        ) # (h * w, 1, d)
        self.attn = nn.MultiheadAttention(self.embed_dim, 8, batch_first=True)

        if self.config.freeze_mask_encoder:
            for param in self.prompt_encoder.mask_downscaling.parameters():
                param.requires_grad = False

    @property
    def embed_dim(self):
        '''
            Dimension of the prompt encoder's output embeddings.
        '''
        return self.prompt_encoder.embed_dim

    @property
    def down_h(self):
        '''
            Height of the downscaled mask feature map.
        '''
        return self.prompt_encoder.image_embedding_size[0]

    @property
    def down_w(self):
        '''
            Width of the downscaled mask feature map.
        '''
        return self.prompt_encoder.image_embedding_size[1]

    def forward(self, span_embeds: list[Tensor], masks: list[Tensor]) -> torch.Tensor:
        '''
        Args:
            span_embeds: list[Tensor] n_spans * (d)
            masks: list[Tensor] n_spans * (h, w)

        Returns:
            Downscaled mask features modulated by the span embeddings, then combined into
            a single feature map of shape (self.embed_dim, self.down_h, self.down_w).
        '''
        assert len(masks) == len(span_embeds)

        if len(span_embeds) == 0: # No mask to start
            mask_feats = rearrange(self.prompt_encoder.no_mask_embed.weight, '1 d -> d 1 1')
            mask_feats = mask_feats.expand(-1, self.down_h, self.down_w) # (d, h, w)
            return mask_feats

        masks = torch.stack(masks, dim=0)[:, None, ...] # (n_spans, 1, h, w)
        span_embeds = torch.stack(span_embeds, dim=0) # (n_spans, d)

        # Downscaling block 1
        mask_feats = self.block1(masks) # (n_spans, c_1, h_1, w_1)
        mask_feats = self.mask_modulator(mask_feats, span_embeds, 0) # (n_spans, c_1, h_1, w_1)
        mask_feats = self.activation1(mask_feats)

        # Downsampling block 2
        mask_feats = self.block2(mask_feats) # (n_spans, c_2, down_h, down_w)
        mask_feats = self.mask_modulator(mask_feats, span_embeds, 1) # (n_spans, c_2, down_h, down_w)
        mask_feats = self.activation2(mask_feats)

        # 1x1 convolution to output dimension
        mask_feats = self.final_conv(mask_feats) # (n_spans, embed_dim, down_h, down_w)

        # Attention pool
        mask_feats = mask_feats + self.prompt_encoder.get_dense_pe() # Positional encoding
        mask_feats = rearrange(mask_feats, 'n c h w -> (h w) n c')

        mask_feats, _ = self.attn(self.queries, mask_feats, mask_feats) # (h * w, 1, d)
        mask_feats = rearrange(mask_feats, '(h w) 1 c -> c h w', h=self.down_h, w=self.down_w)

        return mask_feats

class MaskModulator(nn.Module):
    def __init__(self, span_embed_dim: int, mask_feat_dims: list[int]):
        super().__init__()
        self.span_embed_dim = span_embed_dim
        self.mask_feat_dims = mask_feat_dims

    def _check_valid_block(self, block: int):
        if not (0 <= block < len(self.mask_feat_dims)):
            raise ValueError(f'Invalid block number: {block}; should be in range [0, {len(self.mask_feat_dims) - 1}]')

    @abc.abstractmethod
    def forward(self, mask_feats: Tensor, span_embeds: Tensor, block: int):
        '''
            mask_feats: (..., n_spans, c_i, h_i, w_i)
            span_embeds: (..., n_spans, d)
            block: The block the mask features come from
        '''
        pass

class FilmModulator(MaskModulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.projections = nn.ModuleList([
            nn.Linear(self.span_embed_dim, 2 * mask_feat_dim)
            for mask_feat_dim in self.mask_feat_dims
        ])

    def forward(self, mask_feats: Tensor, span_embeds: Tensor, block: int) -> Tensor:
        '''
            mask_feats: (..., n_spans, c_i, h_i, w_i)
            span_embeds: (..., n_spans, d)
            block: The block the mask features come from
        '''
        self._check_valid_block(block)

        proj = self.projections[block]
        scale, shift = proj(span_embeds).chunk(2, dim=-1) # Each (..., n_spans, c_i)

        scale = rearrange(scale, '... s c_i -> ... s c_i 1 1')
        shift = rearrange(shift, '... s c_i -> ... s c_i 1 1')

        mask_feats = mask_feats * (scale + 1) + shift # (..., n_spans, c_i, h_i, w_i)

        return mask_feats

if __name__ == '__main__':
    from model.LISA import LISAForCausalLM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    # Initialize LISA
    lisa = LISAForCausalLM.from_pretrained(
        'xinlai/LISA-13B-llama2-v1',
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        seg_token_idx=0 # Arbitrary placeholder for the next index added to the tokenizer
    ).to(device)

    lisa.get_model().initialize_vision_modules(lisa.get_model().config)

    # Initialize pooler
    prompt_encoder = lisa.model.visual_model.prompt_encoder
    decoder_embed_dim = lisa.model.visual_model.prompt_encoder.embed_dim

    conv_layers = [m for m in prompt_encoder.mask_downscaling if isinstance(m, nn.Conv2d)]
    mask_modulator = FilmModulator(
        span_embed_dim=decoder_embed_dim,
        mask_feat_dims=[conv_layer.out_channels for conv_layer in conv_layers]
    )

    mask_pooler = TemporalMaskPooler(
        prompt_encoder=prompt_encoder,
        mask_modulator=mask_modulator
    ).to(device, dtype)

    # Test feedback loop with dummy inputs
    img_dim = lisa.model.visual_model.image_encoder.img_size
    N_SPANS = 5

    span_embeds_l = []
    span_embeds = torch.randn(N_SPANS, decoder_embed_dim, device=device, dtype=dtype) # After text_hidden_fcs projection

    masks_l = []
    masks = torch.randint(0, 2, (N_SPANS, 1, img_dim, img_dim), device=device, dtype=dtype)

    image_embeddings = torch.randn(1, decoder_embed_dim, *prompt_encoder.image_embedding_size, device=device, dtype=dtype) # (1, d, h, w)

    with torch.no_grad():
        for i in range(N_SPANS):
            curr_span_embeds = span_embeds[i:i+1][:, None, :] # (1, 1, d) == (batch, n_prompts, d)
            sparse_embeddings, _ = prompt_encoder(None, None, None, text_embeds=curr_span_embeds) # (1, 1, d)
            sparse_embeddings = sparse_embeddings.to(dtype)

            dense_embeddings = mask_pooler(span_embeds_l, masks_l)[None, ...] # (1, c, h, w)
            print(f'Mask Pooler Output {i} shape: {dense_embeddings.shape}')

            low_res_masks, _ = lisa.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=lisa.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            ) # (1, 1, h_3, w_3)

            pred_mask = lisa.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=(img_dim, img_dim),
                original_size=(img_dim, img_dim),
            ) # (1, 1, h, w)
            pred_mask = (pred_mask[0, 0] > 0).to(dtype) # The full-res mask which is NOT fed back into the loop

            print(f'Low Res Mask {i} shape: {low_res_masks.shape}')
            print(f'Pred Mask {i} shape: {pred_mask.shape}')

            # NOTE: PromptEncoder expects low-res masks as input
            # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/predictor.py#L112
            span_embeds_l.append(span_embeds[i]) # (d,)
            masks_l.append(low_res_masks.squeeze()) # (h_2, w_2)
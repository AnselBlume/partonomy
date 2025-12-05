import PIL.Image
from PIL.Image import Image as PILImage
import torch
from rembg import remove, new_session
import logging
from data.partonomy_core.annotation_completion.embed import ImageEmbedder
from data.partonomy_core.annotation_completion.embed.embed import DEFAULT_RESIZE_INTERPOLATION
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HeatmapGeneratorConfig:
    # For heatmap generation
    use_cosine_similarity_for_heatmap: bool = True

    # For heatmap visualization
    threshold_heatmap: bool = False
    heatmap_threshold: float = .7

    use_relative_scaling: bool = False

    use_absolute_scaling: bool = False
    center: int = 0
    clamp_min: int = 0
    clamp_max: float = .3
    scale: int = 1

    heatmap_cmap: str = 'inferno'
    heatmap_blend_ratio: float = .5

class HeatmapGenerator:
    def __init__(
        self,
        embedder: ImageEmbedder,
        config: HeatmapGeneratorConfig = HeatmapGeneratorConfig()
    ):
        self.embedder = embedder
        self.config = config
        self.rembg_session = new_session()

    @torch.no_grad()
    def generate_heatmap(
        self,
        input_image: PILImage,
        query_embed: torch.Tensor
    ):

        # Get image features upscaled to the size of the image
        patch_embeds = self.embedder.get_patch_embeddings([input_image])[0] # (h, w, d)

        # Collapses range but requires less tuning
        if self.config.use_cosine_similarity_for_heatmap:
            logger.debug('Using cosine similarity for heatmap')
            heatmap = torch.cosine_similarity(patch_embeds, query_embed, dim=-1) # (h, w)

        else: # Dot product; potentially more expressive, but requires more tuning of clamp_min, clamp_max, and scale
            logger.debug('Using dot product for heatmap')
            heatmap = patch_embeds @ query_embed

        return heatmap

    def restrict_heatmap_to_foreground(self, heatmap: torch.Tensor, image: PILImage) -> torch.Tensor:
        '''
            heatmap: (h, w) torch.Tensor
            image: PIL.Image.Image
            session: rembg.Session
            is_relative: bool

            Returns: Heatmap restricted to the foreground of the image.
        '''
        foreground_mask = np.array(remove(image, session=self.rembg_session, only_mask=True, post_process_mask=True))
        foreground_mask = torch.from_numpy(foreground_mask).bool()
        background_mask = foreground_mask.logical_not().to(heatmap.device)

        if self.config.use_relative_scaling:
            heatmap[background_mask] = heatmap.min()
        else:
            heatmap[background_mask] = -torch.inf

        return heatmap

    def threshold_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        '''
            heatmap: (h, w) torch.Tensor

            Returns: Thresholded heatmap.
        '''
        return (heatmap > self.config.heatmap_threshold).float()

    def postprocess_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        '''
            heatmap: (h, w) torch.Tensor

            Returns: Postprocessed heatmap.
        '''
        # Normalize heatmap
        logger.debug(f'Heatmap min: {heatmap.min()}, max: {heatmap.max()}')

        assert not (self.config.use_relative_scaling and self.config.use_absolute_scaling), 'Cannot use both relative and absolute scaling'

        # If we're sure the concept is in the image
        if self.config.use_relative_scaling:
            logger.debug('Using relative heatmap; ignoring center, clamp_min, clamp_max, and scale')
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # Normalize to [0, 1]

        # If we're not sure it's in the image. Requires more tuning
        elif self.config.use_absolute_scaling:
            logger.debug(f'Using absolute heatmap; center: {self.config.center}, clamp_min: {self.config.clamp_min}, '
                         f'clamp_max: {self.config.clamp_max}, scale: {self.config.scale}')

            heatmap = (heatmap - self.config.center) * self.config.scale
            heatmap = heatmap.clamp(self.config.clamp_min, self.config.clamp_max)
            heatmap = (heatmap - self.config.clamp_min) / (self.config.clamp_max - self.config.clamp_min) # Normalize to [0, 1]

        if self.config.threshold_heatmap:
            heatmap = self.threshold_heatmap(heatmap)

        return heatmap

    def image_from_heatmap(self, heatmap: torch.Tensor, image: PILImage) -> PILImage:
        '''
            heatmap: (h, w) torch.Tensor: The (usually postprocessed) heatmap
            image: PIL.Image.Image

            Returns: PIL.Image.Image of the heatmap overlaid on the image.
        '''
        if self.embedder.fe.resize_images:
            image = TF.resize(image, heatmap.shape[-2:], interpolation=DEFAULT_RESIZE_INTERPOLATION)

        assert heatmap.shape[-2:] == image.size[::-1], 'Heatmap and image must have the same dimensions'

        # Convert heatmap to RGB
        cmap = plt.get_cmap(self.config.heatmap_cmap)
        heatmap_rgb = cmap(heatmap.numpy())[...,:3] # (h, w, 3); ignore alpha channel of ones
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

        # Convert image to RGB
        image_rgb = np.array(image)

        # Blend heatmap with image
        blended = self.config.heatmap_blend_ratio * heatmap_rgb + (1 - self.config.heatmap_blend_ratio) * image_rgb
        blended = blended.astype(np.uint8)

        return PIL.Image.fromarray(blended)
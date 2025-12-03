from .clip_encoder import CLIPVisionTower
from .NACLIP.clip import clip


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if vision_tower.startswith("naclip"):
        clip_path = "ViT-L/14"
        arch = 'reduced'
        attn_strategy = 'naclip'
        gaussian_std=5.0
        naclip_vision_tower, _ = clip.load(clip_path, jit=False)
        naclip_vision_tower.visual.set_params(arch, attn_strategy, gaussian_std, no_proj=True)
        return naclip_vision_tower
    elif (
        vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "clip" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")

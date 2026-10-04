"""
Pattern-conditioned UNet: embed a fixed checkerboard-like pattern into ad backgrounds.

Training entrypoints: ``python -m pattern_unet.train`` or ``python train_pattern_unet.py``.
Inference: ``python -m pattern_unet.infer`` or ``python infer_pattern_unet.py``.
"""

from pattern_unet.dataset import (
    PatternAdDataset,
    PatternAdDatasetConfig,
    blend_output_with_fg_mask,
    load_image_rgb_with_alpha_composite,
    pil_fg_mask_to_tensor_01,
)
from pattern_unet.generator import (
    WideShallowPatternUNet,
    build_wide_shallow_pattern_unet,
    rgb_to_grayscale,
)

__all__ = [
    "WideShallowPatternUNet",
    "build_wide_shallow_pattern_unet",
    "rgb_to_grayscale",
    "PatternAdDataset",
    "PatternAdDatasetConfig",
    "blend_output_with_fg_mask",
    "load_image_rgb_with_alpha_composite",
    "pil_fg_mask_to_tensor_01",
]

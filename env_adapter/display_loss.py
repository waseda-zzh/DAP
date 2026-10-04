from __future__ import annotations

import numpy as np
import torch


def build_display_colors(icc_path: str | None = None, levels_per_channel: int = 4) -> np.ndarray:
    # Lightweight fallback: RGB grid anchors in [0,1].
    lv = max(2, int(levels_per_channel))
    vals = np.linspace(0.0, 1.0, lv, dtype=np.float32)
    grid = np.stack(np.meshgrid(vals, vals, vals, indexing="ij"), axis=-1).reshape(-1, 3)
    # Convert RGB grid to BGR ordering to match training tensors.
    return grid[:, ::-1].copy()


def compute_display_loss(adapted_bchw: torch.Tensor, display_colors_bgr: torch.Tensor) -> torch.Tensor:
    # Encourage colors to stay near display anchor set.
    b, c, h, w = adapted_bchw.shape
    if c != 3:
        raise ValueError("Expected adapted_bchw with 3 channels")
    x = adapted_bchw.permute(0, 2, 3, 1).reshape(-1, 3)  # (N,3)
    anchors = display_colors_bgr.to(device=x.device, dtype=x.dtype)  # (K,3)
    d2 = (x[:, None, :] - anchors[None, :, :]).pow(2).sum(dim=2)
    min_d2 = d2.min(dim=1).values
    return min_d2.mean()


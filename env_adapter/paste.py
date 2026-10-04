from __future__ import annotations

import torch
import torch.nn.functional as F


def paste_adapted_native_on_scene_homography(
    scene_bgr_bchw: torch.Tensor,
    adapted_native_bchw: torch.Tensor,
    homography_b2s_b33: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scene_bgr_bchw.dim() != 4 or scene_bgr_bchw.size(1) != 3:
        raise ValueError("scene_bgr_bchw must be (B,3,H,W)")
    if adapted_native_bchw.dim() != 4 or adapted_native_bchw.size(1) != 3:
        raise ValueError("adapted_native_bchw must be (B,3,ph,pw)")
    if homography_b2s_b33.dim() != 3 or homography_b2s_b33.size(1) != 3 or homography_b2s_b33.size(2) != 3:
        raise ValueError("homography_b2s_b33 must be (B,3,3)")

    b, _, h, w = scene_bgr_bchw.shape
    _, _, ph, pw = adapted_native_bchw.shape
    inv_h = torch.inverse(homography_b2s_b33)

    xs = torch.linspace(0.0, float(w - 1), w, device=scene_bgr_bchw.device, dtype=scene_bgr_bchw.dtype)
    ys = torch.linspace(0.0, float(h - 1), h, device=scene_bgr_bchw.device, dtype=scene_bgr_bchw.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # (H,W)
    ones = torch.ones((h, w), device=scene_bgr_bchw.device, dtype=scene_bgr_bchw.dtype)
    pts = torch.stack([grid_x, grid_y, ones], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)

    mapped = torch.einsum("bij,bjhw->bihw", inv_h, pts)
    u = mapped[:, 0] / mapped[:, 2].clamp(min=1e-6)
    v = mapped[:, 1] / mapped[:, 2].clamp(min=1e-6)
    u_norm = ((u + 0.5) / float(pw)) * 2.0 - 1.0
    v_norm = ((v + 0.5) / float(ph)) * 2.0 - 1.0
    sample_grid = torch.stack([u_norm, v_norm], dim=-1)

    warped = F.grid_sample(adapted_native_bchw, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    src_mask = torch.ones((b, 1, ph, pw), dtype=scene_bgr_bchw.dtype, device=scene_bgr_bchw.device)
    warped_mask = F.grid_sample(src_mask, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = (warped_mask > 0.5).to(scene_bgr_bchw.dtype)
    out = scene_bgr_bchw * (1.0 - mask) + warped * mask
    return out, mask


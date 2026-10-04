"""
BGR variants of style-matching helpers from reference/patch.py.
Original patch.py uses RGB in blur_to_match; OpenCV loads BGR, so we use COLOR_BGR2GRAY here.
Does not import placement.py (reference/patch.py depends on it).
"""
from __future__ import annotations

import numpy as np
import cv2


def _spatial_hw(bgr: np.ndarray) -> tuple[int, int]:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return 0, 0
    return int(bgr.shape[0]), int(bgr.shape[1])


def _is_empty_spatial(bgr: np.ndarray) -> bool:
    h, w = _spatial_hw(bgr)
    return h <= 0 or w <= 0 or bgr.size == 0


def match_color_statistics(src_bgr: np.ndarray, target_bgr: np.ndarray, blend_factor: float = 0.5) -> np.ndarray:
    """
    Match per-channel color statistics of patch (src) to target ROI, then blend with original.
    blend_factor: higher = keep more of original patch colors.
    """
    if _is_empty_spatial(src_bgr) or _is_empty_spatial(target_bgr):
        return np.clip(src_bgr.astype(np.float32), 0, 255)

    src = src_bgr.astype(np.float32)
    target = target_bgr.astype(np.float32)
    src_mean, src_std = src.mean(axis=(0, 1)), src.std(axis=(0, 1))
    tgt_mean, tgt_std = target.mean(axis=(0, 1)), target.std(axis=(0, 1))
    matched = (src - src_mean) / (src_std + 1e-6) * (tgt_std + 1e-6) + tgt_mean
    blended = blend_factor * src + (1.0 - blend_factor) * matched
    return np.clip(blended, 0, 255)


def match_patch_color(patch_bgr: np.ndarray, roi_bgr: np.ndarray, blend_factor: float = 0.5) -> np.ndarray:
    return match_color_statistics(patch_bgr, roi_bgr, blend_factor=blend_factor)


def blur_to_match(patch_bgr: np.ndarray, roi_bgr: np.ndarray) -> np.ndarray:
    """
    If patch is sharper than ROI (Laplacian variance), apply mild Gaussian blur to patch.
    """
    if _is_empty_spatial(patch_bgr) or _is_empty_spatial(roi_bgr):
        return patch_bgr
    roi_gray = cv2.cvtColor(roi_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    patch_gray = cv2.cvtColor(patch_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    roi_blur = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    patch_blur = cv2.Laplacian(patch_gray, cv2.CV_64F).var()
    out = patch_bgr
    if patch_blur > 0 and roi_blur > 0 and patch_blur > roi_blur:
        blur_ksize = int(np.clip((patch_blur / roi_blur) ** 0.5, 1, 5))
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)
    return out


def add_realistic_noise(patch_bgr: np.ndarray, std: float = 2.0, rng: np.random.RandomState | None = None) -> np.ndarray:
    if _is_empty_spatial(patch_bgr):
        return patch_bgr
    if rng is not None:
        noise = rng.normal(0, std, patch_bgr.shape).astype(np.float32)
    else:
        noise = np.random.normal(0, std, patch_bgr.shape).astype(np.float32)
    return np.clip(patch_bgr.astype(np.float32) + noise, 0, 255)


def apply_patch_style(
    patch_resized_bgr: np.ndarray,
    roi_bgr: np.ndarray,
    *,
    match_color: bool = True,
    blend_factor: float = 0.5,
    blur: bool = True,
    noise: bool = False,
    noise_std: float = 2.0,
    noise_rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Make pasted patch closer to local scene style (same order as reference/patch.py realistic_patch_applier).
    patch_resized_bgr and roi_bgr must be same spatial size.
    """
    if _is_empty_spatial(patch_resized_bgr):
        return patch_resized_bgr
    out = patch_resized_bgr
    if match_color:
        out = match_patch_color(out, roi_bgr, blend_factor=blend_factor)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur:
        out = blur_to_match(out, roi_bgr)
    if noise:
        out = add_realistic_noise(out, std=noise_std, rng=noise_rng)
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out

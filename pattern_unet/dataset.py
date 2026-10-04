from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset


def _list_images_recursively(root_dir: str, exts: Iterable[str]) -> list[str]:
    exts_l = {str(e).lower().lstrip(".") for e in exts}
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    paths: list[str] = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            paths.extend(_list_images_recursively(p, exts))
        else:
            ext = os.path.splitext(name)[1].lower().lstrip(".")
            if ext in exts_l:
                paths.append(p)
    return sorted(paths)


def pil_to_tensor_rgb_01(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL RGB image to torch float tensor in [0,1]:
      (H,W,3) -> (3,H,W)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    # PIL -> torch without external deps.
    import numpy as np

    arr = np.array(img, dtype="float32")  # (H,W,3) uint8->float32
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB image array shape (H,W,3)")
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous() / 255.0
    return t


def load_image_rgb_with_alpha_composite(
    img_path: str,
    *,
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Load an image (possibly palette/with transparency) and composite it onto a background,
    returning RGB image.

    This avoids PIL's Palette transparency UserWarning when calling convert("RGB") directly.
    """
    img = Image.open(img_path)
    # Convert to RGBA first to preserve alpha information regardless of source mode.
    img_rgba = img.convert("RGBA")
    bg_rgba = Image.new("RGBA", img_rgba.size, (*bg_rgb, 255))
    comp_rgba = Image.alpha_composite(bg_rgba, img_rgba)
    return comp_rgba.convert("RGB")


def _resolve_fg_mask_path(ad_path: str, mask_dir: str | None, mask_suffix: str) -> str | None:
    """
    mask_suffix: e.g. "_mask.png" -> <stem>_mask.png in mask_dir (or same dir as ad if mask_dir is None).
    """
    stem, _ = os.path.splitext(os.path.basename(ad_path))
    base_dir = mask_dir.strip() if mask_dir and str(mask_dir).strip() else os.path.dirname(ad_path)
    cand = os.path.join(base_dir, stem + mask_suffix)
    return cand if os.path.isfile(cand) else None


def pil_fg_mask_to_tensor_01(
    mask_path: str,
    size_hw: tuple[int, int],
    *,
    invert: bool = False,
) -> torch.Tensor:
    """Load single-channel mask, resize with nearest, values in [0,1], shape (1,H,W)."""
    import numpy as np

    m = Image.open(mask_path).convert("L")
    m = m.resize((size_hw[1], size_hw[0]), resample=Image.NEAREST)
    arr = np.array(m, dtype="float32") / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).contiguous()
    if invert:
        t = 1.0 - t
    return t.clamp(0.0, 1.0)


def blend_output_with_fg_mask(
    raw_out_bchw: torch.Tensor,
    ad_rgb_bchw: torch.Tensor,
    fg_mask_bchw: torch.Tensor,
) -> torch.Tensor:
    """
    raw_out, ad: (B,3,H,W) in [0,1]
    fg_mask: (B,1,H,W) in [0,1], 1=foreground (keep ad), 0=background (use raw_out).
    """
    fg = fg_mask_bchw.clamp(0.0, 1.0)
    bg = 1.0 - fg
    return fg * ad_rgb_bchw + bg * raw_out_bchw


@dataclass
class PatternAdDatasetConfig:
    ad_images_dir: str
    error_pattern_path: str
    image_size: int = 256
    exts: tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "tiff", "webp")
    pattern_resize_mode: str = "bilinear"  # for tensor resize if needed
    transparent_bg_rgb: tuple[int, int, int] = (255, 255, 255)
    # Foreground mask: 1 = foreground (object, do not fuse grid), 0 = background.
    fg_mask_enabled: bool = False
    fg_mask_dir: str | None = None  # None -> same directory as each ad image
    fg_mask_suffix: str = "_mask.png"  # file name = <ad_stem> + suffix
    fg_mask_invert: bool = False  # set True if your mask uses white=background
    fg_mask_default_all_background: bool = True  # if mask file missing: all 0 (no protected fg)


class PatternAdDataset(Dataset):
    """
    Dataset for unpaired training:
      - Input: ad image
      - Condition: fixed error pattern template
      - Output target: none (losses compare output to ad and pattern)
    """

    def __init__(self, cfg: PatternAdDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.ad_images_dir = cfg.ad_images_dir
        self.image_size = int(cfg.image_size)
        self.transparent_bg_rgb = tuple(cfg.transparent_bg_rgb)
        self.fg_mask_enabled = bool(cfg.fg_mask_enabled)
        self.fg_mask_dir = cfg.fg_mask_dir
        self.fg_mask_suffix = str(cfg.fg_mask_suffix)
        self.fg_mask_invert = bool(cfg.fg_mask_invert)
        self.fg_mask_default_all_background = bool(cfg.fg_mask_default_all_background)

        self.ad_paths = _list_images_recursively(self.ad_images_dir, cfg.exts)
        if not self.ad_paths:
            raise RuntimeError(f"No ad images found under: {self.ad_images_dir}")

        if not os.path.isfile(cfg.error_pattern_path):
            raise FileNotFoundError(f"error_pattern_path not found: {cfg.error_pattern_path}")
        self.pattern_pil = load_image_rgb_with_alpha_composite(
            cfg.error_pattern_path,
            bg_rgb=self.transparent_bg_rgb,
        )
        self.pattern_pil = self.pattern_pil.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        self.pattern_tensor = pil_to_tensor_rgb_01(self.pattern_pil)  # (3,H,W)

    def __len__(self) -> int:
        return len(self.ad_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ad_path = self.ad_paths[idx]
        img = load_image_rgb_with_alpha_composite(ad_path, bg_rgb=self.transparent_bg_rgb)
        orig_w, orig_h = img.size
        img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        ad_t = pil_to_tensor_rgb_01(img)  # (3,H,W) [0,1]
        pattern_t = self.pattern_tensor
        # meta: (orig_h, orig_w) for visualization-only resizing
        meta_hw = torch.tensor([orig_h, orig_w], dtype=torch.int32)

        h = w = self.image_size
        if self.fg_mask_enabled:
            mp = _resolve_fg_mask_path(ad_path, self.fg_mask_dir, self.fg_mask_suffix)
            if mp is not None:
                fg_mask_t = pil_fg_mask_to_tensor_01(
                    mp, (h, w), invert=self.fg_mask_invert
                )
            else:
                if not self.fg_mask_default_all_background:
                    raise FileNotFoundError(
                        f"fg_mask_enabled but mask not found for {ad_path!r} "
                        f"(expected *{self.fg_mask_suffix} under {self.fg_mask_dir or 'same dir'})"
                    )
                fg_mask_t = torch.zeros(1, h, w, dtype=torch.float32)
        else:
            fg_mask_t = torch.zeros(1, h, w, dtype=torch.float32)

        return ad_t, pattern_t, fg_mask_t, meta_hw


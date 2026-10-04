from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EnvAdapterCarlaDatasetConfig:
    carla_root_dir: str
    patch_dir: str
    carla_towns: set[str] | None = None
    error_pattern_path: str | None = None
    error_pattern_dir: str | None = None
    seed: int = 0
    max_frames_per_town: int | None = None
    patch_width: int | None = None


def _to_tensor_chw_01(bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bgr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()


def _list_towns(root: str) -> list[str]:
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d)) and d.lower().startswith("carla_town")]


def _align_error_pattern_to_patch(patch_bgr: np.ndarray, patch_path: str, cfg: EnvAdapterCarlaDatasetConfig) -> torch.Tensor | None:
    path = None
    if cfg.error_pattern_dir:
        stem = os.path.splitext(os.path.basename(patch_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            cand = os.path.join(cfg.error_pattern_dir, stem + ext)
            if os.path.isfile(cand):
                path = cand
                break
    if path is None and cfg.error_pattern_path:
        path = cfg.error_pattern_path
    if not path:
        return None
    p = cv2.imread(path, cv2.IMREAD_COLOR)
    if p is None:
        raise FileNotFoundError(f"Failed reading error pattern: {path}")
    ph, pw = patch_bgr.shape[:2]
    inter = cv2.INTER_AREA if (p.shape[0] * p.shape[1]) > (ph * pw) else cv2.INTER_LINEAR
    rs = cv2.resize(p, (pw, ph), interpolation=inter)
    return _to_tensor_chw_01(rs)


class EnvAdapterCarlaDataset(Dataset):
    def __init__(self, cfg: EnvAdapterCarlaDatasetConfig):
        self.cfg = cfg
        self.seed = int(cfg.seed)
        root = cfg.carla_root_dir
        if not os.path.isdir(root):
            raise FileNotFoundError(f"carla_root_dir not found: {root!r}")
        hom_dir = os.path.join(root, "result")
        if not os.path.isdir(hom_dir):
            raise FileNotFoundError(f"homography dir not found: {hom_dir!r}")

        towns = sorted(cfg.carla_towns) if cfg.carla_towns else _list_towns(root)
        if not towns:
            raise RuntimeError(f"No towns found under {root!r}")

        scenes = []
        for town in towns:
            tdir = town if os.path.isabs(town) else os.path.join(root, town)
            imgs = sorted(glob.glob(os.path.join(tdir, "*.png")))
            if cfg.max_frames_per_town is not None:
                imgs = imgs[: int(cfg.max_frames_per_town)]
            for img in imgs:
                stem = os.path.splitext(os.path.basename(img))[0]
                hp = os.path.join(hom_dir, f"homography_{stem}.npy")
                if os.path.isfile(hp):
                    scenes.append((img, hp, stem))
        if not scenes:
            raise RuntimeError("No CARLA (scene,homography) pairs found.")
        self.scenes = scenes

        patch_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            patch_paths.extend(glob.glob(os.path.join(cfg.patch_dir, ext)))
        patch_paths = sorted(set(patch_paths))
        if not patch_paths:
            raise RuntimeError(f"No patch images found in {cfg.patch_dir!r}")
        patches = []
        kept_paths = []
        for p in patch_paths:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                continue
            if cfg.patch_width is not None and im.shape[1] != int(cfg.patch_width):
                continue
            patches.append(im)
            kept_paths.append(p)
        if not patches:
            raise RuntimeError("No patch images remain after patch_width filter.")
        self.patches = patches
        self.patch_paths = kept_paths

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene_path, hp, stem = self.scenes[idx]
        scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)
        if scene is None:
            raise FileNotFoundError(f"Failed reading scene {scene_path}")
        H = np.load(hp).astype(np.float32)
        if H.shape != (3, 3):
            raise ValueError(f"Bad homography shape {H.shape} at {hp}")
        h, w = scene.shape[:2]

        rng = np.random.RandomState(self.seed + idx * 9973)
        pidx = int(rng.randint(0, len(self.patches)))
        patch = self.patches[pidx]
        pat_t = _align_error_pattern_to_patch(patch, self.patch_paths[pidx], self.cfg)

        # Axis-aligned ROI from projected corners (used for environment stats only).
        ph, pw = patch.shape[:2]
        corners = np.array([[0.0, 0.0, 1.0], [pw - 1.0, 0.0, 1.0], [0.0, ph - 1.0, 1.0], [pw - 1.0, ph - 1.0, 1.0]], dtype=np.float32).T
        out = H @ corners
        u = out[0] / np.maximum(out[2], 1e-6)
        v = out[1] / np.maximum(out[2], 1e-6)
        x1 = int(np.floor(np.min(u))); y1 = int(np.floor(np.min(v)))
        x2 = int(np.ceil(np.max(u))); y2 = int(np.ceil(np.max(v)))
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w)); y2 = max(y1 + 1, min(y2, h))

        item = {
            "scene_bgr": _to_tensor_chw_01(scene),
            "patch_native_bgr": _to_tensor_chw_01(patch),
            "roi_box": torch.tensor([x1, y1, x2, y2], dtype=torch.int64),
            "paste_wh": torch.tensor([x2 - x1, y2 - y1], dtype=torch.int64),
            "homography_b2s": torch.from_numpy(H),
            "stem": stem,
        }
        if pat_t is not None:
            item["pattern_native_bgr"] = pat_t
        return item


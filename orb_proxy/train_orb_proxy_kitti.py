import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from orb_proxy.orb_proxy_model import build_orb_proxy
    from orb_proxy.patch_style_bgr import apply_patch_style
except Exception:
    # Backward-compatible when running this script from inside orb_proxy/ directly.
    from orb_proxy_model import build_orb_proxy
    from patch_style_bgr import apply_patch_style


@dataclass
class BBox2D:
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self, w: int, h: int) -> "BBox2D":
        return BBox2D(
            x1=max(0.0, min(float(self.x1), w - 1.0)),
            y1=max(0.0, min(float(self.y1), h - 1.0)),
            x2=max(0.0, min(float(self.x2), w - 1.0)),
            y2=max(0.0, min(float(self.y2), h - 1.0)),
        )

    @property
    def width(self) -> int:
        return max(1, int(round(self.x2 - self.x1)))

    @property
    def height(self) -> int:
        return max(1, int(round(self.y2 - self.y1)))

    @property
    def as_int(self):
        x1 = int(round(self.x1))
        y1 = int(round(self.y1))
        x2 = int(round(self.x2))
        y2 = int(round(self.y2))
        return x1, y1, x2, y2


def find_image_for_id(images_dir: str, stem: str) -> str:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"]
    for ext in exts:
        matches = glob.glob(os.path.join(images_dir, ext))
        for m in matches:
            if os.path.splitext(os.path.basename(m))[0] == stem:
                return m
    raise FileNotFoundError(f"Cannot find image for stem={stem} in {images_dir}")


def kitti_object_distance_m(x: float, y: float, z: float, metric: str) -> float:
    """
    Distance from camera origin using KITTI 3D location (camera coords, meters).
    metric:
      - z_depth: |z| (KITTI z is depth along optical axis; use abs for safety)
      - euclidean: sqrt(x^2+y^2+z^2)
      - xz_horizontal: sqrt(x^2+z^2) (ignore vertical y)
    """
    m = metric.strip().lower()
    if m == "z_depth":
        return abs(float(z))
    if m == "euclidean":
        return float(math.sqrt(x * x + y * y + z * z))
    if m in ("xz_horizontal", "xz", "horizontal"):
        return float(math.sqrt(x * x + z * z))
    raise ValueError(f"Unknown distance_metric: {metric!r} (use z_depth, euclidean, xz_horizontal)")


def parse_kitti_label_file(
    label_path: str,
    *,
    keep_classes: set[str] | None = None,
    occluded_allow: set[int] | None = None,
    truncated_max: float | None = None,
    skip_classes: set[str] | None = None,
    max_distance_m: float | None = None,
    distance_metric: str = "z_depth",
) -> list[BBox2D]:
    """
    KITTI object detection label format:
      type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom ...

    Filters (AND):
      - skip_classes: dropped first (default DontCare).
      - keep_classes: if non-empty, only these types.
      - occluded_allow: if not None, only these occlusion codes (KITTI int).
      - truncated_max: if not None, only rows with truncated <= this (e.g. 0.0 = non-truncated).
      - max_distance_m: if not None, only rows whose 3D location distance (see distance_metric) <= threshold.
    """
    bboxes = []
    if keep_classes is None:
        keep_classes = set()
    if skip_classes is None:
        skip_classes = {"DontCare"}

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls in skip_classes:
                continue
            if keep_classes and cls not in keep_classes:
                continue

            truncated = float(parts[1])
            occluded = int(float(parts[2]))  # tolerate "0.0" if any

            if truncated_max is not None and truncated > truncated_max + 1e-9:
                continue
            if occluded_allow is not None and occluded not in occluded_allow:
                continue

            if max_distance_m is not None:
                if len(parts) < 15:
                    continue
                xc, yc, zc = map(float, parts[11:14])
                dist = kitti_object_distance_m(xc, yc, zc, distance_metric)
                if dist > max_distance_m + 1e-9:
                    continue

            x1, y1, x2, y2 = map(float, parts[4:8])
            bboxes.append(BBox2D(x1=x1, y1=y1, x2=x2, y2=y2))
    return bboxes


def color_jitter_bgr(patch_bgr: np.ndarray, brightness: float, contrast: float, saturation: float, sharpness: float) -> np.ndarray:
    """
    Simple BGR jitter to mimic attack fusion variations.
    brightness: multiply V in HSV
    contrast: alpha in cv2.convertScaleAbs
    saturation: multiply S in HSV
    sharpness: >=0 apply unsharp kernel
    """
    out = patch_bgr.astype(np.float32)

    # brightness/contrast
    out = out * contrast
    out = out + (brightness - 1.0) * 255.0
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    # saturation
    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # sharpness
    if sharpness > 0:
        amount = float(sharpness)
        kernel = np.array(
            [[0, -amount, 0], [-amount, 1 + 4 * amount, -amount], [0, -amount, 0]],
            dtype=np.float32,
        )
        out = cv2.filter2D(out, -1, kernel)

    return out


def paste_patch_into_context(
    scene_bgr: np.ndarray,
    patch_bgr: np.ndarray,
    bbox: BBox2D,
    context_bbox: BBox2D,
    jitter_rng: np.random.RandomState,
    aug: bool,
    *,
    style_match_color: bool = True,
    style_blend_factor: float = 0.5,
    style_blur: bool = True,
    style_noise: bool = False,
    style_noise_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      composite_context_bgr: Hc x Wc x 3
      mask_context_uint8: Hc x Wc in {0,255}
    """
    h, w = scene_bgr.shape[:2]
    context_bbox = context_bbox.clamp(w, h)
    x1c, y1c, x2c, y2c = context_bbox.as_int
    context = scene_bgr[y1c:y2c, x1c:x2c].copy()
    hc, wc = context.shape[:2]

    bbox_clamped = bbox.clamp(w, h)
    x1, y1, x2, y2 = bbox_clamped.as_int

    # bbox region inside context crop
    x1r = max(0, x1 - x1c)
    y1r = max(0, y1 - y1c)
    x2r = min(wc, x2 - x1c)
    y2r = min(hc, y2 - y1c)

    bw = max(1, x2r - x1r)
    bh = max(1, y2r - y1r)

    patch_resized = cv2.resize(patch_bgr, (bw, bh), interpolation=cv2.INTER_LINEAR)

    if aug:
        # jitter factors (empirical, small range)
        brightness = float(jitter_rng.uniform(0.85, 1.15))
        contrast = float(jitter_rng.uniform(0.85, 1.15))
        saturation = float(jitter_rng.uniform(0.85, 1.15))
        sharpness = float(jitter_rng.uniform(0.0, 0.15))
        patch_resized = color_jitter_bgr(patch_resized, brightness=brightness, contrast=contrast, saturation=saturation, sharpness=sharpness)

    # Same idea as reference/patch.py: align patch to local ROI (color stats + optional blur/noise).
    roi_region = context[y1r:y2r, x1r:x2r].copy()
    if style_match_color or style_blur or style_noise:
        patch_resized = apply_patch_style(
            patch_resized,
            roi_region,
            match_color=style_match_color,
            blend_factor=style_blend_factor,
            blur=style_blur,
            noise=style_noise,
            noise_std=style_noise_std,
            noise_rng=jitter_rng,
        )

    composite = context
    composite[y1r:y2r, x1r:x2r] = patch_resized

    mask = np.zeros((hc, wc), dtype=np.uint8)
    mask[y1r:y2r, x1r:x2r] = 255
    return composite, mask


def paste_patch_into_full_scene(
    scene_bgr: np.ndarray,
    patch_bgr: np.ndarray,
    bbox: BBox2D,
    jitter_rng: np.random.RandomState,
    aug: bool,
    *,
    style_match_color: bool = True,
    style_blend_factor: float = 0.5,
    style_blur: bool = True,
    style_noise: bool = False,
    style_noise_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Paste attack patch into bbox on the **full** scene (same resolution as KITTI image).
    Returns composite HxWx3 BGR, mask HxW uint8 (255 on bbox).
    """
    h, w = scene_bgr.shape[:2]
    bbox_clamped = bbox.clamp(w, h)
    x1, y1, x2, y2 = bbox_clamped.as_int
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    patch_resized = cv2.resize(patch_bgr, (bw, bh), interpolation=cv2.INTER_LINEAR)

    if aug:
        brightness = float(jitter_rng.uniform(0.85, 1.15))
        contrast = float(jitter_rng.uniform(0.85, 1.15))
        saturation = float(jitter_rng.uniform(0.85, 1.15))
        sharpness = float(jitter_rng.uniform(0.0, 0.15))
        patch_resized = color_jitter_bgr(
            patch_resized,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            sharpness=sharpness,
        )

    roi_region = scene_bgr[y1:y2, x1:x2].copy()
    if style_match_color or style_blur or style_noise:
        patch_resized = apply_patch_style(
            patch_resized,
            roi_region,
            match_color=style_match_color,
            blend_factor=style_blend_factor,
            blur=style_blur,
            noise=style_noise,
            noise_std=style_noise_std,
            noise_rng=jitter_rng,
        )

    composite = scene_bgr.copy()
    composite[y1:y2, x1:x2] = patch_resized
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return composite, mask


def scale_image_and_mask_max_side(
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform scale so max(H,W) <= max_side (no upscale)."""
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, mask_uint8
    s = max_side / float(m)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return (
        cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR),
        cv2.resize(mask_uint8, (nw, nh), interpolation=cv2.INTER_NEAREST),
    )


def letterbox_image_and_mask(
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    out_w: int,
    out_h: int,
    fill_bgr: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Fit image inside out_w x out_h preserving aspect; pad with fill_bgr / 0 mask."""
    ih, iw = img_bgr.shape[:2]
    scale = min(out_w / float(iw), out_h / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    ri = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rm = cv2.resize(mask_uint8, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((out_h, out_w, 3), fill_bgr, dtype=np.uint8)
    mcanvas = np.zeros((out_h, out_w), dtype=np.uint8)
    dx = (out_w - nw) // 2
    dy = (out_h - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = ri
    mcanvas[dy : dy + nh, dx : dx + nw] = rm
    return canvas, mcanvas


def compute_orb_loss_teacher(
    composite_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    scales=(1.0, 0.9, 0.8, 0.7),
    nfeatures: int = 2000,
    # Backward-compatible args (kept but response is ignored in "count-only" mode).
    response_weight: float = 0.0,
    percentage_weight: float = 1.0,
) -> float:
    """
    Teacher orb score (count-only), comparable across scales as a *fraction*:
      score = mean_over_scales( percentage_weight * orb_ratio_s )
    Where for each multi-scale pyramid level s:
      orb_ratio = (# ORB keypoints with integer coords in mask==255) / (# keypoints)
    Note:
      We intentionally do NOT multiply orb_ratio by `scale`. That old weighting made
      the scalar target closer to a weighted sum than to an "in-mask fraction", which
      misleadingly matched neither (a) raw scale-1.0 ratio nor (b) a simple average
      of per-scale ratios — and ORBProxyCNN's sigmoid output was easy to misread as
      "percent of keypoints in mask" at full resolution.

      After changing this definition, retrain ORB proxy so its labels stay consistent.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    h, w = composite_bgr.shape[:2]
    if mask_uint8.shape[:2] != (h, w):
        raise ValueError("mask_uint8 shape mismatch with composite_bgr")

    scale_metrics = []
    for scale in scales:
        comp_s = cv2.resize(composite_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_LINEAR)
        mask_s = cv2.resize(mask_uint8, (comp_s.shape[1], comp_s.shape[0]), interpolation=cv2.INTER_NEAREST)

        kp = orb.detect(comp_s, None)
        if len(kp) == 0:
            metric = 0.0
        else:
            count_in_mask = 0
            hh, ww = mask_s.shape[:2]
            for point in kp:
                x = int(point.pt[0])
                y = int(point.pt[1])
                if 0 <= x < ww and 0 <= y < hh and mask_s[y, x] == 255:
                    count_in_mask += 1
            orb_ratio = count_in_mask / float(len(kp))
            metric = float(percentage_weight) * orb_ratio

        # Average in-mask fractions across scales (no extra `* scale` bias).
        scale_metrics.append(metric)

    return float(np.mean(scale_metrics))


class KittiOrbProxyDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str | None,
        attack_patch_paths: list[str],
        kitti_keep_classes: set[str] | None = None,
        kitti_occluded_allow: set[int] | None = None,
        kitti_truncated_max: float | None = None,
        kitti_skip_classes: set[str] | None = None,
        kitti_max_distance_m: float | None = None,
        kitti_distance_metric: str = "z_depth",
        composite_mode: str = "context",
        input_size: int = 224,
        context_scale: float = 1.5,
        full_canvas_width: int = 800,
        full_canvas_height: int = 400,
        full_max_side: int | None = None,
        aug: bool = True,
        max_samples: int | None = None,
        label_paths: list[str] | None = None,
        seed: int = 0,
        style_match_color: bool = True,
        style_blend_factor: float = 0.5,
        style_blur: bool = True,
        style_noise: bool = False,
        style_noise_std: float = 2.0,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.composite_mode = str(composite_mode).strip().lower()
        if self.composite_mode not in ("context", "full_scene"):
            raise ValueError("composite_mode must be 'context' or 'full_scene'")
        self.input_size = int(input_size)
        self.context_scale = float(context_scale)
        self.full_canvas_width = int(full_canvas_width)
        self.full_canvas_height = int(full_canvas_height)
        self.full_max_side = int(full_max_side) if full_max_side is not None else None
        self.aug = bool(aug)
        self.seed = int(seed)
        self.style_match_color = bool(style_match_color)
        self.style_blend_factor = float(style_blend_factor)
        self.style_blur = bool(style_blur)
        self.style_noise = bool(style_noise)
        self.style_noise_std = float(style_noise_std)

        self.kitti_keep_classes = kitti_keep_classes
        self.kitti_occluded_allow = kitti_occluded_allow
        self.kitti_truncated_max = kitti_truncated_max
        self.kitti_skip_classes = kitti_skip_classes if kitti_skip_classes is not None else {"DontCare"}
        self.kitti_max_distance_m = kitti_max_distance_m
        self.kitti_distance_metric = str(kitti_distance_metric).strip()

        if not attack_patch_paths:
            raise ValueError("attack_patch_paths must be a non-empty list of image files.")
        self.attack_patches_bgr: list[np.ndarray] = []
        for p in attack_patch_paths:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"Failed to read attack patch image: {p}")
            self.attack_patches_bgr.append(im)

        self.samples = []
        if label_paths is None:
            if not labels_dir:
                raise ValueError("Either label_paths or labels_dir must be provided.")
            label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

        # Note: we cap label_paths (image-level), not bbox-level, so each image still contributes all its bboxes.
        if max_samples is not None:
            label_paths = sorted(label_paths)[: int(max_samples)]

        for lab in label_paths:
            stem = os.path.splitext(os.path.basename(lab))[0]
            img_path = find_image_for_id(images_dir, stem)
            bboxes = parse_kitti_label_file(
                lab,
                keep_classes=self.kitti_keep_classes,
                occluded_allow=self.kitti_occluded_allow,
                truncated_max=self.kitti_truncated_max,
                skip_classes=self.kitti_skip_classes,
                max_distance_m=self.kitti_max_distance_m,
                distance_metric=self.kitti_distance_metric,
            )
            for bbox in bboxes:
                # store (img_path, bbox) tuples
                self.samples.append((img_path, bbox))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No bbox samples after filters. Check KITTI labels and YAML: "
                "keep_classes, occluded_allow, truncated_max, max_distance_m, skip_classes."
            )

    def __len__(self):
        return len(self.samples)

    def get_scene_bbox_patch_source(
        self, idx: int
    ) -> tuple[np.ndarray, BBox2D, np.ndarray, int, str]:
        """
        Raw inputs for differentiable fusion training (no paste / no ORB here).

        Returns:
          scene_bgr: HxWx3 uint8 BGR
          bbox: clamped to image
          patch_bgr: attack image uint8 BGR (same RNG choice as get_paste_visualization for this idx)
          patch_idx: index into attack_patches_bgr
          stem: image id (label stem)
        """
        img_path, bbox = self.samples[idx]
        scene = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if scene is None:
            raise FileNotFoundError(f"Failed to read image={img_path}")
        h, w = scene.shape[:2]
        bbox = bbox.clamp(w, h)
        jitter_rng = np.random.RandomState(self.seed + idx * 9973)
        patch_idx = int(jitter_rng.randint(0, len(self.attack_patches_bgr)))
        patch_bgr = self.attack_patches_bgr[patch_idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return scene, bbox, patch_bgr, patch_idx, stem

    def get_paste_visualization(self, idx: int) -> tuple[np.ndarray, np.ndarray, float, str]:
        """
        Build the same paste pipeline as training samples.
        Returns:
          composite_model: BGR uint8 — both modes end up full_canvas_width × full_canvas_height
          (context: context crop resized to that size; full_scene: letterbox to that canvas)
          mask_model: uint8 0/255 same spatial size
          orb_loss_teacher: float in [0,1] after clip (ORB on same tensor ORB sees)
          image_stem: KITTI image id for file naming
        """
        img_path, bbox = self.samples[idx]
        scene = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if scene is None:
            raise FileNotFoundError(f"Failed to read image={img_path}")
        h, w = scene.shape[:2]
        bbox = bbox.clamp(w, h)

        jitter_rng = np.random.RandomState(self.seed + idx * 9973)
        patch_idx = int(jitter_rng.randint(0, len(self.attack_patches_bgr)))
        patch_bgr = self.attack_patches_bgr[patch_idx]

        if self.composite_mode == "full_scene":
            composite_bgr, mask_u8 = paste_patch_into_full_scene(
                scene_bgr=scene,
                patch_bgr=patch_bgr,
                bbox=bbox,
                jitter_rng=jitter_rng,
                aug=self.aug,
                style_match_color=self.style_match_color,
                style_blend_factor=self.style_blend_factor,
                style_blur=self.style_blur,
                style_noise=self.style_noise,
                style_noise_std=self.style_noise_std,
            )
            if self.full_max_side is not None:
                composite_bgr, mask_u8 = scale_image_and_mask_max_side(
                    composite_bgr, mask_u8, self.full_max_side
                )
            composite_resized, mask_resized = letterbox_image_and_mask(
                composite_bgr,
                mask_u8,
                self.full_canvas_width,
                self.full_canvas_height,
            )
        else:
            cx = 0.5 * (bbox.x1 + bbox.x2)
            cy = 0.5 * (bbox.y1 + bbox.y2)
            bw = max(1.0, bbox.x2 - bbox.x1)
            bh = max(1.0, bbox.y2 - bbox.y1)
            new_w = bw * self.context_scale
            new_h = bh * self.context_scale
            context_bbox = BBox2D(
                x1=cx - new_w / 2.0,
                y1=cy - new_h / 2.0,
                x2=cx + new_w / 2.0,
                y2=cy + new_h / 2.0,
            )
            composite_context_bgr, mask_context = paste_patch_into_context(
                scene_bgr=scene,
                patch_bgr=patch_bgr,
                bbox=bbox,
                context_bbox=context_bbox,
                jitter_rng=jitter_rng,
                aug=self.aug,
                style_match_color=self.style_match_color,
                style_blend_factor=self.style_blend_factor,
                style_blur=self.style_blur,
                style_noise=self.style_noise,
                style_noise_std=self.style_noise_std,
            )
            composite_resized = cv2.resize(
                composite_context_bgr,
                (self.full_canvas_width, self.full_canvas_height),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_resized = cv2.resize(
                mask_context,
                (self.full_canvas_width, self.full_canvas_height),
                interpolation=cv2.INTER_NEAREST,
            )

        orb_raw = compute_orb_loss_teacher(composite_resized, mask_resized)
        orb_loss = float(np.clip(orb_raw, 0.0, 1.0))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return composite_resized, mask_resized, orb_loss, stem

    def __getitem__(self, idx: int):
        composite_resized, mask_resized, orb_loss, _stem = self.get_paste_visualization(idx)

        img = composite_resized.astype(np.float32) / 255.0
        mask = (mask_resized.astype(np.float32) / 255.0)[:, :, None]
        x = np.concatenate([img, mask], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y = torch.tensor([orb_loss], dtype=torch.float32)
        return x, y


def _collect_carla_scene_homography_pairs(
    carla_root_dir: str,
    carla_towns: list[str] | None = None,
    max_frames_per_town: int | None = None,
) -> list[tuple[str, str, str]]:
    """
    Returns list of (scene_img_path, homography_path, stem).
    Homography files are expected at: <carla_root_dir>/result/homography_<stem>.npy
    """
    if not os.path.isdir(carla_root_dir):
        raise FileNotFoundError(f"data.carla_root_dir not found: {carla_root_dir!r}")
    result_dir = os.path.join(carla_root_dir, "result")
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"data.carla_root_dir/result not found: {result_dir!r}")

    if carla_towns:
        town_dirs: list[str] = []
        for raw in carla_towns:
            s = str(raw).strip()
            if not s:
                continue
            # Support both:
            # 1) town name, e.g. "CARLA_Town01" (resolved under carla_root_dir)
            # 2) explicit town directory path (absolute or repo-relative already resolved by caller)
            if os.path.isabs(s):
                cand = s
            elif os.path.isdir(s):
                cand = s
            else:
                cand = os.path.join(carla_root_dir, s)
            if not os.path.isdir(cand):
                raise FileNotFoundError(
                    f"Invalid entry in data.carla_towns: {raw!r} -> resolved dir {cand!r} not found"
                )
            town_dirs.append(cand)
    else:
        town_dirs = [
            os.path.join(carla_root_dir, d)
            for d in sorted(os.listdir(carla_root_dir))
            if os.path.isdir(os.path.join(carla_root_dir, d)) and d.lower().startswith("carla_town")
        ]
    if not town_dirs:
        raise RuntimeError(f"No CARLA town folders found under {carla_root_dir!r}")

    pairs: list[tuple[str, str, str]] = []
    for town_dir in town_dirs:
        scene_paths = sorted(glob.glob(os.path.join(town_dir, "*.png")))
        if max_frames_per_town is not None:
            scene_paths = scene_paths[: int(max_frames_per_town)]
        for scene_path in scene_paths:
            stem = os.path.splitext(os.path.basename(scene_path))[0]
            h_path = os.path.join(result_dir, f"homography_{stem}.npy")
            if not os.path.isfile(h_path):
                continue
            pairs.append((scene_path, h_path, stem))
    if not pairs:
        raise RuntimeError(
            f"No (scene, homography) pairs found in carla_root_dir={carla_root_dir!r}; "
            "expected <town>/<stem>.png with result/homography_<stem>.npy"
        )
    return pairs


class CarlaOrbProxyDataset(Dataset):
    """
    CARLA-based ORB proxy dataset using perspective homography mapping patch->scene.
    """

    def __init__(
        self,
        scene_h_pairs: list[tuple[str, str, str]],
        attack_patch_paths: list[str],
        full_canvas_width: int = 800,
        full_canvas_height: int = 400,
        full_max_side: int | None = None,
        aug: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
        style_match_color: bool = True,
        style_blend_factor: float = 0.5,
        style_blur: bool = True,
        style_noise: bool = False,
        style_noise_std: float = 2.0,
        patch_width: int | None = None,
    ):
        if not attack_patch_paths:
            raise ValueError("attack_patch_paths must be a non-empty list of image files.")
        self.full_canvas_width = int(full_canvas_width)
        self.full_canvas_height = int(full_canvas_height)
        self.full_max_side = int(full_max_side) if full_max_side is not None else None
        self.aug = bool(aug)
        self.seed = int(seed)
        self.style_match_color = bool(style_match_color)
        self.style_blend_factor = float(style_blend_factor)
        self.style_blur = bool(style_blur)
        self.style_noise = bool(style_noise)
        self.style_noise_std = float(style_noise_std)
        self.patch_width = int(patch_width) if patch_width is not None else None

        pairs = list(scene_h_pairs)
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        self.samples = pairs
        if not self.samples:
            raise RuntimeError("No CARLA samples after filtering.")

        self.attack_patches_bgr: list[np.ndarray] = []
        self.attack_patch_src_paths: list[str] = []
        for p in attack_patch_paths:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"Failed to read attack patch image: {p}")
            if self.patch_width is not None and int(im.shape[1]) != self.patch_width:
                continue
            self.attack_patches_bgr.append(im)
            self.attack_patch_src_paths.append(p)
        if not self.attack_patches_bgr:
            raise RuntimeError(
                f"No attack patches left after width filter patch_width={self.patch_width}."
            )

    def __len__(self):
        return len(self.samples)

    def get_paste_visualization(self, idx: int) -> tuple[np.ndarray, np.ndarray, float, str]:
        scene_path, h_path, stem = self.samples[idx]
        scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)
        if scene is None:
            raise FileNotFoundError(f"Failed to read CARLA scene image: {scene_path}")
        H = np.load(h_path)
        if H.shape != (3, 3):
            raise ValueError(f"Expected homography shape (3,3), got {H.shape} from {h_path}")
        H = H.astype(np.float64)
        hh, ww = scene.shape[:2]

        jitter_rng = np.random.RandomState(self.seed + idx * 9973)
        patch_idx = int(jitter_rng.randint(0, len(self.attack_patches_bgr)))
        patch_bgr = self.attack_patches_bgr[patch_idx]

        # Optional patch-style adaptation before geometric warping.
        if self.style_match_color or self.style_blur or self.style_noise:
            ph, pw = patch_bgr.shape[:2]
            corners = np.array(
                [[0.0, 0.0, 1.0], [pw - 1.0, 0.0, 1.0], [0.0, ph - 1.0, 1.0], [pw - 1.0, ph - 1.0, 1.0]],
                dtype=np.float64,
            ).T
            proj = H @ corners
            u = proj[0] / np.maximum(proj[2], 1e-8)
            v = proj[1] / np.maximum(proj[2], 1e-8)
            x1 = int(np.floor(np.min(u)))
            y1 = int(np.floor(np.min(v)))
            x2 = int(np.ceil(np.max(u)))
            y2 = int(np.ceil(np.max(v)))
            x1 = max(0, min(x1, ww - 1))
            y1 = max(0, min(y1, hh - 1))
            x2 = max(x1 + 1, min(x2, ww))
            y2 = max(y1 + 1, min(y2, hh))
            roi = scene[y1:y2, x1:x2]
            if roi.size > 0:
                patch_bgr = apply_patch_style(
                    patch_bgr,
                    roi,
                    match_color=self.style_match_color,
                    blend_factor=self.style_blend_factor,
                    blur=self.style_blur,
                    noise=self.style_noise,
                    noise_std=self.style_noise_std,
                    noise_rng=jitter_rng,
                )

        # Warp patch and a binary mask into scene coordinates.
        warped_patch = cv2.warpPerspective(patch_bgr, H, (ww, hh), flags=cv2.INTER_LINEAR)
        patch_mask_src = np.full((patch_bgr.shape[0], patch_bgr.shape[1]), 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(patch_mask_src, H, (ww, hh), flags=cv2.INTER_NEAREST)
        warped_mask = (warped_mask > 0).astype(np.uint8) * 255

        composite_bgr = scene.copy()
        m = warped_mask > 0
        composite_bgr[m] = warped_patch[m]

        if self.full_max_side is not None:
            composite_bgr, warped_mask = scale_image_and_mask_max_side(
                composite_bgr, warped_mask, self.full_max_side
            )
        composite_resized, mask_resized = letterbox_image_and_mask(
            composite_bgr,
            warped_mask,
            self.full_canvas_width,
            self.full_canvas_height,
        )

        orb_raw = compute_orb_loss_teacher(composite_resized, mask_resized)
        orb_loss = float(np.clip(orb_raw, 0.0, 1.0))
        return composite_resized, mask_resized, orb_loss, stem

    def __getitem__(self, idx: int):
        composite_resized, mask_resized, orb_loss, _stem = self.get_paste_visualization(idx)
        img = composite_resized.astype(np.float32) / 255.0
        mask = (mask_resized.astype(np.float32) / 255.0)[:, :, None]
        x = np.concatenate([img, mask], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y = torch.tensor([orb_loss], dtype=torch.float32)
        return x, y


def _require_str(d: dict, section: str, key: str) -> str:
    v = d.get(key)
    if v is None or (isinstance(v, str) and not str(v).strip()):
        raise ValueError(f"Missing or empty required config: {section}.{key}")
    return str(v)


def _optional_int(x):
    if x is None:
        return None
    return int(x)


def _parse_kitti_keep_classes(raw) -> set[str] | None:
    """Empty / null = no class filter (except skip_classes)."""
    if raw is None:
        return None
    if isinstance(raw, list):
        s = {str(x).strip() for x in raw if str(x).strip()}
        return s if s else None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def _parse_kitti_occluded_allow(raw) -> set[int] | None:
    """
    KITTI occluded: 0 visible, 1 partly, 2 largely, 3 unknown.
    null / omit = no occlusion filter. Empty list = no filter (same as omit).
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("data.occluded_allow must be a list of integers or null")
    if len(raw) == 0:
        return None
    return {int(x) for x in raw}


def _parse_kitti_truncated_max(raw) -> float | None:
    """null / omit = no truncated filter. 0.0 = only non-truncated objects."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("", "null", "none"):
        return None
    return float(raw)


def _resolve_optional_path(raw, repo_root: str) -> str | None:
    """None / empty / null string -> None. Relative paths joined to repo_root."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s if os.path.isabs(s) else os.path.join(repo_root, s)


def append_epoch_metrics_csv(
    path: str,
    row: dict,
    fieldnames: list[str],
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _csv_float(x: float) -> str:
    if x != x:  # NaN
        return ""
    return f"{x:.8f}"


def overlay_mask_on_bgr(
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    bgr: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Semi-transparent mask overlay for quick inspection."""
    out = img_bgr.copy()
    m = mask_uint8 >= 128
    if not np.any(m):
        return out
    blend = out.astype(np.float32)
    col = np.array(bgr, dtype=np.float32)
    blend[m] = blend[m] * (1.0 - alpha) + col * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def save_paste_visualization_epoch(
    split_datasets: list[tuple[str, KittiOrbProxyDataset]],
    epoch: int,
    out_root: str,
    max_samples: int,
) -> None:
    """Save composite + mask-overlay PNGs for the first `max_samples` indices per split."""
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    for split_name, ds in split_datasets:
        n = min(max_samples, len(ds))
        for i in range(n):
            comp, mask, orb_l, stem = ds.get_paste_visualization(i)
            orb_tag = int(round(float(orb_l) * 10000))
            base = f"{split_name}_i{i:04d}_{stem}_orb{orb_tag:05d}"
            cv2.imwrite(os.path.join(ep_dir, f"{base}_composite.png"), comp)
            cv2.imwrite(
                os.path.join(ep_dir, f"{base}_overlay.png"),
                overlay_mask_on_bgr(comp, mask),
            )


def _parse_kitti_skip_classes(raw) -> set[str]:
    if raw is None:
        return {"DontCare"}
    if isinstance(raw, list):
        s = {str(x).strip() for x in raw if str(x).strip()}
        return s if s else {"DontCare"}
    s = str(raw).strip()
    if not s:
        return {"DontCare"}
    return {x.strip() for x in s.split(",") if x.strip()}


def _repo_root_from_config(config_path: str) -> str:
    """Assume config lives in <repo>/configs/*.yaml."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), ".."))


def _resolve_data_path(p: str, repo_root: str) -> str:
    p = str(p).strip()
    if not p or p.lower() in ("null", "none"):
        return ""
    return p if os.path.isabs(p) else os.path.join(repo_root, p)


def _attack_patch_dir_list(patch_dir) -> list[str]:
    """Normalize YAML attack_patch_dir: one string or a list/tuple of strings."""
    if patch_dir is None:
        return []
    if isinstance(patch_dir, (list, tuple)):
        out: list[str] = []
        for x in patch_dir:
            if x is None:
                continue
            s = str(x).strip()
            if not s or s.lower() in ("null", "none"):
                continue
            out.append(s)
        return out
    s = str(patch_dir).strip()
    if not s or s.lower() in ("null", "none"):
        return []
    return [s]


def collect_attack_patch_paths(data: dict, repo_root: str) -> list[str]:
    """
    Either data.attack_patch_dir (one folder or list of folders: all common image extensions)
    or data.attack_patch_path (single file). Relative paths are resolved against repo_root.
    """
    patch_dirs = _attack_patch_dir_list(data.get("attack_patch_dir"))
    has_dir = bool(patch_dirs)

    single = data.get("attack_patch_path")
    has_file = single is not None and str(single).strip() and str(single).lower() not in ("null", "none")

    if has_dir:
        paths: list[str] = []
        resolved_dirs: list[str] = []
        for entry in patch_dirs:
            d = _resolve_data_path(entry, repo_root)
            if not d or not os.path.isdir(d):
                raise FileNotFoundError(f"data.attack_patch_dir entry is not a directory: {d!r}")
            resolved_dirs.append(d)
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                paths.extend(glob.glob(os.path.join(d, ext)))
        paths = sorted(set(paths))
        if not paths:
            raise ValueError(
                f"No images (*.png/*.jpg) under data.attack_patch_dir={resolved_dirs}"
            )
        return paths

    if has_file:
        p = _resolve_data_path(str(single), repo_root)
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f"data.attack_patch_path not found: {p!r}")
        return [p]

    raise ValueError(
        "Set data.attack_patch_dir (folder or list of folders of ad images) or "
        "data.attack_patch_path (single file) in YAML."
    )


def load_train_config(config_path: str) -> SimpleNamespace:
    """
    Load training + patch_style + data settings from YAML.
    Schema: see configs/orb_proxy_train.yaml
    """
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")

    data = raw.get("data") or {}
    patch_style = raw.get("patch_style") or {}
    training = raw.get("training") or {}

    repo_root = _repo_root_from_config(config_path)

    data_mode = str(data.get("mode", "kitti")).strip().lower()
    if data_mode not in ("kitti", "carla"):
        raise ValueError("data.mode must be one of: kitti, carla")

    images_dir = _resolve_data_path(str(data.get("images_dir", "")), repo_root)
    labels_dir = _resolve_data_path(str(data.get("labels_dir", "")), repo_root)
    carla_root_dir = _resolve_data_path(str(data.get("carla_root_dir", "")), repo_root)

    if data_mode == "kitti":
        if not images_dir:
            images_dir = _require_str(data, "data", "images_dir")
            images_dir = _resolve_data_path(images_dir, repo_root) or images_dir
        if not labels_dir:
            labels_dir = _require_str(data, "data", "labels_dir")
            labels_dir = _resolve_data_path(labels_dir, repo_root) or labels_dir
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"data.images_dir is not a directory: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"data.labels_dir is not a directory: {labels_dir}")
    else:
        if not carla_root_dir:
            carla_root_dir = _require_str(data, "data", "carla_root_dir")
            carla_root_dir = _resolve_data_path(carla_root_dir, repo_root) or carla_root_dir
        if not os.path.isdir(carla_root_dir):
            raise FileNotFoundError(f"data.carla_root_dir is not a directory: {carla_root_dir}")

    attack_patch_paths = collect_attack_patch_paths(data, repo_root)

    keep_raw = data.get("keep_classes", "")
    if keep_raw is None:
        keep_str = ""
    elif isinstance(keep_raw, list):
        keep_str = ",".join(str(x) for x in keep_raw)
    else:
        keep_str = str(keep_raw)

    kitti_keep_classes = _parse_kitti_keep_classes(keep_raw)
    kitti_occluded_allow = _parse_kitti_occluded_allow(data.get("occluded_allow"))
    kitti_truncated_max = _parse_kitti_truncated_max(data.get("truncated_max"))
    kitti_skip_classes = _parse_kitti_skip_classes(data.get("skip_classes"))

    md_raw = data.get("max_distance_m")
    if md_raw is None or (isinstance(md_raw, str) and md_raw.strip().lower() in ("", "null", "none")):
        kitti_max_distance_m = None
    else:
        kitti_max_distance_m = float(md_raw)

    kitti_distance_metric = str(data.get("distance_metric", "z_depth")).strip()
    try:
        kitti_object_distance_m(1.0, 2.0, 3.0, kitti_distance_metric)
    except ValueError as e:
        raise ValueError(f"data.distance_metric: {e}") from e

    composite_mode = str(training.get("composite_mode", "context")).strip().lower()
    if composite_mode not in ("context", "full_scene"):
        raise ValueError("training.composite_mode must be 'context' or 'full_scene'")

    fms = training.get("full_max_side")
    if fms is None or (isinstance(fms, str) and fms.strip().lower() in ("", "null", "none")):
        full_max_side_ns = None
    else:
        full_max_side_ns = int(fms)

    vis = raw.get("visualization") or {}
    vis_split = str(vis.get("split", "train")).strip().lower()
    if vis_split not in ("train", "val", "both"):
        raise ValueError("visualization.split must be one of: train, val, both")

    ns = SimpleNamespace(
        # data
        data_mode=data_mode,
        images_dir=images_dir,
        labels_dir=labels_dir,
        carla_root_dir=carla_root_dir,
        carla_towns=data.get("carla_towns"),
        max_frames_per_town=_optional_int(data.get("max_frames_per_town")),
        carla_patch_width=_optional_int(data.get("carla_patch_width")),
        attack_patch_paths=attack_patch_paths,
        attack_patch_dir=data.get("attack_patch_dir"),
        attack_patch_path=data.get("attack_patch_path"),
        repo_root=repo_root,
        keep_classes=keep_str,
        kitti_keep_classes=kitti_keep_classes,
        kitti_occluded_allow=kitti_occluded_allow,
        kitti_truncated_max=kitti_truncated_max,
        kitti_skip_classes=kitti_skip_classes,
        kitti_max_distance_m=kitti_max_distance_m,
        kitti_distance_metric=kitti_distance_metric,
        # patch_style (explicit flags for train loop)
        patch_style_enabled=bool(patch_style.get("enabled", True)),
        style_match_color=bool(patch_style.get("match_color", True)),
        style_blur=bool(patch_style.get("blur", True)),
        style_noise=bool(patch_style.get("noise", False)),
        style_blend_factor=float(patch_style.get("blend_factor", 0.5)),
        style_noise_std=float(patch_style.get("noise_std", 2.0)),
        # training
        composite_mode=composite_mode,
        input_size=int(training.get("input_size", 224)),
        context_scale=float(training.get("context_scale", 1.5)),
        full_canvas_width=int(training.get("full_canvas_width", 800)),
        full_canvas_height=int(training.get("full_canvas_height", 400)),
        full_max_side=full_max_side_ns,
        val_ratio=float(training.get("val_ratio", 0.1)),
        batch_size=int(training.get("batch_size", 8)),
        lr=float(training.get("lr", 1e-3)),
        epochs=int(training.get("epochs", 5)),
        device=str(training.get("device", "cuda")),
        num_workers=int(training.get("num_workers", 2)),
        seed=int(training.get("seed", 0)),
        max_train_samples=_optional_int(training.get("max_train_samples")),
        max_val_samples=_optional_int(training.get("max_val_samples")),
        out_dir=str(training.get("out_dir", "orb_proxy_ckpt")),
        log_csv=_resolve_optional_path(training.get("log_csv"), repo_root),
        # visualization (paste preview PNGs)
        vis_enabled=bool(vis.get("enabled", False)),
        vis_dir=_resolve_optional_path(vis.get("dir"), repo_root),
        vis_every_epochs=max(1, int(vis.get("every_epochs", 1))),
        vis_max_samples=max(1, int(vis.get("max_samples_per_epoch", 8))),
        vis_split=vis_split,
        # meta (for checkpoints / reproducibility)
        config_path=config_path,
    )

    return ns


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_orb_proxy(in_channels=4)
    model.to(device)

    if args.data_mode == "kitti":
        print(
            "[KITTI filters] keep_classes="
            f"{sorted(args.kitti_keep_classes) if args.kitti_keep_classes else 'ALL'}, "
            f"occluded_allow={sorted(args.kitti_occluded_allow) if args.kitti_occluded_allow else 'ANY'}, "
            f"truncated_max={args.kitti_truncated_max}, "
            f"max_distance_m={args.kitti_max_distance_m} (metric={args.kitti_distance_metric}), "
            f"skip_classes={sorted(args.kitti_skip_classes)}"
        )
        if args.composite_mode == "full_scene":
            print(
                "[composite] full_scene: paste on full image -> "
                f"optional max_side={args.full_max_side} -> "
                f"letterbox {args.full_canvas_width}x{args.full_canvas_height} "
                "(teacher ORB + CNN use this tensor)"
            )
        else:
            print(
                "[composite] context: crop around bbox (context_scale="
                f"{args.context_scale}) -> resize {args.full_canvas_width}x{args.full_canvas_height}"
            )
    else:
        print(
            f"[CARLA] root={args.carla_root_dir} towns={args.carla_towns if args.carla_towns else 'ALL'} "
            f"max_frames_per_town={args.max_frames_per_town} patch_width={args.carla_patch_width}"
        )
        print(
            "[composite] carla homography: warp patch->scene -> "
            f"optional max_side={args.full_max_side} -> "
            f"letterbox {args.full_canvas_width}x{args.full_canvas_height}"
        )

    rng = np.random.RandomState(args.seed)

    use_patch_style = args.patch_style_enabled
    style_match_color = use_patch_style and args.style_match_color
    style_blur = use_patch_style and args.style_blur
    style_noise = use_patch_style and args.style_noise

    if args.data_mode == "kitti":
        # Random split at the image (label file stem) level.
        all_label_paths = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
        if len(all_label_paths) == 0:
            raise RuntimeError(f"No label txt files found in labels_dir={args.labels_dir}")
        stems = [os.path.splitext(os.path.basename(p))[0] for p in all_label_paths]
        rng.shuffle(stems)
        n_train = int(round(len(stems) * (1.0 - args.val_ratio)))
        n_train = max(1, min(n_train, len(stems) - 1))
        train_stems = stems[:n_train]
        val_stems = stems[n_train:]
        train_label_paths = [os.path.join(args.labels_dir, f"{s}.txt") for s in train_stems]
        val_label_paths = [os.path.join(args.labels_dir, f"{s}.txt") for s in val_stems]

        train_ds = KittiOrbProxyDataset(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            label_paths=train_label_paths,
            attack_patch_paths=args.attack_patch_paths,
            kitti_keep_classes=args.kitti_keep_classes,
            kitti_occluded_allow=args.kitti_occluded_allow,
            kitti_truncated_max=args.kitti_truncated_max,
            kitti_skip_classes=args.kitti_skip_classes,
            kitti_max_distance_m=args.kitti_max_distance_m,
            kitti_distance_metric=args.kitti_distance_metric,
            composite_mode=args.composite_mode,
            input_size=args.input_size,
            context_scale=args.context_scale,
            full_canvas_width=args.full_canvas_width,
            full_canvas_height=args.full_canvas_height,
            full_max_side=args.full_max_side,
            aug=True,
            max_samples=args.max_train_samples,
            seed=args.seed,
            style_match_color=style_match_color,
            style_blend_factor=args.style_blend_factor,
            style_blur=style_blur,
            style_noise=style_noise,
            style_noise_std=args.style_noise_std,
        )
        val_ds = KittiOrbProxyDataset(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            label_paths=val_label_paths,
            attack_patch_paths=args.attack_patch_paths,
            kitti_keep_classes=args.kitti_keep_classes,
            kitti_occluded_allow=args.kitti_occluded_allow,
            kitti_truncated_max=args.kitti_truncated_max,
            kitti_skip_classes=args.kitti_skip_classes,
            kitti_max_distance_m=args.kitti_max_distance_m,
            kitti_distance_metric=args.kitti_distance_metric,
            composite_mode=args.composite_mode,
            input_size=args.input_size,
            context_scale=args.context_scale,
            full_canvas_width=args.full_canvas_width,
            full_canvas_height=args.full_canvas_height,
            full_max_side=args.full_max_side,
            aug=False,
            max_samples=args.max_val_samples,
            seed=args.seed + 123,
            style_match_color=style_match_color,
            style_blend_factor=args.style_blend_factor,
            style_blur=style_blur,
            style_noise=style_noise,
            style_noise_std=args.style_noise_std,
        )
    else:
        towns_raw = args.carla_towns
        if towns_raw is None:
            carla_towns = None
        elif isinstance(towns_raw, list):
            carla_towns = [str(x).strip() for x in towns_raw if str(x).strip()]
        else:
            carla_towns = [x.strip() for x in str(towns_raw).split(",") if x.strip()]
        all_pairs = _collect_carla_scene_homography_pairs(
            args.carla_root_dir,
            carla_towns=carla_towns,
            max_frames_per_town=args.max_frames_per_town,
        )
        rng.shuffle(all_pairs)
        n_train = int(round(len(all_pairs) * (1.0 - args.val_ratio)))
        n_train = max(1, min(n_train, len(all_pairs) - 1))
        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train:]

        train_ds = CarlaOrbProxyDataset(
            scene_h_pairs=train_pairs,
            attack_patch_paths=args.attack_patch_paths,
            full_canvas_width=args.full_canvas_width,
            full_canvas_height=args.full_canvas_height,
            full_max_side=args.full_max_side,
            aug=True,
            max_samples=args.max_train_samples,
            seed=args.seed,
            style_match_color=style_match_color,
            style_blend_factor=args.style_blend_factor,
            style_blur=style_blur,
            style_noise=style_noise,
            style_noise_std=args.style_noise_std,
            patch_width=args.carla_patch_width,
        )
        val_ds = CarlaOrbProxyDataset(
            scene_h_pairs=val_pairs,
            attack_patch_paths=args.attack_patch_paths,
            full_canvas_width=args.full_canvas_width,
            full_canvas_height=args.full_canvas_height,
            full_max_side=args.full_max_side,
            aug=False,
            max_samples=args.max_val_samples,
            seed=args.seed + 123,
            style_match_color=style_match_color,
            style_blend_factor=args.style_blend_factor,
            style_blur=style_blur,
            style_noise=style_noise,
            style_noise_std=args.style_noise_std,
            patch_width=args.carla_patch_width,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(
        f"[data] train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"batch_size={args.batch_size} train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    def _vis_output_root() -> str:
        if args.vis_dir:
            p = args.vis_dir
            return p if os.path.isabs(p) else os.path.join(args.repo_root, p)
        od = args.out_dir
        od_abs = od if os.path.isabs(od) else os.path.join(args.repo_root, od)
        return os.path.join(od_abs, "paste_vis")

    if args.vis_enabled:
        vr = _vis_output_root()
        print(
            f"[vis] enabled split={args.vis_split} every_epochs={args.vis_every_epochs} "
            f"max_samples={args.vis_max_samples} -> {os.path.abspath(vr)}"
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    metrics_csv_path = args.log_csv
    if not metrics_csv_path:
        os.makedirs(args.out_dir, exist_ok=True)
        metrics_csv_path = os.path.join(args.out_dir, "orb_proxy_train_metrics.csv")
    else:
        parent = os.path.dirname(os.path.abspath(metrics_csv_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    metrics_fields = [
        "epoch",
        "epochs",
        "lr",
        "train_mse",
        "train_mae",
        "train_rmse",
        "val_mse",
        "val_mae",
        "val_rmse",
        "val_pearson_r",
        "val_pct_err_lt_0.05",
        "val_pct_err_lt_0.10",
        "is_new_best",
        "best_val_mse",
    ]
    print(f"[log] metrics CSV -> {os.path.abspath(metrics_csv_path)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_sse = 0.0
        tr_sae = 0.0
        tr_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)

            pred = model(x).view(-1)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            with torch.no_grad():
                diff = pred - y
                tr_sse += float((diff * diff).sum().item())
                tr_sae += float(diff.abs().sum().item())
            tr_n += bs
            with torch.no_grad():
                batch_mae = float(diff.abs().mean().item())
            pbar.set_postfix(mse=f"{loss.item():.5f}", mae=f"{batch_mae:.5f}")

        train_mse = tr_sse / max(1, tr_n)
        train_mae = tr_sae / max(1, tr_n)
        train_rmse = math.sqrt(train_mse)

        model.eval()
        val_sse = 0.0
        val_sae = 0.0
        val_n = 0
        val_preds: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).view(-1)
                pred = model(x).view(-1)
                diff = pred - y
                bs = x.size(0)
                val_sse += float((diff * diff).sum().item())
                val_sae += float(diff.abs().sum().item())
                val_n += bs
                val_preds.append(pred.detach().float().cpu().numpy())
                val_targets.append(y.detach().float().cpu().numpy())

        val_mse = val_sse / max(1, val_n)
        val_mae = val_sae / max(1, val_n)
        val_rmse = math.sqrt(val_mse)

        if val_n > 0:
            p_all = np.concatenate(val_preds)
            t_all = np.concatenate(val_targets)
            if len(t_all) > 2 and np.std(p_all) > 1e-12 and np.std(t_all) > 1e-12:
                val_r = float(np.corrcoef(p_all, t_all)[0, 1])
            else:
                val_r = float("nan")
            val_acc_005 = float((np.abs(p_all - t_all) < 0.05).mean())
            val_acc_01 = float((np.abs(p_all - t_all) < 0.10).mean())
        else:
            val_r = float("nan")
            val_acc_005 = float("nan")
            val_acc_01 = float("nan")

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train loss_mse={train_mse:.6f} mae={train_mae:.6f} rmse={train_rmse:.6f} | "
            f"val   loss_mse={val_mse:.6f} mae={val_mae:.6f} rmse={val_rmse:.6f} "
            f"pearson_r={val_r:.4f} "
            f"pct_|err|<0.05={val_acc_005 * 100:.2f}% pct_|err|<0.10={val_acc_01 * 100:.2f}%"
        )

        prev_best = best_val
        is_new_best = bool(val_mse < prev_best)
        if is_new_best:
            best_val = val_mse

        append_epoch_metrics_csv(
            metrics_csv_path,
            {
                "epoch": epoch,
                "epochs": args.epochs,
                "lr": args.lr,
                "train_mse": _csv_float(train_mse),
                "train_mae": _csv_float(train_mae),
                "train_rmse": _csv_float(train_rmse),
                "val_mse": _csv_float(val_mse),
                "val_mae": _csv_float(val_mae),
                "val_rmse": _csv_float(val_rmse),
                "val_pearson_r": _csv_float(val_r),
                "val_pct_err_lt_0.05": f"{val_acc_005 * 100:.4f}" if val_acc_005 == val_acc_005 else "",
                "val_pct_err_lt_0.10": f"{val_acc_01 * 100:.4f}" if val_acc_01 == val_acc_01 else "",
                "is_new_best": int(is_new_best),
                "best_val_mse": _csv_float(best_val),
            },
            metrics_fields,
        )

        if args.vis_enabled and (epoch - 1) % args.vis_every_epochs == 0:
            vis_pairs: list[tuple[str, KittiOrbProxyDataset]] = []
            if args.vis_split in ("train", "both"):
                vis_pairs.append(("train", train_ds))
            if args.vis_split in ("val", "both"):
                vis_pairs.append(("val", val_ds))
            vroot = _vis_output_root()
            save_paste_visualization_epoch(vis_pairs, epoch, vroot, args.vis_max_samples)
            print(f"[vis] epoch {epoch}: wrote previews under {os.path.abspath(vroot)}/epoch_{epoch:03d}/")

        if is_new_best:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, "orb_proxy_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_mse": best_val,
                    "args": vars(args),
                    "config_path": getattr(args, "config_path", None),
                },
                ckpt_path,
            )
            print(f"[save] best ckpt: {ckpt_path}")


def main():
    _root = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.normpath(os.path.join(_root, "..", "configs", "orb_proxy_train.yaml"))

    ap = argparse.ArgumentParser(description="Train ORB proxy from YAML config (see configs/orb_proxy_train.yaml).")
    ap.add_argument(
        "--config",
        type=str,
        default=default_cfg,
        help=f"Path to YAML config (default: {default_cfg})",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training.device from YAML (e.g. cuda / cpu).",
    )
    parsed = ap.parse_args()

    args = load_train_config(parsed.config)
    if parsed.device is not None:
        args.device = parsed.device
    train(args)


if __name__ == "__main__":
    main()


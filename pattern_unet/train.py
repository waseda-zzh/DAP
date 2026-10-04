from __future__ import annotations

import argparse
import os
from typing import Any

import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from pattern_unet.generator import build_wide_shallow_pattern_unet, rgb_to_grayscale
from pattern_unet.losses import (
    ContentStyleLossWeights,
    FFTGridSpectralLoss,
    TotalVariationLoss,
    VGG16ContentStyleMSELoss,
)
from pattern_unet.dataset import (
    PatternAdDataset,
    PatternAdDatasetConfig,
    blend_output_with_fg_mask,
)


def _project_root() -> str:
    """
    DAP repository root (parent of the `pattern_unet` package).

    Relative paths in YAML (e.g. data/ad_images_dir) are resolved against this.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_repo_path(path: str, repo_root: str) -> str:
    p = str(path).strip()
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(repo_root, p))


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_pil_rgb_01(x_chw: torch.Tensor):
    x = x_chw.detach().float().clamp(0.0, 1.0).cpu()
    x_hw3 = x.permute(1, 2, 0).numpy()
    import numpy as np
    from PIL import Image

    arr = (x_hw3 * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def save_debug_images(
    *,
    out_dir: str,
    epoch: int,
    ad_t: torch.Tensor,  # (B,3,H,W)
    out_t: torch.Tensor,  # (B,3,H,W)
    pattern_t: torch.Tensor,  # (B,3,H,W)
    orig_hw: torch.Tensor,  # (B,2) int32 [orig_h, orig_w]
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    b = min(1, ad_t.size(0))
    ad = ad_t[:b][0]
    out = out_t[:b][0]
    pat = pattern_t[:b][0]
    orig_h = int(orig_hw[:b][0, 0].item())
    orig_w = int(orig_hw[:b][0, 1].item())

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # tensor_to_pil_rgb_01(ad).save(os.path.join(img_dir, f"epoch_{epoch:04d}_ad.png"))
    # Resize output to match original ad resolution (visualization-only post-process).
    out_resized = F.interpolate(
        out.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    tensor_to_pil_rgb_01(out_resized).save(os.path.join(img_dir, f"epoch_{epoch:04d}_out.png"))
    # tensor_to_pil_rgb_01(pat).save(os.path.join(img_dir, f"epoch_{epoch:04d}_pattern.png"))

    # High-frequency visualization (rough grid proxy).
    out_gray = rgb_to_grayscale(out_resized.unsqueeze(0)).squeeze(0)  # (1,orig_h,orig_w)
    # Estimate "low" by downsample+upsample (cheap and stable).
    h, w = out_gray.shape[-2], out_gray.shape[-1]
    pool = max(2, min(32, h // 8))
    low = F.avg_pool2d(out_gray.unsqueeze(0), kernel_size=pool, stride=pool)
    low_up = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    high = (out_gray - low_up).abs()
    high = high / (high.max().clamp_min(1e-8))
    # Save high as grayscale RGB for convenience.
    high_rgb = high.repeat(3, 1, 1)
    # tensor_to_pil_rgb_01(high_rgb).save(os.path.join(img_dir, f"epoch_{epoch:04d}_high.png"))


@torch.no_grad()
def save_loss_curves_train_only(
    *,
    out_dir: str,
    step_points: list[int],
    train_loss_total: list[float],
    train_loss_content: list[float],
    train_loss_style: list[float],
    train_loss_fft: list[float],
    train_loss_tv: list[float],
    max_points: int,
) -> None:
    """
    Train-only loss curve snapshot.
    Intended to be called every N steps to avoid waiting for full epochs.
    """
    if not step_points:
        return

    # Keep the most recent points to reduce matplotlib work (if max_points>0).
    # If max_points<=0, keep all points to draw from head to tail.
    if max_points > 0 and len(step_points) > max_points:
        step_points = step_points[-max_points:]
        train_loss_total = train_loss_total[-max_points:]
        train_loss_content = train_loss_content[-max_points:]
        train_loss_style = train_loss_style[-max_points:]
        train_loss_fft = train_loss_fft[-max_points:]
        train_loss_tv = train_loss_tv[-max_points:]

    try:
        # Always persist train loss history to CSV (so you can plot later).
        import csv

        csv_path = os.path.join(out_dir, "loss_curves_train.csv")
        rows = zip(
            step_points,
            train_loss_total,
            train_loss_content,
            train_loss_style,
            train_loss_fft,
            train_loss_tv,
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "global_step",
                    "loss_total",
                    "loss_content",
                    "loss_style",
                    "loss_fft",
                    "loss_tv",
                ]
            )
            for r in rows:
                w.writerow(r)

        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        x = step_points
        plt.figure(figsize=(10, 6))
        plt.plot(x, train_loss_total, label="train_total", linewidth=2)

        plt.yscale("log")
        plt.xlabel("global_step")
        plt.ylabel("loss (log scale)")
        plt.title("Train Total Loss Curves")
        plt.grid(True, which="both", linestyle="--", alpha=0.35)
        plt.legend()

        plot_path = os.path.join(out_dir, "loss_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except ModuleNotFoundError:
        # matplotlib missing: CSV is already written above.
        csv_path = os.path.join(out_dir, "loss_curves_train.csv")
        print(f"[vis] matplotlib not found; wrote {os.path.abspath(csv_path)}")
    except Exception as e:  # pragma: no cover
        print(f"[vis] failed to save train loss curves: {e}")


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    import numpy as np

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def load_config(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Train WideShallowPatternUNet with VGG content/style + FFT grid loss.")
    ap.add_argument("--config", type=str, required=True, help="Path to pattern_unet_train.yaml")
    ap.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda or cpu)")
    parsed = ap.parse_args()

    cfg = load_config(parsed.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    repo_root = _project_root()

    data_cfg = cfg.get("data") or {}
    loss_cfg = cfg.get("loss") or {}
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("training") or {}
    optim_cfg = cfg.get("optim") or {}
    vis_cfg = cfg.get("visualization") or {}

    ad_images_dir = _resolve_repo_path(str(data_cfg.get("ad_images_dir", "")).strip(), repo_root)
    error_pattern_path = _resolve_repo_path(str(data_cfg.get("error_pattern_path", "")).strip(), repo_root)
    if not ad_images_dir:
        raise ValueError("data.ad_images_dir is required")
    if not error_pattern_path:
        raise ValueError("data.error_pattern_path is required")

    fg_mask_enabled = bool(data_cfg.get("fg_mask_enabled", False))
    _fmd = data_cfg.get("fg_mask_dir")
    if _fmd is not None and str(_fmd).strip():
        fg_mask_dir = _resolve_repo_path(str(_fmd).strip(), repo_root)
    else:
        fg_mask_dir = None

    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(optim_cfg.get("lr", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    num_workers = int(train_cfg.get("num_workers", 0))

    device_str = parsed.device if parsed.device is not None else str(train_cfg.get("device", "cuda"))
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[setup] device={device}")
    if fg_mask_enabled:
        print(
            f"[setup] fg_mask_enabled=True dir={fg_mask_dir!r} "
            f"suffix={data_cfg.get('fg_mask_suffix', '_mask.png')!r}"
        )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = str(train_cfg.get("out_dir", "pattern_unet_ckpt"))
    os.makedirs(out_dir, exist_ok=True)
    save_every_epochs = int(vis_cfg.get("save_every_epochs", 100))
    debug_max_batch = int(vis_cfg.get("debug_max_batch", 1))
    loss_plot_every_epochs = int(vis_cfg.get("loss_plot_every_epochs", 0))
    loss_plot_every_steps = int(vis_cfg.get("loss_plot_every_steps", 500))
    loss_plot_max_points = int(vis_cfg.get("loss_plot_max_points", 2000))

    # Dataset + split
    ds = PatternAdDataset(
        PatternAdDatasetConfig(
            ad_images_dir=ad_images_dir,
            error_pattern_path=error_pattern_path,
            image_size=image_size,
            exts=tuple(data_cfg.get("ad_exts", ["png", "jpg", "jpeg"])),
            transparent_bg_rgb=tuple(data_cfg.get("transparent_bg_rgb", [255, 255, 255])),
            fg_mask_enabled=fg_mask_enabled,
            fg_mask_dir=fg_mask_dir,
            fg_mask_suffix=str(data_cfg.get("fg_mask_suffix", "_mask.png")),
            fg_mask_invert=bool(data_cfg.get("fg_mask_invert", False)),
            fg_mask_default_all_background=bool(
                data_cfg.get("fg_mask_default_all_background", True)
            ),
        )
    )
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    train_idx, val_idx = _split_indices(len(ds), val_ratio=val_ratio, seed=seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    print(f"[data] total={len(ds)} train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "train DataLoader is empty (len(train_ds) < batch_size with drop_last=True). "
            "Lower training.batch_size or add more images."
        )
    if len(val_loader) == 0:
        raise RuntimeError("val DataLoader is empty; check val_ratio and dataset size.")

    # Model
    model = build_wide_shallow_pattern_unet(
        base_channels=int(model_cfg.get("base_channels", 96)),
        max_channels=int(model_cfg.get("max_channels", 512)),
        num_down=int(model_cfg.get("num_down", 3)),
        convs_per_stage=int(model_cfg.get("convs_per_stage", 2)),
        use_fg_mask=fg_mask_enabled,
    ).to(device)

    # Losses
    content_layers = tuple(loss_cfg.get("vgg_content_layers", ["relu3_3"]))
    style_layers = tuple(loss_cfg.get("vgg_style_layers", ["relu2_2", "relu3_3"]))
    vgg_loss = VGG16ContentStyleMSELoss(
        ContentStyleLossWeights(
            content_weight=float(loss_cfg.get("w_content", 1.0)),
            style_weight=float(loss_cfg.get("w_style", 0.2)),
            content_layers=content_layers,
            style_layers=style_layers,
            style_use_grayscale=bool(loss_cfg.get("style_use_grayscale", True)),
        )
    ).to(device)

    fft_loss = FFTGridSpectralLoss(
        top_quantile=float(loss_cfg.get("fft_top_quantile", 0.995)),
        use_log1p=bool(loss_cfg.get("fft_use_log1p", True)),
        normalize_spectra=bool(loss_cfg.get("fft_normalize_spectra", True)),
    ).to(device)

    tv_loss = TotalVariationLoss().to(device)

    w_fft = float(loss_cfg.get("w_fft", 0.05))
    w_tv = float(loss_cfg.get("w_tv", 1e-6))
    fft_warmup_epochs = float(loss_cfg.get("fft_warmup_epochs", 3.0))

    opt_name = str(optim_cfg.get("name", "adamw")).lower()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optim.name must be adamw or adam")

    # Training
    best_val = float("inf")
    global_step = 0
    steps_per_epoch = max(1, len(train_loader))

    history: dict[str, list[float]] = {
        "train_loss_total": [],
        "train_loss_content": [],
        "train_loss_style": [],
        "train_loss_fft": [],
        "train_loss_tv": [],
        "val_loss_total": [],
        "val_loss_content": [],
        "val_loss_style": [],
        "val_loss_fft": [],
        "val_loss_tv": [],
    }
    # Step-level history for faster visual feedback.
    step_points: list[int] = []
    step_train_loss_total: list[float] = []
    step_train_loss_content: list[float] = []
    step_train_loss_style: list[float] = []
    step_train_loss_fft: list[float] = []
    step_train_loss_tv: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            "loss_total": 0.0,
            "loss_content": 0.0,
            "loss_style": 0.0,
            "loss_fft": 0.0,
            "loss_tv": 0.0,
        }
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for ad, pattern, fg_mask, meta_hw in pbar:
            ad = ad.to(device, non_blocking=True)
            pattern = pattern.to(device, non_blocking=True)
            fg_mask = fg_mask.to(device, dtype=ad.dtype, non_blocking=True)

            raw = model(ad, pattern, fg_mask if fg_mask_enabled else None)
            out = blend_output_with_fg_mask(raw, ad, fg_mask)

            vgg_total, vgg_content, vgg_style = vgg_loss(out, ad, pattern)
            bg_fft = (1.0 - fg_mask) if fg_mask_enabled else None
            L_fft = fft_loss(out, pattern, bg_mask_bchw=bg_fft)
            L_tv = tv_loss(out)

            # Warm up FFT weight to preserve ad look early in training.
            warmup_steps = int(round(fft_warmup_epochs * steps_per_epoch))
            if warmup_steps <= 0:
                fft_scale = 1.0
            else:
                fft_scale = min(1.0, global_step / float(warmup_steps))

            loss_total = vgg_total + (w_fft * fft_scale) * L_fft + w_tv * L_tv

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            opt.step()

            bs = int(ad.size(0))
            running["loss_total"] += float(loss_total.item()) * bs
            running["loss_content"] += float(vgg_content.item()) * bs
            running["loss_style"] += float(vgg_style.item()) * bs
            running["loss_fft"] += float(L_fft.item()) * bs
            running["loss_tv"] += float(L_tv.item()) * bs
            n_batches += bs

            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss_total.item():.4f}",
                    "content": f"{vgg_content.item():.4f}",
                    "style": f"{vgg_style.item():.4f}",
                    "fft": f"{L_fft.item():.4f}",
                    "tv": f"{L_tv.item():.4f}",
                    "fft_scale": f"{fft_scale:.2f}",
                }
            )

            # Record and optionally plot train-only loss curves.
            step_points.append(global_step)
            step_train_loss_total.append(float(loss_total.item()))
            step_train_loss_content.append(float(vgg_content.item()))
            step_train_loss_style.append(float(vgg_style.item()))
            step_train_loss_fft.append(float(L_fft.item()))
            step_train_loss_tv.append(float(L_tv.item()))

            if loss_plot_every_steps > 0 and (global_step % loss_plot_every_steps == 0):
                save_loss_curves_train_only(
                    out_dir=out_dir,
                    step_points=step_points,
                    train_loss_total=step_train_loss_total,
                    train_loss_content=step_train_loss_content,
                    train_loss_style=step_train_loss_style,
                    train_loss_fft=step_train_loss_fft,
                    train_loss_tv=step_train_loss_tv,
                    max_points=loss_plot_max_points,
                )

        # Epoch summary
        for k in list(running.keys()):
            running[k] /= max(1, n_batches)

        # Validate
        model.eval()
        val_running = {k: 0.0 for k in running.keys()}
        val_n = 0
        with torch.no_grad():
            for ad, pattern, fg_mask, meta_hw in val_loader:
                ad = ad.to(device, non_blocking=True)
                pattern = pattern.to(device, non_blocking=True)
                fg_mask = fg_mask.to(device, dtype=ad.dtype, non_blocking=True)

                raw = model(ad, pattern, fg_mask if fg_mask_enabled else None)
                out = blend_output_with_fg_mask(raw, ad, fg_mask)
                vgg_total, vgg_content, vgg_style = vgg_loss(out, ad, pattern)
                bg_fft = (1.0 - fg_mask) if fg_mask_enabled else None
                L_fft = fft_loss(out, pattern, bg_mask_bchw=bg_fft)
                L_tv = tv_loss(out)

                # During validation use full FFT weight.
                loss_total = vgg_total + w_fft * L_fft + w_tv * L_tv

                bs = int(ad.size(0))
                val_running["loss_total"] += float(loss_total.item()) * bs
                val_running["loss_content"] += float(vgg_content.item()) * bs
                val_running["loss_style"] += float(vgg_style.item()) * bs
                val_running["loss_fft"] += float(L_fft.item()) * bs
                val_running["loss_tv"] += float(L_tv.item()) * bs
                val_n += bs

        for k in list(val_running.keys()):
            val_running[k] /= max(1, val_n)

        val_metric = val_running["loss_total"]
        is_best = val_metric < best_val
        if is_best:
            best_val = val_metric

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train total={running['loss_total']:.5f} content={running['loss_content']:.5f} style={running['loss_style']:.5f} "
            f"fft={running['loss_fft']:.5f} tv={running['loss_tv']:.5f} | "
            f"val total={val_running['loss_total']:.5f} (best={best_val:.5f})"
        )

        history["train_loss_total"].append(float(running["loss_total"]))
        history["train_loss_content"].append(float(running["loss_content"]))
        history["train_loss_style"].append(float(running["loss_style"]))
        history["train_loss_fft"].append(float(running["loss_fft"]))
        history["train_loss_tv"].append(float(running["loss_tv"]))

        history["val_loss_total"].append(float(val_running["loss_total"]))
        history["val_loss_content"].append(float(val_running["loss_content"]))
        history["val_loss_style"].append(float(val_running["loss_style"]))
        history["val_loss_fft"].append(float(val_running["loss_fft"]))
        history["val_loss_tv"].append(float(val_running["loss_tv"]))

        # Save checkpoints
        ckpt_last = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val": best_val,
                "cfg": cfg,
            },
            ckpt_last,
        )

        if is_best:
            ckpt_best = os.path.join(out_dir, "ckpt_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_val": best_val,
                    "cfg": cfg,
                },
                ckpt_best,
            )

        # Save debug images
        if epoch == 1 or (save_every_epochs > 0 and epoch % save_every_epochs == 0):
            # Use first val batch (model may still be in eval() from validation).
            with torch.no_grad():
                ad, pattern, fg_mask, meta_hw = next(iter(val_loader))
                ad = ad.to(device, non_blocking=True)
                pattern = pattern.to(device, non_blocking=True)
                fg_mask = fg_mask.to(device, dtype=ad.dtype, non_blocking=True)
                raw = model(ad, pattern, fg_mask if fg_mask_enabled else None)
                out = blend_output_with_fg_mask(raw, ad, fg_mask)
            save_debug_images(
                out_dir=out_dir,
                epoch=epoch,
                ad_t=ad[:debug_max_batch].detach().cpu(),
                out_t=out[:debug_max_batch].detach().cpu(),
                pattern_t=pattern[:debug_max_batch].detach().cpu(),
                orig_hw=meta_hw[:debug_max_batch].detach().cpu(),
            )

        # Optional epoch-level plot; disabled by default since step-level plot is faster.
        if loss_plot_every_epochs > 0 and (epoch == 1 or epoch % loss_plot_every_epochs == 0):
            # Keep it simple: only save a quick train-only snapshot.
            save_loss_curves_train_only(
                out_dir=out_dir,
                step_points=step_points,
                train_loss_total=step_train_loss_total,
                train_loss_content=step_train_loss_content,
                train_loss_style=step_train_loss_style,
                train_loss_fft=step_train_loss_fft,
                train_loss_tv=step_train_loss_tv,
                max_points=loss_plot_max_points,
            )


if __name__ == "__main__":
    main()


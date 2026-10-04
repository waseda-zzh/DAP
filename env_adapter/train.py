from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from env_adapter.dataset_carla import EnvAdapterCarlaDataset, EnvAdapterCarlaDatasetConfig
from env_adapter.display_loss import build_display_colors, compute_display_loss
from env_adapter.model import LumaContrastAdapter, bgr01_to_ycbcr, build_env_adapter
from env_adapter.paste import paste_adapted_native_on_scene_homography
from orb_proxy.orb_proxy_model import build_orb_proxy


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_repo_path(path: str, repo_root: str) -> str:
    p = str(path).strip()
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(repo_root, p))


def _optional_data_path(raw, repo_root: str) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return _resolve_repo_path(s, repo_root)


def _parse_set_like(raw) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        out = {str(x).strip() for x in raw if str(x).strip()}
        return out if out else None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


@dataclass
class TrainLossWeights:
    orb: float
    orb_delta: float
    grad: float
    luma_contrast: float
    display: float


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be mapping")
    return cfg


def _sobel_grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, c, _, _ = x.shape
    kx = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    kx = kx.expand(c, 1, 3, 3)
    ky = ky.expand(c, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def _structure_grad_l1_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    gx_p, gy_p = _sobel_grad_xy(pred)
    gx_r, gy_r = _sobel_grad_xy(ref)
    return (gx_p - gx_r).abs().mean() + (gy_p - gy_r).abs().mean()


def _luma_contrast_hinge_loss(
    adapted_bgr: torch.Tensor,
    pattern_bgr: torch.Tensor,
    mask_gain: float,
    min_delta: float,
    edge_only: bool = False,
    edge_band_radius: int = 1,
) -> torch.Tensor:
    y_ad = bgr01_to_ycbcr(adapted_bgr)[:, 0:1]
    y_pt = bgr01_to_ycbcr(pattern_bgr)[:, 0:1]
    med = torch.median(y_pt.view(y_pt.size(0), -1), dim=1).values.view(-1, 1, 1, 1)
    m_full = torch.sigmoid((y_pt - med) * float(mask_gain))
    if edge_only:
        # Focus luma separation on boundary band only.
        dx = torch.abs(m_full[:, :, :, 1:] - m_full[:, :, :, :-1])
        dy = torch.abs(m_full[:, :, 1:, :] - m_full[:, :, :-1, :])
        edge = torch.zeros_like(m_full)
        edge[:, :, :, 1:] += dx
        edge[:, :, 1:, :] += dy
        edge = edge.clamp(0.0, 1.0)
        if int(edge_band_radius) > 0:
            r = int(edge_band_radius)
            k = 2 * r + 1
            edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=r)
        m_hi = m_full * edge
        m_lo = (1.0 - m_full) * edge
    else:
        m_hi = m_full
        m_lo = 1.0 - m_full
    eps = 1e-6
    mu_hi = (y_ad * m_hi).sum(dim=(2, 3)) / (m_hi.sum(dim=(2, 3)) + eps)
    mu_lo = (y_ad * m_lo).sum(dim=(2, 3)) / (m_lo.sum(dim=(2, 3)) + eps)
    delta = (mu_hi - mu_lo).abs().view(-1)
    return torch.relu(float(min_delta) - delta).mean()


def _bchw_to_bgr_u8(x_bchw: torch.Tensor) -> np.ndarray:
    x = x_bchw.detach().float().cpu().clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def _write_csv(path: str, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _try_save_loss_curves_png(rows: list[dict[str, float | int]], out_path: str) -> bool:
    """
    Save train/val loss curves. If matplotlib is unavailable, silently skip.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not rows:
        return False

    xs = [int(r["epoch"]) for r in rows]
    series = [
        ("train_total", "train_total"),
        ("val_total", "val_total"),
        ("train_orb", "train_orb"),
        ("val_orb", "val_orb"),
        ("train_orb_delta", "train_orb_delta"),
        ("val_orb_delta", "val_orb_delta"),
        ("train_luma_contrast", "train_luma_contrast"),
        ("val_luma_contrast", "val_luma_contrast"),
        ("train_grad", "train_grad"),
        ("val_grad", "val_grad"),
        ("train_display", "train_display"),
        ("val_display", "val_display"),
    ]

    plt.figure(figsize=(10, 6), dpi=150)
    for key, label in series:
        ys = [float(r[key]) for r in rows]
        plt.plot(xs, ys, label=label, linewidth=1.8)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("env_adapter luma-first losses")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Train luma-first environment adapter.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo_root = _project_root()
    data_cfg = cfg.get("data") or {}
    train_cfg = cfg.get("training") or {}
    model_cfg = cfg.get("model") or {}
    loss_cfg = cfg.get("loss") or {}
    orb_cfg = cfg.get("orb_proxy") or {}
    display_cfg = cfg.get("display") or {}
    vis_cfg = cfg.get("visualization") or {}

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device_str = args.device if args.device else str(train_cfg.get("device", "cuda"))
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[setup] device={device}")

    mode = str(data_cfg.get("mode", "carla")).strip().lower()
    if mode != "carla":
        raise ValueError("This redesigned trainer supports data.mode=carla only.")

    ds_cfg = EnvAdapterCarlaDatasetConfig(
        carla_root_dir=_resolve_repo_path(str(data_cfg["carla_root_dir"]), repo_root),
        patch_dir=_resolve_repo_path(str(data_cfg["attack_patch_dir"]), repo_root),
        carla_towns=_parse_set_like(data_cfg.get("carla_towns")),
        error_pattern_path=_optional_data_path(data_cfg.get("error_pattern_path"), repo_root),
        error_pattern_dir=_optional_data_path(data_cfg.get("error_pattern_dir"), repo_root),
        seed=seed,
        max_frames_per_town=int(data_cfg["max_frames_per_town"]) if data_cfg.get("max_frames_per_town") is not None else None,
        patch_width=int(data_cfg["carla_patch_width"]) if data_cfg.get("carla_patch_width") is not None else None,
    )
    ds = EnvAdapterCarlaDataset(ds_cfg)
    tr_idx, va_idx = _split_indices(len(ds), float(train_cfg.get("val_ratio", 0.1)), seed)
    train_ds = Subset(ds, tr_idx)
    val_ds = Subset(ds, va_idx)
    print(f"[data] mode=carla total={len(ds)} train={len(train_ds)} val={len(val_ds)}")

    batch_size = 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    adapter = build_env_adapter(model_cfg).to(device)
    print(f"[model] arch={model_cfg.get('arch','luma_contrast')} ({type(adapter).__name__})")

    orb_proxy = build_orb_proxy(in_channels=4).to(device)
    orb_ckpt = _resolve_repo_path(str(orb_cfg["checkpoint"]), repo_root)
    state = torch.load(orb_ckpt, map_location=device)
    orb_proxy.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    orb_proxy.eval()
    for p in orb_proxy.parameters():
        p.requires_grad = False

    disp_np = build_display_colors(
        icc_path=_resolve_repo_path(str(display_cfg.get("icc_path")), repo_root) if display_cfg.get("icc_path") else None,
        levels_per_channel=int(display_cfg.get("levels_per_channel", 4)),
    )
    display_colors = torch.from_numpy(disp_np).to(device=device, dtype=torch.float32)

    w = TrainLossWeights(
        orb=float(loss_cfg.get("w_orb", 1.0)),
        orb_delta=float(loss_cfg.get("w_orb_delta", 0.0)),
        grad=float(loss_cfg.get("w_grad", 0.2)),
        luma_contrast=float(loss_cfg.get("w_luma_contrast", 0.25)),
        display=float(loss_cfg.get("w_display", 0.01)),
    )
    luma_min = float(loss_cfg.get("luma_contrast_min", 0.14))
    orb_delta_margin = float(loss_cfg.get("orb_delta_margin", 0.03))

    opt = torch.optim.AdamW(adapter.parameters(), lr=float(train_cfg.get("lr", 2e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))

    out_dir = _resolve_repo_path(str(train_cfg.get("out_dir", "env_adapter_ckpt")), repo_root)
    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    csv_path = os.path.join(out_dir, "metrics.csv")
    curves_path = os.path.join(out_dir, "loss_curves.png")

    vis_enabled = bool(vis_cfg.get("enabled", False))
    vis_every = int(vis_cfg.get("every_steps", 200))
    vis_dir = _resolve_repo_path(str(vis_cfg.get("dir", os.path.join(out_dir, "images"))), repo_root)
    if vis_enabled:
        os.makedirs(vis_dir, exist_ok=True)
    global_step = 0

    def run_epoch(loader: DataLoader, train: bool) -> dict[str, float]:
        nonlocal global_step
        adapter.train(mode=train)
        sums = {"total": 0.0, "orb": 0.0, "orb_delta": 0.0, "grad": 0.0, "luma_contrast": 0.0, "display": 0.0}
        n = 0
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in pbar:
            scene = batch["scene_bgr"].to(device, non_blocking=True)
            patch = batch["patch_native_bgr"].to(device, non_blocking=True)
            roi_box = batch["roi_box"].to(device, non_blocking=True)
            H = batch["homography_b2s"].to(device, non_blocking=True)
            pat = batch.get("pattern_native_bgr")
            if pat is not None:
                pat = pat.to(device, non_blocking=True)

            adapted = adapter(patch, scene, roi_box=roi_box, error_pattern_bchw=pat)
            comp, mask = paste_adapted_native_on_scene_homography(scene, adapted, H)
            comp_rs = F.interpolate(comp, size=(int(data_cfg.get("proxy_input_h", 400)), int(data_cfg.get("proxy_input_w", 800))), mode="bilinear", align_corners=False)
            mask_rs = F.interpolate(mask, size=comp_rs.shape[-2:], mode="nearest")
            proxy_x = torch.cat([comp_rs, mask_rs], dim=1)
            orb_score = orb_proxy(proxy_x)
            l_orb = (1.0 - orb_score).mean()

            # Baseline (no adapter): same scene/homography with original patch.
            if w.orb_delta > 0:
                with torch.no_grad():
                    comp_base, mask_base = paste_adapted_native_on_scene_homography(scene, patch, H)
                    comp_base_rs = F.interpolate(
                        comp_base,
                        size=(int(data_cfg.get("proxy_input_h", 400)), int(data_cfg.get("proxy_input_w", 800))),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask_base_rs = F.interpolate(mask_base, size=comp_base_rs.shape[-2:], mode="nearest")
                    proxy_base = torch.cat([comp_base_rs, mask_base_rs], dim=1)
                    orb_base = orb_proxy(proxy_base)
                orb_delta = (orb_score - orb_base).mean()
                l_orb_delta = torch.relu(float(orb_delta_margin) - orb_delta)
            else:
                l_orb_delta = l_orb.new_tensor(0.0)

            l_grad = _structure_grad_l1_loss(adapted, patch)
            ref = pat if pat is not None else patch
            l_luma = _luma_contrast_hinge_loss(
                adapted,
                ref,
                mask_gain=float(getattr(adapter, "mask_gain", 24.0)),
                min_delta=luma_min,
                edge_only=bool(getattr(adapter, "edge_only_contrast", False)),
                edge_band_radius=int(getattr(adapter, "edge_band_radius", 1)),
            )
            l_disp = compute_display_loss(adapted, display_colors)
            total = (
                w.orb * l_orb
                + w.orb_delta * l_orb_delta
                + w.grad * l_grad
                + w.luma_contrast * l_luma
                + w.display * l_disp
            )

            if train:
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
                global_step += 1
                if vis_enabled and vis_every > 0 and global_step % vis_every == 0:
                    stem = str(batch.get("stem", ["sample"])[0] if isinstance(batch.get("stem"), list) else batch.get("stem", "sample"))
                    cv2.imwrite(os.path.join(vis_dir, f"step_{global_step:08d}_{stem}_patch.png"), _bchw_to_bgr_u8(adapted[0:1]))
                    cv2.imwrite(os.path.join(vis_dir, f"step_{global_step:08d}_{stem}_composite.png"), _bchw_to_bgr_u8(comp[0:1]))

            bs = int(scene.size(0))
            n += bs
            sums["total"] += float(total.item()) * bs
            sums["orb"] += float(l_orb.item()) * bs
            sums["orb_delta"] += float(l_orb_delta.item()) * bs
            sums["grad"] += float(l_grad.item()) * bs
            sums["luma_contrast"] += float(l_luma.item()) * bs
            sums["display"] += float(l_disp.item()) * bs
            pbar.set_postfix(total=sums["total"] / max(1, n), orb=sums["orb"] / max(1, n))
        return {k: v / max(1, n) for k, v in sums.items()}

    epochs = int(train_cfg.get("epochs", 30))
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        tr = run_epoch(train_loader, train=True)
        with torch.no_grad():
            va = run_epoch(val_loader, train=False)
        row = {
            "epoch": ep,
            "train_total": tr["total"],
            "train_orb": tr["orb"],
            "train_orb_delta": tr["orb_delta"],
            "train_grad": tr["grad"],
            "train_luma_contrast": tr["luma_contrast"],
            "train_display": tr["display"],
            "val_total": va["total"],
            "val_orb": va["orb"],
            "val_orb_delta": va["orb_delta"],
            "val_grad": va["grad"],
            "val_luma_contrast": va["luma_contrast"],
            "val_display": va["display"],
        }
        rows.append(row)
        print(
            f"[ep {ep:03d}] "
            f"train_total={tr['total']:.4f} val_total={va['total']:.4f} | "
            f"train_orb={tr['orb']:.4f} val_orb={va['orb']:.4f} | "
            f"train_orb_delta={tr['orb_delta']:.4f} val_orb_delta={va['orb_delta']:.4f} | "
            f"train_luma={tr['luma_contrast']:.4f} val_luma={va['luma_contrast']:.4f} | "
            f"train_grad={tr['grad']:.4f} val_grad={va['grad']:.4f} | "
            f"train_disp={tr['display']:.4f} val_disp={va['display']:.4f}"
        )
        torch.save({"model": adapter.state_dict(), "epoch": ep, "row": row}, os.path.join(out_dir, "last.pt"))
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save({"model": adapter.state_dict(), "epoch": ep, "row": row}, os.path.join(out_dir, "best.pt"))
            print(f"[save] best -> {os.path.join(out_dir, 'best.pt')}")
        _write_csv(csv_path, rows)
        curves_ok = _try_save_loss_curves_png(rows, curves_path)
        if curves_ok:
            print(f"[log] loss curves -> {curves_path}")
        else:
            print("[log] loss curves skipped (matplotlib not available).")


if __name__ == "__main__":
    main()


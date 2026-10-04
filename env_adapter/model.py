from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bgr01_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or x.size(1) != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(x.shape)}")
    b, g, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.492 * (b - y) + 0.5
    cr = 0.877 * (r - y) + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr01_to_bgr(yuv: torch.Tensor) -> torch.Tensor:
    if yuv.dim() != 4 or yuv.size(1) != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(yuv.shape)}")
    y = yuv[:, 0:1]
    cb = yuv[:, 1:2] - 0.5
    cr = yuv[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([b, g, r], dim=1).clamp(0.0, 1.0)


def roi_mean_bgr(scene_bchw: torch.Tensor, roi_box_b4: torch.Tensor | None) -> torch.Tensor:
    bsz, _, h, w = scene_bchw.shape
    if roi_box_b4 is None:
        return scene_bchw.mean(dim=(2, 3))
    out = []
    for i in range(bsz):
        x1, y1, x2, y2 = [int(v.item()) for v in roi_box_b4[i]]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        r = scene_bchw[i : i + 1, :, y1:y2, x1:x2]
        out.append(r.mean(dim=(2, 3)).squeeze(0))
    return torch.stack(out, dim=0)


class LumaContrastAdapter(nn.Module):
    """
    Luma-first environment adapter:
    - keep patch structure
    - increase region luminance gap and edge contrast
    - color anchored to environment (ROI + scene), only tiny chroma bias.
    """

    def __init__(
        self,
        scene_base_channels: int = 16,
        hidden: int = 128,
        mask_gain: float = 24.0,
        delta_y_max: float = 0.30,
        edge_gain_max: float = 0.25,
        chroma_bias_max: float = 0.03,
        edge_only_contrast: bool = True,
        edge_band_radius: int = 1,
        chroma_edge_mix: float = 1.0,
    ):
        super().__init__()
        self.mask_gain = float(mask_gain)
        self.delta_y_max = float(delta_y_max)
        self.edge_gain_max = float(edge_gain_max)
        self.chroma_bias_max = float(chroma_bias_max)
        self.edge_only_contrast = bool(edge_only_contrast)
        self.edge_band_radius = max(0, int(edge_band_radius))
        self.chroma_edge_mix = max(0.0, min(float(chroma_edge_mix), 1.0))

        s1 = scene_base_channels
        s2 = scene_base_channels * 2
        s3 = scene_base_channels * 4
        self.scene_conv1 = nn.Sequential(
            nn.Conv2d(3, s1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(s1),
            nn.ReLU(inplace=True),
        )
        self.scene_conv2 = nn.Sequential(
            nn.Conv2d(s1, s2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(s2),
            nn.ReLU(inplace=True),
        )
        self.scene_conv3 = nn.Sequential(
            nn.Conv2d(s2, s3, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(s3),
            nn.ReLU(inplace=True),
        )
        self.scene_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Outputs: alpha_env, d_hi, d_lo, edge_gain, cb_bias, cr_bias
        self.fc = nn.Sequential(
            nn.Linear(s3 + 6, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), 6),
        )

    def _partition(self, patch_bchw: torch.Tensor, error_pattern_bchw: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        y_patch = bgr01_to_ycbcr(patch_bchw)[:, 0:1]
        if error_pattern_bchw is not None:
            if error_pattern_bchw.shape[-2:] != patch_bchw.shape[-2:]:
                error_pattern_bchw = F.interpolate(
                    error_pattern_bchw, size=patch_bchw.shape[-2:], mode="bilinear", align_corners=False
                )
            y_src = bgr01_to_ycbcr(error_pattern_bchw)[:, 0:1]
        else:
            y_src = y_patch
        med = torch.median(y_src.view(y_src.size(0), -1), dim=1).values.view(-1, 1, 1, 1)
        m = torch.sigmoid((y_src - med) * self.mask_gain)
        # Edge mask from partition gradients
        dx = torch.abs(m[:, :, :, 1:] - m[:, :, :, :-1])
        dy = torch.abs(m[:, :, 1:, :] - m[:, :, :-1, :])
        edge = torch.zeros_like(m)
        edge[:, :, :, 1:] += dx
        edge[:, :, 1:, :] += dy
        edge = edge.clamp(0.0, 1.0)
        if self.edge_band_radius > 0:
            k = 2 * self.edge_band_radius + 1
            edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=self.edge_band_radius)
        return m, edge

    def forward(
        self,
        patch_bchw: torch.Tensor,
        scene_bchw: torch.Tensor,
        roi_box: torch.Tensor | None = None,
        error_pattern_bchw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_bchw.dim() != 4 or patch_bchw.size(1) != 3:
            raise ValueError(f"Expected patch (B,3,H,W), got {tuple(patch_bchw.shape)}")
        if scene_bchw.dim() != 4 or scene_bchw.size(1) != 3:
            raise ValueError(f"Expected scene (B,3,H,W), got {tuple(scene_bchw.shape)}")
        if patch_bchw.size(0) != scene_bchw.size(0):
            raise ValueError("patch/scene batch mismatch")

        m, edge = self._partition(patch_bchw, error_pattern_bchw)
        yuv_patch = bgr01_to_ycbcr(patch_bchw)
        y = yuv_patch[:, 0:1]

        s = self.scene_conv1(scene_bchw)
        s = self.scene_conv2(s)
        s = self.scene_conv3(s)
        s = self.scene_pool(s).flatten(1)

        roi_m = roi_mean_bgr(scene_bchw, roi_box)
        scene_m = scene_bchw.mean(dim=(2, 3))
        q = self.fc(torch.cat([s, roi_m, scene_m], dim=1))
        alpha_env, d_hi_raw, d_lo_raw, edge_gain_raw, cb_bias_raw, cr_bias_raw = torch.sigmoid(q).chunk(6, dim=1)

        d_hi = (d_hi_raw * 2.0 - 1.0) * self.delta_y_max
        d_lo = (d_lo_raw * 2.0 - 1.0) * self.delta_y_max
        edge_gain = edge_gain_raw * self.edge_gain_max
        cb_bias = (cb_bias_raw * 2.0 - 1.0) * self.chroma_bias_max
        cr_bias = (cr_bias_raw * 2.0 - 1.0) * self.chroma_bias_max

        # Luma modulation:
        # - edge_only_contrast=True: keep interiors natural, push contrast mostly on boundaries.
        # - else: region-wise + boundary enhancement.
        if self.edge_only_contrast:
            # Region deltas only contribute near boundary band.
            band_delta = edge * (m * d_hi.unsqueeze(-1).unsqueeze(-1) + (1.0 - m) * d_lo.unsqueeze(-1).unsqueeze(-1))
            y_out = y + band_delta
        else:
            y_out = y + m * d_hi.unsqueeze(-1).unsqueeze(-1) + (1.0 - m) * d_lo.unsqueeze(-1).unsqueeze(-1)
        y_out = y_out + edge_gain.unsqueeze(-1).unsqueeze(-1) * edge * (2.0 * m - 1.0)
        y_out = y_out.clamp(0.0, 1.0)

        # Chroma control:
        # - edge_only_contrast=True: preserve interior patch chroma, blend to env chroma near boundary band.
        # - else: use environment chroma on the whole patch.
        patch_cb = yuv_patch[:, 1:2]
        patch_cr = yuv_patch[:, 2:3]
        roi_yuv = bgr01_to_ycbcr(roi_m[:, :, None, None])
        scene_yuv = bgr01_to_ycbcr(scene_m[:, :, None, None])
        cb_env = (alpha_env * roi_yuv[:, 1:2, 0, 0] + (1.0 - alpha_env) * scene_yuv[:, 1:2, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        cr_env = (alpha_env * roi_yuv[:, 2:3, 0, 0] + (1.0 - alpha_env) * scene_yuv[:, 2:3, 0, 0]).unsqueeze(-1).unsqueeze(-1)
        cb_tgt = (cb_env + cb_bias.unsqueeze(-1).unsqueeze(-1)).clamp(0.0, 1.0)
        cr_tgt = (cr_env + cr_bias.unsqueeze(-1).unsqueeze(-1)).clamp(0.0, 1.0)
        if self.edge_only_contrast:
            # edge in [0,1], after dilation by edge_band_radius; interior gets low weight.
            w_edge = (edge * self.chroma_edge_mix).clamp(0.0, 1.0)
            cb_out = patch_cb * (1.0 - w_edge) + cb_tgt * w_edge
            cr_out = patch_cr * (1.0 - w_edge) + cr_tgt * w_edge
        else:
            cb_out = cb_tgt.expand_as(y_out)
            cr_out = cr_tgt.expand_as(y_out)

        return ycbcr01_to_bgr(torch.cat([y_out, cb_out.expand_as(y_out), cr_out.expand_as(y_out)], dim=1))

    def chroma_separation_hinge(
        self,
        patch_bchw: torch.Tensor,
        scene_bchw: torch.Tensor,
        roi_box: torch.Tensor | None,
        min_sep: float,
    ) -> torch.Tensor:
        # Kept for compatibility; luma-first model relies more on luma loss.
        return patch_bchw.new_tensor(0.0)


def build_env_adapter(model_cfg: dict) -> nn.Module:
    arch = str(model_cfg.get("arch", "luma_contrast")).strip().lower()
    if arch in ("luma_contrast", "luma_first", "luma"):
        return LumaContrastAdapter(
            scene_base_channels=int(model_cfg.get("scene_base_channels", 16)),
            hidden=int(model_cfg.get("recolor_hidden", 128)),
            mask_gain=float(model_cfg.get("mask_gain", 24.0)),
            delta_y_max=float(model_cfg.get("delta_y_max", 0.30)),
            edge_gain_max=float(model_cfg.get("edge_gain_max", 0.25)),
            chroma_bias_max=float(model_cfg.get("chroma_bias_max", 0.03)),
            edge_only_contrast=bool(model_cfg.get("edge_only_contrast", True)),
            edge_band_radius=int(model_cfg.get("edge_band_radius", 1)),
            chroma_edge_mix=float(model_cfg.get("chroma_edge_mix", 1.0)),
        )
    raise ValueError(f"Unknown model.arch={arch!r}; expected luma_contrast")


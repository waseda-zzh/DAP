from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import VGG16_Weights, vgg16
except Exception as e:  # pragma: no cover
    raise RuntimeError("torchvision is required for VGG16 losses") from e


def rgb_to_grayscale(rgb_chw: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to grayscale.
    Input:  (B,3,H,W) float32 in [0,1]
    Output: (B,1,H,W)
    """
    if rgb_chw.dim() != 4 or rgb_chw.size(1) != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(rgb_chw.shape)}")
    r, g, b = rgb_chw[:, 0:1], rgb_chw[:, 1:2], rgb_chw[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class VGG16FeatureExtractor(nn.Module):
    """
    Lightweight feature extractor for perceptual/style losses.

    Uses torchvision VGG16's `features` Sequential and returns intermediate activations
    at specified layer indices.
    """

    # Common relu output indices in torchvision VGG16.features:
    # conv1_1=0, relu1_1=1, conv1_2=2, relu1_2=3, pool1=4,
    # conv2_1=5, relu2_1=6, conv2_2=7, relu2_2=8, pool2=9,
    # conv3_1=10, relu3_1=11, conv3_2=12, relu3_2=13, conv3_3=14, relu3_3=15, pool3=16,
    # conv4_1=17 ... relu4_3=22, pool4=23,
    # relu5_3=29
    NAME_TO_INDEX = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_3": 15,
        "relu4_3": 22,
        "relu5_3": 29,
    }

    def __init__(self, layer_names: Iterable[str]):
        super().__init__()
        layer_names = list(layer_names)
        if not layer_names:
            raise ValueError("layer_names must be non-empty")

        layer_indices: list[int] = []
        for name in layer_names:
            n = str(name).strip()
            if n not in self.NAME_TO_INDEX:
                raise ValueError(f"Unknown VGG layer name: {n}. Supported: {sorted(self.NAME_TO_INDEX.keys())}")
            layer_indices.append(self.NAME_TO_INDEX[n])
        self.layer_indices = sorted(set(layer_indices))

        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights)
        self.features = vgg.features

        # Freeze VGG.
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        # ImageNet normalization used by VGG16 weights.
        self.register_buffer("mean", torch.tensor(weights.transforms().mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(weights.transforms().std).view(1, 3, 1, 1))

    def forward(self, x_rgb_bchw: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns activations for all requested layers.
        """
        if x_rgb_bchw.dim() != 4 or x_rgb_bchw.size(1) != 3:
            raise ValueError(f"Expected (B,3,H,W) input, got {tuple(x_rgb_bchw.shape)}")

        # VGG expects 224x224; callers may already resize but we enforce it for safety.
        x = F.interpolate(x_rgb_bchw, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        activations: dict[str, torch.Tensor] = {}
        # We match requested output by index only; convert index -> closest name
        inv_name = {v: k for k, v in self.NAME_TO_INDEX.items()}

        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.layer_indices:
                name = inv_name.get(i, f"idx_{i}")
                activations[name] = h
        return activations


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Gram matrix of VGG features.
    Input: (B,C,H,W)
    Output: (B,C,C) normalized by spatial size.
    """
    if feat.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(feat.shape)}")
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w + 1e-8)


@dataclass
class ContentStyleLossWeights:
    content_weight: float = 1.0
    style_weight: float = 0.2
    content_layers: tuple[str, ...] = ("relu3_3",)
    style_layers: tuple[str, ...] = ("relu2_2", "relu3_3")
    # Per-layer weights (optional)
    content_layer_weights: tuple[float, ...] | None = None
    style_layer_weights: tuple[float, ...] | None = None
    # If true, style features are extracted from grayscale (structure-only)
    style_use_grayscale: bool = True


class VGG16ContentStyleMSELoss(nn.Module):
    """
    Content/style loss using VGG16 features:
      - content: MSE between VGG(out) and VGG(ad) at content_layers
      - style:   AdaIN-style feature statistics matching at style_layers:
                 MSE(mean(out), mean(pattern)) + MSE(std(out), std(pattern))
                 Optional grayscale-only style path (structure over color).
    """

    def __init__(self, weights: ContentStyleLossWeights):
        super().__init__()
        self.weights = weights

        self.content_extractor = VGG16FeatureExtractor(weights.content_layers)
        self.style_extractor = VGG16FeatureExtractor(weights.style_layers)

        if weights.content_layer_weights is not None:
            if len(weights.content_layer_weights) != len(weights.content_layers):
                raise ValueError("content_layer_weights length must match content_layers length")
            self.content_layer_weights = torch.tensor(weights.content_layer_weights, dtype=torch.float32)
        else:
            self.content_layer_weights = None

        if weights.style_layer_weights is not None:
            if len(weights.style_layer_weights) != len(weights.style_layers):
                raise ValueError("style_layer_weights length must match style_layers length")
            self.style_layer_weights = torch.tensor(weights.style_layer_weights, dtype=torch.float32)
        else:
            self.style_layer_weights = None

    @staticmethod
    def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(a, b, reduction="mean")

    @staticmethod
    def _channel_mean_std(feat: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel moments for AdaIN-style statistics matching."""
        if feat.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(feat.shape)}")
        mean = feat.mean(dim=(-2, -1), keepdim=False)  # (B,C)
        var = feat.var(dim=(-2, -1), unbiased=False, keepdim=False)
        std = torch.sqrt(var + eps)  # (B,C)
        return mean, std

    def forward(
        self,
        out_rgb_bchw: torch.Tensor,
        ad_rgb_bchw: torch.Tensor,
        pattern_rgb_bchw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (total_loss, content_loss, style_loss)
        """
        content_feats_out = self.content_extractor(out_rgb_bchw)
        content_feats_ad = self.content_extractor(ad_rgb_bchw)

        if self.weights.style_use_grayscale:
            # style from luminance structure, not chroma; replicate to 3ch for VGG
            out_style_in = rgb_to_grayscale(out_rgb_bchw).repeat(1, 3, 1, 1)
            pat_style_in = rgb_to_grayscale(pattern_rgb_bchw).repeat(1, 3, 1, 1)
        else:
            out_style_in = out_rgb_bchw
            pat_style_in = pattern_rgb_bchw

        style_feats_out = self.style_extractor(out_style_in)
        style_feats_pat = self.style_extractor(pat_style_in)

        content_losses: list[torch.Tensor] = []
        for li, name in enumerate(self.weights.content_layers):
            f_out = content_feats_out[name]
            f_ad = content_feats_ad[name]
            l = self._mse(f_out, f_ad)
            if self.content_layer_weights is not None:
                l = l * self.content_layer_weights[li].to(l.device)
            content_losses.append(l)
        content_loss = sum(content_losses)

        style_losses: list[torch.Tensor] = []
        for li, name in enumerate(self.weights.style_layers):
            f_out = style_feats_out[name]
            f_pat = style_feats_pat[name]
            mu_out, std_out = self._channel_mean_std(f_out)
            mu_pat, std_pat = self._channel_mean_std(f_pat)
            l = self._mse(mu_out, mu_pat) + self._mse(std_out, std_pat)
            if self.style_layer_weights is not None:
                l = l * self.style_layer_weights[li].to(l.device)
            style_losses.append(l)
        style_loss = sum(style_losses)

        total = self.weights.content_weight * content_loss + self.weights.style_weight * style_loss
        return total, content_loss, style_loss


class FFTGridSpectralLoss(nn.Module):
    """
    FFT-based spectral loss to encourage grid-like periodic structure.

    Loss:
      - grayscale + mean removal
      - log(1+|FFT|) amplitude
      - compute mask from template pattern spectrum (top quantile)
      - weighted MSE between spectra in masked frequencies
    """

    def __init__(
        self,
        *,
        top_quantile: float = 0.995,
        use_log1p: bool = True,
        normalize_spectra: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.top_quantile = float(top_quantile)
        self.use_log1p = bool(use_log1p)
        self.normalize_spectra = bool(normalize_spectra)
        self.eps = float(eps)

    @staticmethod
    def _fft_amp(x_gray_bchw: torch.Tensor) -> torch.Tensor:
        # x_gray: (B,1,H,W)
        # Use ortho normalization for scale stability.
        x = x_gray_bchw.squeeze(1)  # (B,H,W)
        # Center amplitude around zero frequency.
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        f = torch.fft.fft2(x, norm="ortho")
        f = torch.fft.fftshift(f, dim=(-2, -1))
        amp = torch.abs(f)  # (B,H,W)
        return amp

    def forward(
        self,
        out_rgb_bchw: torch.Tensor,
        pattern_rgb_bchw: torch.Tensor,
        bg_mask_bchw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out_rgb_bchw.shape != pattern_rgb_bchw.shape:
            raise ValueError("out_rgb_bchw and pattern_rgb_bchw must have the same shape")
        if out_rgb_bchw.dim() != 4 or out_rgb_bchw.size(1) != 3:
            raise ValueError(f"Expected (B,3,H,W), got {tuple(out_rgb_bchw.shape)}")

        out_gray = rgb_to_grayscale(out_rgb_bchw)
        pat_gray = rgb_to_grayscale(pattern_rgb_bchw)

        if bg_mask_bchw is not None:
            if bg_mask_bchw.dim() != 4 or bg_mask_bchw.size(1) != 1:
                raise ValueError(f"bg_mask_bchw must be (B,1,H,W), got {tuple(bg_mask_bchw.shape)}")
            if bg_mask_bchw.shape[2:] != out_gray.shape[2:]:
                raise ValueError("bg_mask_bchw spatial size must match out")
            w = bg_mask_bchw.clamp(0.0, 1.0)
            out_gray = out_gray * w
            pat_gray = pat_gray * w

        amp_out = self._fft_amp(out_gray)  # (B,H,W)
        amp_pat = self._fft_amp(pat_gray)  # (B,H,W)

        if self.use_log1p:
            amp_out = torch.log1p(amp_out)
            amp_pat = torch.log1p(amp_pat)

        if self.normalize_spectra:
            # Normalize by per-sample mean to reduce scale sensitivity.
            mean_out = amp_out.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            mean_pat = amp_pat.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            amp_out = amp_out / mean_out
            amp_pat = amp_pat / mean_pat

        # Create mask from template spectrum only (no gradients through mask).
        amp_pat_det = amp_pat.detach()
        thr = torch.quantile(amp_pat_det.view(amp_pat_det.size(0), -1), self.top_quantile, dim=1, keepdim=True)
        thr = thr.view(-1, 1, 1)
        mask = (amp_pat_det >= thr).float()  # (B,H,W)

        diff2 = (amp_out - amp_pat) ** 2
        num = (diff2 * mask).sum(dim=(-2, -1))
        den = mask.sum(dim=(-2, -1)).clamp_min(self.eps)
        loss = (num / den).mean()
        return loss


class TotalVariationLoss(nn.Module):
    """
    Standard TV loss (encourages spatial smoothness).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, x_rgb_bchw: torch.Tensor) -> torch.Tensor:
        if x_rgb_bchw.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x_rgb_bchw.shape)}")
        dh = torch.abs(x_rgb_bchw[:, :, 1:, :] - x_rgb_bchw[:, :, :-1, :])
        dw = torch.abs(x_rgb_bchw[:, :, :, 1:] - x_rgb_bchw[:, :, :, :-1])
        loss = dh.mean() + dw.mean()
        if self.reduction == "sum":
            loss = dh.sum() + dw.sum()
        return loss


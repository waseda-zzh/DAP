from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_grayscale(rgb_chw: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale.
    Input:  (B,3,H,W) float32 in [0,1]
    Output: (B,1,H,W)
    """
    if rgb_chw.dim() != 4 or rgb_chw.size(1) != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(rgb_chw.shape)}")
    r, g, b = rgb_chw[:, 0:1], rgb_chw[:, 1:2], rgb_chw[:, 2:3]
    # ITU-R BT.601 luma transform.
    return 0.299 * r + 0.587 * g + 0.114 * b


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm: str = "bn", act: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            # Use groups to avoid very small channel edge cases.
            groups = min(32, out_ch)
            self.norm = nn.GroupNorm(groups, out_ch)
        else:
            raise ValueError("norm must be 'bn' or 'gn'")

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError("act must be 'relu' or 'leaky_relu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    """
    Wide shallow building block: (conv-norm-act) x N.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_layers: int = 2,
        norm: str = "bn",
        act: str = "relu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(ConvNormAct(ch, out_ch, norm=norm, act=act))
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideShallowPatternUNet(nn.Module):
    """
    Wide shallow multi-scale UNet with pattern conditioning.

    Key points:
    - depth is small (num_down defaults to 3 => 4 resolutions)
    - channels are relatively wide at each stage
    - decoder merges features via concat + conv
    - error pattern is injected at every decoder resolution (multi-scale fusion)
    """

    def __init__(
        self,
        *,
        ad_in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 96,
        max_channels: int = 512,
        num_down: int = 3,
        convs_per_stage: int = 2,
        pattern_cond_on_gray: bool = True,
        pattern_gray_channels: int = 1,
        use_fg_mask: bool = False,
        norm: str = "bn",
        act: str = "relu",
    ):
        super().__init__()

        if num_down < 1:
            raise ValueError("num_down must be >= 1")

        self.num_down = int(num_down)
        self.pattern_cond_on_gray = bool(pattern_cond_on_gray)
        self.use_fg_mask = bool(use_fg_mask)

        # Channel schedule (wide but not overly deep).
        # Example (base=96, max=512, num_down=3): [96, 192, 384, 512]
        chs: list[int] = []
        for i in range(self.num_down + 1):
            ch_i = int(base_channels * (2**i))
            ch_i = min(ch_i, int(max_channels))
            chs.append(ch_i)
        self.chs = chs

        # Encoder
        # Optional: concat foreground mask (1 ch) so the net knows where to avoid editing.
        # Pattern grayscale is concatenated at full resolution; when use_fg_mask, pattern
        # is gated by background (1-fg_mask) inside forward().
        enc0_in = int(ad_in_channels)
        if self.use_fg_mask:
            enc0_in += 1  # fg_mask channel
        enc0_in += int(pattern_gray_channels) if self.pattern_cond_on_gray else 0
        self.enc0 = ConvBlock(enc0_in, chs[0], num_layers=convs_per_stage, norm=norm, act=act)

        self.downs: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.enc_blocks: nn.ModuleList[nn.Module] = nn.ModuleList()
        for i in range(self.num_down):
            # Strided conv downsampling for learnable capacity.
            down = nn.Conv2d(chs[i], chs[i + 1], kernel_size=3, stride=2, padding=1, bias=True)
            self.downs.append(down)
            self.enc_blocks.append(ConvBlock(chs[i + 1], chs[i + 1], num_layers=convs_per_stage, norm=norm, act=act))

        # Pattern conditioning projections per resolution.
        cond_in_ch = int(pattern_gray_channels) if self.pattern_cond_on_gray else 3
        self.cond_projs = nn.ModuleList()
        for i in range(self.num_down + 1):
            # Map pattern to decoder/skip channel width.
            self.cond_projs.append(nn.Conv2d(cond_in_ch, chs[i], kernel_size=1, bias=True))

        # Decoder
        self.up_proj: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.dec_fuse_blocks: nn.ModuleList[nn.Module] = nn.ModuleList()
        for i in range(self.num_down - 1, -1, -1):
            # i runs over skip resolutions: chs[i]
            # Up feature comes from chs[i+1] and is projected to chs[i] before concat.
            self.up_proj.append(nn.Conv2d(chs[i + 1], chs[i], kernel_size=1, bias=True))
            # concat: up_proj(chs[i]) + skip(chs[i]) + cond(chs[i])
            fuse_in_ch = chs[i] * 3
            self.dec_fuse_blocks.append(ConvBlock(fuse_in_ch, chs[i], num_layers=convs_per_stage, norm=norm, act=act))

        self.out_head = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chs[0], out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ad_rgb_bchw: torch.Tensor,
        pattern_rgb_bchw: torch.Tensor,
        fg_mask_bchw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ad_rgb_bchw:       (B,3,H,W) in [0,1]
        pattern_rgb_bchw: (B,3,H,W) in [0,1] (fixed error pattern)
        fg_mask_bchw:     optional (B,1,H,W) in [0,1], 1=foreground (preserve object), 0=background (allow grid).
                          Required when use_fg_mask=True.
        """
        if ad_rgb_bchw.dim() != 4 or ad_rgb_bchw.size(1) != 3:
            raise ValueError(f"Expected ad_rgb_bchw (B,3,H,W), got {tuple(ad_rgb_bchw.shape)}")
        if pattern_rgb_bchw.shape != ad_rgb_bchw.shape:
            raise ValueError(f"pattern_rgb_bchw must match ad shape. ad={tuple(ad_rgb_bchw.shape)} pattern={tuple(pattern_rgb_bchw.shape)}")

        b, _, h, w = ad_rgb_bchw.shape
        if pattern_rgb_bchw.size(0) != b:
            raise ValueError("Batch size mismatch between ad and pattern")

        if self.use_fg_mask:
            if fg_mask_bchw is None:
                raise ValueError("fg_mask_bchw is required when use_fg_mask=True")
            if fg_mask_bchw.dim() != 4 or fg_mask_bchw.size(1) != 1 or fg_mask_bchw.shape != (b, 1, h, w):
                raise ValueError(
                    f"fg_mask_bchw must be (B,1,H,W) matching ad, got {tuple(fg_mask_bchw.shape)}"
                )
            fg = fg_mask_bchw.clamp(0.0, 1.0)
            bg = 1.0 - fg
        else:
            fg = None
            bg = None

        pattern_gray = rgb_to_grayscale(pattern_rgb_bchw)  # (B,1,H,W)
        if self.use_fg_mask:
            # Gate pattern so conditioning applies mainly on background.
            pattern_gray = pattern_gray * bg

        # Encoder with full-res pattern conditioning.
        if self.pattern_cond_on_gray:
            if self.use_fg_mask:
                x0 = torch.cat([ad_rgb_bchw, fg, pattern_gray], dim=1)
            else:
                x0 = torch.cat([ad_rgb_bchw, pattern_gray], dim=1)
        else:
            x0 = ad_rgb_bchw if not self.use_fg_mask else torch.cat([ad_rgb_bchw, fg], dim=1)
        f0 = self.enc0(x0)  # (B,chs[0],H,W)

        feats: list[torch.Tensor] = [f0]
        x = f0
        for i in range(self.num_down):
            x = self.downs[i](x)  # spatial /2
            x = self.enc_blocks[i](x)
            feats.append(x)

        # Precompute multi-scale pattern condition features aligned to encoder resolutions.
        cond_feats: list[torch.Tensor] = []
        for i in range(self.num_down + 1):
            _, _, hi, wi = feats[i].shape
            p_resized = F.interpolate(pattern_gray if self.pattern_cond_on_gray else pattern_rgb_bchw, size=(hi, wi), mode="bilinear", align_corners=False)
            cond_feats.append(self.cond_projs[i](p_resized))

        # Decoder: start from bottleneck feats[-1]
        x = feats[-1]
        # decoder lists are arranged in reverse i-order (from num_down-1 -> 0)
        fuse_i = 0
        for i in range(self.num_down - 1, -1, -1):
            up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            up = self.up_proj[fuse_i](up)
            skip = feats[i]
            cond = cond_feats[i]
            x = torch.cat([up, skip, cond], dim=1)
            x = self.dec_fuse_blocks[fuse_i](x)
            fuse_i += 1

        return self.out_head(x)


def build_wide_shallow_pattern_unet(
    *,
    base_channels: int = 96,
    max_channels: int = 512,
    num_down: int = 3,
    convs_per_stage: int = 2,
    use_fg_mask: bool = False,
    **kwargs,
) -> WideShallowPatternUNet:
    return WideShallowPatternUNet(
        base_channels=base_channels,
        max_channels=max_channels,
        num_down=num_down,
        convs_per_stage=convs_per_stage,
        use_fg_mask=use_fg_mask,
        **kwargs,
    )


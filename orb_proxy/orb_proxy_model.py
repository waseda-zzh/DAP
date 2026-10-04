import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.act(out)
        return out


class ORBProxyCNN(nn.Module):
    """
    A lightweight CNN regressor that predicts orb_loss (scalar) from a composite/ROI crop.

    Input:
      - image: B x C x H x W (C=3 for RGB, C=4 if we additionally provide a mask channel)

    Output:
      - orb_loss_hat: B (sigmoid constrained to (0,1))
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        num_blocks: int = 3,
        masked_pool_weight: float = 0.8,
    ):
        super().__init__()
        self.masked_pool_weight = float(max(0.0, min(masked_pool_weight, 1.0)))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        channels = base_channels
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        self.blocks = nn.Sequential(*blocks)

        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(channels * 2),
        )

        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

        self.out_act = nn.Sigmoid()

    def forward(self, x):
        """
        Scene-aware + mask-aware regression:
          - Keep full RGB (do not zero background)
          - Use mask as an extra guidance channel
          - Mix masked pooling and global pooling
        """
        # x: B x 4 x H x W (RGB + mask)
        if x.dim() != 4 or x.size(1) < 4:
            raise ValueError(f"Expected x as (B,4,H,W), got {tuple(x.shape)}")

        rgb = x[:, :3]
        m = x[:, 3:4]  # (B,1,H,W) in [0,1]
        x_in = torch.cat([rgb, m], dim=1)

        f = self.stem(x_in)
        f = self.blocks(f)
        f = self.down(f)  # (B,C,Hf,Wf)

        # Mixed pooling at feature resolution:
        # - masked pooling emphasizes patch area attribution
        # - global pooling preserves scene competition/context
        m_rs = F.interpolate(m, size=f.shape[-2:], mode="nearest")
        denom = m_rs.sum(dim=(2, 3), keepdim=False).clamp(min=1e-6)  # (B,1)
        pooled_masked = (f * m_rs).sum(dim=(2, 3), keepdim=False) / denom  # (B,C)
        pooled_global = f.mean(dim=(2, 3), keepdim=False)  # (B,C)
        pooled = self.masked_pool_weight * pooled_masked + (1.0 - self.masked_pool_weight) * pooled_global

        orb_loss_hat = self.head(pooled).squeeze(-1)  # (B,)
        return self.out_act(orb_loss_hat)


def build_orb_proxy(in_channels: int = 4) -> ORBProxyCNN:
    return ORBProxyCNN(in_channels=in_channels)


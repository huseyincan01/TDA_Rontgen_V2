"""V6.0 Faz 2 — Eğitilmiş hafif dedektör modelleri (M5: CenterNet ısı haritası, M6: FCOS-lite).

Omurga: ImageNet-ön-eğitimli ResNet18, maxpool atlanarak stride-2 encoder.
512x512 girdi → 256x256 çıktı (orijinal 1024'e göre stride=4 ⇒ 4px çözünürlük).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

INPUT = 512
STRIDE = 2
HMP = INPUT // STRIDE          # 256


class Encoder(nn.Module):
    """ResNet18 (maxpool atlanmış): c1 (stride2, 64ch, 256x256) + x2 (stride4, 128ch, 128x128)."""

    def __init__(self):
        super().__init__()
        rn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.layer1 = rn.layer1   # stride 2 (maxpool yok)
        self.layer2 = rn.layer2   # stride 4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # 256x256 (stride 2)
        c1 = x
        x2 = self.layer2(self.layer1(x))         # 128x128 (stride 4)
        return c1, x2


class _Decode(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)          # 128 -> 256
        self.fuse = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        c1, x2 = self.enc(x)
        up = self.up(x2)
        return self.fuse(torch.cat([c1, up], dim=1))                 # 256x256, 64ch


class CenterNetHead(nn.Module):
    """M5: ısı haritası (1ch) + kutu yarıçapı (1ch)."""

    def __init__(self):
        super().__init__()
        self.dec = _Decode()
        self.heat = nn.Conv2d(64, 1, 1)
        self.radius = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        f = self.dec(x)
        return self.heat(f), self.radius(f)


class FCOSHead(nn.Module):
    """M6: objectness (1ch) + kutu regresyonu l,t,r,b (4ch)."""

    def __init__(self):
        super().__init__()
        self.dec = _Decode()
        self.obj = nn.Conv2d(64, 1, 1)
        self.box = nn.Conv2d(64, 4, 1)

    def forward(self, x):
        f = self.dec(x)
        return self.obj(f), self.box(f)


def _gaussian_kernel(sigma: float, size: int = 31):
    ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g[None] * g[:, None]
    g /= g.max()
    return g


def build_targets_heat(centers: list, radii: list, device) -> tuple[torch.Tensor, torch.Tensor]:
    """CenterNet ısı haritası hedefleri (HMP x HMP). centers/radii: 512 uzayı (px)."""
    Y = np.zeros((HMP, HMP), dtype=np.float32)
    R = np.zeros((HMP, HMP), dtype=np.float32)
    for (cx, cy), r in zip(centers, radii):
        hx, hy = cx / STRIDE, cy / STRIDE
        sigma = max(1.0, r / STRIDE * 0.3)
        k = _gaussian_kernel(sigma)
        hs = k.shape[0] // 2
        x0, x1 = int(round(hx)) - hs, int(round(hx)) + hs + 1
        y0, y1 = int(round(hy)) - hs, int(round(hy)) + hs + 1
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(HMP, x1), min(HMP, y1)
        kx0, ky0 = sx0 - x0, sy0 - y0
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        Y[sy0:sy1, sx0:sx1] = np.maximum(Y[sy0:sy1, sx0:sx1], k[ky0:ky0 + (sy1 - sy0), kx0:kx0 + (sx1 - sx0)])
        hx_i = int(min(max(int(round(hx)), 0), HMP - 1))
        hy_i = int(min(max(int(round(hy)), 0), HMP - 1))
        R[hy_i, hx_i] = r / STRIDE
    return torch.tensor(Y, device=device), torch.tensor(R, device=device)


def center_focal_loss(pred_heat: torch.Tensor, gt_heat: torch.Tensor, alpha=2.0, beta=4.0):
    pred = torch.sigmoid(pred_heat)
    pos = gt_heat.eq(1.0)
    neg = gt_heat.lt(1.0)
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred.clamp_min(1e-6)) * pos.float()
    neg_loss = -((1 - gt_heat) ** beta) * (pred ** alpha) * torch.log((1 - pred).clamp_min(1e-6)) * neg.float()
    n = pos.sum().clamp_min(1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def _build_box_targets(centers: list, radii: list):
    obj = np.zeros((HMP, HMP), dtype=np.float32)
    box = np.zeros((HMP, HMP, 4), dtype=np.float32)
    for (cx, cy), r in zip(centers, radii):
        x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
        gx0, gx1 = int(np.floor(x0 / STRIDE)), int(np.ceil(x1 / STRIDE))
        gy0, gy1 = int(np.floor(y0 / STRIDE)), int(np.ceil(y1 / STRIDE))
        gx0, gx1 = max(0, gx0), min(HMP, gx1)
        gy0, gy1 = max(0, gy0), min(HMP, gy1)
        for gy in range(gy0, gy1):
            for gx in range(gx0, gx1):
                px, py = gx * STRIDE + STRIDE / 2, gy * STRIDE + STRIDE / 2
                l, t = px - x0, py - y0
                rr, b = x1 - px, y1 - py
                if l > 0 and t > 0 and rr > 0 and b > 0:
                    obj[gy, gx] = 1.0
                    box[gy, gx] = (l / STRIDE, t / STRIDE, rr / STRIDE, b / STRIDE)
    return obj, box

"""2-Kanallı (H0, H1) Persistence Image kesitleri için Küçük Zon CNN mimarisi."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def vector_to_2channel_image(X: np.ndarray) -> np.ndarray:
    """800 boyutlu TDA vektörünü (H0: 400, H1: 400) 2 kanallı (2, 20, 20) tensöre dönüştürür.
    
    Args:
        X: (N, 800) veya (800,) boyutunda dizi.
        
    Returns:
        (N, 2, 20, 20) boyutunda float32 dizi.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    single = False
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
        single = True

    n_samples = X_arr.shape[0]
    h0 = X_arr[:, :400].reshape(n_samples, 20, 20)
    h1 = X_arr[:, 400:].reshape(n_samples, 20, 20)
    stacked = np.stack([h0, h1], axis=1)  # (N, 2, 20, 20)

    if single:
        return stacked[0]
    return stacked


class SmallZoneCNN(nn.Module):
    """(2, 20, 20) girişli hafif konvolüsyonel sinir ağı (Küçük Zon CNN).
    
    Persistence Image'ların 2B topolojik doğum-yaşam (birth-persistence)
    lokal geometrisini konvolüsyon filtreleriyle işler.
    """

    def __init__(self, in_channels: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        # 1. Konvolüsyon Bloğu: 20x20 -> 10x10
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 2. Konvolüsyon Bloğu: 10x10 -> 5x5
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 3. Uyarlamalı Havuzlama: 5x5 -> 3x3
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        # 4. Sınıflandırıcı Başlık
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Giriş: (B, 2, 20, 20), Çıkış: (B,) ikili logit değerleri."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid uygulanmış pozitif sınıf (1) olasılıkları [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

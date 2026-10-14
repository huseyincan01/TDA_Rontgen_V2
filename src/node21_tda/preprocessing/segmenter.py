"""Akciğer ve 5 Anatomik Bölge Segmentasyon Modülü (TorchXRayVision PSPNet).

Bu modül, göğüs röntgenlerini TorchXRayVision PSPNet modeli ile segmentasyon işlemine tabi tutar
ve insan anatomisine (Sağ 3 lob, Sol 2 lob) uygun olarak 5 anatomik bölge üretir:
- Bölge 1: Sağ Üst (Apeks / Klavikula)
- Bölge 2: Sağ Orta (Orta Lob / Hiler Seviye)
- Bölge 3: Sağ Alt (Alt Lob / Diyafram)
- Bölge 4: Sol Üst (Apeks / Aort Kavsi / Sol Üst Lob)
- Bölge 5: Sol Alt (Sol Alt Lob / Kardiyak Kenar / Diyafram)
"""

import time
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv


# 5 Anatomik Bölge Tanımları
ZONE_METADATA = {
    1: {
        "name": "1. Sağ Üst Bölge (Apeks / Klavikula)",
        "short": "Sağ Üst (R-Upper)",
        "lung": "right",
        "level": "upper",
        "file_prefix": "zone_1_sag_ust",
        "color": [0.95, 0.30, 0.15]   # Kırmızı / Turuncu
    },
    2: {
        "name": "2. Sağ Orta Bölge (Orta Lob / Hiler Seviye)",
        "short": "Sağ Orta (R-Mid)",
        "lung": "right",
        "level": "middle",
        "file_prefix": "zone_2_sag_orta",
        "color": [0.95, 0.75, 0.10]   # Sarı / Altın
    },
    3: {
        "name": "3. Sağ Alt Bölge (Alt Lob / Diyafram)",
        "short": "Sağ Alt (R-Lower)",
        "lung": "right",
        "level": "lower",
        "file_prefix": "zone_3_sag_alt",
        "color": [0.15, 0.85, 0.35]   # Yeşil
    },
    4: {
        "name": "4. Sol Üst Bölge (Apeks / Aort Kavsi / Sol Üst Lob)",
        "short": "Sol Üst (L-Upper)",
        "lung": "left",
        "level": "upper",
        "file_prefix": "zone_4_sol_ust",
        "color": [0.65, 0.25, 0.90]   # Mor
    },
    5: {
        "name": "5. Sol Alt Bölge (Sol Alt Lob / Kardiyak Kenar / Diyafram)",
        "short": "Sol Alt (L-Lower)",
        "lung": "left",
        "level": "lower",
        "file_prefix": "zone_5_sol_alt",
        "color": [0.10, 0.65, 0.98]   # Canlı Mavi
    }
}


class PSPNet5ZoneSegmenter:
    """TorchXRayVision PSPNet tabanlı 5-Bölge Akciğer Segmentasyon Motoru."""

    def __init__(self, device: Optional[str] = None, threshold: float = 0.5):
        """Segmentasyon modelini GPU (veya CPU) üzerinde başlatır.
        
        Args:
            device: 'cuda' veya 'cpu'. Belirtilmezse otomatik seçilir.
            threshold: Akciğer olasılık maskeleri için eşik değeri (> 0.5).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # TorchXRayVision PSPNet modeli
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(self.device)
        self.model.eval()
        self.threshold = threshold

        # Model kanal indeksleri
        self.RIGHT_LUNG_CH = 5  # Sağ Akciğer
        self.LEFT_LUNG_CH = 4   # Sol Akciğer

    def preprocess_image(self, img_array: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
        """Görüntüyü [-1024, 1024] aralığına normalize eder ve kare padding ekler.

        Args:
            img_array: (H, W) boyutunda 2D numpy dizisi.

        Returns:
            t_in: (1, 1, L, L) boyutunda GPU tensörü.
            pad_params: (H, W, L, pad_top, pad_bottom, pad_left, pad_right)
        """
        H, W = img_array.shape
        L = max(H, W)

        pad_top = (L - H) // 2
        pad_bottom = L - H - pad_top
        pad_left = (L - W) // 2
        pad_right = L - W - pad_left

        # Dinamik aralık normalizasyonu (8-bit, 12-bit veya 16-bit tespiti)
        max_val_found = float(img_array.max())
        if max_val_found > 4095.0:
            scale_max = 65535.0
        elif max_val_found > 255.0:
            scale_max = 4095.0
        else:
            scale_max = 255.0

        # [-1024.0, 1024.0] aralığına normalize et
        norm_img = (2.0 * (img_array.astype(np.float32) / scale_max) - 1.0) * 1024.0

        # Simetrik kare padding (arkaplanı -1024 ile doldur)
        norm_padded = np.pad(
            norm_img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=-1024.0
        )

        t_in = torch.from_numpy(norm_padded).unsqueeze(0).unsqueeze(0).float().to(self.device)
        pad_params = (H, W, L, pad_top, pad_bottom, pad_left, pad_right)
        return t_in, pad_params

    def segment(self, img_array: np.ndarray) -> Dict:
        """Görüntüden 5 anatomik bölge haritasını üretir.

        Args:
            img_array: (H, W) 2D uint16/uint8/float numpy dizisi.

        Returns:
            dict: {
                "zone_map": (H, W) uint8 (1..5 değerleri, arka plan 0),
                "right_lung": (H, W) bool maske,
                "left_lung": (H, W) bool maske,
                "inference_time_ms": float,
                "right_stats": dict (ymin, ymax, step),
                "left_stats": dict (ymin, ymax, step)
            }
        """
        if len(img_array.shape) == 3:
            img_array = img_array[0]

        H, W = img_array.shape
        t_in, (H, W, L, pad_top, pad_bottom, pad_left, pad_right) = self.preprocess_image(img_array)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            out = self.model(t_in)
            probs = torch.sigmoid(out)
            # 512x512'lik çıktı tensörünü bilinear interpolasyonla (L, L)'ye büyüt
            probs_sq = F.interpolate(probs, size=(L, L), mode="bilinear", align_corners=False)[0]
            # Padding'i geri kırparak orijinal (H, W) boyutuna döndür
            probs_orig = probs_sq[:, pad_top:pad_top+H, pad_left:pad_left+W].cpu().numpy()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        duration_ms = (time.time() - t0) * 1000.0

        # Akciğer maskeleri (> 0.5)
        right_lung = probs_orig[self.RIGHT_LUNG_CH] > self.threshold
        left_lung = probs_orig[self.LEFT_LUNG_CH] > self.threshold

        zone_map = np.zeros((H, W), dtype=np.uint8)
        y_grid = np.indices((H, W))[0]

        # -------------------------------------------------------------
        # A) SAĞ AKCİĞER (3 Seviyeli Bölme - Sağ 3 Lob Anatomisi)
        # -------------------------------------------------------------
        r_stats = {}
        r_indices = np.where(right_lung)
        if len(r_indices[0]) > 0:
            r_ymin, r_ymax = int(np.min(r_indices[0])), int(np.max(r_indices[0]))
            r_step = (r_ymax - r_ymin) / 3.0
            r_cut1 = r_ymin + r_step
            r_cut2 = r_ymin + 2.0 * r_step

            zone_map[right_lung & (y_grid < r_cut1)] = 1                     # 1. Sağ Üst
            zone_map[right_lung & (y_grid >= r_cut1) & (y_grid < r_cut2)] = 2 # 2. Sağ Orta
            zone_map[right_lung & (y_grid >= r_cut2)] = 3                     # 3. Sağ Alt

            r_stats = {
                "ymin": r_ymin, "ymax": r_ymax, "step": r_step,
                "cut1": r_cut1, "cut2": r_cut2
            }

        # -------------------------------------------------------------
        # B) SOL AKCİĞER (2 Seviyeli Bölme - Sol 2 Lob Anatomisi)
        # -------------------------------------------------------------
        l_stats = {}
        l_indices = np.where(left_lung)
        if len(l_indices[0]) > 0:
            l_ymin, l_ymax = int(np.min(l_indices[0])), int(np.max(l_indices[0]))
            l_step = (l_ymax - l_ymin) / 2.0
            l_cut = l_ymin + l_step

            zone_map[left_lung & (y_grid < l_cut)] = 4   # 4. Sol Üst
            zone_map[left_lung & (y_grid >= l_cut)] = 5  # 5. Sol Alt

            l_stats = {
                "ymin": l_ymin, "ymax": l_ymax, "step": l_step,
                "cut": l_cut
            }

        return {
            "zone_map": zone_map,
            "right_lung": right_lung,
            "left_lung": left_lung,
            "inference_time_ms": duration_ms,
            "right_stats": r_stats,
            "left_stats": l_stats
        }

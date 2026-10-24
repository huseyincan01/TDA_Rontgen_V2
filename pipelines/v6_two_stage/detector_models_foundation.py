"""V6.6 — Foundation X-Ray Backbone Stage-1: CXR Vakıf Omurgası + FPN + ASPP + CenterNet.

Neden
-----
M5 (V6.0) omurgası ImageNet (kedi/köpek) ile öneğitilmiş ResNet18 idi. Bu betik
omurgayı, 800.000+ gerçek göğüs röntgeniyle (NIH ChestX-ray14, CheXpert, PadChest,
MIMIC-CXR, OpenI, RSNA, VinDr) eğitilmiş **torchxrayvision** vakıf modelleriyle
değiştirir.

Omurga seçimi
-------------
`resnet50-res512-all` (pc-nih-rsna-siim-vin, 512x512 native) SEÇİLDİ:
  * native girdi 512x512 = bizim kanonik girdimizle BİREBİR (yeniden örnekleme yok
    ⇒ küçük nodüllerin (>4px) çözünürlüğü korunur; densenet121-res224-all 512→224
    küçültmesi isterdi ve bu tam da kaçırdığımız küçük nodülleri silerdi),
  * temiz C2/C3/C4/C5 (1/4, 1/8, 1/16, 1/32) piramidi verir.
`densenet121-res224-all` `--backbone` ile desteklenir (karşılaştırma için).

KRİTİK GİRDİ ÖLÇEĞİ
-------------------
torchxrayvision omurgaları tek kanallı girdiyi [0,1] değil **[-1024, +1024]**
bekler (`xrv.utils.normalize(img, maxval=255)`). `foundation_input()` bunu yapar.

Çıktı sözleşmesi M5 ile AYNI
----------------------------
512 girdi → 256x256 ısı haritası (1024'e stride-4). Bu sayede aday çıkarımı,
değerlendirme ve 2. aşama entegrasyonu **birebir aynı kodla** çalışır.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT = 512
STRIDE = 2
HMP = INPUT // STRIDE          # 256

# torchxrayvision normalize ölçeği
XRV_MAX = 1024.0


# --------------------------------------------------------------------------- #
# Girdi yardımcıları
# --------------------------------------------------------------------------- #
def foundation_input(canon_u8: np.ndarray, flip: bool = False) -> torch.Tensor:
    """[0,255] kanonik uint8 (512x512) → (1,512,512) float, [-1024, +1024] aralığı."""
    a = np.asarray(canon_u8, dtype=np.float32)
    if flip:
        a = np.ascontiguousarray(a[:, ::-1])
    a = (a / 255.0) * 2.0 - 1.0        # [-1, 1]
    a = a * XRV_MAX                     # [-1024, 1024]
    return torch.from_numpy(a).unsqueeze(0)


def foundation_input_aug(canon_u8: np.ndarray, gamma: float = 1.0,
                         scale: float = 1.0) -> np.ndarray:
    """Yoğunluk/gama + rastgele ölçek augmentasyonu (512x512 uint8 döner)."""
    from PIL import Image

    a = np.asarray(canon_u8, dtype=np.float32)
    if gamma != 1.0:
        a = np.clip((a / 255.0) ** gamma * 255.0, 0, 255)
    if scale != 1.0:
        s = int(round(INPUT * scale))
        im = Image.fromarray(a.astype(np.uint8)).resize((s, s), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        out = np.zeros((INPUT, INPUT), dtype=np.float32)
        if s >= INPUT:                                   # merkezden kırp
            o = (s - INPUT) // 2
            out = arr[o:o + INPUT, o:o + INPUT]
        else:                                            # merkeze yerleştir
            o = (INPUT - s) // 2
            out[o:o + s, o:o + s] = arr
        a = out
    return np.clip(a, 0, 255).astype(np.uint8)


def scale_centers(centers: list, radii: list, scale: float):
    """Ölçek augmentasyonuna karşılık gelen merkez/yarıçap dönüşümü (512 uzayı)."""
    c = INPUT / 2.0
    cs = [((cx - c) * scale + c, (cy - c) * scale + c) for cx, cy in centers]
    rs = [r * scale for r in radii]
    return cs, rs


# --------------------------------------------------------------------------- #
# Boyun: ASPP + FPN
# --------------------------------------------------------------------------- #
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — C5 üzerinde (küçük nodül + dev kitle aynı anda)."""

    def __init__(self, in_ch: int, branch_ch: int = 64, out_ch: int = 256,
                 rates: tuple = (1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
            for r in rates
        ])
        # Global-pool dalında BN YOK: 1x1 uzamsalda BN anlamsız ve batch=1'de çöker.
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)
        return self.project(torch.cat(feats + [g], dim=1))


class FPNNeck(nn.Module):
    """C2..C5 → P2 (1/4). ASPP C5'e uygulanır, sonra yukarı doğru birleştirme."""

    def __init__(self, in_chs=(256, 512, 1024, 2048), fpn_ch: int = 256):
        super().__init__()
        self.c5_proj = nn.Sequential(
            nn.Conv2d(in_chs[3], fpn_ch, 1, bias=False),
            nn.BatchNorm2d(fpn_ch), nn.ReLU(inplace=True))
        self.aspp = ASPP(fpn_ch, branch_ch=64, out_ch=fpn_ch)
        self.lat = nn.ModuleList([
            nn.Conv2d(c, fpn_ch, 1) for c in in_chs[:3]
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
                          nn.BatchNorm2d(fpn_ch), nn.ReLU(inplace=True))
            for _ in range(3)
        ])

    def forward(self, c2, c3, c4, c5):
        p5 = self.aspp(self.c5_proj(c5))
        p4 = self.smooth[2](self.lat[2](c4) + F.interpolate(p5, size=c4.shape[-2:],
                                                            mode="nearest"))
        p3 = self.smooth[1](self.lat[1](c3) + F.interpolate(p4, size=c3.shape[-2:],
                                                            mode="nearest"))
        p2 = self.smooth[0](self.lat[0](c2) + F.interpolate(p3, size=c2.shape[-2:],
                                                            mode="nearest"))
        return p2, p3, p4, p5


# --------------------------------------------------------------------------- #
# Tam model
# --------------------------------------------------------------------------- #
class FoundationCenterNet(nn.Module):
    """CXR vakıf omurgası + FPN/ASPP + CenterNet kafası (M5 ile aynı çıktı ölçeği)."""

    def __init__(self, weights: str = "resnet50-res512-all", fpn_ch: int = 256,
                 head_ch: int = 64, cache_dir: str = None):
        """cache_dir: xrv ağırlık önbelleği. Docker'da imaja gömülü dizine
        (ör. /opt/xrv_weights) işaret edilir ⇒ --network=none altında İNDİRME OLMAZ."""
        super().__init__()
        import torchxrayvision as xrv

        if weights.startswith("resnet50"):
            self.bb = xrv.models.ResNet(weights=weights, cache_dir=cache_dir).model
            self.fpn = FPNNeck((256, 512, 1024, 2048), fpn_ch)
            self._top_blocks = ("layer3", "layer4")
            self.is_densenet = False
        else:
            self.bb = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir)
            self.fpn = FPNNeck((256, 512, 1024, 1024), fpn_ch)
            self._top_blocks = ("denseblock3", "denseblock4")
            self.is_densenet = True

        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_ch, head_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_ch), nn.ReLU(inplace=True))
        self.heat = nn.Conv2d(head_ch, 1, 1)
        self.radius = nn.Conv2d(head_ch, 1, 1)
        # True ise yarıçap kafası DETACH edilmiş özelliklerden öğrenir ⇒ gradyanı
        # paylaşılan gövdeye / ısı haritası dalına ASLA geri akmaz (V6.6 denetimi:
        # radius kaybı ağırlığı 0,1 iken kafa çöküyordu; bu bayrak omurgayı korur).
        self.radius_detach = False

        # Kafa sıfır-bias başlangıcı (CenterNet pratiği): ısı haritası prior'u düşük
        nn.init.constant_(self.heat.bias, -2.19)

    # ------------------------------------------------------------------ omurga
    def backbone_features(self, x):
        if self.is_densenet:
            f = self.bb.features
            x = f.relu0(f.norm0(f.conv0(x)))
            x = f.pool0(x)
            c2 = f.denseblock1(x)
            x = f.transition1(c2)
            c3 = f.denseblock2(x)
            x = f.transition2(c3)
            c4 = f.denseblock3(x)
            x = f.transition3(c4)
            c5 = f.denseblock4(x)
            return c2, c3, c4, c5
        b = self.bb
        x = b.relu(b.bn1(b.conv1(x)))
        c2 = b.layer1(b.maxpool(x))
        c3 = b.layer2(c2)
        c4 = b.layer3(c3)
        c5 = b.layer4(c4)
        return c2, c3, c4, c5

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone_features(x)
        p2, _, _, _ = self.fpn(c2, c3, c4, c5)          # 1/4 → 128x128 @512
        f = self.fuse(p2)
        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        fr = f.detach() if self.radius_detach else f
        return self.heat(f), self.radius(fr)            # 256x256

    # ------------------------------------------------- forgetting koruması
    def set_bn_eval(self):
        """Omurganın BN running istatistiklerini dondur (800k CXR istatistiği bozulmasın)."""
        for m in self.bb.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = True

    def freeze_backbone(self):
        for p in self.bb.parameters():
            p.requires_grad = False

    def unfreeze_top(self):
        """Yalnız omurganın üst bloklarını (layer3/layer4 veya denseblock3/4) açar."""
        for name in self._top_blocks:
            mod = getattr(self.bb, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True

    def param_groups(self, lr_backbone: float, lr_head: float):
        bb_params = [p for p in self.bb.parameters() if p.requires_grad]
        head_params = [p for n, p in self.named_parameters() if not n.startswith("bb.")]
        groups = [{"params": head_params, "lr": lr_head, "name": "head"}]
        if bb_params:
            groups.append({"params": bb_params, "lr": lr_backbone, "name": "backbone"})
        return groups

    def trainable_summary(self) -> dict:
        bb = sum(p.numel() for p in self.bb.parameters() if p.requires_grad)
        hd = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("bb."))
        return {"backbone_trainable": int(bb),
                "head_trainable": int(hd),
                "total": int(sum(p.numel() for p in self.parameters()))}


# --------------------------------------------------------------------------- #
# Aday çıkarımı (M5 sözleşmesiyle birebir)
# --------------------------------------------------------------------------- #
def _heat_candidates(heat: np.ndarray, shape, top: int = 2000, thr: float = 0.01):
    from scipy import ndimage as ndi
    maxf = ndi.maximum_filter(heat, size=3)
    ys, xs = np.where((heat >= maxf) & (heat > thr))
    sc = heat[ys, xs]
    order = np.argsort(-sc)[:top]
    H, W = shape
    return [(float((xs[k] * STRIDE + STRIDE / 2) * W / INPUT),
             float((ys[k] * STRIDE + STRIDE / 2) * H / INPUT), float(sc[k])) for k in order]


@torch.no_grad()
def predict(model: FoundationCenterNet, canon_u8: np.ndarray, shape, device,
            tta: bool = True, top: int = 2000):
    """Kanonik [0,255] görüntüden aday listesi. tta=True → yatay çevirme ortalaması.

    `canon_u8`: 512x512 uint8. `shape`: koordinatların döndürüleceği görüntü boyutu.
    """
    x = foundation_input(canon_u8, False).unsqueeze(0).to(device)
    h, _ = model(x)
    heat = torch.sigmoid(h)[0, 0].cpu().numpy()
    if tta:
        xf = foundation_input(canon_u8, True).unsqueeze(0).to(device)
        hf, _ = model(xf)
        heat_f = torch.sigmoid(hf)[0, 0].cpu().numpy()[:, ::-1]
        heat = (heat + heat_f) / 2.0
    return _heat_candidates(heat, shape, top=top)


@torch.no_grad()
def predict_radius(model, canon_u8: np.ndarray, shape, device):
    """Yarıçap kafası çıktısı (aday kutusu için, teşhis amaçlı)."""
    x = foundation_input(canon_u8, False).unsqueeze(0).to(device)
    _, r = model(x)
    return r[0, 0].cpu().numpy()


# --------------------------------------------------------------------------- #
# V6 hattıyla birebir uyumlu tahmin (ev._predict_heat_tta sözleşmesi)
# --------------------------------------------------------------------------- #
def canon_to_512(canon: np.ndarray) -> np.ndarray:
    """1024x1024 float kanonik ([0,255]) → 512x512 uint8."""
    from PIL import Image
    return np.array(Image.fromarray(np.clip(canon, 0, 255).astype(np.uint8)).resize(
        (INPUT, INPUT), Image.Resampling.BILINEAR))


@torch.no_grad()
def predict_tta(model, canon, device, top: int = 1000, thr: float = 0.01, shape=None):
    """Isı haritası düzeyinde TTA (ev._predict_heat_tta ile aynı çıktı sözleşmesi).

    `canon`: kanonik float görüntü. `shape`: koordinatların ölçekleneceği (H, W);
    verilmezse (1024, 1024) varsayılır (NODE21). VinDr gibi farklı boyutlarda
    MUTLAKA geçilmelidir, aksi hâlde GT ile ~%1 ölçek uyuşmazlığı olur.
    """
    import candidates_classical as cc
    model.eval()
    x = foundation_input(canon_to_512(canon), False).unsqueeze(0).to(device)
    xf = torch.flip(x, dims=[3])
    h1 = torch.sigmoid(model(x)[0])[0, 0]
    h2 = torch.flip(torch.sigmoid(model(xf)[0])[0, 0], dims=[1])
    heat = ((h1 + h2) / 2).cpu().numpy()
    from scipy import ndimage as ndi
    maxf = ndi.maximum_filter(heat, size=3)
    ys, xs = np.where((heat >= maxf) & (heat > thr))
    scores = heat[ys, xs]
    order = np.argsort(-scores)
    H, W = shape if shape is not None else (1024, 1024)
    cands = [(float((xs[k] * STRIDE + STRIDE / 2) * W / INPUT),
              float((ys[k] * STRIDE + STRIDE / 2) * H / INPUT),
              float(scores[k])) for k in order]
    return cc._nms(cands, min_dist=20, top=top)


@torch.no_grad()
def predict_single(model, canon, device, top: int = 1000, thr: float = 0.01, shape=None):
    """TTA'sız tahmin (aynı sözleşme)."""
    import candidates_classical as cc
    model.eval()
    x = foundation_input(canon_to_512(canon), False).unsqueeze(0).to(device)
    heat = torch.sigmoid(model(x)[0])[0, 0].cpu().numpy()
    from scipy import ndimage as ndi
    maxf = ndi.maximum_filter(heat, size=3)
    ys, xs = np.where((heat >= maxf) & (heat > thr))
    scores = heat[ys, xs]
    order = np.argsort(-scores)
    H, W = shape if shape is not None else (1024, 1024)
    cands = [(float((xs[k] * STRIDE + STRIDE / 2) * W / INPUT),
              float((ys[k] * STRIDE + STRIDE / 2) * H / INPUT),
              float(scores[k])) for k in order]
    return cc._nms(cands, min_dist=20, top=top)


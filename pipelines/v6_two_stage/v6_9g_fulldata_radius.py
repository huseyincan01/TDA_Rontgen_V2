"""V6.9g — Tam-veri gönderim modelinin dinamik yarıçap kafası (R>0 maske düzeltmesi).

AMAÇ
----
V6.8'de bilimsel modelde (M5_foundation) R>0 maskesiyle eğitilen dinamik yarıçap kafası
sabit 56 px kutuyu geçti. Bu betik, aynı düzeltmeyi **tam-veri gönderim modeline**
(M5_foundation_fulldata.pt) uygular.

KORUMA (Tuzak A'ya saygı)
-------------------------
* Omurga + FPN + fuse + ısı haritası ağırlıkları **%100 DONUK** — hiçbir gradyan akmaz.
* Yalnızca yepyeni, 2 katmanlı `RadiusHead` (64→32→1) eğitilir.
* Kayıp: smooth-L1, **yalnız R>0 (merkez) piksellerinde**, ağırlık 1,0 — V6.8 ile birebir.
* Takvim: 6 epoch, batch 4, lr 1e-3, weight_decay 1e-5, seed 42.

VERİ
----
Tam-veri 1:1: tüm splitlerdeki nodüllü hastalar (1.134) + eşit sayıda temiz kontrol (1.134).
Bu, V6.7 tam-veri eğitiminin kullandığı verinin AYNISIDIR (gönderim modelinin gördüğü veri).

NOT (dürüstlük)
---------------
Tam-veri modeli NODE21'de değerlendirilemez (tüm veri eğitimde). Bu kafanın kalitesi yalnızca
tarif transferiyle güvence altına alınır: aynı maske düzeltmesinin bilimsel modelde kilitli
TEST'te kazandığı V6.9f'de doğrulandı. Buradaki çıktı, doğrudan ölçülmez — beyan edilir.

Kullanım: python pipelines/v6_two_stage/v6_9g_fulldata_radius.py [--epochs 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_7_train_fulldata import _aug_batch, _prep512  # noqa: E402

WORK = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9g_fulldata_radius.json"
SEED, BATCH, EPOCHS, LR = 42, 4, 6, 1e-3


class RadiusHead(nn.Module):
    """V6.8 ile birebir: 64→32→1, ReLU, 3x3 + 1x1."""

    def __init__(self, in_ch=64, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))

    def forward(self, f):
        return self.net(f)


def _feats(model, x):
    """forward()'daki `f` ile birebir aynı 256x256 özellik (V6.8 kopyası)."""
    c2, c3, c4, c5 = model.backbone_features(x)
    p2v, _, _, _ = model.fpn(c2, c3, c4, c5)
    f = model.fuse(p2v)
    return F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9g — tam-veri gönderim modeli dinamik yarıçap kafası (R>0 maske)", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- veri: tüm splitler, 1:1 (V6.7 ile aynı)
    splits = common.split_patients()
    all_pids = list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    pos = [p for p in all_pids if common.load_gt(p)]
    neg_all = [p for p in all_pids if not common.load_gt(p)]
    neg = list(np.random.RandomState(SEED).permutation(neg_all)[:len(pos)])
    print(f"  veri: {len(pos)} nodüllü + {len(neg)} kontrol = {len(pos) + len(neg)}", flush=True)

    print("[1] RAM önbelleği...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    cache, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pid, c512, cen, rad in pool.map(_prep512, list(pos) + list(neg), chunksize=8):
            cache[pid] = (c512, cen, rad)
    items = [(p, True) for p in pos if p in cache] + [(p, False) for p in neg if p in cache]
    print(f"    önbellek: {len(cache)} ({time.time() - t0:.0f}s)", flush=True)

    print("[2] Model yükleniyor (omurga DONUK)...", flush=True)
    model = load_foundation("M5_foundation_fulldata", device, subdir=True)
    for p in model.parameters():
        p.requires_grad = False
    model.radius_detach = True
    print(f"    donuk parametre: {sum(1 for p in model.parameters() if not p.requires_grad)}",
          flush=True)

    head = RadiusHead(64, 32).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-5)
    print(f"[3] Yarıçap kafası eğitimi (R>0, ağırlık 1,0, {args.epochs} epoch)...", flush=True)

    hist = []
    for ep in range(1, args.epochs + 1):
        head.train()
        rng = np.random.RandomState(SEED + ep)
        order = rng.permutation(len(items))
        tot, nb, t0 = 0.0, 0, time.time()
        for s in range(0, len(order), BATCH):
            sel = [items[i] for i in order[s:s + BATCH]]
            if len(sel) < 2:
                continue
            x, Y, R = _aug_batch(sel, cache, rng, device)
            with torch.no_grad():
                f = _feats(model, x)
            pred = head(f)
            pos_mask = (R > 0).float()          # yalnız merkez pikseli (V6.8 B kolu)
            loss = (F.smooth_l1_loss(pred, R, reduction="none") * pos_mask).sum() \
                / pos_mask.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        hist.append({"epoch": ep, "loss": round(tot / max(1, nb), 4),
                     "seconds": round(time.time() - t0, 1)})
        print(f"    epoch {ep}/{args.epochs} | L1 {hist[-1]['loss']:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)

    head_path = WORK / "radius_head_weight1_center_only_fulldata.pt"
    WORK.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), head_path)
    print(f"[4] Kaydedildi: {head_path}", flush=True)

    rep = {
        "experiment": "V6.9g-fulldata-radius-head",
        "purpose": "Tam-veri gönderim modeline R>0 maske düzeltmesi (V6.8 B kolu) transferi",
        "protection": "omurga+FPN+fuse+ısı haritası %100 donuk; yalnız yeni 2 katmanlı RadiusHead eğitildi",
        "data": f"tüm splitler 1:1 ({len(pos)}+{len(neg)}) — V6.7 tam-veri eğitimiyle aynı",
        "recipe": "smooth-L1 ağırlık 1,0 · maske R>0 · lr 1e-3 · batch 4 · seed 42",
        "history": hist,
        "head_ckpt": "submission_fulldata/radius_head_weight1_center_only_fulldata.pt",
        "honest_note": ("Tam-veri modeli NODE21'de değerlendirilemez (tüm veri eğitimde); "
                        "kalite yalnızca tarif transferiyle güvence altına alınır (V6.9f'de "
                        "bilimsel modelde kilitli test kazancı doğrulandı)."),
    }
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()

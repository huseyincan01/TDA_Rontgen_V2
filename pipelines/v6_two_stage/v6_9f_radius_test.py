"""V6.9f — Dinamik yarıçap kafasının KİLİTLİ TEST doğrulaması (sabit 56 px'e karşı).

SORU
----
V6.8'de, R>0 maskesiyle eğitilen dinamik yarıçap kafası **val**'de sabit 56 px kutuyu
geçti (rank_metric 0,9185 > 0,9131). Bu betik, aynı kafanın **kilitli test** setinde de
kazanç sağladığını teyit eder.

ÖNEMLİ — TEST BÜTÇESİ BEYANI
---------------------------
NODE21 test bölmesi proje tarihinde İKİ kez kullanıldı (V6.4 mührü 0,9876 · V6.9 mührü
0,8842/0,9645). Bu koşu **ÜÇÜNCÜ** kullanımdır. Gerekçe: val'de seçilen bir iyileştirmenin
(dinamik kutu) gönderime alınmadan ÖNCE tek-atımlık doğrulanmasıdır — hiçbir hiperparametre
test üzerinde seçilmez, seçim val'de zaten yapılmıştır. Bu beyan, raporda ve commit mesajında
açıkça yer alır.

YÖNTEM
------
Aynı kilitli protokol (3 ölçek + flip, 3×3 tepe, NMS 8 px, top-50), yalnız kutu geometrisi
değişir: sabit 28 px yarı-kenar (56 px kutu) vs öğrenilmiş yarıçap. Isı haritası DEĞİŞMEZ
⇒ aday merkezleri/skorlar ve hasta AUC birebir aynı kalır; değişen tek şey IoU eşleşmesi.

Kullanım: python pipelines/v6_two_stage/v6_9f_radius_test.py [--split test]
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import (dist_nms, heat_maps, official_metric,  # noqa: E402
                                     pick_peaks)

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
WORK = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "submission_fulldata"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_9f_radius_test.json"
SCALES, WIN, NMS_PX, TOPK = (0.9, 1.0, 1.1), 3, 8, 50
BOX_HALF = 28.0                     # sabit 56×56 kutu


class RadiusHead(nn.Module):
    """V6.8'in donuk-özellik yarıçap kafası (state_dict ile birebir uyumlu)."""

    def __init__(self, in_ch=64, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1))

    def forward(self, f):
        return self.net(f)


def _pass(model, pids, device, tag):
    flat, gts, t0 = {}, {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        flat[pid] = [(float(cx), float(cy), float(s), float(r)) for cx, cy, s, r in cs]
        gts[pid] = [(g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
                    for g in common.load_gt(pid)]
        if (i + 1) % 200 == 0:
            print(f"    {tag} {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    return flat, gts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9f — dinamik yarıçap KİLİTLİ TEST doğrulaması", flush=True)
    print("  ⚠ ÜÇÜNCÜ test kullanımı — val'de seçilmiş iyileştirmenin tek-atımlık doğrulaması",
          flush=True)

    model = load_foundation("M5_foundation", device)
    head_path = WORK / "radius_head_weight1_center_only.pt"
    if not head_path.exists():
        raise SystemExit(f"yarıçap kafası yok: {head_path}")
    head = RadiusHead(64, 32).to(device)
    head.load_state_dict(torch.load(head_path, map_location=device))
    head.eval()
    model.radius = head
    model.radius_detach = True
    for p in model.parameters():
        p.requires_grad = False
    print(f"  model=M5_foundation + {head_path.name} (R>0 maske, donuk özellik)", flush=True)

    pids = list(common.split_patients()[args.split])
    flat, gts = _pass(model, pids, device, args.split.upper())
    is_nod = {p: len(gts[p]) > 0 for p in pids}
    n_les = sum(len(g) for g in gts.values())

    f = [(s, pid, cx, cy, r) for pid in pids for cx, cy, s, r in flat[pid]]
    fixed = official_metric(f, len(pids), n_les, gts, is_nod, pids, box_half=BOX_HALF)
    dyn = official_metric(f, len(pids), n_les, gts, is_nod, pids, use_radius=True)
    rs = np.array([r for pid in pids for *_, r in flat[pid]], np.float32)

    rep = {
        "experiment": "V6.9f-radius-test",
        "test_budget_declaration": "ÜÇÜNCÜ test kullanımı — val'de seçilmiş iyileştirmenin tek-atımlık doğrulaması; test üzerinde hiçbir seçim yapılmadı.",
        "split": args.split, "n_patients": len(pids), "n_lesions": int(n_les),
        "radius_head": "submission_fulldata/radius_head_weight1_center_only.pt (V6.8, R>0 maske)",
        "fixed_56px": fixed, "dynamic_radius": dyn,
        "predicted_r1024": {"p10": round(float(np.percentile(rs, 10)), 2),
                            "p50": round(float(np.percentile(rs, 50)), 2),
                            "p90": round(float(np.percentile(rs, 90)), 2)},
        "delta_rank": round((dyn["rank_metric"] or 0) - (fixed["rank_metric"] or 0), 4),
        "note": "Isı haritası değişmez ⇒ AUC iki kolda da aynı olmalı (0,9446 test).",
    }
    print(f"\n  sabit 56px   rank={fixed['rank_metric']} AUC={fixed['auc']} "
          f"S@.25={fixed['sens']['0.25']}", flush=True)
    print(f"  dinamik      rank={dyn['rank_metric']} AUC={dyn['auc']} "
          f"S@.25={dyn['sens']['0.25']}", flush=True)
    print(f"  Δ rank_metric = {rep['delta_rank']:+.4f} | r1024 medyan "
          f"{rep['predicted_r1024']['p50']} px", flush=True)
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()

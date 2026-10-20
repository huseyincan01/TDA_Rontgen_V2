"""V6.0 — Kaçan Nodüllerin Otopsisi + Kriter 2 + Alt-grup tablosu (val, eval-only).

M5+heatTTA ve M5+M6+TTA için K20/K50'de kaçan nodülleri inceler:
  Q1: GT merkezi dilate maske dışında mı (yapısal kaçırma)?
  Q2: Kaçan nodüllerin boyutu (px/mm) ve kontrastı?
  Q3: Isı haritasında tepe yok mu, yoksa tepe var da 51-100+ sıraya mı düşmüş?
Kullanım: python pipelines/v6_two_stage/diagnose_misses.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
from detector_models import CenterNetHead, FCOSHead  # noqa: E402
import benchmark_v6_0_phase2 as p2  # noqa: E402
import eval_variants as ev  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_0_miss_autopsy.json"
KS = (10, 20, 30, 50)
SUBS = ("kucuk", "orta", "buyuk")


def _raw_lung_mask(pid: str) -> np.ndarray:
    import SimpleITK as sitk
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(common.PATIENTS_DIR / pid / "mask_5zones.mha")))
    if m.ndim == 3:
        m = m[0]
    return m > 0


def _best_hit_rank(cands, gt) -> int | None:
    for i, (cx, cy, _) in enumerate(cands, start=1):
        if common.hit_criterion1(cx, cy, gt):
            return i
    return None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m5 = CenterNetHead().to(device); m5.load_state_dict(torch.load(CKPT / "M5_centernet.pt", map_location=device)); m5.eval()
    m6 = FCOSHead().to(device); m6.load_state_dict(torch.load(CKPT / "M6_fcos.pt", map_location=device)); m6.eval()

    pids = p2._nodule_pids("val")
    print(f"Otopsi | val {len(pids)} hastada {sum(len(common.load_gt(p)) for p in pids)} nodül")

    stats = {}
    for cfg in ("M5_heatTTA", "M5M6TTA"):
        stats[cfg] = {
            "hits_k1": {str(k): 0 for k in KS}, "hits_k2": {str(k): 0 for k in KS},
            "sub_k1_K20": {s: [0, 0] for s in SUBS}, "sub_k2_K20": {s: [0, 0] for s in SUBS},
            "n_lesions": 0,
        }
    autopsy = []          # kaçan nodül kayıtları (M5_heatTTA @K50)
    n_miss20 = n_miss50 = 0

    for pid in pids:
        canon, _, _ = p2._prep(pid)
        raw = _raw_lung_mask(pid)
        dil = common.load_lung_mask(pid)
        gts = common.load_gt(pid)

        c5h = ev._predict_heat_tta(m5, canon, device, "M5")
        c6 = p2.predict(m6, canon, device, "M6")
        configs = {"M5_heatTTA": c5h, "M5M6TTA": ev._rrf([c5h, c6])}

        lung_med = float(np.median(canon[raw])) if raw.any() else 0.0

        for cfg, cands in configs.items():
            st = stats[cfg]
            st["n_lesions"] += len(gts)
            for k in KS:
                st["hits_k1"][str(k)] += int(round(common.recall_at_K(cands, gts, k, common.hit_criterion1) * len(gts)))
                st["hits_k2"][str(k)] += int(round(common.recall_at_K(cands, gts, k, common.hit_criterion2) * len(gts)))
            for gt in gts:
                sg = common.subgroup_of(gt["size_mm"])
                h1 = any(common.hit_criterion1(cx, cy, gt) for cx, cy, _ in cands[:20])
                h2 = any(common.hit_criterion2(cx, cy, gt) for cx, cy, _ in cands[:20])
                st["sub_k1_K20"][sg][1] += 1; st["sub_k1_K20"][sg][0] += int(h1)
                st["sub_k2_K20"][sg][1] += 1; st["sub_k2_K20"][sg][0] += int(h2)

        # --- M5_heatTTA için kaçan nodül otopsisi ---
        for gt in gts:
            hit50 = any(common.hit_criterion1(cx, cy, gt) for cx, cy, _ in c5h[:50])
            if hit50:
                continue
            n_miss50 += 1
            hit20 = any(common.hit_criterion1(cx, cy, gt) for cx, cy, _ in c5h[:20])
            if not hit20:
                n_miss20 += 1
            cx, cy = gt["cx"], gt["cy"]
            ix, iy = int(round(cx)), int(round(cy))
            in_raw = bool(raw[iy, ix]) if 0 <= iy < raw.shape[0] and 0 <= ix < raw.shape[1] else False
            in_dil = bool(dil[iy, ix]) if 0 <= iy < dil.shape[0] and 0 <= ix < dil.shape[1] else False
            x0, x1 = int(max(0, gt["x"])), int(min(canon.shape[1], gt["x"] + gt["w"]))
            y0, y1 = int(max(0, gt["y"])), int(min(canon.shape[0], gt["y"] + gt["h"]))
            nod_mean = float(np.mean(canon[y0:y1, x0:x1])) if (x1 > x0 and y1 > y0) else 0.0
            autopsy.append({
                "patient": pid,
                "center": [round(cx, 1), round(cy, 1)],
                "size_px": [round(gt["w"], 1), round(gt["h"], 1)],
                "size_mm": round(gt["size_mm"], 1),
                "in_raw_lung": in_raw,
                "in_dilated_lung": in_dil,
                "contrast": round(nod_mean - lung_med, 1),
                "best_hit_rank": _best_hit_rank(c5h, gt),
                "hit_at_K20": hit20,
            })

    # --- özet ---
    def rc(v, n):
        return round(v / max(1, n), 4)

    summary = {}
    for cfg, st in stats.items():
        summary[cfg] = {
            "recall_k1_froc": {str(k): rc(st["hits_k1"][str(k)], st["n_lesions"]) for k in KS},
            "recall_k2_containment": {str(k): rc(st["hits_k2"][str(k)], st["n_lesions"]) for k in KS},
            "subgroup_k1_K20": {s: (rc(v[0], v[1]) if v[1] else None) for s, v in st["sub_k1_K20"].items()},
            "subgroup_k2_K20": {s: (rc(v[0], v[1]) if v[1] else None) for s, v in st["sub_k2_K20"].items()},
            "n_lesions": st["n_lesions"],
        }

    ranks = [a["best_hit_rank"] for a in autopsy]
    miss_summary = {
        "n_missed_at_K50": n_miss50,
        "n_missed_at_K20": n_miss20,
        "n_gt_lesions": summary["M5_heatTTA"]["n_lesions"],
        "Q1_mask_kaynakli": {
            "disinda_raw_lung": sum(1 for a in autopsy if not a["in_raw_lung"]),
            "disinda_dilated": sum(1 for a in autopsy if not a["in_dilated_lung"]),
        },
        "Q2_boyut_kontrast": {
            "median_size_mm": float(np.median([a["size_mm"] for a in autopsy])) if autopsy else None,
            "median_size_px": float(np.median([max(a["size_px"]) for a in autopsy])) if autopsy else None,
            "median_contrast": float(np.median([a["contrast"] for a in autopsy])) if autopsy else None,
        },
        "Q3_tepe_sirasi": {
            "tepe_yok_tamamen": sum(1 for r in ranks if r is None),
            "rank_51_100": sum(1 for r in ranks if r is not None and 51 <= r <= 100),
            "rank_101_300": sum(1 for r in ranks if r is not None and 101 <= r <= 300),
            "rank_301_plus": sum(1 for r in ranks if r is not None and r > 300),
        },
        "misses": autopsy,
    }

    report = {"experiment": "V6.0", "stage": "kacan nodul otopsisi (val)", "summary": summary,
              "miss_autopsy": miss_summary}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Kritik 1 (FROC) ===")
    for cfg in summary:
        s = summary[cfg]["recall_k1_froc"]
        print(f"  {cfg:10s} " + "  ".join(f"K{k}={s[str(k)]}" for k in KS))
    print("=== Kritik 2 (kapsama) ===")
    for cfg in summary:
        s = summary[cfg]["recall_k2_containment"]
        print(f"  {cfg:10s} " + "  ".join(f"K{k}={s[str(k)]}" for k in KS))
    print("=== Alt grup @K20 (Kritik1 / Kritik2) ===")
    for cfg in summary:
        print(f"  {cfg:10s} " + "  ".join(
            f"{sg}: {summary[cfg]['subgroup_k1_K20'][sg]}/{summary[cfg]['subgroup_k2_K20'][sg]}"
            for sg in SUBS))
    print(f"\n=== KAÇAN NODÜL OTOPSİSİ (M5_heatTTA) ===\n  K50'de kaçan: {n_miss50} / {summary['M5_heatTTA']['n_lesions']}"
          f" | K20'de kaçan: {n_miss20}")
    print(f"  Q1 maske: raw dışı {miss_summary['Q1_mask_kaynakli']['disinda_raw_lung']}, "
          f"dilate dışı {miss_summary['Q1_mask_kaynakli']['disinda_dilated']}")
    print(f"  Q2 boyut/kontrast: medyan {miss_summary['Q2_boyut_kontrast']['median_size_mm']} mm, "
          f"{miss_summary['Q2_boyut_kontrast']['median_size_px']} px, kontrast {miss_summary['Q2_boyut_kontrast']['median_contrast']}")
    print(f"  Q3 tepe sırası: {miss_summary['Q3_tepe_sirasi']}")
    print(f"\nRapor: {OUT}")


if __name__ == "__main__":
    main()

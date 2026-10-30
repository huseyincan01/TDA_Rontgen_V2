"""V6.9k2 — LOSO'nun "veri azalması" kontrastı: val(−JSRT) üzerinde LOSO vs ANA.

NEDEN
-----
V6.9k'de LOSO eğitimi JSRT'yi train'den çıkarınca eğitim seti 1.580 → 1.366 görüntüye
düşüyor. JSRT'de bir düşüş görülürse bunun iki açıklaması olur:

    (i)  **kaynak ezberi**: model JSRT hastalarını hatırlıyordu, artık hatırlamıyor;
    (ii) **veri azalması**: train %13,5 küçüldü, her yerde biraz düştü.

Ayrıştırmak için aynı iki modeli **JSRT içermeyen** val hastalarında kıyaslıyoruz. Orada
kaynak ezberi söz konusu değil (val zaten eğitimde yok) ⇒ ölçülen Δ **genel veri maliyeti**dir.

    DiD = Δ(JSRT)  −  Δ(val, JSRT-dışı)

DiD ≈ 0 ⇒ JSRT'deki fark yalnız veri azalmasıyla açıklanır (kaynak ezberi kanıtı yok).
DiD < 0 ⇒ JSRT'ye özgü ek bir kayıp var.

Bu betik **yalnız 1. aşama** metriklerini ölçer (aday üretimi; TDA/HGB gerekmez) — hızlıdır.
Ağırlık: Recall@K (lezyon) + hasta düzeyi AUC (val pozitifleri vs val kontrolleri).

İzolasyon: hiçbir şey yazmaz; ana artifact'ler salt-okunur. Çıktı ayrı JSON.

Kullanım: python pipelines/v6_two_stage/v6_9k2_val_control.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
from detector_models_foundation import canon_to_512  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402
from v6_6_optimization_audit import dist_nms, heat_maps, pick_peaks  # noqa: E402
from v6_9k_leave_jsrt_out import (KS, LOSO_TAG, MAIN_TAG, SCALES, WIN, NMS_PX, TOPK,  # noqa: E402
                                  _auc, jsrt_patients)

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
OUT = ART / "v6_9k2_val_control.json"
SEED = 42
N_CTRL = 250


def _cands(model, pids, device, tag) -> dict:
    out, t0 = {}, time.time()
    for i, pid in enumerate(pids):
        canon, _, _ = p2._prep(pid)
        heat, rad = heat_maps(model, canon_to_512(canon), device, SCALES)
        cs = dist_nms(pick_peaks(heat, rad, WIN), NMS_PX, top=TOPK)
        out[pid] = [(float(cx), float(cy), float(s)) for cx, cy, s, _r in cs]
        if (i + 1) % 200 == 0:
            print(f"    {tag} {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    {tag} bitti ({time.time() - t0:.0f}s)", flush=True)
    return out


def measure(pids_pos, pids_ctrl, cands, gts) -> dict:
    n_les = sum(len(gts[p]) for p in pids_pos)
    rec = {}
    for K in KS:
        hit = sum(int(round(common.recall_at_K(cands[p], gts[p], K, common.hit_criterion1)
                            * len(gts[p]))) for p in pids_pos if gts[p])
        rec[f"K{K}"] = round(hit / max(1, n_les), 4)

    def top1(p):
        ss = sorted([c[2] for c in cands[p]], reverse=True)
        return float(ss[0]) if ss else 0.0

    def top3(p):
        ss = sorted([c[2] for c in cands[p]], reverse=True)
        return float(np.mean(ss[:3])) if ss else 0.0

    def nors(p):
        ss = sorted([c[2] for c in cands[p]], reverse=True)
        return float(1 - np.prod([1 - min(1.0, max(0.0, x)) for x in ss[:5]]))

    y = [1] * len(pids_pos) + [0] * len(pids_ctrl)
    auc = {how: _auc(y, [f(p) for p in pids_pos] + [f(p) for p in pids_ctrl])
           for how, f in (("top1", top1), ("top3", top3), ("noisyor", nors))}
    return {"n_positive": len(pids_pos), "n_lesions": n_les,
            "n_controls": len(pids_ctrl), "recall_at_K": rec, "mixed_patient_auc": auc}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V6.9k2 — val(−JSRT) kontrastı | device={device}", flush=True)

    js = jsrt_patients()
    js_set = set(js["train"]) | set(js["val"]) | set(js["test"])
    val = [p for p in common.split_patients()["val"] if p not in js_set]
    gts = {p: common.load_gt(p) for p in val}
    pos = [p for p in val if gts[p]]
    ctrl_all = [p for p in val if not gts[p]]
    rng = np.random.RandomState(SEED)
    ctrl = list(rng.permutation(ctrl_all)[:N_CTRL])
    print(f"  val(−JSRT): {len(pos)} pozitif · {len(ctrl)} kontrol "
          f"({sum(len(gts[p]) for p in pos)} lezyon)", flush=True)

    rep = {"experiment": "V6.9k2-val-control",
           "question": ("LOSO'nun JSRT'deki farkı kaynak ezberinden mi, veri azalmasından mı? "
                        "val(−JSRT) üzerinde genel veri maliyeti ölçülür."),
           "split": "val, JSRT hastaları çıkarıldı",
           "protocol": {"scales": list(SCALES), "flip_tta": True, "win": WIN,
                        "nms_px": NMS_PX, "top_k": TOPK, "stage": "yalnız 1. aşama"},
           "models": {}}

    res = {}
    for tag in (LOSO_TAG, MAIN_TAG):
        print(f"\n[{tag}] aday üretimi…", flush=True)
        m = load_foundation(tag, device)
        c = _cands(m, pos + ctrl, device, tag)
        res[tag] = measure(pos, ctrl, c, gts)
        r = res[tag]
        print(f"  {tag}: Recall@K {r['recall_at_K']} | AUC {r['mixed_patient_auc']}", flush=True)
        rep["models"][tag] = r

    d = {k: round(res[LOSO_TAG]["recall_at_K"][k] - res[MAIN_TAG]["recall_at_K"][k], 4)
         for k in res[LOSO_TAG]["recall_at_K"]}
    dauc = {k: (round(res[LOSO_TAG]["mixed_patient_auc"][k]
                       - res[MAIN_TAG]["mixed_patient_auc"][k], 4)
                if res[LOSO_TAG]["mixed_patient_auc"][k] is not None
                and res[MAIN_TAG]["mixed_patient_auc"][k] is not None else None)
            for k in res[LOSO_TAG]["mixed_patient_auc"]}
    rep["delta_loso_minus_main_val"] = {"recall_at_K": d, "mixed_patient_auc": dauc}
    print(f"\n  Δ(val, LOSO − ANA): Recall {d} | AUC {dauc}", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


if __name__ == "__main__":
    main()

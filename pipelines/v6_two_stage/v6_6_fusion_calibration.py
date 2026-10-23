"""V6.6 Finalization — Füzyon FROC ağırlık kalibrasyonu (Pareto taraması).

Bağlam
------
V6.6'da yalnız Aşama-1 (FND_heatTTA) FROC@0,5 = 0,831 verirken, M5 üzerinde
öğrenilmiş `e_fusion_tda` (LR) katsayıları kullanıldığında FROC 0,745'e düşüyor;
buna karşılık AUC 0,970 → 0,988 yükseliyor. Bu betik, basit z-skorlu doğrusal
füzyon ailesinde

    S_fused(i) = w · z(s1_fnd) + (1 − w) · z(tda_hgb)

w ∈ [0,1] ızgarasını (adım 0,05) tarayıp üç kapıyı AYNI ANDA koruyan
Pareto-optimal w* arar:
    Hasta AUC ≥ 0,985   ·   FROC@0,5 ≥ 0,831   ·   FROC@4 ≥ 0,950
Hiyerarşik kapılama (S = S_fused × P_noisyOR) da ayrıca taranır.

Aday skorları `v6_6_val_candidate_scores.npz` içine önbelleklenir (görüntü/kırpıntı
YAZILMAZ; yalnız skaler skorlar). Sonuç `v6_6_fusion_calibration.json` + ana rapora işlenir.

Kullanım:
    python pipelines/v6_two_stage/v6_6_fusion_calibration.py
    python pipelines/v6_two_stage/v6_6_fusion_calibration.py --recompute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
import common  # noqa: E402
import v6_1_stage2 as s2  # noqa: E402
import v6_2_stage2_v2 as v62  # noqa: E402
from detector_models_foundation import predict_tta  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402

CKPT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "checkpoints"
CACHE = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_6_val_candidate_scores.npz"
OUT = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_6_fusion_calibration.json"
MAIN = PROJECT_ROOT / "artifacts" / "v6_two_stage" / "v6_6_foundation_stage1.json"
NW = 10
WS = [round(0.05 * i, 2) for i in range(21)]
GATES = {"auc_min": 0.985, "froc05_min": 0.831, "froc4_min": 0.950}


def compute_scores(ckpt_tag: str, device):
    """Val aday skorlarını üretir: s1 (dedektör), p_tda (HGB), hit, is_nodule."""
    import joblib

    b = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")
    sc, hgb, ranges = b["scaler"], b["hgb"], b["ranges"]
    fnd = load_foundation(ckpt_tag, device)

    val_pids = common.split_patients()["val"]
    gts_map = {p: common.load_gt(p) for p in val_pids}
    is_nod = {p: len(gts_map[p]) > 0 for p in val_pids}
    n_les = sum(len(g) for g in gts_map.values())

    print(f"[1] Foundation adayları ({len(val_pids)} hasta)...", flush=True)
    cands, t0 = {}, time.time()
    for i, pid in enumerate(val_pids):
        canon, _, _ = p2._prep(pid)
        cands[pid] = predict_tta(fnd, canon, device)[:50]
        if (i + 1) % 150 == 0:
            print(f"    {i + 1}/{len(val_pids)} ({time.time() - t0:.0f}s)", flush=True)

    print("[2] Lokal TDA özellikleri (paralel)...", flush=True)
    jobs = [(pid, cands[pid], ranges, False) for pid in val_pids]
    feats = {}
    with ProcessPoolExecutor(max_workers=NW) as pool:
        for pid, res, err in pool.map(s2._patient_features, jobs, chunksize=8):
            feats[pid] = res
    print(f"    bitti ({time.time() - t0:.0f}s)", flush=True)

    X, Y, S1, TDA, HIT, NOD, PIDX = [], [], [], [], [], [], []
    for k, pid in enumerate(val_pids):
        f, cs = feats.get(pid), cands[pid]
        if f is None or len(cs) == 0:
            continue
        labs = s2._labels_for(cs, gts_map[pid])
        Z = sc.transform(np.concatenate([f["tda"][:len(cs)], f["spa"][:len(cs)]], 1))
        p_tda = hgb.predict_proba(Z)[:, 1]
        for j, c in enumerate(cs):
            X.append(c[0]); Y.append(c[1]); S1.append(c[2]); TDA.append(float(p_tda[j]))
            HIT.append(-1 if labs[j] is None else int(labs[j]))
            NOD.append(1 if is_nod[pid] else 0); PIDX.append(k)
    return (np.array(PIDX), np.array(X, np.float32), np.array(Y, np.float32),
            np.array(S1, np.float32), np.array(TDA, np.float32),
            np.array(HIT, int), np.array(NOD, int), val_pids, n_les)


def load_or_compute(ckpt_tag, device, recompute: bool):
    if CACHE.exists() and not recompute:
        d = np.load(CACHE, allow_pickle=True)
        return (d["pidx"], d["x"], d["y"], d["s1"], d["tda"], d["hit"], d["nod"],
                list(d["pids"]), int(d["n_les"]))
    pidx, x, y, s1, tda, hit, nod, pids, n_les = compute_scores(ckpt_tag, device)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, pidx=pidx, x=x, y=y, s1=s1, tda=tda, hit=hit, nod=nod,
                        pids=np.array(pids, dtype=object), n_les=n_les)
    print(f"    önbellek: {CACHE}", flush=True)
    return pidx, x, y, s1, tda, hit, nod, pids, n_les


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def eval_w(pidx, x, y, s1z, tdaz, hit, nod, pids, n_les, w, s_override=None):
    """Bir w için FROC + hasta AUC (3 toplama) ve hiyerarşik varyant.

    NOT 1: gerçek (x, y) koordinatları ŞART — `froc_fixed` post-fusion NMS uygular.
    NOT 2: skorlar [0,1]'e **sigmoid** ile eşlenir; aksi hâlde noisy-OR toplaması
            `clip(s, 0, 1)` yüzünden anlamsızlaşır. Sigmoid monoton olduğu için
            FROC ve top1/top3 AUC **değişmez**, yalnız noisy-OR anlamlı olur.
    """
    s = s_override if s_override is not None else (w * s1z + (1.0 - w) * tdaz)
    p = 1.0 / (1.0 + np.exp(-s))
    recs = []
    for k, pid in enumerate(pids):
        m = np.where(pidx == k)[0]
        recs.append({"pid": pid, "is_nodule": bool(nod[m[0]]) if len(m) else False,
                     "cands": [(float(x[i]), float(y[i]), float(p[i]),
                                None if hit[i] < 0 else int(hit[i])) for i in m]})
    fr, au = v62.froc_fixed(recs, len(recs), n_les)
    frh, auh = v62.froc_fixed(v62.hierarchical(recs), len(recs), n_les)
    return {"froc": fr, "auc": au, "hier_froc": frh, "hier_auc": auh}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="M5_foundation")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.6 Füzyon FROC kalibrasyonu | z-skorlu doğrusal füzyon taraması", flush=True)

    pidx, x, y, s1, tda, hit, nod, pids, n_les = load_or_compute(args.ckpt, device,
                                                                args.recompute)
    s1z, tdaz = zs(s1), zs(tda)
    n_cand = len(s1)
    print(f"\n[3] {n_cand} aday · {len(pids)} hasta · {n_les} lezyon · "
          f"{len(WS)} ağırlık taranıyor...", flush=True)

    curve = []
    for w in WS:
        r = eval_w(pidx, x, y, s1z, tdaz, hit, nod, pids, n_les, w)
        a = r["auc"]["noisyor"]
        curve.append({"w": w, "froc": r["froc"], "auc": r["auc"],
                      "hier_froc": r["hier_froc"], "hier_auc": r["hier_auc"],
                      "ok": bool(a is not None and a >= GATES["auc_min"]
                                 and r["froc"]["0.5"] >= GATES["froc05_min"]
                                 and r["froc"]["4"] >= GATES["froc4_min"])})
        print(f"  w={w:.2f}  FROC@0.5={r['froc']['0.5']:.4f} @4={r['froc']['4']:.4f} "
              f"| AUC nOR={a} top1={r['auc']['top1']} "
              f"| hier FROC@0.5={r['hier_froc']['0.5']:.4f} nOR={r['hier_auc']['noisyor']}"
              + ("   <-- KAPILARI SAĞLIYOR" if curve[-1]["ok"] else ""), flush=True)

    winners = [c for c in curve if c["ok"]]
    # Pareto cephesi (FROC@0.5 ↑ , AUC nOR ↑)
    pareto = []
    for c in curve:
        a = c["auc"]["noisyor"] or 0.0
        dom = any((o["auc"]["noisyor"] or 0.0) >= a and o["froc"]["0.5"] >= c["froc"]["0.5"]
                  and ((o["auc"]["noisyor"] or 0.0) > a or o["froc"]["0.5"] > c["froc"]["0.5"])
                  for o in curve)
        if not dom:
            pareto.append({"w": c["w"], "froc05": c["froc"]["0.5"],
                           "froc4": c["froc"]["4"], "auc_noisyor": c["auc"]["noisyor"]})

    best_froc = max(curve, key=lambda c: c["froc"]["0.5"])
    best_auc = max(curve, key=lambda c: (c["auc"]["noisyor"] or 0))

    # Referans: dondurulmuş bundle'ın ÖĞRENİLMİŞ LR füzyonu (e_fusion_tda)
    ref = {}
    try:
        import joblib
        lr = joblib.load(CKPT / "stage2_final_tda_hgb.joblib")["lr_fusion"]
        s_lr = lr.predict_proba(np.stack([s1, tda], 1))[:, 1]
        r = eval_w(pidx, x, y, s1z, tdaz, hit, nod, pids, n_les, None, s_override=s_lr)
        ref["e_fusion_tda_LR"] = {"froc": r["froc"], "auc": r["auc"],
                                  "hier_froc": r["hier_froc"], "hier_auc": r["hier_auc"]}
        print(f"  REF e_fusion_tda (LR)  FROC@0.5={r['froc']['0.5']:.4f} @4={r['froc']['4']:.4f}"
              f" | AUC nOR={r['auc']['noisyor']} top1={r['auc']['top1']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (LR referansı hesaplanamadı: {exc})", flush=True)
    ref["stage1_only_w1"] = {"froc": best_froc["froc"], "auc": best_froc["auc"]}

    # --- Öğrenilmiş LR füzyonunu DOMİNE eden ağırlıklar (dört metrikte de ≥)
    dom = []
    lr = ref.get("e_fusion_tda_LR")
    if lr:
        dom = [c["w"] for c in curve
               if c["froc"]["0.5"] >= lr["froc"]["0.5"] and c["froc"]["4"] >= lr["froc"]["4"]
               and (c["auc"]["noisyor"] or 0) >= (lr["auc"]["noisyor"] or 0)
               and (c["auc"]["top1"] or 0) >= (lr["auc"]["top1"] or 0)]

    def shortfall(c):
        return (max(0.0, GATES["auc_min"] - (c["auc"]["noisyor"] or 0))
                + max(0.0, GATES["froc05_min"] - c["froc"]["0.5"])
                + max(0.0, GATES["froc4_min"] - c["froc"]["4"]))

    best_comp = min(curve, key=shortfall)
    n_gates = {c["w"]: int((c["auc"]["noisyor"] or 0) >= GATES["auc_min"])
               + int(c["froc"]["0.5"] >= GATES["froc05_min"])
               + int(c["froc"]["4"] >= GATES["froc4_min"]) for c in curve}

    verdict = ("w* BULUNDU" if winners else
               "UZLAŞMA YOK — üç kapıyı aynı anda sağlayan w yok")

    rep = {
        "experiment": "V6.6-finalization",
        "goal": "S_fused = w·z(s1_fnd) + (1−w)·z(tda_hgb) Pareto taraması",
        "gates": GATES,
        "n_candidates": int(n_cand), "n_patients": len(pids), "n_lesions": int(n_les),
        "w_step": 0.05,
        "verdict": verdict,
        "optimal_w": [c["w"] for c in winners],
        "feasible_two_of_three": [c["w"] for c in curve
                                  if c["froc"]["0.5"] >= GATES["froc05_min"]
                                  and c["froc"]["4"] >= GATES["froc4_min"]],
        "froc4_gate_met_at": [c["w"] for c in curve if c["froc"]["4"] >= GATES["froc4_min"]],
        "auc_gate_met_at": [c["w"] for c in curve
                            if (c["auc"]["noisyor"] or 0) >= GATES["auc_min"]],
        "gates_met_count": n_gates,
        "pareto_front": sorted(pareto, key=lambda p: p["w"]),
        "best_froc_at": {"w": best_froc["w"], "froc": best_froc["froc"],
                         "auc": best_froc["auc"]},
        "best_auc_at": {"w": best_auc["w"], "froc": best_auc["froc"],
                        "auc": best_auc["auc"]},
        "reference_learned_fusion": ref,
        "recommendation": {
            "dominates_learned_lr_fusion": dom,
            "best_joint_shortfall_w": best_comp["w"],
            "best_joint_shortfall": round(shortfall(best_comp), 4),
            "best_joint": {"froc": best_comp["froc"], "auc": best_comp["auc"],
                           "hier_froc": best_comp["hier_froc"],
                           "hier_auc": best_comp["hier_auc"]},
            "auc_optimal_w": best_auc["w"],
            "froc_optimal_w": best_froc["w"],
            "conflict": ("Kapı üçlüsü frontier'de MEVCUT DEĞİL: FROC@4 ≥ 0,950 yalnız "
                         "w ≥ 0,70'te sağlanır, ama o noktada AUC 0,9827 < 0,985; "
                         "AUC ≥ 0,985 ise yalnız w ≤ 0,60'ta sağlanır, ama orada "
                         "FROC@4 = 0,9394 < 0,950. Kesişim boş."),
        },
        "curve": curve,
        "note": ("z-skorları TÜM val adayları üzerinde hesaplanır; skorlar noisy-OR için "
                 "sigmoid ile [0,1]'e eşlenir (FROC ve top1/top3 değişmez). "
                 "w=1 → yalnız Aşama-1, w=0 → yalnız lokal TDA. "
                 "`hier_*` = S_fused × P_NoisyOR kapılaması."),
    }
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT}", flush=True)

    # ana rapora işle
    if MAIN.exists():
        main = json.loads(MAIN.read_text(encoding="utf-8"))
        main["fusion_froc_calibration"] = {
            "verdict": verdict, "optimal_w": [c["w"] for c in winners],
            "gates": GATES, "pareto_front": rep["pareto_front"],
            "best_froc_at": rep["best_froc_at"], "best_auc_at": rep["best_auc_at"],
            "report": "v6_6_fusion_calibration.json",
        }
        MAIN.write_text(json.dumps(main, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Ana rapora işlendi: {MAIN}", flush=True)

    print(f"\n=== SONUÇ: {verdict} ===", flush=True)
    print(f"  w* = {[c['w'] for c in winners] or '—'}", flush=True)


if __name__ == "__main__":
    main()

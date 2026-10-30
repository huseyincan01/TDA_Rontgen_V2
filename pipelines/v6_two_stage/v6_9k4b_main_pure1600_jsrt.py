"""V6.9k4b — ANA modelin SAF 1.600 sürümünün JSRT'deki TDA-tek puanı (küçük tamamlayıcı).

§5.1'deki "TDA-tek puan JSRT'de kaynaktan bağımsız taşınıyor" cümlesi için, ana mimarinin
(`M5_foundation` + `stage2_foundation_tda_hgb_pure1600`) JSRT 149 hastasındaki TDA-tek hasta
AUC'sini hesaplar. LOSO tarafı V6.9k4'te zaten var (0,9501).

Kilitli teste dokunmaz; yalnız JSRT (val+test+train JSRT) üzerinde ölçüm yapar.
Kullanım: python pipelines/v6_two_stage/v6_9k4b_main_pure1600_jsrt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402
import v6_9k_leave_jsrt_out as k  # noqa: E402
import v6_9l_spatial_ablation as L  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
OUT = ART / "v6_9k4b_main_pure1600_jsrt.json"
B1600 = CKPT / "stage2_foundation_tda_hgb_pure1600.joblib"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9k4b — ana model (saf 1.600) JSRT TDA-tek ölçümü", flush=True)

    js = k.jsrt_patients()
    js_all = js["train"] + js["val"] + js["test"]
    b = joblib.load(B1600)
    model = load_foundation(k.MAIN_TAG, device)
    print(f"  JSRT {len(js_all)} hasta · bundle n_features={b['hgb'].n_features_in_}",
          flush=True)

    c_js = L._candidates(model, js_all, device, "jsrt-main")
    f_js = L._features(js_all, c_js, b["ranges"], "jsrt-main")
    pk = L.pack(js_all, c_js, f_js)
    p_tda = b["hgb"].predict_proba(b["scaler"].transform(pk["tda"].astype(np.float32)))[:, 1]

    # kontrol havuzu: val kontrolleri (250, seed 42) — LOSO betiğiyle aynı
    sp = common.split_patients()
    js_set = set(js["train"]) | set(js["val"]) | set(js["test"])
    va = [p for p in sp["val"] if p not in js_set]
    rng = np.random.RandomState(42)
    ctrl = list(rng.permutation([p for p in va if not common.load_gt(p)])[:250])
    c_ct = L._candidates(model, ctrl, device, "ctrl-main")
    f_ct = L._features(ctrl, c_ct, b["ranges"], "ctrl-main")
    pk_ct = L.pack(ctrl, c_ct, f_ct)
    p_ct = b["hgb"].predict_proba(b["scaler"].transform(pk_ct["tda"].astype(np.float32)))[:, 1]

    gts = {p: common.load_gt(p) for p in js_all + ctrl}
    cands = {p: c_js[p] for p in js_all}
    cands.update({p: c_ct[p] for p in ctrl})
    s1 = {p: np.asarray([c[2] for c in cands[p]], np.float32) for p in js_all + ctrl}
    scored = {p: (s1[p], np.zeros(0, np.float32)) for p in js_all + ctrl}
    for i, p in enumerate(js_all):
        scored[p] = (s1[p], p_tda[pk["off"][i]:pk["off"][i + 1]])
    for i, p in enumerate(ctrl):
        scored[p] = (s1[p], p_ct[pk_ct["off"][i]:pk_ct["off"][i + 1]])

    m = k.measure(js_all, cands, scored, {"s1_mean": 0, "s1_std": 1, "tda_mean": 0,
                                          "tda_std": 1, "w": k.W},
                  {p: g for p, g in gts.items()}, ctrl_pids=ctrl,
                  ctrl_cands={p: cands[p] for p in ctrl},
                  ctrl_scored={p: scored[p] for p in ctrl})
    out = {"experiment": "V6.9k4b-main-pure1600-jsrt",
           "note": "Ana model (M5_foundation + pure1600) JSRT 149'da TDA-tek hasta AUC.",
           "jsrt_n": len(js_all), "controls": len(ctrl),
           "tda_only": m["mixed_patient_auc"]["tda"],
           "fusion": m["mixed_patient_auc"]["fusion"],
           "recall_at_K": m["recall_at_K"]}
    print(f"  TDA-tek hasta AUC : {out['tda_only']}", flush=True)
    print(f"  Füzyon hasta AUC  : {out['fusion']}", flush=True)
    print(f"  Recall@K          : {out['recall_at_K']}", flush=True)
    Path(OUT).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  kayıt: {OUT.name}", flush=True)


if __name__ == "__main__":
    main()

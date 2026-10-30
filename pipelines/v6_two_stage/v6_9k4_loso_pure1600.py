"""V6.9k4 — Leave-JSRT-Out'un SAF 1.600 sürümü (dış test satırının tutarlılığı).

NEDEN
-----
V6.9k LOSO deneyi 2. aşamayı **1.604** özellikle (1.600 PI + 4 konum vekili) kurmuştu. V6.9m'de
ana mimari **saf 1.600**'e geçtiği için, hocaya sunulan "yüksek kaliteli dış hastane (JSRT)"
satırının da aynı temsille ölçülmesi gerekir. Bu betik aynı LOSO 1. aşama checkpoint'ini
(`M5_nojsrt.pt`) kullanır; yalnız 2. aşama temsili değişir.

    LOSO 1. aşama : M5_nojsrt.pt                    (JSRT eğitimde YOK — değişmedi)
    TDA aralıkları: LOSO bundle'ından devralınır      (JSRT görmedi)
    fit havuzu    : train − JSRT = 683 + 683 = 1.366 hasta
    z-ref         : val(−JSRT)                        (JSRT görmedi)
    dış test      : JSRT 149 hasta (hepsi yabancı)

İki kol aynı önbellekten üretilir:
    A) saf 1.600  → yeni HGB (`stage2_nojsrt_tda_hgb_pure1600.joblib`)
    B) 1.604      → mevcut LOSO bundle'ı yeniden uygulanır (V6.9k sayılarının tekrarı = kontrol)

İzolasyon: eski LOSO 1.604 bundle'ı ve ana checkpoint'ler **silinmez/değiştirilmez**.

Kullanım: python pipelines/v6_two_stage/v6_9k4_loso_pure1600.py [--refresh]
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
import v6_9k_leave_jsrt_out as k  # noqa: E402
import v6_9l_spatial_ablation as L  # noqa: E402
from v6_6_foundation_stage1 import load_foundation  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
CKPT = ART / "checkpoints"
OUT = ART / "v6_9k4_loso_pure1600.json"
CACHE = ART / "_v6_9k4_cache.npz"
LOSO_TAG = k.LOSO_TAG
B1604 = CKPT / "stage2_nojsrt_tda_hgb.joblib"
B1600 = CKPT / "stage2_nojsrt_tda_hgb_pure1600.joblib"
SEED = 42
N_CTRL = 250


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("V6.9k4 — LOSO dış testinin SAF 1.600 sürümü", flush=True)
    print(f"  device={device} | kilitli TEST'e dokunulmaz | LOSO 1. aşama = {LOSO_TAG}",
          flush=True)

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    js = k.jsrt_patients()
    js_set = set(js["train"]) | set(js["val"]) | set(js["test"])
    sp = common.split_patients()
    tr_pids = [p for p in p2._nodule_pids("train") if p not in set(js["train"])]
    ctl_pids = p2._control_pids("train")[:len(tr_pids)]
    va_pids = [p for p in sp["val"] if p not in set(js["val"])]
    rng = np.random.RandomState(SEED)
    ctrl_eval = list(rng.permutation([p for p in va_pids if not common.load_gt(p)])[:N_CTRL])
    js_all = js["train"] + js["val"] + js["test"]
    fit_pids = tr_pids + ctl_pids
    print(f"  fit {len(fit_pids)} · val(-JSRT) {len(va_pids)} · JSRT {len(js_all)} · "
          f"kontrol {len(ctrl_eval)}", flush=True)

    if CACHE.exists() and not args.refresh:
        print(f"\n[0] önbellek: {CACHE.name}", flush=True)
        d = {kk: vv for kk, vv in np.load(CACHE, allow_pickle=True).items()}
    else:
        ranges = joblib.load(B1604)["ranges"]
        model = load_foundation(LOSO_TAG, device)
        print("\n[1] Adaylar (fit)…", flush=True)
        c_fit = L._candidates(model, fit_pids, device, "fit")
        print("[2] Adaylar (val)…", flush=True)
        c_va = L._candidates(model, va_pids, device, "val")
        print("[3] Adaylar (JSRT)…", flush=True)
        c_js = L._candidates(model, js_all, device, "jsrt")
        print("[4] Özellikler (fit)…", flush=True)
        f_fit = L._features(fit_pids, c_fit, ranges, "fit")
        print("[5] Özellikler (val)…", flush=True)
        f_va = L._features(va_pids, c_va, ranges, "val")
        print("[6] Özellikler (JSRT)…", flush=True)
        f_js = L._features(js_all, c_js, ranges, "jsrt")
        d = {}
        for tag, pids, cc, ff in (("fit", fit_pids, c_fit, f_fit),
                                  ("va", va_pids, c_va, f_va),
                                  ("js", js_all, c_js, f_js)):
            pk = L.pack(pids, cc, ff)
            d[f"{tag}_tda"] = pk["tda"]
            d[f"{tag}_sb"] = pk["sb"]
            d[f"{tag}_s1"] = pk["s1"]
            d[f"{tag}_off"] = pk["off"]
            d[f"{tag}_n"] = pk["n"]
            d[f"{tag}_pids"] = np.asarray(pids)
            d[f"{tag}_cx"] = np.asarray([c[0] for p in pids for c in cc[p]], np.float32)
            d[f"{tag}_cy"] = np.asarray([c[1] for p in pids for c in cc[p]], np.float32)
        d["fit_y"] = L.labels_for(fit_pids, c_fit)
        np.savez_compressed(CACHE, **d)
        print(f"  önbellek yazıldı: {CACHE.name}", flush=True)

    fit_pids = [str(p) for p in d["fit_pids"]]
    va_pids = [str(p) for p in d["va_pids"]]
    js_all = [str(p) for p in d["js_pids"]]
    ctrl_eval = [p for p in ctrl_eval if p in set(va_pids)]
    print(f"\n  fit aday {len(d['fit_y'])} (%{100*d['fit_y'].mean():.2f} poz) · "
          f"JSRT aday {int(d['js_n'].sum())}", flush=True)

    # ---------------- kolları kur ----------------
    Xva_tda = d["va_tda"].astype(np.float32)
    Xva_old = np.concatenate([Xva_tda, d["va_sb"].astype(np.float32)], 1)
    arms = {}
    # A) saf 1.600 — yeni HGB
    Xtr = d["fit_tda"].astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         class_weight="balanced",
                                         random_state=SEED).fit(sc.transform(Xtr), d["fit_y"])
    p_va = hgb.predict_proba(sc.transform(Xva_tda))[:, 1]
    ref1600 = {"s1_mean": float(d["va_s1"].mean()), "s1_std": float(d["va_s1"].std()),
               "tda_mean": float(p_va.mean()), "tda_std": float(p_va.std()), "w": k.W}
    arms["saf1600"] = {"bundle": {"hgb": hgb, "scaler": sc,
                                  "ranges": joblib.load(B1604)["ranges"],
                                  "fusion_ref": ref1600,
                                  "stage1_score_included": False},
                       "p_js": hgb.predict_proba(sc.transform(d["js_tda"].astype(np.float32)))[:, 1],
                       "p_va": p_va, "ref": ref1600, "n_features": 1600}
    # B) 1.604 — mevcut LOSO bundle'ı yeniden uygulanır (kontrol)
    b = joblib.load(B1604)
    X_js = np.concatenate([d["js_tda"].astype(np.float32), d["js_sb"].astype(np.float32)], 1)
    arms["ref1604"] = {"bundle": None,
                       "p_js": b["hgb"].predict_proba(b["scaler"].transform(X_js))[:, 1],
                       "p_va": b["hgb"].predict_proba(b["scaler"].transform(Xva_old))[:, 1],
                       "ref": b["fusion_ref"], "n_features": 1604}

    # ---------------- ölçüm ----------------
    gts_js = {p: common.load_gt(p) for p in js_all}
    gts_va = {p: common.load_gt(p) for p in va_pids}
    gts_all = dict(gts_js)
    gts_all.update(gts_va)
    va_index = {p: i for i, p in enumerate(va_pids)}
    cands_js = _cands_from(d, "js", js_all)
    cands_va = _cands_from(d, "va", va_pids)
    ctrl_cands = {p: cands_va[p] for p in ctrl_eval}
    rep = {"experiment": "V6.9k4-loso-pure1600",
           "question": ("V6.9k LOSO dış testi saf 1.600 temsille yeniden ölçüldüğünde JSRT "
                        "sayıları ne olur? (hocaya sunulan satırın tutarlılığı)"),
           "design": {"loso_stage1": LOSO_TAG, "excluded_source": "jsrt",
                      "fit": f"{len(tr_pids)} nodüllü + {len(ctl_pids)} kontrol (JSRT YOK)",
                      "jsrt_eval": len(js_all), "jsrt_lesions": sum(len(g) for g in gts_js.values()),
                      "controls": len(ctrl_eval), "z_ref": "val(-JSRT)",
                      "ranges": "LOSO bundle'ından (JSRT görmedi)"},
           "arms": {}, "bundle_sha_note": "eski 1604 LOSO bundle'ı DEĞİŞMEDİ"}

    for name, a in arms.items():
        s1_js = d["js_s1"].astype(np.float32)
        scored = {p: (s1_js[d["js_off"][i]:d["js_off"][i + 1]],
                      a["p_js"][d["js_off"][i]:d["js_off"][i + 1]])
                  for i, p in enumerate(js_all)}
        s1_va = d["va_s1"].astype(np.float32)
        scored_va = {p: (s1_va[d["va_off"][va_index[p]]:d["va_off"][va_index[p] + 1]],
                         a["p_va"][d["va_off"][va_index[p]]:d["va_off"][va_index[p] + 1]])
                     for p in va_pids}
        m = k.measure(js_all, cands_js, scored, a["ref"], gts_all,
                      ctrl_pids=ctrl_eval, ctrl_cands=ctrl_cands,
                      ctrl_scored={p: scored_va[p] for p in ctrl_eval})
        rep["arms"][name] = m
        print(f"\n  [{name}] ({a['n_features']} özellik)", flush=True)
        print(f"    Recall@K {m['recall_at_K']} · hasta-yakalama {m['patient_hit_top50']}",
              flush=True)
        print(f"    aday-AUC  s1={m['candidate_auc']['stage1']} tda={m['candidate_auc']['tda']} "
              f"füzyon={m['candidate_auc']['fusion']}", flush=True)
        print(f"    kontrol-karışık hasta AUC  {m['mixed_patient_auc']['fusion']}", flush=True)

    rep["delta_saf1600_minus_ref1604"] = {
        "recall_K10": _d(rep["arms"]["saf1600"]["recall_at_K"]["K10"],
                         rep["arms"]["ref1604"]["recall_at_K"]["K10"]),
        "recall_K50": _d(rep["arms"]["saf1600"]["recall_at_K"]["K50"],
                         rep["arms"]["ref1604"]["recall_at_K"]["K50"]),
        "patient_hit_top50": _d(rep["arms"]["saf1600"]["patient_hit_top50"],
                                rep["arms"]["ref1604"]["patient_hit_top50"]),
        "mixed_auc_top1": _d(rep["arms"]["saf1600"]["mixed_patient_auc"]["fusion"]["top1"],
                             rep["arms"]["ref1604"]["mixed_patient_auc"]["fusion"]["top1"]),
        "mixed_auc_noisyor": _d(
            rep["arms"]["saf1600"]["mixed_patient_auc"]["fusion"]["noisyor"],
            rep["arms"]["ref1604"]["mixed_patient_auc"]["fusion"]["noisyor"]),
    }
    print(f"\n  Δ(saf1600 − 1604): "
          f"{json.dumps(rep['delta_saf1600_minus_ref1604'], ensure_ascii=False)}", flush=True)

    # yeni bundle'ı kaydet (eski 1604 LOSO bundle'ı korunur)
    joblib.dump(arms["saf1600"]["bundle"], B1600)
    print(f"  kaydedildi: {B1600.name} (eski {B1604.name} korundu)", flush=True)

    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n  kayıt: {args.out}", flush=True)


def _cands_from(d, tag, pids):
    """Önbellekten aday koordinatları (cx, cy, s1) sözlüğü — GERÇEK koordinatlar."""
    out = {}
    for i, p in enumerate(pids):
        lo, hi = int(d[f"{tag}_off"][i]), int(d[f"{tag}_off"][i + 1])
        out[p] = [(float(d[f"{tag}_cx"][j]), float(d[f"{tag}_cy"][j]),
                   float(d[f"{tag}_s1"][j])) for j in range(lo, hi)]
    return out


def _d(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


if __name__ == "__main__":
    main()

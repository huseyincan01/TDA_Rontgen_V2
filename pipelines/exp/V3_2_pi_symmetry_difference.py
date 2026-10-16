"""V3.2 — PI simetrik farkı: |PI_1 − PI_4| ve |PI_3 − PI_5| ile topolojik asimetri.

Neden bu deney (V3.1'in doğrudan devamı):
    V3.1 (5 zon arası 10 ikili sliced-Wasserstein mesafesi) **KILL** oldu; kök neden
    **zon şemasının simetrik olmaması** (sağ 3 zon, sol 2 zon) ⇒ 10 çiftin yalnızca
    ~2'si karşılaştırılabilir, gerisi **sabit anatomik farkı** ölçüyor.

    **Bu deney yalnızca GERÇEKTEN ayna olan çiftleri kullanır** ve mesafe yerine
    **PI vektörlerinin eleman-eleman farkını** alır:
        * `|PI_1 − PI_4|`  → sağ üst vs sol üst
        * `|PI_3 − PI_5|`  → sağ alt vs sol alt
    Böylece (a) piksel hizalaması (registration) GEREKMEZ — hata kaynağı yok,
    (b) fark **hangi (birth, persistence) hücresinde** asimetri olduğunu korur
    (skaler mesafeden daha zengin), (c) önbellekteki PI'lar kullanılır.

Ön-kayıtlı hipotez (v4 ölçütü):
    Asimetri temsili mevcut PI füzyonunu (0.9430) **d_fuzyon ≥ +0.015** ile geçer
    **veya** PI ile birleştiğinde geçer.

Tasarım (tek değişken: asimetri özelliğinin tanımı; görüntü dalı sabit):
    * **A** = PI zon dalı (taban) — v3 protokolü, zon başına SVM + hasta max-pool
    * **B** = **ayna asimetrisi**: concat(|PI_1−PI_4|, |PI_3−PI_5|) = 1600
    * **C** = PI + ayna asimetrisi
    * **D** = **YANLIŞ ayna kontrolü**: concat(|PI_1−PI_5|, |PI_3−PI_4|) = 1600
      (çapraz eşleştirme: üst vs alt — aynı sayıda özellik, yanlış simetri ekseni)
    * **E** = **KARIŞTIRILMIŞ kontrol**: gerçek çiftler ama ortak zon **başka hastadan**
    * Her kol için: hasta AUC, füzyon AUC (a validation'da), r(ResNet)

    Projenin **dondurulmuş PI** özellikleri kullanılır (yeniden türetme YOK —
    E2.1/E2.6 dersi).

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test hastaları KULLANILMAZ.

Kullanım:
    python pipelines/exp/V3_2_pi_symmetry_difference.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402

EXPERIMENT_ID = "V3.2"
SLUG = "pi_symmetry_difference"
SEED = 42
MIN_DELTA = 0.015
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
BASELINE_FUSION = 0.9430
# Gercek ayna ciftleri (anatominin izin verdigi olcude)
MIRROR_PAIRS = [(1, 4), (3, 5)]      # sag ust <-> sol ust, sag alt <-> sol alt
WRONG_PAIRS = [(1, 5), (3, 4)]       # capraz: ust <-> alt (yanlis eksen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3.2 PI simetrik farki")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/patient_splits.csv")
    p.add_argument("--zone-splits", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    ps = pd.read_csv(args.splits_path)
    ps = ps[ps["split"].isin(["train", "val"])].dropna(subset=["patient_label"]).copy()
    ps["patient_label"] = ps["patient_label"].astype(int)

    zs = pd.read_csv(args.zone_splits).sort_values("vector_index").reset_index(drop=True)
    Xp = np.load(args.pi_features)

    print("=" * 78)
    print("V3.2 — PI SIMETRIK FARKI: |PI_1-PI_4| ve |PI_3-PI_5|")
    print("=" * 78)
    print(f"PI boyutu: {Xp.shape[1]} | ayna ciftleri: {MIRROR_PAIRS} | "
          f"yanlis ciftler: {WRONG_PAIRS}")
    print("TEST HASTALARI KULLANILMIYOR. Piksel hizalamasi YOK (PI uzayinda).")

    # Zon -> PI vektoru sozlugu (yalnizca train+val satirlari)
    need_rows = zs["split"].isin(["train", "val"]).values
    lut: Dict[tuple, np.ndarray] = {}
    for r in zs[need_rows].itertuples(index=False):
        lut[(r.patient_id, int(r.zone_id))] = Xp[r.vector_index]

    pids = list(ps["patient_id"])
    have_all = [p for p in pids if all((p, z) in lut for z in range(1, 6))]
    print("  5 zonesi da mevcut hasta: %d/%d" % (len(have_all), len(pids)))
    ps = ps[ps["patient_id"].isin(have_all)].reset_index(drop=True)
    pids = list(ps["patient_id"])
    y = ps["patient_label"].values
    tr = (ps["split"] == "train").values
    va = (ps["split"] == "val").values
    print(f"  train {int(tr.sum())} | val {int(va.sum())}")

    def asym(pair_list, shuffle_partner=False, rng=None):
        out = np.zeros((len(pids), len(pair_list) * Xp.shape[1]), dtype=np.float32)
        for a, p in enumerate(pids):
            for c, (i, j) in enumerate(pair_list):
                q = pids[rng.randint(0, len(pids))] if shuffle_partner else p
                out[a, c * Xp.shape[1]:(c + 1) * Xp.shape[1]] = np.abs(lut[(p, i)] - lut[(q, j)])
        return out

    X_mirror = asym(MIRROR_PAIRS)
    X_wrong = asym(WRONG_PAIRS)
    X_shuf = asym(MIRROR_PAIRS, True, np.random.RandomState(SEED + 7))
    print(f"  ayna {X_mirror.shape} | yanlis {X_wrong.shape} | karistirilmis {X_shuf.shape}")
    print(f"  SANITY ayna ilk 5: {np.round(X_mirror[0, :5], 4)}")
    print(f"         yanlis ilk 5: {np.round(X_wrong[0, :5], 4)}")
    print(f"  ayna ortalama |fark| = {X_mirror.mean():.4f} | "
          f"yanlis = {X_wrong.mean():.4f} | karistirilmis = {X_shuf.mean():.4f}")

    # PI zon dali: TUM hastalar icin hasta olasiligi
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    rows = []
    for zz in range(1, 6):
        zm = (zs["zone_id"] == zz).values
        m = zm & zs["split"].isin(["train", "val"]).values
        clfz = make_pipeline(StandardScaler(),
                             SVC(C=10.0, gamma="scale", probability=True, random_state=SEED))
        clfz.fit(Xp[zm & trm], zs["zone_label"].values[zm & trm])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": clfz.predict_proba(Xp[m])[:, 1]}))
    pi_pat = pd.concat(rows).groupby("patient_id")["p"].max()
    pi_all = pi_pat.reindex(pids).values
    piv = pi_pat.reindex(ps.loc[va, "patient_id"].values).values

    # Goruntu dali (val)
    Xr = np.load(args.resnet_features)
    rows = []
    for zz in range(1, 6):
        zm = (zs["zone_id"] == zz).values
        m = zm & (zs["split"] == "val").values
        sc = StandardScaler().fit(Xr[zm & trm])
        lr = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(
            sc.transform(Xr[zm & trm]), zs["zone_label"].values[zm & trm])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": lr.predict_proba(sc.transform(Xr[m]))[:, 1]}))
    img_pat = pd.concat(rows).groupby("patient_id")["p"].max()
    val_pids = ps.loc[va, "patient_id"].values
    yv = y[va]
    iv = img_pat.reindex(val_pids).values

    def pat_svm_full(X):
        clf = make_pipeline(StandardScaler(),
                            SVC(C=1.0, gamma="scale", probability=True,
                                class_weight="balanced", random_state=SEED))
        clf.fit(X[tr], y[tr])
        return clf.predict_proba(X)[:, 1]

    print("\n  SVM'ler egitiliyor...", flush=True)
    p_mirror = pat_svm_full(X_mirror)[va]
    p_wrong = pat_svm_full(X_wrong)[va]
    p_shuf = pat_svm_full(X_shuf)[va]
    p_pi_mirror = pat_svm_full(np.concatenate([X_mirror, pi_all[:, None]], axis=1))[va]
    p_pi_wrong = pat_svm_full(np.concatenate([X_wrong, pi_all[:, None]], axis=1))[va]

    print(f"\n--- SONUC (val, hasta duzeyi) ---")
    print(f"  goruntu dali : {roc_auc_score(yv, iv):.4f}")

    def fusion(tda_vec, label):
        best = (-1.0, None)
        for a in A_GRID:
            s = roc_auc_score(yv, a * iv + (1 - a) * tda_vec)
            if s > best[0]:
                best = (s, float(a))
        rr = float(np.corrcoef(iv, tda_vec)[0, 1])
        single = float(roc_auc_score(yv, tda_vec))
        print(f"  {label:26s} tek={single:.4f} | FUZYON={best[0]:.4f} (a={best[1]:.2f}) | "
              f"r={rr:.3f} | d_fuzyon={best[0]-BASELINE_FUSION:+.4f}")
        return {"tda_alone_val_auc": single, "fusion_val_auc": best[0], "fusion_a": best[1],
                "r": rr, "delta_fusion_vs_PI_baseline": best[0] - BASELINE_FUSION}

    results = {
        "A_PI_zonal": fusion(piv, "A PI (taban)"),
        "B_ayna_asimetri": fusion(p_mirror, "B ayna asimetrisi"),
        "C_PI+ayna": fusion(p_pi_mirror, "C PI + ayna"),
        "D_YANLIS_ayna": fusion(p_wrong, "D YANLIS ayna (kontrol)"),
        "E_KARISTIRILMIS": fusion(p_shuf, "E KARISTIRILMIS (kontrol)"),
        "F_PI+yanlis": fusion(p_pi_wrong, "F PI + yanlis ayna"),
    }

    print(f"\n  --- Simetri kaniti ---")
    print(f"  B ayna tek        {results['B_ayna_asimetri']['tda_alone_val_auc']:.4f}")
    print(f"  D yanlis ayna tek {results['D_YANLIS_ayna']['tda_alone_val_auc']:.4f}")
    print(f"  E karistirilmis   {results['E_KARISTIRILMIS']['tda_alone_val_auc']:.4f}")
    print(f"  fark (B - D) = {results['B_ayna_asimetri']['tda_alone_val_auc'] - results['D_YANLIS_ayna']['tda_alone_val_auc']:+.4f}")

    cand = ["B_ayna_asimetri", "C_PI+ayna"]
    best_tag = max(cand, key=lambda k: results[k]["fusion_val_auc"])
    best_delta = results[best_tag]["delta_fusion_vs_PI_baseline"]
    passed = best_delta >= MIN_DELTA
    print(f"\n  En iyi: {best_tag} (d_fuzyon {best_delta:+.4f}) | kriter >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"PI simetrik farki (ayna ciftleri), fuzyonu >= +{MIN_DELTA} iyilestirir.",
        "scope": "validation only; TEST HASTALARI KULLANILMADI; piksel hizalamasi yok",
        "design": {"features": "concat(|PI_i - PI_j|) for mirror pairs (1,4),(3,5) = 1600",
                   "baseline": "proje dondurulmus PI (yeniden turetim yok)",
                   "controls": {"D": "capraz/yanlis ayna (1,5),(3,4)",
                                "E": "gercek ciftler ama ortak zon baska hastadan"},
                   "baseline_fusion_ref": BASELINE_FUSION},
        "n_patients": {"train": int(tr.sum()), "val": int(va.sum())},
        "results": results, "best_arm": best_tag, "best_delta_fusion": best_delta,
        "symmetry_evidence": {
            "B_mirror_alone": results["B_ayna_asimetri"]["tda_alone_val_auc"],
            "D_wrong_mirror_alone": results["D_YANLIS_ayna"]["tda_alone_val_auc"],
            "E_shuffled_alone": results["E_KARISTIRILMIS"]["tda_alone_val_auc"],
            "B_minus_D": results["B_ayna_asimetri"]["tda_alone_val_auc"]
                         - results["D_YANLIS_ayna"]["tda_alone_val_auc"],
        },
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.zone_splits], seed=SEED,
        decision=report["decision"],
        metrics={"best_delta_fusion": best_delta,
                 "B_ayna_alone": results["B_ayna_asimetri"]["tda_alone_val_auc"],
                 "D_yanlis_alone": results["D_YANLIS_ayna"]["tda_alone_val_auc"]},
        notes=f"PI simetrik farki: en iyi {best_tag} (d_fuzyon {best_delta:+.4f}); karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()

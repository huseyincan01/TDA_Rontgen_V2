"""E2.3b — Sağlıklı şablon, DÜZELTİLMİŞ tasarım (doğrusal olmayan sapma).

Neden bu deney (E2.3'ün düzeltmesi):
    E2.3 **geçersiz** çıktı: `X − T_zon` dönüşümü, zon başına uygulanan `StandardScaler`
    tarafından tamamen yutuluyordu — `(X−T) − mean(X−T) = X − mean(X)`. Üç kol birebir
    aynı sonucu verdi (0.8399/0.8400/0.8399) ve hipotez **test edilemedi**.

    **Düzeltme:** sapmayı **doğrusal olmayan** hâle getir. ReLU (`(X−T)₊`) ve mutlak
    değer (`|X−T|`) dönüşümleri per-feature afin **değildir**, dolayısıyla StandardScaler
    bunları yutamaz. Böylece "şablondan sapma bilgi taşıyor mu?" sorusu **gerçekten**
    test edilebilir.

Ön-kayıtlı hipotez:
    Doğrusal olmayan şablon-göreli temsil (ReLU veya mutlak sapma), mutlak PI'ı
    **val** ROC-AUC'sinde **≥ +0.015** geçer.

Tasarım (boyut EŞLEŞTİRİLMİŞ — hepsi 800 boyut):
    * T_zon = zonun **temiz-negatif TRAIN** PI ortalaması (yalnızca train → sızıntı yok).
    * **A (mutlak)**      : X
    * **B (ReLU sapma)** : (X − T_doğru_zon)₊
    * **C (mutlak sapma)**: |X − T_doğru_zon|
    * **D (kontrol)**    : (X − T_yanlış_zon)₊   ← boyut aynı, şablon YANLIŞ

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/E2_3b_healthy_template_nonlinear.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402

EXPERIMENT_ID = "E2.3b"
SLUG = "healthy_template_nonlinear"
SEED = 42
MIN_DELTA = 0.015


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.3b duzeltilmis saglikli sablon (dogrusal olmayan)")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    return p.parse_args()


def svm_val_patient_auc(X: np.ndarray, zs: pd.DataFrame, meta: pd.DataFrame) -> float:
    y = zs["zone_label"].values
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        va = zm & (zs["split"] == "val").values
        clf = make_pipeline(StandardScaler(),
                            SVC(C=10.0, gamma="scale", probability=True, random_state=SEED))
        clf.fit(X[tr], y[tr])
        rows.append(pd.DataFrame({"patient_id": zs.loc[va, "patient_id"].values,
                                  "p": clf.predict_proba(X[va])[:, 1]}))
    pdf = pd.concat(rows).groupby("patient_id")["p"].max().reindex(meta.index)
    return float(roc_auc_score(meta["patient_label"].values, pdf.values))


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    need = (zs["is_train_balanced_sample"].values.astype(bool)
            | (zs["split"] == "val").values
            | (zs["label_state"] == "negative").values)
    sub = zs[need].reset_index(drop=True)
    X = np.load(args.pi_features)[sub["vector_index"].values]
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")
    zone = sub["zone_id"].values

    print("=" * 78)
    print("E2.3b — SAGLIKLI SABLON, DUZELTILMIS TASARIM (dogrusal OLMAYAN sapma)")
    print("=" * 78)
    print(f"Zon: {len(sub)} | PI: {X.shape} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR. Sablon yalnizca TRAIN satirlarindan.")
    print("NOT: E2.3 gecersizdi (afin donusum scaler tarafindan yutuluyordu);")
    print("     ReLU ve |.| afin OLMADIGI icin bu kez gercekten test edilebilir.")

    templ: Dict[int, np.ndarray] = {}
    for z in range(1, 6):
        m = (zone == z) & (sub["split"] == "train").values & (sub["label_state"] == "negative").values
        templ[z] = X[m].mean(axis=0) if int(m.sum()) >= 30 else np.zeros(X.shape[1], dtype=X.dtype)
        print(f"  T_zon{z}: {int(m.sum())} temiz-negatif train zonundan ortalama")

    T_ok = np.stack([templ[z] for z in zone])
    wrong = {z: (z % 5) + 1 for z in range(1, 6)}
    T_wrong = np.stack([templ[wrong[z]] for z in zone])

    arms = {
        "A_mutlak": X,
        "B_relu_sapma": np.maximum(X - T_ok, 0.0),
        "C_mutlak_sapma": np.abs(X - T_ok),
        "D_kontrol_yanlis_zon": np.maximum(X - T_wrong, 0.0),
    }
    print(f"  Kollar (hepsi {X.shape[1]} boyut — BOYUT ESLESTIRILMIS):")
    for k in arms:
        print(f"    {k}")

    aucs: Dict[str, float] = {}
    for k, V in arms.items():
        print(f"\n  SVM ({k})...", flush=True)
        aucs[k] = svm_val_patient_auc(V, sub, meta)
        print(f"    {k} = {aucs[k]:.4f}", flush=True)

    cand = {"B_relu_sapma": aucs["B_relu_sapma"], "C_mutlak_sapma": aucs["C_mutlak_sapma"]}
    best_name = max(cand, key=lambda k: cand[k])
    best_auc = cand[best_name]
    delta = best_auc - aucs["A_mutlak"]
    ctl_delta = aucs["D_kontrol_yanlis_zon"] - aucs["A_mutlak"]
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC ---")
    for k in arms:
        print(f"  {k:24s}: {aucs[k]:.4f}")
    print(f"  En iyi dogrusal-olmayan: {best_name} = {best_auc:.4f}")
    print(f"  Fark (en iyi - mutlak) : {delta:+.4f}   (kriter >= +{MIN_DELTA} -> {passed})")
    print(f"  Kontrol (yanlis zon)   : {ctl_delta:+.4f}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Sablon-goreli sapma ek bilgi katmiyor; mutlak PI yeterli.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Dogrusal olmayan sablon-goreli temsil, mutlak PI'i val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; sablon yalnizca train satirlarindan (sizinti yok)",
        "fixes": "E2.3 gecersizdi: afin (X - T) donusumu zon-basi StandardScaler tarafindan yutuluyordu. "
                 "ReLU ve |.| afin olmadigi icin bu tasarim hipotezi gercekten test eder.",
        "arms": aucs,
        "best_nonlinear": best_name,
        "best_nonlinear_auc": best_auc,
        "absolute_auc": aucs["A_mutlak"],
        "delta": delta,
        "control_wrong_zone_delta": ctl_delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.pi_features], seed=SEED,
        decision=report["decision"],
        metrics={"absolute_val_auc": aucs["A_mutlak"], "best_nonlinear_val_auc": best_auc, "delta": delta},
        notes=f"E2.3 duzeltmesi: en iyi dogrusal-olmayan {best_name} {best_auc:.4f} vs mutlak "
              f"{aucs['A_mutlak']:.4f}; fark {delta:+.4f}; karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()

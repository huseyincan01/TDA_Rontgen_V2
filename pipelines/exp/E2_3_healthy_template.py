"""E2.3 — Sağlıklı şablon çıkarma (relative persistence).

⚠️⚠️ GEÇERSİZ TASARIM — bu betiğin ölçtüğü şey bir NO-OP'tur (2026-10-16) ⚠️⚠️
    `X_rel = X − T_zon` dönüşümü, `make_pipeline(StandardScaler(), SVC())` içindeki
    **zon başına** StandardScaler tarafından tamamen yutulur:

        (X − T) − mean(X − T) = (X − T) − (mean(X) − T) = X − mean(X)

    Yani mutlak ve göreli temsiller **özdeştir**; std de sabit kaymadan etkilenmez.
    Koşu bunu doğruladı: A = 0.8399, B = 0.8400, C = 0.8399 (farklar SVM numeriği).
    ⇒ **Bu betik hipotezi test EDEMEZ.** Genel kural: **per-feature afin dönüşümler
    StandardScaler tarafından yutulur**; etki için dönüşüm **doğrusal olmayan** ya da
    **özellikler-arası** olmalıdır (örn. (X−T)₊ eşikleme) veya **diyagram düzleminde**
    yapılmalıdır.
    Düzeltilmiş tasarımlar [experiments.md](experiments.md) E2.3 girdisinde listelenmiştir.

Neden bu deney (özgün gerekçe):
    Mevcut PI temsili **mutlak**tır: her zonun diyagramı kendi başına kodlanır. Ancak
    E4.4/E4.2 gösterdi ki zarfın büyük kısmı **arka plan** (bronkovasküler döngüler,
    göğüs duvarı) ve bu arka plan **hasta/zon arasında ortak**. O halde asıl bilgi
    **şablondan sapma**da olabilir.

Ön-kayıtlı hipotez:
    Şablon-göreli temsil (`X − T_z`), mutlak PI'ı **val** ROC-AUC'sinde **≥ +0.015** geçer.
    (Test EDİLEMEDİ — yukarıdaki no-op nedeniyle.)

Tasarım (boyut EŞLEŞTİRİLMİŞ — E2.1/E2.2 dersleri):
    * T_z = zon z'nin **temiz-negatif** TRAIN zonlarının ortalama PI'ı (800 boyut).
    * **A (mutlak)**   : X                       — 800 boyut
    * **B (göreli)**   : X − T_zon               — 800 boyut  ← aynı boyut, adil karşılaştırma
    * **C (kontrol)**  : X − T_yanlış_zon        — 800 boyut  ← boyut aynı, şablon YANLIŞ
      (C, kazancın "doğru zonun şablonundan" mı yoksa "herhangi bir çıkarmadan" mı geldiğini ayırır.)
    Sınıflandırıcı, split ve hasta max-pooling **sabit** (RBF-SVM C=10).

Kapsam (test bütçesi tükendi):
    Yalnızca **validation**. Test setine DOKUNULMAZ.
    Şablon YALNIZCA train satırlarından hesaplanır → val'e sızıntı yok.

Kullanım:
    python pipelines/exp/E2_3_healthy_template.py
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

EXPERIMENT_ID = "E2.3"
SLUG = "healthy_template"
SEED = 42
MIN_DELTA = 0.015


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2.3 saglikli sablon (relative persistence)")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--tag", type=str, default="")
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
    print("E2.3 — SAGLIKLI SABLON (RELATIVE PERSISTENCE)")
    print("=" * 78)
    print(f"Zon: {len(sub)} | PI: {X.shape} | esik +{MIN_DELTA}")
    print("TEST SETINE DOKUNULMUYOR. Sablon yalnizca TRAIN satirlarindan.")

    # --- Sablonlar: zon basina, yalnizca TRAIN + temiz-negatif ---
    templ: Dict[int, np.ndarray] = {}
    for z in range(1, 6):
        m = (zone == z) & (sub["split"] == "train").values & (sub["label_state"] == "negative").values
        n = int(m.sum())
        templ[z] = X[m].mean(axis=0) if n >= 30 else np.zeros(X.shape[1], dtype=X.dtype)
        print(f"  T_zon{z}: {n} temiz-negatif train zonundan ortalama")

    T_correct = np.stack([templ[z] for z in zone])
    rng = np.random.RandomState(SEED)
    wrong = {z: (z % 5) + 1 for z in range(1, 6)}   # zon z -> baska bir zon
    T_wrong = np.stack([templ[wrong[z]] for z in zone])

    X_abs = X
    X_rel = X - T_correct
    X_ctl = X - T_wrong
    print(f"  Temsiller (hepsi {X.shape[1]} boyut — BOYUT ESLESTIRILMIS):")
    print(f"    A mutlak | B = X - T_dogru_zon | C = X - T_yanlis_zon")

    print("\n  SVM (A: mutlak PI)...", flush=True)
    auc_A = svm_val_patient_auc(X_abs, sub, meta)
    print(f"    A = {auc_A:.4f}", flush=True)
    print("  SVM (C: yanlis sablon kontrolu)...", flush=True)
    auc_C = svm_val_patient_auc(X_ctl, sub, meta)
    print(f"    C = {auc_C:.4f}", flush=True)
    print("  SVM (B: dogru zon sablonu)...", flush=True)
    auc_B = svm_val_patient_auc(X_rel, sub, meta)
    print(f"    B = {auc_B:.4f}", flush=True)

    delta = auc_B - auc_A
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC ---")
    print(f"  A mutlak PI              : {auc_A:.4f}")
    print(f"  B = X - T(dogru zon)     : {auc_B:.4f}   ({delta:+.4f})")
    print(f"  C = X - T(yanlis zon)    : {auc_C:.4f}   ({auc_C - auc_A:+.4f})  [kontrol]")
    print(f"  Kriter: >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Sablon-goreli temsil ek bilgi katmiyor; mutlak PI yeterli.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"Sablon-goreli temsil (X - T_zon), mutlak PI'i val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; sablon yalnizca train satirlarindan (sizinti yok)",
        "design": {"template": "zon basina temiz-negatif TRAIN PI ortalamasi",
                   "dimensions": "hepsi 800 — boyut eslestirilmis",
                   "control": "C = X - T(yanlis zon): boyut ayni, sablon yanlis"},
        "absolute_val_auc": auc_A,
        "relative_val_auc": auc_B,
        "control_wrong_zone_val_auc": auc_C,
        "delta": delta,
        "control_delta": auc_C - auc_A,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG + (f"_{args.tag}" if args.tag else ""))
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.pi_features], seed=SEED,
        decision=report["decision"],
        metrics={"absolute_val_auc": auc_A, "relative_val_auc": auc_B, "delta": delta},
        notes=f"Sablon-goreli {auc_B:.4f} vs mutlak {auc_A:.4f}; fark {delta:+.4f}; "
              f"kontrol (yanlis zon) {auc_C:.4f}; karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()

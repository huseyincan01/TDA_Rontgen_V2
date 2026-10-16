"""V1.1 — KRİTİK KAPI: E2.6'nın kazancı füzyonda da duruyor mu?

Neden bu deney (v4'ün ilk kapısı):
    E2.6, TDA dalında **+0.0480** val kazancı verdi (0.8399 → 0.8880). ANCAK bu kazanç
    TDA dalının **tek başına** başarımıdır. Makalenin ana sonucu bir **füzyondur** ve
    füzyonun diğer dalı (**ResNet18**) zaten kenar/doku bilgisini görüyor. E2.6'nın
    eklediği **gradyan** bilgisi ResNet'te zaten varsa, kazanç füzyonda **kaybolur**.

    Bu, v4'ü E2.6 üzerine kurmadan ÖNCE cevaplanması gereken sorudur.

Ön-kayıtlı hipotez:
    E2.6-tabanlı füzyon, mevcut (PI-tabanlı) füzyondan **val** ROC-AUC'sinde
    **≥ +0.015** yüksektir. (Δ < +0.015 ise E2.6'nın füzyon değeri yoktur.)

Tasarım (tek değişken: TDA dalının temsili):
    * Görüntü dalı **sabit**: donmuş ResNet18 + lojistik sonda (E0.3 protokolü, önbellekte).
    * **TDA-A** = mevcut PI (800) + RBF-SVM            → füzyon A
    * **TDA-B** = proje PI + 2 gradyan dilimi (2400) + RBF-SVM → füzyon B
    * Her füzyon için ağırlık ve eşik **yalnızca validation'da** seçilir.
    * Kontrol: **TDA-C** = TDA-B'nin dilimleri karıştırılmış hâli (2400) → füzyon C
      (kazanç gerçekten dilim bilgisinden mi, boyuttan mı?)

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/V1_1_fusion_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
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

EXPERIMENT_ID = "V1.1"
SLUG = "fusion_gate"
SEED = 42
GRID, SIGMA = 20, 1.0
MIN_DELTA = 0.015
T_VALUES = [0.0, 0.5, 1.5]
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CACHE = PROJECT_ROOT / "data/processed/features_multiparam/E2_6_diagrams.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1.1 fuzyon kapisi (E2.6 degeri)")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    return p.parse_args()


def rasterize(dgm: np.ndarray, br, pr) -> np.ndarray:
    img = np.zeros((GRID, GRID), dtype=np.float64)
    if len(dgm) == 0:
        return img
    b, pers = dgm[:, 0], dgm[:, 1] - dgm[:, 0]
    ok = np.isfinite(b) & np.isfinite(pers) & (pers > 0)
    if not np.any(ok):
        return img
    b, pers = np.clip(b[ok], *br), np.clip(pers[ok], *pr)
    bw, pw = max(br[1] - br[0], 1e-9), max(pr[1] - pr[0], 1e-9)
    cols = np.minimum(((b - br[0]) / bw * GRID).astype(int), GRID - 1)
    rows = np.minimum(((pers - pr[0]) / pw * GRID).astype(int), GRID - 1)
    np.add.at(img, (rows, cols), pers)
    return gaussian_filter(img, sigma=SIGMA, mode="constant", truncate=3.0)


def ranges_from(diags, train_mask):
    ds = [d for d, m in zip(diags, train_mask) if m and len(d)]
    if not ds:
        return (0.0, 1.0), (0.0, 1.0)
    allb = np.concatenate([d[:, 0] for d in ds])
    allp = np.concatenate([d[:, 1] - d[:, 0] for d in ds])
    return (float(np.percentile(allb, 0.5)), float(np.percentile(allb, 99.5))), \
           (float(np.percentile(allp, 0.5)), float(np.percentile(allp, 99.5)))


def svm_patient_probs(X: np.ndarray, zs: pd.DataFrame, name: str) -> pd.Series:
    y = zs["zone_label"].values
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & zs["is_train_balanced_sample"].values.astype(bool)
        m = zm & (zs["split"] == name).values
        clf = make_pipeline(StandardScaler(),
                            SVC(C=10.0, gamma="scale", probability=True, random_state=SEED))
        clf.fit(X[tr], y[tr])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": clf.predict_proba(X[m])[:, 1]}))
    return pd.concat(rows).groupby("patient_id")["p"].max()


def resnet_patient_probs(X: np.ndarray, zs: pd.DataFrame, name: str) -> pd.Series:
    y = zs["zone_label"].values
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & trm
        m = zm & (zs["split"] == name).values
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(
            sc.transform(X[tr]), y[tr])
        rows.append(pd.DataFrame({"patient_id": zs.loc[m, "patient_id"].values,
                                  "p": clf.predict_proba(sc.transform(X[m]))[:, 1]}))
    return pd.concat(rows).groupby("patient_id")["p"].max()


def best_fusion(img: pd.Series, tda: pd.Series, meta: pd.DataFrame) -> Tuple[float, float]:
    y = meta["patient_label"].values
    iv, tv = img.reindex(meta.index).values, tda.reindex(meta.index).values
    best = (-1.0, None)
    for a in A_GRID:
        s = roc_auc_score(y, a * iv + (1 - a) * tv)
        if s > best[0]:
            best = (s, float(a))
    return best[0], best[1]


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    zs = pd.read_csv(args.splits_path).sort_values("vector_index").reset_index(drop=True)
    need = zs["is_train_balanced_sample"].values.astype(bool) | (zs["split"] == "val").values
    sub = zs[need].reset_index(drop=True)
    meta = sub[sub["split"] == "val"].drop_duplicates("patient_id").set_index("patient_id")

    print("=" * 78)
    print("V1.1 — KRITIK KAPI: E2.6 kazanci FUZYONDA da duruyor mu?")
    print("=" * 78)
    print(f"Zon: {len(sub)} | esik +{MIN_DELTA} | TEST SETINE DOKUNULMUYOR.")

    X_pi = np.load(args.pi_features)[sub["vector_index"].values]
    X_res = np.load(args.resnet_features)[sub["vector_index"].values]

    z = np.load(CACHE, allow_pickle=True)
    trm = sub["is_train_balanced_sample"].values.astype(bool)
    keys = [f"t{i}" for i in range(len(T_VALUES))]
    blocks = {}
    for k in keys:
        h0, h1 = z[f"{k}_h0"], z[f"{k}_h1"]
        r0, r1 = ranges_from(h0, trm), ranges_from(h1, trm)
        blocks[k] = np.stack([np.concatenate([rasterize(h0[i], *r0).ravel(),
                                              rasterize(h1[i], *r1).ravel()])
                              for i in range(len(sub))])
    rng = np.random.RandomState(SEED)
    X_B = np.concatenate([X_pi] + [blocks[k] for k in keys[1:]], axis=1)
    X_C = np.concatenate([X_pi] + [blocks[k][rng.permutation(len(sub))] for k in keys[1:]], axis=1)
    print(f"  TDA temsilleri: A {X_pi.shape} | B (E2.6) {X_B.shape} | C (kontrol) {X_C.shape}")

    print("\n  Goruntu dali (donmus ResNet18 + lojistik)...", flush=True)
    img = resnet_patient_probs(X_res, sub, "val")
    print(f"    goruntu dalI val AUC = {roc_auc_score(meta['patient_label'].values, img.reindex(meta.index).values):.4f}")

    out: Dict[str, Dict[str, float]] = {}
    for tag, X in (("A_PI", X_pi), ("B_E2.6", X_B), ("C_kontrol", X_C)):
        print(f"  TDA dali ({tag}) SVM egitiliyor...", flush=True)
        tda = svm_patient_probs(X, sub, "val")
        tda_auc = float(roc_auc_score(meta["patient_label"].values, tda.reindex(meta.index).values))
        fus_auc, a = best_fusion(img, tda, meta)
        out[tag] = {"tda_alone_val_auc": tda_auc, "fusion_val_auc": fus_auc, "fusion_a": a}
        print(f"    {tag}: TDA-tek={tda_auc:.4f} | FUZYON={fus_auc:.4f} (a={a:.2f})", flush=True)

    base = out["A_PI"]["fusion_val_auc"]
    delta = out["B_E2.6"]["fusion_val_auc"] - base
    ctl = out["C_kontrol"]["fusion_val_auc"] - base
    passed = delta >= MIN_DELTA
    print(f"\n--- SONUC (val fuzyon) ---")
    print(f"  Fuzyon A (mevcut PI)      : {base:.4f}  (a={out['A_PI']['fusion_a']:.2f})")
    print(f"  Fuzyon B (E2.6 dilimleri) : {out['B_E2.6']['fusion_val_auc']:.4f}  "
          f"(a={out['B_E2.6']['fusion_a']:.2f})  fark {delta:+.4f}")
    print(f"  Fuzyon C (kontrol)        : {out['C_kontrol']['fusion_val_auc']:.4f}  fark {ctl:+.4f}")
    print(f"  Kriter: B - A >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if passed:
        print("  => E2.6 kazanci FUZYONDA DURUYOR: v4 bunun uzerine kurulabilir.")
    else:
        print("  => E2.6 kazanci fuzyonda KAYBOLUYOR: TDA-tek basarim artisi ResNet'in")
        print("     zaten bildigi bilgiden geliyor olabilir. v4 yonunu gozden gecir.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"E2.6 tabanli fuzyon, mevcut PI fuzyonundan val'de >= +{MIN_DELTA} yuksektir.",
        "scope": "validation only; test setine dokunulmadi (butce kapali)",
        "single_variable": "TDA dalinin temsili (PI vs PI+gradyan dilimleri); goruntu dali sabit",
        "results": out,
        "baseline_fusion_val_auc": base,
        "delta_e26_minus_pi": delta,
        "delta_control_minus_pi": ctl,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
        "interpretation": ("E2.6 kazanci fuzyonda duruyor" if passed
                           else "E2.6 kazanci fuzyonda kayboluyor"),
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.pi_features, args.resnet_features],
        seed=SEED, decision=report["decision"],
        metrics={"fusion_A_val_auc": base, "fusion_B_val_auc": out["B_E2.6"]["fusion_val_auc"],
                 "delta": delta},
        notes=f"Fuzyon kapisi: A={base:.4f}, B(E2.6)={out['B_E2.6']['fusion_val_auc']:.4f}, "
              f"fark {delta:+.4f}; karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()

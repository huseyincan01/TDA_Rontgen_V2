"""V1.2 — KAPI (devam): Hangi TDA temsili FÜZYONA gerçekten katkı sağlıyor?

Neden bu deney:
    V1.1 kritik bir şey gösterdi: **E2.6'nın TDA-tek kazancı (+0.0480) füzyonda
    +0.0072'ye düşüyor** ⇒ gradyan bilgisi ResNet18'de zaten var, yedekli.

    Bu, v4 için **arama kriterini değiştiriyor**: TDA dalını *tek başına* iyileştirmek
    yeterli değil; asıl ölçüt **füzyon kazancı** olmalı. Bu deney tüm aday temsilleri
    **füzyon ölçütüyle** karşılaştırır.

Ön-kayıtlı hipotez:
    En az bir aday temsil, mevcut PI füzyonunu val'de **≥ +0.015** geçer.

Tasarım (tek değişken: TDA temsilim; görüntü dalı sabit):
    * Görüntü dalı: donmuş ResNet18 + lojistik sonda (sabit).
    * **A** = mevcut PI (800)                          ← referans
    * **B** = E2.6: PI + 2 gradyan dilimi (2400)
    * **C** = E2.1: PI + ters polarite (diphasic, 1600)
    * **D** = E2.6 + E2.1 birleşik (3200)
    Her temsil için TDA-tek AUC **ve** füzyon AUC raporlanır (ikisi arasındaki fark
    "yedeklilik" göstergesidir).

Kapsam (test bütçesi KAPALI):
    Yalnızca **validation**. Test setine DOKUNULMAZ.

Kullanım:
    python pipelines/exp/V1_2_fusion_rep_comparison.py
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

EXPERIMENT_ID = "V1.2"
SLUG = "fusion_rep_comparison"
SEED = 42
GRID, SIGMA = 20, 1.0
MIN_DELTA = 0.015
A_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
CACHE_E26 = PROJECT_ROOT / "data/processed/features_multiparam/E2_6_diagrams.npz"
CACHE_E21 = PROJECT_ROOT / "data/processed/features_diphasic/E2_1_diagrams.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1.2 fuzyon temsil karsilastirmasi")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--pi-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_v2/X_zone_features.npy")
    p.add_argument("--resnet-features", type=Path,
                   default=PROJECT_ROOT / "data/processed/features_resnet18/X_zone_features.npy")
    return p.parse_args()


def rasterize(dgm, br, pr) -> np.ndarray:
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


def ranges_from(diags, trm):
    ds = [d for d, m in zip(diags, trm) if m and len(d)]
    if not ds:
        return (0.0, 1.0), (0.0, 1.0)
    ab = np.concatenate([d[:, 0] for d in ds]); ap = np.concatenate([d[:, 1] - d[:, 0] for d in ds])
    return (float(np.percentile(ab, 0.5)), float(np.percentile(ab, 99.5))), \
           (float(np.percentile(ap, 0.5)), float(np.percentile(ap, 99.5)))


def block_from_arrays(pairs, trm, n):
    """pairs: [(h0_array, h1_array), ...] — her cift bagimsiz bir PI blogu uretir."""
    blocks = []
    for h0, h1 in pairs:
        r0, r1 = ranges_from(h0, trm), ranges_from(h1, trm)
        blocks.append(np.stack([np.concatenate([rasterize(h0[i], *r0).ravel(),
                                                rasterize(h1[i], *r1).ravel()])
                                for i in range(n)]))
    return blocks


def svm_patient_probs(X, zs, name) -> pd.Series:
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


def resnet_patient_probs(X, zs, name) -> pd.Series:
    y = zs["zone_label"].values
    trm = zs["is_train_balanced_sample"].values.astype(bool)
    rows = []
    for z in range(1, 6):
        zm = (zs["zone_id"] == z).values
        tr = zm & trm
        m = zm & (zs["split"] == name).values
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
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
    trm = sub["is_train_balanced_sample"].values.astype(bool)
    n = len(sub)

    print("=" * 78)
    print("V1.2 — HANGI TDA TEMSILI FUZYONA GERCEKTEN KATKI SAGLIYOR?")
    print("=" * 78)
    print(f"Zon: {n} | esik +{MIN_DELTA} | TEST SETINE DOKUNULMUYOR.")
    print("NOT: V1.1'de E2.6'nin TDA-tek kazanci (+0.0480) fuzyonda +0.0072'ye dustu.")
    print("     Bu deney tum adaylari FUZYON olcutuyle karsilastirir.")

    X_pi = np.load(args.pi_features)[sub["vector_index"].values]
    X_res = np.load(args.resnet_features)[sub["vector_index"].values]

    z26 = np.load(CACHE_E26, allow_pickle=True)
    z21 = np.load(CACHE_E21, allow_pickle=True)
    b_e26 = block_from_arrays([(z26["t1_h0"], z26["t1_h1"]), (z26["t2_h0"], z26["t2_h1"])], trm, n)
    b_e21 = block_from_arrays([(z21["B0"], z21["B1"])], trm, n)

    reps: Dict[str, np.ndarray] = {
        "A_PI": X_pi,
        "B_E26_gradyan": np.concatenate([X_pi] + b_e26, axis=1),
        "C_E21_diphasic": np.concatenate([X_pi] + b_e21, axis=1),
        "D_Birlesik": np.concatenate([X_pi] + b_e26 + b_e21, axis=1),
    }
    for k, v in reps.items():
        print(f"  {k}: {v.shape}")

    print("\n  Goruntu dali (donmus ResNet18)...", flush=True)
    img = resnet_patient_probs(X_res, sub, "val")
    img_auc = float(roc_auc_score(meta["patient_label"].values, img.reindex(meta.index).values))
    print(f"    goruntu tek basina = {img_auc:.4f}")

    out: Dict[str, Dict[str, float]] = {}
    for tag, X in reps.items():
        print(f"\n  SVM ({tag})...", flush=True)
        tda = svm_patient_probs(X, sub, "val")
        tda_auc = float(roc_auc_score(meta["patient_label"].values, tda.reindex(meta.index).values))
        fus_auc, a = best_fusion(img, tda, meta)
        out[tag] = {"tda_alone_val_auc": tda_auc, "fusion_val_auc": fus_auc, "fusion_a": a,
                    "redundancy_loss": tda_auc - (tda_auc - 0.8399) if tag != "A_PI" else 0.0}
        print(f"    TDA-tek={tda_auc:.4f} | FUZYON={fus_auc:.4f} (a={a:.2f})", flush=True)

    base = out["A_PI"]["fusion_val_auc"]
    print(f"\n--- SONUC (val) ---")
    print(f"  {'Temsil':22s} {'TDA-tek':>9} {'FUZYON':>9} {'d_fuzyon':>9} {'d_tek':>9}")
    best_tag, best_delta = None, -9.0
    for tag, r in out.items():
        d = r["fusion_val_auc"] - base
        dt = r["tda_alone_val_auc"] - out["A_PI"]["tda_alone_val_auc"]
        print(f"  {tag:22s} {r['tda_alone_val_auc']:9.4f} {r['fusion_val_auc']:9.4f} "
              f"{d:+9.4f} {dt:+9.4f}")
        if tag != "A_PI" and d > best_delta:
            best_delta, best_tag = d, tag
    passed = best_delta >= MIN_DELTA
    print(f"\n  En iyi: {best_tag} (d_fuzyon {best_delta:+.4f}) | kriter >= +{MIN_DELTA} -> {passed}")
    print(f"  KARAR: {'KEEP' if passed else 'KILL'}")
    if not passed:
        print("  => Hicbir aday fuzyona >= +0.015 katki saglamiyor.")
        print("     TDA dalindaki kazananlar ResNet ile YEDEKLI gorunuyor.")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": f"En az bir aday temsil, mevcut PI fuzyonunu val'de >= +{MIN_DELTA} gecer.",
        "scope": "validation only; test setine dokunulmadi",
        "single_variable": "TDA temsili; goruntu dali sabit",
        "image_branch_alone_val_auc": img_auc,
        "representations": out,
        "baseline_fusion_val_auc": base,
        "best_candidate": best_tag,
        "best_fusion_delta": best_delta,
        "criteria": {"delta_ge_0.015": bool(passed)},
        "decision": "KEEP" if passed else "KILL",
        "key_metric_note": "d_fuzyon esas alinir; d_tek tek basina yanilticidir (V1.1).",
    }
    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path, args.pi_features, args.resnet_features],
        seed=SEED, decision=report["decision"],
        metrics={"baseline_fusion_val_auc": base, "best_fusion_delta": best_delta},
        notes=f"Fuzyon temsil karsilastirmasi: en iyi {best_tag} ({best_delta:+.4f}); "
              f"karar={report['decision']}.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()

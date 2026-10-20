"""V5.3 vs V4 (eski temsil): aynı 400 hastada eşleştirilmiş bootstrap — düzeltmenin harici etkisi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/runs/V4_vindr_external_validation/vindr_predictions_matched.csv"
NEW = ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv"
N, SEED = 2000, 42


def main() -> None:
    old = pd.read_csv(OLD).set_index("image_id")
    new = pd.read_csv(NEW).set_index("image_id")
    idx = new.index.intersection(old.index)
    new, old = new.loc[idx], old.loc[idx]
    y = new["patient_label"].values
    a = 0.45

    fus_new = a * new["p_resnet"].values + (1 - a) * new["p_tda"].values
    fus_old = a * old["p_resnet"].values + (1 - a) * old["p_tda"].values

    for name, p_new, p_old in (("TDA", new["p_tda"].values, old["p_tda"].values),
                               ("Füzyon (a=0.45)", fus_new, fus_old)):
        rng = np.random.RandomState(SEED)
        d = []
        for _ in range(N):
            i = rng.randint(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            d.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
        d = np.asarray(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        print(f"{name:16s} eski {roc_auc_score(y, p_old):.4f} → yeni {roc_auc_score(y, p_new):.4f} "
              f"| Δ {roc_auc_score(y, p_new) - roc_auc_score(y, p_old):+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] p={p:.4f}  (n={len(y)})")


if __name__ == "__main__":
    main()

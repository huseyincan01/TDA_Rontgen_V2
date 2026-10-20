"""V5.3: füzyon − ResNet farkının eşleştirilmiş bootstrap %95 GA'sı + alt grup özeti."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "artifacts/runs/V5.3_corrected_tda_vindr/vindr_predictions_v5_3.csv"
REPORT = ROOT / "data/external_vindr/v5_3_corrected_tda_vindr_report.json"
N, SEED = 2000, 42


def main() -> None:
    df = pd.read_csv(PRED)
    y = df["patient_label"].values
    tda, res = df["p_tda"].values, df["p_resnet"].values

    for a in (0.45, 0.25, 0.75):
        fus = a * res + (1 - a) * tda
        rng = np.random.RandomState(SEED)
        d = []
        for _ in range(N):
            i = rng.randint(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            d.append(roc_auc_score(y[i], fus[i]) - roc_auc_score(y[i], res[i]))
        d = np.asarray(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        print(f"a={a:.2f}: füzyon {roc_auc_score(y, fus):.4f} | Δ(resnet) {d.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] p={p:.4f}")

    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    print("\nAlt gruplar (rel_size üçte birlik, V5.3):")
    for k, v in rep["subgroups_scale_free"].items():
        print(f"  {k:16s} n={v['n_positive']:3d} | TDA {v['tda_auc']:.4f} | ResNet {v['resnet_auc']:.4f} "
              f"| Füzyon {v['fusion_auc']:.4f}")
    print("\nV4 karşılaştırması:", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in rep["comparison_vs_v4"].items()})


if __name__ == "__main__":
    main()

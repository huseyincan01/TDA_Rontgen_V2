"""v2 Zon Splitleri ve Dengeli Eğitim Alt Kümesi Oluşturma Modülü.

Kurallar:
- `features_metadata.csv` (v2) ile `patient_splits.csv` (v2) dosyalarını birleştirir.
- Train'de her zon (1..5) için pozitif sayısı kadar negatif örnek seçer (1:1 dengeleme).
- Negatif adaylar YALNIZCA `label_state == 'negative'` olan temiz zonlar arasından seçilir.
- `ambiguous` (%0-%20 örtüşen) ve `unassigned_nodule` satırları ASLA eğitim negatif havuzuna alınmaz.
- `is_zone_metric_sample`: Yalnızca `label_state in ['positive', 'negative']` olan temiz zonlar için True'dur;
  ambiguous ve unassigned satırlar zon metriklerine girmez.
- Test ve validation kümelerinde dengeleme YAPILMAZ (doğal prevalans korunur).
- Çıktı: `data/processed/splits_v2/zone_splits.csv`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_zone_splits_v2(
    features_meta_df: pd.DataFrame,
    patient_splits_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Zon metadata ile hasta splitlerini birleştirir ve train 1:1 temiz negatif dengelemesi yapar."""
    rng = np.random.RandomState(seed)

    # 1. Merge patient split info
    p_map = patient_splits_df.set_index("patient_id")[
        ["split", "zone_pattern", "pos_zone_count"]
    ].to_dict(orient="index")

    zone_df = features_meta_df.copy()
    zone_df["split"] = zone_df["patient_id"].map(lambda pid: p_map[pid]["split"])
    zone_df["zone_pattern"] = zone_df["patient_id"].map(lambda pid: p_map[pid]["zone_pattern"])
    zone_df["pos_zone_count"] = zone_df["patient_id"].map(lambda pid: p_map[pid]["pos_zone_count"])

    if zone_df["split"].isna().any():
        missing_count = int(zone_df["split"].isna().sum())
        raise ValueError(f"{missing_count} zon satırı için split bilgisi eşleşmedi!")

    zone_df["is_train_balanced_sample"] = False
    # ambiguous ve unassigned zonlar zon metriklerine girmez
    zone_df["is_zone_metric_sample"] = zone_df["label_state"].isin(["positive", "negative"])

    zone_balance_stats: Dict[str, Any] = {}

    # 2. Her zon için train dengelemesi yap
    for zone_id in sorted(zone_df["zone_id"].unique()):
        train_z = zone_df[(zone_df["split"] == "train") & (zone_df["zone_id"] == zone_id)]

        pos_idx = train_z[train_z["label_state"] == "positive"].index
        n_pos = len(pos_idx)

        # Negatif adaylar YALNIZCA label_state == "negative" olan temiz zonlar
        neg_candidates_idx = train_z[train_z["label_state"] == "negative"].index
        n_clean_neg_available = len(neg_candidates_idx)

        if n_pos > n_clean_neg_available:
            raise ValueError(
                f"Zon {zone_id} için pozitif sayısı ({n_pos}) mevcut temiz negatif sayısından ({n_clean_neg_available}) fazla!"
            )

        selected_neg_idx = rng.choice(neg_candidates_idx, size=n_pos, replace=False)

        zone_df.loc[pos_idx, "is_train_balanced_sample"] = True
        zone_df.loc[selected_neg_idx, "is_train_balanced_sample"] = True

        # Validation ve Test sayıları
        val_z = zone_df[(zone_df["split"] == "val") & (zone_df["zone_id"] == zone_id)]
        test_z = zone_df[(zone_df["split"] == "test") & (zone_df["zone_id"] == zone_id)]

        zone_balance_stats[f"zone_{zone_id}"] = {
            "zone_name": train_z["zone_name"].iloc[0],
            "train": {
                "positive": n_pos,
                "clean_negative_available": n_clean_neg_available,
                "clean_negative_selected": n_pos,
                "balanced_total": 2 * n_pos,
                "ambiguous_excluded": int((train_z["label_state"] == "ambiguous").sum()),
                "unassigned_excluded": int((train_z["label_state"] == "unassigned_nodule").sum()),
                "total_train_zone_rows": len(train_z),
            },
            "validation": {
                "positive": int((val_z["label_state"] == "positive").sum()),
                "clean_negative": int((val_z["label_state"] == "negative").sum()),
                "ambiguous": int((val_z["label_state"] == "ambiguous").sum()),
                "unassigned": int((val_z["label_state"] == "unassigned_nodule").sum()),
                "clean_metric_total": int(val_z["is_zone_metric_sample"].sum()),
                "total": len(val_z),
            },
            "test": {
                "positive": int((test_z["label_state"] == "positive").sum()),
                "clean_negative": int((test_z["label_state"] == "negative").sum()),
                "ambiguous": int((test_z["label_state"] == "ambiguous").sum()),
                "unassigned": int((test_z["label_state"] == "unassigned_nodule").sum()),
                "clean_metric_total": int(test_z["is_zone_metric_sample"].sum()),
                "total": len(test_z),
            },
        }

    report = {
        "pipeline_version": "v2",
        "seed": seed,
        "total_rows": len(zone_df),
        "total_patients": int(zone_df["patient_id"].nunique()),
        "zone_balance_statistics": zone_balance_stats,
    }

    return zone_df, report

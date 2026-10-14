"""Hasta düzeyinde v2 katmanlı veri bölme (Train/Val/Test) modülü.

Kurallar:
- Eski splitleri ve source_dataset bilgisini KULLANMAZ (sağlıklıların özgün kaynağı kayıp olduğu için).
- Katmanlamada yalnızca `patient_label` ve `zone_pattern` (5-zon pozitiflik örüntüsü, örn. '01000') kullanılır.
- Nadir zon örüntüleri (< 3 örnek) için güvenli açık-tabanlı (deficit-based) fallback uygular.
- Etiket dosyasının SHA-256 hash'ini, seed ve oranları split manifestine yazar.
- Hasta sızıntısını kesinlikle engeller: her hasta yalnızca tek bir split kümesinde (train/val/test) bulunur.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfigV2:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"Oranlar toplamı 1.0 olmalıdır, gelen: {total}")


def compute_file_sha256(filepath: Path) -> str:
    """Dosyanın SHA-256 hash'ini hesaplar."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_patients_v2(
    labels_df: pd.DataFrame,
    labels_file_path: Path | None = None,
    config: SplitConfigV2 | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Hastaları 'patient_label' ve 'zone_pattern' katmanlamasıyla böler."""
    if config is None:
        config = SplitConfigV2()

    rng = np.random.RandomState(config.seed)

    # 1. Hasta bazında tekil tablo ve 5-zon pozitiflik örüntüsü
    piv = labels_df.pivot(index="patient_id", columns="zone_id", values="zone_label")
    sorted_zone_cols = sorted(piv.columns)
    pattern_series = piv[sorted_zone_cols].astype(int).astype(str).agg("".join, axis=1)

    patient_meta = labels_df.groupby("patient_id").agg({
        "patient_label": "first",
    })
    patient_meta["zone_pattern"] = pattern_series
    patient_meta["pos_zone_count"] = piv[sorted_zone_cols].sum(axis=1)
    patient_meta["strat_key"] = (
        patient_meta["patient_label"].astype(str) + "_" + patient_meta["zone_pattern"]
    )

    splits: Dict[str, str] = {}
    fallback_records: List[Dict[str, Any]] = []

    # Grupları büyükten küçüğe sıralayarak deterministik sıra sağla
    strata_groups = list(patient_meta.groupby("strat_key"))
    # Sıralama: grup büyüklüğü azalan, sonra anahtar ismi
    strata_groups.sort(key=lambda x: (-len(x[1]), x[0]))

    for strat_key, group in strata_groups:
        n = len(group)
        pids = group.index.tolist()
        shuffled_ids = rng.permutation(pids).tolist()

        if n >= 3:
            n_val = max(1, int(round(n * config.val_ratio)))
            n_test = max(1, int(round(n * config.test_ratio)))
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1
            n_train = n - n_val - n_test

            for pid in shuffled_ids[:n_train]:
                splits[pid] = "train"
            for pid in shuffled_ids[n_train : n_train + n_val]:
                splits[pid] = "val"
            for pid in shuffled_ids[n_train + n_val :]:
                splits[pid] = "test"
        else:
            # Nadir örüntüler (< 3 hasta): Hedef oran açıklarına (deficit) göre ata
            assigned_this_stratum = []
            for pid in shuffled_ids:
                c_train = sum(1 for s in splits.values() if s == "train")
                c_val = sum(1 for s in splits.values() if s == "val")
                c_test = sum(1 for s in splits.values() if s == "test")
                tot = max(1, c_train + c_val + c_test)

                deficits = [
                    (config.train_ratio - (c_train / tot), "train"),
                    (config.val_ratio - (c_val / tot), "val"),
                    (config.test_ratio - (c_test / tot), "test"),
                ]
                deficits.sort(key=lambda x: x[0], reverse=True)
                chosen_split = deficits[0][1]
                splits[pid] = chosen_split
                assigned_this_stratum.append((pid, chosen_split))

            fallback_records.append({
                "strat_key": strat_key,
                "count": n,
                "assignments": assigned_this_stratum,
            })

    patient_meta["split"] = patient_meta.index.map(splits)
    patient_splits_df = patient_meta.reset_index()

    cols_order = [
        "patient_id",
        "split",
        "patient_label",
        "zone_pattern",
        "pos_zone_count",
    ]
    patient_splits_df = patient_splits_df[cols_order].sort_values("patient_id").reset_index(drop=True)

    # 2. Manifest / Özet rapor
    label_hash = compute_file_sha256(labels_file_path) if labels_file_path and labels_file_path.exists() else "unknown"

    patient_counts_by_split = patient_splits_df["split"].value_counts().to_dict()
    patient_ratios_by_split = (
        patient_splits_df["split"].value_counts(normalize=True).round(4).to_dict()
    )

    label_dist: Dict[str, Dict[str, int]] = {}
    for lbl, grp in patient_splits_df.groupby("patient_label"):
        label_dist[str(lbl)] = grp["split"].value_counts().to_dict()

    summary_report: Dict[str, Any] = {
        "pipeline_version": "v2",
        "seed": config.seed,
        "ratios": {
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
        "labels_file_sha256": label_hash,
        "total_patients": len(patient_splits_df),
        "split_counts": patient_counts_by_split,
        "split_ratios": patient_ratios_by_split,
        "patient_label_distribution": label_dist,
        "stratification_factors": ["patient_label", "zone_pattern"],
        "source_dataset_used_for_stratification": False,
        "fallback_rare_strata_count": len(fallback_records),
        "fallback_rare_strata_details": fallback_records,
    }

    return patient_splits_df, summary_report

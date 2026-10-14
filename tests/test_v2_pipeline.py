"""v2 Veri Hattı Otomatik Birim ve Bütünlük Testleri."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd


def load_v2_files():
    labels_path = PROJECT_ROOT / "data" / "processed" / "labels" / "zone_labels_v2.csv"
    patient_splits_path = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "patient_splits.csv"
    zone_splits_path = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "zone_splits.csv"
    features_path = PROJECT_ROOT / "data" / "processed" / "features_v2" / "X_zone_features.npy"
    features_meta_path = PROJECT_ROOT / "data" / "processed" / "features_v2" / "features_metadata.csv"

    return {
        "labels": pd.read_csv(labels_path),
        "patient_splits": pd.read_csv(patient_splits_path),
        "zone_splits": pd.read_csv(zone_splits_path),
        "features": np.load(features_path),
        "features_meta": pd.read_csv(features_meta_path),
    }


def test_no_patient_leakage(v2):
    p_df = v2["patient_splits"]
    tr_pids = set(p_df[p_df["split"] == "train"]["patient_id"])
    val_pids = set(p_df[p_df["split"] == "val"]["patient_id"])
    ts_pids = set(p_df[p_df["split"] == "test"]["patient_id"])

    assert len(tr_pids & val_pids) == 0, "Train-Val sızıntısı var!"
    assert len(tr_pids & ts_pids) == 0, "Train-Test sızıntısı var!"
    assert len(val_pids & ts_pids) == 0, "Val-Test sızıntısı var!"
    assert len(p_df) == 4882
    assert p_df["patient_id"].nunique() == 4882
    print("  [PASS] test_no_patient_leakage")


def test_zone_splits_1to1_train_balance(v2):
    z_df = v2["zone_splits"]
    for zid in range(1, 6):
        train_z = z_df[(z_df["split"] == "train") & (z_df["zone_id"] == zid)]
        balanced = train_z[train_z["is_train_balanced_sample"]]
        pos_cnt = int((balanced["label_state"] == "positive").sum())
        neg_cnt = int((balanced["label_state"] == "negative").sum())
        assert pos_cnt == neg_cnt, f"Zon {zid} pos/neg dengesiz!"
        assert len(balanced) == 2 * pos_cnt
    print("  [PASS] test_zone_splits_1to1_train_balance")


def test_no_ambiguous_in_train_negatives(v2):
    z_df = v2["zone_splits"]
    balanced_all = z_df[z_df["is_train_balanced_sample"]]
    assert not any(balanced_all["label_state"] == "ambiguous"), "Ambiguous satır train negatifinde!"
    assert not any(balanced_all["label_state"] == "unassigned_nodule"), "Unassigned satır train negatifinde!"
    assert set(balanced_all["label_state"].unique()) == {"positive", "negative"}
    print("  [PASS] test_no_ambiguous_in_train_negatives")


def test_dimensions_and_index_alignment(v2):
    X = v2["features"]
    meta = v2["features_meta"]
    z_df = v2["zone_splits"]

    assert X.shape == (24410, 800), f"X boyutu hatalı: {X.shape}"
    assert len(meta) == 24410
    assert len(z_df) == 24410
    assert (meta["vector_index"].values == np.arange(24410)).all()
    assert (z_df["vector_index"].values == np.arange(24410)).all()
    print("  [PASS] test_dimensions_and_index_alignment")


def test_features_are_finite_and_nonconstant(v2):
    X = v2["features"]
    assert np.isfinite(X).all(), "Özellik matrisinde NaN veya sonsuz değer var!"
    assert float(X[:, :400].std()) > 0, "H0 özellikleri sabit!"
    assert float(X[:, 400:].std()) > 0, "H1 özellikleri sabit!"
    print("  [PASS] test_features_are_finite_and_nonconstant")


def main():
    print("=" * 60)
    print("v2 VERİ HATTI BİRİM TESTLERİ ÇALIŞTIRILIYOR")
    print("=" * 60)
    v2 = load_v2_files()
    test_no_patient_leakage(v2)
    test_zone_splits_1to1_train_balance(v2)
    test_no_ambiguous_in_train_negatives(v2)
    test_dimensions_and_index_alignment(v2)
    test_features_are_finite_and_nonconstant(v2)
    print("=" * 60)
    print("TÜM BİRİM TESTLER BAŞARIYLA GEÇTİ!")
    print("=" * 60)


if __name__ == "__main__":
    main()

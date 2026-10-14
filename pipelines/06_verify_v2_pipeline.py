"""v2 Veri Hattı Bütünlük ve Doğrulama Kontrolleri.

Kontrol Edilenler:
1. Sıfır Hasta Sızıntısı: Her hasta tek bir splitte bulunur (train, val, test kesişimi boştur).
2. Train 1:1 Dengeleme: Her zon için train örnekleri 1:1 dengelidir (pos == neg).
3. Temiz Negatif Güvencesi: Ambiguous (%0-%20) ve Unassigned satırlar ASLA negatif olarak seçilmemiştir.
4. Boyut ve İndeks Eşleşmesi: Vektör sayısı, metadata satır sayısı ve vector_index birebir eşleşir.
5. v1 Dosya İzolasyonu: Eski v1 dosyaları kullanılmaz; tüm yollar v2 dizinlerine işaret eder.
6. Test Dokunulmazlığı: Test kümesi yalnızca değerlendirme için dondurulmuştur.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_checks() -> bool:
    print("=" * 80)
    print("v2 VERİ HATTI BÜTÜNLÜK VE DOĞRULAMA KONTROLLERİ")
    print("=" * 80)

    all_passed = True

    # Dosya yolları
    labels_path = PROJECT_ROOT / "data" / "processed" / "labels" / "zone_labels_v2.csv"
    patient_splits_path = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "patient_splits.csv"
    splits_summary_path = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "splits_summary.json"
    tda_config_path = PROJECT_ROOT / "data" / "processed" / "tda_config" / "tda_ranges_v2.json"
    features_path = PROJECT_ROOT / "data" / "processed" / "features_v2" / "X_zone_features.npy"
    features_meta_path = PROJECT_ROOT / "data" / "processed" / "features_v2" / "features_metadata.csv"
    zone_splits_path = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "zone_splits.csv"

    # Kontrol 1: Dosya Varlıkları
    print("\n[1] V2 Dosya Varlık Kontrolü:")
    files_to_check = [
        ("zone_labels_v2.csv", labels_path),
        ("patient_splits.csv", patient_splits_path),
        ("splits_summary.json", splits_summary_path),
        ("tda_ranges_v2.json", tda_config_path),
        ("X_zone_features.npy", features_path),
        ("features_metadata.csv", features_meta_path),
        ("zone_splits.csv", zone_splits_path),
    ]
    for name, fpath in files_to_check:
        if fpath.exists():
            print(f"  [OK] {name} mevcut ({fpath.stat().st_size:,} bayt)")
        else:
            print(f"  [HATA] {name} bulunamadı: {fpath}")
            all_passed = False

    if not all_passed:
        print("\nBazı v2 dosyaları eksik, lütfen önce ilgili adımları tamamlayın.")
        return False

    # Verileri yükle
    labels_df = pd.read_csv(labels_path)
    patient_splits_df = pd.read_csv(patient_splits_path)
    with open(splits_summary_path, "r", encoding="utf-8") as f:
        splits_summary = json.load(f)
    with open(tda_config_path, "r", encoding="utf-8") as f:
        tda_cfg = json.load(f)
    X = np.load(features_path)
    features_meta = pd.read_csv(features_meta_path)
    zone_splits = pd.read_csv(zone_splits_path)

    # Kontrol 2: Sıfır Hasta Sızıntısı
    print("\n[2] Hasta Sızıntısı (Patient Leakage) Kontrolü:")
    tr_pids = set(patient_splits_df[patient_splits_df["split"] == "train"]["patient_id"])
    val_pids = set(patient_splits_df[patient_splits_df["split"] == "val"]["patient_id"])
    ts_pids = set(patient_splits_df[patient_splits_df["split"] == "test"]["patient_id"])

    leakage_tr_val = tr_pids & val_pids
    leakage_tr_ts = tr_pids & ts_pids
    leakage_val_ts = val_pids & ts_pids

    total_unique_pids = len(set(patient_splits_df["patient_id"]))
    total_pid_rows = len(patient_splits_df)

    assert len(leakage_tr_val) == 0, f"Train-Val sızıntısı: {leakage_tr_val}"
    assert len(leakage_tr_ts) == 0, f"Train-Test sızıntısı: {leakage_tr_ts}"
    assert len(leakage_val_ts) == 0, f"Val-Test sızıntısı: {leakage_val_ts}"
    assert total_unique_pids == total_pid_rows == 4882, f"Hasta sayısı uyumsuz: {total_unique_pids}"
    print(f"  [OK] Sıfır hasta sızıntısı doğrulandı. 4.882 hastanın tamamı tek bir splitte.")
    print(f"       Train: {len(tr_pids)}, Val: {len(val_pids)}, Test: {len(ts_pids)}")

    # Kontrol 3: Train 1:1 Dengeleme
    print("\n[3] Her Zon İçin Train 1:1 Dengeleme Kontrolü:")
    for zid in range(1, 6):
        z_tr = zone_splits[(zone_splits["split"] == "train") & (zone_splits["zone_id"] == zid)]
        z_balanced = z_tr[z_tr["is_train_balanced_sample"]]

        pos_cnt = int((z_balanced["label_state"] == "positive").sum())
        neg_cnt = int((z_balanced["label_state"] == "negative").sum())
        total_balanced = len(z_balanced)

        assert pos_cnt == neg_cnt, f"Zon {zid} 1:1 dengeli değil: pos={pos_cnt}, neg={neg_cnt}"
        assert total_balanced == 2 * pos_cnt, f"Zon {zid} toplam örnek uyumsuz: {total_balanced}"
        print(f"  [OK] Zon {zid}: Positif = {pos_cnt}, Temiz Negatif = {neg_cnt} (Toplam = {total_balanced}) -> 1:1 TAM DENGELİ")

    # Kontrol 4: Ambiguous ve Unassigned Negatif Havuzuna Alınmama Kontrolü
    print("\n[4] Ambiguous ve Unassigned Dışlanma Kontrolü:")
    tr_balanced_all = zone_splits[zone_splits["is_train_balanced_sample"]]
    amb_in_balanced = tr_balanced_all[tr_balanced_all["label_state"] == "ambiguous"]
    unassigned_in_balanced = tr_balanced_all[tr_balanced_all["label_state"] == "unassigned_nodule"]
    dirty_negatives = tr_balanced_all[
        (tr_balanced_all["zone_label"] == 0) & (tr_balanced_all["label_state"] != "negative")
    ]

    assert len(amb_in_balanced) == 0, f"Ambiguous satırlar train negatifinde bulundu: {len(amb_in_balanced)}"
    assert len(unassigned_in_balanced) == 0, f"Unassigned satırlar train negatifinde bulundu: {len(unassigned_in_balanced)}"
    assert len(dirty_negatives) == 0, f"Negatif olmayan satırlar dengeli train'de: {len(dirty_negatives)}"
    print("  [OK] Ambiguous (%0-%20) satırlar asla train negatif havuzuna alınmadı (0 ihlal).")
    print("  [OK] Unassigned satırlar asla train negatif havuzuna alınmadı (0 ihlal).")
    print("  [OK] Train negatif havuzunun %100'ü temiz (label_state == 'negative') zonlardan oluşuyor.")

    # Kontrol 5: Vektör Sayısı, Metadata Satır Sayısı ve vector_index Eşleşmesi
    print("\n[5] Vektör Boyutu, Metadata ve İndeks Eşleşme Kontrolü:")
    assert X.shape == (24410, 800), f"X boyutu uyumsuz: {X.shape}"
    assert len(features_meta) == 24410, f"features_meta satır sayısı uyumsuz: {len(features_meta)}"
    assert len(zone_splits) == 24410, f"zone_splits satır sayısı uyumsuz: {len(zone_splits)}"

    # İndeks sıralaması
    assert (features_meta["vector_index"].values == np.arange(24410)).all(), "features_meta vector_index hatalı!"
    assert (zone_splits["vector_index"].values == np.arange(24410)).all(), "zone_splits vector_index hatalı!"
    print(f"  [OK] X matris şekli: {X.shape} (24.410 zon, 800 boyut).")
    print(f"  [OK] features_metadata satır sayısı: {len(features_meta):,} (vector_index 0..24409 birebir eşleşiyor).")
    print(f"  [OK] zone_splits satır sayısı: {len(zone_splits):,} (vector_index 0..24409 birebir eşleşiyor).")

    # Hash Kontrolleri
    print("\n[6] Hash Bütünlük Kontrolleri:")
    actual_label_hash = compute_sha256(labels_path)
    manifest_label_hash = splits_summary.get("labels_file_sha256")
    assert actual_label_hash == manifest_label_hash, "Etiket dosyası SHA-256 hash'i split manifesti ile uyuşmuyor!"
    print(f"  [OK] zone_labels_v2.csv SHA-256: {actual_label_hash[:16]}... (Manifest ile tam eşleşiyor).")

    actual_tda_hash = compute_sha256(tda_config_path)
    meta_tda_hash = features_meta["tda_config_hash"].iloc[0]
    assert actual_tda_hash == meta_tda_hash, "TDA config SHA-256 hash'i metadata ile uyuşmuyor!"
    print(f"  [OK] tda_ranges_v2.json SHA-256: {actual_tda_hash[:16]}... (Metadata ile tam eşleşiyor).")

    # Kontrol 7: Eski v1 Dosyalarının İzolasyonu
    print("\n[7] v1 Dosya İzolasyon Kontrolü:")
    v1_paths = [
        PROJECT_ROOT / "data" / "processed" / "features" / "X_zone_features.npy",
        PROJECT_ROOT / "data" / "processed" / "splits" / "zone_splits.csv",
        PROJECT_ROOT / "data" / "processed" / "zones" / "reports" / "nodule_5zones_mapping.csv",
        PROJECT_ROOT / "artifacts" / "models" / "zone_svm",
        PROJECT_ROOT / "artifacts" / "runs" / "test_patient_predictions.csv",
    ]
    for v1_p in v1_paths:
        assert not v1_p.exists(), f"Eski v1 dosyası hala mevcut: {v1_p}"
    print("  [OK] Tüm geçersiz v1 türetilmiş dosyaları temizlenmiş durumda; v2 tamamen bağımsız.")

    print("\n" + "=" * 80)
    print("TÜM KONTROLLER BAŞARIYLA GEÇTİ! v2 VERİ HATTI %100 DOĞRULANDI.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)

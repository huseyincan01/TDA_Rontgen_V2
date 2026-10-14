"""v2 TDA Özellik Çıkarımı Boru Hattı.

Kurallar:
- Mevcut zone_*.mha kesitlerinden cubical persistence homology ve v2 Persistence Image çıkarır.
- Yeni `data/processed/labels/zone_labels_v2.csv` etiket dosyasını kullanır.
- H0 ve H1 için v2 ayrı ızgara aralıklarını (`tda_ranges_v2.json`) kullanır.
- Çıktı klasörü: `data/processed/features_v2/`
- Beklenen matris şekli: (24410, 800); zonlar birleştirilmez.
- Metadata'da: `label_state`, `label_rule_version` ve `tda_config_hash` alanları bulunur.
- Çıktı raporunda H0 ve H1 için dolu hücre oranını ve sıfır oranını raporlar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor

ZONE_FILES = {
    1: ("Sağ Üst (R-Upper)", "zone_1_sag_ust.mha"),
    2: ("Sağ Orta (R-Mid)", "zone_2_sag_orta.mha"),
    3: ("Sağ Alt (R-Lower)", "zone_3_sag_alt.mha"),
    4: ("Sol Üst (L-Upper)", "zone_4_sol_ust.mha"),
    5: ("Sol Alt (L-Lower)", "zone_5_sol_alt.mha"),
}


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_one_v2(
    task: Tuple[str, int, str, float, int, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
) -> Dict[str, Any]:
    """Windows ProcessPool için modül düzeyinde tek-zon işçisi (v2)."""
    (
        patient_id,
        zone_id,
        path_string,
        intensity_max,
        grid_size,
        h0_birth_range,
        h0_pers_range,
        h1_birth_range,
        h1_pers_range,
    ) = task

    zone_name, _ = ZONE_FILES[zone_id]
    settings = TDASettings(
        grid_size=grid_size,
        intensity_max=intensity_max,
        h0_birth_range=h0_birth_range,
        h0_persistence_range=h0_pers_range,
        h1_birth_range=h1_birth_range,
        h1_persistence_range=h1_pers_range,
    )
    try:
        image = sitk.GetArrayFromImage(sitk.ReadImage(path_string))
        if image.ndim == 3:
            image = image[0]
        vector, status = ZoneTDAExtractor(settings).extract(image)
        error = ""
    except Exception as exc:
        vector = np.zeros(settings.vector_size, dtype=np.float32)
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    return {
        "patient_id": patient_id,
        "zone_id": zone_id,
        "vector": vector,
        "status": status,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 TDA Özellik Çıkarımı Boru Hattı")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tda_config" / "tda_ranges_v2.json",
        help="tda_ranges_v2.json yapılandırma dosya yolu.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labels" / "zone_labels_v2.csv",
        help="zone_labels_v2.csv etiket dosya yolu.",
    )
    parser.add_argument(
        "--zones-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zones" / "patients",
        help="Hasta kesitleri dizini.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features_v2",
        help="Çıktı dizini.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Paralel işçi sayısı (varsayılan: 10).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Paralel chunk boyutu (varsayılan: 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Test amaçlı hasta sınırı.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config_path.exists():
        raise FileNotFoundError(f"TDA yapılandırma dosyası bulunamadı: {args.config_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Etiket dosyası bulunamadı: {args.labels_path}")
    if not args.zones_dir.exists():
        raise FileNotFoundError(f"Zon klasörü bulunamadı: {args.zones_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Yapılandırma ve etiketleri yükle
    with open(args.config_path, "r", encoding="utf-8") as f:
        tda_cfg = json.load(f)

    tda_config_hash = compute_file_sha256(args.config_path)

    h0_ranges = tda_cfg["ranges"]["h0"]
    h1_ranges = tda_cfg["ranges"]["h1"]

    h0_b_range = tuple(float(x) for x in h0_ranges["birth_range"])
    h0_p_range = tuple(float(x) for x in h0_ranges["persistence_range"])
    h1_b_range = tuple(float(x) for x in h1_ranges["birth_range"])
    h1_p_range = tuple(float(x) for x in h1_ranges["persistence_range"])

    grid_size = 20
    intensity_max = 65535.0

    print("=" * 80)
    print("v2.1 TDA ÖZELLİK ÇIKARIMI BORU HATTI")
    print("=" * 80)
    print(f"Yapılandırma Dosyası: {args.config_path} (SHA-256: {tda_config_hash[:16]}...)")
    print(f"Etiket Dosyası:       {args.labels_path}")
    print(f"Zon Klasörü:          {args.zones_dir}")
    print(f"Çıktı Dizini:         {args.output_dir}")
    print(f"H0 Aralıkları:        Birth={h0_b_range}, Pers={h0_p_range}")
    print(f"H1 Aralıkları:        Birth={h1_b_range}, Pers={h1_p_range}")
    print(f"Grid Size:            {grid_size}x{grid_size} (H0: 400, H1: 400 -> Vektör: 800)")
    print(f"İşçi Sayısı:          {args.workers}")
    print("-" * 80)

    labels_df = pd.read_csv(args.labels_path)
    # labels_df satır sırası deterministik olmalıdır: patient_id, zone_id
    labels_df = labels_df.sort_values(["patient_id", "zone_id"]).reset_index(drop=True)

    if args.limit is not None:
        pids = labels_df["patient_id"].unique()[:args.limit]
        labels_df = labels_df[labels_df["patient_id"].isin(pids)].copy()

    # Görevleri hazırla
    tasks = []
    for _, row in labels_df.iterrows():
        pid = str(row["patient_id"])
        zid = int(row["zone_id"])
        _, filename = ZONE_FILES[zid]
        mha_path = args.zones_dir / pid / filename
        if not mha_path.exists():
            raise FileNotFoundError(f"Zon dosyası bulunamadı: {mha_path}")

        tasks.append((
            pid,
            zid,
            str(mha_path),
            intensity_max,
            grid_size,
            h0_b_range,
            h0_p_range,
            h1_b_range,
            h1_p_range,
        ))

    workers = max(1, min(args.workers, os.cpu_count() or 1))
    print(f"Toplam Görev Sayısı:  {len(tasks):,} (Hasta: {labels_df['patient_id'].nunique():,})")
    print("TDA Vektör Çıkarımı Başlıyor...")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(
            pool.map(extract_one_v2, tasks, chunksize=args.chunksize),
            total=len(tasks),
            desc="TDA v2 Vektörleri",
            unit="zon",
        ))
    elapsed = time.perf_counter() - t0

    # 2. Vektör matrisini oluştur
    vectors = np.stack([r["vector"] for r in results]).astype(np.float32, copy=False)
    assert vectors.shape == (len(tasks), 800), f"Beklenen şekil ({len(tasks)}, 800), gelen: {vectors.shape}"

    features_path = args.output_dir / "X_zone_features.npy"
    np.save(features_path, vectors)
    print(f"\n[OK] Vektör Matrisi Kaydedildi: {features_path} {vectors.shape}")

    # 3. Dolu ve sıfır hücre istatistiklerini hesapla
    h0_vecs = vectors[:, :400]
    h1_vecs = vectors[:, 400:]

    h0_filled_ratio = float(np.mean(h0_vecs != 0.0))
    h0_zero_ratio = float(np.mean(h0_vecs == 0.0))

    h1_filled_ratio = float(np.mean(h1_vecs != 0.0))
    h1_zero_ratio = float(np.mean(h1_vecs == 0.0))

    total_filled_ratio = float(np.mean(vectors != 0.0))
    total_zero_ratio = float(np.mean(vectors == 0.0))

    # 4. Metadata CSV'sini yaz
    meta_df = labels_df.copy()
    meta_df["vector_index"] = list(range(len(vectors)))
    meta_df["tda_config_hash"] = tda_config_hash
    meta_df["extraction_status"] = [r["status"] for r in results]
    meta_df["error"] = [r["error"] for r in results]

    cols_order = [
        "vector_index",
        "patient_id",
        "zone_id",
        "zone_name",
        "zone_label",
        "label_state",
        "patient_label",
        "nodule_ids",
        "max_bbox_overlap",
        "label_rule_version",
        "tda_config_hash",
        "extraction_status",
        "error",
    ]
    meta_df = meta_df[cols_order]

    metadata_path = args.output_dir / "features_metadata.csv"
    meta_df.to_csv(metadata_path, index=False)
    print(f"[OK] Metadata Kaydedildi:       {metadata_path} ({len(meta_df):,} satır)")

    # 5. Extraction Raporu JSON
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1

    report: Dict[str, Any] = {
        "pipeline_version": "v2.1_fixed_grid_gaussian",
        "matrix_shape": list(vectors.shape),
        "total_patients": int(labels_df["patient_id"].nunique()),
        "total_zone_vectors": int(len(vectors)),
        "feature_dimension": 800,
        "grid_size": grid_size,
        "persistence_image_method": "fixed_grid_gaussian",
        "gaussian_sigma_pixels": 1.0,
        "intensity_max": intensity_max,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60.0, 2),
        "tda_config_hash": tda_config_hash,
        "h0_ranges": {
            "birth_range": list(h0_b_range),
            "persistence_range": list(h0_p_range),
        },
        "h1_ranges": {
            "birth_range": list(h1_b_range),
            "persistence_range": list(h1_p_range),
        },
        "sparsity_analysis": {
            "h0_filled_ratio": round(h0_filled_ratio, 4),
            "h0_zero_ratio": round(h0_zero_ratio, 4),
            "h1_filled_ratio": round(h1_filled_ratio, 4),
            "h1_zero_ratio": round(h1_zero_ratio, 4),
            "total_filled_ratio": round(total_filled_ratio, 4),
            "total_zero_ratio": round(total_zero_ratio, 4),
        },
        "status_counts": dict(status_counts),
        "note": "Her satır bağımsız bir (görüntü, zon) TDA vektörüdür; hasta vektörleri birleştirilmemiştir.",
    }

    report_path = args.output_dir / "extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Çıkarım Raporu:            {report_path}")

    print("\n" + "=" * 80)
    print("v2.1 TDA ÖZELLİK ÇIKARIMI TAMAMLANDI")
    print("=" * 80)
    print(f"Toplam Süre:          {elapsed / 60.0:.2f} dakika ({elapsed:.1f} saniye)")
    print(f"Vektör Şekli:         {vectors.shape}")
    print(f"H0 Sıfır Oranı:       %{h0_zero_ratio*100:.2f} (yalnız tanısal bilgi)")
    print(f"H1 Sıfır Oranı:       %{h1_zero_ratio*100:.2f} (yalnız tanısal bilgi)")
    print(f"Durum Sayımları:      {dict(status_counts)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

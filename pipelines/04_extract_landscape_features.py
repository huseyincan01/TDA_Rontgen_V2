"""v2.1 Persistence Landscapes (PL) Özellik Çıkarımı Boru Hattı.

Kurallar:
- MHA zon kesitlerinden kübik kalıcı homoloji (H0 ve H1) çıkarır.
- GUDHI `Landscape(num_landscapes=5, resolution=80, sample_range=[0.0, 255.0])` ile
  her boyut için 400 boyutlu (5 seviye x 80 nokta), toplamda tam 800 boyutlu fonksiyonel vektör üretir.
- Girdi ve etiket: `data/processed/labels/zone_labels_v2.csv`
- Çıktı klasörü: `data/processed/features_landscape_v2/`
- Matris şekli: (24410, 800); satır sırası deterministik olarak labels_df ve zone_splits.csv ile birebirdir.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import gudhi
from gudhi.representations import Landscape
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

_WORKER_EXTRACTOR: ZoneTDAExtractor | None = None
_WORKER_PL_H0: Landscape | None = None
_WORKER_PL_H1: Landscape | None = None


def _init_worker(num_landscapes: int, resolution: int) -> None:
    global _WORKER_EXTRACTOR, _WORKER_PL_H0, _WORKER_PL_H1
    _WORKER_EXTRACTOR = ZoneTDAExtractor(TDASettings())
    dummy_diagram = np.array([[0.0, 255.0]])

    _WORKER_PL_H0 = Landscape(
        num_landscapes=num_landscapes,
        resolution=resolution,
        sample_range=[0.0, 255.0],
    )
    _WORKER_PL_H0.fit([dummy_diagram])

    _WORKER_PL_H1 = Landscape(
        num_landscapes=num_landscapes,
        resolution=resolution,
        sample_range=[0.0, 255.0],
    )
    _WORKER_PL_H1.fit([dummy_diagram])


def extract_one_landscape(task: Tuple[str, int, str]) -> Dict[str, Any]:
    """ProcessPool için tek-zon Landscape çıkarıcısı."""
    global _WORKER_EXTRACTOR, _WORKER_PL_H0, _WORKER_PL_H1
    patient_id, zone_id, path_string = task

    if _WORKER_EXTRACTOR is None or _WORKER_PL_H0 is None or _WORKER_PL_H1 is None:
        _init_worker(5, 80)

    try:
        image = sitk.GetArrayFromImage(sitk.ReadImage(path_string))
        if image.ndim == 3:
            image = image[0]

        filtration = _WORKER_EXTRACTOR.prepare_filtration(image)
        if filtration is None:
            vector = np.zeros(800, dtype=np.float32)
            status = "empty"
            error = "Geçerli doku maskesi bulunamadı"
        else:
            h0, h1 = _WORKER_EXTRACTOR.compute_diagrams(filtration)
            v_h0 = _WORKER_PL_H0.transform([h0])[0] if len(h0) > 0 else np.zeros(400, dtype=np.float32)
            v_h1 = _WORKER_PL_H1.transform([h1])[0] if len(h1) > 0 else np.zeros(400, dtype=np.float32)
            vector = np.concatenate((v_h0, v_h1)).astype(np.float32, copy=False)
            status = "success"
            error = ""
    except Exception as exc:
        vector = np.zeros(800, dtype=np.float32)
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
    parser = argparse.ArgumentParser(description="v2.1 Persistence Landscapes Özellik Çıkarımı")
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
        default=PROJECT_ROOT / "data" / "processed" / "features_landscape_v2",
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
        default=16,
        help="Paralel chunk boyutu (varsayılan: 16).",
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

    if not args.labels_path.exists():
        raise FileNotFoundError(f"Etiket dosyası bulunamadı: {args.labels_path}")
    if not args.zones_dir.exists():
        raise FileNotFoundError(f"Zon klasörü bulunamadı: {args.zones_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_landscapes = 5
    resolution = 80
    feature_dim = 2 * num_landscapes * resolution  # 800

    print("=" * 80)
    print("v2.1 PERSISTENCE LANDSCAPES (PL) ÖZELLİK ÇIKARIMI BORU HATTI")
    print("=" * 80)
    print(f"Etiket Dosyası:       {args.labels_path}")
    print(f"Zon Klasörü:          {args.zones_dir}")
    print(f"Çıktı Dizini:         {args.output_dir}")
    print(f"Landscape Ayarları:   num_landscapes={num_landscapes}, resolution={resolution}")
    print(f"Özellik Boyutu:       H0: {num_landscapes*resolution}, H1: {num_landscapes*resolution} -> Toplam: {feature_dim}")
    print(f"İşçi Sayısı:          {args.workers}")
    print("-" * 80)

    labels_df = pd.read_csv(args.labels_path)
    labels_df = labels_df.sort_values(["patient_id", "zone_id"]).reset_index(drop=True)

    if args.limit is not None:
        pids = labels_df["patient_id"].unique()[:args.limit]
        labels_df = labels_df[labels_df["patient_id"].isin(pids)].copy()

    tasks = []
    for _, row in labels_df.iterrows():
        pid = str(row["patient_id"])
        zid = int(row["zone_id"])
        _, filename = ZONE_FILES[zid]
        mha_path = args.zones_dir / pid / filename
        if not mha_path.exists():
            raise FileNotFoundError(f"Zon dosyası bulunamadı: {mha_path}")
        tasks.append((pid, zid, str(mha_path)))

    workers = max(1, min(args.workers, os.cpu_count() or 1))
    print(f"Toplam Görev Sayısı:  {len(tasks):,} (Hasta: {labels_df['patient_id'].nunique():,})")
    print("Landscape Çıkarımı Başlıyor...")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(num_landscapes, resolution),
    ) as pool:
        results = list(tqdm(
            pool.map(extract_one_landscape, tasks, chunksize=args.chunksize),
            total=len(tasks),
            desc="Persistence Landscapes",
            unit="zon",
        ))
    elapsed = time.perf_counter() - t0

    # Vektör matrisini oluştur
    vectors = np.stack([r["vector"] for r in results]).astype(np.float32, copy=False)
    assert vectors.shape == (len(tasks), feature_dim), f"Beklenen şekil ({len(tasks)}, {feature_dim}), gelen: {vectors.shape}"

    features_path = args.output_dir / "X_zone_features.npy"
    np.save(features_path, vectors)
    print(f"\n[OK] Vektör Matrisi Kaydedildi: {features_path} {vectors.shape}")

    # Dolu ve sıfır hücre istatistikleri
    h0_vecs = vectors[:, :400]
    h1_vecs = vectors[:, 400:]

    h0_filled_ratio = float(np.mean(h0_vecs != 0.0))
    h0_zero_ratio = float(np.mean(h0_vecs == 0.0))
    h1_filled_ratio = float(np.mean(h1_vecs != 0.0))
    h1_zero_ratio = float(np.mean(h1_vecs == 0.0))
    total_filled_ratio = float(np.mean(vectors != 0.0))
    total_zero_ratio = float(np.mean(vectors == 0.0))

    # Metadata CSV'sini yaz
    meta_df = labels_df.copy()
    meta_df["vector_index"] = list(range(len(vectors)))
    meta_df["tda_representation"] = "persistence_landscape"
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
        "tda_representation",
        "extraction_status",
        "error",
    ]
    meta_df = meta_df[cols_order]

    metadata_path = args.output_dir / "features_metadata.csv"
    meta_df.to_csv(metadata_path, index=False)
    print(f"[OK] Metadata Kaydedildi:       {metadata_path} ({len(meta_df):,} satır)")

    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1

    report: Dict[str, Any] = {
        "pipeline_version": "v2.1_persistence_landscape",
        "representation": "persistence_landscape",
        "matrix_shape": list(vectors.shape),
        "total_patients": int(labels_df["patient_id"].nunique()),
        "total_zone_vectors": int(len(vectors)),
        "feature_dimension": feature_dim,
        "num_landscapes": num_landscapes,
        "resolution": resolution,
        "sample_range": [0.0, 255.0],
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_minutes": round(elapsed / 60.0, 2),
        "sparsity_analysis": {
            "h0_filled_ratio": round(h0_filled_ratio, 4),
            "h0_zero_ratio": round(h0_zero_ratio, 4),
            "h1_filled_ratio": round(h1_filled_ratio, 4),
            "h1_zero_ratio": round(h1_zero_ratio, 4),
            "total_filled_ratio": round(total_filled_ratio, 4),
            "total_zero_ratio": round(total_zero_ratio, 4),
        },
        "status_counts": dict(status_counts),
        "note": "Her satır bağımsız bir (görüntü, zon) Persistence Landscape vektörüdür.",
    }

    report_path = args.output_dir / "extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Çıkarım Raporu:            {report_path}")

    print("\n" + "=" * 80)
    print("v2.1 PERSISTENCE LANDSCAPES ÇIKARIMI TAMAMLANDI")
    print("=" * 80)
    print(f"Toplam Süre:          {elapsed / 60.0:.2f} dakika ({elapsed:.1f} saniye)")
    print(f"Vektör Şekli:         {vectors.shape}")


if __name__ == "__main__":
    main()

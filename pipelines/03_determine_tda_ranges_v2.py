"""Train Verisinden Sağlam TDA Izgara Aralıkları Belirleme Boru Hattı.

Kurallar:
- SADECE train hastalarından dengeli ve sabit tohumlu bir örnek seçer.
- Test ve validation görüntülerini ASLA kullanmaz (Data leakage koruması).
- H0 ve H1 için doğum (birth) ve persistence dağılımlarını ayrı hesaplar.
- Tek ortak [0, 255] yerine, H0 ve H1 için ayrı Persistence Image aralıkları türetir.
- Sağlam persentilleri (robust percentiles), örnek hasta ve zon sayısını,
  nihai sınırları ve clipping oranını JSON dosyasına kaydeder.
- Çıktı: data/processed/tda_config/tda_ranges_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.features.extractor import ZoneTDAExtractor

ZONE_FILES = [
    (1, "Sağ Üst", "zone_1_sag_ust.mha"),
    (2, "Sağ Orta", "zone_2_sag_orta.mha"),
    (3, "Sağ Alt", "zone_3_sag_alt.mha"),
    (4, "Sol Üst", "zone_4_sol_ust.mha"),
    (5, "Sol Alt", "zone_5_sol_alt.mha"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train verisinden TDA ızgara aralıklarını belirler.")
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits_v2" / "patient_splits.csv",
        help="patient_splits.csv dosya yolu.",
    )
    parser.add_argument(
        "--patients-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zones" / "patients",
        help="Hasta klasörleri dizini.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tda_config" / "tda_ranges_v2.json",
        help="Çıktı JSON dosya yolu.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=25,
        help="Train'den seçilecek her sınıftan hasta sayısı (varsayılan: 25 pos, 25 neg = 50 hasta).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Örnekleme rastgele tohumu (varsayılan: 42).",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Doğum için alt persentil (varsayılan: 1.0).",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help="Doğum ve persistence için üst persentil (varsayılan: 99.0).",
    )
    return parser.parse_args()


def compute_distribution_stats(arr: np.ndarray, name: str) -> Dict[str, Any]:
    if len(arr) == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p005": float(np.percentile(arr, 0.5)),
        "p01": float(np.percentile(arr, 1.0)),
        "p05": float(np.percentile(arr, 5.0)),
        "p10": float(np.percentile(arr, 10.0)),
        "p25": float(np.percentile(arr, 25.0)),
        "p50_median": float(np.percentile(arr, 50.0)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p75": float(np.percentile(arr, 75.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "p995": float(np.percentile(arr, 99.5)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("v2 TDA IZGARA ARALIĞI BELİRLEME (YALNIZCA TRAIN VERİSİ)")
    print("=" * 80)
    print(f"Splits Dosyası:       {args.splits_path}")
    print(f"Hasta Klasörleri:     {args.patients_dir}")
    print(f"Çıktı Yapılandırması: {args.output_path}")
    print(f"Örneklem:             {args.sample_per_class} Pozitif + {args.sample_per_class} Kontrol (Train içi)")
    print(f"Rastgele Tohum:       {args.seed}")
    print(f"Persentil Aralıkları: Alt: %{args.lower_percentile} | Üst: %{args.upper_percentile}")
    print("-" * 80)

    splits_df = pd.read_csv(args.splits_path)

    # 1. SADECE TRAIN kümesini al ve katı sızıntı kontrolü yap
    train_patients = splits_df[splits_df["split"] == "train"].copy()
    assert len(train_patients[train_patients["split"] != "train"]) == 0, "HATA: Train dışı hasta sızıntısı tespit edildi!"

    train_pos = train_patients[train_patients["patient_label"] == 1].sample(
        n=args.sample_per_class, random_state=args.seed
    )
    train_neg = train_patients[train_patients["patient_label"] == 0].sample(
        n=args.sample_per_class, random_state=args.seed
    )
    sampled_patients = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"Seçilen Train Hastaları: Toplam {len(sampled_patients)} hasta ({len(train_pos)} Pozitif, {len(train_neg)} Kontrol)")
    print(f"İşlenecek Zon Sayısı:   {len(sampled_patients) * 5} kesit")

    # 2. Persistence diyagramlarını çıkar
    extractor = ZoneTDAExtractor()

    h0_births: List[float] = []
    h0_persistences: List[float] = []
    h1_births: List[float] = []
    h1_persistences: List[float] = []

    success_zones = 0
    t0 = time.time()

    for idx, row in sampled_patients.iterrows():
        pid = row["patient_id"]
        p_dir = args.patients_dir / pid

        for zid, zname, zfile in ZONE_FILES:
            mha_path = p_dir / zfile
            if not mha_path.exists():
                continue

            itk_img = sitk.ReadImage(str(mha_path))
            arr = sitk.GetArrayFromImage(itk_img).squeeze()

            filtration = extractor.prepare_filtration(arr)
            if filtration is None:
                continue

            h0, h1 = extractor.compute_diagrams(filtration)
            success_zones += 1

            if len(h0) > 0:
                h0_births.extend(h0[:, 0].tolist())
                h0_persistences.extend((h0[:, 1] - h0[:, 0]).tolist())

            if len(h1) > 0:
                h1_births.extend(h1[:, 0].tolist())
                h1_persistences.extend((h1[:, 1] - h1[:, 0]).tolist())

    elapsed = time.time() - t0
    print(f"Diyagram çıkarımı tamamlandı ({elapsed:.1f}s). İşlenen zon: {success_zones}")
    print(f"Toplam H0 noktası: {len(h0_births):,} | Toplam H1 noktası: {len(h1_births):,}")

    h0_b_arr = np.array(h0_births, dtype=np.float64)
    h0_p_arr = np.array(h0_persistences, dtype=np.float64)
    h1_b_arr = np.array(h1_births, dtype=np.float64)
    h1_p_arr = np.array(h1_persistences, dtype=np.float64)

    # 3. İstatistikleri ve sağlam persentilleri hesapla
    h0_b_stats = compute_distribution_stats(h0_b_arr, "H0 Birth")
    h0_p_stats = compute_distribution_stats(h0_p_arr, "H0 Persistence")
    h1_b_stats = compute_distribution_stats(h1_b_arr, "H1 Birth")
    h1_p_stats = compute_distribution_stats(h1_p_arr, "H1 Persistence")

    # Sınırları belirle:
    # H0 Doğum: p01 -> p99
    # Persistence: 0 -> p99 (persistence daima >= 0)
    # Tam sayılara güvenli yuvarlama (floor / ceil)
    h0_b_low = math.floor(float(np.percentile(h0_b_arr, args.lower_percentile)))
    h0_b_high = math.ceil(float(np.percentile(h0_b_arr, args.upper_percentile)))
    h0_p_low = 0.0
    h0_p_high = math.ceil(float(np.percentile(h0_p_arr, args.upper_percentile)))

    h1_b_low = math.floor(float(np.percentile(h1_b_arr, args.lower_percentile)))
    h1_b_high = math.ceil(float(np.percentile(h1_b_arr, args.upper_percentile)))
    h1_p_low = 0.0
    h1_p_high = math.ceil(float(np.percentile(h1_p_arr, args.upper_percentile)))

    # Güvenlik kontrolü: genişlik 0 olamaz
    if h0_b_high <= h0_b_low:
        h0_b_high = h0_b_low + 10.0
    if h0_p_high <= h0_p_low:
        h0_p_high = 10.0
    if h1_b_high <= h1_b_low:
        h1_b_high = h1_b_low + 10.0
    if h1_p_high <= h1_p_low:
        h1_p_high = 10.0

    h0_birth_range = [float(h0_b_low), float(h0_b_high)]
    h0_pers_range = [float(h0_p_low), float(h0_p_high)]
    h1_birth_range = [float(h1_b_low), float(h1_b_high)]
    h1_pers_range = [float(h1_p_low), float(h1_p_high)]

    # 4. Clipping oranları (aralık dışına taşan nokta oranı)
    h0_b_clipped = float(np.mean((h0_b_arr < h0_b_low) | (h0_b_arr > h0_b_high)))
    h0_p_clipped = float(np.mean((h0_p_arr < h0_p_low) | (h0_p_arr > h0_p_high)))
    h1_b_clipped = float(np.mean((h1_b_arr < h1_b_low) | (h1_b_arr > h1_b_high)))
    h1_p_clipped = float(np.mean((h1_p_arr < h1_p_low) | (h1_p_arr > h1_p_high)))

    output_config: Dict[str, Any] = {
        "config_version": "v2_train_derived",
        "description": "Persistence Image grid ranges derived strictly from balanced train persistence diagrams.",
        "sample_info": {
            "source_split": "train_only",
            "sample_patients_total": len(sampled_patients),
            "sample_positive_patients": len(train_pos),
            "sample_negative_patients": len(train_neg),
            "sample_zones_processed": success_zones,
            "seed": args.seed,
            "patient_ids": sampled_patients["patient_id"].tolist(),
        },
        "percentile_settings": {
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
        },
        "ranges": {
            "h0": {
                "birth_range": h0_birth_range,
                "persistence_range": h0_pers_range,
                "birth_clipping_ratio": round(h0_b_clipped, 4),
                "persistence_clipping_ratio": round(h0_p_clipped, 4),
            },
            "h1": {
                "birth_range": h1_birth_range,
                "persistence_range": h1_pers_range,
                "birth_clipping_ratio": round(h1_b_clipped, 4),
                "persistence_clipping_ratio": round(h1_p_clipped, 4),
            },
        },
        "distributions": {
            "h0_birth": h0_b_stats,
            "h0_persistence": h0_p_stats,
            "h1_birth": h1_b_stats,
            "h1_persistence": h1_p_stats,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("NİHAİ TDA v2 IZGARA ARALIKLARI")
    print("=" * 80)
    print(f"H0 Birth Range:        {h0_birth_range} (Clipping: %{h0_b_clipped*100:.2f})")
    print(f"H0 Persistence Range:  {h0_pers_range} (Clipping: %{h0_p_clipped*100:.2f})")
    print(f"H1 Birth Range:        {h1_birth_range} (Clipping: %{h1_b_clipped*100:.2f})")
    print(f"H1 Persistence Range:  {h1_pers_range} (Clipping: %{h1_p_clipped*100:.2f})")
    print("=" * 80)
    print(f"Yapılandırma dosyası kaydedildi: {args.output_path}")


if __name__ == "__main__":
    main()

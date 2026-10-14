"""v2 Zon Splitleri ve Dengeleme Boru Hattı.

Kullanım:
    python pipelines/02_create_zone_splits_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.data.zone_splits_v2 import build_zone_splits_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 Zon Splitleri ve Dengeleme Boru Hattı")
    parser.add_argument(
        "--features-metadata-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features_v2" / "features_metadata.csv",
        help="features_metadata.csv yolu.",
    )
    parser.add_argument(
        "--patient-splits-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits_v2" / "patient_splits.csv",
        help="patient_splits.csv yolu.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits_v2" / "zone_splits.csv",
        help="Çıktı zone_splits.csv yolu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dengeleme rastgele tohumu (varsayılan: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features_metadata_path.exists():
        raise FileNotFoundError(f"Özellik metadata dosyası bulunamadı: {args.features_metadata_path}")
    if not args.patient_splits_path.exists():
        raise FileNotFoundError(f"Hasta split dosyası bulunamadı: {args.patient_splits_path}")

    print("=" * 80)
    print("v2 ZON SPLITLERİ VE DENGELEME BORU HATTI")
    print("=" * 80)
    print(f"Girdi Metadata:     {args.features_metadata_path}")
    print(f"Girdi Hasta Split:  {args.patient_splits_path}")
    print(f"Çıktı Zon Split:    {args.output_path}")
    print(f"Random Seed:        {args.seed}")
    print("-" * 80)

    meta_df = pd.read_csv(args.features_metadata_path)
    patient_splits_df = pd.read_csv(args.patient_splits_path)

    zone_splits_df, report = build_zone_splits_v2(
        features_meta_df=meta_df,
        patient_splits_df=patient_splits_df,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    zone_splits_df.to_csv(args.output_path, index=False)

    summary_path = args.output_path.parent / "zone_splits_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("v2 ZON DENGELEME VE SPLIT RAPORU")
    print("=" * 80)
    print(f"Toplam Satır: {report['total_rows']:,} | Toplam Hasta: {report['total_patients']:,}")
    print("\nTRAIN BAZINDA 1:1 DENGELEME (Temiz Negatif Havuzundan):")
    print(f"{'Zon':<22} | {'Pos':>5} | {'Dengeli Neg':>11} | {'Dengeli Top':>11} | {'Val Pos/Neg':>12} | {'Test Pos/Neg':>12}")
    print("-" * 80)
    for z_key, z_stat in report["zone_balance_statistics"].items():
        name = z_stat["zone_name"]
        tr = z_stat["train"]
        vl = z_stat["validation"]
        ts = z_stat["test"]
        print(
            f"{name:<22} | {tr['positive']:>5} | {tr['clean_negative_selected']:>11} | {tr['balanced_total']:>11} | "
            f"{vl['positive']:>5}/{vl['clean_negative']:<6} | {ts['positive']:>5}/{ts['clean_negative']:<6}"
        )
    print("=" * 80)
    print(f"Çıktı 1: {args.output_path}")
    print(f"Çıktı 2: {summary_path}")


if __name__ == "__main__":
    main()

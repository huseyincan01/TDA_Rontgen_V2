"""v2 Hasta Düzeyinde Veri Bölme Boru Hattı.

Kullanım:
    python pipelines/02_create_splits_v2.py
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

from node21_tda.data.splits_v2 import SplitConfigV2, split_patients_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 Hasta Düzeyinde Veri Bölme Boru Hattı")
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labels" / "zone_labels_v2.csv",
        help="Girdi zone_labels_v2.csv yolu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits_v2",
        help="Çıktı dizini.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train oranı (varsayılan: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation oranı (varsayılan: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test oranı (varsayılan: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele tohum değeri (varsayılan: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.labels_path.exists():
        raise FileNotFoundError(f"Etiket dosyası bulunamadı: {args.labels_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v2 HASTA BÖLME BORU HATTI")
    print("=" * 80)
    print(f"Girdi Etiket Dosyası: {args.labels_path}")
    print(f"Çıktı Dizini:         {args.output_dir}")
    print(f"Oranlar:              Train: %{args.train_ratio*100:.1f} | Val: %{args.val_ratio*100:.1f} | Test: %{args.test_ratio*100:.1f}")
    print(f"Random Seed:          {args.seed}")
    print("Katmanlama:           YALNIZCA patient_label ve v2 zon pozitiflik örüntüsü (source_dataset YOK)")
    print("-" * 80)

    labels_df = pd.read_csv(args.labels_path)
    config = SplitConfigV2(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    patient_splits_df, report = split_patients_v2(
        labels_df=labels_df,
        labels_file_path=args.labels_path,
        config=config,
    )

    patient_splits_path = args.output_dir / "patient_splits.csv"
    summary_path = args.output_dir / "splits_summary.json"

    patient_splits_df.to_csv(patient_splits_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("v2 HASTA BÖLME RAPORU")
    print("=" * 80)
    print(f"Toplam Hasta Sayısı: {report['total_patients']:,}")
    print("Hasta Dağılımı:")
    for s_name, count in report["split_counts"].items():
        ratio = report["split_ratios"][s_name] * 100
        print(f"  {s_name.upper():<6}: {count:>5} hasta (%{ratio:.2f})")

    print("\nHasta Etiket Dağılımı:")
    for lbl, counts in report["patient_label_distribution"].items():
        lbl_name = "Sağlıklı (0)" if lbl == "0" else "Nodüllü (1)"
        print(f"  {lbl_name:<14}: Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}")

    print(f"\nNadir Örüntü Fallback Sayısı: {report['fallback_rare_strata_count']} nadir grup")
    for fb in report["fallback_rare_strata_details"]:
        print(f"  Örüntü {fb['strat_key']} ({fb['count']} hasta): Atamalar: {fb['assignments']}")

    print(f"\nEtiket Dosyası SHA-256 Hash: {report['labels_file_sha256']}")
    print("=" * 80)
    print(f"Çıktı 1: {patient_splits_path}")
    print(f"Çıktı 2: {summary_path}")


if __name__ == "__main__":
    main()

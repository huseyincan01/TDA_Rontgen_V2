"""v2 Zon Etiketleri Üretim Betiği.

Kullanım:
    python pipelines/02_create_labels_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.data.labels_v2 import generate_zone_labels_v2, LABEL_RULE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 Zon Etiketleri Üretim Betiği")
    parser.add_argument(
        "--overlap-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zones" / "reports" / "nodule_pixel_overlap.csv",
        help="Girdi nodule_pixel_overlap.csv yolu.",
    )
    parser.add_argument(
        "--patients-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zones" / "patients",
        help="Hasta kesitleri dizini.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labels" / "zone_labels_v2.csv",
        help="Çıktı zone_labels_v2.csv yolu.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Pozitiflik eşiği (varsayılan: 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("v2 ZON ETİKETLEME BORU HATTI")
    print("=" * 80)
    print(f"Girdi Örtüşme Dosyası: {args.overlap_path}")
    print(f"Hasta Dizini:          {args.patients_dir}")
    print(f"Çıktı Etiket Dosyası:  {args.output_path}")
    print(f"Pozitiflik Eşiği:      %{args.threshold*100:.0f}")
    print(f"Etiket Kural Versiyonu: {LABEL_RULE_VERSION}")
    print("-" * 80)

    labels_df, report = generate_zone_labels_v2(
        overlap_csv_path=args.overlap_path,
        patients_dir=args.patients_dir,
        threshold=args.threshold,
        rule_version=LABEL_RULE_VERSION,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(args.output_path, index=False)

    report_path = args.output_path.parent / "labels_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("v2 ZON ETİKETLEME RAPORU")
    print("=" * 80)
    print(f"Toplam Hasta Sayısı:   {report['total_patients']:,}")
    print(f"  - Sağlıklı Kontrol:  {report['total_control_patients']:,}")
    print(f"  - Nodüllü Hasta:     {report['total_nodule_patients']:,}")
    print(f"  - Atanamayan Hasta:  {report['unassigned_nodule_patients_count']} (unassigned_nodule)")
    print(f"    Atanamayan Hasta ID'leri: {report['unassigned_nodule_patient_ids']}")
    print(f"Toplam Zon Satırı:     {report['total_zone_rows']:,}")
    print("\nZON BAZINDA DAĞILIM:")
    print(f"{'Zon':<22} | {'Pozitif':>8} | {'Temiz Neg':>10} | {'Ambiguous':>10} | {'Unassigned':>10} | {'Toplam':>8}")
    print("-" * 80)
    for z_key, z_stat in report["zone_summary"].items():
        name = z_stat["zone_name"]
        print(
            f"{name:<22} | {z_stat['positive']:>8} | {z_stat['clean_negative']:>10} | "
            f"{z_stat['ambiguous']:>10} | {z_stat['unassigned_nodule']:>10} | {z_stat['total']:>8}"
        )
    print("=" * 80)
    print(f"Çıktı kaydedildi: {args.output_path}")
    print(f"Rapor kaydedildi: {report_path}")


if __name__ == "__main__":
    main()

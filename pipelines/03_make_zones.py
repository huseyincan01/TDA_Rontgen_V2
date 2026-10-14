"""5-Bölge Akciğer Segmentasyonu ve Hasta Başına Bölge Kırpma Boru Hattı.

Kullanım:
    python pipelines/03_make_zones.py --limit 10
    python pipelines/03_make_zones.py --limit 1000 --output-dir data/processed/zones_pilot
    python pipelines/03_make_zones.py --output-dir data/processed/zones_full
"""

import os
import sys
import argparse
import csv
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

# Proje kökünü sys.path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter, ZONE_METADATA
from node21_tda.preprocessing.zones import ZoneCropper


def load_metadata_and_mappings(data_dir: str) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Metadata ve dosya adı eşleştirmelerini yükler."""
    meta_csv = os.path.join(data_dir, "metadata.csv")
    mapping_csv = os.path.join(os.path.dirname(data_dir), "original_data", "filenames_orig_and_new.csv")
    if not os.path.exists(mapping_csv):
        mapping_csv = os.path.join(data_dir, "filenames_orig_and_new.csv")

    source_dataset_map = {}
    if os.path.exists(mapping_csv):
        with open(mapping_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                node_id = row.get("node21_img_id")
                orig_ds = row.get("orig_dataset", "unknown")
                if node_id:
                    source_dataset_map[f"{node_id}.mha"] = orig_ds
                    source_dataset_map[node_id] = orig_ds

    images_meta = defaultdict(lambda: {"label": 0, "source_dataset": "node21_control", "bboxes": []})

    if os.path.exists(meta_csv):
        with open(meta_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_name = row.get("img_name")
                if not img_name:
                    continue
                label_val = int(row.get("label", 0))
                images_meta[img_name]["label"] = max(images_meta[img_name]["label"], label_val)

                if img_name in source_dataset_map:
                    images_meta[img_name]["source_dataset"] = source_dataset_map[img_name]

                # Bounding box koordinatları
                try:
                    bx = float(row.get("x", 0))
                    by = float(row.get("y", 0))
                    bw = float(row.get("width", 0))
                    bh = float(row.get("height", 0))
                    if bw > 0 and bh > 0:
                        images_meta[img_name]["bboxes"].append((bx, by, bw, bh))
                except (ValueError, TypeError):
                    pass

    return images_meta, source_dataset_map


def main():
    parser = argparse.ArgumentParser(description="5-Bölge Akciğer Segmentasyonu ve Hasta Başına Kesim Pipeline'ı")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(project_root, "dataset_node21", "cxr_images", "proccessed_data"),
        help="Görüntülerin ve metadata.csv'nin bulunduğu dizin (varsayılan: proccessed_data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "data", "processed", "zones"),
        help="Her hasta için klasörlerin ve kesilmiş bölgelerin kaydedileceği çıktı dizini"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="İşlenecek maksimum görüntü sayısı (test ve pilot için örn: 10 veya 1000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="İşlem birimi: 'cuda' veya 'cpu' (varsayılan: otomatik GPU)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Kesim kenar payı pikseli (varsayılan: 5)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Daha önce işlenmiş hasta klasörlerini atla"
    )

    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(images_dir):
        images_dir = args.data_dir

    if not os.path.exists(images_dir):
        print(f"HATA: Görüntü klasörü bulunamadı: {images_dir}")
        sys.exit(1)

    print("=" * 75)
    print(" 5-BÖLGELİ AKCİĞER SEGMENTASYONU VE BÖLGE KESME BORU HATTI")
    print("=" * 75)
    print(f"  Veri Kaynağı: {images_dir}")
    print(f"  Çıktı Dizini: {args.output_dir}")
    print(f"  Her hasta için ayrı klasör: {os.path.join(args.output_dir, 'patients')}")

    # Metadata yükle
    images_meta, _ = load_metadata_and_mappings(args.data_dir)

    # Görüntü dosyalarını listele
    all_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".mha")]
    all_files.sort()
    print(f"  Toplam Tespit Edilen Görüntü: {len(all_files)}")

    if args.limit:
        all_files = all_files[:args.limit]
        print(f"  --limit Parametresi Uygulandı: {len(all_files)} adet işlenecek.")

    # Segmenter ve Cropper'ı başlat
    segmenter = PSPNet5ZoneSegmenter(device=args.device)
    print(f"  Model: TorchXRayVision PSPNet ({segmenter.device.type.upper()})")

    cropper = ZoneCropper(
        output_base_dir=args.output_dir,
        padding=args.padding,
        save_mha=True,
        save_png=True,
        save_overview=True
    )

    all_nodule_records = []
    patient_summaries = []
    total_infer_time = 0.0

    print("-" * 75)
    pbar = tqdm(all_files, desc="İşleniyor", unit="hasta")

    for fname in pbar:
        patient_id = os.path.splitext(fname)[0]
        patient_dir = os.path.join(cropper.patients_dir, patient_id)

        # Önceden işlenmişse atla seçeneği
        if args.skip_existing and os.path.exists(os.path.join(patient_dir, "patient_info.json")):
            continue

        img_path = os.path.join(images_dir, fname)
        try:
            sitk_img = sitk.ReadImage(img_path)
            img_array = sitk.GetArrayFromImage(sitk_img)
            if len(img_array.shape) == 3:
                img_array = img_array[0]

            # 1. Segmentasyon
            seg_res = segmenter.segment(img_array)
            zone_map = seg_res["zone_map"]
            total_infer_time += seg_res["inference_time_ms"]

            meta = images_meta.get(fname, {"label": 0, "source_dataset": "unknown", "bboxes": []})

            # 2. Her hasta için klasör oluştur ve 5 bölgeyi kesip kaydet
            patient_meta = cropper.process_patient(
                patient_id=patient_id,
                sitk_img=sitk_img,
                zone_map=zone_map,
                bboxes=meta["bboxes"],
                source_dataset=meta["source_dataset"],
                label=meta["label"]
            )

            # Toplu rapor kayıtları
            all_nodule_records.extend(patient_meta["nodules"])

            summary_row = {
                "patient_id": patient_id,
                "label": meta["label"],
                "source_dataset": meta["source_dataset"],
                "width": img_array.shape[1],
                "height": img_array.shape[0],
                "nodule_count": len(meta["bboxes"]),
                "z1_pixels": patient_meta["zones"].get(1, {}).get("pixel_count", 0),
                "z2_pixels": patient_meta["zones"].get(2, {}).get("pixel_count", 0),
                "z3_pixels": patient_meta["zones"].get(3, {}).get("pixel_count", 0),
                "z4_pixels": patient_meta["zones"].get(4, {}).get("pixel_count", 0),
                "z5_pixels": patient_meta["zones"].get(5, {}).get("pixel_count", 0),
                "infer_time_ms": round(seg_res["inference_time_ms"], 2)
            }
            patient_summaries.append(summary_row)

        except Exception as e:
            print(f"\nHATA: {fname} işlenirken problem oluştu: {e}")

    # Toplu CSV raporlarını kaydet
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    nodule_csv_path = os.path.join(reports_dir, "nodule_5zones_mapping.csv")
    if all_nodule_records:
        with open(nodule_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_nodule_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_nodule_records)
        print(f"\n[OK] Nodül Bölge Eşleştirme Raporu: {nodule_csv_path}")

    patient_csv_path = os.path.join(reports_dir, "patients_zones_summary.csv")
    if patient_summaries:
        with open(patient_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(patient_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(patient_summaries)
        print(f"[OK] Hasta Bölge İstatistikleri: {patient_csv_path}")

    # Özet istatistikler
    avg_infer = total_infer_time / max(1, len(patient_summaries))
    print("=" * 75)
    print(" TAMAMLANDI!")
    print(f"  Toplam İşlenen Hasta: {len(patient_summaries)}")
    print(f"  Eşleştirilen Nodül Sayısı: {len(all_nodule_records)}")
    print(f"  Ortalama GPU Çıkarım Süresi: {avg_infer:.2f} ms / görüntü")
    print(f"  Klasör Yapısı: {cropper.patients_dir}/<hasta_id>/ (İçinde 5 ayrı kesilmiş bölge)")
    print("=" * 75)


if __name__ == "__main__":
    main()

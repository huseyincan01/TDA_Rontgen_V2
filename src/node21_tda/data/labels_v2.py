"""v2 Zon Etiketleme Modülü.

Bu modül:
- Mevcut `nodule_pixel_overlap.csv` içindeki gerçek piksel örtüşme oranlarını kullanır.
- Bir nodülün (bbox) alanının >= %20'si bir zon maskesindeyse o zonu pozitif (`zone_label=1`, `label_state='positive'`) yapar.
- Aynı bbox iki zona >= %20 taşıyorsa iki zon da pozitif olur.
- Bbox örtüşmesi %0 ile %20 arasındaysa o zon `label_state='ambiguous'` olarak işaretlenir (`zone_label=0`),
  böylece eğitimde ve değerlendirmede negatif havuzuna alınmaz.
- Hiçbir zonu >= %20 örtüşmeye ulaşmayan nodüllü hastalar `label_state='unassigned_nodule'` olarak işaretlenir.
- 0% örtüşmeye sahip zonlar temiz negatif (`zone_label=0`, `label_state='negative'`) olarak etiketlenir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

ZONE_METADATA: Dict[int, str] = {
    1: "Sağ Üst (R-Upper)",
    2: "Sağ Orta (R-Mid)",
    3: "Sağ Alt (R-Lower)",
    4: "Sol Üst (L-Upper)",
    5: "Sol Alt (L-Lower)",
}

LABEL_RULE_VERSION = "v2_overlap20"


def generate_zone_labels_v2(
    overlap_csv_path: Path,
    patients_dir: Path,
    threshold: float = 0.20,
    rule_version: str = LABEL_RULE_VERSION,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Tüm hastalar ve 5 zon için v2 etiket tablosunu üretir."""
    if not overlap_csv_path.exists():
        raise FileNotFoundError(f"Örtüşme raporu bulunamadı: {overlap_csv_path}")
    if not patients_dir.exists():
        raise FileNotFoundError(f"Hasta dizini bulunamadı: {patients_dir}")

    overlap_df = pd.read_csv(overlap_csv_path)

    # 1. Hasta listesi ve hasta etiketlerini topla
    patient_dirs = sorted([p for p in patients_dir.iterdir() if p.is_dir()])
    patient_info_map: Dict[str, Dict[str, Any]] = {}
    for p_dir in patient_dirs:
        pid = p_dir.name
        # Hasta etiketi: nXXXX ise 1, cXXXX ise 0
        if pid.startswith("n"):
            p_label = 1
        elif pid.startswith("c"):
            p_label = 0
        else:
            # Fallback to patient_info.json if available
            info_file = p_dir / "patient_info.json"
            if info_file.exists():
                with open(info_file, "r", encoding="utf-8") as f:
                    p_label = int(json.load(f).get("label", 0))
            else:
                p_label = 0
        patient_info_map[pid] = {"patient_label": p_label}

    # 2. Her hasta için zon bazında nodül örtüşmelerini analiz et
    # patient_nodules[pid] = list of nodule dicts
    grouped_nodules: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in overlap_df.iterrows():
        pid = str(row["image"])
        nid = str(row["nodule_id"])
        nodule_dict = {
            "nodule_id": nid,
            "ratios": {z: float(row[f"z{z}_ratio"]) for z in range(1, 6)},
        }
        grouped_nodules.setdefault(pid, []).append(nodule_dict)

    # 3. Her hastanın zon bazında max örtüşmesini ve >= threshold olup olmadığını belirle
    # Unassigned nodule hastalarını tespit et: nodüllü (patient_label == 1) ama hiçbir zonu >= threshold değil
    unassigned_patients = set()
    for pid, meta in patient_info_map.items():
        if meta["patient_label"] == 1:
            nodules = grouped_nodules.get(pid, [])
            has_pos_zone = False
            for z in range(1, 6):
                max_z_ratio = max((n["ratios"][z] for n in nodules), default=0.0)
                if max_z_ratio >= threshold:
                    has_pos_zone = True
                    break
            if not has_pos_zone:
                unassigned_patients.add(pid)

    # 4. 4.882 hasta x 5 zon = 24.410 satır oluştur
    records: List[Dict[str, Any]] = []

    for pid in sorted(patient_info_map.keys()):
        p_label = patient_info_map[pid]["patient_label"]
        nodules = grouped_nodules.get(pid, [])
        is_unassigned_patient = pid in unassigned_patients

        for zone_id in range(1, 6):
            zone_name = ZONE_METADATA[zone_id]

            if p_label == 0:
                # Kontrol / Sağlıklı hasta: Kesin temiz negatif
                max_overlap = 0.0
                matching_nodule_ids = ""
                zone_label = 0
                label_state = "negative"

            elif is_unassigned_patient:
                # Hiçbir zonu %20'ye ulaşmayan nodüllü hasta
                # İlgili zonda varsa nodül örtüşmesini ve nodül id'lerini kaydet
                z_ratios = [(n["nodule_id"], n["ratios"][zone_id]) for n in nodules]
                max_overlap = max((r[1] for r in z_ratios), default=0.0)
                active_nids = [r[0] for r in z_ratios if r[1] > 0.0]
                matching_nodule_ids = ",".join(active_nids)
                zone_label = 0
                label_state = "unassigned_nodule"

            else:
                # Normal nodüllü hasta
                z_ratios = [(n["nodule_id"], n["ratios"][zone_id]) for n in nodules]
                max_overlap = max((r[1] for r in z_ratios), default=0.0)

                if max_overlap >= threshold:
                    # >= %20 örtüşme: Pozitif
                    zone_label = 1
                    label_state = "positive"
                    active_nids = [r[0] for r in z_ratios if r[1] >= threshold]
                    matching_nodule_ids = ",".join(active_nids)
                elif max_overlap > 0.0:
                    # 0 < örtüşme < %20: Ambiguous (Sınırda / Belirsiz doku)
                    zone_label = 0
                    label_state = "ambiguous"
                    active_nids = [r[0] for r in z_ratios if r[1] > 0.0]
                    matching_nodule_ids = ",".join(active_nids)
                else:
                    # %0 örtüşme: Temiz negatif
                    zone_label = 0
                    label_state = "negative"
                    matching_nodule_ids = ""

            records.append({
                "patient_id": pid,
                "zone_id": zone_id,
                "zone_name": zone_name,
                "zone_label": zone_label,
                "label_state": label_state,
                "patient_label": p_label,
                "nodule_ids": matching_nodule_ids,
                "max_bbox_overlap": round(max_overlap, 4),
                "label_rule_version": rule_version,
            })

    labels_df = pd.DataFrame(records)

    # 5. Raporlama
    summary_report: Dict[str, Any] = {
        "total_patients": len(patient_info_map),
        "total_nodule_patients": sum(1 for m in patient_info_map.values() if m["patient_label"] == 1),
        "total_control_patients": sum(1 for m in patient_info_map.values() if m["patient_label"] == 0),
        "unassigned_nodule_patients_count": len(unassigned_patients),
        "unassigned_nodule_patient_ids": sorted(list(unassigned_patients)),
        "label_rule_version": rule_version,
        "threshold_ratio": threshold,
        "total_zone_rows": len(labels_df),
        "zone_summary": {},
    }

    for zone_id in range(1, 6):
        z_df = labels_df[labels_df["zone_id"] == zone_id]
        pos_count = int((z_df["label_state"] == "positive").sum())
        clean_neg_count = int((z_df["label_state"] == "negative").sum())
        amb_count = int((z_df["label_state"] == "ambiguous").sum())
        unassigned_count = int((z_df["label_state"] == "unassigned_nodule").sum())

        summary_report["zone_summary"][f"zone_{zone_id}"] = {
            "zone_name": ZONE_METADATA[zone_id],
            "positive": pos_count,
            "clean_negative": clean_neg_count,
            "ambiguous": amb_count,
            "unassigned_nodule": unassigned_count,
            "total": len(z_df),
        }

    return labels_df, summary_report

"""5 Anatomik Bölge Kırpma (Cropping), Kayıt ve Nodül Eşleştirme Modülü.

Bu modül, her hasta için ayrı bir klasör oluşturur ve içinde 5 farklı anatomik bölgeyi
(Sağ 3, Sol 2 lob) hem fiziksel .mha formatında hem de görselleştirme için .png formatında kesip saklar.
Ayrıca lezyonları/nodülleri bu bölgelerle eşleştirir.
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure

from node21_tda.preprocessing.segmenter import ZONE_METADATA


class ZoneCropper:
    """Her hasta için 5 anatomik bölgeyi kesip klasörleyen ve nodülleri eşleştiren sınıf."""

    def __init__(
        self,
        output_base_dir: str,
        padding: int = 5,
        save_mha: bool = True,
        save_png: bool = True,
        save_overview: bool = True
    ):
        """
        Args:
            output_base_dir: Hasta klasörlerinin ve çıktıların saklanacağı ana dizin.
            padding: Bölge sınırlarına eklenecek piksel boşluğu (varsayılan: 5).
            save_mha: .mha formatında kesilmiş bölge kaydı.
            save_png: .png formatında görselleştirme kaydı.
            save_overview: 3 panelli hasta özet görseli kaydı.
        """
        self.output_base_dir = output_base_dir
        self.patients_dir = os.path.join(output_base_dir, "patients")
        self.padding = padding
        self.save_mha = save_mha
        self.save_png = save_png
        self.save_overview = save_overview

        os.makedirs(self.patients_dir, exist_ok=True)

    @staticmethod
    def normalize_for_display(img_array: np.ndarray) -> np.ndarray:
        """Görsel gösterim için görüntüyü [0, 1] aralığına normalize eder (1-99 persentil kontrastı)."""
        p1, p99 = np.percentile(img_array, (1, 99))
        if p99 > p1:
            return np.clip((img_array - p1) / (p99 - p1), 0.0, 1.0)
        return np.zeros_like(img_array, dtype=np.float32)

    def map_nodules(
        self,
        zone_map: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]],
        img_name: str,
        source_dataset: str = "unknown"
    ) -> List[Dict]:
        """Verilen nodül bounding box'larını 5 bölge ile eşleştirir.

        Args:
            zone_map: (H, W) uint8 bölge haritası.
            bboxes: [(x, y, w, h), ...] nodül koordinatları listesi.
            img_name: Görsel/hasta adı.
            source_dataset: Kaynak veri seti.

        Returns:
            Eşleştirilmiş nodül kayıtları listesi.
        """
        H, W = zone_map.shape
        records = []

        for idx, (bx, by, bw, bh) in enumerate(bboxes, start=1):
            cx = int(bx + bw / 2.0)
            cy = int(by + bh / 2.0)
            cx_c = max(0, min(W - 1, cx))
            cy_c = max(0, min(H - 1, cy))

            z_val = int(zone_map[cy_c, cx_c])

            # Merkez nokta akciğer dışına ya da sınır çizgisine düşmüşse bbox içi baskın zona bak
            is_fallback = False
            if z_val == 0:
                y_start = max(0, int(by))
                y_end = min(H, int(by + bh))
                x_start = max(0, int(bx))
                x_end = min(W, int(bx + bw))
                roi = zone_map[y_start:y_end, x_start:x_end]
                non_zero = roi[roi > 0]
                if len(non_zero) > 0:
                    z_val = int(np.bincount(non_zero).argmax())
                    is_fallback = True

            meta = ZONE_METADATA.get(z_val, {
                "name": "Retrokardiyak / Akciğer Dışı",
                "short": "Retrokardiyak",
                "lung": "none",
                "level": "none"
            })

            records.append({
                "image": img_name,
                "dataset": source_dataset,
                "nodule_id": idx,
                "bbox_x": int(bx),
                "bbox_y": int(by),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "center_x": cx,
                "center_y": cy,
                "assigned_zone_id": z_val,
                "assigned_zone_name": meta["name"],
                "assigned_zone_short": meta["short"],
                "is_fallback": is_fallback
            })

        return records

    def process_patient(
        self,
        patient_id: str,
        sitk_img: sitk.Image,
        zone_map: np.ndarray,
        bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
        source_dataset: str = "unknown",
        label: int = 0
    ) -> Dict:
        """Bir hasta için özel klasör oluşturur ve 5 farklı bölgeyi kesip kaydeder.

        Args:
            patient_id: Hasta veya görsel kimliği (örn: 'c0001', 'n0239').
            sitk_img: SimpleITK Image nesnesi.
            zone_map: (H, W) uint8 bölge haritası (1..5).
            bboxes: Varsa nodül koordinatları [(x, y, w, h)].
            source_dataset: Kaynak veri kümesi.
            label: 0 (sağlıklı) veya 1 (nodüllü).

        Returns:
            Hasta işlem özeti sözlüğü.
        """
        if bboxes is None:
            bboxes = []

        patient_dir = os.path.join(self.patients_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        orig_spacing = sitk_img.GetSpacing()
        orig_direction = sitk_img.GetDirection()

        img_array = sitk.GetArrayFromImage(sitk_img)
        if len(img_array.shape) == 3:
            img_array = img_array[0]
        H, W = img_array.shape

        # Nodülleri eşle
        nodule_records = self.map_nodules(zone_map, bboxes, patient_id, source_dataset)

        zones_summary = {}

        # -------------------------------------------------------------
        # 5 FARKLI BÖLGENİN HER BİRİNİ AYRI AYRI KES VE KAYDET
        # -------------------------------------------------------------
        for z_id in range(1, 6):
            z_meta = ZONE_METADATA[z_id]
            file_prefix = z_meta["file_prefix"]
            z_mask = (zone_map == z_id)
            pixel_count = int(np.sum(z_mask))

            if pixel_count == 0:
                zones_summary[z_id] = {
                    "zone_name": z_meta["name"],
                    "pixel_count": 0,
                    "bbox": None,
                    "status": "empty"
                }
                continue

            # Bölgenin sınırlayıcı kutusunu (bounding box) bul
            y_indices, x_indices = np.where(z_mask)
            min_y = max(0, int(np.min(y_indices)) - self.padding)
            max_y = min(H, int(np.max(y_indices)) + 1 + self.padding)
            min_x = max(0, int(np.min(x_indices)) - self.padding)
            max_x = min(W, int(np.max(x_indices)) + 1 + self.padding)

            # 1. Maskelenmiş Kesik (Bölge dışı pikseller 0)
            masked_crop = np.zeros((max_y - min_y, max_x - min_x), dtype=img_array.dtype)
            crop_mask = z_mask[min_y:max_y, min_x:max_x]
            crop_raw = img_array[min_y:max_y, min_x:max_x]
            masked_crop[crop_mask] = crop_raw[crop_mask]

            # 2. .mha formatında kaydet (Orijinal fiziksel koordinat ofsetini koruyarak)
            if self.save_mha:
                mha_crop_sitk = sitk.GetImageFromArray(masked_crop)
                mha_crop_sitk.SetSpacing(orig_spacing)
                mha_crop_sitk.SetDirection(orig_direction)
                # Yeni fiziksel orijin noktası
                new_origin = sitk_img.TransformIndexToPhysicalPoint((int(min_x), int(min_y)))
                mha_crop_sitk.SetOrigin(new_origin)

                crop_mha_path = os.path.join(patient_dir, f"{file_prefix}.mha")
                sitk.WriteImage(mha_crop_sitk, crop_mha_path)

            # 3. .png formatında önizleme kaydet
            if self.save_png:
                norm_crop = self.normalize_for_display(masked_crop)
                # Şeffaf veya yüksek kontrastlı görsel
                crop_png_path = os.path.join(patient_dir, f"{file_prefix}.png")
                plt.imsave(crop_png_path, norm_crop, cmap="gray")

            zones_summary[z_id] = {
                "zone_name": z_meta["name"],
                "short_tag": z_meta["short"],
                "pixel_count": pixel_count,
                "crop_shape": [int(max_y - min_y), int(max_x - min_x)],
                "crop_bbox": {
                    "min_y": min_y, "max_y": max_y,
                    "min_x": min_x, "max_x": max_x
                },
                "status": "success"
            }

        # -------------------------------------------------------------
        # TAM 5-BÖLGELİ MASKE (.mha) VE OVERVIEW KAYDI
        # -------------------------------------------------------------
        if self.save_mha:
            mha_full_mask = sitk.GetImageFromArray(zone_map)
            mha_full_mask.CopyInformation(sitk_img)
            sitk.WriteImage(mha_full_mask, os.path.join(patient_dir, "mask_5zones.mha"))

        if self.save_overview:
            self._save_patient_overview(
                patient_dir=patient_dir,
                patient_id=patient_id,
                img_array=img_array,
                zone_map=zone_map,
                nodules=nodule_records,
                source_dataset=source_dataset,
                label=label
            )

        # Hasta özet JSON dosyası
        patient_meta = {
            "patient_id": patient_id,
            "source_dataset": source_dataset,
            "label": label,
            "original_size": [H, W],
            "spacing": list(orig_spacing),
            "zones": zones_summary,
            "nodules": nodule_records
        }
        with open(os.path.join(patient_dir, "patient_info.json"), "w", encoding="utf-8") as f:
            json.dump(patient_meta, f, indent=2, ensure_ascii=False)

        return patient_meta

    def _save_patient_overview(
        self,
        patient_dir: str,
        patient_id: str,
        img_array: np.ndarray,
        zone_map: np.ndarray,
        nodules: List[Dict],
        source_dataset: str,
        label: int
    ):
        """Hasta için 3 panelli özet görselini kaydeder."""
        H, W = img_array.shape
        img_disp = self.normalize_for_display(img_array)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

        # Panel 1: Orijinal CXR + BBox
        axes[0].imshow(img_disp, cmap="gray")
        status_str = "Pozitif (Nodüllü)" if label == 1 else "Negatif (Sağlıklı)"
        axes[0].set_title(f"{patient_id} ({source_dataset.upper()})\nDurum: {status_str}", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        for nod in nodules:
            bx, by, bw, bh = nod["bbox_x"], nod["bbox_y"], nod["bbox_w"], nod["bbox_h"]
            rect = patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor="#FF2A2A", facecolor="none", linestyle="--")
            axes[0].add_patch(rect)
            axes[0].text(bx, max(0, by - 8), "Nodül", color="#FF2A2A", fontsize=9, fontweight="bold", backgroundcolor="black")

        # Panel 2: 5 Renkli Bölge Haritası
        zone_rgb = np.zeros((H, W, 3), dtype=np.float32)
        for z_id in range(1, 6):
            zone_rgb[zone_map == z_id] = ZONE_METADATA[z_id]["color"]
        axes[1].imshow(zone_rgb)
        axes[1].set_title("5 Anatomik Bölge Haritası\n(Sağ: 3 Bölge, Sol: 2 Bölge)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Röntgen Üzerinde Renkli Zon Overlay + Konturlar + Nodül Etiketi
        axes[2].imshow(img_disp, cmap="gray")
        tint_map = np.zeros((H, W, 4), dtype=np.float32)
        for z_id in range(1, 6):
            tint_map[zone_map == z_id, :3] = ZONE_METADATA[z_id]["color"]
            tint_map[zone_map == z_id, 3] = 0.35
        axes[2].imshow(tint_map)

        for z_id in range(1, 6):
            contours = measure.find_contours(zone_map == z_id, 0.5)
            for c in contours:
                axes[2].plot(c[:, 1], c[:, 0], color="white", linewidth=1.0)

        for nod in nodules:
            bx, by, bw, bh = nod["bbox_x"], nod["bbox_y"], nod["bbox_w"], nod["bbox_h"]
            rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor="#FF0000", facecolor="none")
            axes[2].add_patch(rect)
            tag = nod["assigned_zone_short"]
            axes[2].text(bx, max(0, by - 10), tag, color="#FF2A2A", fontsize=9, fontweight="bold", backgroundcolor="black")

        axes[2].set_title("Bölge Overlay ve Eşleşen Nodüller", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(f"Hasta: {patient_id} — Anatomik 5-Bölge Ayrımı ve Kesiti", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(patient_dir, "overview.png"), bbox_inches="tight")
        plt.close(fig)

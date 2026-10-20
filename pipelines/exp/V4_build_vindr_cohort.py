"""V4 — VinDr-CXR harici kohortun (400 görüntü) deterministik yeniden üretimi.

Neden bu betik:
    Harici doğrulama kohortu (150 nodül/kitle + 250 saf normal) ve nodül boyutları
    daha önce **scriptsiz** türetilmişti; ``cohort_with_diameters.csv`` içindeki
    ``nodule_diam_px`` değerlerinin hangi koordinat uzayında olduğu belirsizdi
    (değerler 32–1327 px; kullanılan JPEG'ler 682–1056 px). Bu betik kaynağı
    şeffaflaştırır.

Tanımlar (ön-kayıtlı):
    * **Pozitif (nodül/kitle):** ``class_name`` içinde "nodule" veya "mass" geçen
      görüntüler (annotasyon uzayı bbox'ları).
    * **Saf normal:** tüm anotatörlerin "No finding" dediği görüntüler.
    * **Örnekleme:** ``np.random.RandomState(42)`` ile 150 pozitif + 250 normal
      (önceki kohortla aynı mantık; betik örtüşmeyi raporlar).
    * **Boyut ölçütü:** ``bbox_max_dim_px`` = nodül/kitle kutularının en uzun
      kenarının **annotasyon (orijinal DICOM) uzayındaki** maksimumu.
    * **Ölçek-bağımsız boyut:** ``rel_size = bbox_max_dim_px / image_h`` (JPEG
      yüksekliği). VinDr kohortundaki JPEG'ler indirgenmiş olduğundan ve spacing
      bulunmadığından mm dönüşümü **yapılamaz**; alt gruplar px/mm yerine
      ölçek-bağımsız üçte birlik dilimlerle tanımlanır.

Çıktı:
    data/external_vindr/cohort_metadata.csv        (image_id, patient_label, group)
    data/external_vindr/cohort_with_diameters.csv  (+ bbox_max_dim_px, image_h/w, rel_size)
    data/external_vindr/v4_cohort_build_report.json

Kullanım:
    python pipelines/exp/V4_build_vindr_cohort.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "external_vindr"
IMAGES_DIR = OUT_DIR / "images"

SEED = 42
N_POSITIVE = 150
N_NORMAL = 250


def find_annotations() -> Path:
    try:
        import torchxrayvision as xrv

        cand = Path(xrv.datasets.datapath) / "vinbigdata-train.csv.gz"
        if cand.exists():
            return cand
    except Exception:
        pass
    fallback = PROJECT_ROOT / "venv" / "Lib" / "site-packages" / "torchxrayvision" / "data" / "vinbigdata-train.csv.gz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "VinDr annotasyon CSV'si bulunamadı. Kontrol edin:\n"
        "  python -c \"import torchxrayvision as xrv; print(xrv.datasets.datapath)\""
    )


def main() -> None:
    # Örtüşme kontrolü için eski kohortu önce oku.
    old_ids: list[str] = []
    old_csv = OUT_DIR / "cohort_with_diameters.csv"
    if old_csv.exists():
        try:
            old_ids = pd.read_csv(old_csv)["image_id"].astype(str).tolist()
        except Exception:
            old_ids = []

    ann_path = find_annotations()
    df = pd.read_csv(ann_path)
    print(f"Annotasyon: {ann_path} ({len(df)} satır, {df['image_id'].nunique()} görüntü)")

    # --- Örnekleme (eski kohortla aynı mantık) ---
    class_sets = df.groupby("image_id")["class_name"].apply(lambda s: set(s))
    pure_normal_ids = class_sets[class_sets == {"No finding"}].index.tolist()
    nodule_rows = df[df["class_name"].str.lower().str.contains("nodule|mass", na=False)]
    nodule_ids = nodule_rows["image_id"].unique().tolist()

    rng = np.random.RandomState(SEED)
    positives = rng.choice(nodule_ids, size=N_POSITIVE, replace=False).tolist()
    normals = rng.choice(pure_normal_ids, size=N_NORMAL, replace=False).tolist()
    print(f"Aday: {len(nodule_ids)} pozitif, {len(pure_normal_ids)} saf normal")
    print(f"Seçilen: {len(positives)} pozitif + {len(normals)} normal")

    new_ids = [str(i) for i in positives + normals]
    overlap = len(set(new_ids) & set(old_ids)) if old_ids else None
    if overlap is not None:
        print(f"Eski kohortla örtüşme: {overlap}/{len(new_ids)}")

    # --- Boyutlar (annotasyon uzayı) ---
    nrows = nodule_rows[nodule_rows["image_id"].isin(positives)].copy()
    nrows["bbox_max_dim_px"] = np.maximum(
        nrows["x_max"] - nrows["x_min"], nrows["y_max"] - nrows["y_min"]
    )
    size_map = nrows.groupby("image_id")["bbox_max_dim_px"].max()

    rows = []
    for img_id in positives:
        rows.append({"image_id": img_id, "patient_label": 1, "group": "nodule",
                     "bbox_max_dim_px": float(size_map.get(img_id, np.nan))})
    for img_id in normals:
        rows.append({"image_id": img_id, "patient_label": 0, "group": "normal",
                     "bbox_max_dim_px": np.nan})
    cohort = pd.DataFrame(rows)

    # --- Görüntü boyutları (mevcut JPEG'lerden) ---
    h_list, w_list, present = [], [], []
    for img_id in cohort["image_id"]:
        p = IMAGES_DIR / f"{img_id}.jpg"
        if p.exists():
            w, h = Image.open(p).size
            present.append(True)
        else:
            w, h = np.nan, np.nan
            present.append(False)
        h_list.append(h)
        w_list.append(w)
    cohort["image_h"] = h_list
    cohort["image_w"] = w_list
    cohort["image_present"] = present
    cohort["rel_size"] = cohort["bbox_max_dim_px"] / cohort["image_h"]
    print(f"JPEG mevcut: {int(np.sum(present))}/{len(cohort)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort[["image_id", "patient_label", "group"]].to_csv(
        OUT_DIR / "cohort_metadata.csv", index=False
    )
    cohort.to_csv(OUT_DIR / "cohort_with_diameters.csv", index=False)

    sub = cohort.loc[cohort["patient_label"] == 1, "bbox_max_dim_px"].astype(float)
    report = {
        "method": "v4_deterministic_cohort_build",
        "seed": SEED,
        "source": str(ann_path),
        "logic": {
            "positive": "class_name ~ 'nodule|mass'",
            "normal": "tüm rad_id'ler 'No finding'",
            "size_metric": "bbox_max_dim_px (annotasyon/orijinal DICOM uzayı, maks. anotatör)",
            "scale_free_metric": "rel_size = bbox_max_dim_px / image_h",
            "mm_conversion": "YOK — JPEG kohortu indirgenmiş, DICOM spacing yok",
        },
        "n_positive": len(positives),
        "n_normal": len(normals),
        "n_images_present": int(np.sum(present)),
        "bbox_max_dim_px": {"min": float(sub.min()), "median": float(sub.median()), "max": float(sub.max())},
        "image_h": {"min": float(np.nanmin(h_list)), "median": float(np.nanmedian(h_list)),
                    "max": float(np.nanmax(h_list))},
        "overlap_with_previous_cohort": overlap,
    }
    (OUT_DIR / "v4_cohort_build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Kaydedildi: cohort_metadata.csv, cohort_with_diameters.csv, v4_cohort_build_report.json")


if __name__ == "__main__":
    main()

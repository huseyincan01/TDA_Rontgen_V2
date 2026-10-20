"""VinDr-CXR JPEG'lerini (3x indirgenmiş Kaggle arşivi) kohort için çıkarır.

Bu, ``pipelines/exp/V4_build_vindr_cohort.py`` ile üretilen kohort CSV'sini okur,
yalnızca kohorttaki görüntüleri arşivden ``data/external_vindr/images/`` altına
açar. Arşiv **silinmez** (yeniden üretilebilirlik için).

Arşiv (örnek):
    vinbigdata-competition-jpg-data-3x-downsampled.zip
    https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection

Kullanım:
    python scratch/extract_vindr_images.py --zip <arsiv.zip>
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORT_CSV = PROJECT_ROOT / "data/external_vindr/cohort_with_diameters.csv"
IMAGES_DIR = PROJECT_ROOT / "data/external_vindr/images"


def main() -> None:
    parser = argparse.ArgumentParser(description="VinDr kohort JPEG'lerini çıkar")
    parser.add_argument("--zip", required=True, type=Path, help="VinDr JPEG zip arşivi")
    parser.add_argument("--overwrite", action="store_true", help="Mevcut JPEG'lerin üzerine yaz")
    args = parser.parse_args()

    import pandas as pd

    if not COHORT_CSV.exists():
        raise FileNotFoundError(
            f"Kohort CSV yok: {COHORT_CSV}\nÖnce: python pipelines/exp/V4_build_vindr_cohort.py"
        )
    ids = pd.read_csv(COHORT_CSV)["image_id"].astype(str).tolist()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(args.zip) as zf:
        names = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
        for img_id in ids:
            target = IMAGES_DIR / f"{img_id}.jpg"
            if target.exists() and not args.overwrite:
                extracted += 1
                continue
            member = names.get(f"{img_id}.jpg")
            if member is None:
                print(f"  UYARI: {img_id}.jpg arşivde yok")
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    print(f"Hazır: {extracted}/{len(ids)} görüntü → {IMAGES_DIR}")


if __name__ == "__main__":
    main()

"""V4 — NODE21 referans doku histogramlarının (200 hasta havuzu) yeniden üretimi.

Neden bu betik:
    VinDr-CXR harici doğrulamasında (``V4_external_validation_vindr.py``) histogram
    eşleme için "havuzlanmış referans" gerekir. Bu dosya
    (``data/processed/node21_200pt_ref_histograms.npz``) daha önce **scriptsiz**
    üretilmişti; bu betik onu deterministik ve denetlenebilir biçimde yeniden üretir.

Tasarım:
    * Referans **yalnızca train** split'inden seçilir (val/test'e dokunulmaz).
    * 200 hasta, ``np.random.RandomState(SEED)`` ile hasta kimliğine göre sıralı
      listeden rastgele seçilir.
    * Her zon için yalnızca **doku pikselleri** (``v > 0``) havuzlanır; ızgara
      eşleme referansın değer dağılımını kullandığından satır yapısı önemsizdir,
      bu yüzden dağılım 1B olarak saklanır (yer tasarrufu).
    * Havuz, zon başına en fazla ``MAX_PIXELS_PER_ZONE`` piksele indirgenir
      (yüzdelik kestirimi için fazlasıyla yeterli).

Çıktı:
    data/processed/node21_200pt_ref_histograms.npz   (z1..z5, uint16)
    data/external_vindr/v4_ref_histograms_report.json (hasta listesi + hash)

Kullanım:
    python pipelines/exp/V4_build_ref_histograms.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SEED = 42
N_PATIENTS = 200
MAX_PIXELS_PER_ZONE = 300_000

SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits_v2" / "patient_splits.csv"
ZONES_DIR = PROJECT_ROOT / "data" / "processed" / "zones" / "patients"
OUT_NPZ = PROJECT_ROOT / "data" / "processed" / "node21_200pt_ref_histograms.npz"
OUT_REPORT = PROJECT_ROOT / "data" / "external_vindr" / "v4_ref_histograms_report.json"

ZONE_FILES = {
    1: "zone_1_sag_ust.mha",
    2: "zone_2_sag_orta.mha",
    3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha",
    5: "zone_5_sol_alt.mha",
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    splits = pd.read_csv(SPLITS_PATH)
    train_ids = sorted(splits.loc[splits["split"] == "train", "patient_id"].astype(str))
    rng = np.random.RandomState(SEED)
    picked = sorted(rng.choice(train_ids, size=N_PATIENTS, replace=False).tolist())
    print(f"Train hastası: {len(train_ids)} | Referans için seçilen: {len(picked)} (seed={SEED})")

    pools: dict[int, list[np.ndarray]] = {z: [] for z in range(1, 6)}
    missing = 0
    t0 = time.time()
    for i, pid in enumerate(picked, start=1):
        for z in range(1, 6):
            p = ZONES_DIR / pid / ZONE_FILES[z]
            if not p.exists():
                missing += 1
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            if arr.ndim == 3:
                arr = arr[0]
            tissue = arr[arr > 0]
            if tissue.size:
                pools[z].append(tissue.astype(np.uint16, copy=False))
        if i % 50 == 0:
            print(f"  {i}/{len(picked)} hasta ({time.time() - t0:.0f} s)...")

    if missing:
        print(f"UYARI: {missing} zon dosyası bulunamadı.")

    out: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
    for z in range(1, 6):
        pooled = np.concatenate(pools[z]) if pools[z] else np.zeros(1, dtype=np.uint16)
        if pooled.size > MAX_PIXELS_PER_ZONE:
            sub_rng = np.random.RandomState(SEED + z)
            pooled = pooled[sub_rng.choice(pooled.size, MAX_PIXELS_PER_ZONE, replace=False)]
        out[f"z{z}"] = pooled.astype(np.uint16)
        stats[f"z{z}"] = {
            "n_pixels": int(pooled.size),
            "min": int(pooled.min()),
            "median": float(np.median(pooled)),
            "max": int(pooled.max()),
            "p01": float(np.percentile(pooled, 1)),
            "p99": float(np.percentile(pooled, 99)),
        }
        print(f"  Zon {z}: {pooled.size} piksel | medyan {np.median(pooled):.0f} | "
              f"p01 {np.percentile(pooled, 1):.0f} | p99 {np.percentile(pooled, 99):.0f}")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)

    report = {
        "config_version": "v4_200pt_pooled_ref",
        "source_split": "train_only",
        "seed": SEED,
        "n_patients": len(picked),
        "patient_ids": picked,
        "source": "data/processed/zones/patients/<id>/zone_*.mha (doku pikselleri, v>0)",
        "max_pixels_per_zone": MAX_PIXELS_PER_ZONE,
        "per_zone_stats": stats,
        "output_npz": str(OUT_NPZ.relative_to(PROJECT_ROOT)),
        "output_npz_sha256": sha256(OUT_NPZ),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {OUT_NPZ}")
    print(f"Rapor     : {OUT_REPORT}")


if __name__ == "__main__":
    main()

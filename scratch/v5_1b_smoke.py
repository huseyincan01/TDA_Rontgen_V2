"""V5.1b duman testi: tüm-akciğer p1/p99 + filtrasyon + diyagram."""

from __future__ import annotations

import sys
from pathlib import Path

import SimpleITK as sitk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "exp"))

import V5_1b_fixed_reference as v  # noqa: E402

ZONE = v.ZONE_FILES[1]


def main() -> None:
    pid = "c0001"
    p1, p99 = v._patient_p1p99(pid)
    print(f"{pid} tüm-akciğer p1={p1:.1f} p99={p99:.1f}")

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(v.PATIENTS_DIR / pid / ZONE)))
    fil = v.build_lungref_filtration(arr, p1, p99)
    print("filtration:", None if fil is None else (fil.shape, fil.min(), fil.max(),
          int((fil > 0).sum())))

    h0, h1 = v._worker_diagram((str(v.PATIENTS_DIR / pid / ZONE), p1, p99))
    print("diyagram H0:", len(h0), "H1:", len(h1))
    if len(h1):
        print("  H1 persistence aralığı:", round(float((h1[:, 1] - h1[:, 0]).min()), 2),
              "→", round(float((h1[:, 1] - h1[:, 0]).max()), 2))

    # karşılaştırma: per-kırpıntı (a_density) ile aynı zon
    a1, a99 = np.percentile(arr[arr > 0], [1, 99])
    print(f"karşılaştırma — per-kırpıntı p1={a1:.1f} p99={a99:.1f} vs tüm-akciğer p1={p1:.1f} p99={p99:.1f}")


if __name__ == "__main__":
    main()

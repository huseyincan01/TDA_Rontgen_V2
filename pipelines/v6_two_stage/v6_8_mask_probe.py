"""V6.8 — Yarıçap kaybı maske oranı probu (neden çöküyor?).

`build_targets_heat` ısı haritası hedefini (`Y`) nodül merkezinin çevresinde **31x31**
Gaussian olarak üretir, ama yarıçap hedefini (`R`) **yalnız merkez pikseline** yazar.
V6.6'nın kaybı `Y > 0` maskesi üzerinde ortalandığı için süpervizyonun neredeyse
tamamı "0 tahmin et" der ⇒ kaybın minimumu sabit 0'dır ve kafa çöker.

Bu betik o oranı **gerçek veriyle** ölçer (eğitim yok, GPU gerekmez):

    python pipelines/v6_two_stage/v6_8_mask_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import benchmark_v6_0_phase2 as p2  # noqa: E402
from detector_models import build_targets_heat  # noqa: E402

N = 4


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    dev = torch.device("cpu")
    tot_y = tot_r = 0
    for pid in p2._nodule_pids("train")[:N]:
        _canon, cen, rad = p2._prep(pid)
        Y, R = build_targets_heat(cen, rad, dev)
        n_y, n_r = int((Y > 0).sum()), int((R > 0).sum())
        tot_y += n_y
        tot_r += n_r
        print(f"  {pid}: nodül={len(cen)} · Y>0={n_y} · R>0={n_r} · "
              f"R hedefleri={[round(float(v), 2) for v in R[R > 0]]}")
    print(f"\n  TOPLAM: Y>0={tot_y} · R>0={tot_r} ⇒ süpervizyonun "
          f"%{100 * tot_r / max(1, tot_y):.2f}'i gerçek hedef taşır; "
          f"%{100 - 100 * tot_r / max(1, tot_y):.2f}'i '0 tahmin et' der.")
    print("  ⇒ `Y>0` maskesiyle eğitilen yarıçap kafasının çökmesi beklenen sonuçtur.")


if __name__ == "__main__":
    main()

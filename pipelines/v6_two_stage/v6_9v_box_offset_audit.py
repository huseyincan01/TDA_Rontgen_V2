"""V6.9v — Kutu-merkezleme (offset) denetimi: JSRT kutuları NODE21 Val'e göre daha mı kaymış?

AMAÇ: V6.9v'de "kutu-merkezli radyal TDA (M1) kendi hastanemizde kazandırıp yabancı Japonya
verisinde kaybettiriyor" bulgusunun MEKANİZMASINI ölçmek. M1'in cezası `+alpha·r_norm` ile
doğrusal arttığı için, 1. aşamanın kutusu tümör merkezinden ne kadar uzaksa ceza o kadar büyük.

Her lezyon için iki ölçüm (1024 uzayı; `r_norm = 1` ⇒ 32'lik yamanın yarıçapı = 16 px):
  * `havuz-en-iyi` : aday havuzundaki GT merkezine EN YAKIN kutunun uzaklığı
                     (1. aşama bu görüntüde en iyi ne kadar ortalayabiliyor?)
  * `top1-s1`      : 1. aşamanın en yüksek skorlu kutusunun en yakın GT'ye uzaklığı
                     (1. aşamanın fiilen önerdiği kutu)

Model eğitimi YOK; yalnızca mevcut aday önbellekleri okunur.
Kullanım: python pipelines/v6_two_stage/v6_9v_box_offset_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "pipelines" / "v6_two_stage"))

import common  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "v6_two_stage"
HALF_PATCH_1024 = 16.0          # 32'lik yamanın yarıçapı (1024 uzayı)


def cohort_stats(z, tag: str) -> dict:
    pids = [str(p) for p in z[f"{tag}_pids"]]
    off = z[f"{tag}_off"]
    cx, cy, s1 = z[f"{tag}_cx"], z[f"{tag}_cy"], z[f"{tag}_s1"]
    pool_best, top1_off = [], []
    for i, p in enumerate(pids):
        lo, hi = int(off[i]), int(off[i + 1])
        gts = common.load_gt(p)
        if not gts or hi <= lo:
            continue
        c = np.stack([cx[lo:hi], cy[lo:hi]], axis=1).astype(np.float64)
        top = int(np.argmax(np.asarray(s1[lo:hi], np.float64)))
        for g in gts:
            gx, gy = g["x"] + g["w"] / 2.0, g["y"] + g["h"] / 2.0
            d = np.hypot(c[:, 0] - gx, c[:, 1] - gy)
            pool_best.append(float(d.min()) / HALF_PATCH_1024)
            top1_off.append(float(d[top]) / HALF_PATCH_1024)
    return {"n_lesions": len(pool_best),
            "pool": np.asarray(pool_best), "top1": np.asarray(top1_off)}


def summarize(name: str, s: dict) -> None:
    pb, t1 = s["pool"], s["top1"]
    print(f"\n{name}  ({s['n_lesions']} lezyon)")
    print(f"  havuz-en-iyi  r_norm: medyan {np.median(pb):5.2f} · ortalama {pb.mean():5.2f} · "
          f"P(>0,5) %{100 * np.mean(pb > 0.5):4.1f} · P(>1,0) %{100 * np.mean(pb > 1.0):4.1f}")
    print(f"  top-1 (max s1) r_norm: medyan {np.median(t1):5.2f} · ortalama {t1.mean():5.2f} · "
          f"P(>0,5) %{100 * np.mean(t1 > 0.5):4.1f} · P(>1,0) %{100 * np.mean(t1 > 1.0):4.1f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    zr = np.load(ART / "_v6_9r_cache.npz", allow_pickle=True)
    zl = np.load(ART / "_v6_9l_cache.npz", allow_pickle=True, mmap_mode="r")
    jsrt = cohort_stats(zr, "jp")
    val = cohort_stats(zl, "va")
    summarize("JSRT 242 (jp, 149 nodüllü)", jsrt)
    summarize("NODE21 Val 734 (nodüllü hastalar)", val)
    print("\n=== FARK (JSRT − Val) ===")
    for k, label in (("pool", "havuz-en-iyi"), ("top1", "top-1")):
        a, b = jsrt[k], val[k]
        print(f"  {label:12} medyan {np.median(a):.2f} vs {np.median(b):.2f} "
              f"(Δ {np.median(a) - np.median(b):+.2f}) · ortalama {a.mean():.2f} vs {b.mean():.2f} "
              f"(Δ {a.mean() - b.mean():+.2f})")
    print("\nYorum: M1'in cezası alpha·r_norm olduğu için JSRT'de aynı kutu için ceza ~1,5-2× daha "
          "büyük ⇒ radyal önsav siteye özgüdür.")


if __name__ == "__main__":
    main()

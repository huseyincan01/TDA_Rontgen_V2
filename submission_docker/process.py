#!/usr/bin/env python
"""NODE21 Detection Track — Grand-Challenge gönderim konteyneri (process.py).

Sözleşme
--------
Girdi : /input/images/chest-x-rays/<ad>.mha   (2B (H,W) VEYA 3B (N_z,H,W) yığın)
Çıktı : /output/nodules.json                   ("Multiple 2D bounding boxes")

3B yığın kuralı: NODE21 sunucusu birden fazla 1024x1024 röntgeni TEK bir 3B .mha
olarak verebilir. Kod her z katmanını tek tek işler ve köşelere z indeksini yazar.

Koordinat zinciri (sıfır varsayım):
  ısı haritası 256x256 → kanonik 512x512 → ORİJİNAL görüntü pikseli (x_px, y_px)
  → SimpleITK TransformContinuousIndexToPhysicalPoint([x_px, y_px, z]) → mm.

--network=none güvenliği
-----------------------
* Ağırlıklar imaja GÖMÜLÜDÜR ve yalnız yerel diskten okunur
  (/opt/xrv_weights veya ~/.torchxrayvision/models_data). Hiçbir urllib/requests
  çağrısı tetiklenmez (dosyalar mevcutsa torchxrayvision indirme yapmaz).
* Akciğer maskesi ÇALIŞMA ANINDA PSPNet ile ham röntgenden üretilir; sunucudan
  önceden hesaplanmış maske gelmez.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
XRV_CACHE_DIR = os.environ.get("XRV_CACHE_DIR", None)   # None → ~/.torchxrayvision/models_data
VERSION = "v6.7-fulldata-1.0"
REPO_ROOT = Path(__file__).resolve().parent          # submission_docker/ içinde çalışır


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _guard_downloads() -> None:
    """Hedefli koruma: torchxrayvision'ın İNDİRME fonksiyonunu devre dışı bırakır.

    Küresel socket yaması YAPILMAZ (torch/multiprocessing'i bozabilir). Bunun yerine
    ağırlık dosyaları mevcut olduğu için indirme yoluna hiç girilmez; girilirse
    burada açık hata verilir (--network=none sözleşmesi).
    """
    import torchxrayvision as xrv

    def _blocked(url=None, *a, **k):
        raise RuntimeError(
            f"AĞ İNDİRMESİ ENGELLENDİ (--network=none): {url}\n"
            "Ağırlıklar imaja gömülü olmalı (XRV_CACHE_DIR / "
            "~/.torchxrayvision/models_data).")

    xrv.utils.download = _blocked          # type: ignore[assignment]
    _log("indirme koruması AKTİF (xrv.utils.download engellendi)")


def build_pipeline():
    """Kilitli boru hattını yükler (ağırlıklar YEREL diskten)."""
    # Depo kökünü bul (submission_docker/ bir üst seviyesi ya da gömülü /opt/app)
    for cand in (REPO_ROOT.parent, Path("/opt/app"), Path.cwd()):
        if (cand / "pipelines" / "v6_two_stage" / "v6_7_submission_infer.py").exists():
            root = cand
            break
    else:
        raise RuntimeError("depo kökü bulunamadı (pipelines/v6_two_stage yok)")
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "pipelines" / "v6_two_stage"))

    import v6_7_submission_infer as v67
    cfg = json.loads(v67.CFG_PATH.read_text(encoding="utf-8"))
    _log(f"sözleşme {cfg['version']} | w={cfg['fusion']['w']} (kilitli)")
    pipe = v67.SubmissionPipeline(v67.CFG_PATH, cache_dir=XRV_CACHE_DIR)
    return pipe, cfg


def process_file(path: Path, pipe, cfg) -> list:
    """Bir .mha dosyasını (2B veya 3B yığın) işler → kutu listesi.

    Resmî baseline (`node21challenge/node21_detection_baseline/process.py`) ile
    BİREBİR uyum:
      * x, y → **mm** (piksel × spacing; NODE21'de origin=(0,0) olduğu için
        TransformContinuousIndexToPhysicalPoint ile aynı sonucu verir),
      * z → **slice indeksi** (baseline `np_prediction['slice'][i]` kullanır),
      * köşe SIRASI → **[top_right, top_left, bottom_left, bottom_right]**,
      * koordinatlar **2 ondalığa** yuvarlanır.
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)                 # (z,H,W) veya (H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    nz, H, W = arr.shape
    dim = img.GetDimension()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    _log(f"{path.name}: yığın {nz} katman, {H}x{W}, ITK boyut={dim}, "
         f"spacing={tuple(round(s, 4) for s in spacing)}, origin={origin}")
    if any(abs(o) > 1e-6 for o in origin):
        _log("  ⚠ origin ≠ 0 — resmî baseline origin'i YOK SAYAR (piksel×spacing); "
             "transform değeri ile fark olabilir, x/y transform'dan alınıyor")

    seg = None
    boxes = []
    for z in range(nz):
        img2d = arr[z].astype(np.float32)
        if seg is None:                                # PSPNet bir kez kurulur
            from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter
            seg = PSPNet5ZoneSegmenter(device=pipe.device, cache_dir=XRV_CACHE_DIR)
        dets = pipe.predict_image(img2d, seg=seg)
        _log(f"  z={z}: {len(dets)} aday")
        for d in dets:
            x_min, y_min = float(d["x1"]), float(d["y1"])
            x_max, y_max = float(d["x2"]), float(d["y2"])
            # --- köşeler: resmî baseline sırası (TR, TL, BL, BR) ---
            px = {"top_right": (x_max, y_max), "top_left": (x_min, y_max),
                  "bottom_left": (x_min, y_min), "bottom_right": (x_max, y_min)}
            corners = []
            for nm in ("top_right", "top_left", "bottom_left", "bottom_right"):
                xi, yi = px[nm]
                if dim == 2:
                    pt = img.TransformContinuousIndexToPhysicalPoint([xi, yi])
                else:
                    pt = img.TransformContinuousIndexToPhysicalPoint([xi, yi, float(z)])
                # Resmî baseline mm'yi `piksel × spacing` olarak üretir (origin'i YOK SAYAR).
                # Transform'u kullanıp origin'i çıkararak baseline ile BİREBİR eşitliyoruz.
                x_mm = float(pt[0]) - float(origin[0])
                y_mm = float(pt[1]) - float(origin[1])
                # z = SLICE İNDEKSİ (baseline `np_prediction['slice']` sözleşmesi)
                corners.append([round(x_mm, 2), round(y_mm, 2), float(z)])
            boxes.append({"corners": corners,
                          "probability": round(float(d["score"]), 4)})
    boxes.sort(key=lambda b: -b["probability"])
    return boxes


def main() -> None:
    _log(f"NODE21 gönderim konteyneri {VERSION}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ağırlık önbelleğini doğrula (indirme TETİKLENMEMELİ) — önce kontrol, sonra koruma
    import torchxrayvision as xrv
    cache = XRV_CACHE_DIR or xrv.utils.get_cache_dir()
    need = ["pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt",
            "pspnet_chestxray_best_model_4.pth"]
    for n in need:
        p = Path(os.path.expanduser(cache)) / n
        if not p.exists():
            raise RuntimeError(f"GÖMÜLÜ AĞIRLIK YOK: {p} (--network=none altında indirilemez)")
        _log(f"  ✓ ağırlık: {n} ({p.stat().st_size / 1e6:.1f} MB)")
    _guard_downloads()

    pipe, cfg = build_pipeline()

    # --- ESNEK GİRDİ YOLU ---
    # Resmî baseline `/input/<uuid>.mha` bekler; yeni GC arayüzleri
    # `/input/images/chest-x-rays/<uuid>.mha` kullanır. RECURSIVE tarama ikisini de bulur.
    files = [f for f in sorted(INPUT_DIR.rglob("*.mha")) if f.is_file()]
    if not files:
        exts = (".mha", ".mhd", ".nrrd", ".nii", ".nii.gz")
        files = [f for f in sorted(INPUT_DIR.rglob("*"))
                 if f.is_file() and f.name.lower().endswith(exts)]
    _log(f"{len(files)} girdi dosyası (kök: {INPUT_DIR}, özyinelemeli tarama)")
    for f in files:
        _log(f"  girdi: {f}")

    all_boxes = []
    for f in files:
        try:
            all_boxes.extend(process_file(f, pipe, cfg))
        except Exception as exc:                       # tek dosya hatası tümünü düşürmesin
            _log(f"  !! {f.name}: {type(exc).__name__}: {exc}")

    # --- ÇIKTI: resmî baseline ile BİREBİR (name alanı YOK) ---
    payload = {
        "type": "Multiple 2D bounding boxes",
        "boxes": all_boxes,
        "version": {"major": 1, "minor": 0},
    }
    out = OUTPUT_DIR / "nodules.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    _log(f"YAZILDI {out} — {len(all_boxes)} kutu")


if __name__ == "__main__":
    main()

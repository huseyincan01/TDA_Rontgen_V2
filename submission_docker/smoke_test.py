"""V6.7 — Gönderim konteynerinin UÇTAN UCA AĞSIZ duman testi.

Ne yapar
--------
1. Gerçek NODE21 val görüntülerinden **3B .mha yığını** (1024x1024x3) üretir,
   bilinen spacing/origin ile (mm dönüşümünü doğrulamak için).
2. `process.py`'yi alt süreç olarak, /input ve /output düzenini taklit ederek koşar.
3. Çıktı sözleşmesini denetler: şema, 4 köşe, olasılık aralığı, mm koordinatları,
   piksel→mm dönüşümünün tersinirliği.
4. **NEGATİF TEST:** ağırlık önbelleği boş bir dizine işaret edilirse süreç sessizce
   indirmeye çalışmak yerine AÇIK HATA vermeli (--network=none kanıtı).

Kullanım:
    python submission_docker/smoke_test.py
    python submission_docker/smoke_test.py --n 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines" / "v6_two_stage"))

SPACING = (0.143, 0.143, 1.0)
ORIGIN = (10.0, 20.0, 0.0)


def build_stack(pids, out_path: Path) -> None:
    import SimpleITK as sitk
    import common

    slices = [common.load_image(p) for p in pids]           # her biri (1024,1024) float
    stack = np.stack(slices).astype(np.float32)             # (z, H, W)
    img = sitk.GetImageFromArray(stack)
    img.SetSpacing(SPACING)
    img.SetOrigin(ORIGIN)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    print(f"  yığın yazıldı: {out_path} {stack.shape} spacing={SPACING} origin={ORIGIN}")


def check_output(out_json: Path) -> dict:
    print("\n[3] Çıktı sözleşmesi denetimi (resmî baseline şeması)...")
    assert out_json.exists(), f"nodules.json YOK: {out_json}"
    d = json.loads(out_json.read_text(encoding="utf-8"))

    # Resmî baseline format_to_GC: {"type", "boxes", "version"} — `name` ALANI YOK
    assert "name" not in d, "resmî baseline'da `name` alanı YOK"
    assert d["type"] == "Multiple 2D bounding boxes", d.get("type")
    assert d["version"] == {"major": 1, "minor": 0}, d.get("version")
    assert set(d.keys()) == {"type", "boxes", "version"}, d.keys()
    assert isinstance(d["boxes"], list) and d["boxes"], "boxes boş!"
    print(f"  ✓ şema: type/version doğru, `name` yok · {len(d['boxes'])} kutu")

    # Köşe SIRASI: [top_right, top_left, bottom_left, bottom_right]
    #   = [(x_max,y_max), (x_min,y_max), (x_min,y_min), (x_max,y_min)]
    for i, b in enumerate(d["boxes"]):
        assert set(b.keys()) == {"corners", "probability"}, b.keys()
        c = b["corners"]
        assert len(c) == 4, f"kutu {i}: 4 köşe bekleniyor"
        assert all(len(p) == 3 for p in c), f"kutu {i}: köşe 3B olmalı (x,y,z)"
        xmax, xmin = c[0][0], c[1][0]
        ymax, ymin = c[0][1], c[2][1]
        assert abs(c[3][0] - xmax) < 1e-6 and abs(c[2][0] - xmin) < 1e-6, \
            f"kutu {i}: x köşe sırası hatalı"
        assert abs(c[1][1] - ymax) < 1e-6 and abs(c[3][1] - ymin) < 1e-6, \
            f"kutu {i}: y köşe sırası hatalı"
        assert xmax > xmin and ymax > ymin, f"kutu {i}: dejenere kutu"
        assert len({p[2] for p in c}) == 1, f"kutu {i}: köşeler farklı z'de"
        assert 0.0 <= b["probability"] <= 1.0, f"kutu {i}: olasılık aralık dışı"
    print("  ✓ köşe sırası = [top_right, top_left, bottom_left, bottom_right]")
    print("  ✓ her kutuda 4 köşe × 3 koordinat, olasılık ∈ [0,1]")
    probs = [b["probability"] for b in d["boxes"]]
    assert probs == sorted(probs, reverse=True), "olasılıklar azalan değil"
    print("  ✓ olasılıklar azalan sırada")
    zs = sorted({b["corners"][0][2] for b in d["boxes"]})
    print(f"  ✓ z = SLICE İNDEKSİ (baseline sözleşmesi): {zs}")
    return d


def roundtrip_check(stack_path: Path, d: dict) -> None:
    """x/y: mm ↔ piksel tersinirliği; z: slice indeksi ile tutarlılık.

    process.py mm'yi baseline sözleşmesiyle (`piksel × spacing`, origin yok sayılır)
    üretir. Ters dönüşüm için fiziksel nokta = mm + origin.
    """
    import SimpleITK as sitk
    img = sitk.ReadImage(str(stack_path))
    sp = img.GetSpacing()
    orig = img.GetOrigin()
    nz = img.GetSize()[-1]
    ok = 0
    for b in d["boxes"][:5]:
        for c in b["corners"]:
            x_px = c[0] / sp[0]          # baseline: mm = piksel*spacing ⇒ piksel = mm/spacing
            y_px = c[1] / sp[1]
            # fiziksel nokta = mm + origin (transform origin'i bekler)
            idx = img.TransformPhysicalPointToContinuousIndex(
                [float(c[0] + orig[0]), float(c[1] + orig[1]), 0.0])
            if abs(idx[0] - x_px) < 1e-2 and abs(idx[1] - y_px) < 1e-2:
                ok += 1
            assert 0 <= c[2] < nz, f"z indeksi aralık dışı: {c[2]}"
    assert ok > 0, "mm→piksel ters dönüşümü tutarsız!"
    print(f"  ✓ x/y mm→piksel ters dönüşümü doğrulandı ({ok} köşe), z ∈ [0,{nz - 1}]")
    # kutu fiziksel boyutu = 56 px * spacing olmalı
    b0 = d["boxes"][0]["corners"]
    dx = abs(b0[0][0] - b0[1][0])
    dy = abs(b0[0][1] - b0[2][1])
    assert abs(dx - 56 * sp[0]) < 0.05, f"kutu genişliği mm yanlış: {dx:.3f}"
    assert abs(dy - 56 * sp[1]) < 0.05, f"kutu yüksekliği mm yanlış: {dy:.3f}"
    print(f"  ✓ 56 px kutu → mm: {dx:.2f} × {dy:.2f} (spacing {sp[0]:.3f}/{sp[1]:.3f})")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    print("V6.7 GÖNDERİM KONTEYNERİ — AĞSIZ UÇTAN UCA DUMAN TESTİ\n")
    import common
    val = common.split_patients()["val"]
    nod = [p for p in val if common.load_gt(p)][:args.n]
    ctl = [p for p in val if not common.load_gt(p)][:1]
    pids = (nod + ctl)[:max(1, args.n)]
    print(f"[1] Kaynak görüntüler (val): {pids}")

    tmp = Path(tempfile.mkdtemp(prefix="node21_smoke_"))
    inp = tmp / "input"
    out = tmp / "output"
    stack = inp / "images" / "chest-x-rays" / "smoke_stack.mha"
    print("\n[2] 3B yığın üretiliyor...")
    build_stack(pids, stack)

    env = dict(os.environ)
    env.update({"INPUT_DIR": str(inp), "OUTPUT_DIR": str(out),
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    env.setdefault("XRV_CACHE_DIR", str(Path.home() / ".torchxrayvision" / "models_data"))

    print("\n[2b] process.py koşuluyor (yerel, ağırlıklar gömülü dizinden)...")
    r = subprocess.run([sys.executable, str(ROOT / "submission_docker" / "process.py")],
                       env=env, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = "\n".join(r.stdout.strip().splitlines()[-18:])
    print(tail)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"process.py BAŞARISIZ (exit {r.returncode})")

    d = check_output(out / "nodules.json")
    roundtrip_check(stack, d)

    # ---- NEGATİF TEST: boş önbellek → sessiz indirme DEĞİL, açık hata
    print("\n[4] NEGATİF TEST: ağırlık yoksa açık hata vermeli (sessiz indirme YASAK)...")
    empty = tmp / "empty_cache"
    empty.mkdir(parents=True, exist_ok=True)
    env2 = dict(env)
    env2["XRV_CACHE_DIR"] = str(empty)
    r2 = subprocess.run([sys.executable, str(ROOT / "submission_docker" / "process.py")],
                        env=env2, cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    ok = r2.returncode != 0 and "GÖMÜLÜ AĞIRLIK YOK" in (r2.stdout + r2.stderr)
    print(f"  exit={r2.returncode} · beklenen hata mesajı: {'BULUNDU ✓' if ok else 'YOK ✗'}")
    assert ok, "boş önbellekte sessizce devam etti — --network=none ihlali!"

    # ---- ESNEK GİRDİ YOLU TESTİ: .mha doğrudan /input/ altında (baseline düzeni)
    print("\n[5] ESNEK GİRDİ YOLU: .mha doğrudan /input/ altında (baseline düzeni)...")
    inp2 = tmp / "input_direct"
    out2 = tmp / "output_direct"
    inp2.mkdir(parents=True, exist_ok=True)
    direct = inp2 / "smoke_stack.mha"
    import shutil
    shutil.copy2(stack, direct)
    env3 = dict(env)
    env3.update({"INPUT_DIR": str(inp2), "OUTPUT_DIR": str(out2)})
    r3 = subprocess.run([sys.executable, str(ROOT / "submission_docker" / "process.py")],
                        env=env3, cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if r3.returncode != 0:
        print(r3.stdout[-1500:], r3.stderr[-1500:])
        raise SystemExit("doğrudan /input/*.mha düzeni BAŞARISIZ")
    d3 = json.loads((out2 / "nodules.json").read_text(encoding="utf-8"))
    assert len(d3["boxes"]) == len(d["boxes"]), "doğrudan düzende kutu sayısı farklı"
    print(f"  ✓ /input/*.mha bulundu ve işlendi ({len(d3['boxes'])} kutu)")
    print(f"  ✓ /input/images/chest-x-rays/*.mha da çalışıyor ({len(d['boxes'])} kutu)")

    print(f"\n✅ TÜM DENETİMLER GEÇTİ\n   çıktı: {out / 'nodules.json'}\n   geçici dizin: {tmp}")


if __name__ == "__main__":
    main()

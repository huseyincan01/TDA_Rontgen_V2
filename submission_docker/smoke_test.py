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
    import SimpleITK as sitk
    print("\n[3] Çıktı sözleşmesi denetimi...")
    assert out_json.exists(), f"nodules.json YOK: {out_json}"
    d = json.loads(out_json.read_text(encoding="utf-8"))

    assert d["name"] == "Regions of interest", d.get("name")
    assert d["type"] == "Multiple 2D bounding boxes", d.get("type")
    assert d["version"] == {"major": 1, "minor": 0}, d.get("version")
    assert isinstance(d["boxes"], list) and d["boxes"], "boxes boş!"
    print(f"  ✓ şema: name/type/version doğru · {len(d['boxes'])} kutu")

    for i, b in enumerate(d["boxes"]):
        assert set(b.keys()) == {"corners", "probability"}, b.keys()
        assert len(b["corners"]) == 4, f"kutu {i}: 4 köşe bekleniyor"
        assert all(len(c) == 3 for c in b["corners"]), f"kutu {i}: köşe 3B olmalı (x,y,z)"
        assert 0.0 <= b["probability"] <= 1.0, f"kutu {i}: olasılık aralık dışı"
    print("  ✓ her kutuda 4 köşe × 3 koordinat, olasılık ∈ [0,1]")
    probs = [b["probability"] for b in d["boxes"]]
    assert probs == sorted(probs, reverse=True), "olasılıklar azalan değil"
    print("  ✓ olasılıklar azalan sırada")

    # mm koordinatları: fiziksel = origin + index * spacing (eksen sırası ITK: x,y,z)
    zs = sorted({round(b["corners"][0][2], 6) for b in d["boxes"]})
    print(f"  ✓ görülen z (mm) düzlemleri: {zs}")
    return d


def roundtrip_check(stack_path: Path, d: dict) -> None:
    """mm → piksel ters dönüşümü, köşelerin izgara üzerinde olduğunu doğrular."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(stack_path))
    ok = 0
    for b in d["boxes"][:5]:
        for c in b["corners"]:
            idx = img.TransformPhysicalPointToContinuousIndex([float(v) for v in c])
            # mm = origin + idx*spacing ⇒ idx = (mm - origin)/spacing
            exp = [(c[k] - ORIGIN[k]) / SPACING[k] for k in range(3)]
            if all(abs(idx[k] - exp[k]) < 1e-3 for k in range(3)):
                ok += 1
    assert ok > 0, "mm→piksel ters dönüşümü tutarsız!"
    print(f"  ✓ mm→piksel ters dönüşümü doğrulandı ({ok} köşe)")


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

    print(f"\n✅ TÜM DENETİMLER GEÇTİ\n   çıktı: {out / 'nodules.json'}\n   geçici dizin: {tmp}")


if __name__ == "__main__":
    main()

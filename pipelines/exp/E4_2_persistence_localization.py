"""E4.2 — Persistence özelliklerinin görüntü bölgelerine geri izlenmesi (yorumlanabilirlik).

Ön-kayıtlı hipotez:
    En yüksek kalıcılıklı H1 (ve H0) sınıflarının doğum/ölüm hücreleri, zon
    görüntüsünde **yüksek gradyanlı (kenar) bölgelere** düşer — yani kaburga
    kenarları ve bronkovasküler yapılar gibi arka plan anatomisine karşılık gelir,
    nodül içi dokuya değil.

Yöntem:
    GUDHI `CubicalComplex.cofaces_of_persistence_pairs()` ile her sınıfın
    doğum/ölüm hücresi görüntü koordinatına çevrilir (literature.md §1.5, yerel
    olarak doğrulanmıştı). Bu koordinatlardaki Sobel gradyan büyüklüğü, doku
    ortalamasına oranlanır. Rastgele doku pikselleri kontrol grubudur.

Başarı ölçütü:
    H1 için top-K sınıf gradyan oranı > 1.5 VE rastgele kontrol oranı ~1.0.

Kullanım:
    python pipelines/exp/E4_2_persistence_localization.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gudhi
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import sobel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from node21_tda.experiment import new_run_dir, write_manifest  # noqa: E402
from node21_tda.features.extractor import TDASettings, ZoneTDAExtractor  # noqa: E402

EXPERIMENT_ID = "E4.2"
SLUG = "persistence_localization"
SEED = 42
TOP_K = 10
ZONE_FILES = {
    1: "zone_1_sag_ust.mha",
    2: "zone_2_sag_orta.mha",
    3: "zone_3_sag_alt.mha",
    4: "zone_4_sol_ust.mha",
    5: "zone_5_sol_alt.mha",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4.2 persistence -> goruntu konumu analizi")
    p.add_argument("--splits-path", type=Path,
                   default=PROJECT_ROOT / "data/processed/splits_v2/zone_splits.csv")
    p.add_argument("--zones-dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/zones/patients")
    p.add_argument("--n-per-class", type=int, default=30, help="Pozitif ve negatif zon sayisi")
    return p.parse_args()


def crop_and_filter(image: np.ndarray, settings: TDASettings):
    """`ZoneTDAExtractor.prepare_filtration` ile birebir ayni crop + filtrasyon."""
    valid = np.isfinite(image) & (image > 0)
    if not np.any(valid):
        return None
    ys, xs = np.where(valid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tissue = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    tissue_mask = valid[y0:y1, x0:x1]
    scaled = np.clip(np.rint(tissue * 255.0 / settings.intensity_max), 0, 255).astype(np.uint8)
    filtration = np.zeros_like(scaled, dtype=np.uint8)
    filtration[tissue_mask] = 255 - scaled[tissue_mask]
    return tissue, tissue_mask, filtration


def gradient_magnitude(tissue: np.ndarray) -> np.ndarray:
    gx = sobel(tissue, axis=1, mode="nearest")
    gy = sobel(tissue, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def build_complex(filtration: np.ndarray):
    """Kubik kompleksi kurar ve persistence'i hesaplar (zon basina BIR kez)."""
    cc = gudhi.CubicalComplex(
        dimensions=filtration.shape,
        top_dimensional_cells=filtration.astype(np.float64, copy=False).ravel(),
    )
    cc.compute_persistence()
    return cc


def top_class_gradient(cc, filtration, tissue_mask, grad, dim, top_k):
    """Belirtilen homoloji boyutunda en kalicilikli top_k sinifin gradyan istatistigi."""
    intervals = cc.persistence_intervals_in_dimension(dim)
    if len(intervals) == 0:
        return None
    finite_mask = np.isfinite(intervals[:, 1])
    finite = intervals[finite_mask]
    pairs = np.asarray(cc.cofaces_of_persistence_pairs()[0][dim])
    if len(pairs) != len(finite):
        return None

    order = np.argsort(-(finite[:, 1] - finite[:, 0]))[:top_k]
    shape = filtration.shape
    locs, pers = [], []
    for i in order:
        cells = pairs[i].ravel()
        vals = filtration.ravel()[cells]
        # sublevel filtrasyonu: dogum = kucuk deger, olum = buyuk deger
        birth_cell = cells[int(np.argmin(vals))]
        death_cell = cells[int(np.argmax(vals))]
        rc = [np.unravel_index(int(c), shape) for c in (birth_cell, death_cell)]
        locs.extend(rc)
        pers.append(float(finite[i, 1] - finite[i, 0]))

    g_vals = [float(grad[r, c]) for r, c in locs]
    tissue_grad = float(grad[tissue_mask].mean())
    return {
        "persistence": pers,
        "grad_at_top_classes": float(np.mean(g_vals)) if g_vals else 0.0,
        "grad_tissue_mean": tissue_grad,
        "ratio": float(np.mean(g_vals) / tissue_grad) if g_vals and tissue_grad > 0 else 0.0,
    }


def random_control(tissue_mask, grad, top_k, rng):
    ys, xs = np.where(tissue_mask)
    pick = rng.randint(0, len(ys), size=min(2 * top_k, len(ys)))
    g_vals = [float(grad[ys[i], xs[i]]) for i in pick]
    tissue_grad = float(grad[tissue_mask].mean())
    return {"grad_at_random": float(np.mean(g_vals)), "grad_tissue_mean": tissue_grad,
            "ratio": float(np.mean(g_vals) / tissue_grad) if tissue_grad > 0 else 0.0}


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(SEED)
    settings = TDASettings()
    extractor = ZoneTDAExtractor(settings)

    splits = pd.read_csv(args.splits_path)
    train = splits[splits["is_train_balanced_sample"]].copy()
    pos = train[train["zone_label"] == 1].sample(n=args.n_per_class, random_state=SEED)
    neg = train[train["zone_label"] == 0].sample(n=args.n_per_class, random_state=SEED)
    sample = pd.concat([pos, neg], ignore_index=True)

    print("=" * 78)
    print("E4.2 — PERSISTENCE OZELLIKLERININ GORUNTUYE GERI IZLENMESI")
    print("=" * 78)
    print(f"Ornek: {len(pos)} pozitif + {len(neg)} negatif zon | top-K = {TOP_K}")

    records = []
    for _, row in sample.iterrows():
        pid, zid = row["patient_id"], int(row["zone_id"])
        path = args.zones_dir / pid / ZONE_FILES[zid]
        if not path.exists():
            continue
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if image.ndim == 3:
            image = image[0]

        prepared = crop_and_filter(image, settings)
        if prepared is None:
            continue
        tissue, tissue_mask, filtration = prepared

        # drift korumasi: kendi crop'umuz extractor ile birebir ayni mi?
        ref = extractor.prepare_filtration(image)
        if ref is None or ref.shape != filtration.shape or not np.array_equal(ref, filtration):
            print(f"  UYARI: filtrasyon uyusmazligi ({pid} zon{zid}), atlandi")
            continue

        grad = gradient_magnitude(tissue)
        cc = build_complex(filtration)
        for dim, dname in [(0, "H0"), (1, "H1")]:
            res = top_class_gradient(cc, filtration, tissue_mask, grad, dim, TOP_K)
            ctrl = random_control(tissue_mask, grad, TOP_K, rng)
            if res is None:
                continue
            records.append({
                "patient_id": pid, "zone_id": zid, "patient_label": int(row["patient_label"]),
                "zone_label": int(row["zone_label"]), "dim": dname,
                "ratio_topk": res["ratio"], "ratio_random": ctrl["ratio"],
                "max_persistence": max(res["persistence"]) if res["persistence"] else 0.0,
            })

    df = pd.DataFrame(records)
    summary = {}
    print("\n--- Sonuclar (gradyan orani: top-K sinif / doku ortalamasi) ---")
    for dname in ("H0", "H1"):
        sub = df[df["dim"] == dname]
        if sub.empty:
            continue
        summary[dname] = {
            "n": int(len(sub)),
            "mean_ratio_topk": float(sub["ratio_topk"].mean()),
            "median_ratio_topk": float(sub["ratio_topk"].median()),
            "mean_ratio_random": float(sub["ratio_random"].mean()),
            "mean_max_persistence": float(sub["max_persistence"].mean()),
            "frac_ratio_gt_1p5": float((sub["ratio_topk"] > 1.5).mean()),
        }
        s = summary[dname]
        print(f"  {dname}: n={s['n']:3d} | top-K oran ort={s['mean_ratio_topk']:.3f} "
              f"(medyan {s['median_ratio_topk']:.3f}) | rastgele kontrol={s['mean_ratio_random']:.3f} "
              f"| oran>1.5 olan zon orani=%{100*s['frac_ratio_gt_1p5']:.0f}")

    h1 = summary.get("H1", {})
    passed = h1.get("mean_ratio_topk", 0) > 1.5 and abs(h1.get("mean_ratio_random", 1) - 1.0) < 0.25
    decision = "KEEP" if passed else "KILL"
    print(f"\n  Karar kriteri: H1 oran>1.5 -> {h1.get('mean_ratio_topk',0):.3f} | "
          f"kontrol ~1.0 -> {h1.get('mean_ratio_random',0):.3f}")
    print(f"  KARAR: {decision}")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "En kalicilikli H1/H0 siniflari yuksek gradyanli (kenar/arka plan) bolgelere duser.",
        "top_k": TOP_K,
        "n_positive": len(pos), "n_negative": len(neg),
        "summary": summary,
        "criteria": {"h1_ratio_gt_1.5": bool(h1.get("mean_ratio_topk", 0) > 1.5),
                     "control_near_1": bool(abs(h1.get("mean_ratio_random", 1) - 1.0) < 0.25)},
        "decision": decision,
    }

    run_dir = new_run_dir(EXPERIMENT_ID, SLUG)
    (run_dir / "val_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(run_dir / "per_zone_records.csv", index=False)
    manifest = write_manifest(
        run_dir, EXPERIMENT_ID, inputs=[args.splits_path], seed=SEED, decision=decision,
        metrics={f"{k}_mean_ratio_topk": v["mean_ratio_topk"] for k, v in summary.items()},
        notes="GUDHI cofaces_of_persistence_pairs ile persistence noktalari goruntu koordinatina geri izlendi.",
    )
    print(f"\nRapor    : {run_dir / 'val_report.json'}")
    print(f"Manifest : {manifest}")


if __name__ == "__main__":
    main()

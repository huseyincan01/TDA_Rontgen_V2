"""V6.0 varyant taramasını ana benchmark JSON'una işle + önerilen konfigürasyonu yaz."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "artifacts" / "v6_two_stage" / "v6_0_candidate_benchmark.json"
SWEEP = ROOT / "artifacts" / "v6_two_stage" / "v6_0_variant_sweep.json"


def main() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    v = sweep["variants"]
    ks = ["10", "20", "30", "50"]

    best = {k: max(v, key=lambda m: v[m][k]) for k in ks}
    bench["phase2_variants"] = {
        "split": "val",
        "n_lesions": sweep["n_lesions"],
        "variants": v,
        "best_per_K": {k: {"variant": best[k], "recall": v[best[k]][k]} for k in ks},
        "gate_98_at_K20_K30": {
            "K20": {m: bool(v[m]["20"] >= 0.98) for m in v},
            "K30": {m: bool(v[m]["30"] >= 0.98) for m in v},
            "any_K20": any(v[m]["20"] >= 0.98 for m in v),
            "any_K30": any(v[m]["30"] >= 0.98 for m in v),
        },
        "recommended": {
            "K<=30": "M5_heatTTA (tek model + ısı haritası düzeyinde yatay TTA)",
            "K>=30": "M5+M6+TTA (iki eğitimli model, RRF füzyonu)",
            "note": "Eğitimli+geometrik (M7/M2) karışımı **daha kötü**: iki öğrenilmiş model "
                    "birbirini geometrik filtreden daha iyi tamamlıyor.",
        },
    }
    BENCH.write_text(json.dumps(bench, indent=2, ensure_ascii=False), encoding="utf-8")
    print("phase2_variants işlendi.")
    for k in ks:
        print(f"  K{k}: en iyi {best[k]:18s} {v[best[k]][k]:.4f}")
    g = bench["phase2_variants"]["gate_98_at_K20_K30"]
    print(f"  %98 kapısı: K20={g['any_K20']} K30={g['any_K30']}")


if __name__ == "__main__":
    main()

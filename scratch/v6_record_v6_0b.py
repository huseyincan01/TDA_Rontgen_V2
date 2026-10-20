"""V6.0b (dilasyon 25 + bipolar + 20 epoch) sonuçlarını ana JSON'a kaydet + V6.0 ile karşılaştır."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/v6_two_stage"
BENCH = A / "v6_0_candidate_benchmark.json"
SWEEP = A / "v6_0_variant_sweep.json"
AUTOPSY = A / "v6_0_miss_autopsy.json"

V60 = {  # V6.0 (dil15, mono, 15ep) — önceki koşunun en iyi değerleri
    "M5_heatTTA": {"10": 0.9177, "20": 0.9654, "30": 0.9740, "50": 0.9784},
    "M5+M6+TTA": {"10": 0.8398, "20": 0.9481, "30": 0.9784, "50": 0.9827},
}


def main() -> None:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))["variants"]
    autopsy = json.loads(AUTOPSY.read_text(encoding="utf-8"))

    ks = ["10", "20", "30", "50"]
    best = {k: max(sweep, key=lambda m: sweep[m][k]) for k in ks}

    bench["phase2b"] = {
        "config": {"dilation_px": 25, "input": "bipolar [canon, 255-canon, canon]",
                   "epochs": 20, "backbone": "ResNet18 stride-2", "input_size": 512,
                   "training": "1:1 nodul + temiz kontrol"},
        "split": "val",
        "n_lesions": 231,
        "variants": sweep,
        "best_per_K": {k: {"variant": best[k], "recall": sweep[best[k]][k]} for k in ks},
        "miss_autopsy_summary": autopsy["miss_autopsy"],
        "criteria2": {cfg: autopsy["summary"][cfg]["recall_k2_containment"]
                      for cfg in autopsy["summary"]},
        "subgroups_K20": {cfg: autopsy["summary"][cfg]["subgroup_k1_K20"]
                          for cfg in autopsy["summary"]},
    }
    bench["v6_0_vs_v6_0b"] = {
        "note": "V6.0 (dil15/mono/15ep) vs V6.0b (dil25/bipolar/20ep) — val Kriter1",
        "V6.0": V60,
        "V6.0b": {m: sweep[m] for m in ("M5", "M5_heatTTA", "M5+M6+TTA") if m in sweep},
        "delta_best_per_K": {
            k: {"V6.0": max(V60[m][k] for m in V60),
                "V6.0b": sweep[best[k]][k],
                "delta": round(sweep[best[k]][k] - max(V60[m][k] for m in V60), 4)} for k in ks},
    }
    BENCH.write_text(json.dumps(bench, indent=2, ensure_ascii=False), encoding="utf-8")

    print("V6.0 vs V6.0b (val, Kriter1):")
    for k in ks:
        a = max(V60[m][k] for m in V60)
        b = sweep[best[k]][k]
        print(f"  K{k}: V6.0 {a:.4f} → V6.0b {b:.4f}  ({b - a:+.4f})  [V6.0b en iyi: {best[k]}]")
    ma = autopsy["miss_autopsy"]
    print(f"  Kaçan nodül K50: 5 → {ma['n_missed_at_K50']} | Q3: {ma['Q3_tepe_sirasi']}")
    print("  phase2b + v6_0_vs_v6_0b kaydedildi.")


if __name__ == "__main__":
    main()

"""Kaçan nodül otopsi raporunu okunur biçimde yazdır."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "artifacts/v6_two_stage/v6_0_miss_autopsy.json").read_text(encoding="utf-8"))

print("=== KACAN NODULLER (M5_heatTTA @K50) ===")
for m in R["miss_autopsy"]["misses"]:
    print(f"  {m['patient']:6s} merkez={m['center']} boyut={m['size_px']}px({m['size_mm']}mm) "
          f"kontrast={m['contrast']:+.1f} raw_lung={m['in_raw_lung']} dilated={m['in_dilated_lung']} "
          f"en_iyi_rank={m['best_hit_rank']}")

print("\n=== KRITER 2 (kapsama) + ALT GRUP @K20 ===")
for cfg, v in R["summary"].items():
    print(f"  {cfg}: K2@K20={v['recall_k2_containment']['20']} K2@K30={v['recall_k2_containment']['30']} "
          f"K2@K50={v['recall_k2_containment']['50']}")
    print(f"     altK20 K1={v['subgroup_k1_K20']}  K2={v['subgroup_k2_K20']}")

print("\n=== OZET ===")
print(json.dumps(R["miss_autopsy"]["Q1_mask_kaynakli"], ensure_ascii=False))
print(json.dumps(R["miss_autopsy"]["Q2_boyut_kontrast"], ensure_ascii=False))
print(json.dumps(R["miss_autopsy"]["Q3_tepe_sirasi"], ensure_ascii=False))

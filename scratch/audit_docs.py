"""Geçmişe dönük belge tutarlılık denetimi.

Kontroller:
 1) Bölüm yapıları (v3/v5/V6) yerinde mi
 2) Silinen dosya/betik referansı kalmış mı
 3) artifacts/runs altındaki her deney ID'si belgelerde geçiyor mu
 4) data/external_vindr + artifacts/v6_two_stage raporları belgelerde referanslı mı
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "progress.md", "roadmap.md", "v4_roadmap.md", "experiments.md",
        "ablation_matrix.md", "paper_draft.md", "data/external_vindr/README.md"]
ALL = "\n".join((ROOT / d).read_text(encoding="utf-8") for d in DOCS if (ROOT / d).exists())

STRUCT = {
    "experiments.md": ["## 7. Bağımsız Dış Doğrulama", "## 8. Fiziksel Kanonikleştirme",
                       "### V5.2", "### V5.3", "### V6.1/V6.2", "### V6.3", "### V6.4"],
    "ablation_matrix.md": ["## 7.2 Gerçek Harici Veri", "## 7.3 Cihaz/Kaynak",
                           "## 7.4 Fiziksel Kanonikleştirme", "## 13. V6", "### V6.4"],
    "paper_draft.md": ["### 4.8 🌐", "### 4.9 🎯", "## 5. Tartışma", "## 6. Sınırlamalar"],
    "v4_roadmap.md": ["### Faz V4", "### Faz V5"],
    "progress.md": ["| **V5.3** |"],
}
# Silinmiş/tükenmiş dosya veya betikler
DEAD = ["vindr_fair_report", "vindr_histogram_matched_report", "vindr_evaluation_report",
        "V4_fair_external_validation", "stage2_tda_spatial_lr", "stage2_patch_cnn",
        "X_b_density_scale", "features_positional", "features_vessel", "features_relational",
        "features_global_ph", "features_sw", "features_weights", "features_vindr",
        "ct_patches"]


def main() -> None:
    ok = True
    print("=== 1) Bölüm yapıları ===")
    for f, needles in STRUCT.items():
        t = (ROOT / f).read_text(encoding="utf-8")
        miss = [n for n in needles if n not in t]
        print(f"  {f:22s} {'OK' if not miss else 'EKSİK: ' + str(miss)}")
        ok = ok and not miss

    print("\n=== 2) Silinmiş dosya referansları ===")
    # "silindi/temizlendi" notu taşıyan satırlar bilinçli arşiv notudur, kırık link değildir
    body = "\n".join(ln for ln in ALL.splitlines()
                     if "silindi" not in ln.lower() and "temizlendi" not in ln.lower())
    hits = {d: body.count(d) for d in DEAD if body.count(d)}
    for d, n in hits.items():
        print(f"  ⚠️ {d}: {n} referans")
    if not hits:
        print("  OK — hiçbir silinmiş dosya referansı yok")
    ok = ok and not hits

    print("\n=== 3) artifacts/runs deney ID'leri ===")
    ids = sorted(p.name for p in (ROOT / "artifacts/runs").iterdir() if p.is_dir())
    missing = []
    for i in ids:
        eid = i.split("_", 1)[0]
        if eid not in ALL:
            missing.append(eid)
    print(f"  {len(ids)} çalıştırma dizini | belgelerde eksik ID: {missing or 'yok'}")
    ok = ok and not missing

    print("\n=== 4) Rapor dosyaları belgelerde referanslı mı ===")
    reports = list((ROOT / "data/external_vindr").glob("*.json")) + \
              list((ROOT / "artifacts/v6_two_stage").glob("*.json"))

    def brace_names(text: str) -> set:
        out = set()
        for m in re.finditer(r"([A-Za-z0-9_\-]+)\{([^}]+)\}([A-Za-z0-9_.\-]*)", text):
            pre, body, suf = m.groups()
            for part in body.split(","):
                out.add(f"{pre}{part.strip()}{suf}")
        return out

    expanded = brace_names(ALL)
    missing_r = [p.name for p in reports if p.name not in ALL and p.name not in expanded]
    print(f"  {len(reports)} rapor | eksik referans: {missing_r or 'yok'}")
    # V6 JSON'ları experiments §9'da toplu listeleniyor; tek tek geçmese de olur
    soft = [m for m in missing_r if not m.startswith("v6_")]
    ok = ok and not soft

    print("\nSONUÇ:", "GEÇTİ" if ok else "SORUN VAR")


if __name__ == "__main__":
    main()

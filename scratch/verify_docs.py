"""Belge yapılarının bütünlüğünü doğrula (V5.3 dahil)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    "experiments.md": ["## 7. Bağımsız Dış Doğrulama", "## 8. Fiziksel Kanonikleştirme",
                       "### V5.2", "### V5.3"],
    "ablation_matrix.md": ["## 7.2 Gerçek Harici Veri", "## 7.3 Cihaz/Kaynak",
                           "## 7.4 Fiziksel Kanonikleştirme", "### V5.2", "### V5.3"],
    "v4_roadmap.md": ["### Faz V4", "### Faz V5", "| 2026-10-17 | **A2**", "| 2026-10-17 | **V4**",
                      "| 2026-10-17 | **V5.1**", "| 2026-10-17 | **V5.2**", "| 2026-10-17 | **V5.3**"],
    "progress.md": ["| **V4** |", "| **V5.1** |", "| **V5.2** |", "| **V5.3** |",
                    "> **E3.1 bağımsız doğrulaması**"],
    "paper_draft.md": ["### 4.8 🌐", "### 4.9 🎯", "## 5. Tartışma", "## 6. Sınırlamalar",
                       "V5.1 — 65535 niceleme"],
    "README.md": ["exp/v4-directional-radial"],
    "data/external_vindr/README.md": ["vindr_report_"],
}
FILES = ["v5_1_physical_canonicalization_report.json", "v5_3_corrected_tda_vindr_report.json",
         "a0_a1_metadata_report.json", "a2_source_predictability_report.json", "vindr_summary_report.json"]


def main() -> None:
    ok = True
    for fname, needles in CHECKS.items():
        text = (ROOT / fname).read_text(encoding="utf-8")
        miss = [n for n in needles if n not in text]
        print(f"{fname:34s} {'OK' if not miss else 'EKSİK: ' + str(miss)}")
        ok = ok and not miss
    print()
    for f in FILES:
        p = ROOT / "data/external_vindr" / f
        print(f"{f:48s} {'var' if p.exists() else 'YOK'}")
        ok = ok and p.exists()
    print("\nSONUÇ:", "GEÇTİ" if ok else "SORUN VAR")


if __name__ == "__main__":
    main()

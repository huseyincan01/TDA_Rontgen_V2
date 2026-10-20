"""V5.1 rapor JSON'una sahip (owner) KEEP kararını ve gerekçesini ekler.

Ön-kayıtlı kuralın sonucu DEĞİŞTİRİLMEZ: kural-temelli karar (`decisions`) olduğu gibi
kalır; bunun yanına açık gerekçeli bir `owner_override` bloğu yazılır.

Kullanım:
    python scratch/v5_1_add_owner_override.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "data/external_vindr/v5_1_physical_canonicalization_report.json"

OVERRIDE = {
    "date": "2026-10-17",
    "decision": "KEEP",
    "applies_to": "a_density (65535 niceleme hatasinin duzeltilmesi)",
    "rationale": (
        "A2 kaynak AUC'nin yukselmesi basarisizlik degildir: 4-bit'e sikismis bulanik temsili "
        "8-bit'e (256 seviye) cikardigimiz icin bilgi miktari artmistir; cihaza ozgu doku da "
        "hastalik sinyali de birlikte guclenmistir. Asil hedefler buyuk olcude artmistir: "
        "val label AUC +0.0881, LOO +0.0611 (+TTA ile LOO +0.0718). Bu nedenle 65535 "
        "niceleme hatasinin duzeltilmesi KEEP olarak isaretlenmistir."
    ),
    "note_on_rule": (
        "Kural-temelli karar (decisions) KILL olarak kayitli kaldi; kapi degisikligi bir "
        "sonraki deney icin AYRICA on-kayit edilmelidir (LOO birincil + val guardrail; "
        "kaynak AUC tanisal, kapi degil)."
    ),
    "follow_ups": [
        "V5.2: duzeltilmis TDA -> 5-zon CNN + SVM ensemble ve ince ayarli ResNet18 fuzyonu (val-only)",
        "V5.3: VinDr-CXR harici dogrulamasinin duzeltilmis TDA ile tekrari",
        "V5.1b: sabit referans normalizasyon kontrolu (uyarlanabilir pencere confound'u)",
    ],
}


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["owner_override"] = OVERRIDE
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"owner_override yazıldı → {REPORT.name}")
    print(f"  owner kararı : {OVERRIDE['decision']}")
    print(f"  kural kararı : {report.get('decisions', {})}")


if __name__ == "__main__":
    main()

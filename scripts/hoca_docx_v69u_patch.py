"""HOCA notu (.docx) — V6.9u (kutu-içi geometrik/graf TDA) cümlesini §5 madde 5'e ekler.

`HOCA_BILGILENDIRME_NOTU.md` ile birebir aynı metni, şablonu bozmadan (run düzeyinde) ekler.
Kullanım: python scripts/hoca_docx_v69u_patch.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "HOCA_BILGILENDIRME_NOTU_SEMALI_10.docx"
BACKUP = ROOT / "scratch" / "hoca_docx_backup" / "_SEMALI_10_once_v69u.docx"

ANCHOR = "sabit kutuyu varsayılan olarak koruduk."
MARKER = "Vietoris-Rips graf"

ADD = [
    (" Ayrıca şüpheli kutunun ", False),
    ("içindeki şekli", True),
    (" farklı biçimlerde özetlemeyi de denedik (kutu merkezine uzaklıkla ağırlıklandırma, yapının "
     "piksel cinsinden ", False),
    ("kalınlığı", True),
    (", nokta bulutunun ", False),
    ("Vietoris-Rips graf", True),
    (" topolojisi; her biri kendi makine öğrenmesi modeliyle ayrı ayrı eğitildi). Üçü de ", False),
    ("yalnız şekil özeti kullanıldığında", True),
    (" doğru kutuyu bulmayı belirgin biçimde artırdı (Japonya dış testinde tümör kutusu isabeti ", False),
    ("%2,7 → %6,7", True),
    (", ilk kutunun tümöre denk gelme oranı ", False),
    ("%6,7 → %10,7", True),
    ("; kurum içi doğrulamada ", False),
    ("%12,1 → %20,4", True),
    ("), ancak ", False),
    ("iki aşama birleştirildiğinde ek kazanç sağlamadı", True),
    (": bu özetler 1. aşamanın zaten gördüğü bilgiye daha çok benziyor (1. aşama ile korelasyon "
     "0,22 → 0,27). Bu yüzden mevcut parlaklık tabanlı şekil özetinde kaldık.", False),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = docx.Document(str(DOCX))
    target = None
    for par in d.paragraphs:
        if par.text.strip().startswith("5. Dürüst sınırlılıklar."):
            target = par
            break
    if target is None:
        raise SystemExit("  [!] '5. Dürüst sınırlılıklar.' paragrafı bulunamadı")
    if MARKER in target.text:
        print("  [atlandı] cümle zaten ekli")
        return
    if ANCHOR not in target.text:
        raise SystemExit("  [!] çapa cümlesi ('koruduk.') bulunamadı")

    tmpl = target.runs[-1] if target.runs else None
    for text, bold in ADD:
        r = target.add_run(text)
        r.bold = bold
        if tmpl is not None:
            if tmpl.font.name:
                r.font.name = tmpl.font.name
            if tmpl.font.size is not None:
                r.font.size = tmpl.font.size
    print(f"  [+] {len(ADD)} run eklendi (paragraf uzunluğu {len(target.text)} karakter)")

    if args.dry_run:
        print("  (dry-run: kaydedilmedi)")
        return
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    d.save(str(DOCX))
    print(f"  kaydedildi: {DOCX.name} (yedek: {BACKUP.relative_to(ROOT)})")


if __name__ == "__main__":
    main()

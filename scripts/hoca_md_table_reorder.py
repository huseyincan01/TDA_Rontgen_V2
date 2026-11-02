"""HOCA notu (.md) §4 tablosunu .docx ile aynı sıraya getirir (JSRT satırı ikiye bölünür)."""
from __future__ import annotations

import pathlib
import re

MD = pathlib.Path(__file__).resolve().parents[1] / "HOCA_BILGILENDIRME_NOTU.md"
text = MD.read_text(encoding="utf-8")

lines = text.split("\n")
start = next(i for i, ln in enumerate(lines) if ln.startswith("| 1. Aşama aday yakalama"))
rows = lines[start:start + 6]
assert rows[0].startswith("| 1. Aşama aday") and rows[5].startswith("| Harici veri 2"), rows[5][:60]
recall, cnn, test1, test12, vindr, jsrt = rows


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def row(cs: list[str]) -> str:
    return "| " + " | ".join(cs) + " |"


# --- VinDr: etiket "Harici veri 3"
v = cells(vindr)
v[0] = v[0].replace("Harici veri 1", "Harici veri 3")
vindr_new = row(v)

# --- JSRT: iki satıra böl
j = cells(jsrt)
assert len(j) == 3, j
label_old, result_old, meaning_old = j

label_uncal = (label_old.replace("Harici veri 2", "Harici veri 1")
               + " · **kalibrasyonsuz (sıfır-eğitim)**")
result_uncal = result_old.split(" → ")[0]
tail = "yükseltiyor."
idx = meaning_old.find(tail)
assert idx > 0, "anlam hücresinde 'yükseltiyor.' bulunamadı"
meaning_uncal = meaning_old[:idx + len(tail)]

label_cal = ("Harici veri 2 — aynı Japonya (JSRT) verisi · "
             "**saha kalibrasyonlu (160 etiketli görüntü)**")
result_cal = ("**0,966 (Top-1) / 0,987 (Noisy-OR) hasta AUC · S@0.25 %56,4 · S@0.50 %66,4 · "
              "sıralama ölçütü 0,866**  (sıfır-eğitim: 0,920 / 0,965)")
meaning_cal = (
    "Sıfır-eğitimde zaten güçlü olan başarı, hedef hastaneden alınan **160 etiketli görüntüyle** "
    "(5 katlı çapraz doğrulama; test görüntülerine hiç dokunulmadan, her katlamada model sıfırdan) "
    "kurum içi kilitli test düzeyine taşınıyor: hasta AUC **0,901 → 0,966**, S@0.50 kutu isabeti "
    "**%56,4 → %66,4**, sıralama ölçütü **0,798 → 0,866**. Kutu isabetindeki kazanç neredeyse "
    "tamamen **1. aşamadan** geliyor (S@0.25 %49,0 → %57,7; 2. aşama tek başına yalnızca %50,3); "
    "hasta ayrımında ise **iki katman eşit katkı veriyor** (her biri ≈ +4 puan) ve birlikte "
    "**+6,6 puana** ulaşıyor. **10 etiketli görüntü kazanımın %55'ini, 40 görüntü %76'sını** "
    "getiriyor ⇒ sahada yüzlerce vaka değil, birkaç düzine etiket yeterli. ⚠️ Bu satır **yerel "
    "etiket gerektirir**; sıfır-eğitim iddiası değildir."
)

new_rows = [
    recall,
    cnn,
    test1,
    test12,
    row([label_uncal, result_uncal, meaning_uncal]),
    row([label_cal, result_cal, meaning_cal]),
    vindr_new,
]

out = lines[:start] + new_rows + lines[start + 6:]
MD.write_text("\n".join(out), encoding="utf-8")

for i, ln in enumerate(new_rows):
    print(f"{i + 1}: {ln[:96]}")
print(f"\nmd güncellendi ({len(lines)} → {len(out)} satır)")

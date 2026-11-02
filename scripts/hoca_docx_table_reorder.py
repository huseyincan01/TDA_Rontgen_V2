"""HOCA notu §4 tablosunu yeniden sıralar ve JSRT satırını ikiye böler (kalibresiz / kalibreli).

Yeni sıra (kullanıcı talebi):
  1) 1. Aşama aday yakalama (Recall@K)
  2) Küçük kutuda piksel CNN vs. lokal TDA        (dokunulmadı, gerekçe: TDA tezinin çekirdeği)
  3) Kilitli test (734 hasta) — yalnız 1. aşama
  4) Kilitli test — 1.+2. aşama birleşimi (Füzyon)
  5) Harici veri — Japonya (JSRT) · KALİBRESİZ
  6) Harici veri — Japonya (JSRT) · KALİBRELİ (160 etiketli görüntü)
  7) Harici veri — VinDr-CXR (Vietnam, 8-bit JPEG)   ← en sona

Kullanım:
    python scripts/hoca_docx_table_reorder.py --out scratch/_reorder_preview.docx   # önizleme
    python scripts/hoca_docx_table_reorder.py                                        # gerçek dosya
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "HOCA_BILGILENDIRME_NOTU_SEMALI_10.docx"
BACKUP = ROOT / "scratch" / "hoca_docx_backup" / "_SEMALI_10_once_reorder.docx"

LABEL_UNCAL = ("Harici veri 1 — Yüksek kalite tıbbi format (JSRT, Japonya; 149 nodüllü + 93 sağlıklı "
               "= 242 hasta, eğitimden tamamen izole) · ")
LABEL_UNCAL_B = "kalibrasyonsuz (sıfır-eğitim)"

LABEL_CAL_A = "Harici veri 2 — aynı Japonya (JSRT) verisi · "
LABEL_CAL_B = "saha kalibrasyonlu (160 etiketli görüntü)"

RESULT_UNCAL = [
    ("İlk 10 kutuda ", False), ("%86,6", True), (" · ilk 50 kutuda ", False), ("%95,3", True),
    (" · ", False), ("0,920", True), (" (Top-1) / ", False), ("0,965", True),
    (" (Noisy-OR) hasta AUC · S@0.25 ", False), ("%52,4", True), (" · S@0.50 ", False),
    ("%58,4", True), (" (sıfır-eğitim)", False),
]
RESULT_CAL = [
    ("0,966", True), (" (Top-1) / ", False), ("0,987", True),
    (" (Noisy-OR) hasta AUC · S@0.25 ", False), ("%56,4", True), (" · S@0.50 ", False),
    ("%66,4", True), (" · sıralama ölçütü ", False), ("0,866", True),
    ("  (sıfır-eğitim: 0,920 / 0,965)", False),
]
MEANING_UNCAL = [
    ("Dış hastanenin görüntüleri sıkıştırılmamış yüksek kaliteli (12-bit ", False), (".mha", False),
    (") olduğunda, sistem ", False), ("hiç kalibrasyon yapmadan (sıfır-eğitimle)", True),
    (" 242 saf Japon röntgeninde ilk 50 kutuda tümörlerin ", False), ("%95,3’ünü (142/149)", True),
    (" yakalıyor. Hastanın en güçlü tek kutusuna göre (", False), ("Top-1", True), (") ", False),
    ("0,920", True), (", kutuların ortak olasılığına göre (", False), ("Noisy-OR", True), (") ", False),
    ("0,965", True),
    (" hasta AUC elde ediliyor; 2. aşama topolojik katman hasta düzeyinde tek başına ", False),
    ("0,970 AUC", True),
    (" üretirken (kutu merkezini bilmediği için tek başına kutu koyamaz: S@0.25 %2,7), kilitli "
     "%55/%45 birleşiminde 1. aşamanın hasta ayrımını ", False), ("0,849 → 0,920 (+7,1 puan)", True),
    (" ve S@0.25 kutu isabetini ", False), ("%49,7 → %52,4 (+2,7 puan)", True), (" yükseltiyor.", False),
]
MEANING_CAL = [
    ("Sıfır-eğitimde zaten güçlü olan başarı, hedef hastaneden alınan ", False),
    ("160 etiketli görüntüyle", True),
    (" (5 katlı çapraz doğrulama; test görüntülerine hiç dokunulmadan, her katlamada model sıfırdan) "
     "kurum içi kilitli test düzeyine taşınıyor: hasta AUC ", False), ("0,901 → 0,966", True),
    (", S@0.50 kutu isabeti ", False), ("%56,4 → %66,4", True), (", sıralama ölçütü ", False),
    ("0,798 → 0,866", True),
    (". Kutu isabetindeki kazanç neredeyse tamamen ", False), ("1. aşamadan", True),
    (" geliyor (S@0.25 %49,0 → %57,7; 2. aşama tek başına yalnızca %50,3); hasta ayrımında ise ", False),
    ("iki katman eşit katkı veriyor", True),
    (" (her biri ≈ +4 puan) ve birlikte ", False), ("+6,6 puana", True), (" ulaşıyor. ", False),
    ("10 etiketli görüntü kazanımın %55’ini, 40 görüntü %76’sını", True),
    (" getiriyor ⇒ sahada yüzlerce vaka değil, birkaç düzine etiket yeterli. ⚠️ Bu satır ", False),
    ("yerel etiket gerektirir", True), ("; sıfır-eğitim iddiası değildir.", False),
]


def _cell_template(cell):
    """Hücrenin ilk run'ından (yazı tipi adı, boyutu) şablonu al."""
    for par in cell.paragraphs:
        for r in par.runs:
            return r.font.name, r.font.size
    return None, None


def _clear(cell) -> None:
    """Hücreyi boşalt (paragraf biçimini koru, fazla paragrafları sil)."""
    pars = cell.paragraphs
    for par in pars[1:]:
        par._element.getparent().remove(par._element)
    for r in list(cell.paragraphs[0].runs):
        r._element.getparent().remove(r._element)


def _fill(cell, pieces) -> None:
    """Hücreyi temizleyip verilen (metin, kalın) parçalarıyla yeniden yaz (biçimi korur)."""
    name, size = _cell_template(cell)
    _clear(cell)
    par = cell.paragraphs[0]
    for text, bold in pieces:
        r = par.add_run(text)
        r.bold = bold
        if name:
            r.font.name = name
        if size is not None:
            r.font.size = size


def _truncate_after(cell, needle: str) -> bool:
    """Hücre metnini `needle` sonrasında kes (kalan run/paragrafları sil)."""
    pars = cell.paragraphs
    for pi, par in enumerate(pars):
        for ri, r in enumerate(par.runs):
            idx = r.text.find(needle)
            if idx >= 0:
                r.text = r.text[:idx + len(needle)]
                for r2 in list(par.runs)[ri + 1:]:
                    r2._element.getparent().remove(r2._element)
                for p2 in pars[pi + 1:]:
                    p2._element.getparent().remove(p2._element)
                return True
    return False


def _drop_last_run(cell) -> None:
    for par in reversed(cell.paragraphs):
        if par.runs:
            par.runs[-1]._element.getparent().remove(par.runs[-1]._element)
            return


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DOCX))
    args = ap.parse_args()

    d = docx.Document(str(DOCX))
    tb = d.tables[0]
    rows = list(tb.rows)
    if len(rows) != 7:
        raise SystemExit(f"Beklenen 7 satır, bulunan {len(rows)} — tablo daha önce değiştirilmiş olabilir")

    jsrt_tr = rows[6]._tr
    vindr_tr = rows[5]._tr
    vindr_cell0_runs = [r for par in rows[5].cells[0].paragraphs for r in par.runs]
    # JSRT satırının bir kopyasını sona ekle (kalibreli satır bunun üzerine kurulacak)
    tb._tbl.append(copy.deepcopy(jsrt_tr))
    rows = list(tb.rows)
    uncal, cal = rows[6], rows[7]

    # ---- 1) kalibresiz satır: etiket + sonuç + anlam
    _fill(uncal.cells[0], [(LABEL_UNCAL, False), (LABEL_UNCAL_B, True)])
    _fill(uncal.cells[1], RESULT_UNCAL)
    if not _truncate_after(uncal.cells[2], "yükseltiyor."):
        raise SystemExit("  [!] 'yükseltiyor.' bulunamadı — anlam hücresi beklenen biçimde değil")

    # ---- 2) kalibreli satır
    _fill(cal.cells[0], [(LABEL_CAL_A, False), (LABEL_CAL_B, True)])
    _fill(cal.cells[1], RESULT_CAL)
    _fill(cal.cells[2], MEANING_CAL)

    # ---- 3) VinDr satırı en sona + etiketi "Harici veri 3"
    for r in vindr_cell0_runs:
        if "Harici veri 1" in r.text:
            r.text = r.text.replace("Harici veri 1", "Harici veri 3", 1)
            break
    order = [rows[0]._tr, rows[1]._tr, rows[2]._tr, rows[3]._tr, rows[4]._tr,
             uncal._tr, cal._tr, vindr_tr]
    for tr in order:
        tb._tbl.remove(tr)
    for tr in order:
        tb._tbl.append(tr)

    print("  yeni satır sırası:")
    for i, row in enumerate(tb.rows):
        print(f"    {i}: {row.cells[0].text.strip()[:72]}")

    out = Path(args.out)
    if out.resolve() != DOCX.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        d.save(str(out))
        print(f"  önizleme yazıldı: {out}")
        return
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    d.save(str(DOCX))
    print(f"  kaydedildi: {DOCX.name} (yedek: {BACKUP.relative_to(ROOT)})")


if __name__ == "__main__":
    main()

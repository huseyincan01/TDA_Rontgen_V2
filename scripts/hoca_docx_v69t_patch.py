"""HOCA notu (.docx) — V6.9t (242 Saf JSRT saha kalibrasyonu) yaması.

`HOCA_BILGILENDIRME_NOTU.md` içine işlenen üç değişikliği, kullanıcının şablonunu BOZMADAN
(yalnız run/cell/paragraph düzeyinde) Word belgesine uygular:

  1. §4 tablosu, JSRT satırı — "Sonuç" hücresine saha kalibrasyonu sayıları eklenir.
  2. §4 tablosu, JSRT satırı — "Ne anlama geliyor" hücresine kalibrasyon cümlesi eklenir.
  3. §5 — yeni "3. Yüksek Kaliteli Harici Veride Saha Kalibrasyonu" paragrafı eklenir;
     eski 3 → 4, eski 4 → 5 olarak yeniden numaralanır.

Kullanım:  python scripts/hoca_docx_v69t_patch.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import docx
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "HOCA_BILGILENDIRME_NOTU_SEMALI_10.docx"
BACKUP = ROOT / "scratch" / "hoca_docx_backup" / "_SEMALI_10_once_v69t.docx"

JSRT_CELL = "Harici veri 2"
RESULT_ADD = (" → saha kalibrasyonu (160 etiketli görüntü): 0,966 AUC · "
              "S@0.25 %56,4 · S@0.50 %66,4 · sıralama ölçütü 0,866")

MEANING_ADD = [
    (" Üstelik bu hastaneden yalnız ", False),
    ("160 etiketli görüntü", True),
    (" alınarak yapılan saha kalibrasyonunda (5 katlı çapraz doğrulama, test görüntülerine hiç "
     "dokunulmadan) hasta AUC ", False),
    ("0,966", True),
    ("’ya, S@0.50 kutu isabeti ", False),
    ("%66,4", True),
    ("’e, sıralama ölçütü ", False),
    ("0,866", True),
    ("’ya çıkıyor — yani sıfır-eğitimde zaten yüksek olan başarı, az sayıda yerel etiketle "
     "kurum içi kilitli test düzeyine (", False),
    ("0,966", True),
    (") taşınıyor", False),
]

NEW_LEAD = "3. Yüksek Kaliteli Harici Veride Saha Kalibrasyonu — Simetriyi Tamamladık."
NEW_BODY = (
    " Vietnam (düşük kaliteli JPEG) verisinde yaptığımız saha kalibrasyonunun aynısını Japonya "
    "(yüksek kaliteli kayıpsız) verisinde de yaptık. Hedef hastaneden N = 0, 10, 20, 40, 80, 120, "
    "160 etiketli görüntü alındığında, 5 katlı çapraz doğrulamayla ve test görüntülerine hiç "
    "dokunmadan, hem 1. aşama kutusunu çizen başlık hem de 2. aşama topolojik sınıflandırıcı yerel "
    "olarak kalibre edildi (her katlamada model sıfırdan, dondurulmuş başlangıç ağırlıklarından "
    "yeniden türetildi; bir katlamanın öğrendiği ağırlık diğerine geçmedi). Sonuç: hasta AUC "
    "0,901 → 0,966 (aynı ölçüm yolundaki tabana göre +6,6 puan; hiç kalibrasyon yapılmayan 3 "
    "ölçekli 0,920 tabanına göre +4,7 puan), S@0.50 kutu isabeti %56,4 → %66,4, sıralama ölçütü "
    "0,798 → 0,866. Kutu isabetindeki kazanç neredeyse tamamen 1. aşamadan geliyor (S@0.25 %49,0 → "
    "%57,7; 2. aşama tek başına yalnızca %50,3’e çıkarıyor). Hasta ayrımında ise iki katman eşit "
    "katkı veriyor (her biri ≈ +4 puan: 1. aşama 0,941, 2. aşama 0,940); iki aşama birlikte +6,6 "
    "puan ile tek başlarına ulaştıklarının ötesine geçiyor (S@0.50’de %66,4) ⇒ tamamlayıcılık "
    "korunuyor. Dahası, 10 etiketli görüntü kazanımın %55’ini, 40 görüntü "
    "%76’sını getiriyor ⇒ sahada yüzlerce vaka değil, birkaç düzine etiket yeterli. Böylece iki uç "
    "birbirini tamamlıyor: görüntü kalitesi düştükçe sıfır-eğitim başarısı düşüyor ama kalibrasyon "
    "kazancı büyüyor (Vietnam 8-bit JPEG’de 0,638 → 0,794, +15,6 puan), kalite yüksek olduğunda "
    "sıfır-eğitim zaten güçlü ve kalibrasyon onu kurum içi kilitli test düzeyine taşıyor (Japonya "
    "12-bit .mha’da 0,920 → 0,966, +4,7 puan). Bu satırların yerel etiket gerektirdiği ve "
    "sıfır-eğitim iddiası olmadığı açıkça belirtilmelidir."
)

OLD3 = "3. Ön-eğitim veri örtüşmesi skoru şişiriyor mu?"
NEW4 = "4. Ön-eğitim veri örtüşmesi skoru şişiriyor mu?"
OLD4 = "4. Dürüst sınırlılıklar."
NEW5 = "5. Dürüst sınırlılıklar."

# Metin hassasiyeti: "kazancın tamamı 1. aşamadan" yalnız KUTU isabeti için doğru; hasta AUC'de
# iki katman eşit katkı veriyor (Kol1 +0,039 / Kol2 +0,041). İlk yazımdaki cümle düzeltilir.
_FIX_HEAD = "Kazancın neredeyse tamamı 1. aşamadan geliyor (S@0.25 kutu isabeti"
_FIX_TAIL = "Dahası, 10 etiketli görüntü kazanımın"
_FIX_NEW = (
    "Kutu isabetindeki kazanç neredeyse tamamen 1. aşamadan geliyor (S@0.25 %49,0 → %57,7; 2. aşama "
    "tek başına yalnızca %50,3’e çıkarıyor). Hasta ayrımında ise iki katman eşit katkı veriyor (her "
    "biri ≈ +4 puan: 1. aşama 0,941, 2. aşama 0,940); iki aşama birlikte +6,6 puan ile tek başlarına "
    "ulaştıklarının ötesine geçiyor (S@0.50’de %66,4) ⇒ tamamlayıcılık korunuyor. "
)


def _append_runs(par, pieces: list[tuple[str, bool]], size=None) -> None:
    for text, bold in pieces:
        r = par.add_run(text)
        r.bold = bold
        if size is not None:
            r.font.size = size


def _insert_paragraph_before(par):
    """`par`dan hemen önce, aynı stille boş bir paragraf açar."""
    return par.insert_paragraph_before("", style=par.style)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DOCX.exists():
        raise SystemExit(f"Belge bulunamadı: {DOCX}")
    d = docx.Document(str(DOCX))

    # ---- 1) tablo hücreleri
    tb = d.tables[0]
    row = next(r for r in tb.rows if r.cells[0].text.strip().startswith(JSRT_CELL))
    cell_res, cell_mean = row.cells[1], row.cells[2]

    if "saha kalibrasyonu (160" in cell_res.text:
        print("  [atlandı] Sonuç hücresi zaten güncel")
    else:
        p = cell_res.paragraphs[-1]
        r = p.add_run(RESULT_ADD)
        r.bold = True
        if p.runs and p.runs[0].font.size:
            r.font.size = p.runs[0].font.size
        print("  [+] Sonuç hücresi güncellendi")

    if "saha kalibrasyonunda" in cell_mean.text:
        print("  [atlandı] Anlam hücresi zaten güncel")
    else:
        p = cell_mean.paragraphs[-1]
        _append_runs(p, MEANING_ADD, size=(p.runs[0].font.size if p.runs else None))
        print("  [+] Anlam hücresi güncellendi")

    # ---- 2) §5 yeni paragraf + yeniden numaralama
    paras = d.paragraphs
    p3 = next((p for p in paras if p.text.strip().startswith(OLD3[:28])), None)
    p4 = next((p for p in paras if p.text.strip().startswith(OLD4)), None)
    already = p3 is None
    if already:
        print("  [atlandı] §5 paragrafı/numaralama daha önce uygulanmış")
    elif any(p.text.strip().startswith(NEW_LEAD[:28]) for p in paras):
        print("  [atlandı] Yeni §5 paragrafı zaten var")
    else:
        new_p = p3.insert_paragraph_before("", style=p3.style)
        r1 = new_p.add_run(NEW_LEAD)
        r1.bold = True
        r2 = new_p.add_run(NEW_BODY)
        r2.bold = False
        # yazı tipi/boyutunu kaynak paragraftan devral
        if p3.runs:
            for r in (r1, r2):
                r.font.name = p3.runs[0].font.name
                r.font.size = p3.runs[0].font.size
        pf_new, pf_src = new_p.paragraph_format, p3.paragraph_format
        for attr in ("space_before", "space_after", "line_spacing", "first_line_indent"):
            val = getattr(pf_src, attr, None)
            if val is not None:
                setattr(pf_new, attr, val)
        print("  [+] Yeni §5 paragrafı eklendi")

    # ---- 3) numaralar
    if not already:
        for par, old, new in ((p3, OLD3, NEW4), (p4, OLD4, NEW5)):
            if par is None:
                continue
            done = False
            for r in par.runs:
                if r.text.startswith(old.split()[0] + " "):
                    r.text = r.text.replace(old.split()[0], new.split()[0], 1)
                    done = True
                    break
            print(f"  [{'✓' if done else '!'}] numaralama: {old[:2]} → {new[:2]}")

    # ---- 4) metin hassasiyeti düzeltmesi (idempotent)
    fixed = 0
    for par in d.paragraphs:
        for run in par.runs:
            if _FIX_HEAD in run.text:
                i = run.text.find(_FIX_HEAD)
                j = run.text.find(_FIX_TAIL, i)
                if j > i:
                    run.text = run.text[:i] + _FIX_NEW + run.text[j:]
                    fixed += 1
    print(f"  [{'✓' if fixed else '='}] metin hassasiyeti düzeltmesi ({fixed} run)")

    if args.dry_run:
        print("  (dry-run: kaydedilmedi)")
        return
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    d.save(str(DOCX))
    print(f"  kaydedildi: {DOCX.name} (yedek: {BACKUP.relative_to(ROOT)})")


if __name__ == "__main__":
    main()

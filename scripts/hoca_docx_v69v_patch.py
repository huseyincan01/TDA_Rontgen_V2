"""HOCA notu (.md + .docx) — V6.9v (kalibre radyal TDA) bulgusunu §5 madde 5'e ekler."""
import pathlib

MD_ANCHOR = "mevcut parlaklık tabanlı şekil özetinde kaldık."
MD_ADD = (
    " Bu özeti bir kez daha, **daha adil koşullarla** sınadık: önceki denemede şekil özetinin "
    "eksenleri gerçek veri aralığına göre ölçeklenmemişti (ölüm değerlerinin tamamı grafiğin tek "
    "satırına sıkışıyordu) ve tek bir \"merkez cezası\" şiddeti denenmişti. Eksenleri eğitim "
    "verisinden türetip üç farklı şiddetle yeniden eğittiğimizde, kendi hastanemizin **kilitli "
    "test setinde altı ölçütün tamamı iyileşti** (hasta AUC **0,9661 → 0,9750**, S@0.25 "
    "**%72,1 → %73,4**, S@0.50 **%76,4 → %78,6**, sıralama ölçütü **0,9047 → 0,9146**). Ancak aynı "
    "özet **yabancı Japonya verisinde geriletti** (0,8206 → 0,8133) ve nedeni ölçüldü: yabancı "
    "hastanede 1. aşama tümörün merkezini daha az tutturuyor — şüpheli kutuların **%47'si** yamanın "
    "\"göbeği\" dışında kalıyor (kendi verimizde %24) — dolayısıyla merkez cezası orada doğru "
    "tümörü de cezalandırıyor. Bu özet 1. aşama ile korelasyonu **artırmıyor** (r ≈ 0,22), yani "
    "gerçekten ek bilgi taşıyor; buna rağmen taşınmadığı için ana modeli **değiştirmedik** ve "
    "sadeleştirme + dış dayanıklılık tercihimizi koruduk. Bu satırı, kendi hastanemizde çalışan ama "
    "hastaneye özgü bir \"konum önsavı\" olarak dürüstçe belgeliyoruz."
)

p = pathlib.Path("HOCA_BILGILENDIRME_NOTU.md")
t = p.read_text(encoding="utf-8")
assert MD_ANCHOR in t, "md çapası bulunamadı"
assert "yabancı Japonya verisinde geriletti" not in t, "zaten eklenmiş"
p.write_text(t.replace(MD_ANCHOR, MD_ANCHOR + MD_ADD, 1), encoding="utf-8")
print("HOCA md güncellendi | bozuk:", p.read_text(encoding='utf-8').count(chr(0xFFFD)))

# ---- docx yaması (run düzeyi, idempotent)
import shutil
import docx

DOCX = pathlib.Path("HOCA_BILGILENDIRME_NOTU_SEMALI_10.docx")
BACKUP = pathlib.Path("scratch/hoca_docx_backup/_SEMALI_10_once_v69v.docx")
ANCHOR = "mevcut parlaklık tabanlı şekil özetinde kaldık."
MARK = "geriletti"

RUNS = [
    (" Bu özeti bir kez daha, ", False), ("daha adil koşullarla", True),
    (" sınadık: önceki denemede şekil özetinin eksenleri gerçek veri aralığına göre "
     "ölçeklenmemişti (ölüm değerlerinin tamamı grafiğin tek satırına sıkışıyordu) ve tek bir "
     "\u201cmerkez cezası\u201d şiddeti denenmişti. Eksenleri eğitim verisinden türetip üç farklı "
     "şiddetle yeniden eğittiğimizde, kendi hastanemizin ", False),
    ("kilitli test setinde altı ölçütün tamamı iyileşti", True),
    (" (hasta AUC ", False), ("0,9661 → 0,9750", True), (", S@0.25 ", False),
    ("%72,1 → %73,4", True), (", S@0.50 ", False), ("%76,4 → %78,6", True),
    (", sıralama ölçütü ", False), ("0,9047 → 0,9146", True),
    ("). Ancak aynı özet ", False), ("yabancı Japonya verisinde geriletti", True),
    (" (0,8206 → 0,8133) ve nedeni ölçüldü: yabancı hastanede 1. aşama tümörün merkezini daha az "
     "tutturuyor — şüpheli kutuların ", False), ("%47\u2019si", True),
    (" yamanın \u201cgöbeği\u201d dışında kalıyor (kendi verimizde %24) — dolayısıyla merkez cezası "
     "orada doğru tümörü de cezalandırıyor. Bu özet 1. aşama ile korelasyonu ", False),
    ("artırmıyor", True), (" (r ≈ 0,22), yani gerçekten ek bilgi taşıyor; buna rağmen taşınmadığı "
     "için ana modeli ", False), ("değiştirmedik", True),
    (" ve sadeleştirme + dış dayanıklılık tercihimizi koruduk. Bu satırı, kendi hastanemizde "
     "çalışan ama hastaneye özgü bir \u201ckonum önsavı\u201d olarak dürüstçe belgeliyoruz.", False),
]

d = docx.Document(str(DOCX))
target = next((x for x in d.paragraphs if x.text.strip().startswith("5. Dürüst sınırlılıklar.")), None)
assert target is not None, "docx: madde 5 bulunamadı"
if MARK in target.text:
    print("docx: zaten ekli")
else:
    assert ANCHOR in target.text, "docx: çapa bulunamadı"
    tmpl = target.runs[-1] if target.runs else None
    for text, bold in RUNS:
        r = target.add_run(text)
        r.bold = bold
        if tmpl is not None:
            if tmpl.font.name:
                r.font.name = tmpl.font.name
            if tmpl.font.size is not None:
                r.font.size = tmpl.font.size
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    d.save(str(DOCX))
    print(f"docx: {len(RUNS)} run eklendi · madde 5 uzunluk {len(target.text)} karakter")

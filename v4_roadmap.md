# NODE21 TDA — v4 Araştırma Programı

> **Bu belge nedir?** v3 **kapandı** ve `v3-publication-ready` etiketiyle donduruldu. Bu, onun üstüne kurulan **yeni** program.
> **v3 dokümanları bu dalda DEĞİŞTİRİLMEZ.** v3'ün tam hâli: `git switch main` veya `git show v3-publication-ready`.
> **Dal:** `exp/v4-directional-radial` · **v3 temeli:** commit `78cace3`

---

## 0. v3 mirası — ne donduruldu

| | |
| :--- | :--- |
| **Ana sonuç** | Füzyon test ROC-AUC **0.9516**, doğruluk **%89.4** |
| **TDA katkısı** | İnce ayarlı güçlü baseline'a **+0.0232** [0.0132, 0.0338], p=0.0000 |
| **Mekanizma** | PL üst-zarf çökmesi (E4.4) · tamamlayıcılık r ≈ 0.30 |
| **Kapsam** | Hasta düzeyi sınıflandırma; **lokalizasyon yok** (E4.7) |
| **Test bütçesi** | **6/6 TÜKENDİ** — v3 test seti kapalı |
| **Kayıtlar** | [ablation_matrix.md](ablation_matrix.md) · [experiments.md](experiments.md) · [paper_draft.md](paper_draft.md) · `artifacts/figures/F1..F8` |

---

## 1. 🔑 v4'ü doğuran bulgu (V1.1 + V1.2)

v3'ün sonunda iki **val düzeyinde doğrulanmış** TDA iyileştirmesi vardı. v4'ün ilk işi bunların **füzyona** taşınıp taşınmadığını ölçmekti:

| Temsil | TDA **tek başına** | **FÜZYON** | **d_fuzyon** | d_tek |
| :--- | :---: | :---: | :---: | :---: |
| **A** — mevcut PI (800) | 0.8399 | 0.9430 | — | — |
| **B** — E2.6 gradyan dilimleri (2400) | **0.8880** | 0.9501 | **+0.0072** | **+0.0480** |
| **C** — E2.1 diphasic (1600) | 0.8609 | 0.9410 | **−0.0019** | +0.0210 |
| **D** — birleşik (3200) | 0.8861 | 0.9512 | **+0.0082** | +0.0461 |

* Görüntü dalı tek başına: **0.8887** · Kriter: d_fuzyon ≥ **+0.015** · **Sonuç: hiçbiri geçemedi (KILL)**

### 🚨 Bu ne anlama geliyor

1. **TDA-tek başarım artışı, füzyon kazancına dönüşmüyor.** E2.6 +0.0480 → sadece **+0.0072**; E2.1 +0.0210 → **−0.0019** (füzyona **zarar**).
2. ⇒ **TDA dalındaki iyileştirmeler, ResNet18'in zaten gördüğü bilgiyi yakalıyor** — yani **yedekli**.
3. ⇒ **v3'ün iyileştirme arayışı, füzyon açısından YANLIŞ ölçütü optimize ediyordu.** TDA dalını *doğru* yapmak, füzyonu *daha iyi* yapmıyor.
4. ✅ **v3'ün kendi iddiası bundan etkilenmez:** v3 "TDA temsili iyileşti" demedi; "TDA + görüntü füzyonu" dedi ve o sonuç (0.9516) **aynen geçerli**.

---

## 2. v4 tezi

> **Topolojik temsiller, görüntü temsilini TAMAMLAYACAK biçimde tasarlanmalıdır — "daha doğru" olacak biçimde değil.**

**Ölçüt değişikliği:**

| | v3 ölçütü | **v4 ölçütü** |
| :--- | :--- | :--- |
| Hedef | TDA dalının tek başına AUC'si | **Füzyon AUC'si (d_fuzyon)** |
| Başarı | Δ ≥ +0.015 (tek başına) | **Δ ≥ +0.015 (füzyonda)** |
| Teşhis | — | **r(TDA, ResNet)** — düşük r + yüksek AUC = değerli |

**Yeni tanı metrikleri:**
* `d_fuzyon` — asıl karar ölçütü
* `r(TDA, ResNet)` — yedeklilik göstergesi (v3 temeli: **r ≈ 0.30**)
* **"Yedeklilik kaybı"** = d_tek − d_fuzyon (E2.6'da 0.0480 − 0.0072 = **0.0408**; E2.1'de 0.0210 − (−0.0019) = **0.0229**)

---

## 3. Faz planı

### Faz V0 — Altyapı ✅ TAMAM
- v3 donduruldu (`v3-publication-ready` tag)
- v4 dalı açıldı
- **V1.1** füzyon kapısı (KILL) · **V1.2** temsil karşılaştırması (KILL)

### Faz V1 — ✅ TAMAM: "Mevcut iyileştirmeler füzyonda duruyor mu?"
**Sonuç: HAYIR.** v4'ün yönü bu bulguyla belirlendi.

### Faz V2 — **Tamamlayıcılık arayışı** (v4'ün kalbi)

Adaylar, **d_fuzyon** ölçütüyle değerlendirilir:

| # | Aday | Hipotez | Neden umut verici |
| :-: | :--- | :--- | :--- |
| **V2.1** | **Konum filtrasyonları** (yön/yükseklik `f + t·y`, `f + t·x`) | Uzamsal düzen, ResNet'in *global* özelliklerinden farklı | Konum bilgisi ResNet'in avgpool'unda **kaybolur** ⇒ yedekli olma ihtimali düşük |
| **V2.2** | **Radyal filtrasyon** (`f + t·r`) | Aynı gerekçe | Aynı |
| **V2.3** | **Düşük çözünürlüklü/anatomik ölçek** — çok kaba PI (ızgara 5-8) | ResNet dokuya bakar, kaba topoloji **organ düzeyi** yapıyı kodlar | Ölçek ayrışması ⇒ düşük yedeklilik |
| **V2.4** | **Hasta düzeyi global PH** — 5 zonun birleşik akciğer maskesi üzerinde **tek** diyagram | Global yapı, zon-zon ilişkileri | v3'te **hiç denenmedi**; E4.7/E1.2 "sinyal global" diyor ⇒ tam da orası |

> **V2.4 dikkat çekici:** v3'ün en güçlü *negatif* bulgusu (E4.7 + E1.2 + E3.6: sinyal global/arka plan) aslında **v4 için bir tasarım ipucu** — madem sinyal global, **global bir temsil** kurmak mantıklı.

### Faz V3 — İlişkisel temsiller (V2 sonrası eklendi)
- §7'deki yapısal çıkarımdan doğdu: görüntü dalı zonları bağımsız işlediği için **zonlar arası** ilişki tamamlayıcı olabilir.
- **V3.1** zon-zon topolojik mesafeler · **V3.2** PI simetrik farkı — ikisi de **KILL** (bkz. §6).

### Faz V3-B — Bileşen birleştirme + sıkıştırma (⏸️ askıda)
> Kimlik çakışmasını önlemek için ayrı adlandırıldı (V3.x kimlikleri ilişkisel deneylere verildi). **Birleştirilecek kazanan aday olmadığı için askıda.**
- Kazanan adayları birleştir
- **PCA sıkıştırma**: v3 bulgusu (E2.5b) PI'ın 800 boyutu **192'ye indirilebiliyor, kayıpsız** ⇒ blok başına ~200 boyut, 4 blok = 800. Bu, aşırı uyumu azaltır ve çok bloklu temsilleri mümkün kılar.

### Faz V4 — Doğrulama
- **Harici veri** (JSRT / ChestX-ray14) ile dış doğrulama — **asıl sıçrama**
- v4 test protokolü: **yeni ve önceden beyan edilmiş küçük bir bütçe**; v3 test setinin yeniden kullanımı **açıkça raporlanır** (kümülatif kullanım sayısıyla birlikte)

---

## 4. v4 protokol kuralları

| Kural | Uygulama |
| :--- | :--- |
| **Tek değişken** | v3'teki gibi |
| **Ana ölçüt** | **d_fuzyon ≥ +0.015** (tek başına AUC **karar ölçütü değil**) |
| **Boyut kontrolü** | Her aday için karıştırılmış kol (v3 pratiği) |
| **Baseline bütünlüğü** | Baseline **yeniden türetilmez**; projenin dondurulmuş özellikleri kullanılır (E2.1/E2.6 dersi) |
| **Kural 10** | ≥3 seed / deterministik yöntemde tekrar koşu |
| **v3 kayıtları** | **Değiştirilmez** — v4 kayıtları bu belgeye ve `experiments.md`'nin v4 bölümüne yazılır |

---

## 5. Durum tablosu

| Faz | Durum |
| :--- | :--- |
| **V0** Altyapı | ✅ tamam |
| **V1** Kapı testi | ✅ tamam — **KILL** (v4 yönü belirlendi) |
| **V2** Tamamlayıcılık arayışı | ✅ tamam — V2.1/V2.2, V2.3, V2.4 hepsi **KILL** |
| **V3** İlişkisel temsiller | ✅ tamam — V3.1, V3.2 **KILL** |
| **V3-B** Birleştirme + PCA | ⏸️ askıda — birleştirilecek kazanan yok |
| **V4** Harici doğrulama | Planlandı |

---

## 6. Karar günlüğü (v4)

| Tarih | Deney | Karar | Özet |
| :--- | :--- | :--- | :--- |
| 2026-10-16 | **V1.1** | **KILL** ⚠️ | **Füzyon kapısı.** E2.6'nın TDA-tek kazancı (+0.0480) füzyonda **+0.0072**'ye düşüyor. Kontrol (dilimler karıştırılmış) füzyonda 0.9090, taban 0.9430 ⇒ TDA kalitesi füzyonu etkiliyor, ama E2.6'nın katkısı **ResNet ile yedekli**. |
| 2026-10-16 | **V1.2** | **KILL** 🎯 | **Temsil karşılaştırması (füzyon ölçütüyle).** d_fuzyon: E2.6 **+0.0072** · E2.1 diphasic **−0.0019** (*füzyona zarar*) · birleşik **+0.0082**. Hiçbiri +0.015'i geçmiyor ⇒ **TDA iyileştirmeleri ResNet ile yedekli.** |
| 2026-10-16 | **SONUÇ** | 🔑 | **Ölçüt değişikliği:** v4, TDA-tek AUC yerine **d_fuzyon** ve **r(TDA, ResNet)** optimize eder. Aday tasarımı **tamamlayıcılık** eksenine kayar. |
| 2026-10-16 | **V2.1 + V2.2** | **KILL** ⚠️🎯 | **Konum tabanlı filtrasyonlar (dikey `f+t·y`, yatay `f+t·x`, radyal `f+t·r`), t ∈ {0.5, 1.5}.** Val-only. TDA-tek **+0.0256 / +0.0273 / +0.0280** (hepsi kazandı) **AMA füzyon −0.0053 / −0.0022 / −0.0026** (hepsi kaybetti). **Ve kritik olan:** korelasyon **r = 0.293 → 0.392 / 0.415 / 0.393** ⇒ **konum filtrasyonları ResNet ile YEDEKLİLİĞİ ARTIRIYOR.** Kontrol (karıştırılmış): TDA-tek 0.7649, füzyon 0.9178, r=0.238. ⇒ *"Konum bilgisi ResNet'in avgpool'unda kaybolur"* hipotezim **ÇÜRÜDÜ**; ResNet konum bilgisini koruyor. 🎯 **v4'ün tezi nicel olarak doğrulandı:** d_tek ↑ ⇒ r ↑ ⇒ d_fuzyon ↓. |
| 2026-10-16 | **V2.4** | **KILL** ⚠️ | **Hasta düzeyi global PH.** Birleşik akciğer maskesi üzerinde **tek** diyagram, 3414 train / 734 val hasta. ⚠️ İlk iki koşu **geçersizdi**: `sitk.Resample` spacing hatası geniş `except` tarafından **sessizce yutuldu** → 4148 boş diyagram, şüpheli 40 s çıkarım süresi ve **tam 0.5000 AUC**. Düzeltme: blok ortalaması + **%5'ten fazla boş diyagramda hata fırlatan sanity koruması**. Geçerli sonuç: tek başına **0.6998** (zon PI'ı 0.8399), füzyon **0.8940** (taban 0.9430), **r = 0.357** (taban 0.293). ⇒ *"Global temsil daha az yedekli olur"* hipotezim **çürüdü**: hem daha zayıf hem daha yedekli. |
| 2026-10-16 | **V2.3** | **KILL** ⚠️ | **Ölçek ayrımı.** Aynı diyagramlar ızgara {6,10,15,20}. Kaba ızgaralar hem tek başına (0.7165→0.7253) hem füzyonda (0.9109→0.9134) **biraz daha kötü**; r neredeyse sabit (0.209–0.217). 🎁 **Yan gözlem:** 72 boyutlu ızgara-6, tek başına **0.7165** — 800 boyutlu ızgara-20'nin (0.7253) neredeyse aynısı ⇒ **E2.5b'nin yedeklilik bulgusu bağımsız olarak teyit edildi.** |
| 2026-10-16 | **V3.1** | **KILL** 🔑 | **İlişkisel temsil — 5 zon arasındaki 10 ikili topolojik mesafe.** Sliced Wasserstein (gudhi'nin `wasserstein`'ı `ot` paketi olmadığı için çalışmıyor; H0 diyagramları 2444 nokta olduğundan tam LP de pratik değil), diyagram başına en kalıcı 100 sınıf. 20 özellik (10 çift × H0/H1), hasta düzeyi. **Kontrol: partner zonlar FARKLI hastalardan.** Sonuç: **B ilişkisel 0.6284** vs **E kontrol 0.5979** ⇒ fark yalnızca **+0.0305** (≈ gürültü) ⇒ **ilişkisel sinyal yok denecek kadar az.** C (PI+ilişkisel) füzyonda **0.9370**, taban **0.9430** ⇒ **−0.0060**; r yine **arttı** (0.293→0.323). **KILL.** |
| 2026-10-16 | **V3.2** | **KILL** 🔑 | **PI simetrik farkı** — piksel hizalaması olmadan, doğrudan PI uzayında: `concat(|PI_1−PI_4|, |PI_3−PI_5|)` = 1600 özellik (projenin **dondurulmuş** PI'ı, yeniden türetim yok). Hasta düzeyi, 3414 train / 734 val, **test yok**. Sonuçlar: **B ayna 0.8118** · **D yanlış ayna (çapraz 1↔5, 3↔4) 0.8389** · **E karıştırılmış 0.6731** · **C (PI+ayna) füzyon 0.9496** (taban 0.9430) ⇒ **d_fuzyon +0.0066** · F (PI+yanlış) 0.9441. **Karar: KILL** (eşik +0.015). |

### 🔑 V3.1 neden başarısız oldu — ve bu ne söylüyor

**Zon şemamız SİMETRİK DEĞİL:** sağ akciğer **3 zon** (üst/orta/alt), sol akciğer **2 zon** (üst/alt) — çünkü kalp yer kaplıyor.

⇒ 10 çiftten yalnızca **~2'si anatomik olarak karşılaştırılabilir** (1↔4 üst, 3↔5 alt); kalan 8 çift **farklı anatomik bölgeleri** karşılaştırıyor.

⇒ Ölçülen mesafeler ağırlıklı olarak **zon TİPLERİ arasındaki sabit anatomik farkı** kodluyor (her hastada aynı) — **hastaya özgü asimetriyi değil.** Kontrol kolunun neredeyse aynı sonucu vermesi bunu doğruluyor.

**⇒ Bu, Çözüm 2'yi (görüntü düzeyinde ayna çıkarma) işaret ediyor:** ayna işlemi **zon şemasına ihtiyaç duymaz**, tüm akciğer alanını doğrudan karşılaştırır ve simetri problemi böylece ortadan kalkar.

### 🔑 V3.2'den çıkan dört bulgu

**1. ✅ Çapraz-zon PI farkları GERÇEKTEN bilgi taşıyor (V3.1'in aksine).**
`B ayna 0.8118` vs `E karıştırılmış 0.6731` ⇒ **+0.139**. V3.1'in skaler mesafeleri (0.6284 vs 0.5979 = +0.03) neredeyse bilgi taşımıyordu; **eleman-eleman PI farkı çok daha güçlü.**

**2. ❌ AMA sinyal SİMETRİYE ÖZGÜ DEĞİL — hipotez çürüdü.**
`D yanlış ayna (0.8389)` > `B doğru ayna (0.8118)` ⇒ **−0.0271**. Yani "sağ üst ↔ sol üst" karşılaştırması, "sağ üst ↔ sol alt" karşılaştırmasından **daha iyi değil, daha kötü.**
⇒ Bilgilendirici olan **ayna simetrisi değil**, sadece **"iki farklı bölgenin farkı"**. Muhtemelen dikey topolojik gradyanı yakalıyor.

**3. 📚 Bu, literatürle BİREBİR örtüşüyor.**
[Evans ve ark. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC9684389/): global gist **"görüntünün yoğunluğundan veya ASİMETRİSİNDEN değil"**, global yapıdan geliyor. ⇒ **Bağımsız olarak aynı sonucu bulduk.**

**4. ⚠️ Füzyon kazancı (+0.0066) gürültü mertebesinde.** 734 val hastası için AUC standart hatası ~0.012–0.015 ⇒ +0.0066 **anlamlı değil**. (Bootstrap CI hesaplanmadı; hesaplanmalı.)

**⇒ Simetri/aşırılık ekseni kapandı.** Sıradaki anlamlı yön, V3.1 ve V3.2'nin ortak dersini kullanan **farklı bir ilişkisellik türü** ya da doğrudan **mimari** (tüm görüntü + skip connection).

---

## 7. 🔬 Kritik gözlem: neden hepsi yedekli çıkıyor?

| Temsil | d_tek | d_fuzyon | r |
| :--- | :---: | :---: | :---: |
| (PI tabanı) | — | — | **0.293** |
| E2.1 diphasic | +0.0210 | **−0.0019** | — |
| E2.6 gradyan | +0.0480 | **+0.0072** | — |
| V2.1 dikey | +0.0256 | **−0.0053** | 0.392 |
| V2.1 yatay | +0.0273 | **−0.0022** | 0.415 |
| V2.1 radyal | +0.0280 | **−0.0026** | 0.393 |
| V2.4 global PH | −0.1401 | **−0.0490** | 0.357 |
| V3.1 ilişkisel mesafe | — | **−0.0060** | 0.323 |
| V3.2 PI simetrik fark | — | **+0.0066** | — |

**Desen:** TDA dalında başarımı artıran her müdahale, **ResNet'e benzerliği de artırıyor** ⇒ füzyon kazancı yok.

**🔑 Yapısal çıkarım (V3 öncesi hipotez):** Görüntü dalımız (ResNet18) her zonu **BAĞIMSIZ** işleyip max-pool yapıyor. Dolayısıyla:
* Tek-zon içi temsil iyileştirmeleri ⇒ **kaçınılmaz olarak yedekli** (ResNet de aynı zonu görüyor)
* **Zonlar ARASI ilişki** kuran temsiller ⇒ görüntü dalının **yapısal olarak temsil EDEMEDİĞİ** bilgi

⇒ Bu çıkarım **Faz V3'ü doğurdu** (ilişkisel temsiller).

**❌ Güncelleme (V3.1 + V3.2 sonrası):** İlişkisel hipotez de **tutmadı** — skaler zon mesafeleri bilgi taşımıyor (V3.1), PI farkları bilgi taşıyor ama simetriye özgü değil ve füzyon kazancı gürültü mertebesinde (V3.2, +0.0066). ⇒ Önceden tanımlanmış TDA eklentilerinin hiçbiri füzyon ölçütünü geçemedi. Açık yönler: **mimari** (tüm görüntü / uçtan uca öğrenilen füzyon) veya **V4 harici doğrulama**.

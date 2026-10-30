# NODE21 TDA — Araştırma Yol Haritası ve Deney Yönetimi (v3 Programı)

> Bu belge, kilitli **v2.1 hattı** (bkz. [progress.md](progress.md)) üzerine kurulacak **v3 araştırma programının** planıdır: iş kolları, deney backlog'u, bağımlılıklar, öldürme kriterleri ve yönetim protokolü.
> Sonuç kaydı: [experiments.md](experiments.md) — her deney burada ID'siyle raporlanır.

**Amaç:** Yayınlanabilir bir katkı üretmek. Katkı "SOTA nodül deteksiyonu" değil; **topolojik temsilin tamamlayıcı + yorumlanabilir bilgi taşıdığını kanıtlamak** ve bunu sağlam bir ablasyon matrisiyle desteklemek.

---

## 0. Yönetim Protokolü (Değişmez Kurallar)

Bu 8 kural, "her fikri denemek" ile "sonuçların güvenilir olması" arasındaki dengeyi kurar.

1. **Tek değişken kuralı.** Her deney yalnızca **bir** faktörü değiştirir (ör. yalnız kemik bastırma, yalnız çok ölçek). İki faktörü aynı anda değiştiren deney yalnızca "kombinasyon" olarak etiketlenir ve tek başına kanıt sayılmaz.
2. **Dondurulmuş test seti.** Model/eşik/hiperparametre seçimi **yalnızca validation** üzerinde yapılır. Test setine (734 hasta) yalnızca **faz sonlarında** dokunulur.
3. **Test seti bütçesi.** Program boyunca en fazla ~6 test değerlendirmesi yapılır (her fazın en iyi 1–2 modeli). Gerekçe: test setini tekrar tekrar kullanmak yayın bütünlüğünü bozar.
4. **Ön-kayıt (pre-registration).** Her deney başlamadan önce hipotezi ve **başarı eşiğini** backlog tablosuna yazılır. Deney bitince yalnızca "tuttu/tutmadı" kararı verilir; eşik sonradan değiştirilemez.
5. **Kayıt protokolü.** Her deney şu alanlarla kaydedilir: `ID, iş kolu, hipotez, tek değişken, bağımlılık, config, git commit, val metriği, test metriği, karar (KEEP/KILL), not`.
6. **Anlamlılık eşiği.** Bir iyileşme "gerçek" sayılır ancak **val ROC-AUC farkı ≥ +0.015** ve **N=2000 bootstrap %95 CI kesişmiyorsa**. Bunun altındaki farklar gürültü kabul edilir (mevcut bootstrap altyapısı: [07_compute_bootstrap_ci.py](pipelines/07_compute_bootstrap_ci.py)).
7. **Öldürme kriteri.** Yüksek riskli her deney için önceden "hangi durumda bırakırım" tanımlanır (bkz. §6). Kriter tetiklenirse deney **KILL** işaretlenir ve zaman harcanmaz.
8. **Karar günlüğü.** Her hafta sonunda roadmap.md durum tablosu + experiments.md güncellenir; verilen KEEP/KILL kararlarının **gerekçesi** yazılır. Bu, yazım aşamasında "neden bunu seçtiniz?" sorularının cevabıdır.
9. **SPLIT BEYANI (ZORUNLU).** Verimiz NODE21'in **halka açık training setidir** (4882 = 1134 + 3748; yerel olarak doğrulandı). Test setimiz bu setin **kendi iç bölünmesidir**; NODE21'in resmî test setleri (281 experimental + 298 final) **özeldir ve kullanılmamıştır**. Bu nedenle **leaderboard ile karşılaştırma yapılmaz**; her sonuç "kendi iç split'imizde" olarak sunulur. Sızıntı yoktur (hasta düzeyi, 0 kesişim). Ayrıntı: [literature.md §3.4](literature.md).
10. **ÇOK-SEED RAPORLAMA.** E4.6'da CNN'in seed varyansı **±0.02–0.03** ölçüldü (val AUC 0.8321–0.8922). Bu nedenle makalede raporlanan model performansları **en az 3 seed ortalaması** (± std) olarak verilir. Tek-seed sonuç verilirse yanında **seed varyansı** belirtilir. Sınıflandırıcı-olmayan analizler (E4.4, E2.7, E4.2) ve bootstrap CI'lı füzyon karşılaştırmaları bu kuralın dışındadır.

### Anlamlı "kanıt" ne demek?

| Kanıt seviyesi | Gereksinim |
| :--- | :--- |
| **Zayıf** | Tek seed, val farkı < +0.015 → gürültü, kayda değmez |
| **Orta** | En az 3 seed ortalaması, val farkı ≥ +0.015 |
| **Güçlü (yayın)** | 3+ seed, val ≥ +0.015, bootstrap CI kesişmiyor, faz sonunda testte doğrulandı |

---

## 1. Hesaplama Bütçesi ve Darboğazlar

| Kaynak | Doğrulanmış durum |
| :--- | :--- |
| GPU | **NVIDIA RTX 4050 Laptop, 6 GB VRAM** (fiziksel olarak var, boşta: 0% kullanım, 5.921 MiB boş) |
| ✅ **PyTorch** | **`torch 2.14.1+cu126`** — `cuda.is_available() = True`, **RTX 4050 Laptop (6 GB VRAM, 20 SM)**. E0.0 **tamamlandı** (2026-10-14). CPU-only dönemde üretilen tüm sonuçlar **geçerli** (sürüm aynı, yalnızca derleme değişti) |
| VRAM sınırı | 6 GB → pretrainli omurga için küçük batch, ~224px girdi |
| Disk | C: üzerinde **~54 GB boş** |
| Pretrained ağırlık | `~/.cache/torch` **yok** → ilk çalıştırmada indirme gerekir (internet şart) |
| Özellik çıkarımı | 24.410 zon ≈ **36 dk** (10 worker, CPU/GUDHI) → her yeni ön-işleme tam yeniden çıkarım |
| Kütüphaneler | gudhi 3.13.0 ✓ · scikit-image 0.26.0 ✓ (**`frangi/sato/meijering` hazır** → E1.2 ucuz) · SimpleITK 2.5.6 ✓ · torchxrayvision 1.5.5 ✓ |
| Veri | 4.882 hasta, 24.410 zon, 800 boyut PI (78 MB) |
| Kilitli test | 734 hasta (172 nodül / 562 kontrol) |

**Yönetim sonucu:** Ön-işleme varyantları (kemik/damar bastırma vb.) **toplu** çıkarılır (gece boyu tek seferde tüm varyantlar), sonra sınıflandırıcı deneyleri hızlıca koşturulur. GPU'yu beklerken bağımsız iş kolları (ör. değerlendirme/analiz) ilerletilir.

---

## 2. İş Kolları

| Kod | İş Kolu | Amaç |
| :--- | :--- | :--- |
| **W0** | Altyapı & Referans | Deney kaydı, referans yeniden üretim, pretrainli omurga altyapısı |
| **W1** | Görüntü Ön-İşleme | Kemik/damar bastırma, normalizasyon, atlas kaydı → **temiz topolojik sinyal** |
| **W2** | Topoloji (TDA) | Çift yön, çok ölçek, arka plan çıkarma, ağırlıklı filtrasyon, OT vektörleştirme |
| **W3** | Mimari & Füzyon | Pretrainli CNN + TDA füzyonu, iki aşamalı candidate-based, ensemble |
| **W4** | Değerlendirme & Yorumlanabilirlik | Robustness, hata analizi, radyolojik eşleme |
| **W5** | Yayın | Ablasyon matrisi, figürler, yazım |

---

## 3. Fazlar ve Çıkış Kriterleri

| Faz | İçerik | Süre (tahmini) | Çıkış kriteri |
| :--- | :--- | :--- | :--- |
| **Faz 0 — Altyapı** | E0.1, E0.2, E0.3 (E0.0 ertelendi) | ~1 hafta ✅ | Registry ✅ · referans birebir ✅ · güçlü baseline kuruldu ✅ — **CPU yeterli çıktı, CUDA gerekmedi** |
| **Faz 1 — Hızlı kazanç (CPU)** | E4.4, E2.7, E2.8, E3.4, E4.1 | ~1–2 hafta | PL mekanizma analizi tamamlandı · en az 1 KEEP |
| **Faz 2 — Füzyon (ana iddia)** | E3.1, E3.2 | ~1–2 hafta | TDA'nın güçlü CNN'e **tamamlayıcı** katkısı val'de CI ile kanıtlı |
| **Faz 3 — Görüntü ön-işleme + topoloji** | E1.1, E1.2, E2.1, E2.2, E2.3, E2.4 | ~3–4 hafta | En az 1 topolojik/görüntü novelty KEEP |
| **Faz 4 — Yüksek risk** | E3.3, E2.5, E1.4, E2.6 | ~3–4 hafta | En az 1 deney başarılı **veya** gerekçeli KILL (öğrenilmiş) |
| **Faz 5 — Yayın + harici doğrulama** | E4.2, E4.3, E5.1, E5.2, E5.3 | ~3 hafta | Ablasyon matrisi tam · taslak metin + figürler |

---

## 4. Deney Backlog'u

Durum: `Planlandı` → `Devam` → `KEEP` / `KILL` / `Parkta`

| ID | İş Kolu | Hipotez (ön-kayıtlı) | Bağımlılık | Efor | Risk | Beklenen Etki | Durum |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **E0.0** | W0 | CUDA PyTorch kurulunca GPU eğitimi tekrar çalışır | — | 0.5g | Düşük | **Kritik** | **TAMAMLANDI** ✅ cu126 |
| **E0.1** | W0 | Deney kayıt protokolü + dizin sözleşmesi kurulabilir | — | 0.5g | Düşük | Altyapı | **KEEP** ✅ |
| **E0.2** | W0 | PI CNN (0.8631) bit-bit yeniden üretilebilir | — | 0.5g | Düşük | Referans | **KEEP** ✅ birebir |
| **E0.3** | W0 | Pretrainli omurga (ResNet/EfficientNet) zon kesitlerinde çalışır | — | 1g | Düşük | Altyapı | **BASELINE** ✅ test 0.8349 |
| **E0.4** | W0 | ResNet18 son-blok ince ayarı donmuş baseline'ı val'de ≥ +0.015 geçer | E0.3 | 1g | Orta | **Yüksek** (geçerlilik) | **KEEP** ⚠️ val 0.9487 ama aşırı uyum + test gerekli |
| **E0.5** | W0 | Düzenlileştirme aşırı uyumu giderir (train loss > 0.05) ve zon AUC ≥ 0.87 kalır | E0.4 | 1g | Orta | Yüksek | **KILL** 🐛 iki koşul birlikte sağlanmadı |
| **E0.6** | W0 | TDA, **ince ayarlı** görüntü modeline de test'te ≥ +0.015 katkı sağlar | E0.4 | 1g | Düşük | **Çok Yüksek** (ana iddia) | **KEEP** 🏆 +0.0232 [0.0132, 0.0338] |
| **E3.4** | W3 | CNN + SVM soft-voting tek modellerden ≥ +0.015 AUC verir | E0.2 | 0.5g | Düşük | Orta | **KEEP** ✅ test 0.8776 |
| **E4.1** | W4 | %10/%20/%30 overlap kurallarında model sıralaması değişmez | E0.2 | 1g | Düşük | Yayın kanıtı | **KILL** ⚠️ %10 +0.0186 > tolerans |
| **E4.6** | W4 | %10 kuralı, çok-seed val doğrulamasında %20'yi ≥ +0.015 geçer | E4.1 | 1.5g | Orta | Yüksek | **KILL** ⚠️ ortalama +0.0085 |
| **E1.1** | W1 | Kemik bastırma H1 kemik gürültüsünü azaltır, val AUC ≥ +0.015 artar | — | 2g | Orta | **Yüksek** | ⛔ **UYGULANAMAZ** (~28 saat CPU) |
| **E1.2** | W1 | Frangi damar bastırma H1 damar döngülerini temizler, AUC ≥ +0.015 artar | — | 2g | Orta | **Yüksek** | **KILL** ⚠️ monoton zarar (−0.0338) |
| **E3.1** | W3 | Pretrainli CNN + TDA geç füzyon > her ikisi tek başına (CI ile) | E0.3 | 1.5g | Düşük | **Çok Yüksek** (ana iddia) | **KEEP** 🏆 test 0.9211 |
| **E3.2** | W3 | TDA özellikleri CNN özellik vektörüne eklenince AUC artar | E0.3 | 1.5g | Orta | **Çok Yüksek** | **KILL** ⚠️ orta seviye −0.0056 |
| **E2.1** | W2 | Çift yön (sublevel+superlevel) filtrasyon tek yönden ≥ +0.015 iyi | — | 1.5g | Düşük | **Yüksek** | **KEEP** ✅ +0.0210 (boyut kontrolü ✓) |
| **E2.2** | W2 | Çok ölçekli PI tek ölçekten ≥ +0.015 iyi | E2.1 | 2g | Orta | Orta-Yüksek | **KILL** ⚠️ +0.0053 < 0.015 |
| **E2.3** | W2 | Sağlıklı şablon çıkarma (relative persistence) AUC artırır | E0.2 | 2.5g | Yüksek | **Yüksek** (novelty) | ⛔ Geçersiz tasarım → **E2.3b KILL** (geçerli) |
| **E2.4** | W2 | Aday-olasılık ağırlıklı filtrasyon H1'i nodüle odaklar | E1.2 | 2.5g | Yüksek | Orta | ⛔ **İPTAL** — bağımlılık E1.2 KILL; E3.6 yerel sinyali zayıf buldu (0.687) |
| **E4.2** | W4 | Kalıcılık özellikleri radyolojik yapılara eşlenebilir | E0.2 | 2g | Orta | **Yüksek** (yayın) | **KILL** ⚠️ H0 ✅ / H1 ❌ |
| **E4.5** | W4 | En kalıcılıklı H1 sınıfları düzgün (düşük-gradyanlı) bölgelerde oluşur | E4.2 | 1.5g | Orta | Orta | ✅ **E4.2 İLE KAPANDI** — H1 medyan gradyan oranı 1.021 (≈rastgele) ⇒ ne kenarda ne düzgün bölgede |
| **E4.3** | W4 | Hangi zonlarda sistematik hata yapıyoruz? | E0.2 | 1g | Düşük | Orta | **KILL** ⚠️ zayıf zon yok |
| **E4.7** | W4 | Modelin maksimum olasılığı gerçek nodül zonuna şanstan sık düşer (lokalite testi) | E4.3 | 1g | Orta | **Yüksek** (geçerlilik) | **KILL** 🚨 lokalizasyon YOK |
| **E3.5** | W3 | Tek global model, 5-zon max-pooling'e denk mi? (zon ayrıştırmasının katkısı) | E4.7 | 2g | Orta | **Yüksek** (kapsam) | **KEEP** ⚠️ eşdeğer (fark −0.0076) |
| **E3.7** | W3 | Ana sonuç (füzyon) 3 seed ortalamasında tek-seed'den <0.03 sapar | E3.1 | 1g | Düşük | **Yüksek** (Kural 10) | **KEEP** ✅ 0.9190 ± 0.0056 |
| **E3.3** | W3 | İki aşamalı candidate-based TDA yanlış pozitifi azaltır | E1.2 | 4g | Yüksek | Yüksek | **İPTAL** ⛔ E3.6: tavan ~0.69 |
| **E2.5** | W2 | Sliced Wasserstein / PersLay, PI'dan ≥ +0.015 iyi | E0.2 | 3g | Yüksek | Orta | **KILL** ⚠️ −0.1504 (boyut-eşit kontrolde de aynı) |
| **E1.4** | W1 | Atlas kaydı arka plan topolojisini tutarlılaştırır | E1.1 | 4g | Yüksek | Orta | 🚫 **ENGELLİ** — bağımlılık E1.1 uygulanamaz (~28 saat CPU, eksik paketler) |
| **E2.6** | W2 | Çok parametreli persistence ek bilgi taşır (`multipers`/MMA olgunluğu doğrulandı) | E2.2 | 5g+ | Yüksek | Orta | **KEEP** 🎉 +0.0480 (en büyük kazanç) |
| **E4.4** | W4 | **PL üst-zarf (envelope) çökmesi nicel olarak gösterilebilir** | E0.2 | 1.5g | Düşük | **Çok Yüksek** | **KEEP** ✅ |
| **E2.7** | W2 | PI ağırlık fonksiyonu (Adams `w=p` vs max-distance/linear/arctan) AUC'yi değiştirir | E0.2 | 0.5g | Düşük | Orta | **KILL** ⚠️ Adams zaten en iyi |
| **E2.8** | W2 | σ / ızgara çözünürlüğü sonuçları anlamlı değiştirmez (sağlamlık kanıtı) | E0.2 | 1g | Düşük | Orta | **KEEP** ✅ iki kol sağlam |
| **E5.3** | W5 | Harici doğrulama (kaynak-dışı genelleme) genellemeyi kanıtlar | E3.1 | 3g | Yüksek | Yüksek | **KEEP** ✅ ortalama 0.8468 |
| **E5.1** | W5 | Tüm ablasyonlar tek matriste tutarlı sunulabilir | Faz 1–4 | 2g | Düşük | Yayın | **TAMAM** ✅ [ablation_matrix.md](ablation_matrix.md) |
| **E5.2** | W5 | Metin + figürler hakem beklentilerini karşılar | E5.1 | 4g | Orta | Yayın | **TASLAK** 📝 [paper_draft.md](paper_draft.md) |

---

## 5. Bağımlılık Grafiği

```
                        ┌──────────────────────────────────────────┐
                        │  FAZ 0 — ALTYAPI (paralel koşabilir)     │
                        │  E0.1 registry · E0.2 referans · E0.3 omurga │
                        └───────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              v                         v                          v
   ┌────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐
   │ W1 GÖRÜNTÜ         │   │ W3 FÜZYON (ana iddia)  │   │ W4 DEĞERLENDİRME │
   │ E1.1 kemik bastır  │   │ E3.1 geç füzyon        │   │ E3.4 ensemble    │
   │ E1.2 damar bastır  │   │ E3.2 orta füzyon       │   │ E4.1 robustness  │
   └─────────┬──────────┘   └────────────────────────┘   │ E4.3 hata analizi│
             │                                           └──────────────────┘
             v
   ┌────────────────────────────────────────┐
   │ W2 TOPOLOJİ (temiz sinyal üzerine)     │
   │ E2.1 çift yön → E2.2 çok ölçek         │
   │ E2.3 arka plan çıkarma (novelty)       │
   │ E2.4 ağırlıklı filtrasyon ← E1.2       │
   └─────────┬──────────────────────────────┘
             v
   ┌────────────────────────────────────────┐
   │ W5 / FAZ 5 — E4.2 yorumlanabilirlik    │
   │ E5.1 ablasyon matrisi → E5.2 yazım     │
   └────────────────────────────────────────┘
```

**Kritik yol:** `E0.3 (omurga) → E3.1 (füzyon)`. Bu, yayının ana iddiasını taşır; öncelik burada.

---

## 6. Öldürme Kriterleri (Kill Criteria)

| ID | Bırakma koşulu |
| :--- | :--- |
| **E1.1 / E1.2** | Yeniden çıkarım sonrası val ROC-AUC artışı < +0.015 **veya** H1 doluluk oranı değişmiyorsa → KILL (36 dk extraction maliyeti haklı değil) |
| **E2.2** | Çok ölçek, tek ölçeğe val'de < +0.015 → KILL |
| **E2.3** | Şablon çıkarma val'de < +0.015 **veya** şablon kontrollerde kararsızsa (yüksek varyans) → KILL |
| **E2.4** | Aday haritası zaten AUC'yi domine ediyorsa (yani TDA katkısı marjinal) → KILL |
| **E2.5** | 3 gün içinde çalışan bir eğitim prototipi yoksa **veya** val < +0.015 → KILL |
| **E2.6** | Hesaplama > 1 saat/zon **veya** 2 haftada prototip yok → Parkta kalır (KILL) |
| **E3.3** | Aday üretici recall < %80 ise (tavan düşük) → KILL |
| **E1.4** | Kayıt sonrası zon hizalaması görsel olarak bozuluyorsa → KILL |

---

## 7. Dizin ve Adlandırma Sözleşmesi

Çok sayıda deneyi karıştırmamak için tek bir sözleşme uygulanır:

```
configs/experiments/<ID>_<slug>.yaml          # deney konfigürasyonu (tek değişken burada)
pipelines/exp/<ID>_<slug>.py                  # deneye özgü betik (gerekirse)
artifacts/runs/<ID>_<slug>/                   # çıktılar (gitignored — kalıcı kayıt experiments.md'de)
    ├── manifest.json                         # git commit + config hash + seed + bağımlılıklar
    ├── val_report.json
    └── test_report.json                      # YALNIZCA faz sonunda
data/processed/features_v3_<slug>/            # yeni ön-işleme varyantlarının özellikleri
```

> **Not:** `artifacts/` ve `data/processed/` git tarafından yok sayılır. Bu yüzden **kalıcı kanıt kaydı `experiments.md` ve bu belgedeki durum tablosudur.** Manifest'ler yerelde tutulur; kritik sayısal sonuçlar mutlaka `experiments.md`'ye yazılır.

---

## 8. Haftalık Ritim

| Gün | İş |
| :--- | :--- |
| Pazartesi | Haftanın deneyini seç, hipotezi + başarı eşiğini backlog'a **ön-kaydet** |
| Salı–Perşembe | Deneyi koştur; ön-işleme varyantlarını gece toplu çıkar |
| Cuma | Sonuçları `experiments.md`'ye yaz, KEEP/KILL kararı ver, roadmap durum tablosunu güncelle |
| Faz sonu | En iyi modeli **bir kez** test setinde değerlendir (test bütçesi §0.3) |

---

## 9. Durum Tablosu (Canlı)

| Faz | Deneyler | Tamamlanan | Aktif | KEEP | KILL |
| :--- | :--- | :---: | :--- | :---: | :---: |
| Faz 0 | E0.0, E0.1, E0.2, E0.3 | **4/4** ✅ | — | 4 | 0 |
| Faz 1 | E4.4, E2.7, E2.8, E3.4, E4.1, E4.6 | **6/6** ✅ | — | 3 | 3 |
| Faz 2 | E3.1, E3.2, E3.7 | **3/3** ✅ | — | 2 | 1 |
| Faz 3 | E1.1, E1.2, E2.1, E2.2, E2.3, E2.4 | 4/6 | E2.3 | 1 | 2 |
| Faz 4 | E3.3, E2.5, E1.4, E2.6 | 3/4 | E1.4 (engelli) | 1 | 2 |
| Faz 5 | E4.2, E4.3, E4.7, E5.1, E5.2, E5.3 | **6/6** ✅ | — | 1 | 3 |
| **Ek deneyler** | E3.5, E3.6, E4.5*, E0.4, E0.5, E0.6, E5.3b | **7** | — | 2 | 4 |

> **Sayım notu (doğrulanmış):** Backlog tablosunda **34 satır** vardır: **12 KEEP/BASELINE**, **2 TAMAM** (E0.0, E5.1), **1 TASLAK** (E5.2), **9 KILL**, **2 İPTAL** (E3.3, E1.1) ve **8 hâlâ "Planlandı"** (E2.1, E2.2, E2.3, E2.4, E4.5, E2.5, E1.4, E2.6).
> ⚠️ **Önceki sayım hatalıydı:** Daha önce "41 satır / 14 KEEP / 16 KILL / 28 deney" diye rapor ettim. İki hata vardı: (a) kill-criteria tablosundaki 3 sütunlu satırlar da backlog deseniyle eşleşiyordu (41 → gerçekte 34), (b) varyantlar (E0.5 tarifleri, E5.3b) ayrı deney sayılmıştı. Doğrusu: **`experiments.md`'de 23 ayrı deney kimliği**; 34 backlog maddesinden **15'i sonuç üretti**, 9'u KILL, 2'si iptal, 8'i planlı.

**Güncel liderlik (hasta düzeyi test ROC-AUC, 734 hasta):**

| Sıra | Model | Test ROC-AUC | Durum |
| :-: | :--- | :---: | :--- |
| 🥇 | **E0.6 Füzyon (ince ayarlı ResNet18 + TDA ensemble)** | **0.9516** (doğruluk %89.4) | KEEP 🏆 |
| 🥈 | E0.6 İnce ayarlı ResNet18 (tek başına) | 0.9284 | baseline |
| 🥉 | E3.1/E3.7 Füzyon (donmuş ResNet18 + TDA) | 0.9190 ± 0.0056 (3 seed) | KEEP |
| 🥈 | E3.4 TDA ensemble (CNN + SVM) | 0.8776 | KEEP |
| 🥉 | TDA PI küçük CNN (v2.1) | 0.8631 | kilitli referans |
| 4 | E0.3 ResNet18 (donmuş) + lojistik regresyon | 0.8349 | baseline |
| 5 | RBF-SVM (PI) | 0.8289 | kilitli referans |
| 6 | k-NN (PI) | 0.7162 | kilitli referans |
| 7 | Ham görüntü küçük CNN | 0.7043 | kilitli referans |
| 8 | PL küçük CNN | 0.5450 | kilitli referans |
| 9 | PL RBF-SVM | 0.5217 | kilitli referans |

> ✅ **TEST SETİ BÜTÇESİ: 6/6 KULLANILDI** (E0.2, E3.4, E0.3, E3.1, E3.7, **E0.6**). Bütçe **tükendi**; son değerlendirme hakem itirazını çürütmeye ayrıldı ve **başarılı oldu**. Bundan sonraki tüm çalışma **validation** üzerinde veya **yeni veri** ile yapılmalı; dondurulmuş test seti artık **kapalıdır**.

---

## 10. Karar Günlüğü

| Tarih | ID | Karar | Gerekçe |
| :--- | :--- | :--- | :--- |
| 2026-10-14 | **E0.1** | **KEEP** | Manifest altyapısı (`src/node21_tda/experiment.py`) + `configs/experiments/` sözleşmesi kuruldu, smoke test geçti. |
| 2026-10-17 | **V4** | **EXTERNAL** 🌐 | **Bağımsız dış doğrulama (VinDr-CXR, 400 GÖRÜNTÜ: 150 nodül/kitle + 250 saf normal) — VinDr'de sıfır eğitim.** TDA dalı dondurulmuş NODE21 bundle'ları; ResNet18 E0.6 tarifiyle **deterministik** yeniden eğitilir (val zon AUC 0.8707; E0.6 kaydı 0.8703; Δ=3,8·10⁻⁴ — E0.6 ağırlıkları hiç kaydedilmemişti) ve VinDr'de yalnızca çıkarım yapılır. Üç kalibrasyon kolu: `linear` TDA **0.5274** / `matched` (200-hasta havuzu, yalnızca dokuya eşleme) TDA **0.5409** / `matched_scalenorm` (akciğer-yüksekliği ölçek normalizasyonu) TDA **0.5110**; ResNet18 üç kolda **0.6204** [0.5609; 0.6763]. **Kilitli füzyon (a=0.45) her kolda ResNet'in ALTINDA** (0.5328/0.5438/0.5352; Δ = −0.088/−0.077/−0.085, CI'lar sıfırı dışlar) ve **harici veride optimal a = 1.00** ⇒ TDA katkısı sıfıra yakın. Ölçek normalizasyonu TDA'yı kurtarmıyor ⇒ çöküş yalnızca yoğunluk/ölçek farkıyla açıklanamaz. Nodül boyutu alt grupları **ölçek-bağımsız** `rel_size` üçte birlikleriyle verilir (mm dönüşümü imkânsız: 3× indirgenmiş JPEG, spacing yok). Donanım teşhisi: yalnızca meta-veri val AUC **0.6850** (kısmi kısayol); eşit-boyut kaynak-dışı TDA 0.7631 ⇒ sinyal tamamen sahte değil. **Sonuç:** v3'ün 0.9516'lık kurum-içi sonucu geçerli, ancak **kapsamı kurum içi**; sistem lezyon kutulayıcı değil, **kalibre edilmesi gereken kurum-içi risk triyaj** altyapısıdır. Kayıtlar: `experiments.md §7`, `paper_draft.md §4.8`. |
| 2026-10-14 | **E0.2** | **KEEP** | v2.1 PI küçük CNN referansı **bit-bit yeniden üretildi** (16/16 metrik; test ROC-AUC 0.8631, TP/FP/TN/FN 137/114/448/35). Pipeline deterministik. Yan bulgu: eğitim CPU'da ~27 s → özellik tabanlı deneyler GPU'suz yapılabilir. |
| 2026-10-14 | **E4.4** | **KEEP** | **PL çöküşünün mekanizması nicel olarak kanıtlandı.** H1 zarf tepeleri λ1→λ5 boyunca 110→90 (birbirine yakın, enerji eşit dağılmış) → zarf, her zonda ≥5 çok uzun ömürlü arka plan H1 sınıfıyla **doymuş**. Poz/neg sinyal zarfın yalnızca **%1–2'si** (≈50–100× bastırılma); H0'da λ1'de %0.02. Bu, yayının **1 numaralı özgünlük iddiasının** deneysel kanıtı. |
| 2026-10-14 | **E3.4** | **KEEP** | **Yeni lider: ensemble test ROC-AUC 0.8776** (w=0.55, hasta düzeyi soft voting). Val farkı +0.0214 (≥0.015 ✓), test farkı +0.0145, bootstrap %95 CI **[+0.0034, +0.0260]** sıfırı dışlıyor, p=0.010. NODE21 rank-1'i (0.868) geçiyor, NODE21 ensemble'ına (0.877) ulaşıyor. Zon düzeyi füzyon (varyant B) daha kötü (0.8629) → CNN ve SVM tamamlayıcı. |
| 2026-10-14 | **E4.2** | **KILL** ⚠️ | Ön-kayıtlı H1 hipotezi **çürütüldü**: en kalıcılıklı H1 sınıflarının gradyan oranı medyan **1.021** (kenarlarda değil), rastgele kontrol 0.981. **Ancak H0 güçlü:** oran **6.176** (medyan 6.608), zonların **%90'ında** >1.5 → H0 PI ağırlıklı olarak **arka plan/sınır** kodluyor. Ayrıca persistence→piksel geri izleme **yeteneği** kuruldu (literatürdeki boşluk). Takip deneyi E4.5 kaydedildi. |
| 2026-10-14 | **E0.3** | **BASELINE** ✅ | Güçlü aynı-split baseline kuruldu: **donmuş ImageNet ResNet18 + lojistik regresyon**, test ROC-AUC **0.8349** (val 0.8884 → **−0.0535 genelleme kaybı**). TDA CNN (0.8631) ve ensemble (0.8776) bu baseline'ı **test'te geçiyor**. Hakemin "neden güçlü CNN değil?" itirazı kapatıldı. CPU'da 849 s. |
| 2026-10-15 | **E0.4** | **KEEP** ⚠️ | **ResNet18 son-blok ince ayarı (GPU, 5 s/epoch — 41× hızlanma).** Val hasta AUC **0.9487** vs dondurulmuş 0.8884 ⇒ **+0.0603** (eşik ✓). **AMA ciddi aşırı uyum:** eğitim kaybı 0.5894→0.0012 (ezberleme), zon AUC 0.859'da plato. Hasta AUC'sindeki sıçrama muhtemelen **max-pooling + aşırı özgüven** artefaktı; ayrıca ince ayarlı model val'de 15 checkpoint'le seçildiği için val karşılaştırması **iyimser**. 🚨 **Stratejik risk:** val'de ince ayarlı görüntü modeli (0.9487) füzyona (0.9572) çok yakın ⇒ eğer test'te de tutarsa **"TDA tamamlayıcıdır" iddiası zayıflar** (füzyon kazancı zayıf baseline'dan geliyor olabilir). **Son test hakkı bu soruya ayrılmalı.** |
| 2026-10-15 | **E0.5** | **KILL** 🐛 | **Düzenlileştirilmiş ince ayar taraması.** 🐛 **Önce geçersiz sonuç:** önbellek E0.4'ten **9380** satır (test dahil), E0.5 ise **5710** satır kullanıyordu → görüntüler etiketlerle **kaydı**, model şans düzeyinde kaldı (val AUC 0.58). Boyut koruması eklendi. **Düzeltilmiş sonuç:** `mild` (hflip, wd=1e-4, do=0.3) train loss 0.0012 / zon AUC **0.8706** / hasta **0.9546**; `full` (artırma+wd=1e-3) train loss **0.1247** ✓ ama zon AUC **0.8439** / hasta 0.9368. ⇒ Hiçbiri iki koşulu birlikte sağlamadı (**KILL**), ama **hasta val AUC üç tarifte de 0.937–0.955** ⇒ ince ayar kazancı tarife duyarsız ve füzyona çok yakın ⇒ **E0.6 zorunlu.** |
| 2026-10-15 | **E0.6** | **KEEP** 🏆 | **BELİRLEYİCİ TEST (son test hakkı) — ana iddia güçlü baseline karşısında AYAKTA.** İnce ayarlı ResNet18 tek başına test **0.9284** (donmuş 0.8349, **+0.0935**); TDA ensemble 0.8629; **FÜZYON (a=0.45) test 0.9516, doğruluk %89.4**. **TDA katkısı +0.0232, %95 CI [+0.0132, +0.0338], p=0.0000** ⇒ füzyon kazancı **zayıf baseline artefaktı DEĞİL**; literatürde öngörülen **1 numaralı hakem itirazı kanıtla çürütüldü.** ⚠️ Küçük tutarsızlık **çözüldü**: TDA-tek 0.8629 = E3.4'ün *zon düzeyi füzyon* varyantı; hasta düzeyi varyant 0.8776 olup **TDA-tek için esas alınan değerdir** (E0.6 füzyonu dolayısıyla **muhafazakâr**). **TEST BÜTÇESİ TÜKENDİ (6/6).** |
| 2026-10-15 | **E3.2** | **KILL** ⚠️ | **Orta seviye füzyon (özellik birleştirme) vs geç füzyon** — val-only, test kullanılmadı. **Geç füzyon 0.9538** (a=0.60) · **orta seviye (PI ⊕ ResNet → LR) 0.9482** ⇒ fark **−0.0056**, kriter ≥ +0.015 ✗. ⇒ **Özellik düzeyinde birleştirme karar düzeyinden iyi değil**; sebep muhtemelen TDA dalındaki **doğrusal olmayan** bileşenin (CNN+RBF-SVM) karar füzyonunda korunması. **Geç füzyon tasarımı doğrulandı.** ⚠️ Küçük yeniden üretilebilirlik notu: TDA ensemble val bu koşuda 0.8699, E3.4'te 0.8821 — bundle yeniden yüklemesi tam örtüşmüyor; **TDA-tek için E3.4 değeri esas.** **Faz 2 tamamlandı (3/3).** |
| 2026-10-16 | **E2.5** | **KILL** ⚠️ | **Sliced Wasserstein gömlemesi vs Kalıcılık İmgesi** — val-only, test kullanılmadı. H0/H1 diyagramları yeniden hesaplandı (önbellekte yoktu); K=96 çapaya SW mesafesi gömlemesi (192 boyut) + RBF-SVM **0.6896** vs PI (800 boyut) + aynı SVM **0.8399** ⇒ **fark −0.1504**, kriter ≥ +0.015 ✗. ⇒ **Izgara kuantalaması PI'ın darboğazı değil**; E2.8'in ızgara duyarlılığı "ızgara bilgi kaybediyor" demek değilmiş. ⚠️ Kapsam sınırı: bu deney *çapa-mesafesi* SW gömlemesini test eder, saf SW çekirdeğini değil. **PI temsili korunur.** Not: E2.5 GPU'dan faydalanamaz (GUDHI + sklearn CPU-only). |
| 2026-10-16 | **E2.1** | **KEEP** ✅ | **Çift yönlü (diphasic) filtrasyon — programın ilk gerçek temsil iyileştirmesi.** Tek polarite (projenin PI'ı) **0.8399** → çift polarite (sublevel + superlevel, 1600 boyut) **0.8609** ⇒ **+0.0210** (eşik ✓), val-only. 🚨 **İlk koşu confound'luydu (+0.0646 şişkin):** baseline'ı bu betikte yeniden türetmiştim (yüzdelik aralıklar) → 0.7198'e düştü; projenin **donmuş aralıkları** (H1 birth **[239,250]**) bilgiyi doğru banda odaklıyor. **Ders: baseline'ı yeniden türetmek kazancı şişirir.** 🎯 **Boyut kontrolü (E2.1b):** B kolunu karıştırınca (boyut aynı 1600, bilgi yok) sonuç **0.7803** — PI'ın **altında** ⇒ boyut eklemek tek başına **zarar veriyor**; kazanç **ters polaritenin bilgisinden**. ⚠️ **Test bütçesi tükendi** ⇒ 0.9516'lık ana sonuca eklenemez; **gelecek test bütçesi için öncelikli aday**. |
| 2026-10-16 | **E2.2** | **KILL** ⚠️ | **Çok ölçekli PI** — val-only, test kullanılmadı, **özellikler E2.8 önbelleğinden** (yeniden çıkarım yok). Tek ölçek σ=1.0 **0.8399** · σ=1.5 (en iyi) **0.8457** · **çok ölçek (4 σ, 3200 boyut) 0.8510** ⇒ **+0.0053**, kriter ≥ +0.015 ✗. 🎯 **Boyut kontrolü** (diğer ölçekler karıştırılmış, 3200 boyut) **0.6598** ⇒ bilgisiz boyut eklemek **şiddetle zararlı**; +0.0053 gerçek ama yetersiz bilgi. 🔑 **E2.1 ile karşıtlık:** **polarite +0.0210 (KEEP)** vs **ölçek +0.0053 (KILL)** ⇒ **topolojik kazanç "hangi kümeleri filtrelediğinden" geliyor, "diyargramı nasıl yumuşattığından" değil.** Temsil tasarımı için yön gösterici. **Faz 3 → 4/6.** |
| 2026-10-16 | **E2.3** | ⛔ **GEÇERSİZ** | **Sağlıklı şablon (relative persistence) — tasarım no-op çıktı, hipotez test EDİLEMEDİ.** Üç kol birebir aynı: mutlak **0.8399**, `X − T(doğru zon)` **0.8400**, `X − T(yanlış zon)` **0.8399**. Sebep matematiksel: scaler **zon başına** uygulanıyor ve sabit vektör çıkarmak yutuluyor — `(X−T) − mean(X−T) = X − mean(X)`. ⇒ **Per-feature afin dönüşümler StandardScaler tarafından absorbe edilir.** 📌 **Ders:** bir ön-işleme adımını ölçmeden önce, sonraki adımın onu **matematiksel olarak** yutup yutmadığı kontrol edilmeli. ✅ **Düzeltilmiş tasarımlar:** (1) doğrusal olmayan sapma `(X−T)_+`, (2) diyagram düzleminde relativasyon, (3) havuzlanmış sınıflandırıcı (protokolü değiştirir). **KILL olarak kaydedilmedi; hipotez açık.** |
| 2026-10-16 | **E2.3b** | **KILL** ✅ | **E2.3'ün düzeltilmiş tasarımı — hipotez ilk kez GERÇEKTEN test edildi.** Doğrusal olmayan sapma kullandım (ReLU `(X−T)₊`, mutlak `|X−T|`); bunlar afin **olmadığı** için StandardScaler yutamaz. Sonuç: mutlak PI **0.8399** · ReLU sapma **0.7530** (−0.0869) · mutlak sapma **0.6365** (−0.2034) · kontrol (yanlış zon) **0.8276** (−0.0124). ⇒ **Şablon-göreli sapma bilgi katmıyor, yok ediyor.** 🔑 **Kontrol kolu asıl bulgu:** *yanlış* zonun şablonunu çıkarmak (0.8276) *doğru* zonunkinden (0.7530) **daha iyi** ⇒ **şablon zon-özel FAYDALI bilgi taşıyor** (ortak arka plan gürültüsü değil). 🔗 **E1.2 + E2.3b:** "arka planı bastır, lokal sinyali açığa çıkar" öncülü **iki bağımsız yoldan çürütüldü**. Hipotez **kapandı**. |
| 2026-10-16 | **E2.6** | **KEEP** 🎉 | **Çok parametreli persistence (dilim yaklaşımı) — programın EN BÜYÜK val kazanımı.** `f_t = f_norm + t·g_norm` (g = Sobel gradyan), t ∈ {0, 0.5, 1.5}. Val-only, test kullanılmadı. Proje PI'ı (tek dilim) **0.8399** → **proje PI + 2 gradyan dilimi (2400 boyut) 0.8880** ⇒ **+0.0480** (eşik ✓). 🎯 **Boyut kontrolü** (dilimler karıştırılmış) **0.6925** — A'nın çok altında ⇒ kazanç **gerçek bilgiden**. 🚨 **İlk koşu şişkindi (+0.0883):** t=0 kolunu bu betikte türetmiştim → baseline 0.7253; **proje PI'ı** ile fark **+0.0480**. **Ders tekrarlandı: baseline'ı yeniden türetmek kazancı şişirir.** 🔑 **Kritik içgörü:** gradyanı **ikinci parametre yapmak işe yarıyor**, ama gradyanla **sınıf elemek** (E1.2) zarar veriyordu ⇒ **elemek değil, KATMANLAŞTIRMAK** doğru strateji. ⚠️ Test bütçesi kapalı ⇒ 0.9516'ya eklenemez; **gelecek test için E2.1 ile birlikte öncelikli aday (hatta daha güçlüsü).** **Faz 4 → 3/4.** |
| 2026-10-16 | **TDA-BUNDLE** | ✅ **ÇÖZÜLDÜ** | **TDA-tek tutarsızlığı kök nedeni bulundu — hata yok, varyant farkı.** E3.4 raporu iki varyant saklıyor: **Varyant A** (iki dalı *önce* hasta düzeyine max-pool, *sonra* birleştir) → val 0.8821 / **test 0.8776**; **Varyant B** (önce zon düzeyinde birleştir, sonra max-pool) → val **0.8699** / test **0.8629**. E0.6 ve E3.2 koddan **B'yi** yeniden üretmiştir → **birebir aynı (val 0.8699, test 0.8629)**. İkisi de **w=0.55** ⇒ **ağırlık değil, varyant farkı**; **kayıtlı bundle'lar E3.4'ü tam yeniden üretiyor** (yeniden üretilebilirlik sorunu **YOK**). 📌 **Sonuç:** E0.6 füzyonu (0.9516) TDA'nın **zayıf varyantıyla** ölçülmüş ⇒ **muhafazakâr**; TDA katkısı **+0.0232** bir **alt sınırdır**. Gelecek test değerlendirmesinde **Varyant A** kullanılmalı. |
| 2026-10-16 | **KAPANIŞ** | ✅ | **Program kapanışı — kalan 4 maddenin tümü gerekçeli karara bağlandı.** **E2.4** ⛔ İPTAL (bağımlılık E1.2 KILL; E3.6 yerel sinyali zayıf buldu) · **E4.5** ✅ E4.2 ile KAPANDI (H1 medyan gradyan oranı 1.021 ≈ rastgele) · **E1.4** 🚫 ENGELLİ (bağımlılık E1.1 uygulanamaz) · **E2.6** ⏸️ ERTELENDİ (E2.2 yalnızca +0.0053; test bütçesi kapalı ⇒ val-only). ⇒ **Artık hiçbir backlog maddesi "Planlandı" durumunda değildir.** Test bütçesi 6/6 tükenmiş, dondurulmuş test seti kapalıdır; ana sonuç **füzyon test ROC-AUC 0.9516 / %89.4 doğruluk**, val düzeyinde doğrulanmış tek iyileştirme adayı **E2.1 çift yönlü filtrasyon (+0.0210)**. |
| 2026-10-16 | **E2.5b** | **KILL** 🎯 | **Boyut confound kontrolü.** İlk karşılaştırmada PI **800**, SW gömlemesi **192** boyuttu (E3.6'daki boyut confound'unun aynısı). **PI, PCA ile 192 boyuta indirildi: val AUC birebir aynı (0.8399 → 0.8399)** ve boyut-eşit fark **yine −0.1504**. ⇒ **Boyut confound'u YOKTUR; fark gerçekten temsilden gelir.** 🎁 **Bonus:** PI'ın 800 boyutu **fazlasıyla yedekli** — 192'ye indirmek hiç kayıp vermiyor (hesaplama/depolama notu). Kontrol raporda `dimensionality_control` alanı olarak kayıtlıdır. |
| 2026-10-15 | **E5.3b** | **KEEP** ⚠️🎯 | **Kaynak-dışı genellemede eşit-boyut kontrolü.** Katların dengeli train boyutları farklıydı (1034/1876/2222 zon) ⇒ önceki 0.8468 **iyimserdi**. Tüm katlar **1034 zona** eşitlendi: chestxray14 0.7821 (değişmedi) · padchest 0.8499 → **0.7625** (−0.0874) · jsrt 0.9085 → **0.8847** (−0.0238) ⇒ **ortalama 0.8098**, domain-shift bedeli **−0.109** (önceki tahmin −0.072). ✅ Genelleme yine de gerçek (eşiğin üstünde), ancak **makalede raporlanacak sayı 0.8098'dir.** 📌 **Metodolojik ders:** kaynak-dışı genelleme çalışmalarında eğitim verisi miktarı mutlaka eşitlenmelidir. |
| 2026-10-14 | **E3.1** | **KEEP** 🏆 | **Yayının ana sonucu: füzyon test ROC-AUC 0.9211** (ResNet 0.8349 + TDA 0.8776, a=0.55). Test farkı **+0.0862**, bootstrap %95 CI **[+0.0632, +0.1099]**, **p≈0.0000**. **Bağımsız doğrulama:** iki temsil arasındaki korelasyon yalnızca **r≈0.30–0.35** (neredeyse dik) → tamamlayıcılık gerçek; etiket permütasyonu AUC=0.4378 → **sızıntı yok**. GPU gerekmedi. |
| 2026-10-14 | **E0.0** | **TAMAMLANDI** ✅ | Kullanıcı CUDA PyTorch'u kurdu: **`torch 2.14.1+cu126`** + `torchvision 0.29.1+cu126` (CPU sürümleriyle **birebir aynı sürüm**, yalnızca derleme farkı). Doğrulandı: `cuda.is_available()=True`, **RTX 4050 Laptop (6 GB VRAM, 20 SM)**. ⇒ Ağır görüntü işleri artık CPU'yu kilitlemeyecek; E0.4 gibi ince ayar deneyleri GPU'da dakikalara inecek. |
| 2026-10-14 | **E2.7** | **KILL** ⚠️ | Ağırlık fonksiyonu karşılaştırması (val-only, 5.710 zon, **test kullanılmadı**): **Adams `p` zaten en iyi (0.8736)**. Farklar hep negatif — arctan −0.0096, √p −0.0112, sabit −0.0145, **p² −0.0189**. `p²`'nin en kötü olması **E4.4'ü bağımsız destekliyor**. Mevcut PI tasarımı **doğrulanmış** sayılır. |
| 2026-10-14 | **E4.1** | **KILL** ⚠️ | Overlap sağlamlığı (hasta split'i dondurulmuş, **test kullanılmadı**): **%10 = 0.8922 (+0.0186)**, %20 = 0.8736, %30 = 0.8693 (−0.0043). En büyük fark **0.0186 ≥ tolerans 0.015** ⇒ sonuçlar kuraldan bağımsız sağlam **değil**; monoton eğilim var (gevşek eşik → iyi). ⚠️ Tek seed; **%10'a geçiş çok-seed doğrulaması gerektirir** → yeni deney **E4.6**. Yan kanıt: %20 sonucu E2.7 ile birebir aynı (0.8736) ⇒ PI rasterleştirmesi doğrulandı. |
| 2026-10-14 | **E4.6** | **KILL** ⚠️ | Eşleştirilmiş 5 seed: ortalama fark **+0.0085** (std 0.0096), pozitif seed 4/5. Eşik +0.015 **aşılamadı** ⇒ E4.1'in +0.0186'sı **gürültü artefaktı**ydı (seed 42 beşin en büyüğü). **%20 kuralı korunur.** 📌 Yan bulgu: CNN seed varyansı **±0.02–0.03** (val 0.8321–0.8922) → makalede ana sonuçlar için **çok-seed ortalama** raporlanmalı; füzyon kazancı (+0.0862) bu gürültünün çok üzerinde olduğu için iddia sağlam. |
| 2026-10-14 | **E2.8** | **KEEP** ✅ | σ/ızgara duyarlılığı (3 seed ortalaması, **test kullanılmadı**, tolerans 0.03). Kol A (σ): 0.5→0.8460 (**−0.0236**), **1.0→0.8697**, 1.5→0.8734 (+0.0038), 2.0→0.8712. Kol B (ızgara): 15→0.8545 (−0.0152), **20→0.8697**, 25→0.8469 (−0.0227). Her iki kol da **sağlam** ⇒ sonuç σ/ızgara seçimine aşırı duyarlı değil. Mevcut σ=1.0 / 20×20 korunur. **Faz 1 tamamlandı (6/6).** |
| 2026-10-14 | **E4.3** | **KILL** ⚠️ | Zon bazlı hata analizi (val-only, yeni model eğitilmedi): **zayıf zon yok** — her zon tek başına 0.74–0.84 hasta düzeyi AUC veriyor. Karar 5 zona yayılıyor (tek-zon hakimiyeti yok); FN'ler "kıl payı" (ort 0.581 vs eşik 0.636). 📌 **Kritik çıkarım:** zon-bazlı sinyalin önemli kısmı **hasta-global** olabilir (saf lokal olsaydı nodülsüz zonlar şans düzeyinde kalırdı) → yeni **E4.7 lokalite testi** kaydedildi; makalede sınırlama olarak belirtilmeli. |
| 2026-10-14 | **E4.7** | **KILL** 🚨 | **Lokalite testi — modeller lokalize ETMİYOR** (val, 171 nodül-pozitif hasta, **test kullanılmadı**). İsabet oranı şans düzeyinde (0.228–0.298 vs şans 0.268; binom p=0.21–0.90) ve **zon düzeyi AUC yalnızca 0.48–0.53**. ⇒ Hasta düzeyi 0.9211 sonucu **gerçek** ama mekanizma "nodülü doğru zonda bulmak" **değil**. **"5-zon lokalizasyon" yenilik iddiası DÜŞTÜ**; doğru ifade "5-zon paralel ayrıştırma + max-pooling". Geçerli kalan katkılar: füzyon (E3.1), tamamlayıcılık (r≈0.30), PL mekanizması (E4.4). Yeni aday **E3.5** (global model vs 5-zon) kaydedildi. |
| 2026-10-14 | **E3.5** | **KEEP** ⚠️ | **Havuzlanmış tek model vs 5-zon özel modeli** (3 seed, val-only, **test kullanılmadı**): 0.8585 vs 0.8661 → fark **−0.0076** < tolerans 0.015 ⇒ **eşdeğer**. Havuzlanmış model model başına **5× veri** görüyor ama daha iyi değil ⇒ zon bölünmesi **model topluluğu çeşitliliği** sağlıyor, **anatomik uzmanlaşma** değil. ⚠️ E4.7 + E3.5 birlikte: mimari *"5 kırpıntı üzerinde model topluluğu + max-pooling"* olarak, **lokalizasyon/uzmanlaşma iddiası olmadan** sunulmalı. |
| 2026-10-14 | **E1.1** | ⛔ **UYGULANAMAZ** | Kemik bastırma (Qure.ai) — README CPU'da **~1 dk/görüntü**; gereken ~1700 hasta ⇒ **~28 saat**; ayrıca `cv2`/`huggingface_hub` kurulu değil. Deney KILL değil **ertelendi** (GPU gelirse yeniden değerlendirilir). Yerine E1.2 uygulandı. |
| 2026-10-14 | **E1.2** | **KILL** ⚠️ | Frangi vesselness ile H1 filtrelemesi (3 seed, val-only, **test kullanılmadı**): filtreleme **monoton zarar veriyor** — base 0.8697 → v90 0.8606 → v75 0.8547 → v50 **0.8358** (−0.0338). ⇒ **Damar kaynaklı H1 sınıfları gürültü değil, faydalı.** 📌 **Üç deney tutarlı bir sonuca bağlanıyor:** E4.4 (zarf arka planla doygun) + E4.7 (lokalizasyon yok) + E1.2 (arka planı kesmek zararlı) ⇒ **ayırt edici bilgi lokal nodül topolojisinde değil, global/arka plan topolojisindedir.** "Arka planı bastırınca lokal sinyal açılır" beklentisi **çürütüldü**. |
| 2026-10-14 | **E3.6** | **KILL** 🎯 | **Oracle-lokalize TDA testi** (ground-truth bbox ile nodül merkezine oturan yama vs **boyut-eşleştirilmiş** kontrol; train+val, **test kullanılmadı**): val yama AUC yalnızca **0.6866**. 🚨 Confound uyarısı: eşleştirilmemiş ilk koşuda kontroller sabit 60×60 iken pozitifler değişkendi → sınıflandırıcı **yama boyutunu** öğrendi ve AUC **0.994** çıktı; boyut eşleştirilince **0.687**'ye düştü. ⇒ **Nodülün tam yerini bilsek bile lokal TDA ayrımı zayıf** ⇒ **aday-bazlı mimarinin (E3.3) tavanı düşük, İPTAL.** Hasta düzeyi 0.9211 ağırlıklı olarak **global sinyalden** geliyor. **Seçenek A** (daraltılmış dürüst kapsamla yayın) benimsenir. |
| 2026-10-14 | **E3.7** | **KEEP** ✅ | **Ana sonucun 3-seed doğrulaması (Kural 10 uyumu)** — test seti seed belirsizliğini ölçmek için bir kez kullanıldı. Seed 42/43/44 → test ROC-AUC **0.9238 / 0.9111 / 0.9222** ⇒ **0.9190 ± 0.0056** (tek-seed 0.9211'e fark **−0.0021**). Füzyon ağırlığı üç seed'de de **a=0.55**. TDA-tek dal **0.8750 ± 0.0079**. ⇒ Tek-seed sonucumuz **şanslı değildi**; **makalenin ana sayısı artık 0.9190 ± 0.0056**. |
| 2026-10-14 | **E5.3** | **KEEP** ✅ | **Kaynak-dışı genelleme (leave-one-source-out)** — dondurulmuş test kullanılmadı; her kat kendi split'ini kurar. Füzyon: chestxray14 **0.7821**, padchest **0.8499**, jsrt **0.9085** ⇒ ortalama **0.8468** (rastgele split 0.9190, **−0.072 domain-shift bedeli**). ⇒ Model görülmemiş kaynağa genelliyor. ⚠️ Kısmi confound: chestxray14 (en büyük kaynak) dışarıda kalınca dengeli train 1034 zona düşüyor (diğer katlarda 1876/2222) → düşüşün bir kısmı veri azalması. |

---

## 11. Literatür Revizyonu (v3.1 — 2026-10-14)

Dört kolda yapılan literatür taraması ([literature.md](literature.md)) planı şu noktalarda değiştirdi:

| Değişiklik | Gerekçe |
| :--- | :--- |
| **§0'a kural 9 (SPLIT BEYANI) eklendi** | Verimizin NODE21 halka açık training seti olduğu **yerel olarak doğrulandı** (4882 = 1134 + 3748) → leaderboard karşılaştırması geçersiz |
| **E4.4 (PL üst-zarf mekanizması) eklendi → Faz 1** | Literatürde **en güçlü novelty**; ucuz (mevcut PL verisi + görselleştirme) |
| **E2.7, E2.8 eklendi → Faz 1** | Moon et al. ağırlık fonksiyonu + 625-kombinasyon σ taraması → ucuz sağlamlık kanıtı |
| **E1.1/E1.2 Faz 1 → Faz 3'e taşındı** | Yeniden özellik çıkarımı gerektiriyor (pahalı); ucuz topolojik analizler öne alındı |
| **E2.6 (MPH) "Parkta"dan çıkarıldı** | `multipers`/MMA olgunluğu doğrulandı → risk düşük-orta |
| **E5.3 (harici doğrulama) eklendi** | Hakem genelleme kanıtı isteyecek |
| **E0.3 → E3.1/E3.2 kritik yolda pekişti** | Literatür "aynı split'te güçlü baseline"yı **zorunlu** kılıyor |

### En önemli dört literatür sonucu

1. **Skorumuz artık açık ara güçlü.** E3.4 ensemble **0.8776**, E3.1 füzyon **0.9211** hasta düzeyi test ROC-AUC. NODE21 rank-1 (0.868) ve ensemble (0.877) **geçildi**; güçlü ResNet18 baseline'ı (0.8349) hem TDA tek başına hem füzyon geçiyor.
2. **PL'nin çökmesi en güçlü özgün iddiamız.** Literatür PL'yi "kararlı" diye övüyor; bizim 0.5450'ye çöküşümüz tipik beklentinin tersi ve üst-zarf mekanizmasıyla açıklanabilir → **E4.4** bunu kanıta çevirmeli.
3. **Kemik/damar bastırma + kalıcı homoloji kombinasyonu literatürde yok** → Ana özgünlük iddiası. Topo-CXR (ICCVW 2023) ise tam tersini (*"ön-işleme gerekmez"*) savunuyor → net karşı-pozisyon.
4. **Görev/metrik uyumu:** NODE21'in AUC'si zaten görüntü düzeyi + **max agregasyon**; bizim hasta skorumuz da max-pooling. Çerçeveleme: *"yorumlanabilir ve hafif bir topolojik temsil, ağır detektör topluluklarının görüntü düzeyi AUC seviyesine yaklaşıyor."*

### Yenilik iddia EDİLMEMESİ gerekenler

"İlgisiz yapılar diyagramı domine eder → ROI kısıtlaması gerekir" (Vandaele 2020/2022) · "PI ve PL veri setine bağlı farklı davranır" (Barnes 2021, Sulowska 2026) · "topoloji tek başına CNN'e denk, füzyonla artar" (TDA-Net) · "topoloji gerçek histolojiyi, radyomik radyolog skorunu taklit eder" (Vandaele 2022).

---

## 12. Devam Noktası (Handoff — güncellendi 2026-10-14, round 4)

**Durum:** Faz 0 ✅ tamam · Faz 1 **5/6** (yalnız E2.8 kaldı) · Faz 2 kısmi (1/2) · Faz 5 kısmi (1/5, 1 KILL).
Çalışma ağacı **temiz**, tüm iş **commit'li**. Deney manifest'leri `artifacts/runs/` altında (gitignored, yerelde mevcut).

**Ana sonuç (yayının çekirdeği):** E3.1 füzyonu → **0.9211** hasta düzeyi test ROC-AUC (NODE21 ensemble'ı +0.044 geçiyor).

**Kapanan deneyler:** E0.1 ✅ · E0.2 ✅ · E4.4 ✅ · E3.4 ✅ · E0.3 ✅ (baseline) · E3.1 ✅ · E4.2 ⚠️ · E2.7 ⚠️ · E4.1 ⚠️ · E4.6 ⚠️

**Devam edilecek ilk işler** — test bütçesi harcanmadan, yalnızca validation'da karar verilecek:

| Öncelik | Deney | Not |
| :-: | :--- | :--- |
| 1 | **E2.8** — σ/ızgara duyarlılık taraması | **Faz 1'in kalan tek maddesi.** Yeniden çıkarım gerekir (~10 dk/varyant) |
| 2 | **E2.1/E2.2** — çift yön filtrasyon + çok ölçek (Faz 3) | Yeni topolojik katkı arayışı; yeniden çıkarım gerekir |
| 3 | **E2.3** — SEDT-3 / sağlıklı şablon filtrasyonu | Literatürde boşluk; kemik maskesiyle H1 süzme |
| 4 | **E4.5** — E4.2 takibi (H1 düzgün bölgelerde mi oluşuyor?) | Temsilci döngü hücreleri gerekir |
| 5 | **E3.2** — orta seviye füzyon (özellik birleştirme) | E3.1 geç füzyonu kanıtladı; bu derinleştirme |
| 6 | **E4.3** — zon bazlı hata analizi | Ucuz, yayın için değerli |

**Test bütçesi: 6/6 TÜKENDİ.** Dondurulmuş test seti **kapalıdır**; son değerlendirme hakem itirazını (zayıf baseline) çürütmeye ayrıldı ve **başarılı oldu** (E0.6). Kalan işler **E5.2 yazım** ve test harcamayan analizlerdir.

⚠️ **Yeni protokol kuralı 10 (çok-seed raporlama):** E4.6'da CNN seed varyansı **±0.02–0.03** ölçüldü → makalede model performansları **≥3 seed ortalaması** olarak raporlanmalı.

**Ortam notları:**
* ✅ `torch 2.14.1+cu126` — **CUDA AKTİF** (RTX 4050, 6 GB VRAM). E0.0 **tamamlandı**; ağır görüntü işleri GPU'da, CPU serbest.
* Pretrained ağırlıklar indirildi: `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`.
* `data/processed/features_resnet18/X_zone_features.npy` (24410, 512) — E3.1/E3.2 tarafından yeniden kullanılabilir.
* `data/processed/features_weights/E2_7_zone_features.npy` — 5 ağırlık varyantı (test dışı zonlar). *Not: bu önbellek arşiv temizliğinde silindi; E2.7 kayıtları `experiments.md`'de duruyor.*
* Tüm deney betikleri: `pipelines/exp/`.

**Devam komutu:** Program **kapandı** (bkz. **§13 Program Kapanışı**). Yeni çalışma yalnızca **validation** veya **yeni veri** ile yapılabilir; dondurulmuş test seti kapalıdır.

---

## 13. Program Kapanışı (2026-10-16)

### 13.1 Nihai sayım — **hiçbir madde "Planlandı" değil**

| Kategori | Sayı |
| :--- | :---: |
| **Backlog maddesi (toplam)** | **34** |
| KEEP / BASELINE | 14 |
| TAMAM (E0.0, E5.1) | 2 |
| TASLAK (E5.2) | 1 |
| KILL | 12 |
| İPTAL / ENGELLİ (E3.3, E1.4) | 2 |
| GEÇERSİZ tasarım (E2.3) | 1 |
| KAPANDI (E4.5 — E4.2 ile) | 1 |
| İPTAL — bağımlılık geçersiz (E2.4) | 1 |

`experiments.md`'de **28 ayrı deney kimliği** belgelenmiştir (14 KEEP · 13 KILL · 1 uygulanamaz). Not: "34 backlog maddesi" ile "28 deney kimliği" **farklı sayımlardır** — backlog iptal/geçersiz/ertelenen maddeleri ve alt-varyantları ayrı satır sayar.

### 13.2 Faz durumu

| Faz | Durum |
| :--- | :--- |
| **Faz 0 — Altyapı** | ✅ **4/4** (+ E0.4/E0.5/E0.6 ince ayar & belirleyici test) |
| **Faz 1 — Ön-işleme** | ✅ **6/6** |
| **Faz 2 — Füzyon** | ✅ **3/3** |
| **Faz 3 — Topoloji** | 4/6 (E2.4 iptal · E2.3 geçersiz tasarım) |
| **Faz 4 — Yüksek risk** | 2/4 koşuldu (E2.5 KILL · **E2.6 KEEP**) · E3.3 iptal · E1.4 engelli |
| **Faz 5 — Yayın** | ✅ **6/6** (ablasyon matrisi + makale taslağı) |

### 13.3 Teslim edilebilirler — hepsi tamam

| Teslim | Durum |
| :--- | :--- |
| Her deneyin val/test sonucu kayıtlı (`experiments.md` + durum tablosu) | ✅ |
| Tek-değişken ve ön-kayıt protokolü uygulandı | ✅ |
| Öldürme kriterleri uygulandı (12 KILL + 1 iptal + 1 geçersiz) | ✅ |
| **Yayınlanabilir ablasyon matrisi** (`ablation_matrix.md`, 335 satır) | ✅ |
| Makale taslağı (`paper_draft.md`, 297 satır) | ✅ |

### 13.4 Makalenin taşıyıcı iddiaları

1. 🏆 **Füzyon: test ROC-AUC 0.9516, doğruluk %89.4** — TDA, **ince ayarlı** güçlü baseline'a bile **+0.0232** [0.0132, 0.0338] katıyor ⇒ füzyon kazancı zayıf baseline artefaktı **değil**
2. **PL zarf mekanizması** — H1 zarfının %98–99.6'sı arka plan; sınıf sinyali %1–2
3. **Tamamlayıcılık r ≈ 0.30** — sızıntı kontrolü şans (0.4378)

### 13.5 Val düzeyinde doğrulanmış gelecek iş — **iki güçlü aday**

| # | İyileştirme | Val AUC | Kazanç | Boyut kontrolü |
| :-: | :--- | :---: | :---: | :--- |
| 🥇 | **E2.6 çok parametreli dilimler** (`f + t·Sobel gradyan`) | 0.8399 → **0.8880** | **+0.0480** | ✓ (karıştırılmış: 0.6925) |
| 🥈 | **E2.1 çift yönlü (diphasic) filtrasyon** (sublevel+superlevel) | 0.8399 → **0.8609** | **+0.0210** | ✓ (karıştırılmış: 0.7803) |

Her ikisi de **test bütçesi tükendiği için** 0.9516'lık ana sonuca **eklenememiştir** ve **yeni bir test bütçesinin öncelikli adaylarıdır** (E2.6 daha güçlü). İkisi birleştirilebilir.

🔑 **Yön gösterici içgörüler:**
* **Polarite +0.0210 (KEEP) vs ölçek +0.0053 (KILL)** ⇒ kazanç *hangi kümelerin filtrelendiğinden* gelir, diyagramın yumuşatılmasından değil.
* **`f + t·gradyan` (katmanlaştırma) işe yarıyor; gradyanla sınıf elemek (E1.2) zarar veriyor** ⇒ **elemek değil, katmanlaştırmak**.
* **Şablon-göreli sapma zarar veriyor** ve yanlış şablon doğrudan daha iyi (E2.3b) ⇒ arka plan/şablon yapısı **gürültü değil, bilgi taşıyıcısı**. "Arka planı bastır" öncülü **E1.2 + E2.3b ile iki yoldan çürütüldü**.

### 13.6 Kapanışta kaydedilen rigor dersleri

| # | Ders | Deney |
| :-: | :--- | :--- |
| 1 | Tek-seed parlak bulgu gürültü olabilir → **çok-seed eşleştirilmiş** test şart | E4.1 → E4.6 |
| 2 | "Fazla iyi" sonuçları **confound** açısından denetle (boyut eşleştirmesi) | E3.6 (0.994 → 0.687) |
| 3 | Önbellek satır sayısı eşleşmezse **görüntü/etiket kayar** (şans düzeyi AUC) | E0.5 |
| 4 | **Baseline'ı yeniden türetmek kazancı şişirir** | E2.1 (+0.0646 → +0.0210) |
| 5 | **Per-feature afin dönüşümler StandardScaler tarafından yutulur** | E2.3 (no-op) |
| 6 | Kaynak-dışı genellemede **eğitim verisi miktarı eşitlenmeli** | E5.3b (0.8468 → 0.8098) |

### 13.7 Durum

**Test bütçesi 6/6 tükendi** — dondurulmuş test seti **kapalıdır**. Yeni deneyler yalnızca **validation** veya **yeni veri** ile yapılabilir. Program **yayına hazırdır**.

---

## 14. V6.9 sonrası ek (v3 protokolü dondurulmuş, bütçe muhasebesi güncel)

v3 protokolü (§0–§13) **kapalıdır** ve değiştirilmez. Ancak §0.3'teki *test bütçesi*
muhasebesi, sonraki hatlarla birlikte şöyle okunmalıdır:

| Dönem | Test kullanımı | Ayar | Sonuç |
| :--- | :--- | :---: | :--- |
| v3 (E0.2, E3.4, E0.3, E3.1, E3.7, E0.6) | **6/6 bütçe tükendi** | protokol içi | headline 0,9516 |
| V6.4 | 1 ek mühür | **yok** | birleşik hasta AUC 0,9876 |
| **V6.9** | 1 ek mühür | **yok** (ön-kayıtlı, hash kilitli) | rank_metric **0,8842** · AUC **0,9645** |

⇒ **Kural:** yeni bir test sonucu ancak **yeni bir veri kümesiyle** üretilebilir; aynı
bölmede üçüncü bir "mühür" mühür sayılmaz. Ayrıca v3'ün 6/6 bütçesi **protokol içi**
değerlendirmelerdir; V6'nın iki mührü farklı model ailelerine aittir ve ikisi de
seçim yapmadan koşulmuştur.

⚠️ **Yöntemsel ders (§13.6'daki derslere ek):** "Val'de seç, val'de raporla" bir sızıntı
değildir ama **ölçülebilir bir iyimserliktir** — V6'da 296 + 19 + 12 değerlendirme
yapıldı ve dürüst test değeri **0,029** daha düşük çıktı. Yeni hatlarda **seçim bütçesi
beyan edilmeli** ve dürüst sayı ayrı bir mühürle üretilmelidir.
Ayrıntı: [LEAK_AUDIT.md](LEAK_AUDIT.md) · [experiments.md §13](experiments.md)

**Ek (V6.9d) — veri kaynağı örtüşmesi:** "Yeni veri" kuralı, ön-eğitimli modeller
kullanıldığında **iki** katmanlıdır: (i) değerlendirme seti yeni mi, (ii) o setin
**kaynakları** omurganın ön-eğitiminde var mı? NODE21'de ikincisi vardır: pozitiflerin
%82,9'u NIH ChestX-ray14 + PadChest'ten. Kutular yeni olduğu için lokalizasyon metriği
sızıntıya kapalıdır, ama kaynak-ayrıştırmalı ölçüm ve **beyan** zorunludur
([LEAK_AUDIT.md](LEAK_AUDIT.md) §9).

**Ek (V6.9f) — dinamik kutu (yarıçap kafası):** V6.8'de `R>0` maskesiyle düzeltilen dinamik
yarıçap kafası val'de 0,9185 (sabit 56 px: 0,9131) idi; **kilitli testte resmî skor EŞİT**
(0,8842 = 0,8842), yalnızca S@0.5+ noktalarında +0,013 kazandı ⇒ val kazancı test'e taşınmadı.
Sabit 56 px muhafazakâr varsayılan olarak kalır ([experiments.md §V6.9.8](experiments.md)).

> ⚠️ **ZORUNLU TODO — Docker Gönderimi Öncesi (V6.9g):** Grand-Challenge sunucusuna fiilî Docker
> gönderimi yapılacağı vakit, `M5_foundation_fulldata.pt` modelinin yarıçap kafası V6.8'deki
> `R>0` maske düzeltmesiyle donuk gövde üzerinde yeniden eğitilecek
> ([v6_9g_fulldata_radius.py](pipelines/v6_two_stage/v6_9g_fulldata_radius.py)) ve
> `submission_docker/process.py` sabit 56×56 px yerine öğrenilmiş dinamik kutuyu kullanacak
> şekilde güncellenecektir. **⚠️ Ancak V6.9f, dinamik kutunun resmî rank_metric'i
> DEĞİŞTİRMEDİĞİNİ gösterdi** (0,8842 = 0,8842) ⇒ bu adım artık zorunlu bir iyileştirme değil,
> FROC-optimal dağıtım hedefleniyorsa uygulanacak bir **seçenektir**.

**Ek (V6.9k) — Leave-JSRT-Out:** JSRT hastalarının 107'si NODE21 `train` bölmesindeydi ⇒
kaynak-dışı genelleme iddiası için JSRT eğitimden **tamamen** çıkarılıp iki aşama aynı reçeteyle
yeniden eğitildi (izole: ayrı checkpoint/bundle, ana artifact SHA-256'ları değişmedi). JSRT dış
testinde (149 hasta, hepsi yabancı) **Recall@K10 0,8523 · K50 0,9396 · hasta AUC 0,9395**.
Ana modelle farkın büyük kısmı **in-sample** ezber; hiç görülmeyen 42 hastada kalan fark
**−0,0432** [−0,0809; −0,0064] ve **veri azalmasıyla açıklanamıyor** (JSRT-dışı val'de iki model
eşit: Δ +0,0013) ⇒ fark-farkı **−0,0435**. **Kural:** kilitli test sayıları `in-fold` etiketlenir,
kaynak-dışı genelleme iddiasında **LOSO** sayıları kullanılır ([LEAK_AUDIT.md](LEAK_AUDIT.md) §9.7).

> 📌 **TODO — bilinçli olarak YAPILMADI (V6.9k eşleştirilmiş bütçe kontrolü):** LOSO'nun
> "veri azalması" bileşeni V6.9k2'de *miktar* için ölçüldü (val(−JSRT)'de Δ ≈ 0). Daha sıkı
> kontrol, **aynı 1.366 görüntü bütçesiyle JSRT'yi koruyup 107 *başka* kaynak pozitifini
> (ör. chestxray14) çıkarmak** olurdu; bu, "kaynak kimliği" ile "kaynak sayısı"nı birebir
> eşleştirip DiD'yi tam nedensel hâle getirir. Maliyet ≈ 1 eğitim koşusu (~20 dk) + aday/özellik
> (~15 dk). **Neden yapılmadı:** sonuçların hiçbiri bu kontrole bağlı değil (V6.9k2 miktar
> etkisini zaten ≈ 0 buldu) ve mevcut hüküm bu kontrol olmadan da muhafazakâr yönde.
> **Yapılırsa:** `v6_9k_leave_jsrt_out.py`'ye `--exclude-source` / `--budget` parametreleri
> eklenip ikinci bir LOSO koşusu + `v6_9k2` tipi val kontrastı çalıştırılmalı; çıktı
> `artifacts/v6_two_stage/v6_9k4_*.json` gibi **ayrı ad alanına** yazılmalı (ana mimariye
> sızmamalı). İlgili kayıtlar: [LEAK_AUDIT.md](LEAK_AUDIT.md) §10 "Açık iş",
> [experiments.md](experiments.md) §V6.9.12, [ablation_matrix.md](ablation_matrix.md) §17.8.

> 📌 **TODO — ERTELENDİ (yakın vadede yapılmayacak) · Konteyner 1.604'te kaldı**
> (mimari uyum denetimi V6.9o bulgusu; 30.10.2026 kararı: **şimdilik ertelendi**) —
> Bilimsel hat saf 1.600 kullanıyor (`stage2_foundation_tda_hgb_pure1600.joblib`), **ama
> dağıtılan konteyner hâlâ 1.604**: `submission_config.json → stage2.bundle =
> submission_fulldata/stage2_tda_hgb_fulldata.joblib` (`n_features_in_ = 1604`) ve
> `v6_7_submission_infer.py:97-103` dört uzamsal sütunu hesaplayıp `concat([tda, spa])` yapıyor.
> **Neden şimdi yapılmıyor:** (i) fulldata stage-2 bundle'ı yeniden eğitim gerektiriyor,
> (ii) kullanıcı 30.10.2026'da bu işi **yakın vadede yapmayacağını** bildirdi ⇒ TODO açık
> kalıyor, "yapılacaklar" listesinde **düşük öncelikli** sayılır. Etkisi zaten beyan edildi:
> `README.md` "Konteyner" satırı ve [submission_docker/README.md](submission_docker/README.md)
> konteynerin 1.604 olduğunu açıkça yazıyor. **Yapılırsa:** `v6_7_build_fulldata_stage2.py` saf
> 1.600 ile koşturulup `submission_fulldata/stage2_tda_hgb_fulldata_pure1600.joblib` üretilmeli,
> `v6_7_submission_infer.py` içindeki `spa` dalı kaldırılmalı, `submission_config.json` yeni
> sha256 ile güncellenmeli ve `smoke_test.py` yeniden koşturulmalı. Not: dış-domainde 4 konum
> sütunu **hafif faydalı** olduğu için (VinDr sıfır-eğitim −0,018) bu geçiş isteğe bağlı bir
> tercihtir; bilimsel raporlanan sayılarla **uyum** için gereklidir.

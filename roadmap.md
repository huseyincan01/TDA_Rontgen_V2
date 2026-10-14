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
| ⚠️ **PyTorch** | **`torch 2.14.1+cpu`** (`cuda.device_count() = 0`). **E0.2'de ölçüldü:** küçük CNN 5 zonu **~27 s**'de CPU'da eğitiyor ⇒ TDA/özellik tabanlı deneyler GPU'suz yapılabilir. CUDA yalnızca görüntü tabanlı pretrainli omurga (E0.3/E3.1/E3.2) için gerekli → **E0.0** |
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
| **Faz 0 — Altyapı** | E0.0, E0.1, E0.2, E0.3 | ~1 hafta | CUDA torch geri kuruldu · registry çalışıyor · 0.8631 referansı yeniden üretildi · pretrainli omurga GPU'da smoke test geçti |
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
| **E0.0** | W0 | ⚠️ **BLOCKER:** CUDA PyTorch kurulunca GPU eğitimi tekrar çalışır | — | 0.5g | Düşük | **Kritik** | Planlandı |
| **E0.1** | W0 | Deney kayıt protokolü + dizin sözleşmesi kurulabilir | — | 0.5g | Düşük | Altyapı | **KEEP** ✅ |
| **E0.2** | W0 | PI CNN (0.8631) bit-bit yeniden üretilebilir | — | 0.5g | Düşük | Referans | **KEEP** ✅ birebir |
| **E0.3** | W0 | Pretrainli omurga (ResNet/EfficientNet) zon kesitlerinde çalışır | — | 1g | Düşük | Altyapı | Planlandı |
| **E3.4** | W3 | CNN + SVM soft-voting tek modellerden ≥ +0.015 AUC verir | E0.2 | 0.5g | Düşük | Orta | Planlandı |
| **E4.1** | W4 | %10/%20/%30 overlap kurallarında model sıralaması değişmez | E0.2 | 1g | Düşük | Yayın kanıtı | Planlandı |
| **E1.1** | W1 | Kemik bastırma H1 kemik gürültüsünü azaltır, val AUC ≥ +0.015 artar | — | 2g | Orta | **Yüksek** | Planlandı |
| **E1.2** | W1 | Frangi damar bastırma H1 damar döngülerini temizler, AUC ≥ +0.015 artar | — | 2g | Orta | **Yüksek** | Planlandı |
| **E3.1** | W3 | Pretrainli CNN + TDA geç füzyon > her ikisi tek başına (CI ile) | E0.3 | 1.5g | Düşük | **Çok Yüksek** (ana iddia) | Planlandı |
| **E3.2** | W3 | TDA özellikleri CNN özellik vektörüne eklenince AUC artar | E0.3 | 1.5g | Orta | **Çok Yüksek** | Planlandı |
| **E2.1** | W2 | Çift yön (sublevel+superlevel) filtrasyon tek yönden ≥ +0.015 iyi | — | 1.5g | Düşük | Orta | Planlandı |
| **E2.2** | W2 | Çok ölçekli PI tek ölçekten ≥ +0.015 iyi | E2.1 | 2g | Orta | Orta-Yüksek | Planlandı |
| **E2.3** | W2 | Sağlıklı şablon çıkarma (relative persistence) AUC artırır | E0.2 | 2.5g | Yüksek | **Yüksek** (novelty) | Planlandı |
| **E2.4** | W2 | Aday-olasılık ağırlıklı filtrasyon H1'i nodüle odaklar | E1.2 | 2.5g | Yüksek | Orta | Planlandı |
| **E4.2** | W4 | Kalıcılık özellikleri radyolojik yapılara eşlenebilir | E0.2 | 2g | Orta | **Yüksek** (yayın) | Planlandı |
| **E4.3** | W4 | Hangi zonlarda sistematik hata yapıyoruz? | E0.2 | 1g | Düşük | Orta | Planlandı |
| **E3.3** | W3 | İki aşamalı candidate-based TDA yanlış pozitifi azaltır | E1.2 | 4g | Yüksek | Yüksek | Planlandı |
| **E2.5** | W2 | Sliced Wasserstein / PersLay, PI'dan ≥ +0.015 iyi | E0.2 | 3g | Yüksek | Orta | Planlandı |
| **E1.4** | W1 | Atlas kaydı arka plan topolojisini tutarlılaştırır | E1.1 | 4g | Yüksek | Orta | Planlandı |
| **E2.6** | W2 | Çok parametreli persistence ek bilgi taşır (`multipers`/MMA olgunluğu doğrulandı) | E2.2 | 5g+ | Yüksek | Orta | Planlandı 🔓 |
| **E4.4** | W4 | **PL üst-zarf (envelope) çökmesi nicel olarak gösterilebilir** | E0.2 | 1.5g | Düşük | **Çok Yüksek** | Planlandı |
| **E2.7** | W2 | PI ağırlık fonksiyonu (Adams `w=p` vs max-distance/linear/arctan) AUC'yi değiştirir | E0.2 | 0.5g | Düşük | Orta | Planlandı |
| **E2.8** | W2 | σ / ızgara çözünürlüğü sonuçları anlamlı değiştirmez (sağlamlık kanıtı) | E0.2 | 1g | Düşük | Orta | Planlandı |
| **E5.3** | W5 | Harici doğrulama (JSRT / ChestX-ray14 nodül alt kümesi) genellemeyi kanıtlar | E3.1 | 3g | Yüksek | Yüksek | Planlandı |
| **E5.1** | W5 | Tüm ablasyonlar tek matriste tutarlı sunulabilir | Faz 1–4 | 2g | Düşük | Yayın | Planlandı |
| **E5.2** | W5 | Metin + figürler hakem beklentilerini karşılar | E5.1 | 4g | Orta | Yayın | Planlandı |

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
| Faz 0 | E0.0, E0.1, E0.2, E0.3 | 2/4 | E0.3 | 2 | 0 |
| Faz 1 | E4.4, E2.7, E2.8, E3.4, E4.1 | 0/5 | — | 0 | 0 |
| Faz 2 | E3.1, E3.2 | 0/2 | — | 0 | 0 |
| Faz 3 | E1.1, E1.2, E2.1, E2.2, E2.3, E2.4 | 0/6 | — | 0 | 0 |
| Faz 4 | E3.3, E2.5, E1.4, E2.6 | 0/4 | — | 0 | 0 |
| Faz 5 | E4.2, E4.3, E5.1, E5.2, E5.3 | 0/5 | — | 0 | 0 |

**Referans noktası (v2.1, kilitli):** PI CNN test ROC-AUC **0.8631** · RBF-SVM **0.8289** · k-NN **0.7162** · Ham CNN **0.7043** · PL CNN **0.5450** · PL SVM **0.5217**

---

## 10. Karar Günlüğü

| Tarih | ID | Karar | Gerekçe |
| :--- | :--- | :--- | :--- |
| 2026-10-14 | **E0.1** | **KEEP** | Manifest altyapısı (`src/node21_tda/experiment.py`) + `configs/experiments/` sözleşmesi kuruldu, smoke test geçti. |
| 2026-10-14 | **E0.2** | **KEEP** | v2.1 PI küçük CNN referansı **bit-bit yeniden üretildi** (16/16 metrik; test ROC-AUC 0.8631, TP/FP/TN/FN 137/114/448/35). Pipeline deterministik. Yan bulgu: eğitim CPU'da ~27 s → özellik tabanlı deneyler GPU'suz yapılabilir. |

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

1. **Skorumuz zayıf değil.** 0.8631 hasta düzeyi AUC, NODE21 rank-1 detektörünün **görüntü düzeyi AUC'sine (0.868)** denk; NODE21 baseline'ının (0.816) ve ChestX-ray14 nodül AUC'sinin (0.67) üstünde. Sorun skor değil, **çerçeveleme ve karşılaştırma eksikliği**.
2. **PL'nin çökmesi en güçlü özgün iddiamız.** Literatür PL'yi "kararlı" diye övüyor; bizim 0.5450'ye çöküşümüz tipik beklentinin tersi ve üst-zarf mekanizmasıyla açıklanabilir → **E4.4** bunu kanıta çevirmeli.
3. **Kemik/damar bastırma + kalıcı homoloji kombinasyonu literatürde yok** → Ana özgünlük iddiası. Topo-CXR (ICCVW 2023) ise tam tersini (*"ön-işleme gerekmez"*) savunuyor → net karşı-pozisyon.
4. **Görev/metrik uyumu:** NODE21'in AUC'si zaten görüntü düzeyi + **max agregasyon**; bizim hasta skorumuz da max-pooling. Çerçeveleme: *"yorumlanabilir ve hafif bir topolojik temsil, ağır detektör topluluklarının görüntü düzeyi AUC seviyesine yaklaşıyor."*

### Yenilik iddia EDİLMEMESİ gerekenler

"İlgisiz yapılar diyagramı domine eder → ROI kısıtlaması gerekir" (Vandaele 2020/2022) · "PI ve PL veri setine bağlı farklı davranır" (Barnes 2021, Sulowska 2026) · "topoloji tek başına CNN'e denk, füzyonla artar" (TDA-Net) · "topoloji gerçek histolojiyi, radyomik radyolog skorunu taklit eder" (Vandaele 2022).

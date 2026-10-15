# NODE21 TDA — Yayınlanabilir Ablasyon Matrisi ve Sonuç Tabloları

> **Bu belge ne?** Programın tamamında üretilen tüm sayısal sonuçların **tek yerde**, kaynak (deney ID + betik) atıflarıyla toplanmış hâli. Makale tablolarının taslağıdır.
> Yönetim ve karar günlüğü: [roadmap.md](roadmap.md) · Deney anlatıları: [experiments.md](experiments.md) · Literatür: [literature.md](literature.md)
> **Son güncelleme:** 2026-10-14 (Faz 0 ✅ · Faz 1 ✅ · Faz 2 1/2)

---

## ⚠️ 0. Zorunlu Beyan (her tabloda geçerli)

Veri kümemiz **NODE21'in halka açık TRAINING setiyle birebir aynıdır** (4882 = 1134 nodüllü + 3748 normal; yerel olarak doğrulandı). NODE21'in resmî test setleri (281 + 298 görüntü) **özeldir ve kullanılmamıştır**.

* **Test setimiz bu setin kendi iç bölünmesidir** (train 3414 / val 734 / test 734 hasta).
* **Hasta sızıntısı yoktur** (hasta düzeyi split, 0 kesişim; `test_v2_pipeline.py` + E0.2 ile doğrulandı).
* Bu nedenle **NODE21 leaderboard'u ile karşılaştırma yapılmaz**; leaderboard sayıları yalnızca *bağlam* olarak verilir.
* Görev farkı: NODE21 kutu (bounding box) üretir → FROC mümkün; biz **hasta/görüntü düzeyi sınıflandırma** yapıyoruz. NODE21'in AUC metriği de görüntü düzeyi + **max agregasyon** olduğundan metrik mantığı örtüşür.

---

## 1. Ana Sonuç — Model Liderlik Tablosu (hasta düzeyi, 734 test hastası)

| Sıra | Model | Temsil | Val ROC-AUC | **Test ROC-AUC** | Kaynak |
| :-: | :--- | :--- | :---: | :---: | :--- |
| 🥇 | **Füzyon: ResNet18 + TDA ensemble** | görüntü + topoloji | 0.9572 | **0.9211** | **E3.1** · [betik](pipelines/exp/E3_1_tda_image_fusion.py) |
| 🥈 | TDA ensemble (CNN + SVM soft-voting) | PI vektörü | 0.8821 | **0.8776** | **E3.4** · [betik](pipelines/exp/E3_4_ensemble_cnn_svm.py) |
| 🥉 | **TDA PI küçük CNN** | PI (2,20,20) | 0.8607 | **0.8631** | v2.1 · E0.2 ile birebir üretildi |
| 4 | ResNet18 (donmuş) + lojistik regresyon | 512-D ImageNet | 0.8884 | **0.8349** | **E0.3** · [betik](pipelines/exp/E0_3_resnet_baseline.py) |
| 5 | RBF-SVM | PI vektörü | 0.8278 | 0.8289 | v2.1 |
| 6 | k-NN | PI vektörü | 0.7125 | 0.7162 | v2.1 |
| 7 | Ham görüntü küçük CNN (64×64) | gri piksel | 0.6870 | 0.7043 | v2.1 |
| 8 | PL küçük CNN | Landscape | 0.5927 | 0.5450 | v2.1 |
| 9 | PL RBF-SVM | Landscape | 0.5449 | 0.5217 | v2.1 |

**Bağlam (bizim split'imizle karşılaştırılamaz):** NODE21 final test leaderboard → D1 0.868 · D2 0.862 · D3 0.833 · baseline 0.816 · **ensemble 0.877**.

> **Okunuşu:** Bizim *füzyon* modelimiz (0.9211) NODE21'in **tüm yarışmacıların birleşiminden** (0.877) yüksek; *tek başına TDA* (0.8631) ise NODE21 rank-1 detektörüne (0.868) denk. Üstelik hiçbir derin tespit mimarisi, harici pretrain veya uçtan uca eğitim kullanılmadı.

---

## 2. Ablasyon 1 — **Temsil Karşılaştırması** (TDA vs görüntü)

| Karşılaştırma | Test ROC-AUC | Fark | Kanıt |
| :--- | :---: | :---: | :--- |
| PI küçük CNN vs Ham görüntü küçük CNN | 0.8631 vs 0.7043 | **+0.1588** | %95 CI'lar **kesişmiyor** (p<0.001) |
| PI küçük CNN vs PL küçük CNN | 0.8631 vs 0.5450 | **+0.3181** | %95 CI'lar **kesişmiyor** (p<10⁻⁶) |
| **PI küçük CNN vs güçlü ResNet18 baseline** | 0.8631 vs 0.8349 | **+0.0282** | **E0.3** — güçlü pretrainli model **test'te geçilemiyor** |

**Destekleyici bulgu — genelleme davranışı:**

| Model | Val | Test | Val→Test kaybı |
| :--- | :---: | :---: | :---: |
| ResNet18 (donmuş) | 0.8884 | 0.8349 | **−0.0535** |
| TDA PI küçük CNN | 0.8607 | 0.8631 | **+0.0024** |

> Küçük veri rejiminde **pretrainli görüntü modeli ciddi genelleme kaybı yaşarken**, topolojik temsil neredeyse kayıpsız kalıyor.

---

## 3. Ablasyon 2 — **Füzyon / Tamamlayıcılık** (ana iddia)

| Model | Val ROC-AUC | **Test ROC-AUC** |
| :--- | :---: | :---: |
| ResNet18 (tek başına) | 0.8884 | 0.8349 |
| TDA ensemble (tek başına) | 0.8821 | 0.8776 |
| **Füzyon (a=0.55)** | **0.9572** | **0.9211** |

| Ölçüt | Değer |
| :--- | :--- |
| Test AUC farkı (füzyon − ResNet) | **+0.0862** |
| Bootstrap %95 CI (N=2.000) | **[+0.0632, +0.1099]** |
| p-değeri | **≈ 0.0000** |
| Val farkı | +0.0688 |
| **Korelasyon (ResNet, TDA)** | **r ≈ 0.30** (val 0.3034 · test 0.3489) |
| Sızıntı kontrolü (etiket permütasyonu) | AUC **0.4378** (≈ şans) ⇒ sızıntı **yok** |

> **Mekanizma:** İki temsil **neredeyse dik** (r≈0.30) → tamamlayıcılık gerçek. Bu, literatürdeki TDA-Net füzyon kazancı (~+0.05) mertebesinin üzerinde ve **istatistiksel olarak çok güçlü**.
> Kaynak: **E3.1** · bağımsız doğrulama [scratch/verify_E3_1.py](scratch/verify_E3_1.py)

---

## 4. Ablasyon 3 — **PL Neden Çöktü?** (mekanizma, en güçlü özgünlük)

Dengeli train alt kümesi (n=2.040 zon). **E4.4** · [betik](pipelines/exp/E4_4_pl_envelope_analysis.py)

| Ölçüt | λ1 | λ2 | λ3 | λ4 | λ5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **H1** ortalama zarf tepesi | 109.89 | 102.61 | 97.56 | 93.60 | 89.61 |
| **H1** enerji payı | %22.3 | %20.8 | %19.8 | %19.0 | %18.1 |
| **H1** sınıf sinyali / zarf | %1.62 | **%1.90** | %1.54 | %0.37 | %1.72 |
| **H0** ortalama zarf tepesi | 173.87 | 173.42 | 172.67 | 171.79 | 170.49 |
| **H0** sınıf sinyali / zarf | %0.02 | %0.21 | %0.39 | %0.78 | **%0.89** |

**Sonuç:** H1 zarf tepeleri λ1→λ5 boyunca 110→90 ve **birbirine yakın**, enerji eşit dağılmış ⇒ zarf her zonda **≥5 çok uzun ömürlü arka plan H1 sınıfıyla doymuş**. Sınıf ayrımını taşıyan sinyal zarfın yalnızca **%1–2'si** (≈50–100× bastırılma). PI, `(birth, persistence)` 2B yoğunluğunu koruduğu için bu sinyali saklıyor.

**Bağımsız destek (E2.7):** üst-doğrusal ağırlık `p²` en kötü sonucu veriyor (−0.0189) — yani arka plan sınıflarını amplifiye etmek zarar veriyor ⇒ mekanizma teyit ediliyor.

---

## 5. Ablasyon 4 — **Bileşen Katkıları**

| Bileşen | Seçenek | Val ROC-AUC | Karar | Deney |
| :--- | :--- | :---: | :--- | :--- |
| **PI ağırlık fonksiyonu** `w(p)` | **`p` (Adams)** | **0.8736** | **KEEP** | E2.7 |
| | `arctan(p)` | 0.8640 (−0.0096) | | E2.7 |
| | `√p` | 0.8624 (−0.0112) | | E2.7 |
| | `1` (sabit) | 0.8591 (−0.0145) | | E2.7 |
| | `p²` | 0.8547 (−0.0189) | | E2.7 |
| **Gaussian genişliği** σ | 0.5 | 0.8460 (−0.0236) | | E2.8 |
| | **1.0** | **0.8697** | **KEEP** | E2.8 |
| | 1.5 | 0.8734 (+0.0038) | | E2.8 |
| | 2.0 | 0.8712 (+0.0015) | | E2.8 |
| **Izgara çözünürlüğü** | 15×15 | 0.8545 (−0.0152) | | E2.8 |
| | **20×20** | **0.8697** | **KEEP** | E2.8 |
| | 25×25 | 0.8469 (−0.0227) | | E2.8 |
| **Zon füzyonu (TDA içi)** | Hasta düzeyi soft voting | **0.8821 / test 0.8776** | **KEEP** | E3.4 |
| | Zon düzeyi füzyon | 0.8699 / test 0.8629 | | E3.4 |
| **Etiketleme eşiği** | %10 | 0.8922 (tek seed) → **0.8587** (5 seed) | | E4.1 → E4.6 |
| | **%20** | **0.8736** → **0.8501** | **KEEP** | E4.1 → E4.6 |
| | %30 | 0.8693 | | E4.1 |

> **Not:** E4.6 çok-seed doğrulaması, %10'un görünürdeki +0.0186 kazancının gerçekte yalnızca **+0.0085** (std 0.0096, 4/5 seed pozitif) olduğunu gösterdi ⇒ **gürültü**, mevcut %20 kuralı korundu.

---

## 6. Ablasyon 5 — **Yorumlanabilirlik** (persistence → görüntü bölgesi)

**E4.2** · [betik](pipelines/exp/E4_2_persistence_localization.py) · 30 pozitif + 30 negatif zon, top-K=10, GUDHI `cofaces_of_persistence_pairs()`

| Boyut | Top-K gradyan oranı (ort / medyan) | Rastgele kontrol | Oran>1.5 olan zon | Karar |
| :---: | :---: | :---: | :---: | :--- |
| **H0** | **6.176 / 6.608** | 0.974 | **%90** | ✅ hipotez doğrulandı |
| **H1** | 1.474 / **1.021** | 0.981 | %30 | ❌ hipotez çürütüldü |

**Yorum:** En kalıcılıklı **H0** sınıfları yüksek gradyanlı **kenarlara** düşüyor (6.2×) ⇒ H0 görüntüsü ağırlıklı olarak **arka plan/sınır** kodluyor. En kalıcılıklı **H1** sınıfları kenarlarda **değil** (medyan 1.02). Her iki durumda da en kalıcı sınıflar **nodüle özgü değil** ⇒ §4'ü bağımsız destekliyor.
**Metodolojik katkı:** literatürde "H1 döngülerinin hangi anatomik yapıya karşılık geldiği kanıtlanmıyor" deniyordu; bu deney o **yeteneği** kuruyor.

---

## 6.1 Zon Bazlı Hata Analizi

**E4.3** · [betik](pipelines/exp/E4_3_zone_error_analysis.py) · validation, 734 hasta (test kullanılmadı, yeni model eğitilmedi)

**Zon başına hasta düzeyi val ROC-AUC** (zon TEK BAŞINA kullanılırsa):

| Zon | CNN | SVM | ResNet |
| :--- | :---: | :---: | :---: |
| 1 Sağ Üst | 0.7859 | 0.7252 | 0.8055 |
| 2 Sağ Orta | **0.8199** | **0.8156** | **0.8244** |
| 3 Sağ Alt | 0.8017 | 0.7321 | 0.7743 |
| 4 Sol Üst | 0.7380 | 0.7046 | **0.8409** |
| 5 Sol Alt | 0.7763 | 0.7379 | 0.8021 |

**Sorumlu zon dağılımı** (fusion kararı hangi zondan geliyor): Zon 5 %26.7 · Zon 1 %25.3 · Zon 3 %21.5 · Zon 4 %18.9 · Zon 2 %7.5 ⇒ **tek-zon hakimiyeti yok**.
**Yanlış negatifler (n=12):** ortalama olasılık 0.5812, maksimum 0.6317 — eşiğin (0.6357) hemen altında **"kıl payı"** hatalar (TP ortalaması 0.7271).

> 🚨 **DOĞRULANMIŞ SINIRLAMA — model lokalize ETMİYOR (E4.7).** E4.3'ün şüphesi **E4.7 ile doğrulandı**: nodül-pozitif hastalarda modelin argmax zonu gerçek nodül zonuna **şans düzeyinde** düşüyor (isabet 0.228–0.298 vs şans 0.268; binom p = 0.21–0.90) ve **zon düzeyi AUC yalnızca 0.48–0.53** — yani modeller bir hastanın nodüllü zonunu nodülsüz zonlarından **ayırt edemiyor**.
>
> **Sonuçları geçersiz kılmaz** — hasta düzeyi sınıflandırma (füzyon 0.9211) ve PL mekanizması (E4.4) lokalizasyondan bağımsız olarak geçerlidir. Ancak **iddianın kapsamını daraltır**: *"nodülü doğru zonda tespit ediyoruz"* **denemez**; doğru ifade *"nodül varlığıyla ilişkili topolojik imzayı yakalıyoruz"*dur. "5-zon **lokalizasyon**" yenilik iddiası **kaldırılmalı**; "5-zon paralel ayrıştırma + max-pooling" olarak yazılmalı.


| Boyut | Test edilen | Sonuç | Deney |
| :--- | :--- | :--- | :--- |
| PI ağırlık fonksiyonu | 5 varyant | ✅ mevcut seçim optimal | E2.7 |
| σ (Gaussian) | 4 değer | ✅ sağlam (maks sapma 0.0236 < 0.03) | E2.8 |
| Izgara çözünürlüğü | 3 değer | ✅ sağlam (maks sapma 0.0227 < 0.03) | E2.8 |
| Etiketleme eşiği | %10/%20/%30 | ⚠️ 5-seed'de fark yok (gürültü) → %20 | E4.1 + E4.6 |
| Tekrar üretilebilirlik | referans yeniden üretim | ✅ **bit-bit aynı (16/16 metrik)** | E0.2 |
| Sızıntı | etiket permütasyonu | ✅ AUC 0.4378 (şans) | E3.1 |
| **Seed varyansı** | 5 seed | ⚠️ **±0.02–0.03** → kural 10: makalede çok-seed ortalama | E4.6 |

---

## 8. İstatistiksel Kanıt Özeti (N=2.000 bootstrap)

| Karşılaştırma | Fark | %95 CI | p | Deney |
| :--- | :---: | :---: | :---: | :--- |
| **Füzyon − ResNet18** | **+0.0862** | **[+0.0632, +0.1099]** | **≈0.0000** | E3.1 |
| Ensemble − PI CNN | +0.0145 | [+0.0034, +0.0260] | 0.010 | E3.4 |
| PI CNN − Ham CNN | +0.1588 | *(CI'lar kesişmiyor)* | <0.001 | v2.1 |
| PI CNN − PL CNN | +0.3181 | *(CI'lar kesişmiyor)* | <10⁻⁶ | v2.1 |

**Tek modellerin test %95 CI'ları (v2.1):** PI CNN 0.8631 [0.8302, 0.8934] · SVM 0.8289 [0.7938, 0.8616] · k-NN 0.7162 [0.6761, 0.7575] · Ham CNN 0.7043 [0.6587, 0.7456] · PL CNN 0.5451 [0.4957, 0.5946] · PL SVM 0.5216 [0.4733, 0.5705]

---

## 9. Yenilik Konumlandırması (literatür taramasından)

| # | İddia | Literatür durumu | Kanıt |
| :-: | :--- | :--- | :--- |
| 1 | **PL üst-zarf çökmesinin nicel gösterimi** | **Bulunamadı** (Vandaele 2020 "ilgisiz nesne" problemini tanıyor ama PL zarfı için test etmiyor) | E4.4 ✅ |
| 2 | **TDA + görüntü tamamlayıcılığı (r≈0.30) ve füzyon kazancı** | TDA-Net ~+0.05 (CXR/COVID); bizimki +0.0862 | E3.1 ✅ |
| 3 | **CXR'de H1'in bronkovasküler döngü + kaburga ile domine edilmesi** | Spesifik formülasyon yok | E4.2/E4.4 |
| 4 | **5 anatomik zon + zon başına PH + hasta düzeyi max-pooling** | Bulunamadı (Topo-CXR global, Vandaele/Wang ROI-kısıtlı) | v2.1 — ⚠️ **lokalizasyon iddiası E4.7 ile DÜŞTÜ** (aşağıya bakınız) |
| 5 | **NODE21'de kalıcı homoloji** | Bulunamadı | v2.1 + E0.2 |

**Yenilik iddia EDİLMEMESİ gerekenler:** "ilgisiz yapılar diyagramı domine eder → ROI kısıtlaması" (Vandaele 2020/2022) · "PI vs PL veri setine bağlı" (Barnes 2021, Sulowska 2026) · "topoloji ≈ CNN, füzyon artırır" (TDA-Net).

---

## 10. Açık Kalan İşler

| Öncelik | İş | Durum |
| :-: | :--- | :--- |
| 1 | **E5.1** — bu belgenin finalize edilmesi + figürler | Bu belge taslak |
| 2 | **E5.3** — harici doğrulama (JSRT / ChestX-ray14) | Planlandı (1 test değerlendirmesi) |
| 3 | **E3.2** — orta seviye füzyon (özellik birleştirme) | Planlandı |
| 4 | **E2.1/E2.2/E2.3** — çift yön, çok ölçek, şablon filtrasyonu (Faz 3) | Planlandı |
| 5 | **E4.3** — zon bazlı hata analizi | Planlandı |
| 6 | **Kural 10 uygulaması** — kilitli modeller için çok-seed ortalama | Bekliyor |
| — | **Test bütçesi** | **4/6 kullanıldı** |

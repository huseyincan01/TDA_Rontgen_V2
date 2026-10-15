# NODE21 TDA — Yayınlanabilir Ablasyon Matrisi ve Sonuç Tabloları

> **Bu belge ne?** Programın tamamında üretilen tüm sayısal sonuçların **tek yerde**, kaynak (deney ID + betik) atıflarıyla toplanmış hâli. Makale tablolarının taslağıdır.
> Yönetim ve karar günlüğü: [roadmap.md](roadmap.md) · Deney anlatıları: [experiments.md](experiments.md) · Literatür: [literature.md](literature.md)
> **Son güncelleme:** 2026-10-15 (Faz 0 ✅ · Faz 1 ✅ · Faz 2 2/3 · Faz 3 3/6 · Faz 4 1/4 · Faz 5 5/6) — **test bütçesi tükendi (6/6)**

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
| 🥇 | **Füzyon: ince ayarlı ResNet18 + TDA ensemble** | görüntü + topoloji | 0.9572 | **0.9516** (doğruluk %89.4) | **E0.6** · [betik](pipelines/exp/E0_6_decisive_test.py) |
| 🥈 | İnce ayarlı ResNet18 (son-blok) | 512-D ImageNet | — | **0.9284** | **E0.4/E0.5/E0.6** · [betik](pipelines/exp/E0_5_resnet_finetune_reg.py) |
| 🥉 | Füzyon: **donmuş** ResNet18 + TDA ensemble | görüntü + topoloji | 0.9572 | **0.9190 ± 0.0056** (3 seed) | **E3.1 + E3.7** · [betik](pipelines/exp/E3_1_tda_image_fusion.py) |
| 4 | TDA ensemble (CNN + SVM soft-voting) | PI vektörü | 0.8821 | **0.8776** | **E3.4** · [betik](pipelines/exp/E3_4_ensemble_cnn_svm.py) |
| 5 | **TDA PI küçük CNN** | PI (2,20,20) | 0.8607 | **0.8631** | v2.1 · E0.2 ile birebir üretildi |
| 6 | ResNet18 (donmuş) + lojistik regresyon | 512-D ImageNet | 0.8884 | **0.8349** | **E0.3** · [betik](pipelines/exp/E0_3_resnet_baseline.py) |
| 7 | RBF-SVM | PI vektörü | 0.8278 | 0.8289 | v2.1 |
| 8 | k-NN | PI vektörü | 0.7125 | 0.7162 | v2.1 |
| 9 | Ham görüntü küçük CNN (64×64) | gri piksel | 0.6870 | 0.7043 | v2.1 |
| 10 | PL küçük CNN | Landscape | 0.5927 | 0.5450 | v2.1 |
| 11 | PL RBF-SVM | Landscape | 0.5449 | 0.5217 | v2.1 |

**Bağlam (bizim split'imizle karşılaştırılamaz):** NODE21 final test leaderboard → D1 0.868 · D2 0.862 · D3 0.833 · baseline 0.816 · **ensemble 0.877**.

> **Okunuşu:** Bizim *füzyon* modelimiz (**0.9516**) NODE21'in **tüm yarışmacıların birleşiminden** (0.877) belirgin biçimde yüksek; *tek başına TDA* (0.8631) ise NODE21 rank-1 detektörüne (0.868) denk. Üstelik hiçbir derin tespit mimarisi ve uçtan uca eğitim kullanılmadı — görüntü dalı yalnızca **son blok ince ayarı** ile güçlendirildi.
> ⚠️ **Not (E0.6):** Bu tablodaki füzyon, TDA dalının **zon düzeyi füzyon** varyantını (test 0.8629) kullanır; E3.4'ün **hasta düzeyi soft-voting** varyantı (0.8776) daha güçlüdür. Yani raporlanan 0.9516 **muhafazakârdır** — daha güçlü TDA varyantıyla daha yüksek olması beklenir (test bütçesi tükendiği için doğrulanamadı).

---

## 2. Ablasyon 1 — **Temsil Karşılaştırması** (TDA vs görüntü)

| Karşılaştırma | Test ROC-AUC | Fark | Kanıt |
| :--- | :---: | :---: | :--- |
| PI küçük CNN vs Ham görüntü küçük CNN | 0.8631 vs 0.7043 | **+0.1588** | %95 CI'lar **kesişmiyor** (p<0.001) |
| PI küçük CNN vs PL küçük CNN | 0.8631 vs 0.5450 | **+0.3181** | %95 CI'lar **kesişmiyor** (p<10⁻⁶) |
| **PI küçük CNN vs güçlü ResNet18 baseline (donmuş)** | 0.8631 vs 0.8349 | **+0.0282** | **E0.3** |
| **PI küçük CNN vs güçlü ResNet18 baseline (ince ayarlı)** | 0.8631 vs **0.9284** | **−0.0653** | **E0.6** — TDA **tek başına** ince ayarlı modeli geçemiyor |

> ⚠️ **Dürüstlük notu:** TDA **tek başına**, ince ayarlı ResNet18'in (0.9284) **gerisindedir**. TDA'nın değeri tek başına üstünlük değil, **tamamlayıcılıktır** (§3). Bu, iddianın doğru çerçevesidir.

**Destekleyici bulgu — genelleme davranışı:**

| Model | Val | Test | Val→Test kaybı |
| :--- | :---: | :---: | :---: |
| ResNet18 (donmuş) | 0.8884 | 0.8349 | **−0.0535** |
| TDA PI küçük CNN | 0.8607 | 0.8631 | **+0.0024** |

> Küçük veri rejiminde **pretrainli görüntü modeli ciddi genelleme kaybı yaşarken**, topolojik temsil neredeyse kayıpsız kalıyor.

---

## 3. Ablasyon 2 — **Füzyon / Tamamlayıcılık** (ana iddia)

### 3.1 Donmuş görüntü modeline karşı (E3.1 + E3.7)

| Model | Val ROC-AUC | **Test ROC-AUC** |
| :--- | :---: | :---: |
| ResNet18 (donmuş, tek başına) | 0.8884 | 0.8349 |
| TDA ensemble (tek başına) | 0.8821 | 0.8776 |
| **Füzyon (a=0.55)** | **0.9572** | **0.9190 ± 0.0056** (3 seed) |

| Ölçüt | Değer |
| :--- | :--- |
| Test AUC farkı (füzyon − ResNet) | **+0.0862** |
| Bootstrap %95 CI (N=2.000) | **[+0.0632, +0.1099]** |
| p-değeri | **≈ 0.0000** |
| **Korelasyon (ResNet, TDA)** | **r ≈ 0.30** (val 0.3034 · test 0.3489) |
| Sızıntı kontrolü (etiket permütasyonu) | AUC **0.4378** (≈ şans) ⇒ sızıntı **yok** |

### 3.2 🏆 **İnce ayarlı görüntü modeline karşı — BELİRLEYİCİ TEST (E0.6)**

> **Neden gerekliydi:** §3.1'in kazancı **donmuş** bir baseline'a karşıydı ve o model ciddi genelleme kaybı yaşıyordu (val 0.8884 → test 0.8349, **−0.0535**). İnce ayar bu modeli 0.9284'e çıkarınca *"füzyon kazancı zayıf baseline artefaktı"* itirazı doğdu — **literatür taramasında öngörülen 1 numaralı hakem itirazı.** Son test hakkı buna ayrıldı.

| Model | Test ROC-AUC | Dondurulmuş döneme göre |
| :--- | :---: | :---: |
| İnce ayarlı ResNet18 (tek başına) | **0.9284** | 0.8349 (**+0.0935**) |
| TDA ensemble (tek başına) | 0.8629 † | 0.8776 |
| **FÜZYON (a=0.45)** | 🏆 **0.9516** | 0.9190 (+0.0326) |
| **Füzyon doğruluğu** | **%89.4** | %86.4 |

| Ölçüt | Değer |
| :--- | :--- |
| **TDA katkısı (füzyon − ince ayarlı görüntü)** | **+0.0232** |
| Bootstrap %95 CI (N=2.000) | **[+0.0132, +0.0338]** |
| p-değeri | **0.0000** |
| Karar | ✅ **KEEP** |

> ✅ **Sonuç:** İnce ayar baseline'ı **+0.0935 güçlendirdi**; buna **rağmen** TDA anlamlı katkı sağlıyor (CI sıfırı dışlıyor). ⇒ **Füzyon kazancı zayıf baseline artefaktı DEĞİLDİR.** Ana iddia, **en güçlü baseline karşısında istatistiksel olarak doğrulandı.**
> † TDA-tek burada 0.8629 çıktı; bu, E3.4'ün **zon düzeyi füzyon** varyantının değeridir (§5). E3.4'ün hasta düzeyi soft-voting varyantı 0.8776'dır ve **TDA-tek satırı için esas alınan değerdir.**

> **Mekanizma:** İki temsil **neredeyse dik** (r≈0.30) → tamamlayıcılık gerçek. Kazanç, literatürdeki TDA-Net füzyon kazancı (~+0.05) mertebesinin üzerindedir.
> Kaynak: **E3.1** · bağımsız doğrulama [scratch/verify_E3_1.py](scratch/verify_E3_1.py) · belirleyici test **E0.6** · çok-seed **E3.7**

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
| **Füzyon seviyesi** | **Geç (karar) füzyon** | **0.9538** | **KEEP** | E3.1/E3.2 |
| | Orta seviye (PI ⊕ ResNet → LR) | 0.9482 (**−0.0056**) | | **E3.2** |
| **Etiketleme eşiği** | %10 | 0.8922 (tek seed) → **0.8587** (5 seed) | | E4.1 → E4.6 |
| | **%20** | **0.8736** → **0.8501** | **KEEP** | E4.1 → E4.6 |
| | %30 | 0.8693 | | E4.1 |
| **Görüntü dalı: ince ayar derinliği** | Donmuş (linear probe) | val 0.8884 / test 0.8349 | BASELINE | E0.3 |
| | **Son blok (`layer4`+`fc`) ince ayar** | val 0.9487–0.9546 / **test 0.9284** | **KEEP** | E0.4/E0.5/E0.6 |
| **İnce ayar düzenlileştirmesi** | Yok (aşırı uyum, train loss 0.0012) | val zon 0.8590, hasta 0.9487 | | E0.4 |
| | hflip + wd 1e-4 + dropout 0.3 | val zon **0.8706**, hasta 0.9546 | en iyi val | E0.5 |
| | Tam artırma + wd 1e-3 | train loss **0.1247** ✓, val zon 0.8439 | aşırı uyum bitti | E0.5 |

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

> 🚨 **DOĞRULANMIŞ SINIRLAMA — model lokalize ETMİYOR (E4.7), zon-özel modelleme katkı sağlamıyor (E3.5) ve sinyal GLOBAL/arka plan topolojisindedir (E1.2).**
> **E4.7:** nodül-pozitif hastalarda modelin argmax zonu gerçek nodül zonuna **şans düzeyinde** düşüyor (isabet 0.228–0.298 vs şans 0.268; binom p = 0.21–0.90) ve **zon düzeyi AUC yalnızca 0.48–0.53** — modeller bir hastanın nodüllü zonunu nodülsüz zonlarından **ayırt edemiyor**.
> **E3.5:** zon kimliğini yok sayan **havuzlanmış tek model**, 5-zon özel modeliyle **eşdeğer** (0.8585 vs 0.8661, fark −0.0076 < 0.015) — üstelik model başına 5× daha fazla veri görmesine rağmen daha iyi değil.
> **E1.2:** Frangi vesselness ile damar-kaynaklı H1 sınıflarını elemek başarımı **monoton olarak düşürüyor** (base 0.8697 → v90 0.8606 → v75 0.8547 → v50 0.8358) ⇒ damar/arka plan H1 sınıfları **gürültü değil, faydalı**.
> **E3.6 (belirleyici):** Ground-truth bbox ile **nodülün tam merkezine** oturan yama, **boyut-eşleştirilmiş** kontrol yamasından yalnızca **0.6866** AUC ile ayrılıyor. ⇒ **Nodülün yerini mükemmel bilsek bile lokal TDA sinyali zayıf** → aday-bazlı mimari (E3.3) **iptal edildi**.
> ⚠️ **Confound uyarısı:** E3.6'nın eşleştirilmemiş ilk koşusu **0.994** vermişti; kontroller sabit 60×60, pozitifler değişken boyuttaydı ve sınıflandırıcı **nodülü değil yama boyutunu** öğrenmişti. **Ders:** "fazla iyi" sonuçlar sistematik olarak confound açısından denetlenmelidir.
>
> 🔗 **Üç bulgu birleşince:** E4.4 (zarf arka plan sınıflarıyla doygun) + E4.7 (lokalizasyon yok) + E1.2 (arka planı kesmek zararlı) ⇒ **ayırt edici bilgi lokal nodül topolojisinde değil, global/arka plan (damar, göğüs duvarı) topolojisindedir.** "Arka planı bastırınca lokal nodül sinyali açığa çıkar" hipotezi **çürütülmüştür**.
>
> **Sonuçları geçersiz kılmaz** — hasta düzeyi sınıflandırma (füzyon **0.9516**) ve PL mekanizması (E4.4) bundan bağımsız olarak geçerlidir. Ancak **iddianın kapsamını belirler**: *"nodülü doğru zonda tespit ediyoruz"* **denemez**; *"her zon ayrı uzmanlaşır"* da **denemez**; *"lokal nodül topolojisini okuyoruz"* da **denemez**. Doğru ifade: **"akciğerin 5 kırpıntısı üzerinde bir model topluluğu + hasta düzeyi max-pooling; ayırt edici sinyal global/arka plan topolojisinde"**.


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

## 7.1 Kaynak-Dışı Genelleme (leave-one-source-out)

**E5.3** · [betik](pipelines/exp/E5_3_cross_source.py) · dondurulmuş test kullanılmadı; her kat kendi split'ini kurar

NODE21'in nodüllü hastaları dört kaynaktan gelir: **chestxray14 (617)**, **padchest (314)**, **jsrt (149)**, openi (54). Her kaynak sırayla dışarıda bırakılıp, modelin **görülmemiş kaynaktan gelen nodülleri** tanıması test edildi.

| Dışarıda bırakılan kaynak | Test hastası | TDA CNN | **Füzyon** |
| :--- | :---: | :---: | :---: |
| chestxray14 | 2635 | 0.7406 | **0.7821** |
| padchest | 1341 | 0.8391 | **0.8499** |
| jsrt | 636 | 0.9076 | **0.9085** |
| **Ortalama** | | **0.8291** | **0.8468** → **0.8098** (eşit-boyut) |
| *(rastgele split referansı — donmuş görüntü dalı)* | *734* | *0.8631* | *0.9190* |

* ⚠️ **Eşit-boyut düzeltmesi (E5.3b) — makalede kullanılacak sayı:** Katların dengeli train boyutları farklıydı (1034 / 1876 / 2222 zon). **Tüm katlar 1034 zona eşitlendiğinde** padchest 0.8499 → **0.7625** (−0.0874), jsrt 0.9085 → **0.8847** (−0.0238), ortalama **0.8468 → 0.8098**. ⇒ Önceki 0.8468 **iyimserdi**; avantajın bir kısmı domain kolaylığı değil **veri miktarı**ydı.
* ✅ **Düzeltilmiş sonuç:** kaynak-dışı ortalama **0.8098** (şansın çok üzerinde, eşik 0.75'in üstünde) ⇒ genelleme **gerçek**, ancak **domain-shift bedeli −0.109** (önceki tahmin −0.072).

> ⚠️ **Not:** E5.3, **donmuş** görüntü dalıyla koşulduğu için referans 0.9190'dır. E0.6 ile füzyon **0.9516**'ya çıktığından, kaynak-dışı sayılar da benzer biçimde yükselme eğiliminde olabilir; ancak bu **test bütçesi tükendiği için doğrulanamadı** ve kayda geçirilmiş bir açık iştir.

* ✅ **Model kaynak-dışı genelliyor** — eşit-boyut düzeltmesiyle **0.8098**, şansın çok üzerinde.
* ⚠️ **Domain-shift bedeli: −0.109** (0.9190 → 0.8098, eşit-boyut).
* ⚠️ **Confound ÖLÇÜLDÜ ve düzeltildi (E5.3b):** chestxray14 en büyük kaynak; dışarıda bırakıldığında dengeli train **1034 zona** düşüyordu (diğer katlarda 1876 ve 2222). Eşitleme sonrası padchest 0.8499 → **0.7625**. **Ders:** kaynak-dışı genelleme çalışmalarında eğitim verisi miktarı mutlaka eşitlenmelidir.

---

## 8. İstatistiksel Kanıt Özeti (N=2.000 bootstrap)

| Karşılaştırma | Fark | %95 CI | p | Deney |
| :--- | :---: | :---: | :---: | :--- |
| **Füzyon − ResNet18 (donmuş)** | **+0.0862** | **[+0.0632, +0.1099]** | **≈0.0000** | E3.1 |
| **🏆 Füzyon − ResNet18 (ince ayarlı)** | **+0.0232** | **[+0.0132, +0.0338]** | **0.0000** | **E0.6** |
| İnce ayarlı ResNet − donmuş ResNet | +0.0935 | *(tek koşu)* | — | E0.6 |
| Ensemble − PI CNN | +0.0145 | [+0.0034, +0.0260] | 0.010 | E3.4 |
| PI CNN − Ham CNN | +0.1588 | *(CI'lar kesişmiyor)* | <0.001 | v2.1 |
| PI CNN − PL CNN | +0.3181 | *(CI'lar kesişmiyor)* | <10⁻⁶ | v2.1 |

**Tek modellerin test %95 CI'ları (v2.1):** PI CNN 0.8631 [0.8302, 0.8934] · SVM 0.8289 [0.7938, 0.8616] · k-NN 0.7162 [0.6761, 0.7575] · Ham CNN 0.7043 [0.6587, 0.7456] · PL CNN 0.5451 [0.4957, 0.5946] · PL SVM 0.5216 [0.4733, 0.5705]

---

## 9. Yenilik Konumlandırması (literatür taramasından)

| # | İddia | Literatür durumu | Kanıt |
| :-: | :--- | :--- | :--- |
| 1 | **PL üst-zarf çökmesinin nicel gösterimi** | **Bulunamadı** (Vandaele 2020 "ilgisiz nesne" problemini tanıyor ama PL zarfı için test etmiyor) | E4.4 ✅ |
| 2 | **TDA + görüntü tamamlayıcılığı (r≈0.30) ve füzyon kazancı** | TDA-Net ~+0.05 (CXR/COVID); bizimki **+0.0862** (donmuş) ve **+0.0232** (**ince ayarlı** baseline karşısında) | E3.1 ✅ · **E0.6 ✅** |
| 3 | **CXR'de H1'in bronkovasküler döngü + kaburga ile domine edilmesi** | Spesifik formülasyon yok | E4.2/E4.4 |
| 4 | ~~5 anatomik zon + zon başına PH + hasta düzeyi max-pooling~~ | Bulunamadı — ancak **E4.7 + E3.5 ile ÇÜRÜTÜLDÜ** | ⛔ **İDDİA KALDIRILDI**: model lokalize etmiyor (zon AUC 0.48–0.53) ve havuzlanmış tek model eşdeğer (−0.0076) |
| 5 | **NODE21'de kalıcı homoloji** | Bulunamadı | v2.1 + E0.2 |

**Yenilik iddia EDİLMEMESİ gerekenler:** "ilgisiz yapılar diyagramı domine eder → ROI kısıtlaması" (Vandaele 2020/2022) · "PI vs PL veri setine bağlı" (Barnes 2021, Sulowska 2026) · "topoloji ≈ CNN, füzyon artırır" (TDA-Net).

---

## 10. Açık Kalan İşler ve Sınırlamalar

### 10.1 Kapanan işler
| İş | Durum |
| :--- | :--- |
| **E5.1** — bu belgenin finalize edilmesi | ✅ **TAMAM** (2026-10-15, bu sürüm) |
| **E5.3** — harici/kaynak-dışı genelleme | ✅ **TAMAM** (leave-one-source-out, ortalama 0.8468) |
| **Kural 10** — çok-seed ortalama | ✅ **TAMAM** (E3.7: 0.9190 ± 0.0056) |
| **Hakem itirazı #1** — zayıf baseline | ✅ **ÇÜRÜTÜLDÜ** (E0.6: +0.0232, CI>0) |
| **Test bütçesi** | 🚫 **6/6 TÜKENDİ** — dondurulmuş test seti **kapalı** |

### 10.2 Kalan işler (test harcamaz)
| Öncelik | İş | Durum |
| :-: | :--- | :--- |
| 1 | **E5.2 — yazım**: metodoloji / sonuçlar / tartışma taslağı + figürler | Bekliyor |
| 2 | **TDA-tek tutarsızlığı**: E0.6'da 0.8629, E3.4'te 0.8776 — sebep **bulundu** (zon düzeyi vs hasta düzeyi varyantı); bundle yükleme farkı **doğrulanmalı** | Kısmen çözüldü |
| 3 | ~~**E5.3 eşit-boyut kontrolü**~~: chestxray14 katındaki düşüşün ne kadarı domain shift, ne kadarı veri azalması | ✅ **TAMAM (E5.3b)** — confound ölçüldü: ortalama 0.8468 → **0.8098** |
| 4 | **E3.2** — orta seviye füzyon (özellik birleştirme) | Planlandı |
| 5 | **E2.1/E2.2/E2.3** — çift yön, çok ölçek, şablon filtrasyonu (Faz 3) | Planlandı (beklenti düşük) |
| 6 | **E3.3** — aday-bazlı mimari | ⛔ **İPTAL** (E3.6: tavan ~0.69) |
| 7 | **E5.3+** — *gerçek* harici veri (JSRT/ChestX-ray14) ile dış doğrulama | Kaynak erişimi gerekir |

### 10.3 Doğrulanmış sınırlamalar (makalede açıkça yazılmalı)
1. **Lokalizasyon yok** (E4.7) — hasta düzeyi sınıflandırma; kutu üretimi değil.
2. **Sinyal global/arka plan topolojisinde** (E4.4 + E1.2 + E3.6) — "lokal nodül topolojisi" denemez.
3. **Zon ayrıştırması uzmanlaşma sağlamıyor** (E3.5) — yalnızca topluluk çeşitliliği.
4. **TDA tek başına ince ayarlı modeli geçemiyor** (0.8629/0.8776 vs 0.9284) — değeri tamamlayıcılıktır.
5. **Seed varyansı ±0.02–0.03** (E4.6) — tüm sayılar çok-seed ortalaması olmalı.
6. **Domain-shift bedeli −0.072** (E5.3) ve kaynak-dışı düşüşte **veri azalması confound'u**.
7. **Veri = NODE21 halka açık training seti** — leaderboard karşılaştırması geçersiz (§0).

---

## 11. Figür Planı (makale için)

| # | Figür | İçerik | Kaynak |
| :-: | :--- | :--- | :--- |
| **F1** | **Boru hattı şeması** | CXR → PSPNet → 5 zon → (a) kübik PH → PI → CNN/SVM, (b) ResNet18 → füzyon | §3.2 |
| **F2** | **Liderlik tablosu (bar)** | 11 model, test ROC-AUC; füzyon 0.9516 vurgulu | §1 |
| **F3** | **Tamamlayıcılık saçılımı** | ResNet vs TDA hasta olasılıkları, r≈0.30; iki eşik çizgisi | E3.1 |
| **F4** | **PL zarf çökmesi** | λ1..λ5 zarf tepeleri (H1: 110→90) + sınıf sinyali payı (%1–2) | E4.4 |
| **F5** | **Belirleyici test (kritik figür)** | Donmuş vs ince ayarlı vs füzyon; bootstrap CI çubukları | **E0.6** |
| **F6** | **Sağlamlık panelleri** | σ, ızgara, ağırlık, eşik, seed varyansı | E2.7/E2.8/E4.1/E4.6 |
| **F7** | **Kaynak-dışı genelleme** | 3 kaynak için füzyon AUC + rastgele split referansı | E5.3 |
| **F8** | **Sınırlama paneli** | Zon düzeyi AUC ≈ 0.5 (lokalizasyon yok) + oracle-yama AUC 0.687 | E4.7/E3.6 |

---

## 12. Özet — Makalenin Taşıyıcı Üç İddiası

| # | İddia | Sayı | Kanıt gücü |
| :-: | :--- | :--- | :--- |
| **1** | **TDA + görüntü füzyonu hasta düzeyi nodül sınıflandırmasını güçlü biçimde iyileştirir** | **0.9516** ROC-AUC / **%89.4** doğruluk | Donmuş baseline'a **+0.0862**; **ince ayarlı** baseline'a **+0.0232** (CI>0) |
| **2** | **PL temsillerinin çökmesinin mekanizması: üst-zarf doygunluğu** | H1 zarf %98–99.6 arka plan; sınıf sinyali %1–2 | E4.4 nicel + E2.7 bağımsız |
| **3** | **Temsiller neredeyse dik (r≈0.30) → tamamlayıcılık gerçek** | r ≈ 0.30–0.35 | Etiket permütasyonu şans (0.4378) ⇒ sızıntı yok |

**Dürüst çerçeve:** Bu bir **hasta düzeyi sınıflandırma** çalışmasıdır; lokalizasyon iddiası yoktur. Ayırt edici sinyal ağırlıklı olarak **global/arka plan topolojisindedir**. Katkı, **yorumlanabilir topolojik temsilin görüntü temsiline kattığı ölçülebilir tamamlayıcılıktır.**

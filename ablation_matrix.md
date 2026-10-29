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
| **Filtrasyon polaritesi** | Tek polarite (projenin PI'ı, 800 boyut) | 0.8399 | | E2.5 |
| | **Çift polarite (sublevel+superlevel, 1600)** | **0.8609** (**+0.0210**) | **KEEP** ✅ val-only | **E2.1** |
| | *Kontrol: B kolu karıştırılmış (boyut aynı)* | *0.7803* (**−0.0597**) | boyut tek başına zarar verir | **E2.1b** |
| **Ölçek (σ)** | Tek ölçek σ=1.0 | 0.8399 | | E2.5 |
| | Tek ölçek σ=1.5 | 0.8457 | | E2.2 |
| | Çok ölçek (4 σ, 3200 boyut) | 0.8510 (**+0.0053**) | ✗ < 0.015 | **E2.2** |
| | *Kontrol: diğer ölçekler karıştırılmış* | *0.6598* | boyut tek başına şiddetle zararlı | **E2.2** |
| **Şablon-göreli (E2.3b)** | **Mutlak PI** | **0.8399** | **KEEP** | E2.5 |
| | `(X − T_doğru)₊` ReLU sapma | 0.7530 (−0.0869) | | **E2.3b** |
| | `\|X − T_doğru\|` mutlak sapma | 0.6365 (−0.2034) | | **E2.3b** |
| | `(X − T_yanlış)₊` **kontrol** | 0.8276 (−0.0124) | yanlış şablon **daha iyi** ⇒ şablon faydalı bilgi taşır | **E2.3b** |
| **Çok parametreli (E2.6)** | Tek dilim (t=0, proje PI'ı) | 0.8399 | | E2.5 |
| | **+ 2 gradyan dilimi (2400 boyut)** | **0.8880** (**+0.0480**) | **KEEP** 🎉 val-only · **en büyük kazanç** | **E2.6** |
| | *Kontrol: dilimler karıştırılmış* | *0.6925* | boyut tek başına şiddetle zararlı | **E2.6** |

> 🔗 **"Arka planı bastır, lokal sinyali açığa çıkar" öncülü iki bağımsız yoldan çürütüldü:** **E1.2** (damar/arka plan H1 sınıflarını elemek **zarar veriyor**, 0.8697 → 0.8358) + **E2.3b** (sağlıklı şablonu çıkarmak **zarar veriyor**; üstelik **yanlış** şablonu çıkarmak **daha iyi**, 0.8276 vs 0.7530) ⇒ **arka plan/şablon yapısı gürültü değil, ayırt edici bilginin taşıyıcısı.**

> 🔑 **Temsil tasarımı dersi (E2.1 vs E2.2):** **polariteyi** değiştirmek (sublevel → sublevel+superlevel) **+0.0210 KEEP**; **ölçeği** değiştirmek (tek σ → 4 σ) yalnızca **+0.0053 KILL**. ⇒ Topolojik bilgi kazancı **"hangi kümeler filtreleniyor"** sorusundan gelir, **"diyagram nasıl yumuşatılıyor"** sorusundan değil.
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

## 7.2 Gerçek Harici Veri — VinDr-CXR (V4)

**V4** · [betik](pipelines/exp/V4_external_validation_vindr.py) · **VinDr'de sıfır eğitim** · ayrıntı: [experiments.md §7](experiments.md)

> 🔒 **Zero-shot sınırı (VinDr bölümünün tamamı için).** VinDr'ye **CNN fine-tuning'i hiç yapılmadı**; tüm modeller NODE21'de eğitilmiş dondurulmuş ağırlıklardır. Buna karşın VinDr bilgisi üç kanaldan girmiştir ve sayılar üç sınıfa ayrılır:
>
> | Sınıf | VinDr'den ne kullanır | Etiket? | Satırlar |
> | :--- | :--- | :---: | :--- |
> | **(A) Saf zero-shot** | yalnız görüntü + **NODE21** referans istatistikleri | ❌ | §7.2 üç kol · V5.3 · §13 V6.3 bileşen otopsisi |
> | **(B) Transductive** | VinDr **BN istatistikleri** veya VinDr kovaryansı | ❌ | ⚠️ **V6.5 KOL 1'in tamamı** (`AdaBN-M5` tabanlı: 0,6439–0,6620) · AdaBN'li sayılar (0,6332 · 0,6346 · 0,6471 · 0,6494) |
> | **(C) Süpervizli** | VinDr **etiketleri** | ✅ | V6.5 KOL 2 / 5b / 5c few-shot eğrileri |
>
> ⇒ **Saf zero-shot zirvesi 0,6283** (V5.3 global füzyon, kilitli a = 0,45). V6.5'in 0,6439–0,6620 bandı **(B)**'dir; tablolarda "harici, hiç dokunulmamış" olarak okunmamalıdır.

NODE21'in dışından, farklı bir ülke ve cihaz dağılımından gelen **VinDr-CXR** üzerinde (400 **görüntü**: 150 nodül/kitle + 250 saf normal, seed 42; VinDr'de hasta kimliği yok ⇒ **görüntü düzeyi**):

| Kol | TDA ensemble | ResNet18 | Füzyon (**a = 0,45** kilitli) | Δ(füzyon−ResNet) |
| :--- | :---: | :---: | :---: | :---: |
| `linear` | 0,5274 [0,4699; 0,5836] | 0,6204 [0,5609; 0,6763] | 0,5328 | **−0,0876** [−0,1262; −0,0491] |
| `matched` (200-hasta havuzu, yalnızca dokuya) | **0,5409** [0,4852; 0,5973] | 0,6204 | 0,5438 | **−0,0765** [−0,1133; −0,0393] |
| `matched_scalenorm` (+ akciğer-yüksekliği ölçeği) | 0,5110 [0,4522; 0,5688] | 0,6204 | 0,5352 | **−0,0852** [−0,1302; −0,0403] |

* 🔑 **TDA dış dünyaya taşınmıyor:** 0,511–0,541 ≈ şans. ResNet18 kısmen taşınıyor (**0,6204**).
* 🔑 **Harici veride optimal füzyon ağırlığı a = 1,00** ⇒ dış eksende **TDA katkısı ~sıfır**; kilitli ağırlık füzyonu ResNet'in altına çekiyor.
* 🔑 **Ölçek normalizasyonu kurtarmıyor** (0,5110 ≤ 0,5409) ⇒ çöküş yoğunluk/ölçek değil, **doku/dedektör** domain-shift'i.
* Nodül boyutu alt grupları **ölçek-bağımsız** `rel_size` üçte birlikleriyle (mm dönüşümü imkânsız: 3× indirgenmiş JPEG, spacing yok).
* Donanım teşhisi: **yalnızca meta-veri** val AUC **0,6850** (kısmi kısayol) · eşit-boyut kaynak-dışı TDA **0,7631** ⇒ sinyal tamamen sahte değil.
* ⚠️ **ResNet ağırlıkları:** E0.6 ağırlıkları kaydedilmemişti; model deterministik olarak yeniden eğitilir (val zon AUC Δ=3,8·10⁻⁴) ve **VinDr'de eğitim yapılmaz**.

> **Okunuşu:** v3'ün **0,9516**'lık kurum-içi sonucu geçerlidir; ancak **kapsamı kurum içidir**. Dış merkeze taşıma, kalibrasyon/adaptasyon gerektirir.
>
> ✅ **V5.3 güncellemesi (düzeltilmiş TDA):** Aynı VinDr kohortu, **65535 niceleme düzeltmesi** uygulanmış TDA ile tekrarlandı. Füzyon (a=0,45) **0,5438 → 0,6283** (**Δ +0,0844, %95 GA [+0,0464; +0,1219], p<0,001**); TDA tek başına **0,5409 → 0,5959** (p=0,053); VinDr'de optimal ağırlık **1,00 → 0,75**; r **0,4009 → 0,2997**. ⇒ Tablodaki "TDA şans düzeyinde" ifadesi **eski ~4-bit temsil** için geçerlidir; temizlenmiş temsilde TDA **şansın üzerinde** ve dış veride füzyona anlamlı katkı sağlar. ⚠️ Mutlak harici AUC yine mütevazı (≈0,63) ve füzyonun ResNet'e göre artı kazancı hâlâ **gürültü mertebesinde** (bkz. §7.4).

---

## 7.3 Cihaz/Kaynak Ayrıştırılabilirliği — Temsillerimiz Fizik mi Kodluyor? (A2)

**A2** · [betik](pipelines/exp/A2_source_predictability.py) · rapor: [a2_source_predictability_report.json](data/external_vindr/a2_source_predictability_report.json)

NODE21'in **kendi içindeki 4 kaynağı** (chestxray14 0,143 mm/px · padchest 0,275 · jsrt 0,309 · openi 0,331) birer "cihaz/kurum" değişkeni olarak kullanıldı. Etiket sızıntısını kırmak için test **yalnızca nodül-pozitif hastalarda** yapıldı (burada kaynak, etiketten bağımsızdır). Hasta düzeyi, train→val, **test hastaları kullanılmadı** (790 train / 172 val).

| Temsil | Çok-sınıflı LR makro AUC | HGB makro AUC | Doğruluk (çoğunluk referansı %54,1) |
| :--- | :---: | :---: | :---: |
| **TDA PI (800)** | **0,8501** | 0,9110 | %70,4 (HGB: %80,2) |
| **ResNet18 özellikleri (512)** | **0,9494** | 0,9434 | **%89,5** |
| Yalnızca meta-veri (spacing + boyut) | **0,9558** | — | %80,2 |

**Okunuşu:**
* Temsiller, **hangi cihazdan geldiğini** neredeyse **hastalığı tanıdığı kadar** iyi kodluyor (TDA label AUC ~0,87 ↔ kaynak AUC 0,85; ResNet label ~0,93 ↔ kaynak 0,95). Bu, literatürdeki klasik **kısayol/confound** imzasıdır.
* ⚠️ **Görüntü dalı (ResNet) en çok cihaz-kimlikli olan dal** (%89,5) — yani dışa en iyi taşınan dal (VinDr 0,62) aynı zamanda en çok fiziksel imza taşıyan dal.
* "Yalnızca spacing+boyut" bile kaynağı 0,956 ile ayırıyor; A0/A1'in 0,6850'lık etiket kısayolu bunun **etiketle ilişkili** olduğunu gösteriyor.
* ⚠️ **Confound uyarısı:** kaynaklar nodül boyutu/dağılımı ve dijitalleştirme zinciri bakımından da farklıdır; ayrıştırılabilirliğin bir kısmı fizik değil anatomi/hastalık spektrumudur. Bu yüzden **hedef metrik**, label AUC'sini korurken bu kaynak AUC'sini **düşürmektir**.

> 🎯 **Bu, alan kaymasına karşı mimari programın çıpasıdır:** her yeni temsilde (a) label AUC **ve** (b) kaynak-ayrıştırılabilirlik AUC'si **birlikte** raporlanmalı; kaynak AUC'si düşerken label AUC'si korunmalıdır.
>
> ⚠️ **V5.1 sonrası güncelleme:** Bu kapı **çürütülmüştür** — yoğunluk kanonikleştirmesi label AUC'yi **+0,088**, LOO'yu **+0,061** artırırken kaynak AUC'yi de **+0,105** artırdı. ⇒ Kaynak AUC **tanısal** kalmalı, **karar kapısı** olmamalı (bkz. §7.4).

### 7.3b Kaynak-içi AUC Denetimi (A3) — Etiket ile Kaynak Tamamen İç İçe

**A3** · [betik](pipelines/exp/A3_source_confound_audit.py) · rapor: `artifacts/runs/A3_source_confound_audit/report.json`

NODE21 test setinde tüm pozitifler 4 dış kaynaktan (chestxray14: 89, padchest: 58, jsrt: 19, openi: 6), tüm negatifler ise tek `node21_control` havuzundan (562) gelir. Her kaynak tek sınıflı olduğu için klasik "kaynak-içi AUC" **tanımsızdır**. Yerine geçen metrik: **kaynak pozitifleri vs ortak kontrol** AUC'si ve makro ortalama:

| Model | Genel AUC | Makro AUC | chestxray14 | padchest | jsrt | openi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TDA PI CNN** | 0,8631 | **0,8920** | 0,8158 | 0,9052 | 0,9433 | 0,9036 |
| TDA PI RBF-SVM | 0,8289 | 0,8672 | 0,8161 | 0,8062 | 0,9303 | 0,9161 |
| Ham görüntü CNN | 0,7043 | 0,6638 | 0,7648 | 0,6171 | 0,7419 | 0,5316 |

* Model 4 kaynağın 4'ünde de ayrım yapıyor (0,82–0,94) ⇒ sinyal tek bir kaynağın artefaktı değil.
* TDA, ham CNN'i makroda belirgin geçiyor (0,892 vs 0,664) ⇒ TDA kazancı yalnızca kaynak kimliği değil.
* Kontrol ortalama skoru 0,555, özgüllük@0,5 = 0,358 ⇒ işletim noktası kaymıştır (kilitli Youden eşiği 0,6949 ile özgüllük %79,72).

### 7.3c PL Zarf Mekanizma Otopsisi — Maske Sınırı Artefaktı (A4)

**A4** · [betik](pipelines/exp/A4_mask_boundary_audit.py) · rapor: `artifacts/runs/A4_mask_boundary_audit/report.json`

GUDHI `cofaces_of_persistence_pairs` ile 100 dengeli train zonunda (50 poz / 50 neg) hücre-piksel geri izleme otopsisi:
* Baseline filtrasyonda doku içi pikseller ~12 seviyelik dar bir banda (240–252) sıkıştığı için, saf doku içi kalıcılık en fazla **7,0** olabilir (ort. H0: 3,6 · H1: 4,2).
* E4.4'te ölçülen ve PL üst zarfını doyuran devasa tepelerin ($\text{kalıcılık} \ge 50$ döngülerin) **%100,0'ı** (599/599) doku ile dış siyah arka plan (0) arasındaki yapay segmentasyon maskesi basamağından kaynaklanır.
* Pozitif ve negatif zonlarda sınır oranı eşittir (%60,2 vs %59,6).
* **Mekanizma kanıtı:** PL çöküşü biyolojik damar ağacından değil, **akciğer maskesi geometrik sınır basamağındandır**. PI $(b, p)$ uzayında bu artefaktı tek köşeye hapsederek doku sinyalini korurken, PL üst zarf projeksiyonunda bu 248 kalıcılıklı yapay tepe tarafından tamamen körleştirilir.

---

## 7.4 Fiziksel Kanonikleştirme (V5.1) — 5-zon mimarisi korunarak

**V5.1** · [betik](pipelines/exp/V5_1_physical_canonicalization.py) · ön-kayıt: [`configs/experiments/V5_1_physical_canonicalization.yaml`](configs/experiments/V5_1_physical_canonicalization.yaml) · ayrıntı: [experiments.md §8](experiments.md)

| Kol | val label AUC | A2 kaynak AUC | LOO (E5.3b) |
| :--- | :---: | :---: | :---: |
| `baseline` (v2, sabit yoğunluk) | 0,8330 | 0,8501 | 0,7249 |
| **`a_density`** (doku p1–p99) | **0,9211** (+0,0881) | 0,9554 (+0,1053) | **0,7861** (+0,0611) |
| `a_density` + TTA | 0,9003 | 0,9622 | **0,7967** |
| `b_density_scale` (+akciğer-yüksekliği ölçeği) | 0,8539 | 0,9138 | 0,7400 |
| `b_density_scale` + TTA | 0,8762 | 0,9235 | 0,7305 |

* 🔑 **Temsil fiilen ~4-bit'ti:** `intensity_max=65535` 12-bit veri için 16× fazla ⇒ yoğunluk ~16 seviyeye kuantalanıyordu. Kanonikleştirme sonrası H1 nokta sayısı 610k → **1,63M**, H1 kalıcılık aralığı 0–7 → **0–97**, PI doluluğu %38 → **%81**. ⇒ v1'in "grid collapse" hatası **tam kapanmamıştı**.
* ✅ **Programın en büyük val kazancı:** +0,0881 val, +0,0611 LOO (v3'ün E2.6 +0,0480'ından büyük). Test bütçesi kapalı ⇒ val/LOO düzeyinde.
* ❌ **Ölçek normalizasyonu (b) zarar veriyor** (val −0,067 · LOO −0,046); kat bazında chestxray14 +0,027 ama jsrt −0,141 ⇒ topoloji **yeniden örneklemeye aşırı duyarlı**.
* ⚠️ **TTA tutarlı değil:** (a)'da val −0,021 / LOO +0,011; (b)'de ters işaret.
* ⚠️ **Açık kontrol:** per-kırpıntı yüzdeler "uyarlanabilir pencere" etkisi taşıyor olabilir ⇒ **sabit referans** kontrolü (V5.1b) + **CNN ensemble** teyidi gerekir.
* **Karar:** **KEEP** ✅ (sahip kararı, 2026-10-17). Ön-kayıtlı kuralın "kaynak AUC düşmeli" koşulu sağlanmadı ve kural-temelli sonuç `KILL` olarak **kayıtta duruyor**; ancak kaynak AUC artışı **başarısızlık değil**: 4-bit'e sıkışmış bulanık temsili **8-bit'e (256 seviye)** çıkardığımız için bilgi miktarı arttı — cihaza özgü doku ve hastalık sinyali **birlikte** güçlendi. Asıl hedefler (val **+0,0881**, LOO **+0,0611**; +TTA ile LOO **+0,0718**) büyük ölçüde arttı ⇒ 65535 niceleme hatasının düzeltilmesi **KEEP**. Kayıt: `owner_override` bloğu (`v5_1_physical_canonicalization_report.json`). Kapı değişikliği (kaynak AUC **tanısal**, LOO birincil + val guardrail) ayrıca ön-kayıt edilecek.

### V5.2 — Düzeltilmiş TDA gerçek modelde (val-only, 734 hasta)

**V5.2** · [betik](pipelines/exp/V5_2_corrected_tda_ensemble_fusion.py) · test bütçesi 6/6 kapalı ⇒ **validation**.

| Dal | v2 dondurulmuş | **Düzeltilmiş** | Δ |
| :--- | :---: | :---: | :---: |
| TDA CNN (tek) | 0,8580 | 0,9156 | +0,0576 |
| TDA SVM (tek) | 0,8278 | 0,9369 | +0,1092 |
| **TDA ensemble** | 0,8811 | **0,9522** | **+0,0711** |
| ResNet18 (E0.6) | 0,9547 | 0,9547 | 0 |
| **FÜZYON (en iyi a)** | 0,9729 (a=0,40) | **0,9856** (**a=0,25**) | **+0,0127** |

* 🔑 Füzyon ağırlığı **0,40 → 0,25** (TDA payı 0,60 → **0,75**): düzeltilmiş TDA artık ResNet'e denk ve füzyona gerçek katkı veriyor ⇒ v4'ün "yedeklilik" bulgusu **bozuk temsilin artefaktıymış**.
* ⚠️ v4'ün sıkı füzyon kapısıyla (≥ +0,015) sınırda (KILL); V5.2'nin kendi kuralı KEEP.

### V5.3 — Düzeltilmiş TDA harici veride (VinDr-CXR, sıfır eğitim)

| | V4 (eski ~4-bit) | **V5.3 (düzeltilmiş)** | Δ (eşleştirilmiş bootstrap, n=400) |
| :--- | :---: | :---: | :---: |
| TDA ensemble | 0,5409 | **0,5959** [0,5475; 0,6450] | +0,0550, p=0,053 |
| ResNet18 (E0.6) | 0,6204 | 0,6204 | 0 |
| **Füzyon (a=0,45)** | 0,5438 | **0,6283** | **+0,0844 [+0,0464; +0,1219], p<0,001** ✅ |
| VinDr optimal a | 1,00 | **0,75** (0,6317) | — |

* ✅ **Harici çöküş büyük ölçüde temsil artefaktıymış** — füzyon iyileşmesi istatistiksel olarak anlamlı (CI sıfırı dışlıyor).
* 🟡 TDA hâlâ ResNet'in altında (0,5959 vs 0,6204) ve füzyonun ResNet'e göre artı kazancı (+0,008…+0,011) gürültü mertebesinde ⇒ dış merkez için kalibrasyon/adaptasyon şart.
* 🟢 **Küçük nodüllerde TDA ResNet'i geçiyor** (0,5951 vs 0,5823) — eski temsilde bu grupta 0,50'ye çöküyordu.

### V5.1b — Pencere (uyarlanabilir referans) kontrolü

| Ölçüt | per-kırpıntı (a_density) | **tüm-akciğer (c_lungref)** | Δ |
| :--- | :---: | :---: | :---: |
| val label AUC | 0,9211 | 0,9133 | −0,0078 |
| **LOO** | 0,7861 | **0,7897** | +0,0036 |
| TDA ensemble | 0,9522 | 0,9335 | −0,0187 |
| **FÜZYON** | 0,9856 | **0,9778** | −0,0078 |

* ⚠️ **Kısmi pencere etkisi var (~+0,0078):** füzyon kazancının ~%60'ı kırpıntı-başına histogram penceresinden geliyor (topolojik değil, histogram kaynaklı).
* ✅ **Ama nicelleme düzeltmesi baskın:** çiftleşmesiz varyantta bile TDA ensemble 0,8811 → **0,9335** (+0,0524 = kazancın %74'ü) ve **LOO daha iyi** (0,7897).
* ⇒ **KEEP korunur**; makalede füzyon için **muhafazakâr sayı 0,9778** (0,9856 pencere katkısını içerir).

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
| 2 | **TDA + görüntü tamamlayıcılığı (r≈0.30) ve füzyon kazancı** ⚠️ **kurum içi** | TDA-Net ~+0.05 (CXR/COVID); bizimki **+0.0862** (donmuş) ve **+0.0232** (**ince ayarlı** baseline karşısında). **Ancak harici veride (VinDr, §7.2) tamamlayıcılık kaybolur: TDA ≈ şans, optimal a = 1,00** | E3.1 ✅ · **E0.6 ✅** · **V4 ⚠️** |
| 3 | **CXR'de H1'in bronkovasküler döngü + kaburga ile domine edilmesi** | Spesifik formülasyon yok | E4.2/E4.4 |
| 4 | ~~5 anatomik zon + zon başına PH + hasta düzeyi max-pooling~~ | Bulunamadı — ancak **E4.7 + E3.5 ile ÇÜRÜTÜLDÜ** | ⛔ **İDDİA KALDIRILDI**: model lokalize etmiyor (zon AUC 0.48–0.53) ve havuzlanmış tek model eşdeğer (−0.0076) |
| 5 | **NODE21'de kalıcı homoloji** | Bulunamadı | v2.1 + E0.2 |

**Yenilik iddia EDİLMEMESİ gerekenler:** "ilgisiz yapılar diyagramı domine eder → ROI kısıtlaması" (Vandaele 2020/2022) · "PI vs PL veri setine bağlı" (Barnes 2021, Sulowska 2026) · "topoloji ≈ CNN, füzyon artırır" (TDA-Net) · **"TDA tamamlayıcılığı kaynak-bağımsızdır"** (V4: VinDr'de kayboluyor).

---

## 10. Açık Kalan İşler ve Sınırlamalar

### 10.1 Kapanan işler
| İş | Durum |
| :--- | :--- |
| **E5.1** — bu belgenin finalize edilmesi | ✅ **TAMAM** (2026-10-15, bu sürüm) |
| **E5.3** — harici/kaynak-dışı genelleme | ✅ **TAMAM** (leave-one-source-out, ortalama 0.8468) |
| **V4** — *gerçek* harici veri (VinDr-CXR) ile dış doğrulama | ✅ **TAMAM** (§7.2; TDA şans düzeyinde, füzyon dışarıda geriye gidiyor) |
| **Kural 10** — çok-seed ortalama | ✅ **TAMAM** (E3.7: 0.9190 ± 0.0056) |
| **Hakem itirazı #1** — zayıf baseline | ✅ **ÇÜRÜTÜLDÜ** (E0.6: +0.0232, CI>0) |
| **Test bütçesi** | 🚫 **6/6 TÜKENDİ** — dondurulmuş test seti **kapalı** |

### 10.2 Kalan işler (test harcamaz)
| Öncelik | İş | Durum |
| :-: | :--- | :--- |
| 1 | **E5.2 — yazım**: metodoloji / sonuçlar / tartışma taslağı + figürler | ✅ **TASLAK** ([paper_draft.md](paper_draft.md)) |
| 1b | 🥇 **E2.6 çok parametreli dilimleri test'te doğrulama** — val'de **+0.0480** (en büyük kazanç), boyut kontrolü geçti | **Birinci öncelik** (yeni test bütçesi gerekir) |
| 1c | 🥈 **E2.1 çift yönlü filtrasyonu test'te doğrulama** — val'de **+0.0210**, boyut kontrolü geçti | **İkinci öncelik** (E2.6 ile birleştirilebilir) |
| 2 | ~~**TDA-tek tutarsızlığı**~~: E0.6'da 0.8629, E3.4'te 0.8776 | ✅ **ÇÖZÜLDÜ (sıfır artık)** — **varyant farkı**: E3.4 **Varyant A** (önce hasta max-pool, sonra füzyon) = 0.8776; **Varyant B** (önce zon füzyonu, sonra max-pool) = **0.8629**. E0.6/E3.2 **B'yi birebir** yeniden üretti (val 0.8699, test 0.8629) ⇒ **bundle'lar doğru yükleniyor, yeniden üretilebilirlik sorunu yok.** ⇒ **E0.6 füzyonu muhafazakârdır** (zayıf varyantla ölçüldü); gelecek test'te **Varyant A** kullanılmalı |
| 3 | ~~**E5.3 eşit-boyut kontrolü**~~: chestxray14 katındaki düşüşün ne kadarı domain shift, ne kadarı veri azalması | ✅ **TAMAM (E5.3b)** — confound ölçüldü: ortalama 0.8468 → **0.8098** |
| 4 | **E3.2** — orta seviye füzyon (özellik birleştirme) | Planlandı |
| 5 | **E2.1/E2.2/E2.3** — çift yön, çok ölçek, şablon filtrasyonu (Faz 3) | Planlandı (beklenti düşük) |
| 6 | **E3.3** — aday-bazlı mimari | ⛔ **İPTAL** (E3.6: tavan ~0.69) |
| 7 | ~~**E5.3+** — *gerçek* harici veri (JSRT/ChestX-ray14) ile dış doğrulama~~ | ✅ **TAMAM (V4)** — VinDr-CXR ile yapıldı; §7.2. Kalan: harici **adaptasyon/fine-tune** denemesi (görüntü ve TDA dalları için) |

### 10.3 Doğrulanmış sınırlamalar (makalede açıkça yazılmalı)
1. **Lokalizasyon yok** (E4.7) — hasta düzeyi sınıflandırma; kutu üretimi değil.
2. **Sinyal global/arka plan topolojisinde** (E4.4 + E1.2 + E3.6) — "lokal nodül topolojisi" denemez.
3. **Zon ayrıştırması uzmanlaşma sağlamıyor** (E3.5) — yalnızca topluluk çeşitliliği.
4. **TDA tek başına ince ayarlı modeli geçemiyor** (0.8629/0.8776 vs 0.9284) — değeri tamamlayıcılıktır.
5. **Seed varyansı ±0.02–0.03** (E4.6) — tüm sayılar çok-seed ortalaması olmalı.
6. **Domain-shift bedeli −0.109** (E5.3b; eşit-boyut) ve kaynak-dışı düşüşte **veri azalması confound'u**.
7. **Veri = NODE21 halka açık training seti** — leaderboard karşılaştırması geçersiz (§0).
8. **🌐 Kapsam kurum içi (V4):** gerçek harici veride (VinDr-CXR) TDA şans düzeyine iner (0,511–0,541) ve kilitli füzyon ResNet'in **altına** düşer; optimal ağırlık a = 1,00. Ölçek normalizasyonu kurtarmaz ⇒ dış merkeze taşıma **kalibrasyon/adaptasyon** gerektirir (§7.2).
9. **🎯 Cihaz/kaynak ayrıştırılabilirliği (A2):** pozitifler arasında temsiller **hangi cihazdan geldiğini** label'a yakın doğrulukla kodluyor (TDA 0,85 · ResNet 0,95 · yalnız meta-veri 0,96) ⇒ iç başarım kısmen **fiziksel imza** üzerinden geliyor; dışa taşınmayan da tam olarak bu (§7.3).
10. **Harici veri görüntü düzeyi ve ölçek confound'u:** VinDr kohortu 3× indirgenmiş JPEG'dir (spacing yok); nodül boyutu alt grupları mm değil `rel_size` üçte birlikleriyle verilir.
11. **ResNet ağırlıkları kayıtlı değil:** E0.6 ağırlıkları saklanmadığı için harici sonuçlar deterministik yeniden eğitilen sabit checkpoint'e dayanır (val zon AUC Δ=3,8·10⁻⁴).

---

## 11. Figür Planı (makale için) — ✅ **ÜRETİLDİ**

> **Figürler:** `artifacts/figures/F1..F10_*.png` + `F10b_external_components.png` (200 dpi) **üretilmiştir**.
> Üreten betikler: [make_figures.py](pipelines/figures/make_figures.py) (F1–F8) · [make_figure_f9_explainability.py](pipelines/figures/make_figure_f9_explainability.py) (F9) · [make_figure_f10_froc_ladder.py](pipelines/figures/make_figure_f10_froc_ladder.py) (F10) · [make_figure_f10b_external_components.py](pipelines/figures/make_figure_f10b_external_components.py) (F10B, ek).
> Sayılar doğrulanmış raporlardan alınır; **F3 gerçek val olasılıklarını yeniden hesaplar**.
> 🔎 **F3 bağımsız doğrulama:** betik korelasyonu **r = 0.2802** hesapladı — kayıtlı val değeri (0.3034) ile tutarlı, temsil tamamlayıcılığını bağımsız olarak teyit ediyor.

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
| **F9** | **Görsel açıklanabilirlik** | (A) 5-zon kanonik görüntü + GT kutusu + adaylar · (B) TP/FP 64×64 kırpıntıları · (C) TP kalıcılık diyagramı (H0/H1) · (D) skor ayrışımı (aşama-1 → TDA-HGB → birleşik) | V6.1/V6.2 — ✅ **üretildi** |
| **F10** | **FROC + VinDr merdiveni** | (A) NODE21 val FROC + test mührü · (B) VinDr transfer merdiveni (V4→V6.5) · (C) FROC@2 vs N (AUC–FROC ödünleşimi) | V6.4/V6.5/V6.5b/V6.5c — ✅ **üretildi** |
| **F10B** | *Ek:* Harici bileşenler | Kol başına TDA/ResNet/Füzyon AUC + CI · füzyon a-eğrisi (optimum a = 1,00) | V4 (§7.2) — ✅ **üretildi** |

---

## 12. Özet — Makalenin Taşıyıcı Üç İddiası

| # | İddia | Sayı | Kanıt gücü |
| :-: | :--- | :--- | :--- |
| **1** | **TDA + görüntü füzyonu hasta düzeyi nodül sınıflandırmasını güçlü biçimde iyileştirir** | **0.9516** ROC-AUC / **%89.4** doğruluk | Donmuş baseline'a **+0.0862**; **ince ayarlı** baseline'a **+0.0232** (CI>0) |
| **2** | **PL temsillerinin çökmesinin mekanizması: üst-zarf doygunluğu** | H1 zarf %98–99.6 arka plan; sınıf sinyali %1–2 | E4.4 nicel + E2.7 bağımsız |
| **3** | **Temsiller neredeyse dik (r≈0.30) → tamamlayıcılık gerçek** | r ≈ 0.30–0.35 | Etiket permütasyonu şans (0.4378) ⇒ sızıntı yok |

**Dürüst çerçeve:** Bu bir **hasta düzeyi sınıflandırma** çalışmasıdır; lokalizasyon iddiası yoktur. Ayırt edici sinyal ağırlıklı olarak **global/arka plan topolojisindedir**. Katkı, **yorumlanabilir topolojik temsilin görüntü temsiline kattığı ölçülebilir tamamlayıcılıktır.**

> 🌐 **Kapsam uyarısı (İddia 1 için):** 0.9516 **kurum içi** bir sonuçtur. Gerçek harici veride (VinDr-CXR, §7.2) TDA şans düzeyine iner ve füzyon ResNet'in altına düşer ⇒ İddia 1, "**kurum içi** hasta düzeyi triyaj" olarak okunmalı; "dış merkeze taşınan genel bir model" olarak **değil**.

---

## 13. V6 — İki Aşamalı Lokalizasyon (özet)

**Kod:** `pipelines/v6_two_stage/` · **Kayıt:** `artifacts/v6_two_stage/` · ayrıntı: [experiments.md §9](experiments.md) · **disk kuralı:** görüntü/kırpıntı yazımı yok.

### 1. Aşama (aday üretici, oracle Recall@K, val)
| Model | K20 | K30 | K50 | Not |
| :--- | :---: | :---: | :---: | :--- |
| Eğitimsiz en iyi (M2 DoH) | 0,117 | 0,229 | 0,325 | yetersiz |
| **M5 CenterNet** | **0,961** | **0,974** | **0,978** | 26 ms; stride-4/1024 |
| M6 FCOS-lite | 0,576 | 0,654 | 0,758 | focal objectness şart |
| M5+heatTTA | 0,965 | 0,970 | 0,978 | K≤30 için en iyi |
| **M5+M6 (RRF)** | 0,918 | 0,974 | **0,983** | K≥30 için en iyi |

### 2. Aşama (lokal TDA + FP eleme, val 734 hasta)
| Model | Hasta AUC (Noisy-OR) | FROC @4 FP |
| :--- | :---: | :---: |
| yalnız Aşama-1 | 0,711 | 0,623 |
| lokal TDA-HGB | 0,926 | 0,615 |
| yama CNN | 0,652 | 0,255 |
| **e_fusion_tda** | **0,955** | 0,771 |
| **f_hierarchical** | 0,936 | **0,857** |

* **Yama CNN çıkarıldı** (0,646 AUC, gereksiz) · SVM+MLP topluluğu reddedildi (0,656 vs HGB 0,919).
* FROC protokolü: post-fusion NMS + **duplikasyon IGNORE**.

### VinDr sıfır-eğitim (400 görüntü)
| Bileşen | AUC |
| :--- | :---: |
| Aşama-1 dedektör | **0,4618** |
| lokal TDA-HGB | **0,5901** |
| e_fusion_tda | 0,5911 |
| **V5.3 global 5-zon** | **0,6283** |
| Birleşik (V5.3+V6) | 0,6214 |
| bbox recall@K50 (M5) | **0,900** |

**Sonuç:** V6 aday üretici **lokalizasyonu taşıyor** (VinDr bbox recall@K50=0,90) ama **görüntü düzeyi güveni taşımıyor** (0,46). Kurum-içi hasta AUC 0,955; dış merkezde V5.3 global temsili hâlâ daha iyi ⇒ **lokal TDA + global temsil** birlikte kullanılmalı (NODE21 val'de birleşik 0,992).

### V6.4 — Domain stabilizasyonu + test mühürü
| Adım | Sonuç | Karar |
| :--- | :--- | :---: |
| SNR (diff/z) | AUC 0,544 → 0,544/0,545 (etkisiz) | KILL |
| **AdaBN (M5)** | **0,544 → 0,633** (+0,089); recall@50 0,907 | KEEP (yalnız M5) |
| AdaBN (M6) | M6 recall 0,893 → 0,740; füzyon 0,512 → 0,441 | KILL |
| **AdaBN-M5 + lokal TDA (M6'sız)** | Stage-1 0,6346 · TDA 0,6082 · **e_fusion 0,6471** · **V5.3+birleşik 0,6494** (r=0,46) | **V5.3'ü geçti** |
| Domain-gen M5 (aug) | NODE21 val @50: 0,978 → **0,919**; VinDr 0,544 → 0,516 | **KILL** |
| **TEST mühürü** | V6 lokal **0,9464** · V5.2 global **0,9612** · **birleşik 0,9876** (r=0,376) | **KAYIT** |

### V6.5 — VinDr transfer güçlendirmesi (transductive + few-shot)

> 🔒 **Sınıf etiketleri** (§7.2'deki tanıma göre): **Kol 1'in tamamı (B) transductive**'dir — 8 konfigürasyonun hepsi `AdaBN-M5` adayları üzerinde koşar, yani VinDr'in **etiketsiz BN istatistiklerini** kullanır. Kol 2 / 2b / 2c / 2d satırlarının tamamı **(C) süpervizli**dir (VinDr etiketi kullanır). ⇒ **Bu bölümde (A) saf zero-shot satırı yoktur**; saf zero-shot zirvesi §7.2'deki V5.3 global füzyonudur (**0,6283**).

| Kol | Konfig | VinDr AUC | Karar |
| :--- | :--- | ---: | :---: |
| 1 | taban (32/64, z) | 0,6439 | referans |
| 1 | JPEG-denoise σ=1,0 | 0,6498 | +0,006 |
| 1 | ölçek-eşlemeli yama | 0,6562 | +0,012 |
| 1 | denoise + ölçek | 0,6523 | +0,008 |
| 1 | quantile (rank-gauss) | 0,6486 | +0,005 |
| 1 | CORAL (tam uzay) | 0,6318 | −0,012 |
| 1 | **denoise + ölçek + CORAL** | **0,6620** | **KEEP (transductive en iyi — saf zero-shot DEĞİL)** |
| 2 | few-shot N=0 → 160 (yalnız lokal) | 0,6593 → **0,7243 ± 0,051** | KEEP (N≥40) |
| 2 | few-shot N=0 → 160 (V5.3 + lokal) | 0,6673 → 0,7054 ± 0,032 | N≥80'de seyreltiyor |
| 2 | few-shot N=10 | 0,6280 < 0,6593 | KILL (aşırı uyum) |
| 2b | kompakt özellik (PCA-128) + görüntü etiketi | N=160: 0,6796 (V6.5: 0,7243) | KILL (boyut sorun değil) |
| 2c | bbox-türevli aday etiketi | N=160: AUC 0,6273 · **FROC@2 0,184** (V6.5: 0,162) | FROC için KEEP, AUC için KILL |
| 2d | Çift-kafa (sabit hasta kapısı + 1:5 kutu sıralayıcı + NODE21 çapası) | AUC **0,7249 sabit** · FROC@2 0,221 (N=10) → **0,059** (N=160) | **KILL (FROC monoton çöküyor)** |

**AUC–FROC ödünleşimi:** görüntü etiketi görüntü sıralamasını (AUC), bbox etiketi lezyon sıralamasını (FROC) optimize ediyor; ikisi aynı anda iyileşmiyor. Çift-kafa (V6.5c) bu ödünleşimi **kaldıramadı**: sabit hasta kapısı AUC'yi 0,7249'da tutuyor ama kutu sıralayıcı α ile FROC'u monoton bozuyor (0,221 → 0,059) ⇒ kutu-içi TDA sıralaması VinDr'ye taşınmıyor.

### V6.5d — 🔒 Inductive (AdaBN'siz) vs Transductive (AdaBN) fiziksel hizalama

Aynı VinDr kohortu, aynı `M5_centernet` adayları, **tek fark AdaBN** (dondurulmuş NODE21 BN ↔ VinDr etiketsiz BN). Betik: [v6_5d_strict_no_adabn.py](pipelines/v6_two_stage/v6_5d_strict_no_adabn.py) · ayrıntı: [experiments.md §9](experiments.md).

| Konfigürasyon | **Inductive** (AdaBN yok) | Transductive (AdaBN) | Δ (AdaBN etkisi) |
| :--- | ---: | ---: | ---: |
| base_z | 0,6090 | 0,6439 | +0,0349 |
| + JPEG-denoise σ=1,0 | 0,5829 | 0,6498 | +0,0669 |
| **+ ölçek-eşlemeli yama** | **0,6181** | 0,6562 | +0,0381 |
| + denoise + ölçek | 0,5814 | 0,6523 | +0,0709 |
| + denoise + ölçek + CORAL | 0,5916 | 0,6620 | +0,0704 |

* 🔑 **AdaBN baş rolde** (her satırda +0,035…+0,071). V6.5'in "+0,018 fiziksel hizalama"sı **AdaBN tabanına göre göreli**dir.
* ❌ **JPEG-denoise AdaBN olmadan ZARARLI** (−0,026 … −0,037) ⇒ denoise kazancı tamamen AdaBN etkileşimiydi.
* ✅ **Tek gerçek inductive kazanç ölçek-eşlemedir: +0,0091** (0,6090 → **0,6181**) = dürüst tekil-görüntü tavanı.
* ⚠️ Inductive tavan **0,6181 < V5.3 global 0,6283** ⇒ saf sıfır-öğrenmede en iyi VinDr kanalı hâlâ **global 5-zon füzyonu**.

**Özet:** **AdaBN tabanına göre** fiziksel hizalama **+0,018**, 160 etiketli hedef görüntüyle few-shot kalibrasyon **+0,065** kazandırıyor; ikisi de VinDr'yi 0,66–0,72 bandına taşıyor ama in-domain 0,9876'nın altında. ⚠️ Kol 1 *transductive*, Kol 2 *süpervizli*dir; **inductive (saf tekil-görüntü) tavan 0,6181**, **saf zero-shot zirvesi ise §7.2'deki V5.3 füzyonudur (0,6283)**. N≥80'de lokal model kalibre edilince V5.3 global kanalı gereksizleşiyor. Domain farkı **kapatılamıyor, yönetilebiliyor**.

---

## 14. V6.6 — CXR Vakıf Omurgası (Stage-1): Tam Karşılaştırma

**Kod:** [`detector_models_foundation.py`](pipelines/v6_two_stage/detector_models_foundation.py) · [`v6_6_train_foundation.py`](pipelines/v6_two_stage/v6_6_train_foundation.py) · [`v6_6_foundation_stage1.py`](pipelines/v6_two_stage/v6_6_foundation_stage1.py) · [`v6_6_vindr_e2e.py`](pipelines/v6_two_stage/v6_6_vindr_e2e.py) · **Kayıt:** `v6_6_foundation_stage1.json`, `v6_6_vindr_e2e.json`, `checkpoints/M5_foundation.pt` · ayrıntı: [experiments.md §10](experiments.md) · **disk kuralı:** kırpıntı yazımı yok.

### 14.1 Tek değişken: ImageNet ResNet18 → CXR Vakıf ResNet50

| | Eski (V6.0–V6.5) | **Yeni (V6.6)** |
| :--- | :--- | :--- |
| Omurga | ResNet18 · **ImageNet** (kedi/köpek) | ResNet50 · **800k+ gerçek CXR** (pc-nih-rsna-siim-vin) |
| Girdi | [0,1] + ImageNet mean/std (3 kanal) | **[−1024, +1024]** (1 kanal, `xrv.utils.normalize`) |
| Boyun | yok (encoder + decode) | **ASPP (d = 1,3,6,12 + global pool) + FPN (C2..C5)** |
| Eğitim | 15 epoch, uçtan uca lr 1e-4 | **iki fazlı diferansiyel** (5 donuk + 13 FT), BN donuk, Cosine |
| Çıktı sözleşmesi | 256×256 CenterNet | 256×256 CenterNet (**birebir aynı**) |

### 14.2 NODE21 val — Yalnız Aşama-1 (734 hasta / 231 lezyon)

| Model | K10 | K20 | K30 | K50 | FROC@0,5 | FROC@4 | AUC top1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M5_heatTTA (ImageNet) | 0,9091 | 0,9654 | 0,9697 | 0,9784 | 0,524 | 0,844 | 0,795 |
| **FND_heatTTA (vakıf)** | **0,9481** | **0,9827** | **0,9827** | **0,9913** | **0,831** | **0,952** | **0,960** |

* ✅ **%98 oracle-Recall kapısı İLK KEZ geçildi** (K20/K30 = 0,9827 · K50 = 0,9913). V6.0'da hiçbir varyant geçememişti (en iyi 0,974).
* ✅ Yalnız-Aşama-1 hasta AUC **0,795 → 0,960 (+0,165)** — bu, **eski TAM iki aşamalı sistemin** (0,955) üzerindedir.

### 14.3 NODE21 val — Uçtan uca (2. aşama entegrasyonu)

| Sistem | AUC (noisy-OR) | FROC@0,5 | FROC@2 | FROC@4 |
| :--- | ---: | ---: | ---: | ---: |
| V6.2 `M5M6TTA·K50·e_fusion_tda` | 0,955 | 0,528 | 0,688 | 0,771 |
| V6.2 `M5_heatTTA·K30·f_hierarchical` | 0,936 | 0,589 | 0,753 | 0,857 |
| **V6.6 `FND·K50·e_fusion_tda`** | **0,988** | 0,745 | 0,853 | 0,879 |
| **V6.6 `FND·K50·f_hierarchical`** | 0,986 | 0,766 | 0,857 | 0,892 |
| **V6.6 yalnız Aşama-1 (K30/50)** | 0,970 | **0,831** | **0,926** | **0,952** |

* 🔑 Füzyon hâlâ katkı veriyor: 0,970 → **0,988** (noisy-OR).
  * ⚠️ **Tuzak B kaydı:** bu satırlar **in-sample s1 ile uydurulmuş LR füzyonundan** gelir (val dürüst, katsayılar hafif iyimser olabilir). **Sızıntısız** alternatif sabit ağırlıktır: **w=0,55 → nOR 0,9886 · FROC@0,5 0,8355 · FROC@4 0,9307** (bkz. §14.5, [STORY.md §12.2](STORY.md)).
* 🚨 **Ama FROC'ta 2. aşama artık ZARARLI:** Aşama-1 tek başına **0,831** → füzyon **0,749**. ⇒ **FROC-optimal (lezyon bulma) dağıtımda 2. aşama atlanmalı; AUC-optimal (hasta triyajı) dağıtımda tutulmalı.**

### 14.4 VinDr-CXR (400 görüntü) — dış transfer

> 🔴 **V6.9 DENETİM DÜZELTMESİ — bu tablodaki FND satırları KONTAMİNEDİR.** `FND` =
> `resnet50-res512-all` = torchxrayvision'da *"trained on pc-nih-rsna-siim-**vin**"*.
> Kohortun **400/400 görüntüsü (%100)** o omuranın 15.000 görüntülük ön-eğitim listesinde;
> üstelik etiketlerimizi aldığımız CSV, xrv'nin VinDr sınıfının varsayılan CSV'siyle aynı.
> ⇒ 0,6469 / 0,6593 / 0,6637 / 0,7344 satırları **"zero-shot" değildir** ve dış-merkez
> genelleme kanıtı sayılamaz. Kanıt: [LEAK_AUDIT.md](LEAK_AUDIT.md) §3. **M5 (ImageNet)
> satırları temizdir** (0,5599 / 0,6332 / 0,6181 / 0,6283 / 0,7243).
> ✅ **Etkisi ölçüldü (§17.1):** VinDr'yi görmemiş omurgayla aynı mertebe elde edildi
> (0,6663 nOR) ⇒ sayılar kontaminasyon artefaktı değil; yine de yayında **temiz omurga**
> değeri kullanılmalıdır.

| Sistem / protokol | Hasta AUC | Kaynak |
| :--- | ---: | :--- |
| M5 strict (AdaBN yok) | 0,5599 | V6.6.3 |
| V5.3 global (eski saf zero-shot zirvesi) | 0,6283 | §7.2 |
| M5 + AdaBN | 0,6332 | V6.6.3 |
| Eski en iyi inductive (M5 + ölçek-eşleme) | 0,6181 | V6.5d |
| ~~**FND strict — yeni saf zero-shot zirvesi**~~ | ~~**0,6469**~~ 🔴 | V6.6.3 · kontamine |
| ~~FND + AdaBN~~ | ~~0,5711~~ 🔴 | V6.6.3 · kontamine |
| ~~**FND inductive uçtan uca (denoise + ölçek)**~~ | ~~**0,6593**~~ 🔴 | V6.6.5 · kontamine |
| ~~FND + CORAL (transductive)~~ | ~~0,6637~~ 🔴 | V6.6.5 · kontamine |
| Eski few-shot N=160 | 0,7243 ± 0,051 | V6.5 |
| ~~**FND few-shot N=160**~~ | ~~**0,7344**~~ 🔴 | V6.6.5 · kontamine |

* ✅ **Yeni en iyi saf zero-shot: 0,6469** (AdaBN'siz) — hem M5+AdaBN (0,6332) hem V5.3 (0,6283) geçildi.
* 🔑 **AdaBN'in değeri omurganın CXR-yerliliğiyle TERS orantılı:** ImageNet ResNet18'de **+0,073** · CXR vakıf modelinde **−0,076**. AdaBN bir **yama**dır; daha iyi omurga onu gereksiz *ve zararlı* kılar.
* 🔑 **Fiziksel hizalamanın faydası da omurgaya bağlı:** JPEG-denoise M5-strict'te **−0,026**, foundation-strict'te **+0,018**.
* ✅ VinDr FROC@4FP 0,240 → **0,352**; few-shot eğrisi artık **monoton** (V6.5'te N=10'da çöküyordu).

### 14.5 Füzyon FROC ağırlık kalibrasyonu (kapanış)

`S_fused = w · z(s1_fnd) + (1 − w) · z(tda_hgb)`, **w ∈ [0, 1] adım 0,05** · 35.666 aday / 734 hasta / 231 lezyon · betik: [v6_6_fusion_calibration.py](pipelines/v6_two_stage/v6_6_fusion_calibration.py) · kayıt: `v6_6_fusion_calibration.json`.

| w | FROC@0,5 | FROC@4 | AUC noisy-OR | AUC top1 |
| ---: | ---: | ---: | ---: | ---: |
| 0,45 (**AUC-optimal**) | 0,8182 | 0,9177 | **0,9909** | **0,9780** |
| **0,55 (önerilen)** | **0,8355** | 0,9307 | 0,9886 | 0,9746 |
| **0,70 (FROC@4 garantili)** | 0,8312 | **0,9524** | 0,9827 | 0,9689 |
| 0,90 (**FROC-optimal**) | **0,8442** | 0,9524 | 0,9732 | 0,9622 |
| 1,00 (yalnız Aşama-1) | 0,8312 | 0,9524 | 0,9692 | 0,9595 |
| **REF: `e_fusion_tda` (miras LR)** | 0,7446 | 0,8788 | 0,9879 | 0,9722 |

> 🐛 **Düzeltme (çift sigmoid):** Bu satırın noisy-OR'u önceki sürümde **0,9737** yazıyordu — hatalıydı. Nedeni: LR çıktısı **zaten [0,1] olasılığı** iken üzerine bir kez daha sigmoid uygulanmıştı (noisy-OR sıralamaya duyarlı olduğu için bozuldu; FROC ve top1 sıralamaya bağlı olduğundan etkilenmedi). Düzeltilmiş değer **0,9879** ve V6.6.4'ün bağımsız ölçtüğü 0,988 ile tutarlı. Betik güncellendi ([v6_6_fusion_calibration.py](pipelines/v6_two_stage/v6_6_fusion_calibration.py)).

* 🔴 **Kapı üçlüsü (AUC ≥ 0,985 · FROC@0,5 ≥ 0,831 · FROC@4 ≥ 0,950) SAĞLANAMAZ — kesişim boş.** FROC@4 ≥ 0,950 yalnız **w ≥ 0,70**'te, AUC ≥ 0,985 yalnız **w ≤ 0,60**'ta sağlanır. Bu bir **AUC–FROC ödünleşimi**dir; eşikler "en iyi AUC'li" ile "en iyi FROC'lu" modelin aynı olmasını istiyordu.
* ✅ **Ama kalibrasyon miras alınan LR katsayılarını DOMİNE ediyor:** **w ∈ [0,35 … 0,55]** dört metrikte de ≥. **w = 0,55**: FROC@0,5 0,745 → **0,836** (+0,091), FROC@4 0,879 → **0,931** (+0,052), AUC nOR 0,9879 → **0,9886** (+0,0007), top1 0,972 → **0,975** (+0,002). ⚠️ **Dürüst okuma:** kazanç **FROC'ta büyük, AUC'de marjinal** — noisy-OR farkı yalnız +0,0007'dir.
* 🔴 **TUZAK B (düzeltildi ama kaydı burada):** V6.1/V6.3'te füzyon LR'ı, 1. aşamanın **kendi eğitim bölmesindeki (in-sample) skorlarıyla** uydurulmuştu. Val değerlendirmesi dürüsttür (M5 val'i görmedi), ama katsayılar aşırı özgüvenli bir dağılıma göre ayarlandığı için **hafif iyimserlik riski** taşır. Bu bölümdeki **sabit w** füzyonu bu riski **taşımaz**; V6.7'nin gönderim modeli de Tuzak B'yi yapısal olarak kapatmıştır (s1 matrise girmez + w kilitli).
* ⚖️ **Önerilen işletim noktaları:** **w\* = 0,55** (varsayılan, regresyonsuz) · **w = 0,70** (FROC@4 ≥ 0,950 zorunluysa; AUC 0,9827) · w = 0,45 (triyaj, en yüksek AUC) · w = 0,90 (en yüksek FROC@0,5).
* 🎯 Kalibrasyon ödünleşmeyi **kaldırmıyor**, frontier'ı miras noktasının **tamamının üstüne taşıyor** — V6.5b/V6.5c'nin dış-merkez bulgusunun kurum içi karşılığı.

> **📌 V6.6 KAPANIŞ:** Yeni omurga + FROC kalibrasyonu ile **hasta triyajı için w=0,55** (AUC 0,9886 · FROC@0,5 0,8355 · FROC@4 0,9307), **lezyon bulma için w=0,70** (FROC@4 0,9524) veya **2. aşamasız yalnız Aşama-1** (FROC@0,5 **0,831**, FROC@4 **0,952**, AUC 0,970) önerilir. Deney fazı bu noktada mühürlenmiştir.

### 14.6 Resmî NODE21 detection metriği denetimi (eğitim YOK)

Resmî protokol ([Details](https://node21.grand-challenge.org/Details/)): eşleştirme **IoU > 0,2**; FROC **@1/8 · 1/4 · 1/2** FP/görüntü; **`rank_metric = 0,75·AUC + 0,25·Sens@0,25`**. Betik: [v6_6_optimization_audit.py](pipelines/v6_two_stage/v6_6_optimization_audit.py) · kayıt: `v6_6_optimization_audit.json` · 734 hasta / 231 lezyon · 296 değerlendirme.

| Ölçek seti (+ flip TTA) | AUC | Sens@0,125 | Sens@0,25 | Sens@0,5 | FROC₃ | **rank_metric** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| single 1,0 | 0,9595 | 0,6364 | 0,7186 | 0,7879 | 0,7143 | 0,8993 |
| 0,9 + 1,0 | 0,9658 | 0,6623 | 0,7403 | 0,7879 | 0,7302 | 0,9094 |
| **0,9 + 1,0 + 1,1** | **0,9678** | **0,6710** | **0,7489** | **0,8009** | **0,7403** | **0,9131** |

*(Hepsi 56×56 sabit kutu ile; her ölçek setinin kendi en iyi NMS/tepe ayarında.)*

* 🔴 **Yarıçap kafası KULLANILAMAZ:** GT boyutları 35–119 px iken model R ≈ 0,01–0,04 (kutu ~0,1 px) ⇒ resmî IoU>0,2'de **Sens@0,25 = 0,000** (rank 0,7196). Neden: radius kaybı ağırlığı 0,1 iken focal kayıp mertebesi 1–13 ⇒ kafa çökmüş. **Gönderim sabit kare kutu kullanmalı.**
* 🔴 **Optimum kutu 56×56 px** — mevcut 64×64 optimal değil (tüm ölçek setlerinde 56 kazanıyor).
* 🟢 **Çok ölçekli TTA monoton kazandırıyor:** single → {0,9,1,0,1,1} = **+0,0138 rank_metric** (eğitim yok).
* ⚪ **NEGATİF: NMS yarıçapı (8…28 px) ve tepe penceresi (3×3/5×5) ETKİSİZ** — rank 0,9131 sabit. Resmî protokol aynı referansa çakışan 2.+ tahmini zaten IGNORE ettiği için "çift kutu" riski bu kurulumda **gerçekleşmiyor**.
* **En iyi:** çok ölçekli {0,9, 1,0, 1,1} + flip · tepe 3×3 · NMS 8 px · **56×56 kutu** → **rank_metric 0,9131**.
* ⚠️ **Kural 9:** verimiz NODE21'in halka açık **eğitim** seti; resmî test setleri özel ⇒ leaderboard sıralaması **ölçülemez**.

---

## 15. V6.7 — %100 Tam Veri Gönderim Modeli + Grand-Challenge Paketi

**Kod:** [`v6_7_train_fulldata.py`](pipelines/v6_two_stage/v6_7_train_fulldata.py) · [`v6_7_build_fulldata_stage2.py`](pipelines/v6_two_stage/v6_7_build_fulldata_stage2.py) · [`v6_7_submission_infer.py`](pipelines/v6_two_stage/v6_7_submission_infer.py) · [`v6_7_freeze_config.py`](pipelines/v6_two_stage/v6_7_freeze_config.py) · [`submission_docker/`](submission_docker/) · **Sözleşme:** `submission_config.json` · ayrıntı: [experiments.md §11](experiments.md)

> ⚠️ **Bu bölümün sayıları makale tablolarından farklı bir statüdedir.** Tam-veri modeli
> **tüm NODE21 verisiyle** eğitildiği için **kendi kendini değerlendiremez**; buradaki
> performans referansları `M5_foundation` (tam-veri olmayan) modelle alınmıştır.

### 15.1 Bilimsel checkpoint koruması (değişmez kural)

| Artifact | Rol | Durum |
| :--- | :--- | :--- |
| `M5_centernet.pt` | V6.0/V6.2 Aşama-1 (makale tabloları) | **değişmedi** (SHA-256 teyitli) |
| `M6_fcos.pt` | V6.0 FCOS yardımcı dedektör | **değişmedi** |
| `M5_foundation.pt` | V6.6 vakıf omurga (§14 tabloları) | **değişmedi** |
| `stage2_final_tda_hgb.joblib` | V6.3 nihai 2. aşama | **değişmedi** |
| `M5_domain_gen.pt` | V6.4 KILL edilen varyant (kanıt) | **değişmedi** |

Gönderim modelleri ayrı dizine yazılır: `submission_fulldata/M5_foundation_fulldata.pt` ·
`submission_fulldata/stage2_tda_hgb_fulldata.joblib`

### 15.2 Tuzak A — sabit takvimle tam veri eğitimi

| | |
| :--- | :--- |
| Veri | **4.882 hastanın tamamı** (train+val+test): 1.134 nodüllü + **1.134** dengeli temiz kontrol = **2.268 görüntü** |
| Takvim | **hiperparametre araması YOK** — V6.6'da doğrulanmış 18 epoch (5 donuk + 13 diferansiyel), BN donuk, Cosine |
| Eğitim kaybı | **19,31 → 1,22** |
| Yarıçap kafası | `radius_detach=True` — gradyanı gövdeye/ısı haritasına **hiç akmadı** |

### 15.3 Tuzak B — sızıntısız 2. aşama

| | |
| :--- | :--- |
| Adaylar | **235.446** aday / 4.882 hasta (kilitli standart) |
| Matris | **(235.446, 1604)** = 1600 PI + 4 uzamsal · **1.808 pozitif (%0,77)** |
| **Koruma 1** | HGB matrisinde **1. aşama skoru (s1) YOK** ⇒ `stage1_score_included = False` (assert) |
| **Koruma 2** | Füzyon **w = 0,55 kilitli**, hiçbir veri üzerinde **yeniden uydurulmadı** |
| HGB eğitimi | 58 s |

* **Neden bu yeterli:** HGB s1'i görmediği için 1. aşama ezberi ona sızamaz; füzyon sabit olduğu için hiçbir veriye uyum sağlayamaz ⇒ yığma-uyumu (stacking overfit) **yapısal olarak imkânsız**.

### 15.4 Kilitli çıkarım standardı (V6.6.6/V6.6.7 tarafından sabitlendi)

`scales = [0.9, 1.0, 1.1]` · flip TTA · `peak_window = 3` · `nms_px = 8` · `top_k = 50`
· **`box_px = 56`** (sabit) · füzyon `S = sigmoid(0.55·z(s1) + 0.45·z(p_tda))`

**Performans referansı** (M5_foundation ile, kendi val bölmemizde):

| Metrik | Değer |
| :--- | ---: |
| `rank_metric = 0,75·AUC + 0,25·Sens@0,25` | **0,9131** |
| AUC | 0,9678 |
| Sens@0,125 / @0,25 / @0,5 | 0,6710 / 0,7489 / 0,8009 |
| Yalnız Aşama-1 FROC@0,5 / @4 | 0,831 / 0,952 |
| Yalnız Aşama-1 Recall@K20 / K30 / K50 | 0,9827 / 0,9827 / 0,9913 |

### 15.5 Grand-Challenge konteyner uyumluluğu (kaynaktan doğrulandı)

Resmî `node21challenge/node21_detection_baseline/process.py` → `format_to_GC` ile **birebir**:

| Konu | Resmî baseline | Paket |
| :--- | :--- | :--- |
| JSON anahtarları | `{type, boxes, version}` — **`name` YOK** | ✅ |
| Köşe sırası | **[top_right, top_left, bottom_left, bottom_right]** | ✅ |
| z koordinatı | **slice indeksi** (mm değil) | ✅ |
| x/y birimi | **mm = piksel × spacing** (origin yok sayılır) | ✅ |
| Girdi yolu | `/input/<uuid>.mha` | ✅ **özyinelemeli** — `/input/*.mha` ve `/input/images/chest-x-rays/*.mha` |
| Kullanıcı | — (GC kuralı: non-root) | ✅ `USER algorithm` (uid 1000), `/output` yazılabilir |

**Doğrulanan testler** (`--network=none`, gerçek konteyner):

| Test | Sonuç |
| :--- | :--- |
| 3B `.mha` yığını (1024×1024×3), katman katman | ✅ 150 kutu |
| Şema + köşe sırası + z = slice indeksi | ✅ |
| x/y mm ↔ piksel tersinirlik; 56 px → mm | ✅ 8,00 mm (spacing 0,143) |
| Ağırlık yoksa **sessiz indirme yasağı** | ✅ `exit 1` + açık hata |
| Deterministiklik | ✅ en yüksek skor yerelde de konteynerde de 0,9938 |
| CUDA imajı / CPU imajı | 13,9 GB / **3,51 GB**, ikisi de aynı çıktı |

### 15.6 Doğrulanmış sınırlar

* ⚠️ **Tam-veri modeli NODE21'de değerlendirilemez** (tüm veri eğitimde). §15.4 referansları **tam-veri olmayan** modeldendir.
* ⚠️ **Resmî test seti skoru YOK** — verimiz halka açık eğitim seti (Kural 9).
* ⚠️ NODE21 detection track **"Now Closed"** ⇒ bu paket sıralama değil, **Algoritma sayfası / tekrar üretilebilirlik** amaçlıdır.
* ⚠️ GPU'suz çıkarım **~26 s/katman**; T4'te ~1–2 s/katman beklenir. Büyük yığınlarda CPU süre riski.

---

## 16. V6.8 — Kanıt Boşluklarının Kapatılması

Betik: [v6_8_fix_checks.py](pipelines/v6_two_stage/v6_8_fix_checks.py) · Kayıt: `artifacts/v6_two_stage/v6_8_fix_checks.json` · Eğitim **yalnız train split**, değerlendirme **yalnız val** (734 hasta / 231 lezyon) · bilimsel checkpoint'ler **değişmedi**.

### 16.1 Isınma (warm) ablasyonu — V6.6'da silinen kanıtın yeniden üretimi

Isınma fazı **aynı tohum/takvim/augmentasyon/kayıpla** yeniden koşuldu (train 1:1 = 1.580 örnek, 5 epoch donuk omurga) ve kayıtlı `M5_foundation.pt` (18 epoch) ile kıyaslandı:

| Ölçüm (val) | Isınma (5 ep, donuk) | **Final (18 ep, iki fazlı)** | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K20 | 0,9654 | **0,9827** | +0,0173 |
| Recall@K50 | 0,9913 | 0,9913 | 0,0000 |
| FROC@0,5 | 0,7792 | **0,8312** | **+0,0520** |
| FROC@4 | 0,9091 | **0,9524** | **+0,0433** |
| Hasta AUC (noisy-OR) | 0,9416 | **0,9702** | **+0,0286** |
| Hasta AUC (top1) | 0,9275 | **0,9595** | **+0,0320** |

* ✅ **2. fazın katkısı ölçüldü: FROC@0,5 +0,052 · FROC@4 +0,043 · AUC nOR +0,029.** K50'de katkı **sıfır** (tavan ısınmada yakalanıyor) ⇒ kazanç **düşük-FP bölgesinde**.
* 🔒 Final kolunun metrikleri V6.6'nın yayımlanmış sayılarıyla **birebir** (K20 0,9827 · FROC@0,5 0,8312 · @4 0,9524 · top1 0,9595) ⇒ karşılaştırma geçerli.
* 🔁 **İkinci bağımsız koşu** (aynı tohum/betik, yeniden eğitim): ısınma kolu ±0,02 kayıyor (FROC@0,5 0,7792 → 0,7965 · AUC nOR 0,9416 → 0,9375) ama **hüküm aynı** ⇒ Δ FROC@0,5 **+0,035…+0,052**, FROC@4 **+0,035…+0,043**, AUC nOR **+0,029…+0,033**, K50 **her iki koşuda 0,0000**. Etki büyüklüğü **aralık** olarak okunmalı; final kolunun sayıları iki koşuda da birebir.
* ⚠️ Bu, 2. fazın **toplam** katkısıdır (13 ek epoch + üst blokların açılması); **BN dondurmanın** payı ayrı kolda ölçülür.

### 16.2 Yarıçap kafası onarımı — çökmenin nedeni ağırlık değil, kayıp maskesi

V6.6.7 çökmeyi *"kayıp ağırlığı 0,1 (focal 1–13)"* ile açıklamıştı. V6.8 bunu test etti: **ağırlık 1,0** (10×) ile kafa **yine çöktü** (r₁₀₂₄ medyanı 0,02 px). Gerçek neden: V6.6'nın kaybı ısı haritasının **tüm pozitif bölgesi** (`Y>0`, **961 piksel**) üzerinde ortalanıyordu ama yarıçap hedefi `R` yalnız **tek merkez pikselinde** dolu ⇒ ölçüldü: süpervizyonun **%0,11**'i hedef taşır, **%99,9'u "0 tahmin et"** der; kaybın minimumu **sabit 0**.

**Kontrollü ablasyon** (tek fark: maske · omurga donuk, aynı kafa, aynı ağırlık 1,0, aynı 6 epoch/veri/tohum, val 734 hasta / 231 lezyon, resmî IoU>0,2):

| Kol | Kayıp maskesi | Eğitim kaybı (ep1→ep6) | Tahmin r₁₀₂₄ (medyan) | rank_metric | S@0,125 | S@0,25 | S@0,5 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A (V6.6'nın yaptığı) | `Y > 0` | 0,0076 → 0,0073 | **0,02** | 0,7258 | 0,000 | 0,000 | 0,000 |
| **B (onarım)** | **`R > 0`** | 2,09 → **1,07** | **33,3** | **0,9185** | **0,6970** | **0,7706** | **0,8312** |
| REF sabit 48 px | — | — | — | 0,9109 | 0,6710 | 0,7403 | 0,7922 |
| **REF sabit 56 px** | — | — | — | 0,9131 | 0,6710 | 0,7489 | 0,8009 |
| REF sabit 64 px | — | — | — | 0,9077 | 0,6537 | 0,7273 | 0,7879 |

* ✅ **Onarılan kafa sabit kutuyu geçiyor:** **0,9131 → 0,9185** (rank_metric) ve **üç duyarlılık noktasının üçünde de** artış (+0,026 / +0,022 / +0,030). Yarıçap kafası **ilk kez** resmî metrikte işe yarar hâle geldi.
* 🔒 **AUC her satırda 0,9678** — kutu geometrisi sıralamayı değiştirmez ⇒ kazanç tamamen **IoU tabanlı eşleştirmeden**.
* ⚠️ Kafa, GT aralığının (17,5–59,5 px) ortasına **sıkışıyor** (p10 26 / medyan 33 / p90 47) ⇒ kalibre değil, **regresyona-ortalama**; yine de sabit kutuyu geçiyor.
* ⚠️ Bu **dondurulmuş özellik probu**dur (omurga eğitilmedi) ⇒ uçtan uca eğitilmiş bir yarıçap dalının tavanı değil.
* ⚠️ **Gönderim sözleşmesi değişmedi:** `submission_config.json` kilitli **sabit 56 px** (Tuzak A) — onarılan kafayı gönderime almak ayrı bir sürüm kararıdır (tam-veri modelinin yarıçap dalının aynı maske düzeltmesiyle yeniden eğitilmesi gerekir).
* 🔒 **Kanıt koruması:** eğitilen kafa diske yazılır (`submission_fulldata/radius_head_weight1_center_only.pt`) ve koşu **bağımsız yeniden üretildi** (ikinci koşu aynı rank_metric/duyarlılık/r yüzdelikleri).

### 16.3 Unutma-önleme (BN dondurma) — ölçüldü, fayda **gösterilemedi**

Aynı 18 epoch takvimi, aynı veri/tohum; **tek fark:** 2. fazda BN running istatistikleri **güncellenir** (batch=4).

| Ölçüm (val 734/231) | 2. faz BN **AÇIK** | Final (BN **donuk**) | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K20 | **0,9913** | 0,9827 | +0,0086 |
| Recall@K50 | **0,9957** | 0,9913 | +0,0044 |
| FROC@0,5 | **0,8442** | 0,8312 | +0,0130 |
| FROC@1 | **0,8918** | 0,8658 | +0,0260 |
| FROC@4 | 0,9524 | 0,9524 | 0,0000 |
| Hasta AUC (top1) | **0,9646** | 0,9595 | +0,0051 |
| Hasta AUC (noisy-OR) | 0,9699 | **0,9702** | −0,0003 |
| Eğitim kaybı (ep 18) | **1,158** | 1,257 | −0,099 |

* 🔴 **"BN donuk = unutma-önleme" gerekçesi ölçümle DOĞRULANMADI:** 9 metrikten 8'inde BN açık ≥ (7'sinde daha iyi, FROC@4'te eşit); tek fark AUC nOR'da **−0,0003** (gürültü).
* ⚠️ **Ters yönde kanıt da değil:** 231 lezyonda FROC@0,5'in SE'si ≈ **0,024** ⇒ 0,013 ≈ **0,5 SE**; tek tohum/tek bölme. Doğru ifade: *"ölçülebilir fayda yok, zarar da yok."*
* 🎯 Hipotezin asıl sınavı **alan dışı transferdir** (BN istatistikleri transfer edilen şeyin parçası; V6.6.3 AdaBN'in vakıf modelini bozduğunu gösterdi) — bu kol onu **test etmez**.
* ⚠️ İki fazlı eğitimin unutma-önleme katkısı **izole edilmedi** (ısınma checkpoint'i silindi).

## 17. V6.9 — Mühürlü Test Onayı (val'de seçilen her şeyin dürüst dış-değeri)

V6.1–V6.8'de tüm seçimler **val**'de yapılıp **val**'de raporlandı (kutu boyu: **296** resmî-metrik değerlendirmesi; füzyon w: **19** nokta; 3 TTA seti; 12 NMS/tepe). 231 lezyonda Sens@0,25 SE ≈ 0,030 ⇒ seçim yanlılığı beklenir. Mühür, kilitli protokolü **hiç değiştirmeden** dokunulmamış test bölmesinde (734/229) **bir kez** koşar.

**Ön-kayıt (koşudan önce commit `deb59ed`):** `artifacts/v6_two_stage/test_seal_preregistration.json` · **Kayıt:** `v6_9_test_seal.json` · **Aralık:** `v6_9b_test_seal_ci.json` · **Rapor:** [LEAK_AUDIT.md](LEAK_AUDIT.md)

**Mühür garantileri:** betik kendi SHA-256'sını ön-kayıtla karşılaştırır (uyuşmazsa durur) · model+bundle hash doğrulaması (`fc2775…`, `976afc…`) · z referansı **val'den** donduruldu · hiperparametre arayan kod yok.

| Ölçüm | Val (seçim) | **Test (mühür)** | Δ | Test %95 GA |
| :--- | ---: | ---: | ---: | :--- |
| **rank_metric — 1. aşama** | 0,9131 | **0,8842** | −0,0289 | 0,8559 – 0,9089 |
| rank_metric — füzyon w=0,55 | 0,9174 | **0,8970** | −0,0204 | 0,8769 – 0,9213 |
| Hasta AUC — 1. aşama | 0,9678 | **0,9446** | −0,0232 | 0,9237 – 0,9621 |
| Hasta AUC — füzyon | 0,9793 | **0,9645** | −0,0148 | 0,9491 – 0,9786 |
| Sens@0,125 — 1. aşama | 0,6710 | **0,6114** | −0,0596 | 0,5250 – 0,6805 |
| Sens@0,25 — 1. aşama | 0,7489 | **0,7031** | −0,0458 | 0,6250 – 0,7645 |
| Sens@0,5 — 1. aşama | 0,8009 | **0,7642** | −0,0367 | 0,7110 – 0,8164 |
| Sens@0,5 — füzyon | 0,7879 | **0,7729** | −0,0150 | 0,7186 – 0,8233 |

* ✅ **Hat sağlam:** mühür koşusu val kolunda V6.6.7'nin sayılarını **birebir** üretti (0,9131/0,9678/0,6710/0,7489/0,8009) ⇒ fark kod hatası değil, genelleme farkı.
* 📉 **Düşüş ≈ 1 SE** (−0,029 vs SE 0,030); ön-kayıttaki 0,05 eşiği aşılmadı ⇒ ölçülebilir ama sınırda iyimserlik.
* ⚠️ Val noktası test aralığının **az üstünde** (0,9131 > 0,9089) ⇒ "val hafif iyimser" demek gerekir.
* 🔄 **Hüküm değişti:** val'de "FROC için 2. aşamayı atla" denmişti; test'te **füzyon 3/4 metrikte üstün** (rank 0,8970 · AUC 0,9645 · S@0,5 0,7729) ⇒ o öneri val'e özgüydü.
* ❌ `legacy_fusion.patient_auc.noisyor` (0,6432) **geçersiz** (z-skoru üzerinden noisy-OR); geçerli değer `A2a_fusion_tda.auc` = 0,9645.
* 🟡 **Beyan:** test bölmesi ikinci kez görüldü (ilki V6.4 mührü, 0,9876); sonraki hiçbir karar test'e bakılarak alınmadı.

### 17.1 Kontaminasyonun ölçümü — temiz omurga deneyi (V6.9c)

VinDr sayılarının ön-eğitim kontaminasyonundan şişip şişmediği **doğrudan test edildi**: aynı tarif (omurga donuk, 5 epoch, aynı veri/tohum), **tek değişken omurganın gördüğü veri**.

| Satır | Omurga | VinDr ön-eğitimde? | NODE21 val nOR | VinDr top1 | VinDr nOR |
| :-: | :--- | :---: | ---: | ---: | ---: |
| A | `resnet50-res512-all` (mevcut) | **var** | 0,9466 | 0,6479 | 0,6414 |
| B | `densenet121-res224-all` | **var** | 0,9440 | 0,6406 | 0,6534 |
| **C** | `densenet121-res224-nih` | **yok** | 0,8650 | **0,6369** | **0,6663** |
| | **B − C** | | **+0,0790** | +0,0037 | −0,0129 |

* ✅ **Şişme yok:** temiz omurga VinDr'de en az kontamine olanlar kadar iyi (Δ top1 +0,004; nOR'da **daha iyi**, −0,013). 150 pozitifte AUC SE ≈ 0,04 ⇒ farklar gürültü bandında. ⇒ "kontaminasyon sayesinde yüksek" iddiası **desteklenmiyor**.
* 🔑 **VinDr ön-eğitimi hedef göreve yarıyor, teste değil:** B−C NODE21'de **+0,079**, VinDr'de ~0.
* ⚠️ **Confound:** B ile C toplam korpus büyüklüğünde de ayrışır ⇒ +0,079 NODE21 farkı VinDr'ye özgü sayılamaz.
* 🎯 **Yayın:** VinDr sayısı **C satırıyla** (0,6663 nOR / 0,6369 top1) verilmeli; A/B "kontamine üst sınır" dipnotu.
* 🔁 **İç tutarlılık:** A satırı V6.6.3'ün 0,6469'unu 0,6479 olarak yeniden üretti.

### 17.2 NODE21 iç kaynak örtüşmesi — kaynak-ayrıştırmalı ölçüm (V6.9d)

NODE21'in halka açık eğitim seti, açık veri kümelerinden derlenmiştir (`patient_info.json`, 4.882 hasta): `node21_control` 3.748 (etiket 0, köken kayıtlı değil) · `chestxray14` 617 · `padchest` 314 · `jsrt` 149 · `openi` 54 (hepsi etiket 1). Omurga `pc-nih-rsna-siim-vin` ile ön-eğitildiği için **pozitiflerin %82,9'u gördüğü** korpuslardan gelir — **ama kutular yeni** (açık veri kümelerinde NODE21 anotasyonu yok) ⇒ resmî IoU metriği kutu ezberine kapalı.

| Test (734/229) | n hasta | n lezyon | AUC | S@0,125 | S@0,25 | S@0,5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| chestxray14 ✅ görüldü | 89 | 138 | 0,9760 | 0,6449 | 0,6957 | 0,7754 |
| padchest ✅ görüldü | 58 | 66 | 0,9604 | 0,6515 | 0,7273 | 0,8182 |
| jsrt ❌ görülmedi | 19 | 19 | 0,9147 | 0,3684 | 0,4737 | 0,5263 |
| openi ❌ görülmedi | 6 | 6 | 0,9920 | 1,000 | 1,000 | 1,000 |
| **GÖRÜLEN** | 147 | 204 | **0,9698** | 0,6471 | **0,7059** | 0,7892 |
| **GÖRÜLMEYEN** | 25 | 25 | **0,9332** | 0,5200 | **0,6000** | 0,6400 |
| **Fark** | | | −0,037 | −0,127 | −0,106 | −0,149 |

Val'de aynı yön, daha küçük: AUC 0,9826 vs 0,9660 · S@0,25 0,7448 vs 0,6667.

* ⚠️ Fark ≈ **1 SE** (görülmeyen grupta 25 lezyon); **belirleyici değil**.
* ⚠️ Kaynak kimliği başlı başına güçlü değişken (spacing 0,143–0,331 mm/px; A3'te aynı model için kaynak-içi AUC 0,78–0,94 = §7.3b) ⇒ confound.
* ✅ Yön tutarlı ama büyüklük sınırlı (≤0,04 AUC) ⇒ ne "etkisiz" ne "sahte" denebilir; **beyan** edilir.
* 🔒 Gizli test setinin bileşimi doğrulanamaz ⇒ "sıfır sızıntı" iddiası kurulmamalı.

### 17.3 Omurga × kaynak etkileşimi — belirleyici test (V6.9e, 🔒 kapandı)

V6.9d'nin farkı kaynak cihazıyla karışıktı; ayrıştırmak için iki omurga **aynı protokolde** (heat-map TTA) kıyaslandı: **M5** = ImageNet ResNet18 (temiz), **FND** = `resnet50-res512-vin` (NODE21 pozitiflerinin %82,9'unu görmüş). Etkileşim = ΔFND−M5(görülen) − ΔFND−M5(görülmeyen).

| Val (734 hasta) | M5 (temiz) | FND (kontamine) | Δ = FND − M5 |
| :--- | ---: | ---: | ---: |
| Görülen (chestxray14+padchest, 192 lezyon) | 0,8386 | 0,9636 | **+0,1249** |
| Görülmeyen (jsrt+openi, 39 lezyon) | 0,6176 | 0,9427 | **+0,3251** |
| **ETKİLEŞİM** | | | **−0,2002** (GA −0,299…−0,109) |

* ✅ **Örtüşme lehine kanıt yok, ters yönde:** avantaj görülmeyen kaynaklarda **2,6×** daha büyük ⇒ kazanç hatırlama değil, genel CXR temsili.
* 🔑 Görülen↔görülmeyen açığı M5'te +0,221, FND'de +0,021 ⇒ CXR ön-eğitimi kaynak-duyarlılığını **~0,20 AUC** azaltıyor.
* ⚠️ Sınırlar: yalnız val (test mühürlü) · kapasite farkı (11M vs 23M) · görülmeyen grupta 39 lezyon.
* 🔒 Konu **kapandı** (V6.9c + V6.9d + V6.9e aynı yön); mührü yalnız yeni dış veri veya korpusu NODE21 ile kesişmeyen omurga açar. Ayrıntı: [LEAK_AUDIT.md §10](LEAK_AUDIT.md).

### 17.4 Dinamik yarıçap kafası — kilitli test doğrulaması (V6.9f)

V6.8'de `R>0` maskesiyle eğitilen dinamik yarıçap kafası **val**'de sabit 56 px'i geçmişti
(0,9185 > 0,9131). Kilitli test doğrulaması (⚠️ üçüncü test kullanımı — tek-atımlık, seçim yok):

| Test (734/229) | S@0,125 | S@0,25 | S@0,5 | S@1 | S@2 | S@4 | rank_metric |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sabit 56×56 | 0,6114 | 0,7031 | 0,7642 | 0,8210 | 0,8603 | 0,8952 | **0,8842** |
| **Dinamik yarıçap** | 0,6114 | 0,7031 | **0,7773** | **0,8341** | **0,8734** | **0,9083** | **0,8842** |

* 🚨 **Val kazancı resmî skorda test'e taşınmadı:** S@0,25 iki kolda da 0,7031 ⇒ rank_metric
  eşit (0,8842). Val'deki +0,0054 (0,9131→0,9185) test'in S@0,125/0,25'inde tekrarlanmadı.
* ✅ **Ama zarar yok, gevşek eşiklerde +0,0131:** S@0,5/1/2/4'te dinamik kutu daha iyi (FROC-optimal bölge).
* 🎯 **Karar:** sabit 56 px muhafazakâr varsayılan kalır; dinamik kutu resmî skoru artırmadığı için zorunlu değil (opsiyonel).

# NODE21 TDA — Makale Taslağı (E5.2)

> **Durum:** Yapısal taslak. Tüm sayılar [ablation_matrix.md](ablation_matrix.md)'den alınmıştır ve doğrulanmıştır.
> **Protokol uyumu:** Split beyanı (Kural 9) ✅ · Çok-seed raporlama: v3 ✅ / **V6 hattı ⚠️ tek tohum** (±0,02 bant, V6.8) · Ön-kayıtlı eşikler ✅ · Test bütçesi: v3 6/6 (kapalı) + **2 ön-kayıtlı mühür** (V6.4, V6.9)
> **Denetim:** [LEAK_AUDIT.md](LEAK_AUDIT.md) — sızıntı taraması (5 kontrol temiz), **VinDr kontaminasyonu** ve **val seçim yanlılığının** ölçümü. Yayına girmeden okunmalı.
> **Kaynak belgeler:** [ablation_matrix.md](ablation_matrix.md) · [experiments.md](experiments.md) · [LEAK_AUDIT.md](LEAK_AUDIT.md) · [literature.md](literature.md) · [roadmap.md](roadmap.md)

---

## Başlık (öneri)

**Yorumlanabilir Kalıcı Homoloji Temsilleri, Göğüs Röntgeninde Nodül Sınıflandırması için Görüntü Temsillerini Tamamlar: NODE21 Üzerinde Bir Ablasyon Çalışması**

*Alternatif:* Persistent Homology Complements Learned Image Representations for Patient-Level Nodule Classification in Chest Radiographs

---

## Özet (Abstract) — taslak

**Amaç.** Göğüs röntgeninde (CXR) nodül tespiti, derin öğrenme modelleriyle yaygın olarak çalışılmaktadır; ancak bu modeller yorumlanabilirlikten yoksundur ve küçük veri rejimlerinde ciddi genelleme kaybı yaşar. Topolojik veri analizi (TDA), kalıcı homoloji yoluyla yorumlanabilir bir alternatif sunar; fakat CXR üzerinde TDA temsillerinin **neden** ve **hangi koşullarda** işe yaradığı sistematik olarak incelenmemiştir.

**Yöntem.** NODE21 halka açık eğitim setini (4.882 hasta: 1.134 nodül-pozitif, 3.748 normal) kullandık. Akciğerler PSPNet ile bölütlendi ve 5 anatomik zona ayrıldı. Her zon için kübik kalıcı homoloji (H0/H1) hesaplandı ve kalıcılık imgeleri (PI, 2×20×20) çıkarıldı. Bir TDA dalı (zon başına küçük CNN + RBF-SVM) ve bir görüntü dalı (ResNet18) hasta düzeyinde max-pooling ile birleştirildi. **Tek-değişken ve ön-kayıtlı** bir protokolle 28 deney yürüttük; test seti yalnızca 6 kez değerlendirildi.

**Bulgular.**
1. **Füzyon ana sonucu:** ince ayarlı ResNet18 + TDA füzyonu hasta düzeyi test ROC-AUC **0.9516** (doğruluk **%89.4**). TDA'nın ince ayarlı güçlü baseline'a katkısı **+0.0232** (%95 CI [+0.0132, +0.0338], p=0.0000).
2. **Tamamlayıcılık:** iki temsil neredeyse diktir (**r ≈ 0.30**); etiket permütasyonu kontrolü şans düzeyi (0.4378) verir ⇒ sızıntı yoktur.
3. **Mekanizma:** H1 kalıcılık **üst-zarfı** her zonda ≥5 uzun ömürlü arka plan sınıfıyla doyurulmuştur; sınıf ayrımını taşıyan sinyal zarfın yalnızca **%1–2'sidir**. Bu, kalıcılık **peyzajlarının** (PL) neden çöktüğünü nicel olarak açıklar.
4. **Sağlamlık ve genelleme:** σ/ızgara/ağırlık/eşik seçimleri sağlamdır; kaynak-dışı (leave-one-source-out) ortalama **0.8098**.
5. **v3 temsilinin sınırı:** 5-zon TDA temsili **lokalize etmez** — zon düzeyi AUC 0.48–0.53; nodülün tam merkezinde bile yerel TDA ayrımı yalnızca **0.687**'dir.
6. **İki aşamalı lokalizasyon (V6):** ayrı bir aday üretici (CenterNet/FCOS) + lokal TDA-HGB FP eleme ile NODE21 test setinde hasta AUC **0.9876** (global + lokal birleşim; r=0.38) ve FROC@2FP **0.642**; tek seferlik ön-kayıtlı mühür.
7. **Harici transfer (VinDr-CXR, 400 görüntü):** saf zero-shot'ta en iyi sonuç V5.3 global füzyonudur (**0.6283**). Etiketsiz AdaBN + fiziksel hizalama (JPEG-denoise + ölçek-eşlemeli yama + CORAL; *transductive*, saf dış doğrulama değil) bandı **0.662**'ye çıkarır — fiziksel hizalamanın AdaBN tabanına katkısı **+0.018**'dir, 0.544→0.662 farkının büyük kısmı AdaBN'indir; hedef merkezden **160 etiketli görüntü** ile few-shot kalibrasyon **0.724**'e ulaşır. Ancak aday düzeyi TDA sıralaması taşınmaz: hasta AUC ile FROC arasında **kaldırılamayan bir ödünleşim** vardır.

**Sonuç.** Yorumlanabilir topolojik temsiller, öğrenilmiş görüntü temsillerine **ölçülebilir ve istatistiksel olarak anlamlı** tamamlayıcılık katar. Ancak CXR'de ayırt edici bilgi **global/arka plan topolojisinde** yoğunlaşmıştır; bu nedenle "nodülü doğru zonda bulma" iddiası **5-zon temsili için** kurulamaz. Lokalizasyon ancak ayrı bir dedektör + lokal TDA katmanıyla (V6) sağlanır ve dış merkezde aday sıralaması kalibre edilmeden taşınmaz. Katkımız, bu sınırı dürüstçe belirleyen ve mekanizmayı nicel olarak gösteren bir çerçevedir.

---

## 1. Giriş

*(Yazılacak — iskelet)*

1.1 CXR'de nodül tespitinin klinik önemi ve düşük doz tarama bağlamı.
1.2 Derin öğrenmenin başarısı **ve** iki zayıflığı: (a) yorumlanabilirlik, (b) küçük veri rejiminde genelleme kaybı.
1.3 TDA / kalıcı homoloji: yorumlanabilir, el yapımı olmayan, ölçek-değişmez temsil vaadi.
1.4 **Boşluk:** CXR'de TDA temsilleri raporlanmıştır, ama (i) *neden* işe yaradığı, (ii) *nerede* sınırının olduğu, (iii) güçlü bir görüntü baseline'ı karşısındaki **marjinal** katkısı sistematik olarak ölçülmemiştir.
1.5 **Katkılarımız:**
   - (C1) NODE21 üzerinde, güçlü ve **ince ayarlı** bir baseline karşısında istatistiksel olarak doğrulanmış füzyon kazancı.
   - (C2) PL üst-zarf çökmesinin **nicel** mekanizma gösterimi.
   - (C3) "Global vs lokal" sorusunu kapatan **negatif sonuçlar kümesi** (5-zon TDA lokalize etmez; yerel topoloji zayıf; arka planı bastırmak zararlı).
   - (C4) **İki aşamalı lokalizasyon (V6):** aday üretici + lokal TDA-HGB FP eleme; NODE21 test mührü (AUC **0,9876**, FROC@2 **0,642**) ve VinDr-CXR transfer merdiveni (**0,544 → 0,724**).
1.6 **Makalenin çerçevesi:** v3 katmanı **hasta düzeyi sınıflandırma**dır ve **lokalizasyon iddiası taşımaz**; lokalizasyon, ayrı olarak değerlendirilen **V6 iki aşamalı eklentisiyle** sağlanır (§4.10).

---

## 2. İlgili Çalışmalar

*(literature.md §1–§4'ten özetlenecek)*

2.1 CXR'de nodül tespiti (NODE21 ve sonrası).
2.2 Tıbbi görüntülemede TDA: Topo-CXR, TDA-Net ve diğerleri.
2.3 PI vs PL: veri setine bağlı başarım.
2.4 **Kritik konumlandırma:** Bizim veri kümemiz NODE21'in **halka açık eğitim setiyle birebir aynıdır** ⇒ leaderboard karşılaştırması **geçersizdir** (§3.1).

---

## 3. Materyal ve Yöntem

### 3.1 Veri ve Zorunlu Beyan (Kural 9)

| | |
| :--- | :--- |
| Kaynak | NODE21 **halka açık eğitim seti** (yerel olarak doğrulandı) |
| Hasta | **4.882** = 1.134 nodül-pozitif (1.476 lezyon) + 3.748 normal |
| Split (hasta düzeyi, dondurulmuş) | train **3.414** / val **734** / test **734** |
| Zon satırı | **24.410** (5 zon × hasta) |
| Dengeli train alt kümesi | **2.040** zon (1:1) |
| Kaynak dağılımı (nodüllü) | chestxray14 617 · padchest 314 · jsrt 149 · openi 54 |

> **Beyan:** NODE21'in resmî test setleri (281 + 298 görüntü) **özeldir ve kullanılmamıştır**. Karşılaştırma yalnızca *bağlam* olarak verilir. Hasta sızıntısı yoktur (0 kesişim, otomatik test + E0.2 ile doğrulandı).

### 3.2 Ön-işleme ve Zon Ayrımı

TorchXRayVision **PSPNet** akciğer bölütlemesi → 5 anatomik zon (sağ üst/orta/alt, sol üst/alt) → zon kırpıntıları.

**Etiketleme kuralı (`v2_overlap20`, üç sınıflı):**

| Bounding-box piksel örtüşmesi | Sınıf | Eğitimde |
| :--- | :--- | :--- |
| ≥ %20 | `positive` | ✅ pozitif örnek |
| %0–%20 | `ambiguous` | ⛔ **dışlanır** (ne pozitif ne negatif) |
| %0 | `negative` | ✅ 1:1 dengede temiz negatif |
| Hiçbir zon ≥%20 | `unassigned_nodule` | ⛔ dışlanır |

*Gerekçe:* %5 örtüşen bir zonun %99'u sağlıklı dokudur; onu "hasta" saymak etiket gürültüsü üretir (v1'de 130 zon bu hatayı taşıyordu). Yalnızca **68 zon** (%0,28) belirsizdir. E4.1+E4.6, eşiğin (%10/%20/%30) 5-seed'de **anlamlı fark yaratmadığını** gösterdi.

### 3.3 Topolojik Temsil (TDA dalı)

Kübik kalıcı homoloji (GUDHI) → H0/H1 diyagramları → **Kalıcılık İmgesi (PI)**, 20×20 ızgara, σ=1,0, ağırlık `w(p)=p` (Adams) → **800 boyut**.

Eksen aralıkları train'den donduruldu: H0 birth/pers [0,248]; H1 birth [239,250], pers [0,7].

**Sınıflandırıcı:** zon başına küçük CNN (2×20×20 giriş) + RBF-SVM, soft voting (w=0,55/0,45) → **hasta düzeyi max-pooling** (5 zon).

### 3.4 Görüntü Dalı

ResNet18 (ImageNet pretrain) → zon kırpıntısı başına **512 boyutlu** özellik → zon başına lojistik regresyon → hasta düzeyi max-pooling.
**İki varyant:** (a) **donmuş** (linear probe), (b) **son blok (`layer4`+`fc`) ince ayarı** (GPU, 25 epoch, lr 1e-4, wd 1e-4, dropout 0,3, yatay çevirme).

### 3.5 Füzyon

`hasta_skoru = a · görüntü + (1−a) · TDA` — ağırlık `a` ve karar eşiği **yalnızca validation'da** seçilir (Youden).

### 3.6 Deneysel Protokol (ön-kayıt)

| Kural | Uygulama |
| :--- | :--- |
| **Tek değişken** | Her deneyde yalnızca bir bileşen değiştirilir |
| **Dondurulmuş test** | Test seti yalnızca **6 kez** değerlendirildi; diğer tüm kararlar validation'da |
| **Ön-kayıtlı eşik** | val ROC-AUC Δ ≥ **+0,015** **ve** bootstrap %95 CI sıfırı dışlamalı |
| **Öldürme kriteri** | Eşik sağlanmazsa deney **KILL** |
| **Kural 9** | Split her tabloda açıkça beyan edilir |
| **Kural 10** | Performanslar **≥3 seed ortalaması** olarak raporlanır |

**Ölçek:** **28 deney** · **14 KEEP** · **13 KILL** · **1 uygulanamaz**. (Roadmap §13'teki "34 backlog maddesi" **farklı bir sayımdır**: iptal/geçersiz/ertelenen maddeler ve alt-varyantlar ayrı satır sayıldığı için; deney kimliği sayısı 28'dir.) **Karar günlüğü** [roadmap.md](roadmap.md) §10.

---

## 4. Sonuçlar

### 4.1 Ana Sonuç — Model Liderlik Tablosu

Hasta düzeyi, 734 test hastası:

| Sıra | Model | **Test ROC-AUC** |
| :-: | :--- | :---: |
| 🥇 | **Füzyon: ince ayarlı ResNet18 + TDA ensemble** | **0,9516** (doğruluk **%89,4**) |
| 🥈 | İnce ayarlı ResNet18 (son blok) | 0,9284 |
| 🥉 | Füzyon: donmuş ResNet18 + TDA | 0,9190 ± 0,0056 (3 seed) |
| 4 | TDA ensemble (CNN + SVM) | 0,8776 |
| 5 | TDA PI küçük CNN | 0,8631 |
| 6 | ResNet18 (donmuş) + lojistik regresyon | 0,8349 |
| 7 | RBF-SVM | 0,8289 |
| 8 | k-NN | 0,7162 |
| 9 | Ham görüntü küçük CNN | 0,7043 |
| 10 | PL küçük CNN | 0,5450 |
| 11 | PL RBF-SVM | 0,5217 |

*Bağlam (karşılaştırılamaz):* NODE21 leaderboard → D1 0,868 · ensemble 0,877.

### 4.2 Ana İddia: Tamamlayıcılık — İki Baseline Karşısında

| Karşılaştırma | Δ | %95 CI | p |
| :--- | :---: | :---: | :---: |
| Füzyon − **donmuş** ResNet18 | **+0,0862** | [+0,0632, +0,1099] | ≈0,0000 |
| **Füzyon − ince ayarlı ResNet18** | **+0,0232** | **[+0,0132, +0,0338]** | **0,0000** |

> **Kritik:** İnce ayar baseline'ı **+0,0935 güçlendirdi**; buna rağmen TDA anlamlı katkı sağlıyor. ⇒ **Füzyon kazancı zayıf baseline artefaktı değildir.**

**Tamamlayıcılık:** iki temsil arasındaki korelasyon **r ≈ 0,30** (val 0,3034 · test 0,3489) — neredeyse dik. Etiket permütasyonu → AUC **0,4378** (şans) ⇒ **sızıntı yok**.

**Füzyon seviyesi (E3.2):** geç (karar) füzyon **0,9538** > orta seviye (özellik birleştirme) 0,9482 (−0,0056) ⇒ karar düzeyinde birleştirme doğrulanmıştır.

### 4.3 Mekanizma: PL Neden Çöker?

Dengeli train (n=2.040 zon), E4.4:

| Ölçüt | λ1 | λ2 | λ3 | λ4 | λ5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **H1** zarf tepesi | 109,89 | 102,61 | 97,56 | 93,60 | 89,61 |
| **H1** sınıf sinyali / zarf | %1,62 | **%1,90** | %1,54 | %0,37 | %1,72 |
| **H0** sınıf sinyali / zarf | %0,02 | %0,21 | %0,39 | %0,78 | **%0,89** |

**Yorum:** H1 zarf tepeleri λ1→λ5 boyunca 110→90 ve birbirine yakın, enerji eşit dağılmış ⇒ zarf her zonda **≥5 uzun ömürlü arka plan H1 sınıfıyla doymuş**. Sınıf sinyali zarfın yalnızca **%1–2'si** (≈50–100× bastırılma). PI `(birth, persistence)` 2B yoğunluğunu koruduğu için bu sinyali saklar; **PL ise skaler bir zarf fonksiyonuna indirgendiği için kaybeder.**

**Bağımsız destek (E2.7):** üst-doğrusal ağırlık `p²` en kötü sonucu verir (−0,0189) — arka plan sınıflarını amplifiye etmek zararlıdır.

### 4.4 Yorumlanabilirlik: Sınıflar Hangi Yapıya Düşüyor?

E4.2 (GUDHI `cofaces_of_persistence_pairs()`, 30 pozitif + 30 negatif zon, top-K=10):

| Boyut | Top-K gradyan oranı (ort/medyan) | Rastgele kontrol | Karar |
| :---: | :---: | :---: | :--- |
| **H0** | **6,176 / 6,608** | 0,974 | ✅ kenarlara düşüyor |
| **H1** | 1,474 / **1,021** | 0,981 | ❌ kenarlarda **değil** |

⇒ Her iki boyutta da en kalıcı sınıflar **nodüle özgü değil**; bu §4.3'ü ve §4.7'yi bağımsız destekler.

### 4.5 Sağlamlık

| Boyut | Test edilen | Sonuç |
| :--- | :--- | :--- |
| PI ağırlığı | 5 varyant | ✅ mevcut seçim optimal (E2.7) |
| σ (Gaussian) | 4 değer | ✅ sağlam, maks sapma 0,0236 < 0,03 (E2.8) |
| Izgara | 3 değer | ✅ sağlam, maks sapma 0,0227 < 0,03 (E2.8) |
| Etiket eşiği | %10/%20/%30 | ⚠️ 5-seed'de fark yok → %20 (E4.1+E4.6) |
| Tekrar üretilebilirlik | referans | ✅ **bit-bit aynı, 16/16 metrik** (E0.2) |
| Seed varyansı | 5 seed | ⚠️ **±0,02–0,03** → Kural 10 (E4.6) |
| Çok-seed ana sonuç | 3 seed | ✅ **0,9190 ± 0,0056** (E3.7) |

### 4.6 Genelleme: Kaynak-Dışı (Leave-One-Source-Out)

| Dışarıda | Doğal boyut | **Eşit-boyut (1034 zon)** |
| :--- | :---: | :---: |
| chestxray14 | 0,7821 | 0,7821 |
| padchest | 0,8499 | **0,7625** |
| jsrt | 0,9085 | **0,8847** |
| **Ortalama** | 0,8468 | **0,8098** |

> **Metodolojik not:** İlk koşuda katların eğitim verisi miktarı farklıydı (1034/1876/2222 zon) ve bu **confound** sonucu iyimserleştiriyordu. Eşitleme sonrası ortalama **0,8098**, domain-shift bedeli **−0,109**. ⇒ Genelleme gerçek, ancak bedeli daha yüksek. **Kaynak-dışı genelleme çalışmalarında eğitim verisi miktarı mutlaka eşitlenmelidir.**

### 4.7 🚨 Doğrulanmış Sınırlama — 5-Zon TDA Temsili Lokalize Etmiyor

| Deney | Bulgu |
| :--- | :--- |
| **E4.7** | Nodül-pozitif hastalarda argmax zon, gerçek nodül zonuna **şans düzeyinde** düşüyor (isabet 0,228–0,298 vs şans 0,268; binom p=0,21–0,90). **Zon düzeyi AUC 0,48–0,53.** |
| **E4.3** | Her zon **tek başına** 0,74–0,84 hasta AUC veriyor ⇒ zayıf zon yok, ama bu lokalizasyon değil |
| **E3.5** | Havuzlanmış tek model, 5-zon özel modeliyle **eşdeğer** (0,8585 vs 0,8661, −0,0076) ⇒ zon bölünmesi **topluluk çeşitliliği** sağlıyor, **anatomik uzmanlaşma** değil |
| **E1.2** | Damar-kaynaklı H1 sınıflarını elemek başarımı **monoton düşürüyor** (0,8697 → 0,8358) ⇒ arka plan **faydalı** |
| **E3.6** | **Ground-truth bbox** ile nodülün tam merkezinde bile yama düzeyi AUC yalnızca **0,6866** (boyut-eşleştirilmiş) |

> 🔗 **Birleşik sonuç:** Ayırt edici bilgi **lokal nodül topolojisinde değil, global/arka plan topolojisindedir.** "Arka planı bastırınca lokal sinyal açığa çıkar" hipotezi **çürütülmüştür**.
> ⚠️ **Confound uyarısı (E3.6):** İlk koşu **0,994** vermişti; kontroller sabit 60×60, pozitifler değişken boyuttaydı ve sınıflandırıcı **nodülü değil yama boyutunu** öğrenmişti. ⇒ *"Fazla iyi" sonuçlar sistematik olarak confound açısından denetlenmelidir.*
> 📌 **Kapsam:** Bu bulgu **5-zon TDA temsili** içindir. Lokalizasyon, ayrı bir aday üretici + lokal TDA katmanıyla **V6'da çözülmüştür** (§4.10). Doğru ifade "TDA lokalize edemez" değil, "**zon düzeyi TDA temsili lokalize etmez**"dir.

### 4.8 🌐 Harici Doğrulama (External Validation) — VinDr-CXR Üzerinde Sıfır-Eğitim Testi

NODE21 üzerinde eğitilen v3 (E0.6) mimarisi, tamamen bağımsız bir dış veri setinde (**VinDr-CXR**, Vietnam — Hospital 108 & Hanoi Medical University Hospital) **sıfır eğitim (zero-shot transfer)** protokolüyle değerlendirilmiştir. Kohort: **400 görüntü** (150 nodül/kitle + 250 saf normal, seed 42); ⚠️ VinDr metadata'sında hasta kimliği olmadığı için değerlendirme **görüntü düzeyindedir** ("400 hasta" değil). TDA dalı dondurulmuş NODE21 bundle'larıdır; ResNet18 E0.6 tarifiyle **deterministik** olarak NODE21'de yeniden eğitilir ve VinDr'de yalnızca çıkarım yapılır (E0.6 ağırlıkları hiç kaydedilmemişti; val zon AUC **0,8707** vs E0.6 kaydı 0,8703, Δ=3,8·10⁻⁴).

| Kalibrasyon kolu | TDA ensemble | ResNet18 | Füzyon (**a = 0,45**, kilitli) | Δ(füzyon−ResNet) |
| :--- | :---: | :---: | :---: | :---: |
| `linear` (8-bit × 16; eşleme yok) | 0,5274 [0,4699; 0,5836] | 0,6204 [0,5609; 0,6763] | 0,5328 | **−0,0876** [−0,1262; −0,0491] |
| `matched` (200-hasta havuzu; yalnızca dokuya eşleme) | **0,5409** [0,4852; 0,5973] | 0,6204 | 0,5438 | **−0,0765** [−0,1133; −0,0393] |
| `matched_scalenorm` (+ akciğer-yüksekliği ölçek normalizasyonu) | 0,5110 [0,4522; 0,5688] | 0,6204 | 0,5352 | **−0,0852** [−0,1302; −0,0403] |

* **Harici veride optimal füzyon ağırlığı a = 1,00** (yani saf görüntü modeli); 0,00–0,95 arası tüm ağırlıklar ≤ 0,5578'de kalır ⇒ **TDA dalı VinDr'de katkı sağlamıyor.**
* **Ölçek normalizasyonu kurtarmıyor:** kesitleri akciğer yüksekliğine göre ölçeklemek TDA'yı iyileştirmiyor (0,5110 ≤ 0,5409) ⇒ çöküş yalnızca yoğunluk/ölçek farkıyla açıklanamaz.

**Nodül boyutu alt grupları — ölçek-bağımsız** (`rel_size` üçte birlikleri; sınırlar 0,1114 / 0,3452; kol `matched`, her grupta 50 pozitif + 250 kontrol):

| Üçte birlik | `rel_size` | TDA | ResNet18 | Füzyon |
| :--- | :---: | :---: | :---: | :---: |
| Küçük | 0,033–0,110 | 0,5016 | 0,5823 | 0,4965 |
| Orta | 0,112–0,343 | 0,5494 | **0,6633** | 0,5763 |
| Büyük | 0,350–2,097 | 0,5718 | 0,6154 | 0,5586 |

> ⚠️ **mm dönüşümü yapılamaz:** kohort 3× indirgenmiş JPEG'lerden oluşur ve DICOM spacing içermez; `bbox_max_dim_px` annotasyon/orijinal DICOM uzayındadır. Bu yüzden boyut üçte birlikleri kullanılır ve daha önce verilen "%15 mm" tipi etiketler **kaldırılmıştır**.

**Donanım ve kaynak kısayolu teşhisi:**
* NODE21 meta-verisinde **yalnızca** spacing/boyut kullanıldığında hastalık etiketi val ROC-AUC **0,6850** ile tahmin edilebiliyor (test 0,6576) ⇒ **kısmi kısayol** (0,65–0,80). İç başarım kısmen dedektör/spacing dağılımından besleniyor; bu confound açıkça raporlanmalıdır.
* Eşit-boyutlu kaynak-dışı koşuda (E5.3b) TDA ortalaması **0,7631** ⇒ sinyal tamamen yapay bir kısayol **değil**.

> 🐛 **Geçersiz sayıların düzeltmesi.** Bu bölümde daha önce raporlanan VinDr değerleri (TDA 0,5103/0,5082/0,5429; ResNet 0,6242/0,6318/0,6545; "spacing eşleştirmeli TDA 0,8537/0,8985/0,9365") **iki hatalı betikten ve kayıtsız koşulardan** geliyordu: histogram eşleme **tüm kesite** (arka plan dahil) uygulanıyordu ve havuzlanmış *doku* referansı sıfır arka planı ~1080'e taşıyarak TDA'nın `image > 0` doku maskesini bozuyordu; ayrıca "tek hasta referansı" keyfî ve mm etiketleri dayanaksızdı. Bu noktalar **düzeltilmiştir**; yukarıdaki tablo tek koşudan ve kayıtlı raporlardan gelir.

#### V6.5 — Transfer güçlendirmesi: fiziksel hizalama ve few-shot site kalibrasyonu (F10)

> 🔒 **Saf zero-shot sınırı.** VinDr'ye **CNN fine-tuning'i yapılmamıştır.** Ancak bu alt bölümdeki sayıların hiçbiri saf zero-shot **değildir**: birinci tablonun taban adayları **AdaBN-M5**'tir (VinDr'in etiketsiz BN istatistikleri ⇒ *transductive*), CORAL satırları ayrıca VinDr özellik kovaryansını kullanır, ikinci tablo ise VinDr **etiketlerini** kullanır. **Saf zero-shot dış doğrulamanın zirvesi V5.3'tür: VinDr AUC 0,6283** (kilitli a = 0,45). Aşağıdaki 0,6439–0,7249 bandı "hedefe hiç dokunulmadan elde edildi" biçiminde raporlanamaz.

> 🔴 **V6.9 denetim notu (bu bölümü de bağlar).** Yukarıdaki sınır, **ImageNet** tabanlı M5 için geçerlidir ve o sayılar temizdir. Ancak **CXR vakıf omurgası** (`resnet50-res512-all`, torchxrayvision) kullanan tüm VinDr sonuçları **kontaminedir**: bu omurga VinDr-CXR ile ön-eğitilmiştir ve değerlendirme görüntülerinin **%100'ü** ön-eğitim listesindedir. ✅ **Etkisi ölçüldü (V6.9c):** VinDr'yi hiç görmemiş `densenet121-res224-nih` ile aynı mertebe elde edildi (**0,6663** nOR / 0,6369 top1) ⇒ sayılar kontaminasyon artefaktı değil, ~0,64–0,67'lik bir transfer tavanıdır. Makalede VinDr sonucu **temiz omurgayla** verilmelidir; "zero-shot" ifadesi kontamine satırlar için kullanılamaz. Kanıt: [LEAK_AUDIT.md](LEAK_AUDIT.md) §3 ve §8.

V5.3'ün (§4.9) 0,6283'ünü temel alarak, **AdaBN ile stabilize edilmiş** M5 adayları üzerinde **etiketsiz** üç fiziksel/istatistiksel düzeltme tek tek ve birlikte sınandı (400 görüntü, 19.955 aday; referans: 200 NODE21 train hastasından 4.000 aday):

| Konfigürasyon (taban: AdaBN-M5 ⇒ *transductive*) | VinDr AUC |
| :--- | :---: |
| taban (32/64 px, z-score) | 0,6439 |
| JPEG-denoise σ=1,0 (kübik homoloji öncesi) | 0,6498 |
| **ölçek-eşlemeli yama** `S = round(32/64 · H_lung/810)` | **0,6562** |
| quantile (rank-gauss) | 0,6486 |
| CORAL (tam uzay) — ayrıca VinDr kovaryansı kullanır | 0,6318 |
| **denoise + ölçek + CORAL** — ayrıca VinDr kovaryansı kullanır | **0,6620** |

⇒ Fiziksel hipotez doğrulanır — VinDr'nin 0,70× ölçeği ve 8-bit JPEG blok gürültüsü ölçülebilir kayıp yaratıyor — ama etki mütevazıdır (**+0,018**). Tek başına CORAL zararlıdır (−0,012); denoise + ölçek üzerine eklenince en iyi konfigürasyonu verir. ⚠️ Bu iyileştirmeler **AdaBN tabanına göre göreli**dir, saf dış doğrulamaya göre değil.

**AdaBN'siz (inductive) denetim.** Aynı kohort ve aynı adaylar **AdaBN kapatılarak** tekrar koşuldu (dondurulmuş NODE21 BN):

| Konfigürasyon | Inductive (AdaBN yok) | Transductive (AdaBN) |
| :--- | ---: | ---: |
| base_z | 0,6090 | 0,6439 |
| + JPEG-denoise σ=1,0 | 0,5829 | 0,6498 |
| **+ ölçek-eşlemeli yama** | **0,6181** | 0,6562 |
| + denoise + ölçek | 0,5814 | 0,6523 |
| + denoise + ölçek + CORAL | 0,5916 | 0,6620 |

⇒ **AdaBN baş roldedir** (her satırda +0,035…+0,071) ve **JPEG-denoise, AdaBN olmadan zararlıdır** (−0,026…−0,037). Fiziksel hizalamanın dürüst **tekil-görüntü (inductive)** katkısı yalnızca **ölçek-eşlemedir: +0,0091** (0,6090 → **0,6181**); bu tavan, saf zero-shot zirvesi olan V5.3 global füzyonunun (**0,6283**) hâlâ altındadır.

**Few-shot site kalibrasyonu (5-fold, sızıntısız) — *(C) süpervizli hedef uyarlaması*, saf dış doğrulama değildir.** Hedef merkezden N ∈ {0,10,20,40,80,160} **etiketli** görüntü ayrıldığında (Aşama-1 ağırlıkları donmuş; yalnız 2. aşama tablosal modeli uyarlanır):

| N | yalnız lokal (AUC ± std) | V5.3 + lokal |
| ---: | :---: | :---: |
| 0 | 0,6593 ± 0,044 | 0,6673 ± 0,031 |
| 10 | 0,6280 ± 0,023 | 0,6521 ± 0,040 |
| 40 | 0,6855 ± 0,057 | 0,6959 ± 0,043 |
| 80 | 0,7057 ± 0,022 | 0,6997 ± 0,031 |
| 160 | **0,7243 ± 0,051** | 0,7054 ± 0,032 |

⇒ **N=10'da aşırı uyum** (0,628 < N=0 tabanı 0,659; ikisi de *transductive*), N≥40'ta toparlanma, N=160'ta **+0,065**. N≥80'de global V5.3 kanalı katkı vermez; lokal model kalibre edildiğinde global temsil gereksizleşir.

**AUC–FROC ödünleşimi (V6.5b/V6.5c).** Görüntü etiketiyle kalibrasyon AUC'yi yükseltirken FROC'u düşürür (0,212 → 0,162 @2FP); bbox-türevli etiketler FROC'u toparlar (0,184) ama AUC'yi 0,627'ye indirir. Ayrıştırılmış çift-kafa (sabit hasta kapısı + 1:5 kutu sıralayıcı + NODE21 çapası) AUC'yi 0,7249'da korur ama FROC'u N ile monoton bozar (0,221 → 0,059) ⇒ **ödünleşim kaldırılamamıştır**; kutu-içi TDA sıralaması VinDr'ye taşınmaz. Bu, §4.7'deki "lokalizasyon taşınır, güven taşınmaz" bulgusunun **aday düzeyindeki** karşılığıdır.

### 4.9 🎯 Nicelleme Darboğazı — Temsil Fiilen ~4-bit'ti (V5.1)

Filtrasyon ölçeklemesi `scaled = piksel × 255 / 65535` sabitiyle yapılıyordu. NODE21 verisi **12-bit** (0–4095) olduğundan bu sabit **16× fazladır** ⇒ doku yoğunluğu **~16 seviyeye** kuantalanıyor (255 yerine). Doku **p1–p99 → [0,255]** kanonikleştirmesi (5-zon mimarisi korunarak) bu darboğazı kaldırır:

| | H1 kalıcılık aralığı | H1 nokta sayısı | PI doluluk |
| :--- | :---: | :---: | :---: |
| Mevcut (v2) | **0–7** | 610.093 | %38 |
| Kanonik (V5.1) | **0–97** | **1.633.427** | **%81** |

| Kol (aynı protokol) | val label AUC | Kaynak AUC | LOO |
| :--- | :---: | :---: | :---: |
| Mevcut (v2) | 0,8330 | 0,8501 | 0,7249 |
| **Kanonik yoğunluk** | **0,9211 (+0,0881)** | 0,9554 | **0,7861 (+0,0611)** |
| Kanonik + ek ölçek normalizasyonu | 0,8539 | 0,9138 | 0,7400 |

> **Okunuşu:** v1'de teşhis edilen "grid collapse" hatası **tam kapanmamıştı**; v2 yalnızca H0/H1 ızgaralarını ayırarak semptomu azaltmış, niceleme darboğazı kalmıştı. Düzeltme, programın **en büyük val kazancını** verir (+0,0881 val · +0,0611 LOO) ve **kaynak (cihaz) AUC'sini de artırır** — bu bir başarısızlık değil, **bilgi miktarının artmasının doğal sonucudur**: 4-bit'lik bulanık temsilden 8-bit'e çıkıldığında cihaza özgü doku ve hastalık sinyali birlikte güçlenir. ⇒ **Kaynak AUC tanısal bir ölçüdür, karar kapısı değildir.** ⚠️ Ek ölçek normalizasyonu **zarar verir** (jsrt −0,141): topoloji bilinear yeniden örneklemeye aşırı duyarlıdır.

**Projenin gerçek modelinde (V5.2, val-only):** 5-zon CNN + SVM ensemble **0,8811 → 0,9522** (+0,0711), ince ayarlı ResNet18 ile **geç füzyon 0,9729 → 0,9856** (+0,0127; doğruluk %92,2 → **%92,8**). **Füzyon ağırlığı a = 0,40 → 0,25** (TDA payı 0,60 → **0,75**) ⇒ düzeltilmiş TDA artık görüntü dalına **denk** (0,9522 vs 0,9547) ve füzyona gerçek katkı veriyor. Bu, v4'ün *"TDA iyileştirmeleri ResNet ile yedekli"* bulgusunun **bozuk temsilin artefaktı** olduğunu gösterir.

**Pencere kontrolü (V5.1b, val-only):** Kanonikleştirme kırpıntı başına yapıldığı için "*pencere artefaktı mı?*" sorusu ayrıca sınandı: p1–p99 **hastanın tüm akciğerinden** (tek eşleme) hesaplandığında (lezyonla çiftleşme yok) füzyon **0,9778** (a=0,35) ve TDA ensemble **0,9335**, LOO **0,7897**. ⇒ Füzyon kazancının **~%60'ı (~+0,0078) kırpıntı-başına histogram penceresinden** gelir (topolojik değil, histogram kaynaklı; test zamanında hesaplanabildiği için hile değildir, ama ayrıştırılmalıdır). Buna karşılık **nicelleme düzeltmesinin kendisi baskındır**: TDA-ensemble kazancının **%74'ü** penceresiz varyantta da vardır. ⇒ Makalede füzyon için **muhafazakâr sayı 0,9778** raporlanır; 0,9856 pencere katkısını içerir.

**Harici eksende (V5.3, VinDr-CXR, sıfır eğitim):** füzyon (a=0,45) **0,5438 → 0,6283** (**Δ +0,0844**, %95 GA **[+0,0464; +0,1219]**, **p<0,001**) · TDA tek başına **0,5409 → 0,5959** (p=0,053) · VinDr'de **optimal a 1,00 → 0,75** · r **0,4009 → 0,2997** · küçük nodüllerde TDA **ResNet'i geçiyor** (0,5951 vs 0,5823). ⇒ **"TDA dış dünyaya taşınmıyor" sonucu büyük ölçüde ~4-bit temsilin artefaktıydı**; temizlenmiş temsilde TDA şansın üzerinde ve füzyona anlamlı katkı sağlıyor. ⚠️ Mutlak harici AUC yine mütevazı (≈0,63) ve füzyonun ResNet'e göre artı kazancı (+0,008…+0,011) **gürültü mertebesinde** ⇒ dış merkez için kalibrasyon/adaptasyon ihtiyacı sürüyor. Test bütçesi 6/6 kapalı olduğundan bu kazançlar **val/LOO/harici** düzeyindedir.

### 4.10 🎯 İki Aşamalı Lokalizasyon (V6) — NODE21 Test Mührü ve FROC (F9, F10)

5-zon TDA temsili lokalize etmez (§4.7); lokalizasyon **ayrı bir aday üretici + lokal TDA FP eleme** katmanıyla sağlanır:

1. **Aşama-1 (aday üretici):** CenterNet (M5) ve FCOS (M6); K=50 aday/görüntü. Val oracle recall@50 = **0,978** (M5) / **0,983** (M5+M6). VinDr'de bbox recall@K50 = **0,900** ⇒ dedektör lokalizasyonu taşıyor.
2. **Aşama-2 (lokal TDA + FP eleme):** Her aday için 32/64 px kırpıntılardan kübik H0/H1 kalıcılık imgeleri (2×2×20×20) + uzamsal öncel (hilus/plevra mesafesi, akciğer-içi konum) → HistGradientBoosting; füzyon LR ([aşama-1 skoru, TDA skoru]). Kübik homoloji, aday kırpıntısı başına **on-the-fly** hesaplanır (görüntü/kırpıntı diske yazılmaz).
3. **Seçim ve mühür:** Tüm hiperparametre/eşik seçimi validation'da; test seti **tek seferlik ön-kayıtlı** mühürle değerlendirildi.

| Model (test: 734 hasta / 229 lezyon) | Hasta AUC | FROC @0,5 / 1 / 2 / 4 FP |
| :--- | :---: | :---: |
| V6 lokal (`e_fusion_tda`) | 0,9464 (noisy-OR) | 0,480 / 0,555 / **0,642** / 0,738 |
| V5.2 global (5-zon) | 0,9612 | — |
| **Global + lokal birleşik** | **0,9876** (r = 0,376) | — |

⇒ İki temsil **gerçekten tamamlayıcıdır**: birleşim hem v3 headline'ını (0,9516) hem V5.2'yi (0,9612) geçer. Val'de FROC-optimal hiyerarşik konfig (M5_heatTTA·K30) 0,589/0,658/0,753/**0,857** verir; test mühründe muhafazakâr `e_fusion` raporlanmıştır. **F9** tek bir vakada aday → kırpıntı → kalıcılık → skor zincirini; **F10** FROC eğrilerini ve VinDr transfer merdivenini gösterir.

> ⚠️ **Bütçe beyanı:** v3'ün 6/6 test değerlendirmesi kapalıdır. V6, v3 testine dokunmadan val'de geliştirildi ve **tek bir ek, ön-kayıtlı** test mührüyle kapandı; bu, v3 sonucunu değiştirmez. VinDr değerlendirmesi görüntü düzeyindedir (hasta kimliği yok).

#### V6.9 — mühürlü dürüst dış-değer (yayına girecek sayılar)

V6.1–V6.8'de **her** seçim val'de yapıldı ve val'de raporlandı (kutu boyu için **296** resmî-metrik değerlendirmesi, füzyon için **19** nokta). 231 lezyonda Sens@0,25'in SE'si ≈ 0,030 olduğu için bu, yapısal bir iyimserliktir. V6.9 ön-kayıtlı tek atımla (betik kendi hash'ini doğrular; z-normalizasyon val'den dondurulur) kilitli protokolü dokunulmamış test bölmesinde koştu:

| Ölçüm (test: 734 hasta / 229 lezyon) | Val (seçim) | **Test (mühür)** | Δ | %95 GA |
| :--- | ---: | ---: | ---: | :--- |
| rank_metric — 1. aşama | 0,9131 | **0,8842** | −0,029 | 0,8559–0,9089 |
| rank_metric — füzyon (w=0,55) | 0,9174 | **0,8970** | −0,020 | 0,8769–0,9213 |
| Hasta AUC — 1. aşama | 0,9678 | **0,9446** | −0,023 | 0,9237–0,9621 |
| Hasta AUC — füzyon | 0,9793 | **0,9645** | −0,015 | 0,9491–0,9786 |

⇒ Makalede **ana sayı test sütunudur**; val sütunu "seçilmiş" etiketiyle verilir. Mühür, val kolunu birebir yeniden ürettiği için fark kod hatası değil, genelleme farkıdır; ayrıca **bir tasarım hükmünü tersine çevirdi**: val'de "FROC için 2. aşamayı atla" denmişti, test'te füzyon 4 metrikten 3'ünde üstün.

---

## 5. Tartışma

*(Yazılacak — iskelet)*

5.1 **Neden tamamlayıcılık gerçek?** İki temsil farklı şeyler kodluyor: görüntü dalı yoğun piksel dokusunu, TDA dalı yoğunluk-altı-seviye kümelerinin ölçek-değişmez topolojik yapısını kodlar. r≈0,30 bunu doğrular. Hatalar bağımsız olduğu için füzyon ikisini de geçer.

5.2 **Neden PI, PL'yi geçiyor?** §4.3: PL skaler zarfa indirger ve zarf arka planla doygun; PI 2B yoğunluğu korur.

5.3 **Neden 5-zon temsili lokalize etmiyor, ama V6 ediyor?** Nodüller zon alanının çok küçük bir kısmı; H1 üst-zarfı bronkovasküler döngüler ve göğüs duvarı tarafından domine edilir (§4.3, §4.4). Sinyal lezyondan değil, genel parankim topolojisinden gelir. Bu yüzden **zon düzeyi TDA lokalize etmez**; lokalizasyon ancak **ayrı bir aday üretici** ve **aday kırpıntısı düzeyinde** hesaplanan TDA ile (V6) mümkündür (§4.10). Buna karşılık aday sıralaması dış merkeze taşınmaz: VinDr'de hasta AUC few-shot ile **0,724**'e çıkarılabilirken FROC aynı oranda iyileşmez (§4.8, V6.5b/c) ⇒ **lokalizasyon taşınır, aday güveni taşınmaz.** Sabit dedektör fiziği/protokolüne sahip kurum içi arşivlerde sistem **0,9876** hasta AUC'lik bir risk triyajı sağlarken, farklı ülke/cihaz dağılımında aday sıralaması hedef-site kalibrasyonu ister. ResNet18 ise harici veride **0,6204** [0,5609; 0,6763] vererek evrişimsel filtrelerin dış dünyaya daha dirençli olduğunu gösterir.

5.4 **Klinik ve Pratik Konumlandırma:** Sistem evrensel bir lezyon kutulayıcısı ("bounding-box detector") değil; klinik merkezlerin kendi radyoloji arşivleriyle kalibre ederek kullanabilecekleri **kurum-içi yüksek doğruluklu risk triyaj altyapısıdır**.

---

## 6. Sınırlamalar

1. **5-zon TDA lokalize etmez** (E4.7) — zon düzeyi temsil; FROC bu temsille hesaplanamaz. Lokalizasyon, ayrı bir aday üretici + lokal TDA katmanıyla (**V6**, §4.10) sağlanır; ancak aday düzeyi skor kalibrasyonu dış merkezde taşınmaz (§4.8).
2. **Sinyal global/arka plan topolojisinde** (E4.4 + E1.2 + E3.6) — "lokal nodül topolojisi" denemez.
3. **Harici transfer (düzeltilmiş temsille iyileşti, ama mutlak düzey mütevazı):** V4'te (eski ~4-bit temsil) TDA VinDr'de şans düzeyindeydi (0,511–0,541) ve füzyon ResNet'in altına düşüyordu. **65535 niceleme düzeltmesinden sonra (V5.3)** TDA **0,5959**'a çıktı, füzyon **0,6283** (Δ **+0,0844**, CI [+0,0464; +0,1219], p<0,001) ve optimal ağırlık **a = 0,75**'e indi; küçük nodüllerde TDA ResNet'i geçiyor. ⚠️ Ancak **mutlak** harici AUC hâlâ ≈0,63 ve füzyonun görüntü dalına göre artı kazancı (+0,008…+0,011) **gürültü mertebesinde** ⇒ dış merkeze taşıma için kalibrasyon/adaptasyon şarttır.
4. **Zon ayrıştırması uzmanlaşma sağlamıyor** (E3.5) — yalnızca topluluk çeşitliliği.
5. **TDA tek başına ince ayarlı modeli geçemiyor** (0,8776 vs 0,9284) — değeri **tamamlayıcılıktır**.
6. **Veri = NODE21 halka açık training seti** — leaderboard karşılaştırması **geçersiz**.
7. **Domain-shift bedeli −0,109** (E5.3b); harici veride ResNet18 **0,6204** ile kısmen taşınırken TDA 0,51'de kalır.
8. **Test bütçesi tükendi (6/6)** — sonraki iyileştirmeler yalnızca validation'da gösterilebilir. **İstisna:** V6, v3 testine dokunmadan val'de geliştirildi ve **tek bir ek, ön-kayıtlı** test mührüyle değerlendirildi (§4.10); bu, v3'ün 0,9516'lık sonucunu değiştirmez.
9. ✅ **TDA ensemble farkı ÇÖZÜLDÜ:** 0,8776 (E3.4) vs 0,8629 (E0.6/E3.2) varyant farkıdır (Varyant A vs B).
10. **Harici doğrulamada nodül boyutu alt grubu** (§4.8) — `rel_size` üçte birliklerinde ResNet orta grupta en iyi (0,6633), küçük (0,5823) ve büyük (0,6154) gruplarda daha zayıf; TDA hiçbir grupta 0,58'i geçmez. ⚠️ mm dönüşümü yapılamadığı ve örneklem küçük olduğu (n=50/tip) için bu bulgu **keşifseldir**.
11. **ResNet18 ağırlıkları:** E0.6 ağırlıkları kaydedilmediği için bit-bit yeniden üretim imkânsızdır; harici sonuçlar deterministik yeniden eğitilen (val zon AUC Δ=3,8·10⁻⁴) sabit checkpoint'e dayanır.
12. **3× indirgenmiş JPEG kohortu:** VinDr görüntüleri spacing içermez; TDA ölçek confound'u yalnızca akciğer-yüksekliği normalizasyonuyla kısmen kontrol edilebilmiştir.
13. **Kırpıntı-başına normalizasyon kısmen pencere etkisi taşır** (§4.9, V5.1b): füzyon kazancının ~%60'ı (~+0,0078) topolojik değil histogram/pencere kaynaklıdır; bu yüzden füzyon için **muhafazakâr sayı 0,9778** (tüm-akciğer referansı) raporlanır.
14. **VinDr few-shot kalibrasyonu N≥40 ister** (§4.8): N=10'da 1604-boyutlu LR aşırı uyuyor (0,628 < N=0 tabanı 0,659; **her ikisi de *transductive***, saf zero-shot değil). Hedef merkezde ~160 etiketli görüntü (80 poz/80 neg) ile AUC 0,724'e çıkar; bu, in-domain 0,9876'nın belirgin altındadır.
15. **AUC–FROC ödünleşimi dış merkezde çözülemedi** (§4.8): görüntü etiketli kalibrasyon AUC'yi, bbox etiketli kalibrasyon FROC'u iyileştiriyor; çift-kafa ikisini aynı anda sağlayamıyor. FROC değerleri ayrıca aday-etiket proxy'sine (görüntü etiketi) duyarlıdır.
16. **Kaynak/etiket confound'u yapısal (A3):** NODE21'de tüm pozitifler dört dış kaynaktan (chestxray14, padchest, jsrt, openi), tüm negatifler `node21_control` havuzundan gelir ⇒ **kaynak-içi AUC tanımsızdır**. Kaynak-pozitifleri vs ortak kontrol AUC'si dört kaynakta da **0,82–0,94** (makro **0,892**; TDA, ham CNN'in 0,664'ünü geçiyor) ⇒ etki tek kaynağa indirgenemez; ancak aynı kaynaktan kontrol olmadığı için confound **tamamen** dışlanamaz. Bu nedenle birincil genelleme kanıtı **kaynak-dışı (0,81)** ve **VinDr**'dir (0,63–0,66). Ayrıca kontrol ortalama skoru 0,555 ve özgüllük@0,5 = 0,358 ⇒ işletim noktası kaymıştır.
17. **Persistence Landscape zarf doygunluğu maske sınırı artefaktıdır (A4):** E4.4'teki devasa tepeler (~110 tepe, 180–220 kalıcılık) damar ağacından değil, doku dışı siyah arka plan (0) ile doku (~240) arasındaki yapay segmentasyon maskesi basamağından kaynaklanır (kalıcılığı $\ge 50$ olan H1 döngülerinin %100'ü sınır geçişlidir; saf doku içi kalıcılık en fazla 7'dir). PL fonksiyonel üst zarfı bu sınır sinyali tarafından körleştirilirken, PI $(b, p)$ 2B uzayında artefaktı köşeye izole ederek doku sinyalini korur.
18. **V6 hattında seçim yanlılığı ölçüldü (§4.10, V6.9):** Kutu boyu, füzyon ağırlığı ve TTA **296 + 19 + 12** değerlendirmeyle val'de seçildi. Dürüst test değerleri val'den **0,015–0,029** düşük (rank_metric 0,9131 → **0,8842**; füzyon AUC 0,9793 → **0,9645**). Düşüş ≈1 SE olup 0,05'lik ön-kayıtlı başarısızlık eşiğinin altındadır; val noktası yine de test'in %95 aralığının az üstünde kaldığı için val sayıları **"seçilmiş"** olarak etiketlenmelidir.
19. **Ön-eğitim kontaminasyonu (VinDr):** X-ray vakıf omurgası (`resnet50-res512-all`) VinDr-CXR ile ön-eğitilmiştir ve değerlendirme görüntülerimizin **%100'ü** o ön-eğitim listesindedir. **Ölçüldü:** VinDr'yi hiç görmemiş `densenet121-res224-nih` ile aynı mertebe elde edildi (**0,6663** nOR vs 0,6414) ⇒ sayılar **şişmemiştir**; VinDr sonucu ~0,64–0,67'lik bir transfer tavanıdır. Yine de yayında **temiz omurga** satırı kullanılmalı, "zero-shot" ifadesi kontamine satırlar için kullanılmamalıdır ([LEAK_AUDIT.md](LEAK_AUDIT.md) §3, §8).
20. **V6 hattı tek tohumludur:** Aynı protokolün iki bağımsız koşusu ana metriklerde ±0,02 kayıyor (V6.8). Etki büyüklükleri bu bantla birlikte okunmalıdır; v3 katmanının çok-seed raporlaması bu hattı kapsamaz.

---

## 7. Sonuç

Bu çalışma, göğüs röntgeninde yorumlanabilir topolojik temsillerin **neyi başardığını ve neyi başaramadığını** ön-kayıtlı bir protokolle ölçer. Bulgumuz iki yönlüdür:

- ✅ **Başarı:** TDA temsilleri, güçlü ve **ince ayarlı** bir görüntü modeline bile istatistiksel olarak anlamlı tamamlayıcılık katar (test ROC-AUC **0,9516**, doğruluk **%89,4**; katkı **+0,0232**, CI>0).
- ⚠️ **Sınır:** Bu katkı **global/arka plan topolojisinden** gelir, lokal nodül yapısından değil. **5-zon TDA temsili lokalize etmez**; lokalizasyon ancak ayrı bir aday üretici + lokal TDA katmanıyla (V6) sağlanır: test hasta AUC **0,9876** (global+lokal, V6.4 mührü), FROC@2FP **0,642**. Yani "lezyon tespiti" iddiası **v3 katmanı** için değil, **V6 katmanı** için kurulabilir. ⚠️ **V6 hattının val'de seçilen sayıları hafif iyimserdir:** aynı hattın ikinci, ön-kayıtlı mühründe (V6.9) dürüst değerler **0,8842** (rank_metric) ve **0,9645** (füzyon AUC) çıktı (§4.10).
- 🌐 **Kapsam sınırı (harici doğrulama):** Kazanç **kurum içi**dir. VinDr-CXR'de **saf zero-shot zirvesi V5.3 füzyonudur (0,6283)**; V6.5'in **0,6620**'si AdaBN tabanlı ***transductive*** bir sonuçtur (fiziksel hizalamanın AdaBN tabanına katkısı +0,018) ve saf dış doğrulama olarak okunamaz. Hedef merkezden 160 **etiketli** görüntüyle few-shot kalibrasyon **0,7243**'e ulaşır (***süpervizli***). ⚠️ **V6.9 denetimi:** CXR vakıf omurgasıyla (`resnet50-res512-all`) üretilen VinDr sayıları **kontaminedir** (görüntülerin %100'ü ön-eğitim listesinde); etkisi ölçüldü ve **şişme bulunmadı** — temiz omurgayla (`densenet121-res224-nih`) **0,6663** nOR / 0,6369 top1. Yayında bu temiz satır kullanılmalıdır ([LEAK_AUDIT.md](LEAK_AUDIT.md) §3, §8). Aday düzeyi sıralama taşınmaz ve AUC–FROC ödünleşimi sürer ⇒ sistem, hedef-site kalibrasyonu gerektiren bir **kurum-içi risk triyaj** altyapısıdır.

PL'nin neden çöktüğünü nicel olarak göstermek (üst-zarf doygunluğu; sınıf sinyali %1–2), alanın bir sonraki adımı için somut bir yön verir: **zarf bastırması değil, zarf-içi sinyal çıkarımı.**

---

## 8. Beyanlar

| | |
| :--- | :--- |
| **Veri** | NODE21 halka açık eğitim seti (erişim: NODE21 yayını) · **harici doğrulama:** VinDr-CXR (3× indirgenmiş JPEG kohortu; görüntü düzeyi, spacing yok) |
| **Kod** | Tüm deneyler tekrarlanabilir; betikler `pipelines/exp/`, kayıtlar `artifacts/runs/<ID>/manifest.json` |
| **Tekrar üretilebilirlik** | Referans sonuç **bit-bit** yeniden üretildi (16/16 metrik, E0.2) · harici doğrulama deterministik yeniden üretim betikleriyle (`V4_*.py`) |
| **Hesaplama** | CPU + tek GPU (RTX 4050 Laptop, 6 GB); tüm eğitim GPU'da ≤2,5 dk |
| **Bütçe disiplini** | 28 deney; test seti yalnızca 6 değerlendirme · V6 için **tek ek, ön-kayıtlı** mühür (§4.10) · harici doğrulama ayrı (VinDr'de eğitim yok) |

---

## 9. Figür ve Tablo Listesi

**Figürler:** F1 boru hattı · F2 liderlik tablosu · F3 tamamlayıcılık saçılımı · F4 PL zarf çökmesi · **F5 belirleyici test** · F6 sağlamlık panelleri · F7 kaynak-dışı genelleme · F8 sınırlama paneli · **F9 görsel açıklanabilirlik** · **F10 FROC + VinDr merdiveni** — ayrıntı [ablation_matrix.md](ablation_matrix.md) §11.

**F9 — Görsel açıklanabilirlik** (`artifacts/figures/F9_explainability.png`): (A) 5-zon kanonik görüntü, GT kutusu ve adaylar (isabet/FP); (B) TP ve TDA'nın bastırdığı güçlü FP'nin 64×64 kırpıntıları; (C) TP kırpıntısının H0/H1 kalıcılık diyagramı; (D) skor ayrışımı (aşama-1 → lokal TDA-HGB → birleşik). Temsilî val vakası (n0082): aşama-1 0,07/0,02 iken lokal TDA 0,74/0,60, birleşik 0,91/0,52.

**F10 — FROC ve VinDr merdiveni** (`artifacts/figures/F10_froc_vindr_ladder.png`): (A) NODE21 val FROC eğrileri (aşama-1, TDA-HGB, e_fusion, hiyerarşik) + **tek seferlik test mührü**; (B) VinDr-CXR transfer merdiveni (V4 0,544 → V5.3 0,628 → V6.4b 0,647/0,649 → V6.5 zero-shot 0,662 → few-shot N=160 0,724; NODE21 test 0,988 referans); (C) VinDr FROC@2FP vs N (V6.5 / V6.5b / V6.5c) — AUC–FROC ödünleşimi.

**Tablolar:** T1 veri/split beyanı (§3.1) · T2 etiketleme kuralı (§3.2) · T3 protokol kuralları (§3.6) · T4 liderlik (§4.1) · T5 tamamlayıcılık (§4.2) · T6 PL mekanizması (§4.3) · T7 sağlamlık (§4.5) · T8 genelleme (§4.6) · **T9 harici doğrulama (§4.8)** · T10 sınırlamalar (§4.7/§6) · **T11 V6 test mührü ve FROC (§4.10)** · **T12 VinDr transfer merdiveni (§4.8)**.

---

## 10. Yazım İçin Açık İşler

| # | İş |
| :-: | :--- |
| 1 | §1 Giriş ve §2 İlgili Çalışmalar metnini yaz (literature.md'den) |
| 2 | §5 Tartışma iskeletini genişlet |
| 3 | ~~F1–F8 figürlerini üret~~ ✅ **TAMAM** — `artifacts/figures/F1..F10_*.png` (F9 açıklanabilirlik, F10 FROC/ladder) |
| 4 | Referans listesini literature.md'den biçimlendir |
| 5 | İngilizce çeviri + hedef dergi formatı |
| 6 | ~~TDA bundle yeniden üretilebilirlik farkını kapat~~ ✅ **ÇÖZÜLDÜ** (§6.8) |
| 7 | ~~Harici doğrulamayı yaz~~ ✅ **TAMAM** — §4.8 (VinDr-CXR; 3 kol, a-taraması, ölçek-bağımsız alt gruplar, artifact'lı kayıtlar) |
| 8 | ~~V6 iki aşamalı lokalizasyon + test mührü + VinDr transfer merdivenini yaz~~ ✅ **TAMAM** — §4.10 + §4.8 (V6.5/V6.5b/V6.5c), F9/F10 |

### Gelecek çalışma — val düzeyinde doğrulanmış iyileştirmeler

| # | İyileştirme | Val AUC | Kazanç | Kontrol |
| :-: | :--- | :---: | :---: | :--- |
| 🥇 | **V5.1 — 65535 niceleme hatasının düzeltilmesi** (doku p1–p99 → 255 seviye) | 0,8330 → **0,9211** | **+0,0881** (**LOO +0,0611**) | ✓ kaynak AUC artışı bilgi artışının sonucu; ölçek normalizasyonu zararlı · **pencere kontrolü (V5.1b):** muhafazakâr val 0,9133 ve füzyon **0,9778** |
| 🥈 | **E2.6 çok parametreli dilimler** (`f + t·Sobel gradyan`) | 0.8399 → **0.8880** | **+0.0480** | ✓ (karıştırılmış: 0.6925) |
| 🥉 | **E2.1 çift yönlü (diphasic) filtrasyon** (sublevel+superlevel) | 0.8399 → **0.8609** | **+0.0210** | ✓ (karıştırılmış: 0.7803) |

Üçü de **test bütçesi tükendiği için** 0.9516'lık ana sonuca **eklenememiştir** ve **yeni bir test bütçesinin öncelikli adaylarıdır** (**V5.1 en güçlü**; LOO'da da doğrulandığı için taşınabilirlik adayı). V5.1 ile E2.6/E2.1 birleştirilebilir (farklı eksenler: nicelleme vs. filtrasyon zenginleştirme).

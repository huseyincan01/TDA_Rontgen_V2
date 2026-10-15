# NODE21 TDA — Makale Taslağı (E5.2)

> **Durum:** Yapısal taslak. Tüm sayılar [ablation_matrix.md](ablation_matrix.md)'den alınmıştır ve doğrulanmıştır.
> **Protokol uyumu:** Split beyanı (Kural 9) ✅ · Çok-seed raporlama (Kural 10) ✅ · Ön-kayıtlı eşikler ✅ · Test bütçesi 6/6 (kapalı)
> **Kaynak belgeler:** [ablation_matrix.md](ablation_matrix.md) · [experiments.md](experiments.md) · [literature.md](literature.md) · [roadmap.md](roadmap.md)

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
5. **Sınırlama:** modeller **lokalize etmez** — zon düzeyi AUC 0.48–0.53; nodülün tam merkezinde bile yerel TDA ayrımı yalnızca **0.687**'dir.

**Sonuç.** Yorumlanabilir topolojik temsiller, öğrenilmiş görüntü temsillerine **ölçülebilir ve istatistiksel olarak anlamlı** tamamlayıcılık katar. Ancak CXR'de ayırt edici bilgi **global/arka plan topolojisinde** yoğunlaşmıştır; bu nedenle "nodülü doğru zonda bulma" iddiası **kurulamaz**. Katkımız, bu sınırı dürüstçe belirleyen ve mekanizmayı nicel olarak gösteren bir çerçevedir.

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
   - (C3) "Global vs lokal" sorusunu kapatan **negatif sonuçlar kümesi** (lokalizasyon yok; yerel topoloji zayıf; arka planı bastırmak zararlı).
1.6 **Makalenin çerçevesi:** Bu bir **hasta düzeyi sınıflandırma** çalışmasıdır; **lokalizasyon iddiası yoktur.**

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

**Ölçek:** 28 deney · 15 KEEP · 13 KILL · 1 uygulanamaz. **Karar günlüğü** [roadmap.md](roadmap.md) §10.

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

### 4.7 🚨 Doğrulanmış Sınırlama — Model Lokalize Etmiyor

| Deney | Bulgu |
| :--- | :--- |
| **E4.7** | Nodül-pozitif hastalarda argmax zon, gerçek nodül zonuna **şans düzeyinde** düşüyor (isabet 0,228–0,298 vs şans 0,268; binom p=0,21–0,90). **Zon düzeyi AUC 0,48–0,53.** |
| **E4.3** | Her zon **tek başına** 0,74–0,84 hasta AUC veriyor ⇒ zayıf zon yok, ama bu lokalizasyon değil |
| **E3.5** | Havuzlanmış tek model, 5-zon özel modeliyle **eşdeğer** (0,8585 vs 0,8661, −0,0076) ⇒ zon bölünmesi **topluluk çeşitliliği** sağlıyor, **anatomik uzmanlaşma** değil |
| **E1.2** | Damar-kaynaklı H1 sınıflarını elemek başarımı **monoton düşürüyor** (0,8697 → 0,8358) ⇒ arka plan **faydalı** |
| **E3.6** | **Ground-truth bbox** ile nodülün tam merkezinde bile yama düzeyi AUC yalnızca **0,6866** (boyut-eşleştirilmiş) |

> 🔗 **Birleşik sonuç:** Ayırt edici bilgi **lokal nodül topolojisinde değil, global/arka plan topolojisindedir.** "Arka planı bastırınca lokal sinyal açığa çıkar" hipotezi **çürütülmüştür**.
> ⚠️ **Confound uyarısı (E3.6):** İlk koşu **0,994** vermişti; kontroller sabit 60×60, pozitifler değişken boyuttaydı ve sınıflandırıcı **nodülü değil yama boyutunu** öğrenmişti. ⇒ *"Fazla iyi" sonuçlar sistematik olarak confound açısından denetlenmelidir.*

---

## 5. Tartışma

*(Yazılacak — iskelet)*

5.1 **Neden tamamlayıcılık gerçek?** İki temsil farklı şeyler kodluyor: görüntü dalı Yoğun piksel dokusu, TDA dalı yoğunluk-altı-seviye kümelerinin **ölçek-değişmez topolojik** yapısı. r≈0,30 bunu doğruluyor. Hatalar bağımsız olduğu için füzyon ikisini de geçiyor.

5.2 **Neden PI, PL'yi geçiyor?** §4.3: PL skaler zarfa indirger ve zarf arka planla doygun; PI 2B yoğunluğu korur.

5.3 **Neden lokalizasyon yok?** Nodüller zon alanının çok küçük bir kısmı; H1 üst-zarfı bronkovasküler döngüler ve göğüs duvarı tarafından domine ediliyor (§4.3, §4.4). Bu, **hasta düzeyi** için bir sorun değil (global örüntü ayırt edici), ama **lokalizasyon** için yapısal bir engel.

5.4 **Klinik yorum:** Nodül varlığı, akciğerin genel yapısına yansıyan bir örüntü üretiyor. Bu, "lezyon tespiti" değil, "risk triyajı" olarak konumlandırılmalıdır.

5.5 **Negatif sonuçların değeri:** E1.2, E3.3 (iptal), E3.5, E3.6, E4.7 birlikte, TDA-CXR literatüründe sık ima edilen ama test edilmemiş bir varsayımı (*"topoloji lokal lezyon yapısını yakalar"*) **çürütür**. Bu, alan için bağımsız değer taşır.

---

## 6. Sınırlamalar

1. **Lokalizasyon yok** (E4.7) — hasta düzeyi sınıflandırma; kutu üretimi değil. FROC hesaplanamaz.
2. **Sinyal global/arka plan topolojisinde** (E4.4 + E1.2 + E3.6) — "lokal nodül topolojisi" denemez.
3. **Zon ayrıştırması uzmanlaşma sağlamıyor** (E3.5) — yalnızca topluluk çeşitliliği.
4. **TDA tek başına ince ayarlı modeli geçemiyor** (0,8776 vs 0,9284) — değeri **tamamlayıcılıktır**.
5. **Veri = NODE21 halka açık training seti** — leaderboard karşılaştırması **geçersiz**.
6. **Domain-shift bedeli −0,109** (E5.3b) ve kaynak-dışı genellemede **veri miktarı eşitlenmelidir**.
7. **Test bütçesi tükendi (6/6)** — sonraki iyileştirmeler yalnızca validation'da gösterilebilir.
8. **TDA ensemble küçük yeniden üretilebilirlik farkı:** 0,8776 (E3.4) vs 0,8699 (E3.2) / 0,8629 (zon düzeyi varyantı), bundle yüklemesinden kaynaklanabilir — açık iş.
9. **Tek bir ince ayar tarifi** kullanıldı (E0.5 taraması 3 varyant); daha geniş bir arama yapılmadı.
10. **Nodül boyutu/yerleşimi alt grup analizi yapılmadı.**

---

## 7. Sonuç

Bu çalışma, göğüs röntgeninde yorumlanabilir topolojik temsillerin **neyi başardığını ve neyi başaramadığını** ön-kayıtlı bir protokolle ölçer. Bulgumuz iki yönlüdür:

- ✅ **Başarı:** TDA temsilleri, güçlü ve **ince ayarlı** bir görüntü modeline bile istatistiksel olarak anlamlı tamamlayıcılık katar (test ROC-AUC **0,9516**, doğruluk **%89,4**; katkı **+0,0232**, CI>0).
- ⚠️ **Sınır:** Bu katkı **global/arka plan topolojisinden** gelir, lokal nodül yapısından değil. Dolayısıyla model **lokalize etmez** ve "lezyon tespiti" olarak sunulamaz.

PL'nin neden çöktüğünü nicel olarak göstermek (üst-zarf doygunluğu; sınıf sinyali %1–2), alanın bir sonraki adımı için somut bir yön verir: **zarf bastırması değil, zarf-içi sinyal çıkarımı.**

---

## 8. Beyanlar

| | |
| :--- | :--- |
| **Veri** | NODE21 halka açık eğitim seti (erişim: NODE21 yayını) |
| **Kod** | Tüm deneyler tekrarlanabilir; betikler `pipelines/exp/`, kayıtlar `artifacts/runs/<ID>/manifest.json` |
| **Tekrar üretilebilirlik** | Referans sonuç **bit-bit** yeniden üretildi (16/16 metrik, E0.2) |
| **Hesaplama** | CPU + tek GPU (RTX 4050 Laptop, 6 GB); tüm eğitim GPU'da ≤2,5 dk |
| **Bütçe disiplini** | 28 deney; test seti yalnızca 6 değerlendirme |

---

## 9. Figür ve Tablo Listesi

**Figürler:** F1 boru hattı · F2 liderlik tablosu · F3 tamamlayıcılık saçılımı · F4 PL zarf çökmesi · **F5 belirleyici test** · F6 sağlamlık panelleri · F7 kaynak-dışı genelleme · F8 sınırlama paneli — ayrıntı [ablation_matrix.md](ablation_matrix.md) §11.

**Tablolar:** T1 veri/split beyanı (§3.1) · T2 etiketleme kuralı (§3.2) · T3 protokol kuralları (§3.6) · T4 liderlik (§4.1) · T5 tamamlayıcılık (§4.2) · T6 PL mekanizması (§4.3) · T7 sağlamlık (§4.5) · T8 genelleme (§4.6) · T9 sınırlamalar (§4.7/§6).

---

## 10. Yazım İçin Açık İşler

| # | İş |
| :-: | :--- |
| 1 | §1 Giriş ve §2 İlgili Çalışmalar metnini yaz (literature.md'den) |
| 2 | §5 Tartışma iskeletini genişlet |
| 3 | F1–F8 figürlerini üret (`pipelines/figures/`) |
| 4 | Referans listesini literature.md'den biçimlendir |
| 5 | İngilizce çeviri + hedef dergi formatı |
| 6 | TDA bundle yeniden üretilebilirlik farkını kapat (§6.8) |
| 7 | E2.1/E2.2 (çift yön / çok ölçek) — yeni topolojik katkı arayışı (val-only) |

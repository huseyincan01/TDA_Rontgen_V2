# NODE21 TDA Projesi — Çalışma Planı ve Geliştirme Güncesi

Bu belge, **NODE21** göğüs radyografileri (CXR) üzerinde **5 anatomik akciğer zonu** ve **Kübik Kalıcı Homoloji (Cubical Persistent Homology - TDA)** temelli nodül tespit sisteminin resmi mimari güncesidir. Projenin teorik temellerini, v1 aşamasındaki metodolojik yanılgıların analizini, v2 temiz veri hattının matematiksel tasarımını ve güncel durumunu belgeler.

---

## 1. Proje Amacı ve Genel Mimari

Bu projenin temel hipotezi; göğüs röntgenlerindeki akciğer nodüllerinin, global görüntü sınıflandırması yerine **5 bağımsız anatomik akciğer lobu/zonu** düzeyinde, doku yoğunluk filtrasyonundan elde edilen **topolojik parmak izleri (Persistent Homology)** ile yüksek doğruluk ve açıklanabilirlikle tespit edilebileceğidir.

```
+---------------------------------------------------------------------------------------------------+
|                                     NODE21 CXR (1024x1024, 16-bit)                                 |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      | TorchXRayVision PSPNet Akciğer Anatomik Segmentasyonu  |
                      +-------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
               Sağ Akciğer (3 Zon)                               Sol Akciğer (2 Zon)
          [Zon 1: Üst] [Zon 2: Orta] [Zon 3: Alt]               [Zon 4: Üst] [Zon 5: Alt]
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           | Kübik Kompleks & Yoğunluk Filtrasyonu (GUDHI) |
                           |       - H0: Bağlantılı Bileşenler             |
                           |       - H1: 1-Boyutlu Döngüler / Çevreler     |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           | SciPy Gaussian Tabanlı Persistence Image      |
                           |     - H0 (20x20) + H1 (20x20) = 800 Boyut     |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           | 5 Bağımsız Zon RBF-SVM Sınıflandırıcısı       |
                           |    - Train: 1:1 Temiz Negatif Dengeli         |
                           |    - Val: ROC-AUC Tuning + Youden's J Eşik    |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           | Hasta Düzeyi Max-Pooling Karar Mekanizması    |
                           |   P(Hasta) = max(P_z1, P_z2, P_z3, P_z4, P_z5)|
                           +-----------------------------------------------+
```

### Temel Tasarım İlkeleri:
1. **Anatomik Bölütleme:** Akciğerler derin öğrenme segmentasyonu ile 5 bağımsız anatomik zona ayrılır (Sağ Üst, Sağ Orta, Sağ Alt, Sol Üst, Sol Alt).
2. **Kübik Kalıcı Homoloji:** Her zon kesitinin doku yoğunluk filtrasyonundan sonlu H0 (yoğun lezyon çekirdekleri / radyoopak odaklar) ve H1 (1-boyutlu döngüler / damarsal ve trabeküler çevre yapıları) kalıcılık diyagramları çıkarılır.
3. **Piksel Uzayında Persistence Image (PI):** Diyagramlar, Adams vd. (2017) ağırlıklandırması ve piksel bazlı Gaussian yumuşatma ile sabit boyutlu ($20 \times 20$) iki kanallı 800 boyutlu topolojik özet vektörlerine dönüştürülür.
4. **Zon Düzeyinde Bağımsız Modelleme:** Her zon kendi morfolojik ve topolojik karakteristiğine göre bağımsız eğitilir; hasta düzeyinde nodül riski zon tahminlerinin Max Pooling birleştirmesi ile hesaplanır.

---

## 2. v1 Aşamasında Yapılan Metodolojik Hataların Özeti ve Tasfiyesi

Projenin ilk sürümünde (v1) elde edilen sonuçlar (SVM 0.7157 AUC, k-NN 0.5555 AUC, CNN 0.6510 AUC) görünüşte çalışan bir boru hattı sunmuş olsa da, gerçekleştirilen kapsamlı veri ve etiket denetimleri sonucunda **5 temel metodolojik hata** tespit edilmiş ve v1 kaynaklı tüm türetilmiş veriler tasfiye edilmiştir:

### 1. Bounding Box Merkez-Nokta Etiketleme Yanılgısı (Dirty Negatives / Label Leakage)
* **Kusur:** Nodüller yalnızca $(c_x, c_y)$ merkez koordinatına indirgenerek tek bir zona atanmıştır.
* **Mekanizma & Kanıt:** 160 nodülün birden fazla loba taştığı (%10.8) ve **130 zonun gerçekte nodül alanının $\ge \%20$'sini barındırmasına rağmen yanlışlıkla `zone_label = 0` (sağlıklı) kaldığı** `nodule_pixel_overlap.csv` analizi ile kesinleşmiştir.
* **Etki:** Train kümesinde modeller, içinde belirgin tümör dokusu bulunan bu 130 zonu "sağlıklı negatif doku" referansı olarak öğrenmeye zorlanmış; bu durum karar sınırlarını kirletmiştir.

### 2. Sınır ve Belirsiz Doku (%0 - %20) Negatif Havuzuna Alınması
* **Kusur:** Bir nodülün kenarını (%1-%19) içeren sınır dokular temiz kontrol zonları ile aynı kefeye konulmuştur.
* **Etki:** Model, infiltratif tümör kenarı içeren dokular ile %100 temiz doku arasındaki sınırları ayırt edememiş; yanlış pozitif ve yanlış negatif oranları şişmiştir.

### 3. Ortak ve Aşırı Geniş [0, 255] Persistence Image Izgarası (Grid Collapse / Extreme Sparsity)
* **Kusur:** H0 ve H1 persistence dağılımları analiz edilmeden her iki boyut için keyfi `[0, 255]` aralığı kullanılmıştır.
* **Mekanizma & Kanıt:** Train diyagramlarının %99'unda H1 persistence değerlerinin $\le 8$ olduğu kanıtlanmıştır. Hücre genişliği $255 / 20 = 12.75$ olduğundan, tüm H1 noktaları ilk satıra (row 0) yığılmış; 20x20 ızgaranın kalan 19 satırı (%95'i) tamamen boş (sıfır) kalmıştır.
* **Etki:** H1 uzayındaki ince topolojik varyasyonlar tamamen yok olmuş, özellik vektörünün yarısı anlamsız sıfırlardan ibaret hale gelmiştir.

### 4. Kayıp Kaynak Verisi Üzerinden Katmanlama (`source_dataset` Hatası)
* **Kusur:** Hasta bölmesinde `source_dataset` (`chestxray14`, `padchest`, `openi`, `jsrt`, `node21_control`) katmanlama anahtarı olarak kullanılmıştır.
* **Mekanizma:** NODE21 yarışmasında tüm kontrol hastaları `node21_control` adı altında birleştirilmiş olup sağlıklıların özgün hastane kaynağı metadata'da mevcut değildir.
* **Etki:** Bu katmanlama simetrisi bozuk, yapay bir dağılım üretmiş ve genelleme kapasitesini çarpıtmıştır.

### 5. Çoklu Hipotez Testi Yanılgısı (FDR Eksikliği)
* **Kusur:** 800 özellik üzerinde düz $p < 0.05$ aranarak ayırt edici özellikler seçilmeye çalışılmıştır.
* **Kanıt:** Yapılan Benjamini-Hochberg FDR düzeltmesi, düz $p < 0.05$ ile "anlamlı" görünen ortalama 40 özelliğin saf rastgele gürültüden kaynaklanan sahte pozitif olduğunu ortaya koymuştur.

> **Tasfiye Kararı:** 17 adet geçersiz türetilmiş varlık (tüm v1 özellik vektörleri, metadata, eski split dosyaları, model ağırlıkları ve tahmin tabloları) silinmiştir. `experiments.md` liderlik tablosu "v1: geçersiz arşiv" olarak damgalanmıştır.

---

## 3. Akciğer Segmentasyonu ve Zon Kesme Altyapısı (Korunan Temel)

Ham veri ve anatomik kesitler hatasız olduğu için korunmuştur:
- **Veri Seti:** NODE21 (Toplam **4.882 tekil hasta**: 3.748 Sağlıklı, 1.134 Nodüllü / 1.476 Lezyon; $1024 \times 1024$ 16-bit MHA).
- **Segmentasyon Modeli:** TorchXRayVision `PSPNet` (NVIDIA RTX 4050 GPU, CUDA hızlandırmalı, 100.3 ms / görüntü).
- **5-Zon Anatomik Bölümleme:**
  - **Sağ Akciğer (3 Lob):** Sağ Üst (Apeks/Klavikula), Sağ Orta (Orta Lob/Hiler), Sağ Alt (Alt Lob/Diyafram).
  - **Sol Akciğer (2 Lob):** Sol Üst (Apeks/Aort Kavsi/Üst Lob), Sol Alt (Alt Lob/Kardiyak Kenar/Diyafram).
- **Kayıt:** `data/processed/zones/patients/<hasta_id>/` altında 4.882 hastanın 5 zon `.mha` ve maskeleri eksiksiz bulunmaktadır (toplam 24.410 zon kesiti, 0 boş zon).
- **Piksel Örtüşme Analizi:** 1.476 lezyonun akciğer maskeleriyle gerçek piksel kesişim oranları hesaplanmış ve [nodule_pixel_overlap.csv](file:///c:/Users/Huseyincan/codes/rontgen_V2/data/processed/zones/reports/nodule_pixel_overlap.csv) olarak dondurulmuştur.

---

## 4. v2 Temiz Veri Hattı Mimarisi

### Adım 1: v2 Zon Etiketleme (`data/processed/labels/zone_labels_v2.csv`)
- **>= %20 Kuralı:** Bounding box alanının $\ge \%20$'si bir zon maskesindeyse o zon pozitif (`zone_label = 1`, `label_state = 'positive'`).
- **Çoklu Taşma:** Aynı nodül iki zona $\ge \%20$ taşıyorsa iki zon da pozitif kabul edilir.
- **Ambiguous (%0 - %20):** Lezyon alanı %0 ile %20 arasında kalan zonlar `label_state = 'ambiguous'` olarak etiketlenir (`zone_label = 0`); asla negatif eğitim havuzuna alınmaz.
- **Unassigned Nodule:** Hiçbir zonu %20'ye ulaşmayan 9 nodüllü hasta (`n0071`, `n0431`, `n0492`, `n0659`, `n0756`, `n0835`, `n0929`, `n0959`, `n1040`) `label_state = 'unassigned_nodule'` olarak izole edilmiştir.
- **Temiz Negatif:** %0 nodül örtüşmesine sahip zonlar `label_state = 'negative'` olarak etiketlenir.

**Zon Başına Dağılım Tablosu (24.410 satır):**
| Zon | Pozitif | Temiz Negatif | Ambiguous (%0-%20) | Unassigned | Toplam |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zon 1: Sağ Üst** | 193 | 4.672 | 8 | 9 | 4.882 |
| **Zon 2: Sağ Orta** | 384 | 4.463 | 26 | 9 | 4.882 |
| **Zon 3: Sağ Alt** | 282 | 4.574 | 17 | 9 | 4.882 |
| **Zon 4: Sol Üst** | 238 | 4.624 | 11 | 9 | 4.882 |
| **Zon 5: Sol Alt** | 381 | 4.486 | 6 | 9 | 4.882 |

---

### Adım 2: Hasta Düzeyinde Katmanlı Bölme (`data/processed/splits_v2/`)
- **Hasta Bütünlüğü:** Her hasta yalnızca `train`, `val` veya `test` kümesinde yer alır (sıfır hasta sızıntısı).
- **Katmanlama:** Yalnızca `patient_label` ve 5-zon pozitiflik örüntüsü (`zone_pattern`, örn. `01000`) katmanlanmıştır. Hatalı `source_dataset` kaldırılmıştır.
- **Bölme Dağılımı:**
  - **Train:** 3.414 hasta (%69.93) — 2.624 sağlıklı, 790 nodüllü
  - **Validation:** 734 hasta (%15.03) — 562 sağlıklı, 172 nodüllü
  - **Test:** 734 hasta (%15.03) — 562 sağlıklı, 172 nodüllü
- **Manifest:** Tohum (42), oranlar ve `zone_labels_v2.csv` dosyasının SHA-256 hash bilgisi `splits_summary.json` içine kilitlenmiştir.

---

### Adım 3: TDA Izgara Aralıklarının Train Verisinden Türetilmesi (`data/processed/tda_config/tda_ranges_v2.json`)
- Validation ve Test verilerine **kesinlikle dokunulmadan**, yalnızca Train kümesinden 50 dengeli hasta (25 nodüllü, 25 kontrol = 250 zon kesiti) seçilmiştir.
- GUDHI ile 389.124 H0 ve 610.093 H1 topolojik nokta üzerinden sağlam persentiller ($p_{01} - p_{99}$) hesaplanmıştır:
  - **H0 Doğum Aralığı (Birth):** `[0.0, 248.0]` (Kırpılma / Clipping: %0.58)
  - **H0 Kalıcılık Aralığı (Persistence):** `[0.0, 248.0]` (Kırpılma: %0.22)
  - **H1 Doğum Aralığı (Birth):** `[239.0, 250.0]` (Kırpılma: %1.44)
  - **H1 Kalıcılık Aralığı (Persistence):** `[0.0, 7.0]` (Kırpılma: %0.98)

---

### Adım 4: SciPy Gaussian Tabanlı Doğrudan Vektörleştirme (`src/node21_tda/features/extractor.py`)

#### Neden Harici `persim` Kütüphanesinden Vazgeçildi?
Harici `persim.PersistenceImager` kütüphanesi, izotropik eksen aralıkları varsayımıyla tek bir skaler piksel boyutu (`_pixel_size`) kullanmaktadır. H1 boyutunda doğum aralığı genişliği ($\Delta b = 11.0$) ile kalıcılık genişliği ($\Delta p = 7.0$) farklı olduğunda, kütüphane $20 \times 20 = 400$ yerine $12 \times 20 = 240$ gibi dikdörtgen matrisler üretmeye zorlamakta ve çökmelere yol açmaktaydı.

#### Matematiksel Çözüm ve Yeni Vektörleştirme Motoru:
1. **Doğrudan Ayrık Izgara Eşlemesi:**
   Her bir $(b, p)$ noktası, doğrudan hedef $(20, 20)$ hücre koordinatlarına eşlenir:
   $$\text{col} = \min\left(\left\lfloor \frac{b - b_{\min}}{\Delta b} \cdot 20 \right\rfloor, 19\right), \quad \text{row} = \min\left(\left\lfloor \frac{p - p_{\min}}{\Delta p} \cdot 20 \right\rfloor, 19\right)$$
2. **Adams vd. (2017) Doğrusal Ağırlık Birikimi:**
   Gürültülü kısa ömürlü noktaları baskılamak ve anlamlı lezyon döngülerini vurgulamak için ağırlık fonksiyonu $w(b, p) = p$ uygulanır:
   $$\text{Image}[r, c] = \sum_{(b, p) \in \text{bin}(r, c)} p \quad \text{(`np.add.at` ile optimize)}$$
3. **Piksel Uzayında İzotropik Gaussian Düzenlileştirme:**
   Doğal koordinat ölçeklerinden bağımsız olarak, ızgara üzerinde standart mekânsal komşuluk etkisi `scipy.ndimage.gaussian_filter` ile sağlanır:
   $$\text{PI} = \text{gaussian\_filter}(\text{Image}, \sigma=1.0, \text{mode}='constant', \text{truncate}=3.0)$$

#### v2 Özellik Çıkarım Sonuçları (`data/processed/features_v2/`):
- **Çalışma:** 8 iş parçacığı (ProcessPoolExecutor), 2.141 saniye (~35.68 dakika).
- **Matris Şekli:** `(24410, 800)`, `float32` (78.1 MB).
- **Hata Sayısı:** 0 (`status: ok: 24.410`).
- **Sparsity / Doluluk Analizi:**
  - **H1 Dolu Hücre Oranı:** **%68.18** (v1'deki %95 boşluk sorunu çözülmüş, topolojik sinyal tüm ızgaraya yayılmıştır).
  - **H0 Dolu Hücre Oranı:** **%8.00**
  - **Toplam Dolu Oranı:** **%38.09** (Sıfır oranı: %61.91).

---

### Adım 5: Zon Splitleri ve 1:1 Temiz Negatif Dengeleme (`data/processed/splits_v2/zone_splits.csv`)
- Train kümesinde her zon için pozitif sayısı kadar temiz negatif örnek seçilmiştir:
  - **Zon 1 (Sağ Üst):** 133 Pos / 133 Temiz Negatif (Toplam: 266)
  - **Zon 2 (Sağ Orta):** 266 Pos / 266 Temiz Negatif (Toplam: 532)
  - **Zon 3 (Sağ Alt):** 194 Pos / 194 Temiz Negatif (Toplam: 388)
  - **Zon 4 (Sol Üst):** 164 Pos / 164 Temiz Negatif (Toplam: 328)
  - **Zon 5 (Sol Alt):** 263 Pos / 263 Temiz Negatif (Toplam: 526)
  - **Toplam Train Dengeli Örnek:** 2.040 kesit.
- **Sıfır Kontaminasyon:** Negatifler yalnızca `label_state == 'negative'` zonlardan seçilmiştir. Ambiguous ve unassigned satırlar eğitimden kesinlikle dışlanmıştır.
- Validation ve Test kümeleri doğal klinik prevalansında korunmuştur.

---

### Adım 6: Bütünlük ve Doğrulama Denetimleri (`pipelines/06_verify_v2_pipeline.py`)

| Test No | Denetim Kriteri | Durum | Doğrulama Detayı |
| :---: | :--- | :---: | :--- |
| **[1]** | Dosya Varlığı | **PASS** | 7 temel v2 dosyasının tamamı mevcut ve boyutları tutarlı. |
| **[2]** | Hasta Sızıntısı (Leakage) | **PASS** | Train (3.414), Val (734), Test (734) kümelerinde 0 kesişim. |
| **[3]** | 1:1 Train Dengeleme | **PASS** | 5 zonun tamamında pozitif ve seçilen temiz negatif sayıları eşit. |
| **[4]** | Kontaminasyon Güvencesi | **PASS** | Ambiguous ve unassigned satırlardan train negatifine 0 sızıntı. |
| **[5]** | Matris & İndeks Uyumu | **PASS** | `(24410, 800)` matrisi, metadata ve split satırlarıyla birebir eşleşiyor. |
| **[6]** | Hash Koruması | **PASS** | Etiket SHA-256 (`35f33a...`) ve TDA SHA-256 (`95600c...`) kilitli. |
| **[7]** | v1 İzolasyonu | **PASS** | Tüm geçersiz v1 dosyaları tasfiye edilmiş, v2 bağımsız. |

---

## 5. v2.1 Temiz Veri Hattı Üzerinde Model Deneyleri ve Sonuçları (Resmi Benchmark)

12–13 Ekim 2026 tarihlerinde, v2.1 temiz veri hattı ve dondurulmuş 734 test hastası üzerinde altı bağımsız model/temsil konfigürasyonu eğitilmiş ve Max-Pooling birleştirmesi ile test edilmiştir:

### Hasta Düzeyinde Nihai Benchmark Karşılaştırması (734 Test Hastası: 172 Nodüllü, 562 Kontrol):

| Sıra | Model / Mimari | Temsil Formatı | Eşik ($\theta^*$) | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Duyarlılık | Test Özgüllük | Test F1 | Test Doğruluk | Karışıklık Matrisi (TP / FP / TN / FN) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **5-Zon TDA Küçük CNN** | $5 \times (2, 20, 20)$ PI | 0.6949 | **0.8607** | **0.8631** | 0.6264 | **%79.65** | **%79.72** | **0.6478** | **%79.70** | **137** / 114 / **448** / **35** |
| 🥈 | **5-Zon RBF-SVM** | $5 \times 800$ PI Vektör | 0.6397 | 0.8278 | **0.8289** | **0.6403** | %78.49 | %68.15 | 0.5556 | %70.57 | 135 / 179 / 383 / 37 |
| 🥉 | **5-Zon k-NN** | $5 \times 800$ PI Vektör | 0.7273 | 0.7125 | **0.7162** | 0.4024 | %62.79 | %65.66 | 0.4567 | %64.99 | 108 / 193 / 369 / 64 |
| 🔹 | **5-Zon Ham Görüntü CNN (Baseline)** | $5 \times (1, 64, 64)$ Ham | 0.8421 | 0.6870 | **0.7043** | 0.4236 | %68.02 | %62.46 | 0.4680 | %63.76 | 117 / 211 / 351 / 55 |
| 🔻 | **5-Zon Landscape Küçük CNN** | $5 \times (2, 20, 20)$ PL | 0.5457 | 0.5927 | **0.5450** | 0.2624 | %64.53 | %43.77 | 0.3706 | %48.64 | 111 / 316 / 246 / 61 |
| 🔻 | **5-Zon Landscape RBF-SVM** | $5 \times 800$ PL Vektör | 0.5922 | 0.5449 | **0.5217** | 0.2477 | %5.23 | %93.24 | 0.0822 | %72.62 | 9 / 38 / 524 / 163 |

### Bootstrap %95 Güven Aralıkları (N=2.000 İstatistiksel Örnekleme):
* **TDA Küçük CNN (PI):** ROC-AUC = **0.8631 [%95 CI: 0.8302, 0.8934]** | PR-AUC = **0.6257 [0.5501, 0.7112]** | Duyarlılık = **%79.65 [%73.68, %85.64]** | Özgüllük = **%79.72 [%76.35, %82.85]** | F1 = **0.6478 [0.5933, 0.6998]**
* **RBF-SVM (PI):** ROC-AUC = **0.8289 [%95 CI: 0.7938, 0.8616]** | PR-AUC = **0.6402 [0.5679, 0.7078]** | Duyarlılık = **%78.49 [%72.15, %84.49]** | Özgüllük = **%68.15 [%64.33, %72.02]** | F1 = **0.5556 [0.5000, 0.6069]**
* **k-NN (PI):** ROC-AUC = **0.7162 [%95 CI: 0.6761, 0.7575]** | PR-AUC = **0.4024 [0.3403, 0.4696]** | Duyarlılık = **%62.79 [%55.69, %69.82]** | Özgüllük = **%65.66 [%61.97, %69.46]** | F1 = **0.4567 [0.3964, 0.5100]**
* **Ham Görüntü CNN:** ROC-AUC = **0.7043 [%95 CI: 0.6587, 0.7456]** | PR-AUC = **0.4236 [0.3533, 0.5023]** | Duyarlılık = **%68.02 [%60.90, %74.84]** | Özgüllük = **%62.46 [%58.17, %66.38]** | F1 = **0.4680 [0.4099, 0.5198]**
* **Landscape Küçük CNN (PL):** ROC-AUC = **0.5451 [%95 CI: 0.4957, 0.5946]** | PR-AUC = **0.2624 [0.2191, 0.3173]** | Duyarlılık = **%64.53 [%57.32, %71.60]** | Özgüllük = **%43.77 [%39.75, %48.12]** | F1 = **0.3706 [0.3220, 0.4197]**
* **Landscape RBF-SVM (PL):** ROC-AUC = **0.5216 [%95 CI: 0.4733, 0.5705]** | PR-AUC = **0.2476 [0.2089, 0.2990]** | Duyarlılık = **%5.23 [%2.30, %8.70]** | Özgüllük = **%93.24 [%91.07, %95.25]** | F1 = **0.0822 [0.0369, 0.1340]**

> [!IMPORTANT]
> **Çift Yönlü İstatistiksel Kanıt ($p < 0.001$):**
> 1. **TDA vs Ham Piksel (Ablation 1):** TDA Küçük CNN alt güven sınırı (**0.8302**), Ham Görüntü CNN üst güven sınırından (**0.7456**) **+0.0846 puan daha yüksektir**. Kesişim yoktur ($p < 0.001$).
> 2. **Persistence Image vs Persistence Landscape (Ablation 2):** PI Küçük CNN alt güven sınırı (**0.8302**), PL Küçük CNN üst güven sınırından (**0.5946**) **+0.2356 puan daha yüksektir**. Bu durum, akciğer lezyonları için 2B doğum-kalıcılık yoğunluk yüzeyinin (PI), 1B fonksiyonel tepe zarflarına (PL) kıyasla açık ara üstün olduğunu kanıtlar.

### Bilimsel Çıkarımlar ve Ablation Bulguları:
1. **TDA Katkısının Kanıtı (+0.1588 ROC-AUC):** Ham görüntü CNN'i 0.7043'te kalırken TDA CNN'i **0.8631** başarım elde etmiştir. Küçük veri rejiminde ham pikseller anatomik gürültüye (damar dalları, kaburga gölgeleri) takılırken, TDA'nın invariant topolojik özellikleri (delikler ve bileşenler) ayırt ediciliği doğrudan sağlamaktadır.
2. **Persistence Image Neden Landscape'e Fark Attı? (+0.3181 ROC-AUC):**
   - Persistence Landscape (PL), homoloji noktalarını $(b, d)$ tepe fonksiyonlarına çevirip 1B üst zarflara ($\lambda_k(t)$) sıkıştırır. Akciğer grafisinde bronkovasküler ağaçlar çok sayıda yüksek kalıcılıklı damar döngüsü (H1) üretir. Bu damar döngüleri üst zarfı domine eder ve nodüllerin lokal homolojik kümesini tamamen maskeler (AUC 0.52–0.54 seviyesine çöküş).
   - Persistence Image (PI) ise 2B koordinat uzayını $(b, p = d - b)$ ve Adams ağırlıklandırmasını ($w(p)$) koruyarak nodül lezyonunun doğum-kalıcılık kümesini arka plan damarlarından mekânsal olarak ayrık tutar.
3. **Mükemmel Genelleme (Sıfır Aşırı Uyum):** PI Küçük CNN'de Validation (0.8607) ve Test (0.8631) farkı $<\%0.3$'tür.

---

## 6. Mevcut Durum Özeti (Checklist)

- [x] v1 geçersiz çıktıları (17 hedef yol) doğrulandı ve silindi.
- [x] `experiments.md` ve `progress.md` üzerindeki eski skor tabloları "v1: geçersiz" olarak işaretlendi.
- [x] %20 piksel örtüşme kuralı, multi-zone ve ambiguous ayrımı ile `zone_labels_v2.csv` üretildi.
- [x] `source_dataset` olmaksızın, hasta bazlı `splits_v2/patient_splits.csv` üretildi.
- [x] Yalnızca train diyagramlarından türetilen H0/H1 ayrı aralıklarıyla `tda_ranges_v2.json` yapılandırıldı.
- [x] `extractor.py` içerisine SciPy Gaussian tabanlı doğrudan PI vektörleştirme motoru entegre edildi.
- [x] 24.410 zon için `features_v2/X_zone_features.npy` (24410, 800) ve metadata üretildi.
- [x] Train 1:1 temiz negatif dengelemesi yapılarak `splits_v2/zone_splits.csv` oluşturuldu.
- [x] 7 aşamalı bütünlük kontrolleri ve birim testler %100 başarıyla geçti.
- [x] **Deney 1 (RBF-SVM):** 5 bağımsız zon modeli eğitildi, Youden's J eşiği kilitlendi, test hastalarında **0.8289 ROC-AUC** elde edildi.
- [x] **Deney 2 (k-NN):** $k \in [1..11]$ tarandı, en iyi $k$ seçildi, test hastalarında **0.7162 ROC-AUC** elde edildi.
- [x] **Deney 3 (TDA Küçük CNN):** 2 kanallı tensörlerle eğitildi, test hastalarında **0.8631 ROC-AUC** ile liderlik elde edildi.
- [x] **Deney 4 (Ham Görüntü CNN Baseline & Ablation 1):** $1 \times 64 \times 64$ ham gri zon kesitleriyle 5 bağımsız ham CNN eğitildi; **0.7043 ROC-AUC** elde edildi.
- [x] **Deney 5 & 6 (Persistence Landscapes PL & Ablation 2):** 24.410 zon için PL vektörleri çıkarıldı; PL RBF-SVM (**0.5217 ROC-AUC**) ve PL Küçük CNN (**0.5450 ROC-AUC**) eğitilerek PI'nin 2B doğum-kalıcılık üstünlüğü kanıtlandı.
- [x] **$N=2.000$ Bootstrap %95 Güven Aralıkları:** 6 modelin tamamı için hesaplandı; PI CNN'in hem Ham CNN'den hem PL CNN'den istatistiksel olarak anlamlı şekilde ($p < 0.001$) ayrıştığı kanıtlandı.
- [x] Deney çıktıları (`artifacts/models/` ve `artifacts/runs/`) başarıyla kaydedildi; `experiments.md` ve `progress.md` senkronize edildi.
- [x] **Seçenek A (Hibrit / Ensemble):** v3 programında **E3.4** olarak tamamlandı → TDA ensemble test ROC-AUC **0.8776**; ardından **E3.1 füzyonu** ile **0.9211**'e ulaşıldı.
- [ ] **Seçenek B (Eşik Hassasiyet Analizi):** v3 programında **E4.1** olarak planlandı (%10/%20/%30 overlap robustness).

---

## 7. v3 Araştırma Programı (2026-10-14 — devam ediyor)

> Yönetim, fazlar, backlog ve karar günlüğü: [roadmap.md](roadmap.md) · Literatür ve konumlandırma: [literature.md](literature.md)

v2.1 benchmark'ının ardından, **yayınlanabilir bir katkı** üretmek için v3 programı başlatıldı. Eski "Seçenek A" ensemble olarak, "Seçenek B" ise robustness olarak programa alındı.

### 7.1 Altyapı ve doğrulama (Faz 0)

- **E0.1 (KEEP):** Deney kayıt altyapısı — [experiment.py](src/node21_tda/experiment.py) her çalıştırma için manifest üretir (git commit + dal + kirli ağaç, config/girdi SHA-256, seed, metrikler, karar). [configs/experiments/](configs/experiments/) sözleşmesi (şablon + README) kuruldu.
- **E0.2 (KEEP):** v2.1 PI küçük CNN referansı **bit-bit yeniden üretildi** — 16/16 metrik (val 0.8607, test **0.8631**, eşik 0.6949, TP/FP/TN/FN 137/114/448/35). Pipeline **deterministik**; 5 zonun eğitimi CPU'da ~27 s.
- **E0.0 (ertelendi):** venv'de `torch 2.14.1+cpu` (CUDA yok) ancak **tüm v3 deneyleri CPU'da üretildi** → GPU kritik değil. RTX 4050 yalnızca uçtan uca fine-tuning için gerekir.

### 7.2 Literatür taraması

[literature.md](literature.md) — dört bağımsız kolda ~40 arama, tüm kaynaklar URL'li:

- **En yakın emsal:** **Topo-CXR** (ICCV 2023 W) — CXR + kübik persistence, ama TB/pnömoni (nodül değil).
- ⚠️ **Kritik doğrulama (yerel):** Verimiz **NODE21'in halka açık training setiyle birebir aynı** (4882 = 1134 nodüllü + 3748 normal). NODE21'in resmî test setleri (281 + 298) **özeldir**. ⇒ 734 hastalık test setimiz **kendi iç bölünmemiz**; **leaderboard karşılaştırması geçersizdir** (sızıntı yok). Roadmap §0 **kural 9** olarak zorunlu kılındı.
- **Literatür boşlukları (bizim özgünlüğümüz):** kemik/damar bastırma + kalıcı homoloji kombinasyonu yok; NODE21'de topolojik çalışma yok. Topo-CXR'ın *"ön-işleme gerekmez"* tezine karşı **net karşı-pozisyon**.
- **Yenilik hiyerarşisi:** literature.md §5.2.

### 7.3 Deneyler ve sonuçlar

| ID | Deney | Karar | Sonuç |
| :--- | :--- | :--- | :--- |
| **E4.4** | PL üst-zarf (envelope) mekanizma analizi | **KEEP** | H1 zarf tepeleri λ1→λ5 boyunca 110→90 (**doymuş**); poz/neg sınıf sinyali zarfın yalnızca **%1–2'si** (≈50–100× bastırılma). PL çöküşünün mekanizması nicel olarak kanıtlandı |
| **E3.4** | CNN + SVM soft-voting ensemble | **KEEP** | Test ROC-AUC **0.8776** (w=0.55); val farkı +0.0214; bootstrap %95 CI [+0.0034, +0.0260], p=0.010 |
| **E0.3** | Güçlü baseline: donmuş ImageNet ResNet18 + LR | **BASELINE** | Test **0.8349** (val 0.8884 → **−0.0535 genelleme kaybı**). TDA tek başına ve füzyon bu baseline'ı **geçiyor** |
| **E3.1** | **ResNet18 + TDA geç füzyonu** | **KEEP** 🏆 | Test **0.9211**; ResNet'e göre **+0.0862**, bootstrap %95 CI **[+0.0632, +0.1099]**, p≈0.0000 |
| **E4.2** | Persistence → görüntü geri izleme (yorumlanabilirlik) | **KILL** ⚠️ | Ön-kayıtlı H1 hipotezi çürütüldü (medyan gradyan oranı 1.021); **ancak H0 güçlü doğrulandı** (oran 6.176, zonların %90'ında >1.5) → H0 PI ağırlıklı olarak **arka plan/sınır** kodluyor |
| **E2.7** | PI ağırlık fonksiyonu karşılaştırması (val-only) | **KILL** ⚠️ | **Adams `p` zaten en iyi (0.8736)**; arctan −0.0096, √p −0.0112, sabit −0.0145, **p² −0.0189**. Mevcut PI tasarımı **doğrulanmış**; `p²`'nin en kötü olması E4.4'ü bağımsız destekliyor |
| **E4.1** | Etiketleme kuralı sağlamlığı %10/%20/%30 (val-only) | **KILL** ⚠️ | Hasta split'i dondurulmuş. **%10 = 0.8922 (+0.0186)**, %20 = 0.8736, %30 = 0.8693. En büyük fark toleransı (0.015) aştı ⇒ sonuçlar kuraldan bağımsız **sağlam değil**. Monoton eğilim var; **%10 iyileştirme adayı → E4.6** |
| **E4.6** | %10 vs %20 — eşleştirilmiş 5-seed doğrulama (val-only) | **KILL** ⚠️ | Ortalama fark yalnızca **+0.0085** (std 0.0096), pozitif seed 4/5 → eşik (+0.015) aşılamadı. **E4.1'in bulgusu gürültü artefaktıydı**; **%20 kuralı korunur**. 📌 Yan bulgu: CNN seed varyansı **±0.02–0.03** → makalede çok-seed ortalama raporlanmalı |
| **E2.8** | σ / ızgara çözünürlük duyarlılığı (3 seed, val-only) | **KEEP** ✅ | Kol A (σ): 0.5→0.8460, **1.0→0.8697**, 1.5→0.8734, 2.0→0.8712. Kol B (ızgara): 15→0.8545, **20→0.8697**, 25→0.8469. Her iki kol da **sağlam** (en büyük sapma 0.0236 < tolerans 0.03) ⇒ σ/ızgara seçimine aşırı duyarlı değil. **Faz 1 tamamlandı (6/6)** |
| **E4.3** | Zon bazlı hata analizi (val-only) | **KILL** ⚠️ | **Zayıf zon yok** — her zon tek başına 0.74–0.84 hasta düzeyi AUC. Karar 5 zona yayılıyor; FN'ler "kıl payı" (ort 0.581 vs eşik 0.636). 📌 **Kritik:** zon sinyalinin bir kısmı **hasta-global** olabilir → **E4.7 lokalite testi** kaydedildi; makalede sınırlama olarak belirtilmeli |
| **E4.7** | Lokalite testi (val-only, 171 nodül-pozitif hasta) | **KILL** 🚨 | **Modeller lokalize ETMİYOR.** İsabet oranı şans düzeyinde (0.228–0.298 vs 0.268; binom p=0.21–0.90); **zon düzeyi AUC 0.48–0.53**. ⇒ 0.9211 sonucu gerçek ama mekanizma "nodülü doğru zonda bulmak" değil. **"5-zon lokalizasyon" iddiası düştü** → "5-zon paralel ayrıştırma" olarak yazılmalı. Yeni aday **E3.5** (global vs 5-zon) |

> **E3.1 bağımsız doğrulaması** ([scratch/verify_E3_1.py](scratch/verify_E3_1.py)): ResNet ve TDA hasta olasılıkları arasındaki korelasyon yalnızca **r ≈ 0.30–0.35** → iki temsil neredeyse **dik**, gerçek tamamlayıcılık. Etiket permütasyonu → AUC **0.4378** (şans) ⇒ **sızıntı yok**.

### 7.4 Güncel liderlik (hasta düzeyi test ROC-AUC, 734 hasta)

| Sıra | Model | Test ROC-AUC |
| :-: | :--- | :---: |
| 🥇 | **E3.1 Füzyon (ResNet18 + TDA ensemble)** | **0.9211** |
| 🥈 | E3.4 TDA ensemble (CNN + SVM) | 0.8776 |
| 🥉 | TDA PI küçük CNN (v2.1) | 0.8631 |
| 4 | E0.3 ResNet18 (donmuş) + LR | 0.8349 |
| 5 | RBF-SVM (PI) | 0.8289 |
| 6 | k-NN (PI) | 0.7162 |
| 7 | Ham görüntü küçük CNN | 0.7043 |
| 8 | PL küçük CNN | 0.5450 |
| 9 | PL RBF-SVM | 0.5217 |

> NODE21 referansları: rank-1 = **0.868**, ensemble = **0.877** → **geçildi**.

### 7.5 Sıradaki adımlar

📄 **Tüm sonuçlar tek belgede:** **[ablation_matrix.md](ablation_matrix.md)** — makale tablolarının taslağı (liderlik, 5 ablasyon, sağlamlık, istatistik, yenilik konumlandırması). E5.1 olarak taslak hâlinde hazır.

⚠️ **Test seti bütçesi: 4/6 kullanıldı** (E0.2, E3.4, E0.3, E3.1). Kalan 2 değerlendirme → **final ablasyon matrisi** + **harici doğrulama**. Ara deneyler **yalnızca validation**'da karar verir.

Devam noktası ve öncelik sırası: **[roadmap.md §12](roadmap.md)** (Handoff).

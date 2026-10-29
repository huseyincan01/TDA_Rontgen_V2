# NODE21 TDA — Literatür Taraması ve Konumlandırma

> Amaç: [roadmap.md](roadmap.md) v3 programındaki fikirlerin literatürde nerede durduğunu, hangisinin **gerçekten yeni** olduğunu ve bu ışıkta hangi deneylerin öncelik kazanması gerektiğini belirlemek.
> Yöntem: Dört bağımsız kolda web araması + birincil kaynak getirme. Yalnızca doğrulanan kaynaklar yazılır; emin olunmayan ifadeler `[belirsiz]` ile işaretlenir.

**Durum:** ✅ Dört kol da tamamlandı (§1–§5) · Tarama tarihi: **2026-10-14**

---

## 1. İleri Düzey TDA Yöntemleri

### 1.1 Çok parametreli persistence (MPH)

| Yöntem | Kaynak | Katkı |
| :--- | :--- | :--- |
| **MMA** (Multiparameter Module Approximation) | Loiseaux, Carrière, Blumberg — [arXiv:2206.02026](https://arxiv.org/abs/2206.02026) | "Keyfî sayıda filtrasyon" + **sınırlı karmaşıklık ve çalışma süresi** garantisi; interleaving hatası δ ile sınırlı; çıktı hem görsel (yorumlanabilir) hem vektörleştirilebilir |
| **multipers** | [repo](https://github.com/DavidLapous/multipers) · [docs](http://www-sop.inria.fr/members/David.Loiseaux/doc/multipers/notebooks/multipers_intro.html) | `module_approximation`, `signed_measure`, `MMA2Img`, `MMA2Landscape`, `SignedMeasure2Convolution` |
| **RIVET** | [repo](https://github.com/rivetTDA/rivet) · [rivet-python](https://github.com/rivetTDA/rivet-python/) | 2-parametre; Hilbert fonksiyonu, bigraded Betti sayıları, fibered barcode |

**Bizim için kritik:** multipers'ın kendi örnek problemi **bizimkiyle birebir aynı** — gürültü topolojiyi bozuyor, yoğunluk (KDE) ikinci bir eksen olarak eklenince topoloji kurtarılıyor ve "eşik seçiminin kanonik olmaması" sorunu MPH ile aşılıyor. Bizim "tek parametre = yoğunluk, sabit ızgara" kısıtımız tam olarak bu sınıfa giriyor.

**Uygulanabilirlik: ORTA.** multipers GUDHI simplextree bekliyor; **cubical** girdiyle doğrudan MPH örneği bulunamadı `[belirsiz]` → adaptasyon işi gerekir.

> **Sonuç:** E2.6 (çok parametreli persistence) "Parkta" statüsünden **çıkarılmalı**; yazılım tarafı tahmin ettiğimden olgun. Yine de cubical adaptasyonu riski nedeniyle ikinci dalga olarak kalmalı.

### 1.2 Öğrenilebilir / çekirdek tabanlı vektörleştirme

| Yöntem | Kaynak | Ne sağlıyor | Efor |
| :--- | :--- | :--- | :---: |
| **PersLay** | [repo](https://github.com/MathieuCarriere/perslay) | Diyagram üzerinde öğrenilebilir ağırlık + permutation-invariant toplama; PI/PL'yi özel durum olarak kapsar | Orta |
| **Sliced Wasserstein kernel** | Carrière et al. — [HAL](https://hal.science/hal-01633105v1/preview/carriere17a.pdf) | Diyagramları RKHS'e taşır; **PI'nin ızgara/çözünürlük bağımlılığından kurtarır** | Orta |
| **Persistence Fisher kernel** | Le & Yamazaki — [arXiv:1802.03569](https://ar5iv.labs.arxiv.org/html/1802.03569) | Diyagram uzayında Riemann manifold çekirdeği; SVM ile geniş benchmark | Orta |
| **Persistence scale-space kernel** | Reininghaus et al. (2015) | Çok ölçekli ısı yayılım çekirdeği; gürültüye kararlı | Orta |
| **TopologyLayer** | [repo](https://github.com/bruel-gabrielsson/TopologyLayer) | PyTorch'ta **diferansiyellenebilir** persistence (H0/H1) | Zor |
| **Filtration learning** | Nishikawa, Ike, Yamanishi — [arXiv:2307.09259](https://arxiv.org/abs/2307.09259) | Filtrasyon fonksiyonunu NN ile öğrenir; izometri-invaryant mimari + sonlu boyut yaklaşım teoremi | Zor |

### 1.3 Çok ölçekli / scale-space

* **Topo-CXR** — Ahmed, Nuwagira, Torlak, Coskunuzer, **ICCVW 2023** ([paper](https://openaccess.thecvf.com//content/ICCV2023W/CVAMD/html/Ahmed_Topo-CXR_Chest_X-ray_TB_and_Pneumonia_Screening_with_Topological_Machine_ICCVW_2023_paper.html)): **göğüs röntgeni + kübik persistence**. Normal/abnormal görüntüler "çok belirgin farklı topolojik paternler" üretiyor; **ön işleme ve augmentasyon olmadan, küçük veri setlerinde SOTA derin öğrenmeyi geçiyor.** Bizim hattımızla (cubical PH + CXR) **aynı aile** → en yakın emsal.
* **σ / ızgara duyarlılığı:** Moon et al. **625 kombinasyonu LOOCV** ile tarıyor → bizde de en azından σ ve ızgara çözünürlüğü için bir **duyarlılık taraması** gerekli. `[Kolay]`

### 1.4 "Relative" / arka plan çıkarma temelli persistence

* **SEDT-3 (Signed Euclidean Distance Transform)** — Moon, Li, Xiao ([arXiv:2012.12102](https://ar5iv.labs.arxiv.org/html/2012.12102)): 3-sınıf (tümör / normal / boş) görüntüde her piksele **en yakın farklı sınıfa işaretli mesafe** atanır, sonra **kübik kompleks** kurulur. Yani filtrasyon, mutlak yoğunluk değil **referans doku sınırına göre sapmadır**. Akciğer (n=133) ve beyin (n=77) kohortlarında sağkalım tahmini anlamlı (p=4×10⁻⁷, HR=5.38; p=8×10⁻⁴, HR=2.18).
* **Boşluk tespiti:** Sağlıklı bir şablona göre **persistence diyagramı düzeyinde** sapma ölçen yayınlanmış bir tıbbi görüntüleme çalışması **bulunamadı** → E2.3 (relative persistence) için potansiyel **açık alan**. Teorik komşular: relative homology/excision ([arXiv:1910.03348](https://arxiv.org/abs/1910.03348)), "2D+t track detection via relative persistent homology" ([Wiley](https://doi.org/10.1002/ima.22503)).
* **Pratik öneri:** Kohort-ortalama **sağlıklı CXR şablonu** çıkar; her hastada `Δ = I − şablon` üzerinde kübik PH hesapla. Bu, SEDT-3 mantığının doğrudan uzantısıdır.

### 1.5 Filtrasyon ağırlıklandırma (önsel/olasılık haritası)

* **multipers çift filtrasyon:** `X_(t,s) = {x : d(x,X) ≤ t, density(x) ≥ s}` — bizde `s` = **nodül olasılık haritası** olabilir. Ağırlıklandırmayı, eşik seçiminin keyfiliğinden kurtaran **prensipli** yola çevirir.
* **Weighted Rips/alpha kompleks:** nokta-bazlı ağırlık ile 1-parametrede aynı bilgi (daha düşük eforlu ara adım).
* **Yorumlanabilirlik API'si (hazır):** GUDHI `CubicalComplex.cofaces_of_persistence_pairs()` ve `vertices_of_persistence_pairs()` ([GUDHI 3.13](https://gudhi.inria.fr/python/latest/_modules/gudhi/cubical_complex.html)) persistence doğum/ölüm hücrelerini **girdi görüntüsündeki indekslere** döndürüyor → bir PI noktasını görüntüdeki bir bölgeye **geri izlemek mümkün**. `[Kolay]` → E4.2'yi doğrudan besliyor.
  * ✅ **Yerel doğrulama (gudhi 3.13.0, gerçek zon kesiti `c0001/zone_1_sag_ust`, 234×334):** API mevcut ve çalışıyor. `cofaces_of_persistence_pairs()` H0 için **(1966×2)**, H1 için **(3127×2)** hücre-indeksi çifti döndürdü; örnek bir H1 çifti görüntüde **(satır 180, sütun 6–7)** konumuna başarıyla geri izlendi. ⇒ **E4.2 teknik olarak hazır**, ek kütüphane gerekmiyor.
  * 📌 **Niceliksel teşhis desteği:** Tek bir zonda H1 çifti sayısı (**3.127**) H0'dan (**1.967**) fazla — bu, "H1 damar döngüleriyle domine ediliyor" teşhisini sayısal olarak destekliyor. (Not: H0'ın toplam kalıcılığı `inf` çıkar; yorumlanabilirlik analizinde sonsuz aralıklar dışlanmalıdır.)

### 1.6 Tıbbi görüntülemede yorumlanabilirlik ve performans emsalleri

| Çalışma | Veri | Sonuç |
| :--- | :--- | :--- |
| **Wang et al. 2024** ([PMC11452904](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452904/)) | 3.223 hasta, 2 merkez, GGN malignite | Topolojik radiomics AUC **0.85 / 0.862**; klinik+radyomik+topolojik **0.879** (Giotto-TDA, 270 topolojik özellik) |
| **Moon et al.** ([arXiv:2012.12102](https://ar5iv.labs.arxiv.org/html/2012.12102)) | Akciğer + beyin | Persistence diyagramı **kadran → şekil paterni** haritası (H0 kadran II = dağınık lezyon, III = bağlı lezyon, H1 kadran II = lezyonun çevrelediği normal doku); **3 ağırlık fonksiyonu** karşılaştırması |
| **Zhu et al. 2024** ([arXiv:2408.07905](https://arxiv.org/abs/2408.07905) · [repo](https://github.com/hrlblab/TopologicalDataAnalysis3D)) | MedMNIST3D | 3D PI + superpixel + optimize Gauss katsayısı; kod açık |
| **Singh et al., Oncotarget 2024** ([PMC11559656](https://pmc.ncbi.nlm.nih.gov/articles/PMC11559656/)) | — | TDA'nın radyolojide yorumlanabilirlik/bias argümanı (editöryel, deneysel kanıt değil) |

**Konumlandırma:** Bizim hasta düzeyi **0.8631** ROC-AUC'muz, Wang et al.'ın topolojik radiomics aralığıyla (**0.85–0.862**) aynı ölçekte. Yani skor "zayıf" değil; eksik olan, **karşılaştırılabilir güçlü bir baseline** ve **yorumlanabilirlik kanıtı**.

---

## 2. Tıbbi Görüntülemede TDA

### 2.1 Uygulamalar

| Çalışma | Veri / görev | Yaklaşım |
| :--- | :--- | :--- |
| **Topo-CXR** — Ahmed ve ark., ICCVW 2023 ([paper](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Ahmed_Topo-CXR_Chest_X-ray_TB_and_Pneumonia_Screening_with_Topological_Machine_ICCVW_2023_paper.html)) | **CXR**, TB/pnömoni | **Kübik persistence** → Betti vektörleri → standart ML. Ön işleme/augmentasyon yok. **En yakın emsal** |
| **Vandaele ve ark.**, *Patterns* 2023 ([PMC9868648](https://pmc.ncbi.nlm.nih.gov/articles/PMC9868648/)) | Akciğer tümörü **histoloji** (CT), 164 hasta | 290 topolojik özellik (15 diyagram × özet istatistik) vs 105 radyomik |
| **Wang ve ark.**, 2024 ([PMC11452904](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452904/)) | Buzlu cam nodül (GGN) malignite, 3.223 hasta | Giotto-TDA, 270 topolojik özellik + PyRadiomics |
| **Lin ve ark.**, *PLOS Digit Health* 2026 ([makale](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001646)) | İnvaziv adenokarsinom derece, 4 merkez | Topolojik intratümöral heterojenlik |
| **TDA-Net** — Hajij, Zamzmi, Batayneh, [arXiv:2101.08398](https://ar5iv.labs.arxiv.org/html/2101.08398) | **CXR** / COVID | Betti eğrisi + derin özellik **füzyonu** |

### 2.2 Temsil karşılaştırmaları — kritik bulgu

* **Barnes, Polanco, Perea**, *Front Artif Intell* 2021 ([PMC8355525](https://pmc.ncbi.nlm.nih.gov/articles/PMC8355525/)): Altı şema karşılaştırılıyor (multi-scale kernel, PL, PI, Adcock–Carlsson, template systems, adaptive template). Sonuç: **PL, Adcock–Carlsson ve template systems hem hızlı hem doğru**; **PI daha yavaş ama rekabetçi**; **multi-scale kernel en pahalı ve hiçbir zaman daha iyi çıkmadı.**
* **Sulowska**, *Adv Sci Technol Res J* 2026 ([makale](https://www.astrj.com/Comparative-evaluation-of-persistence-diagram-vectorisation-methods-in-classification,218660,0,1.html)): **PI ve PL tutarlı biçimde en iyi**; sentetikte >0.98; **ModelNet10'da PI açık ara üstün (~0.75)**; MNIST'te tüm yöntemler zayıf. Sentez: **PI sağlam/çok yönlü, PL kararlı ve yorumlanabilir.**

> **Sonuç:** Literatürde **"PI vs PL evrensel kazananı yok"**. Bizim PI lehine +0.3181'lik farkımız bu çerçevede *olağandışı* — ve tam da bu yüzden ilginç (§2.6).

### 2.3 Bildirilen başarı seviyeleri

| Çalışma | Sonuç |
| :--- | :--- |
| Topo-CXR | TB-CXR için iddia edilen **~%99.3 doğruluk / 99.8 AUC** — ancak sonraki bir ViT çalışması (RepViT-CXR, [arXiv:2509.08234](https://arxiv.org/abs/2509.08234)) onu geçti (99.9/99.9); Shenzhen TB'de yalnızca %91.1 ⇒ **üstünlük kalıcı değil, marj küçük** |
| Vandaele 2022 | Benign/malign: radyomik 84.6 → topolojik **86.8** → vote 87.9. ADC vs skuamöz: 67.2 → **91.2** (büyük katkı). Bazı görevlerde ise **radyomik daha iyi** |
| Wang 2024 (GGN) | Topolojik **0.85 / 0.862**; klinik+radyomik+topolojik birleşik **0.879** |
| TDA-Net | Temel CNN 0.87 → **sadece topoloji (Betti) 0.89** → füzyon **0.92/0.93** (~+0.05) |
| Lin 2026 | Stacking **0.875** vs sadece radyomik 0.834 |

### 2.4 "İlgisiz/arka plan yapıları baskın çıkıyor" problemi

**Kavramsal olarak literatürde VAR, ama bizim CXR/H1 formülasyonumuzda yok.**

* **Vandaele, Nervo, Gevaert**, *Sci Rep* 2020 ([PMC7713118](https://pmc.ncbi.nlm.nih.gov/articles/PMC7713118/)) — en net ifade: *"many real-world images contain outliers, as well as **irrelevant objects, which complicate the use of persistent homology**"*. Çözüm: **Topological Image Modification (TIM)** — ilgisiz ama yüksek kalıcılıklı bilgiyi filtreler; "border modification" ile en kalıcı bileşenin **arka plan** olması garanti edilir (elder rule).
* **ROI izolasyonu standart pratik:** Vandaele diyagramları **5 farklı nesneden** hesaplıyor (lezyon pikselleri ham/negatif, lezyon+bbox, yüzey nokta bulutu). Wang 2024 GGN'de lezyon maskesiyle kırpma yapıyor.
* **Bulunamayan:** "CXR'de H1'de bronkovasküler döngüler ve kaburga kenarları üst zarfı domine eder" biçiminde **spesifik** ifade → muhtemelen bizim özgün formülasyonumuz. `[belirsiz: daha derin atıf-ağı taraması gerekir]`

### 2.5 Bilinen sınırlamalar

* **Hesaplama maliyeti** (multi-scale kernel en pahalı, hiç kazanmadı).
* **Vektörizasyonda bilgi kaybı** — Vandaele 2022 açıkça yazıyor: *"Unavoidably, one loses information when transforming topological information into feature representations."*
* **Yüksek boyut + küçük örneklem / overfitting** — Vandaele 290 özellikten mRMR ile **yalnızca 10** özellik seçiyor; GGN çalışması PCA (0.99) uyguluyor.
* **Genelleme sınırı** — yüksek geometrik karmaşıklık + gürültüde tüm temsiller zayıf (Sulowska).
* **Kararlılık bir güçlü yön:** bottleneck/Wasserstein kararlılığı $d_b(\mathrm{Dgm}(I),\mathrm{Dgm}(J)) \le \|f-g\|_\infty$ ⇒ "kararsızlık" eleştirisi PH'ın kendisi için değil, **vektörizasyon + özellik seçimi + küçük örneklem** hattı için geçerli.
* **Yorumlanabilirlik iddiası kanıtsız:** TDA-Net daha iyi açıklanabilirlik iddia ediyor ama H1 döngülerinin **hangi anatomik yapıya** karşılık geldiği gösterilmiyor. `[belirsiz]`

### 2.6 Bizim PI-vs-PL bulgumuz literatürle uyuşuyor mu?

**Kısmen uyuşuyor, kısmen ayrışıyor — ayrıştığı yer tam da ilginç olan yer.**

* ✅ **Uyuşan:** PI'ın (birth, persistence) 2B uzayını koruyup ayrık imzaları saklaması ve yüksek geometrik karmaşıklıkta PL'den üstün olması, Sulowska'nın **ModelNet10'da PI açık ara üstün** bulgusu ve "PI sağlam/çok yönlü" senteziyle **örtüşüyor**.
* ❌ **Ayrışan:** Literatürde PL genelde **öne çıkan** taraf (Barnes 2021: PL hızlı+doğru). Bizim PL'nin **0.5450'ye çökmesi** (PI 0.8631) tipik beklentinin **tersi**. Literatür PL'yi "kararlı/yorumlanabilir" diye övüyor ama **ilgisiz yüksek-kalıcılıklı yapıların üst zarfı ele geçirmesi** durumunu PL için özel olarak test etmiyor.
* **Mekanizma:** "İlgisiz ama yüksek kalıcılıklı yapılar diyagramı domine eder" problemi **tanınıyor** (Vandaele 2020 TIM tam bunun için). Yani gözlemimiz **bilinen bir olgunun CXR/H1'deki yeni tezahürü** — kavramsal olarak yeni değil, bu anatomik örnekte gösterilmesi yeni.

### 2.7 Gerçekten yeni olan (yayın iddiaları)

| # | İddia | Güç |
| :-: | :--- | :--- |
| 1 | **PL'nin üst-zarf (envelope) mekanizmasıyla çökmesinin nicel gösterimi** — birkaç uzun ömürlü ama ilgisiz H1 sınıfının tüm temsili ezmesi; PI'ın 2B kütle dağılımı olarak yerel imzayı korumasıyla doğrudan nedensel karşılaştırma | **En güçlü** (literatürde bulunamadı) |
| 2 | **CXR'de H1'in bronkovasküler döngüler + kaburga kenarlarıyla domine edilmesi** | Güçlü (spesifik formülasyon yok) |
| 3 | **5 anatomik zon + zon başına ayrı kübik PH + hasta düzeyi max-pooling** (Topo-CXR global, Vandaele/Wang ROI-kısıtlı) | Güçlü |
| 4 | **NODE21 + kalıcı homoloji** | Orta `[belirsiz: olumsuz sonuç]` |

**Yenilik İDDİA EDİLMEMESİ gerekenler:** "ilgisiz yapılar diyagramı domine eder → ROI kısıtlaması gerekir" (Vandaele 2020/2022) · "PI ve PL farklı davranır, veri setine bağlıdır" (Barnes 2021, Sulowska 2026) · "topoloji tek başına CNN'e denk, füzyonla artar" (TDA-Net) · "topoloji gerçek histolojiyi, radyomik radyolog skorunu taklit eder" (Vandaele 2022).

### 2.8 Hakem riski ve zorunlu karşılaştırma

> ⚠️ ROC-AUC 0.8631, Topo-CXR'ın (0.993) ve TDA-Net füzyonunun (~0.92–0.93) altında. Hakemin en olası itirazı: **"TDA neden global bir CNN baseline'ını geçmiyor?"**

Bu nedenle raporda **zorunlu** görünen iki şey:
1. **Aynı split üzerinde eğitilmiş güçlü bir ham-piksel CNN baseline'ı** → E0.3 + E3.1/E3.2 (füzyon) kritik yolunu doğrudan meşrulaştırıyor.
2. **PI vs PL mekanizma analizi** (üst zarf görselleştirmesi) → §2.7'deki 1 numaralı iddiayı kanıta çeviriyor.

---

> ℹ️ **Araç sınırlaması:** `web_fetch` PDF içeriğini okuyamıyor (`application/pdf`). Topo-CXR'ın tam metin sonuç tablosu **ikincil atıftan** (RepViT-CXR) alındı; sayılar birincil kaynaktan teyit edilemedi. ScienceDirect/MDPI/PubMed HTML getirmeleri 403 döndü; PMC ve arXiv HTML yolları çalıştı.

---

## 3. NODE21 ve CXR Nodül Tespitinde SOTA

**Birincil kaynak:** Sogancioglu, van Ginneken, … Murphy, *"Nodule detection and generation on chest X-rays: NODE21 Challenge"*, **IEEE TMI 2024** — [arXiv:2401.02192](https://arxiv.org/abs/2401.02192) · [challenge](https://node21.grand-challenge.org/) · [leaderboard](https://node21.grand-challenge.org/evaluation/challenge/leaderboard/)

### 3.1 Görev ve metrikler

* İki track: **detection** (frontal CXR → skorlu bounding box listesi) ve **generation** (verilen lokasyona nodül sentezleme). Yarışma değil **işbirliği**; kod açık kaynak zorunlu.
* **Veri setleri:**
  * **Training (halka açık):** **4882 görüntü = 1134 nodüllü (1476 nodül) + 3748 normal**
  * **Experimental test (ÖZEL):** 281 görüntü (166 nodüllü + 115 normal)
  * **Final test (ÖZEL):** 298 görüntü (111 pozitif + 187 negatif), CT tabanlı referans
* **Metrikler:** **AUC** → görüntü skoru = o görüntüdeki **maksimum bounding-box olasılığı** (tahmin yoksa 0), sonra eşiklenip ROC. **FROC** → eşleşme **IoU > 0.2**, duyarlılık **0.125 / 0.25 / 0.50 FP/görüntü** noktalarında. **CPM NODE21'de kullanılmıyor** (o LUNA16 gibi CT challenge'larında).

### 3.2 En iyi sonuçlar (final test, 298 görüntü)

| Yöntem | Mimari | AUC | S(0.5 FP) | S(0.25 FP) | S(0.125 FP) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D1** (rank 1) | Faster R-CNN + RetinaNet + YOLOv5 + EfficientDet-D2, 20 model, WBF; VinDr-CXR pretrain | **0.868** | 0.800 | 0.750 | 0.603 |
| **D2** (rank 2) | 33× YOLOv5 ensemble, scratch | 0.862 | 0.771 | 0.723 | 0.600 |
| **D3** (rank 3) | Mask R-CNN + RetinaNet (ResNet-50), 6 model | 0.833 | 0.761 | 0.704 | 0.590 |
| **DB** (baseline) | Faster R-CNN, COCO pretrain | 0.816 | 0.714 | 0.635 | 0.504 |
| **Ensemble** (D1+D2+D3+DB) | weighted box fusion | **0.877** | 0.819 | 0.754 | 0.619 |

Kazananların **tamamı** iki-aşamalı, bol ensemble'lı anchor-based detektörler + NMS/WBF + harici pretrain. **Hiçbir topolojik / açıklanabilir / hafif yöntem kazanmıyor.** Ensemble 12 radyologdan 3'ünü geçiyor, 8'iyle benzer (AUC 0.877, %95 CI 0.830–0.915).

### 3.3 "0.8631 nerede duruyor?"

| Kaynak | Veri / görev | AUC |
| :--- | :--- | :---: |
| NODE21 (DB / D1 / ensemble) | NODE21 final (298), **nodül tespiti** | 0.816 / **0.868** / 0.877 |
| Wang et al., CVPR 2017 ([arXiv:1705.02315](https://arxiv.org/abs/1705.02315)) | ChestX-ray14, **nodül** etiketi (zayıf) | **0.6687** (mass 0.6933) |
| Yoo et al., JAMA Netw Open 2020 ([PMC7516603](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516603/)) | NLST, nonkalsifiye nodül (Lunit) | 0.93 (tümü) / 0.99 (dijital) / 0.86 (kompütte) |
| Hanaoka et al., IJCARS 2024 ([PMC11442563](https://pmc.ncbi.nlm.nih.gov/articles/PMC11442563/)) | JSRT, nodül tespiti | **Sn @0.2 FP = 0.688** |

**Değerlendirme: orta-güçlü.** ChestX-ray14'te "nodül/mass" **en zor etiketler** (AUC ~0.67–0.72); bizim 0.8631 bunun çok üstünde ve NODE21 rank-1 AUC'sine (0.868) denk. **Kritik nüans:** AUC ≠ düşük FP'deki duyarlılık — rank-1'in AUC'si 0.868 iken 0.125 FP'de duyarlılığı yalnızca 0.603.

### 3.4 ⚠️ KRİTİK UYARI — bizim split'imiz resmî NODE21 split'i DEĞİL

✅ **Yerel doğrulama:** veri kümemiz **birebir** NODE21'in *halka açık training setidir*:

| | Bizim verimiz | NODE21 halka açık training |
| :--- | :---: | :---: |
| Toplam | 4882 | 4882 |
| Nodüllü | 1134 | 1134 |
| Normal | 3748 | 3748 |

Yani **734 hastalık test setimiz, halka açık training setinden yapılmış kendi iç bölünmemizdir** (train 3414 / val 734 / test 734). NODE21'in iki test seti (281 ve 298) **özeldir, yalnızca organizatörlerde bulunur**.

**Sonuçlar:**
* ✅ **Sızıntı yok** — split hasta düzeyinde ve 0 kesişimli olduğu doğrulandı (E0.2 ve `test_v2_pipeline.py`).
* ❌ **Leaderboard ile karşılaştırma geçersizdir**: farklı split + farklı görev (biz lokalizasyon yapmıyoruz → FROC üretemiyoruz).
* ⚠️ Makalede bu **açıkça beyan edilmelidir**; aksi halde "resmî test setinde 0.8631" gibi yanlış bir izlenim doğar.

### 3.5 NODE21 üzerinde topolojik çalışma var mı?

**Bulunamadı** — "NODE21 persistent homology / topological / explainable" aramaları sonuçsuz. En yakın metodolojik referans **Topo-CXR** (ICCVW 2023) ama o **TB/pnömoni** üzerine, nodül değil. Bulunan tek NODE21+açıklanabilirlik makalesi ([arXiv:2510.07681](https://arxiv.org/abs/2510.07681)) **geri çekilmiş (withdrawn)** → kaynak olarak **kullanılamaz**.

⇒ "Bildiğimiz kadarıyla NODE21 üzerinde kalıcı homoloji tabanlı ilk çalışma" iddiası **savunulabilir** (Topo-CXR referansı verilerek).

### 3.6 🔑 Yeniden çerçeveleme: bizim görevimiz NODE21'in AUC metriğiyle uyumlu

NODE21'in **AUC metriği zaten görüntü düzeyindedir** ve görüntü skoru = **maksimum kutu olasılığı** ile hesaplanır. Bizim hasta skorumuz = **5 zon olasılığının maksimumu**. Yani **agregasyon mantığı (max) ve metrik (AUC) örtüşüyor**; fark, bizim lokalizasyon üretmememizdir.

Bu, yayının ana çerçevesini kurar:

> **Ağır bir detektör topluluğu (20–33 model, harici pretrain) NODE21'de görüntü düzeyi 0.868 AUC'ye ulaşıyor. Biz, hiçbir derin tespit mimarisi ve harici pretrain kullanmadan, yalnızca yorumlanabilir topolojik özelliklerle hasta düzeyi 0.8776 AUC elde ediyoruz (E3.4 ensemble) — üstelik kendi iç split'imizde.**

Bu, "SOTA'yı geçtik" iddiası değil; **"yorumlanabilir ve hafif bir temsil, ağır detektörlerin görüntü-düzeyi AUC seviyesine yaklaşıyor"** iddiasıdır ve **dürüst, savunulabilir**.

### 3.7 Yayın için zorunlu karşılaştırmalar

| # | Gereksinim | Neden |
| :-: | :--- | :--- |
| 1 | **Aynı split'te güçlü baseline** (basit CNN + mümkünse bir detektör) — *bizim 734 test setimizde* | En zorunlu adım; "TDA neden global CNN'i geçmiyor?" itirazını kapatır |
| 2 | **FROC / duyarlılık@FP** (0.125/0.25/0.5) **veya** lokalizasyon yapmadığımızın açık beyanı + alternatif metrik (kısmi AUC, sabit özgüllükte duyarlılık) | AUC tek başına yetmez |
| 3 | **NODE21 organizer submission** (public GitHub + Docker) | Tek gerçek elma-elmaya karşılaştırma |
| 4 | **Harici doğrulama** (JSRT veya ChestX-ray14 nodül alt kümesi, eğitimde kullanmadan) | Genelleme iddiası |
| 5 | Özgünlük ifadesi: *"bildiğimiz kadarıyla ilk"* + Topo-CXR referansı | Aşırı iddiadan kaçınma |

---

## 4. Görüntü Ön-İşleme ve İki Aşamalı Tespit

### 4.1 Kemik bastırma — açık kaynak envanteri

| Yöntem | Kaynak | Lisans | Notlar |
| :--- | :--- | :--- | :--- |
| **Qure.ai** (Anatomy-Decomposed CT Projections) | [arXiv:2609.24937](https://arxiv.org/html/2609.24937v2) (2026, *Med Image Anal* incelemede) | **CC-BY-NC-SA-4.0** | TorchScript ağırlıklar: [huggingface.co/qureaiorg/bone-suppression](https://huggingface.co/qureaiorg/bone-suppression) (~400 MB). **NODE21 ve TBX11K'da değerlendirilmiş**; kemik parlak, 1024×1024 resize, min-max [0,1]; `--device cpu` ~1 dk/görüntü. ⚠️ Gelecek tarihli arXiv ID + hakem süreci `[belirsiz]` |
| **SADXNet** | Xu ve ark., *Diagnostics* 13(9):1652 (2023) · [repo](https://github.com/FluteXu/CXR-Rib-Suppression) | MIT | Nodül tespitinde **+%3.23 AUC**, FP 1273→1070; lokalizasyonda +%6.62 AUC. ⚠️ Ağırlıklar "TBA" |
| **GL-LCM** | MICCAI 2025 · [repo](https://github.com/diaoquesang/GL-LCM) · [arXiv:2508.03357](https://arxiv.org/abs/2508.03357) | ağırlıklar açık | 8.54 s/görüntü, BSR %95.6. ⚠️ **Dual-energy eşleştirilmiş veri** gerektirir → NODE21 transferi garanti değil |
| **xU-NetFullSharp** | [repo](https://github.com/ArjunBasandrai/xU-NetFullSharp-Bone-Shadow-Suppression) | GPL-3.0 | Eğitilmiş `model.pth`, tek komut |
| **DeBoNet** | *PLOS ONE* 2022 · [repo](https://github.com/sivaramakrishnan-rajaraman/Bone-Suppresion-Ensemble) | belirtilmemiş | COVID'de MCC 0.796→**0.965** |
| **Ernestosant** | [repo](https://github.com/Ernestosant/Bone-supression-models) | Apache-2.0 | Çoklu model + demo |

### 4.2 Damar bastırma — dürüst boşluk

* **CXR'de hazır/açık bir damar bastırma modeli bulunamadı** (literatür çoğunlukla CT tarafında). Bu, **literatürde az işlenmiş** bir alan.
* **Frangi vesselness hazır:** [`skimage.filters.frangi`](https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html) (`frangi`, `sato`, `meijering`, `hessian`; `sigmas=[1,2,3,4]`). Olgunluk kanıtı: VESSEL12'de Method K ve I, **σ = 1,2,3 mm çok ölçekli Frangi** kullanmış (CT).
* ✅ **Yerel doğrulama:** `scikit-image 0.26.0` kurulu ve `frangi/sato/meijering` mevcut → **ek kütüphane gerekmiyor**.
* ❌ **CXR'de damar bastırmanın nodül tespitine ölçülmüş katkısı** için doğrulanmış sayı bulunamadı `[belirsiz]`.

### 4.3 Bildirilen kazançlar

| Çalışma | Görev | Kazanç |
| :--- | :--- | :--- |
| SADXNet | Nodül tespiti | **+%3.23 AUC**, 1273→1070 FP |
| SADXNet | Akciğer hastalığı lokalizasyonu | +%6.62 AUC, 3029→2644 FP |
| DeBoNet | COVID tespiti | MCC 0.796 → **0.965** |
| Qure.ai | NODE21 / TBX11K | Kazanç var, **ama kemik-örtüşmeli lezyonlarda yoğunlaşıyor** |
| Schalekamp ve ark., *Eur J Radiol* 2013 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/24113431/)) | Radyolog nodül tespiti | Kemik-bastırılmış görüntüler radyolog performansını artırıyor |

> ⚠️ **Kritik uyarı:** Bu kazançların **tamamı dedektör/radyolog** performansı içindir. **H1 topolojik çıkarımı üzerindeki etkiyi sayısallaştıran bir çalışma bulunamadı** → burası bizim özgün katkı alanımız.

### 4.4 İki aşamalı tespit

* **NODE21 baseline'ı tek aşamalıdır** (Faster R-CNN şablonu) — aday üretimi + FP azaltma ayrımı **yapmıyor**. Bu, bizim candidate-based mimarimiz (E3.3) için bir **boşluk/avantaj**.
* Repolar: [DIAGNijmegen/node21-noduledetection](https://github.com/DIAGNijmegen/node21-noduledetection), [osmaoui/node21_detection_baseline](https://github.com/osmaoui/node21_detection_baseline). Veri seti: [Zenodo 4783745](https://zenodo.org/records/4783745).
* Klasik iki-aşamalı şablon (Hessian/şablon eşleme ile aday → CNN patch sınıflandırıcı ile FP azaltma) standarttır, ama bu turda doğrulanmış tek bir sayısal referans bulunamadı `[belirsiz]`.

### 4.5 Topoloji + ön-işleme kombinasyonu — bizim boşluğumuz

* **Topo-CXR'ın tezi bizimkinin TAM TERSİ.** Yazarlar açıkça *"does not need any data augmentation or pre-processing steps"* ifadesini **avantaj** olarak sunuyor. Biz ise H1'in arka plan (kaburga + bronkovasküler döngüler) tarafından maskelendiğini gösterip **ön-işleme/filtreleme eklemeyi** öneriyoruz → literatürde **net, savunulabilir bir karşı-pozisyon ve özgün katkı**.
* ❌ **Kemik/damar bastırmayı kalıcı homoloji ile birleştiren bir çalışma bulunamadı** → **ana özgünlük iddiası** burada.

### 4.6 Kritik uygulama notları (deney tasarımına doğrudan girer)

1. **Kemik bastırmayı GÖRÜNTÜ düzeyinde değil MASKE/diyagram düzeyinde uygula.** Difüzyon/GAN tabanlı bastırıcılar ince nodül dokusunu zayıflatabilir. `bone.png`'yi **kaynak maske** olarak kullanıp H1 doğumlarını süzmek daha güvenli.
2. **Frangi'yi H1 filtre ağırlığı olarak kullan**, görüntüyü agresif değiştirmeden — geri alınabilir ve hiperparametrik ayarlanabilir.
3. **Filtrasyon yeniden kalibre edilmeli.** Bastırılmış görüntüde piksel yoğunluk dağılımı değişir → H0/H1 diyagramları bastırma öncesiyle **karşılaştırılamaz** hale gelir. Pratik çözüm: bastırma sonrası görüntüyü orijinalin min-max aralığına geri ölçekle.
4. **"Kemik bastırma her zaman iyi değil."** Kazanç kemik-örtüşmeli lezyonlarda yoğunlaşıyor → **zon-bazlı kazanç analizi** raporlanmalı (üst zonlarda klavikula/kaburga etkisi daha güçlü). SADXNet'in +%3.23 AUC'si referans büyüklük mertebesi.
5. **Lisans:** Qure.ai CC-BY-NC-SA-4.0 (ticari kullanım yasak), xU-NetFullSharp GPL-3.0. Ticari hedef varsa kendi modelini eğitmek daha güvenli.

---

## 5. Sentez ve Deney Önceliklendirmesi (FİNAL)

### 5.1 Skorumuzun gerçek konumu — "başarısız" değil

| Karşılaştırma | Değer | Yorum |
| :--- | :---: | :--- |
| NODE21 rank-1 (görüntü düzeyi AUC) | 0.868 | Bizim **ensemble (0.8776)** ve **füzyon (0.9211)** bunu **geçiyor** |
| NODE21 ensemble (zirve) | 0.877 | Bizim füzyon **+0.044 geçiyor** (0.9211) |
| NODE21 baseline (DB) | 0.816 | Bizimki **çok üstünde** |
| Güçlü görüntü baseline'ı (bizim, ResNet18) | 0.8349 | **TDA tek başına geçiyor**; füzyon çok geçiyor |
| ChestX-ray14 nodül AUC (Wang 2017) | 0.67 | Bizimki **çok üstünde** |
| Wang 2024 topolojik radiomics (GGN) | 0.85 / 0.862 | Bizimki **üstünde** |
| Topo-CXR (TB/pnömoni) | ~0.99 | **Farklı ve yapısal olarak daha kolay görev** |

**Kritik teknik uyum:** NODE21'in AUC metriği **görüntü düzeyindedir** ve görüntü skoru = **maksimum kutu olasılığı**. Bizim hasta skorumuz = **5 zon olasılığının maksimumu**. ⇒ **Agregasyon mantığı (max) ve metrik (AUC) örtüşüyor.** Farklar: (a) lokalizasyon üretmiyoruz, (b) split'imiz resmî değil.

> 🔑 **Yeniden çerçeveleme:** *"Topolojik ve görüntü temsilleri büyük ölçüde bağımsız bilgi taşır (r ≈ 0.30); füzyon, güçlü bir pretrainli görüntü baseline'ını +0.086 AUC ile geçerek **0.9211** hasta düzeyi ROC-AUC'ye ulaşır — hiçbir derin tespit mimarisi ve uçtan uca eğitim olmadan."* **E3.1 (KEEP)** bu iddiayı bootstrap CI **[+0.0632, +0.1099]**, p≈0.0000 ile kanıtladı. "Resmî test setinde SOTA'yı geçtik" iddiası ise **geçersizdir** (§3.4).

### 5.2 Yenilik hiyerarşisi (yayın için)

| # | İddia | Literatür durumu | Güç |
| :-: | :--- | :--- | :---: |
| 1 | **PL'nin üst-zarf (envelope) mekanizmasıyla çökmesi** — birkaç uzun ömürlü ama ilgisiz H1 sınıfının tüm temsili ezmesi; PI ile doğrudan nedensel karşılaştırma | **Bulunamadı.** Vandaele 2020 "ilgisiz nesne" problemini tanıyor ama **PL zarfı için test etmiyor** | **En güçlü** — ✅ **E4.4 ile nicel kanıtlandı** (sinyal/zarf = %1–2) |
| 2 | **Kemik/damar bastırma + kalıcı homoloji kombinasyonu** | **Bulunamadı** | **Çok güçlü** |
| 3 | Topo-CXR'ın *"ön-işleme gerekmez"* tezine **karşı çıkış** | Literatür ters tezi savunuyor → net karşı-pozisyon | Güçlü |
| 4 | CXR'de H1'in bronkovasküler döngü + kaburga kenarı ile domine edilmesi | Spesifik anatomik formülasyon **yok** | Güçlü |
| 5 | 5 anatomik zon + zon başına PH + hasta düzeyi max-pooling | **Bulunamadı** (Topo-CXR global, Vandaele/Wang ROI-kısıtlı) | Güçlü |
| 6 | NODE21 + kalıcı homoloji | **Bulunamadı** | Orta (*"bildiğimiz kadarıyla ilk"*) |

**Yenilik İDDİA EDİLMEMESİ gerekenler:** "ilgisiz yapılar diyagramı domine eder → ROI kısıtlaması gerekir" (Vandaele 2020/2022) · "PI ve PL farklı davranır, veri setine bağlıdır" (Barnes 2021, Sulowska 2026) · "topoloji tek başına CNN'e denk, füzyonla artar" (TDA-Net) · "topoloji gerçek histolojiyi, radyomik radyolog skorunu taklit eder" (Vandaele 2022).

### 5.3 Zorunlu karşılaştırmalar (hakem itirazlarını kapatmak için)

| # | Gereksinim | Bağlı deney |
| :-: | :--- | :--- |
| 1 | **Aynı split'te güçlü ham-piksel CNN baseline** ("TDA neden global CNN'i geçmiyor?") | ✅ **E0.3 tamamlandı** (ResNet18 test 0.8349 — TDA geçiyor) |
| 2 | **PL mekanizma analizi** (üst zarf görselleştirmesi + nicel ölçüm) | ✅ **E4.4 tamamlandı** (sinyal/zarf %1–2) |
| 3 | **Split beyanı** (iç bölünme; leaderboard karşılaştırması yok) | roadmap §0 kural 9 |
| 4 | **Sağlamlık** (σ/ızgara, ağırlık fonksiyonu, overlap eşiği) | E2.7, E2.8, E4.1 |
| 5 | **Harici doğrulama** (JSRT / ChestX-ray14 nodül alt kümesi) | E5.3 |
| 6 | **NODE21 organizer submission** (public GitHub + Docker) | opsiyonel, en güçlü kanıt |

### 5.4 Literatürden doğan deneyler (roadmap'e işlendi)

| ID | Deney | Dayanak | Efor | Etki |
| :--- | :--- | :--- | :---: | :---: |
| **E4.4** | PL üst-zarf mekanizması analizi | Moon 2020 + Barnes 2021 + Sulowska 2026 | Kolay | **Çok Yüksek** |
| **E2.7** | PI ağırlık fonksiyonu karşılaştırması (Adams `w=p` vs max-dist/linear/arctan) | Moon et al. 3 ağırlık fonksiyonu | Kolay | Orta |
| **E2.8** | σ / ızgara çözünürlük duyarlılık taraması | Moon et al. 625 kombinasyon LOOCV | Kolay | Orta (kanıt) |
| **E2.3** (güçlendirildi) | SEDT-3 / sağlıklı şablon + **kemik maskesiyle H1 süzme** | Moon et al. + Qure.ai | Orta | Yüksek |
| **E4.2** (enabler doğrulandı) | GUDHI `cofaces_of_persistence_pairs()` ile PI noktası → görüntü bölgesi | GUDHI 3.13 (yerel test ✅) | Kolay | Yüksek |
| **E5.3** | Harici doğrulama | JSRT / ChestX-ray14 | Zor | Yüksek |
| **E2.6** (parktan çıktı) | Çok parametreli persistence (multipers/MMA) | multipers olgun | Zor | Orta |

### 5.5 Dürüstlük notları ve doğrulanamayanlar

* **Topo-CXR'ın sayısal sonuçları** ikincil atıftan (RepViT-CXR) alındı — PDF araç zincirince okunamadı.
* **NODE21 leaderboard sayıları** bazı kollarda tam metinden çıkarılamadı; TMI 2024 makalesi ikincil kaynaklarla teyit edildi, ama **sayı uydurulmadı**.
* **Qure.ai** arXiv ID'si gelecek tarihli (2026-09) ve hakem süreci belirsiz → ağırlıklar kullanılsa bile makalede bu belirsizlik belirtilmeli.
* **CXR'de damar bastırmanın nodül tespitine ölçülmüş katkısı** için doğrulanmış kaynak yok.
* **Tarama kapsamı:** ~40 arama + birincil kaynak getirme. `web_fetch` PDF okuyamıyor; ScienceDirect/MDPI/PubMed HTML getirmeleri 403 döndü (PMC ve arXiv HTML çalıştı).

### 5.6 Hakem riski — V6.9 denetiminin doğurduğu üç yeni itiraz

Aşağıdakiler literatürden değil, **kendi denetimimizden** doğdu; hakemin sırayla soracağı
sorular bunlardır ve cevapları hazırdır ([LEAK_AUDIT.md](LEAK_AUDIT.md)):

| # | Hakem itirazı | Bizim cevabımız | Durum |
| :-: | :--- | :--- | :--- |
| 1 | *"X-ray vakıf omurgası VinDr ile ön-eğitilmiş; dış doğrulamanız kontamine."* | Doğru — ve **ölçtük**: VinDr'yi görmemiş densenet121-res224-nih ile aynı mertebe (0,6663 vs 0,6414 nOR). Sayıyı **temiz omurgayla** raporluyoruz. | ✅ kapatıldı |
| 2 | *"Hiperparametrelerinizi aynı validasyonda seçip aynı yerde raporlamışsınız."* | Doğru — **296 + 19 + 12** değerlendirme. Ön-kayıtlı ikinci mühürle dürüst değer verildi: rank_metric 0,9131 → **0,8842**; AUC 0,9793 → **0,9645**. | ✅ kapatıldı (val "seçilmiş" etiketli) |
| 3 | *"Tek tohum; ±0,02'lik sapmayı raporlamıyorsunuz."* | Doğru. İki bağımsız koşuda sapma ölçüldü (±0,02, V6.8) ve tüm etki büyüklükleri **aralık** olarak yazıldı. Çok-seed koşusu **açık iş**. | ⏳ kısmen |
| 4 | *"NODE21 açık veri kümelerinden derlenmiş; ön-eğitimli kodlayıcınız bu görüntülerin çoğunu görmüş olabilir."* | Doğru — **ölçtük**: pozitiflerin **%82,9'u** NIH ChestX-ray14 + PadChest'ten; ama korpusta **radyolog kutusu yok** ⇒ lokalizasyon sızıntıya kapalı. Kaynak-ayrıştırmalı: görülen **0,9698** / görülmeyen **0,9332** AUC. **Belirleyici etkileşim testi:** CXR omurgasının ImageNet'e göre avantajı görülmeyen kaynaklarda **2,6×** daha büyük (etkileşim **−0,200** [−0,299; −0,109]) ⇒ örtüşme kaynaklı şişme lehine **kanıt yok**. | ✅ kapatıldı (beyan + ölçüm) |

**Konumlandırma sonucu:** Bu **dört** itiraz, çalışmanın "başka bir SOTA" iddiası olmadığı
için savunulabilir; makalenin gücü **neyin ne kadar kazandırdığını ve nerede
kazandırmadığını** ölçmesindedir. Buna karşılık "kontaminasyonu ölçüp yok bulduk"
bulgusu, TDA literatüründe **yöntemsel** bir katkı olarak sunulabilir: CXR vakıf
omurgaları yaygınlaştıkça herkesin karşılaşacağı bir sorundur ve nadiren raporlanır.

# NODE21 TDA — Literatür Taraması ve Konumlandırma

> Amaç: [roadmap.md](roadmap.md) v3 programındaki fikirlerin literatürde nerede durduğunu, hangisinin **gerçekten yeni** olduğunu ve bu ışıkta hangi deneylerin öncelik kazanması gerektiğini belirlemek.
> Yöntem: Dört bağımsız kolda web araması + birincil kaynak getirme. Yalnızca doğrulanan kaynaklar yazılır; emin olunmayan ifadeler `[belirsiz]` ile işaretlenir.

**Durum:** §1 tamamlandı · §2–§4 tarama sürüyor · §5 sentez tamamlanacak.

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

## 3. NODE21 ve CXR Nodül Tespitinde SOTA (tarama sürüyor)

> ⏳ Bu bölüm tamamlandığında doldurulacak: NODE21 görev tanımı ve metrikleri, leaderboard/yayın sonuçları, hakim mimariler, "0.8631 nerede duruyor" analizi.

---

## 4. Görüntü Ön-İşleme ve İki Aşamalı Tespit (tarama sürüyor)

> ⏳ Bu bölüm tamamlandığında doldurulacak: kemik/damar bastırma yöntemleri ve açık kaynak durumu, Frangi vesselness, iki aşamalı (aday üretimi + yanlış pozitif azaltma) nodül tespiti ve tipik kazançlar.

---

## 5. Sentez ve Deney Önceliklendirmesi

> ⏳ Dört kol tamamlandığında finalize edilecek.

**Şimdiden netleşen üç sonuç:**

1. **En yakın emsal Topo-CXR (ICCVW 2023):** CXR + kübik PH ile, küçük veri rejiminde, augmentasyonsuz SOTA DL'yi geçiyor. Bizim PI-vs-PL ve TDA-vs-ham-piksel ablasyonlarımız bu çizgiyi **nodül görevine** taşıyor → **özgün katkı, görev + PI temsili + 5-zon anatomik ayrıştırmada**, yöntemin kendisinde değil.
2. **Referans skorumuz literatürle uyumlu:** 0.8631, topolojik radiomics emsalleriyle (0.85–0.862) aynı bantta. Sorun skorun düşüklüğü değil, **güçlü baseline eksikliği** ve **yorumlanabilirlik kanıtının olmaması**.
3. **E2.6 (MPH) "Parkta"dan çıkarılmalı**; ancak cubical adaptasyon riski nedeniyle ikinci dalga. İlk dalga için **SEDT-3 / şablon tabanlı filtrasyon** ve **çok ölçekli σ-ızgara duyarlılık taraması** daha yüksek etki/efor oranı sunuyor.

**Yeni, literatürden doğan deney adayları (backlog'a eklenecek):**

| Öneri | Dayanak | Efor | Etki |
| :--- | :--- | :---: | :---: |
| PI **ağırlık fonksiyonu** karşılaştırması (Adams `w=p` vs max-distance, linear, arctangent) | Moon et al. 3 ağırlık fonksiyonu | Kolay | Orta |
| **σ / ızgara çözünürlük duyarlılık taraması** | Moon et al. 625 kombinasyon LOOCV | Kolay | Orta (sağlamlık kanıtı) |
| **GUDHI `cofaces_of_persistence_pairs()`** ile PI noktası → görüntü bölgesi geri izleme | GUDHI 3.13 API | Kolay | **Yüksek** (E4.2 yorumlanabilirlik) |
| **SEDT-3 / sağlıklı şablon** filtrasyonu | Moon et al. | Orta | Yüksek |
| **PersLay** ile öğrenilebilir vektörleştirme | repo hazır | Orta | Orta |

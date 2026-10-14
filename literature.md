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

### 1.6 Tıbbi görüntülemede yorumlanabilirlik ve performans emsalleri

| Çalışma | Veri | Sonuç |
| :--- | :--- | :--- |
| **Wang et al. 2024** ([PMC11452904](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452904/)) | 3.223 hasta, 2 merkez, GGN malignite | Topolojik radiomics AUC **0.85 / 0.862**; klinik+radyomik+topolojik **0.879** (Giotto-TDA, 270 topolojik özellik) |
| **Moon et al.** ([arXiv:2012.12102](https://ar5iv.labs.arxiv.org/html/2012.12102)) | Akciğer + beyin | Persistence diyagramı **kadran → şekil paterni** haritası (H0 kadran II = dağınık lezyon, III = bağlı lezyon, H1 kadran II = lezyonun çevrelediği normal doku); **3 ağırlık fonksiyonu** karşılaştırması |
| **Zhu et al. 2024** ([arXiv:2408.07905](https://arxiv.org/abs/2408.07905) · [repo](https://github.com/hrlblab/TopologicalDataAnalysis3D)) | MedMNIST3D | 3D PI + superpixel + optimize Gauss katsayısı; kod açık |
| **Singh et al., Oncotarget 2024** ([PMC11559656](https://pmc.ncbi.nlm.nih.gov/articles/PMC11559656/)) | — | TDA'nın radyolojide yorumlanabilirlik/bias argümanı (editöryel, deneysel kanıt değil) |

**Konumlandırma:** Bizim hasta düzeyi **0.8631** ROC-AUC'muz, Wang et al.'ın topolojik radiomics aralığıyla (**0.85–0.862**) aynı ölçekte. Yani skor "zayıf" değil; eksik olan, **karşılaştırılabilir güçlü bir baseline** ve **yorumlanabilirlik kanıtı**.

---

## 2. Tıbbi Görüntülemede TDA (tarama sürüyor)

> ⏳ Bu bölüm, ilgili araştırma kolu tamamlandığında doldurulacak: TDA'nın akciğer/CXR'deki uygulamaları, diyagram temsillerinin karşılaştırılması, bildirilen başarı seviyeleri, "arka plan topolojisi baskın çıkıyor" probleminin literatürde ele alınışı.

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

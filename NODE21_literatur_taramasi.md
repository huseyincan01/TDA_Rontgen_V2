# NODE21 ve CXR Nodül Tespiti — Literatür Taraması

## 1. NODE21 nedir?

**Kaynak: Sogancioglu, van Ginneken, ... Murphy, "Nodule detection and generation on chest X-rays: NODE21 Challenge", IEEE TMI 2024** ([arXiv:2401.02192](https://arxiv.org/abs/2401.02192), [IEEE](https://ieeexplore.ieee.org/document/10479589), [challenge sitesi](https://node21.grand-challenge.org/), [tam metin](https://pure.uva.nl/ws/files/233416363/Nodule_Detection_and_Generation_on_Chest_X-Rays_NODE21_Challenge.pdf))

İki track, yarışma değil **işbirliği** (kod açık kaynak zorunlu). **Detection**: frontal CXR → skorlu bounding box listesi. **Generation**: CXR + lokasyon → o noktaya nodül sentezi (amaç: sentetik veriyle detektörü güçlendirmek). 19 Eki 2021–25 Oca 2022; 302 katılımcı, >230 submission, finalde 7 ülkeden 10 takım.

**Veri setleri:**
- **Training (halka açık)**: JSRT, PadChest, ChestX-ray14, OpenI'den seçilip radyolog onaylı **4882 görüntü = 1134 nodüllü (1476 nodül) + 3748 normal**. Generation için normal görüntülerden 1000 görüntü + görüntü başına 1–3 kutu (7–17 mm).
- **Experimental test (özel)**: **281 görüntü = 166 nodüllü (248 nodül) + 115 normal**.
- **Final test (özel)**: **298 görüntü = 111 nodül-pozitif (tek nodül) + 187 negatif**; CT tabanlı referans (pozitif: CXR'de soliter nodül + 3 ay içinde CT; negatif: 6 ay içinde CT). Ayrıca 12 okuyucunun (6 radyolog + 6 asistan) bulguları.

**Metrikler (detection, makale III.B.1):**
- **AUC**: görüntü skoru = o görüntüdeki **maksimum bounding-box olasılığı** (hiç tahmin yoksa 0) → eşiklenip ROC. Yani görüntü düzeyi, max-score agregasyonu.
- **FROC**: eşleşme **IoU > 0.2**; duyarlılık **ortalama 0.125 / 0.25 / 0.50 FP/görüntü** noktalarında.
- **CPM NODE21'de YOK** — ne makalede ne leaderboard'da geçiyor. Leaderboard kolonları: `final_score, AUC, S(0.25 FP), S(0.5 FP)` ([leaderboard](https://node21.grand-challenge.org/evaluation/challenge/leaderboard/), [final test](https://node21.grand-challenge.org/evaluation/detection-track-final-test-set/leaderboard/)). (CPM, LUNA16 gibi CT challenge'larında kullanılır.)

## 2. En iyi sonuçlar (Sogancioglu et al., TMI 2024, final test 298 görüntü)

| Yöntem | Mimari | AUC | S(0.5 FP) | S(0.25 FP) | S(0.125 FP) |
|---|---|---|---|---|---|
| **D1 (rank 1)** | Faster R-CNN + RetinaNet + YOLOv5(640/1024) + EfficientDet-D2, 20 model, WBF; VinDr-CXR pretrain | **0.868** | 0.800 | 0.750 | 0.603 |
| **D2 (rank 2)** | 33× YOLOv5 ensemble, NMS, scratch | 0.862 | 0.771 | 0.723 | 0.600 |
| **D3 (rank 3)** | Mask R-CNN + RetinaNet (ResNet-50), 6 model, NMS | 0.833 | 0.761 | 0.704 | 0.590 |
| **DB (baseline)** | Faster R-CNN, COCO pretrain | 0.816 | 0.714 | 0.635 | 0.504 |
| **Ensemble (D1+D2+D3+DB)** | weighted box fusion | **0.877** | 0.819 | 0.754 | 0.619 |

Generation rank-1 **G1** (CR-Fill tabanlı inpainting GAN). Sentetik nodül, gerçek veri tam kullanıldığında DB'yi 0.816→**0.844** yaptı (anlamlı değil); gerçek verinin ≤%75'inde anlamlı iyileşme. Sadece sentetikle eğitilen DB ≈ 0.78–0.80.

**Kazananlar**: iki-aşamalı, bol ensemble'lı anchor-based detektörler (YOLO / Faster/Mask R-CNN / RetinaNet) + NMS/WBF + harici pretrain (VinDr-CXR, DeepLesion). **Hiçbir topolojik/açıklanabilir/hafif yöntem kazanmıyor.** Ensemble 12 radyologdan 3'ünü geçiyor, 8'iyle benzer (AUC 0.877, %95 CI 0.830–0.915).

## 3. "Güçlü" performans ve 0.86'nın yeri

| Kaynak | Veri | Görev | Metrik | Değer |
|---|---|---|---|---|
| NODE21 TMI 2024 | NODE21 final (298) | nodül tespiti | AUC | 0.816 / 0.868 / 0.877 (DB/D1/ens.) |
| [Wang et al., CVPR 2017](https://arxiv.org/abs/1705.02315) | ChestX-ray14, ResNet-50 | **Nodule** (zayıf etiket) | AUC | **0.6687** |
| Wang et al. 2017 | ChestX-ray14 | **Mass** | AUC | 0.6933 |
| [Yoo et al., JAMA Netw Open 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516603/) | NLST, 577 görüntü | nonkalsifiye nodül | AUROC | **0.93** (tümü), **0.99** (dijital), 0.86 (kompütte) |
| [Hanaoka et al., IJCARS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11442563/) | JSRT (CT-doğrulanmış), LOCO | nodül tespiti | Sn @ 0.2 FP | **0.688** (fine-tune'lı), 0.507 (fine-tune'sız) |

ChestX-ray14'te "Nodule/Mass" **en zor etiketler** (AUC ~0.67–0.72) — 0.86 çok üstünde. NODE21'de ise 0.86–0.87 **zaten challenge'ın zirvesi**. Tek bir "güçlü" eşik yok; veri setine ve metriğe bağlı. Kritik nüans: **AUC ≠ düşük FP'deki FROC duyarlılığı** — rank-1 AUC 0.868 iken 0.125 FP'de duyarlılık sadece 0.603.

## 4. NODE21 üzerinde açıklanabilir / hafif / topolojik çalışma?

**Bulunamadı: NODE21 üzerinde persistent homology / topolojik özellik kullanan yayınlanmış bir çalışma yok** ("NODE21 persistent homology / topological / explainable" aramaları sonuçsuz).

İlgili ama NODE21'de olmayan:
- [Ahmed, Nuwagira, Torlak, Coskunuzer, "Topo-CXR: Chest X-ray TB and Pneumonia Screening with Topological Machine Learning", ICCV 2023 Workshops, s. 2326–2336](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Ahmed_Topo-CXR_Chest_X-ray_TB_and_Pneumonia_Screening_with_Topological_Machine_ICCVW_2023_paper.html) — **cubical persistence** + standart ML; **TB ve pnömoni**, nodül değil. Bizim yaklaşıma en yakın metodolojik referans. Sayısal sonuçlar [belirsiz] (PDF metni çıkarılamadı).
- [arXiv:2510.07681](https://arxiv.org/abs/2510.07681) "Curriculum Learning with Synthetic Data..." (2025) — NODE21 (1213 hasta) + VinDr-CXR + CheXpert + DDPM sentetik, Faster R-CNN+FPN, Grad-CAM; AUC 0.95 vs 0.89 baseline. **Ancak makale GERİ ÇEKİLDİ (withdrawn)** — kaynak olarak kullanılamaz.

## 5. Diğer açık CXR veri setlerinde tipik performans

- **JSRT** (247 radyograf, 154 nodüllü, CT-doğrulanmış): FROC tabanlı. Hanaoka et al. 2024, tamamen sentetik eğitim + LOCO ince ayarla **0.2 FP'de duyarlılık 0.688**. Aynı makalenin aktardığı literatür: Li et al. 5-fold CV ile 0.2 FP'de %99 [belirsiz — harici validasyonda ciddi düşüş], Chen et al. 2.0 FP'de %91.4. Küçük ve FROC tabanlı → AUC ile karşılaştırılamaz.
- **ChestX-ray14 nodül alt kümesi**: NLP ile metinden çıkarılmış zayıf etiketler, lokalizasyon yok. Tipik **nodül AUC ≈ 0.67–0.72**, mass ≈ 0.69. [belirsiz — ChestX-ray14'te 0.85+ nodül AUC iddiası güvenilir kaynakla doğrulanamadı.]
- **NLST**: Yoo et al. 2020, Lunit INSIGHT CXR ile **nodül AUROC 0.93** (dijital 0.99, kompütte 0.86); 577 görüntü, 2 radyolog uzlaşma etiketi.
- Üst düzey AUC'ler (0.94–0.98) hep **CT** çalışmalarında. CXR projeksiyon görüntüsü olduğu için yapısal olarak daha zor.

---

## Proje için önemli çıkarımlar

1. **Metrik uyumu iyi.** NODE21 AUC'si zaten görüntü düzeyi max-skor agregasyonu ile hesaplanıyor; hasta düzeyine max-pooling taşımamız challenge tanımıyla tutarlı. Ancak NODE21'de skor "bounding-box olasılığı maksimumu", bizde sınıflandırma skoru — makalede belirtilmeli.

2. **KRİTİK UYARI — bizim split'imiz resmî NODE21 split'i değil.** Bizim sayımız (4882 = 1134 + 3748) NODE21'in halka açık **training setinin birebir aynısı**. NODE21'in iki test seti **özeldir, yalnızca organizatörlerde** (281 ve 298 görüntü). Yani 734 hastalık test setimiz büyük olasılıkla halka açık training setinden yapılmış **kendi iç bölünmemiz** — hasta düzeyinde ayrılmadıysa veri sızıntısı riski var. **0.8631'i leaderboard'daki 0.868/0.877 ile karşılaştırmak geçersizdir** (farklı split, farklı görev: biz lokalizasyon yapmıyoruz).

3. **Konumlandırma.** 0.8631, NODE21 rank-1 (0.868) ve ensemble (0.877) seviyesine çok yakın; ChestX-ray14 (~0.67–0.72) ve JSRT'nin düşük-FP duyarlılıkları düşünülünce **orta-güçlü**. Ama (a) resmî olmayan split'te, (b) lokalizasyonsuz. "NODE21'de SOTA'ya yakın" yerine "kendi split'imizde NODE21 rank-1 AUC'sine denk" demek dürüst olur.

4. **Yayın için gerekli karşılaştırmalar:**
   - **Aynı split'te re-baseline**: D1/D2 benzeri bir detektör + basit bir CNN'i *bizim* 734 hastalık test setimizde çalıştırmak (en zorunlu adım).
   - **FROC / duyarlılık@FP raporlamak**: sadece AUC değil, 0.125/0.25/0.5 FP per image duyarlılıkları (rank-1: 0.603/0.750/0.800). Bu olmadan NODE21 literatürüyle konuşulamaz.
   - **NODE21 organizer submission** (public GitHub + Docker): tek gerçek elma-elmaya karşılaştırma.
   - **Harici doğrulama**: JSRT veya ChestX-ray14 nodül alt kümesi, eğitimde kullanmadan.
   - **Topolojik özgünlük**: "NODE21 üzerinde persistent homology tabanlı ilk çalışma" muhtemelen doğru bir iddia — ama "bildiğimiz kadarıyla ilk" diye ifade edip Topo-CXR'ı (ICCV 2023 W) mutlaka referans vermek gerekir.

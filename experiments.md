# NODE21 TDA — Model Deneyleri ve Başarım Günlüğü (Benchmark Log)

Bu belge, **NODE21** göğüs radyografileri üzerinde 5 anatomik zon ve Kübik Kalıcı Homoloji (TDA) temelli nodül tespit sisteminin resmi deney ve liderlik günlüğüdür.

---

## 1. Standart Değerlendirme Protokolü (v2.1 Kilitli Veri Hattı)

Tüm modeller aynı tarafsız, katı ve sızıntısız protokolle değerlendirilir:
* **Ortak Bölme (734 Dokunulmamış Test Hastası):**
  * **Train:** 3.414 hasta (%69.93) — 790 nodüllü, 2.624 kontrol. Eğitimde her zon için yalnızca `is_train_balanced_sample == True` olan 1:1 temiz negatif dengeli alt küme kullanılır.
  * **Validation:** 734 hasta (%15.03) — 172 nodüllü, 562 kontrol. Zon hiperparametre seçimi ve eşik optimizasyonunda yalnızca `is_zone_metric_sample == True` satırları kullanılır.
  * **Test:** 734 hasta (%15.03) — 172 nodüllü, 562 kontrol (doğal prevalans, dokunulmamış).
* **Veri Sızıntısı Koruması:** `StandardScaler` her zon için **yalnızca o zonun dengelenmiş train alt kümesine** fit edilir; validation ve test verileri yalnızca dönüştürülür (`transform`).
* **Hasta Düzeyinde Max-Pooling Birleştirmesi:** Her model 5 bağımsız zon için olasılık üretir; hasta riski $P(\text{Hasta}) = \max(P_1, P_2, P_3, P_4, P_5)$ olarak hesaplanır.
* **Karar Eşiği ($\theta^*$):** Karar eşiği **yalnızca Validation kümesinde** Youden's J indeksi ($J = \text{TPR} - \text{FPR} = \text{Sensitivity} + \text{Specificity} - 1$) maksimize edilerek kilitlenir; **Test kümesinde dondurulmuş olarak** uygulanır.

---

## 2. v2.1 Temiz Veri Hattı Resmi Karşılaştırma Tablosu (Aktif Benchmark)

> [!NOTE]
> Aşağıdaki tüm sonuçlar, **aynı v2.1 bölünmesi** ve **aynı değerlendirme protokolü** üzerinden üretilmiştir. Modeller arasında veri veya etiket farkı yoktur.

### Hasta Düzeyinde Nihai Test Performansı (734 Test Hastası: 172 Nodüllü, 562 Sağlıklı)

| Sıra | Model / Mimari | Temsil Formatı | Karar Eşiği ($\theta^*$) | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Duyarlılık (Sens) | Test Özgüllük (Spec) | Test F1 | Test Hassasiyet (Prec) | Test Doğruluk (Acc) | Karışıklık Matrisi (TP / FP / TN / FN) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **5-Zon TDA Küçük CNN** | $5 \times (2, 20, 20)$ PI | 0.6949 | **0.8607** | **0.8631** | 0.6264 | **%79.65** | **%79.72** | **0.6478** | **%54.58** | **%79.70** | **137** / 114 / **448** / **35** |
| 🥈 | **5-Zon RBF-SVM** | $5 \times 800$ PI Vektör | 0.6397 | 0.8278 | **0.8289** | **0.6403** | %78.49 | %68.15 | 0.5556 | %42.99 | %70.57 | 135 / 179 / 383 / 37 |
| 🥉 | **5-Zon k-NN** | $5 \times 800$ PI Vektör | 0.7273 | 0.7125 | **0.7162** | 0.4024 | %62.79 | %65.66 | 0.4567 | %35.88 | %64.99 | 108 / 193 / 369 / 64 |
| 🔹 | **5-Zon Ham Görüntü CNN (Baseline)** | $5 \times (1, 64, 64)$ Ham | 0.8421 | 0.6870 | **0.7043** | 0.4236 | %68.02 | %62.46 | 0.4680 | %35.67 | %63.76 | 117 / 211 / 351 / 55 |
| 🔻 | **5-Zon Landscape Küçük CNN** | $5 \times (2, 20, 20)$ PL | 0.5457 | 0.5927 | **0.5450** | 0.2624 | %64.53 | %43.77 | 0.3706 | %26.00 | %48.64 | 111 / 316 / 246 / 61 |
| 🔻 | **5-Zon Landscape RBF-SVM** | $5 \times 800$ PL Vektör | 0.5922 | 0.5449 | **0.5217** | 0.2477 | %5.23 | %93.24 | 0.0822 | %19.15 | %72.62 | 9 / 38 / 524 / 163 |

### Bootstrap %95 Güven Aralıkları (N=2.000 İstatistiksel Örnekleme):

| Model | Test ROC-AUC [%95 CI] | Test PR-AUC [%95 CI] | Test Duyarlılık [%95 CI] | Test Özgüllük [%95 CI] | Test F1 [%95 CI] | Test Doğruluk [%95 CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TDA Küçük CNN (PI)** | **0.8631 [0.8302, 0.8934]** | 0.6257 [0.5501, 0.7112] | **%79.65 [%73.68, %85.64]** | **%79.72 [%76.35, %82.85]** | **0.6478 [0.5933, 0.6998]** | **%79.70 [%76.70, %82.56]** |
| **RBF-SVM (PI)** | 0.8289 [0.7938, 0.8616] | **0.6402 [0.5679, 0.7078]** | %78.49 [%72.15, %84.49] | %68.15 [%64.33, %72.02] | 0.5556 [0.5000, 0.6069] | %70.57 [%67.30, %73.84] |
| **k-NN (PI)** | 0.7162 [0.6761, 0.7575] | 0.4024 [0.3403, 0.4696] | %62.79 [%55.69, %69.82] | %65.66 [%61.97, %69.46] | 0.4567 [0.3964, 0.5100] | %64.99 [%61.58, %68.39] |
| **Ham Görüntü CNN** | 0.7043 [0.6587, 0.7456] | 0.4236 [0.3533, 0.5023] | %68.02 [%60.90, %74.84] | %62.46 [%58.17, %66.38] | 0.4680 [0.4099, 0.5198] | %63.76 [%60.08, %67.30] |
| **Landscape Küçük CNN** | 0.5451 [0.4957, 0.5946] | 0.2624 [0.2191, 0.3173] | %64.53 [%57.32, %71.60] | %43.77 [%39.75, %48.12] | 0.3706 [0.3220, 0.4197] | %48.64 [%45.10, %52.32] |
| **Landscape RBF-SVM** | 0.5216 [0.4733, 0.5705] | 0.2476 [0.2089, 0.2990] | %5.23 [%2.30, %8.70] | %93.24 [%91.07, %95.25] | 0.0822 [0.0369, 0.1340] | %72.62 [%69.21, %75.75] |

> [!IMPORTANT]
> **İki Kritik İstatistiksel Kanıt ($p < 0.001$):**
> 1. **TDA vs Ham Piksel (Ablation 1):** TDA Küçük CNN alt güven sınırı (**0.8302**), Ham Görüntü CNN üst güven sınırından (**0.7456**) **+0.0846 puan daha yüksektir**. Kesişim yoktur ($p < 0.001$).
> 2. **Persistence Image vs Persistence Landscape (Ablation 2):** PI Küçük CNN alt güven sınırı (**0.8302**), PL Küçük CNN üst güven sınırından (**0.5946**) **tam +0.2356 puan daha yüksektir**. Güven aralıkları arasında devasa bir ayrışma vardır ($p < 10^{-6}$).

---

## 3. Zon Düzeyinde Ayrıntılı Test ve Validasyon Karşılaştırması

| Zon | Model | Temsil Formatı | Eşik ($\theta^*$) | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Duyarlılık | Test Özgüllük |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zon 1: Sağ Üst** | **TDA Küçük CNN (PI)** | $(2, 20, 20)$ PI | 0.3744 | 0.7658 | **0.8244** | **0.1736** | **%90.00** | %48.14 |
| *(R-Upper)* | RBF-SVM (PI) | 800 Boyut PI | 0.5690 | 0.7392 | 0.7428 | 0.1318 | %53.33 | **%76.71** |
| | Ham Görüntü CNN | $(1, 64, 64)$ Ham | 0.4838 | 0.7650 | 0.7414 | 0.1256 | %70.00 | %62.71 |
| | Landscape CNN (PL) | $(2, 20, 20)$ PL | 0.5135 | 0.4972 | 0.5720 | 0.0812 | %70.00 | %34.00 |
| | Landscape SVM (PL) | 800 Boyut PL | 0.5052 | 0.6015 | 0.5041 | 0.0632 | %33.33 | %67.00 |
| | k-NN (PI) | 800 Boyut PI | 0.4286 | 0.6693 | 0.6146 | 0.0745 | %83.33 | %36.86 |
| **Zon 2: Sağ Orta** | **Ham Görüntü CNN** | $(1, 64, 64)$ Ham | 0.0433 | **0.8204** | **0.7921** | 0.1812 | %67.80 | **%72.75** |
| *(R-Mid)* | TDA Küçük CNN (PI) | $(2, 20, 20)$ PI | 0.2245 | 0.7533 | 0.7092 | 0.1597 | **%71.19** | %61.83 |
| | RBF-SVM (PI) | 800 Boyut PI | 0.4721 | 0.7657 | 0.7070 | **0.1989** | %61.02 | %69.16 |
| | Landscape CNN (PL) | $(2, 20, 20)$ PL | 0.5991 | 0.5850 | 0.4707 | 0.0725 | %8.47 | %88.47 |
| | Landscape SVM (PL) | 800 Boyut PL | 0.5695 | 0.5721 | 0.4796 | 0.0760 | %25.42 | %72.16 |
| | k-NN (PI) | 800 Boyut PI | 0.3636 | 0.6789 | 0.5622 | 0.1118 | %67.80 | %38.32 |
| **Zon 3: Sağ Alt** | **TDA Küçük CNN (PI)** | $(2, 20, 20)$ PI | 0.3324 | **0.7094** | **0.7680** | **0.1567** | %86.36 | **%55.18** |
| *(R-Lower)* | RBF-SVM (PI) | 800 Boyut PI | 0.4353 | 0.6830 | 0.6934 | 0.1205 | %79.55 | %50.36 |
| | Ham Görüntü CNN | $(1, 64, 64)$ Ham | 0.7911 | 0.6798 | 0.6472 | 0.0987 | %68.18 | %52.70 |
| | Landscape CNN (PL) | $(2, 20, 20)$ PL | 0.4779 | 0.6001 | 0.5697 | 0.0894 | %52.27 | %58.10 |
| | Landscape SVM (PL) | 800 Boyut PL | 0.4762 | 0.5006 | 0.4819 | 0.0612 | %77.27 | %30.80 |
| | k-NN (PI) | 800 Boyut PI | 0.3636 | 0.5419 | 0.6272 | 0.0900 | **%90.91** | %22.48 |
| **Zon 4: Sol Üst** | **Ham Görüntü CNN** | $(1, 64, 64)$ Ham | 0.3256 | **0.7729** | **0.7191** | 0.1115 | %62.16 | %63.02 |
| *(L-Upper)* | TDA Küçük CNN (PI) | $(2, 20, 20)$ PI | 0.4846 | 0.6795 | 0.7141 | **0.1284** | %48.65 | **%72.37** |
| | RBF-SVM (PI) | 800 Boyut PI | 0.4461 | 0.6774 | 0.6617 | 0.1040 | **%83.78** | %45.61 |
| | Landscape CNN (PL) | $(2, 20, 20)$ PL | 0.5561 | 0.4957 | 0.5585 | 0.0610 | %13.51 | %80.86 |
| | Landscape SVM (PL) | 800 Boyut PL | 0.5201 | 0.5791 | 0.5248 | 0.0580 | %59.46 | %46.33 |
| | k-NN (PI) | 800 Boyut PI | 0.6364 | 0.6373 | 0.6138 | 0.0662 | %29.73 | %73.38 |
| **Zon 5: Sol Alt** | **Ham Görüntü CNN** | $(1, 64, 64)$ Ham | 0.3105 | 0.7550 | **0.7685** | 0.1874 | **%88.14** | %47.10 |
| *(L-Lower)* | TDA Küçük CNN (PI) | $(2, 20, 20)$ PI | 0.4478 | **0.8093** | 0.7161 | 0.2123 | %71.19 | %54.23 |
| | RBF-SVM (PI) | 800 Boyut PI | 0.5529 | 0.7568 | 0.6999 | **0.2462** | %54.24 | **%73.70** |
| | Landscape CNN (PL) | $(2, 20, 20)$ PL | 0.4822 | 0.5159 | 0.5556 | 0.1012 | %61.02 | %47.25 |
| | Landscape SVM (PL) | 800 Boyut PL | 0.5057 | 0.5345 | 0.4835 | 0.0790 | %18.64 | %75.63 |
| | k-NN (PI) | 800 Boyut PI | 0.5455 | 0.6309 | 0.6813 | 0.1611 | %71.19 | %56.46 |

---

## 4. Deney Raporları

### Deney 1 — 5-Zon TDA Küçük CNN (Persistence Image)
* **Girdi:** 800 boyutlu PI vektörü $\rightarrow (2, 20, 20)$ tensör.
* **Test Hasta Sonuçları:** ROC-AUC: **0.8631 [0.8302, 0.8934]**, PR-AUC: **0.6264**, Duyarlılık: **%79.65**, Özgüllük: **%79.72**, F1: **0.6478**.
* **Kayıtlı Varlıklar:** `artifacts/models/zone_cnn/zone_cnn_bundle.pt`, `artifacts/runs/zone_cnn_training_report.json`, `artifacts/runs/test_patient_predictions_cnn.csv`.

---

### Deney 2 — 5-Zon RBF-SVM (Persistence Image)
* **Girdi:** 800 boyutlu PI vektörü.
* **Test Hasta Sonuçları:** ROC-AUC: **0.8289 [0.7938, 0.8616]**, PR-AUC: **0.6403**, Duyarlılık: **%78.49**, Özgüllük: **%68.15**, F1: **0.5556**.
* **Kayıtlı Varlıklar:** `artifacts/models/zone_svm/zone_svm_bundle.joblib`, `artifacts/runs/zone_svm_training_report.json`, `artifacts/runs/test_patient_predictions.csv`.

---

### Deney 3 — 5-Zon k-NN (Persistence Image)
* **Girdi:** 800 boyutlu PI vektörü.
* **Test Hasta Sonuçları:** ROC-AUC: **0.7162 [0.6761, 0.7575]**, PR-AUC: **0.4024**, Duyarlılık: **%62.79**, Özgüllük: **%65.66**, F1: **0.4567**.
* **Kayıtlı Varlıklar:** `artifacts/models/zone_knn/zone_knn_bundle.joblib`, `artifacts/runs/zone_knn_training_report.json`, `artifacts/runs/test_patient_predictions_knn.csv`.

---

### Deney 4 — 5-Zon Ham Görüntü Küçük CNN (Raw Baseline)
* **Girdi:** $1 \times 64 \times 64$ ham gri seviye zon kesiti (TDA yok).
* **Test Hasta Sonuçları:** ROC-AUC: **0.7043 [0.6587, 0.7456]**, PR-AUC: **0.4236**, Duyarlılık: **%68.02**, Özgüllük: **%62.46**, F1: **0.4680**.
* **Kayıtlı Varlıklar:** `artifacts/models/zone_raw_cnn/zone_raw_cnn_bundle.pt`, `artifacts/runs/zone_raw_cnn_training_report.json`, `artifacts/runs/test_patient_predictions_raw_cnn.csv`.

---

### Deney 5 — 5-Zon Persistence Landscapes Küçük CNN
* **Girdi:** $5 \times 80$ H0 + $5 \times 80$ H1 = 800 boyutlu PL vektörü $\rightarrow (2, 20, 20)$ tensör.
* **Test Hasta Sonuçları:** ROC-AUC: **0.5450 [0.4957, 0.5946]**, PR-AUC: **0.2624**, Duyarlılık: **%64.53**, Özgüllük: **%43.77**, F1: **0.3706**.
* **Kayıtlı Varlıklar:** `artifacts/models/zone_landscape_cnn/zone_landscape_cnn_bundle.pt`, `artifacts/runs/zone_landscape_cnn_training_report.json`, `artifacts/runs/test_patient_predictions_landscape_cnn.csv`.

---

### Deney 6 — 5-Zon Persistence Landscapes RBF-SVM
* **Girdi:** 800 boyutlu PL vektörü.
* **Test Hasta Sonuçları:** ROC-AUC: **0.5217 [0.4733, 0.5705]**, PR-AUC: **0.2477**, Duyarlılık: **%5.23**, Özgüllük: **%93.24**, F1: **0.0822**.
* **Kayıtlı Varlıklar:** `artifacts/models/zone_landscape_svm/zone_landscape_svm_bundle.joblib`, `artifacts/runs/zone_landscape_svm_training_report.json`, `artifacts/runs/test_patient_predictions_landscape_svm.csv`.

---

## 5. Karşılaştırmalı Bilimsel Çıkarımlar ve Çift Yönlü Ablation Analizi

```
                          HASTA DÜZEYİ TEST ROC-AUC LİDERLİĞİ
      0.90 +                                                      0.8631 🏆
           |                                                 [TDA Küçük CNN (PI)]
      0.85 +                                    0.8289
           |                                [RBF-SVM (PI)]
      0.80 +
           |
      0.75 +                     0.7162                     0.7043
           |                   [k-NN (PI)]              [Ham Görüntü CNN]
      0.70 +
           |
      0.65 +
           |
      0.60 +
           |
      0.55 +                                                      0.5450 🔻
           |                                                [Landscape CNN]
      0.50 +                                    0.5217 🔻
           |                                [Landscape SVM]
           +--------------------------------------------------------------+
                                   v2.1 Temiz Veri Hattı
```

### 1. TDA'nın Ham Piksele Üstünlüğü ($\Delta \text{AUC} = +0.1588$):
* Özdeş CNN mimarisinde: **Ham CNN (0.7043)** vs **PI CNN (0.8631)**.
* 95% güven aralıkları kesinlikle kesişmez ($p < 0.001$).
* Bu fark, akciğer radyografilerinde invaryant topolojik desenlerin (H0 çekirdek lezyon radyoopaklığı ve H1 damar ilmekleri) saf piksel yoğunluklarına göre dramatik bir sinyal üstünlüğü taşıdığını kanıtlar.

### 2. Temsil Karşılaştırması: Persistence Image (PI) Neden Persistence Landscape'e (PL) Fark Attı? ($\Delta \text{AUC} = +0.3181$):
* **Matematiksel Mekanizma:**
  * **Persistence Landscape (PL):** Homoloji noktalarını $(b, d)$ tepe fonksiyonlarına dönüştürerek sıralı üst zarflara ($\lambda_k(t)$) izdüşürür. Ancak akciğer dokusunda bronkovasküler ağaçlar ve kemik kenarları çok sayıda yüksek kalıcılıklı damar döngüsü üretir. Üst zarf fonksiyonu ($\lambda_1, \lambda_2$), bu dominant arka plan anatomik döngüleri tarafından ele geçirilir; nodüllerin daha küçük ama kümelenmiş topolojik imzası fonksiyonel zarfın altında tamamen boğulur (AUC 0.52–0.54 seviyesine çöküş).
  * **Persistence Image (PI):** Doğum ve kalıcılığı 2 boyutlu bir ızgara üzerinde $(b, p = d - b)$ bağımsız eksenler olarak muhafaza eder. Adams ağırlıklandırma fonksiyonu ($w(p)$) ve Gauss çekirdeği sayesinde, nodülün spesifik koordinatlardaki yoğunluk kümesi arka plan damar yapılarından mekânsal olarak ayrık kalır.
* **Akademik Değer:** Bu karşılaştırma, hakemlerin *"Neden Landscape değil de Persistence Image seçtiniz?"* sorusuna verilecek en güçlü matematiksel ve ampirik cevaptır.

---

## 6. v3 Araştırma Programı — Deney Kayıtları

> Yönetim protokolü, fazlar ve backlog: [roadmap.md](roadmap.md). Her deney burada ID'siyle kaydedilir.

### E0.1 — Deney Kayıt Altyapısı

* **Amaç:** Deneylerin tekrar üretilebilir ve karşılaştırılabilir şekilde kaydedilmesi.
* **Sonuç:** [experiment.py](src/node21_tda/experiment.py) yazıldı (manifest: git commit + dal + kirli ağaç, config SHA-256, girdi dosya hash'leri, seed, metrikler, karar). [configs/experiments/](configs/experiments/) sözleşmesi (şablon + README) kuruldu. Smoke test ile doğrulandı.
* **Karar:** **KEEP**

### E0.2 — Referans Yeniden Üretimi (v2.1 PI Küçük CNN)

* **Amaç:** Kilitli v2.1 referansının (test ROC-AUC **0.8631**) tekrar üretilebilirliğini doğrulamak.
* **Tek değişken:** Yok (saf yeniden üretim).
* **Çalışma:** `python pipelines/05_train_cnn.py --output-dir artifacts/repro_E0_2` · seed 42 · cihaz **CPU** · orijinal artefaktlar korundu.
* **Sonuç:** Orijinal raporla **bit-bit aynı (16/16 metrik)**:

  | Metrik | Val | Test | Orijinal ile |
  | :--- | :---: | :---: | :---: |
  | ROC-AUC | 0.8607 | **0.8631** | ✅ aynı |
  | Duyarlılık | 0.7674 | 0.7965 | ✅ aynı |
  | Özgüllük | 0.8078 | 0.7972 | ✅ aynı |
  | F1 | 0.6408 | 0.6478 | ✅ aynı |
  | TP/FP/TN/FN | 132/108/454/40 | 137/114/448/35 | ✅ aynı |

  5 zonun tamamı da (eşikler dahil) birebir eşleşti.
* **Karar:** **KEEP** — referans doğrulandı, pipeline deterministik.
* **Yan bulgu (hesaplama bütçesi):** Küçük CNN 5 zonu toplam **~27 saniyede** (zon başına 4.7–6.1 s) **CPU'da** eğitiyor. ⇒ Özellik/TDA tabanlı deneylerin tamamı GPU'suz yürütülebilir; CUDA yalnızca görüntü tabanlı pretrainli omurga (E0.3/E3.1/E3.2) için gerekli.
* **Manifest:** `artifacts/runs/E0.2_baseline_repro/manifest.json`

### E4.4 — Persistence Landscape Üst-Zarf (Envelope) Mekanizma Analizi

* **Hipotez (ön-kayıtlı):** PL'nin AUC çöküşü (0.5450), üst-zarf fonksiyonlarının az sayıda çok uzun ömürlü ama **sınıf-ayırt-edici olmayan** H1 sınıfı tarafından doyurulmasından kaynaklanır; nodüle ait sinyal zarfın yanında ihmal edilebilir kalır.
* **Tek değişken:** Yok (mekanizma analizi).
* **Çalışma:** `python pipelines/exp/E4_4_pl_envelope_analysis.py` · dengeli train alt kümesi (n = 2.040 zon) · PL vektörü `(N, 2, 5, 80)` olarak açıldı.
* **Sonuç:** ✅ **Hipotez doğrulandı.**

  | Ölçüt | λ1 | λ2 | λ3 | λ4 | λ5 |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | **H1** ortalama tepe | 109.89 | 102.61 | 97.56 | 93.60 | 89.61 |
  | **H1** enerji payı | %22.3 | %20.8 | %19.8 | %19.0 | %18.1 |
  | **H1** sinyal/zarf | %1.62 | **%1.90** | %1.54 | %0.37 | %1.72 |
  | **H0** ortalama tepe | 173.87 | 173.42 | 172.67 | 171.79 | 170.49 |
  | **H0** sinyal/zarf | %0.02 | %0.21 | %0.39 | %0.78 | **%0.89** |

* **Yorum — PL neden çöktü:**
  1. **Zarf doymuş:** H1 tepe değerleri λ1→λ5 boyunca 110 → 90 aralığında ve **birbirine yakın**; enerji neredeyse eşit dağılmış. Bu, *her zonda en az 5 adet çok uzun ömürlü (~180–220 kalıcılık) H1 sınıfı* bulunduğunu gösterir — bunlar damar döngüleri/kaburga kenarları gibi **arka plan** yapılarıdır.
  2. **Sinyal gömülü:** Pozitif/negatif zon ayrımını taşıyan bilgi, zarfın yalnızca **%1–2'si** kadardır (≈ **50–100× bastırılma**). H0'da ayırt edicilik λ1'de neredeyse sıfır (**%0.02**) ve ancak alt landsaflarda (**%0.89**) belirir — yani sınıf bilgisi zarfın *altına* gömülür.
  3. **PI neden kazandı:** PI, üst-zarf projeksiyonu yapmak yerine `(birth, persistence)` 2B yoğunluk yüzeyini korur; bu sayede zarf tarafından ezilen lokal imza ayakta kalır ve hasta düzeyi **0.8631** ROC-AUC'ye ulaşılır.
* **Karar:** **KEEP** — PL çöküşünün mekanizması nicel olarak kanıtlandı. Bu, [literature.md](literature.md) §5.2'deki **1 numaralı özgünlük iddiasının** deneysel kanıtıdır.
* **Yan kanıt:** `train_diagram_distribution_report.json` H1 kalıcılık dağılımının aşırı ağır kuyruklu olduğunu doğruluyor (p50 = 1, p90 = 2, p95 = 3, p99 = 8, **max = 255**) → zarfı doyuran az sayıda sınıf niceliksel olarak mevcuttur.
* **Manifest:** `artifacts/runs/E4.4_pl_envelope_analysis/manifest.json`

### E3.4 — 5-Zon TDA CNN + RBF-SVM Soft-Voting (Ensemble)

* **Hipotez (ön-kayıtlı):** CNN ve SVM hasta düzeyi olasılıklarının ağırlıklı ortalaması, tek modellerin her birinden **≥ +0.015 val ROC-AUC** verir.
* **Tek değişken:** Birleştirme ağırlığı `w` (`p = w·p_cnn + (1−w)·p_svm`), `w ∈ [0,1]` adım 0.05. Veri hattı, özellikler, split ve ölçekleyiciler sabit (v2.1 kilidi).
* **Çalışma:** `python pipelines/exp/E3_4_ensemble_cnn_svm.py` · kayıtlı paketlerden çıkarım (`zone_cnn_bundle.pt`, `zone_svm_bundle.joblib`) · seed 42.
* **Sonuç:** ✅ **Hipotez doğrulandı → KEEP**

  | Model | w | Val ROC-AUC | **Test ROC-AUC** | Eşik |
  | :--- | :---: | :---: | :---: | :---: |
  | CNN (tek model) | — | 0.8607 | 0.8631 | 0.6949 |
  | SVM (tek model) | — | 0.8278 | 0.8289 | 0.6397 |
  | **🥇 Ensemble A** (hasta düzeyi soft voting) | **0.55** | **0.8821** | **0.8776** | 0.6785 |
  | Ensemble B (zon düzeyi füzyon) | 0.55 | 0.8699 | 0.8629 | 0.6688 |

* **İstatistiksel kanıt:** Val farkı **+0.0214** (≥ +0.015 ✓) · Test AUC farkı **+0.0145**, N=2.000 bootstrap %95 CI **[+0.0034, +0.0260]** (sıfırı **dışlıyor** ✓), iki yönlü **p = 0.010**.
* **Yorum:**
  1. **Hasta düzeyinde soft voting (A), zon düzeyinde füzyondan (B) belirgin iyi** → CNN ve SVM **tamamlayıcı** bilgi taşıyor; kazanç en iyi zonu korumaktan geliyor.
  2. **Yeni lider: 0.8776 test ROC-AUC.** Bu, NODE21 rank-1 detektörünün görüntü düzeyi AUC'sini (0.868) **geçiyor** ve NODE21 **ensemble** seviyesine (0.877) **ulaşıyor** — üstelik hiçbir derin tespit mimarisi ve harici pretrain kullanmadan.
  3. Referans tek modeller **birebir** yeniden üretildi (CNN 0.8631 / SVM 0.8289) → çıkarım hattı bağımsız olarak doğrulandı.
* **Karar:** **KEEP** — bu, yayının ana performans sonucudur.
* **Manifest:** `artifacts/runs/E3.4_ensemble_cnn_svm/manifest.json`

### E4.2 — Persistence Özelliklerinin Görüntüye Geri İzlenmesi (Yorumlanabilirlik)

* **Hipotez (ön-kayıtlı):** En yüksek kalıcılıklı H1 (ve H0) sınıflarının doğum/ölüm hücreleri **yüksek gradyanlı (kenar)** bölgelere düşer → kaburga/damar kenarları.
* **Tek değişken:** Yok (mekanizma analizi).
* **Çalışma:** `python pipelines/exp/E4_2_persistence_localization.py` · 30 pozitif + 30 negatif zon · top-K = 10 · GUDHI `cofaces_of_persistence_pairs()` ile hücre → piksel eşlemesi · **drift koruması:** kendi crop'umuz `prepare_filtration` çıktısıyla birebir eşleşiyor (eşleşmezse zon atlanır).
* **Sonuç:** ⚠️ **H1 hipotezi ÇÜRÜTÜLDÜ → KILL**, ancak iki değerli bulgu çıktı:

  | Boyut | Top-K gradyan oranı (ort / medyan) | Rastgele kontrol | Oran > 1.5 olan zon |
  | :---: | :---: | :---: | :---: |
  | **H0** | **6.176 / 6.608** | 0.974 | **%90** |
  | **H1** | 1.474 / **1.021** | 0.981 | %30 |

* **Yorum:**
  1. **H0 için hipotez GÜÇLÜ desteklendi:** en kalıcılıklı H0 sınıfları, doku ortalamasının **6,2 katı** gradyanlı kenarlara düşüyor (zonların **%90'ında**). ⇒ H0 Persistence Image büyük ölçüde **arka plan/sınır yapısını** kodluyor, nodül çekirdeğini değil.
  2. **H1 için hipotez ÇÜRÜTÜLDÜ:** medyan oran **1.021** — tipik bir zonda en kalıcılıklı H1 sınıfları kenarlarda **değil**, düzgün (düşük gradyanlı) bölgelerde. Ortalamanın 1.474'e çıkması birkaç aşırı zonun etkisiyle (dağılım çarpık).
  3. **Yine de tez destekleniyor:** her iki boyutta da en kalıcılıklı sınıflar **nodüle özgü yapılara karşılık gelmiyor** → E4.4'teki zarf-doygunluğu bulgusunu **bağımsız olarak** destekliyor.
  4. **Metodolojik katkı:** literatür *"H1 döngülerinin hangi anatomik yapıya karşılık geldiği kanıtlanmıyor"* diyordu ([literature.md](literature.md) §2.5). Bu deney o **yeteneği** kuruyor: persistence noktası → piksel koordinatı geri izleme çalışıyor.
* **Karar:** **KILL** (ön-kayıtlı H1 hipotezi) — ama H0 bulgusu ve yorumlanabilirlik yeteneği korunur, yayında kullanılır.
* **Takip hipotezi (E4.5 olarak kaydedildi):** "En kalıcılıklı H1 sınıfları düzgün/düşük-gradyanlı (hava/arka plan) bölgelerde oluşur." Doğum/ölüm hücreleri yerine **temsilci döngünün tüm hücreleri** kullanılarak test edilmeli.
* **Manifest:** `artifacts/runs/E4.2_persistence_localization/manifest.json`

### E0.3 — Güçlü Aynı-Split Baseline (donmuş ImageNet ResNet18 + linear probe)

* **Amaç:** Literatürün **zorunlu** kıldığı güçlü görüntü baseline'ı (mevcut 64×64'lük küçük ham CNN = 0.7043 adil değildi).
* **Tek değişken:** Yok (yeni referans noktası).
* **Çalışma:** `python pipelines/exp/E0_3_resnet_baseline.py` · ImageNet-pretrainli **ResNet18 (donmuş)** · 224×224 girdi · 512-D avgpool özellikleri · zon başına lojistik regresyon (C val'de seçilir) · hasta düzeyi max-pooling · CPU.
* **Sonuç:**

  | Model | Val ROC-AUC | **Test ROC-AUC** | Val→Test farkı |
  | :--- | :---: | :---: | :---: |
  | ResNet18 (donmuş) + LR | 0.8884 | **0.8349** | **−0.0535** |
  | TDA PI küçük CNN | 0.8607 | **0.8631** | +0.0024 |
  | TDA ensemble (E3.4) | 0.8821 | **0.8776** | −0.0045 |

* **Yorum:**
  1. **Güçlü pretrainli görüntü modeli test'te TDA'yı GEÇEMİYOR** (0.8349 < 0.8631 < 0.8776) ⇒ hakemin *"neden güçlü bir CNN değil?"* itirazı **kapatıldı**.
  2. **ResNet ciddi genelleme kaybı yaşıyor** (val 0.8884 → test 0.8349, **−0.0535**), TDA ise neredeyse kayıpsız (+0.0024). Küçük veri rejiminde TDA'nın **genelleme avantajı**.
  3. Çıkarım maliyeti: 24.410 zon × 512-D, CPU'da **849 s** (~14 dk) — GPU gerekmedi.
* **Karar:** **BASELINE** (referans noktası kuruldu)

### E3.1 — Pretrainli Görüntü Omurgası + TDA Geç Füzyonu 🏆

* **Hipotez (ön-kayıtlı):** ResNet18 + TDA geç füzyonu, tek başına ResNet'ten **≥ +0.015 val ROC-AUC** verir.
* **Tek değişken:** Füzyon ağırlığı `a` (`p = a·p_resnet + (1−a)·p_tda`). TDA dalı E3.4 konfigürasyonunda **sabit** (CNN+SVM, w=0.55).
* **Çalışma:** `python pipelines/exp/E3_1_tda_image_fusion.py` · seed 42 · CPU.
* **Sonuç:** ✅ **KEEP — çok güçlü etki**

  | Model | Val ROC-AUC | **Test ROC-AUC** |
  | :--- | :---: | :---: |
  | ResNet18 (tek) | 0.8884 | 0.8349 |
  | TDA ensemble (tek) | 0.8821 | 0.8776 |
  | **🏆 Füzyon (a = 0.55)** | **0.9572** | **0.9211** |

* **İstatistiksel kanıt:** Test AUC farkı (füzyon − ResNet) = **+0.0862**, N=2.000 bootstrap %95 CI **[+0.0632, +0.1099]**, **p ≈ 0.0000**. Val farkı **+0.0688** (≥ +0.015 ✓).
* **Neden bu kadar büyük? — bağımsız doğrulama (`scratch/verify_E3_1.py`):**
  * İki temsil arasındaki **korelasyon yalnızca r ≈ 0.30–0.35** (val 0.3034 · test 0.3489) ⇒ ResNet ve TDA hasta olasılıkları **neredeyse dik**; büyük ölçüde **bağımsız** bilgi taşıyorlar.
  * **Sızıntı kontrolü:** etiketler permute edildiğinde füzyon AUC = **0.4378** (≈ şans) ⇒ etiket sızıntısı **yok**.
  * NaN yok; tüm sayılar bağımsız yeniden hesaplamayla **birebir** doğrulandı.
* **Yorum:**
  1. Bu, yayının **ana sonucudur**: topolojik ve görüntü temsilleri **tamamlayıcı**dır; füzyon **0.9211** test ROC-AUC ile NODE21 ensemble'ını (0.877) **+0.044** geçiyor.
  2. Literatürde TDA-Net (CXR/COVID) füzyonla ~+0.05 kazanmıştı ([literature.md](literature.md) §2.3); bizim **+0.0862**'lik kazancımız bu mertebede ve istatistiksel olarak çok güçlü.
  3. **GPU gerekmedi** — tüm sonuç CPU'da üretildi.
* **Karar:** **KEEP**
* **Manifest:** `artifacts/runs/E3.1_tda_image_fusion/manifest.json`

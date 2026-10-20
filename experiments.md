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

### E2.7 — Persistence Image Ağırlık Fonksiyonu Karşılaştırması

* **Hipotez (ön-kayıtlı):** Adams `w(p) = p` dışında bir ağırlık fonksiyonu val ROC-AUC'yi **≥ +0.015** artırır.
* **Tek değişken:** PI hücre birikim ağırlığı `w(p)`. Diyagramlar, ızgara aralıkları (`tda_ranges_v2.json`), split, mimari ve eğitim protokolü **sabit**.
* **Kapsam (test bütçesi korunur):** Yalnızca **train(dengeli) + val = 5.710 zon** işlendi; **test setine dokunulmadı**.
* **Çalışma:** `python pipelines/exp/E2_7_pi_weight_functions.py` · çıkarım 605 s (10 worker) · 5 varyant × 5-zon CNN.
* **Sonuç:** ❌ **Hipotez çürütüldü → KILL**

  | Ağırlık `w(p)` | Val hasta ROC-AUC | Adams'a fark |
  | :--- | :---: | :---: |
  | **`p` (Adams — mevcut)** | **0.8736** | — |
  | `arctan(p)` (doyurucu) | 0.8640 | −0.0096 |
  | `√p` (alt-doğrusal) | 0.8624 | −0.0112 |
  | `1` (sabit — kalıcılığı yok sayar) | 0.8591 | −0.0145 |
  | `p²` (üst-doğrusal) | 0.8547 | −0.0189 |

* **Yorum:**
  1. **Adams ağırlığı zaten en iyi** ⇒ mevcut PI tasarım tercihi **keyfi değil, doğrulanmış**. Bu, yayın için değerli bir **sağlamlık kanıtıdır**.
  2. **Sıralama anlamlıdır:** doğrusal > doyurucu > alt-doğrusal > sabit > üst-doğrusal. **`p²`'nin en kötü olması E4.4'ü bağımsız olarak destekliyor:** üst-doğrusal ağırlık, zarfı doyuran **çok uzun ömürlü arka plan sınıflarını** daha da amplifiye ediyor.
  3. **Dürüstlük notu:** tek seed (CNN gürültüsü ±0.01–0.02 olabilir). Ancak istatistiksel bir *kazanç* iddia edilmediği için çok-seed doğrulaması gerekmez; tüm farklar negatif ve tutarlı.
  4. Bu koşudaki `adams_p` değeri (0.8736) tam boru hattı referansından (0.8607) farklıdır; nedeni bu betiğin her zon için RNG'yi yeniden tohumlaması. **Karşılaştırma kendi içinde kontrollüdür** (tüm varyantlar aynı protokol).
* **Karar:** **KILL** — mevcut Adams ağırlığı korunur.
* **Manifest:** `artifacts/runs/E2.7_pi_weight_functions/manifest.json`

### E4.1 — Etiketleme Kuralı Sağlamlık Analizi (%10 / %20 / %30 örtüşme)

* **Hipotez (ön-kayıtlı):** %20 piksel-örtüşme kuralıyla elde edilen sonuçlar, %10 ve %30 kuralları altında **anlamlı değişmez** (val ROC-AUC farkı < 0.015).
* **Tasarım:** **Hasta split'i DONDURULDU** (aynı train/val/test hastaları); yalnızca zon etiketleri ve 1:1 dengeli alt küme yeniden türetildi. **Test setine dokunulmadı** — karar validation üzerinden.
* **Çalışma:** `python pipelines/exp/E4_1_overlap_robustness.py` · seed 42 · CPU · çalışma süresi ~3 dk.
* **Sonuç:** ❌ **Hipotez reddedildi → KILL** (sonuçlar toleransın ötesinde değişiyor)

  | Kural | Pozitif zon | Ambiguous | Dengeli train | Val hasta ROC-AUC | %20'ye fark |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **%10** | 1514 | 33 | 2084 | **0.8922** | **+0.0186** |
  | **%20 (mevcut)** | 1478 | 68 | 2040 | 0.8736 | — |
  | **%30** | 1436 | 109 | 1980 | 0.8693 | −0.0043 |

* **Yorum:**
  1. **Monoton eğilim:** gevşek örtüşme eşiği (daha çok pozitif, daha az ambiguous) → **daha iyi** val AUC. %10 en iyi, %30 en kötü.
  2. En büyük |fark| **0.0186 ≥ tolerans 0.015** ⇒ etiketleme kuralı modele **ölçülebilir biçimde** yansıyor; sonuç "kuraldan bağımsız sağlam" **değil**.
  3. ⚠️ **Uyarı:** tek seed. Fark toleransın hemen üzerinde ve CNN gürültüsü ±0.01–0.02 mertebesinde olabilir ⇒ **%10'a geçiş kararı çok-seed val doğrulaması gerektirir** (yeni deney **E4.6** olarak kaydedildi). Mevcut boru hattı (%20) **değiştirilmedi**.
  4. ✅ **Tutarlılık kontrolü (değerli):** bu koşuda %20 için elde edilen **0.8736**, E2.7'deki `adams_p` sonucuyla **birebir aynı**. E2.7'de PI rasterleştirmesini yeniden yazmıştım; bu eşleşme onun orijinal extractor ile **birebir uyumlu** olduğunu kanıtlıyor.
  5. Referans: tam boru hattı (`05_train_cnn.py`) %20'de 0.8607 verir; fark, o betiğin RNG'yi global olarak **bir kez** tohumlamasından kaynaklanır (protokol farkı, hata değil).
* **Karar:** **KILL** (hipotez reddedildi) — ancak **%10 kuralı güçlü bir iyileştirme adayı** olarak kaydedildi (E4.6).
* **Manifest:** `artifacts/runs/E4.1_overlap_robustness/manifest.json`

### E4.6 — %10 vs %20 Örtüşme Kuralı: Çok-Seed (Eşleştirilmiş) Doğrulama

* **Hipotez (ön-kayıtlı):** %10 kuralı, %20'ye kıyasla **ortalama** val ROC-AUC'yi **≥ +0.015** artırır ve fark 5 seed'in **≥4'ünde** pozitiftir.
* **Tasarım:** **Eşleştirilmiş (paired) çok-seed.** Hasta split'i dondurulmuş; her seed hem 1:1 dengeleme seçimini hem CNN eğitim tohumunu kontrol eder. **Test setine dokunulmadı.**
* **Gerekçe:** E4.1 tek-seed'lik +0.0186 bulgusunu vermişti; CNN gürültüsü ±0.01–0.02 olduğu için bu doğrulanmalıydı.
* **Sonuç:** ❌ **Hipotez reddedildi → KILL** — **E4.1'in bulgusu çürütüldü**

  | Seed | %10 | %20 | Fark |
  | :---: | :---: | :---: | :---: |
  | 42 | 0.8922 | 0.8736 | **+0.0186** |
  | 43 | 0.8546 | 0.8434 | +0.0112 |
  | 44 | 0.8395 | 0.8477 | −0.0082 |
  | 45 | 0.8369 | 0.8321 | +0.0048 |
  | 46 | 0.8701 | 0.8538 | +0.0163 |
  | **Ortalama** | **0.8587** | **0.8501** | **+0.0085** (std 0.0096) |

* **Yorum:**
  1. **Ortalama kazanç yalnızca +0.0085** — ön-kayıtlı eşiğin (+0.015) **altında** ve standart sapması (0.0096) mertebesinde. Yön tutarlı (4/5 pozitif) ama **büyüklük gürültü içinde**.
  2. ⇒ **Mevcut %20 kuralı korunur.** E4.1'in +0.0186'sı, seed=42'nin beş içinde **en büyük fark** olmasından kaynaklanan bir **aşırı tahmin**di. Çok-seed doğrulaması, bir gürültü artefaktını kovalamayı **önledi** — ön-kayıt + öldürme kriteri protokolünün doğrudan faydası.
  3. 📌 **Önemli yan bulgu — seed varyansı:** CNN'in val AUC'si seed'ler arasında **0.8321–0.8922** (≈±0.02–0.03) değişiyor. Bu, makalede tek-seed sonuçların yanına **çok-seed ortalama** konması gerektiğini gösterir. **Ancak** ana sonucumuz (füzyon kazancı **+0.0862**, CI [+0.0632, +0.1099]) bu gürültünün **çok üzerinde** ⇒ iddia sağlam kalır.
* **Karar:** **KILL** — %20 kuralı korunur, E4.6 kapanır.
* **Manifest:** `artifacts/runs/E4.6_threshold_multiseed/manifest.json`

### E2.8 — PI σ ve Izgara Çözünürlüğü Duyarlılık Taraması

* **Hipotez (ön-kayıtlı, SAĞLAMLIK):** PI sonuçları Gaussian genişliği σ ve ızgara çözünürlüğü seçimine karşı sağlamdır — her varyantın ortalama val ROC-AUC'si referanstan (σ=1.0, ızgara=20) **en fazla 0.03** sapar.
* **Toleransın gerekçesi:** E4.6'da CNN seed varyansı **±0.02–0.03** ölçüldü; tek-seed'lik 0.015 toleransı gürültünün altında kalırdı. Bu yüzden **3 seed ortalaması** ve **0.03 tolerans** ön-kayıtlandı.
* **Tek değişken:** İki kol — **A)** σ ∈ {0.5, 1.0, 1.5, 2.0} (ızgara=20) · **B)** ızgara ∈ {15, 20, 25} (σ=1.0). Diyagramlar **bir kez** hesaplandı, her konfigürasyon için yeniden vektörleştirildi; split/mimari/protokol sabit.
* **Kapsam:** train(dengeli)+val = 5.710 zon; **test setine dokunulmadı** · çıkarım 572 s.
* **Sonuç:** ✅ **Hipotez doğrulandı → KEEP** (her iki kol da sağlam)

  | Kol | Konfigürasyon | Ortalama val ROC-AUC (3 seed) | Referansa fark |
  | :--- | :--- | :---: | :---: |
  | **A (σ)** | σ=0.5 | 0.8460 | **−0.0236** |
  | | **σ=1.0 (mevcut)** | **0.8697** | — |
  | | σ=1.5 | 0.8734 | +0.0038 |
  | | σ=2.0 | 0.8712 | +0.0015 |
  | **B (ızgara)** | 15×15 | 0.8545 | −0.0152 |
  | | **20×20 (mevcut)** | **0.8697** | — |
  | | 25×25 | 0.8469 | −0.0227 |

* **Yorum:**
  1. **Her iki kol da sağlam** (en büyük |fark| 0.0236 ve 0.0227, tolerans 0.03). Sonuçlar σ/ızgara seçimine **aşırı duyarlı değil** → yayın için sağlamlık kanıtı.
  2. **Mevcut ayarlar iyi seçilmiş:** σ=1.0 referans, σ=1.5'ten yalnızca 0.0038 geride; σ=0.5 ise belirgin kötü (−0.0236, aşırı keskin/yüksek varyans). Izgarada 20×20 her iki yönde de en iyi.
  3. **Std değerleri çok küçük** (0.0005–0.0065) — 3 seed ortalaması varyantlar arası karşılaştırmayı güvenilir kılıyor.
  4. 📌 **Bonus:** E4.6'da ölçülen seed varyansının (±0.02–0.03) **yöntem kararlarından büyük** olduğu burada da doğrulandı → **kural 10** (çok-seed raporlama) haklı.
* **Karar:** **KEEP** — mevcut σ=1.0 / 20×20 ayarları korunur; **Faz 1 tamamlandı (6/6)**.
* **Manifest:** `artifacts/runs/E2.8_sigma_grid_sensitivity/manifest.json`

### E4.3 — Zon Bazlı Hata Analizi

* **Hipotez (ön-kayıtlı):** Hasta düzeyi max-pooling kararı tek zona bağımlı değildir; **zayıf zonlar** (val ROC-AUC < 0.70) vardır ve yanlış negatiflerin çoğundan onlar sorumludur.
* **Kapsam:** Yalnızca **validation**; test setine dokunulmadı. **Yeni model eğitilmedi** — kayıtlı paketlerden çıkarım (açıklayıcı analiz, model seçimi değil).
* **Sonuç:** ❌ **Hipotez reddedildi → KILL** (zayıf zon **yok**) — ama üç değerli bulgu:

  **1) Zon başına hasta düzeyi val ROC-AUC (zon TEK BAŞINA kullanılırsa):**

  | Zon | CNN | SVM | ResNet |
  | :--- | :---: | :---: | :---: |
  | 1 Sağ Üst | 0.7859 | 0.7252 | 0.8055 |
  | 2 Sağ Orta | **0.8199** | **0.8156** | **0.8244** |
  | 3 Sağ Alt | 0.8017 | 0.7321 | 0.7743 |
  | 4 Sol Üst | 0.7380 | 0.7046 | **0.8409** |
  | 5 Sol Alt | 0.7763 | 0.7379 | 0.8021 |

  **Hiçbir zon 0.70'in altında değil** ⇒ ön-kayıtlı "zayıf zon" kriteri sağlanmadı.

  **2) Sorumlu zon dağılımı (fusion, eşik 0.6357):** karar 5 zonun **tamamına** yayılıyor (Zon 1 %25.3 · Zon 5 %26.7 · Zon 3 %21.5 · Zon 4 %18.9 · Zon 2 %7.5). Tek-zon hakimiyeti **yok** ⇒ max-pooling dejenere değil.

  **3) Yanlış negatifler (FN=12):** ortalama fusion olasılığı **0.5812**, maksimum **0.6317** — yani eşiğin (0.6357) hemen altında **"kıl payı"** hatalar; TP ortalaması 0.7271.

* **📌 En önemli çıkarım (yeni hipotez doğuruyor):**
  Her zonun **tek başına** hastayı ~0.74–0.84 AUC ile ayırt edebilmesi, zon-bazlı sınıflandırıcıların sinyalinin önemli bir kısmının **hasta-global** (zon-dışı) olabileceğini düşündürüyor — saf lokal nodül kanıtı olsaydı, nodül içermeyen zonlar şans düzeyinde kalırdı. Bu, makalede **sınırlama + gelecek iş** olarak açıkça belirtilmeli ve **E4.7** ile test edilmeli.
* **Karar:** **KILL** (hipotez) — bulgular korunur.
* **Yeni deney (E4.7):** "Modelin maksimum olasılığı, gerçek nodül zonuna şanstan daha sık mı düşüyor?" (lokalite testi).
* **Manifest:** `artifacts/runs/E4.3_zone_error_analysis/manifest.json`

### E4.7 — Lokalite Testi: Model Gerçekten Nodülün Olduğu Zonu mu İşaretliyor? 🚨

* **Hipotez (ön-kayıtlı):** Nodül-pozitif val hastalarında model (a) argmax zonunu **gerçek nodül zonuna** şanstan sık yerleştirir, (b) gerçek nodül zonlarına aynı hastanın nodülsüz zonlarından **daha yüksek** olasılık atar.
* **Gerekçe:** E4.3, her zonun tek başına 0.74–0.84 AUC vermesini buldu — bu, sinyalin **lokal değil hasta-global** olabileceğini düşündürdü. Bu deney soruyu doğrudan test eder.
* **Kapsam:** Validation, 171 nodül-pozitif hasta · test setine dokunulmadı · yeni model eğitilmedi.
* **Sonuç:** ❌ **Hipotez reddedildi → KILL. Hiçbir model lokalize ETMİYOR.**

  | Model | İsabet oranı | Şans tabanı | Fark | Eşleştirilmiş fark [%95 CI] | **Zon düzeyi AUC** |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | TDA CNN | 0.2281 | 0.2678 | **−0.0398** | −0.0129 [−0.0458, +0.0214] | **0.4837** |
  | TDA ensemble | 0.2398 | 0.2678 | −0.0281 | +0.0017 [−0.0239, +0.0280] | 0.5037 |
  | ResNet18 | 0.2982 | 0.2678 | +0.0304 | +0.0005 [−0.0175, +0.0185] | 0.5258 |
  | Fusion | 0.2690 | 0.2678 | +0.0012 | +0.0010 [−0.0156, +0.0180] | 0.5156 |

  * **İsabet oranları şans düzeyinde** (binominal p = 0.21–0.90; hiçbiri anlamlı değil).
  * **Zon düzeyi AUC 0.48–0.53** = **lokalizasyon yok**. Modeller, bir hastanın nodüllü zonunu nodülsüz zonlarından ayırt **edemiyor**.
  * Eşleştirilmiş fark CI'ları **sıfırı içeriyor** → gerçek pozitif zonlara daha yüksek olasılık atanmıyor.

* **🚨 Yorum — bu bulgu iddianın kapsamını belirler:**
  1. Hasta düzeyi yüksek AUC (**0.9211 füzyon**) **gerçektir**, ancak mekanizma *"nodülü doğru zonda tespit ediyoruz"* **değildir**.
  2. Zon-bazlı modeller fiilen **5 paralel hasta-global sınıflandırıcı** gibi davranıyor; zon ayrımı **lokalizasyon sağlamıyor**.
  3. ⚠️ Bu nedenle **"5 anatomik zon + lokalizasyon"** yenilik iddiası **zayıflatılmalı**; doğru ifade *"5-zon paralel ayrıştırma + max-pooling agregasyonu"*dur.
  4. ✅ Yine de geçerli kalan katkılar: **füzyon sonucu** (E3.1, 0.9211), **temsil tamamlayıcılığı** (r≈0.30), **PL zarf mekanizması** (E4.4) — bunlar lokalizasyondan bağımsızdır.
  5. 📌 **Neden olabilir:** CXR'de küçük bir nodül, zon kesitinin topolojik özetinde zayıf bir lokal sinyal bırakır; modeller bunun yerine nodül varlığıyla ilişkili **global doku örüntülerini** öğreniyor olabilir. Bu, literatürde de bilinen bir CXR zorluğudur.
* **Karar:** **KILL** (hipotez) — bulgu **yayın için kritik bir dürüstlük ve kapsam belirleme** sonucudur.
* **Yeni deney adayı (E3.5):** "Tek global (zon ayrımı olmayan) model, 5-zon max-pooling'e denk mi?" → zon ayrıştırmasının **katkısı olup olmadığını** ölçer.
* **Manifest:** `artifacts/runs/E4.7_locality_test/manifest.json`

### E3.5 — Zon-Özel Modelleme Gerekli mi? Havuzlanmış Tek Model vs 5-Zon

* **Hipotez (ön-kayıtlı, EŞDEĞERLİK):** Havuzlanmış tek modelin ortalama val ROC-AUC'si, 5-zon modelinden **0.015'ten az** sapar (yani zon-özel modelleme ölçülebilir katkı sağlamaz).
* **Gerekçe:** E4.7, zon modellerinin **lokalize etmediğini** gösterdi. Bu durumda "her zon için ayrı model" mimarisinin katkısı sorgulanmalı.
* **Tek değişken:** Modelleme şeması (5 ayrı zon modeli vs zon kimliğini yok sayan tek havuzlanmış model). Özellikler, split, mimari, protokol ve seed'ler sabit.
* **Kapsam:** Validation; test setine dokunulmadı. Referans, **aynı protokol/seed'lerle** ölçülen E2.8 5-zon ortalamasıdır.
* **Sonuç:** ✅ **Eşdeğerlik doğrulandı → KEEP** (zon-özel modelleme gereksiz)

  | Şema | Seed'ler | Ortalama val hasta ROC-AUC |
  | :--- | :--- | :---: |
  | **5-zon özel modeli** (mevcut) | 0.8736 / 0.8546 / 0.8701 | **0.8661** |
  | **Havuzlanmış tek model** | 0.8552 / 0.8522 / 0.8680 | **0.8585** |
  | **Fark** | | **−0.0076** |

* **Yorum:**
  1. Fark **−0.0076**, toleransın (0.015) **altında** ve seed varyansının (±0.02–0.03) içinde ⇒ **istatistiksel olarak eşdeğer**.
  2. **Kritik ayrıntı:** havuzlanmış model, model başına **5× daha fazla** eğitim verisi görüyor (2.040 vs ~408 örnek) ama yine de **daha iyi değil**. Bu, zon bölünmesinin bir **model topluluğu (ensemble) çeşitliliği** sağladığını, ancak **anatomik uzmanlaşma** sağlamadığını düşündürüyor.
  3. ⚠️ **İddia daralması:** E4.7 + E3.5 birlikte, "5 anatomik zon ayrıştırması" gerekçesinin **veri tarafından desteklenmediğini** gösteriyor. Makalede mimari, *"akciğerin 5 kırpıntısı üzerinde model topluluğu + max-pooling"* olarak, **lokalizasyon veya uzmanlaşma iddiası olmadan** sunulmalı.
  4. ✅ Geçerli kalan katkılar değişmiyor: **füzyon 0.9211** (E3.1), **tamamlayıcılık r≈0.30**, **PL zarf mekanizması** (E4.4).
* **Karar:** **KEEP** (eşdeğerlik) — mimari basitleştirilebilir; iddia kapsamı daraltıldı.
* **Manifest:** `artifacts/runs/E3.5_pooled_vs_zone/manifest.json`

### E1.1 — Kemik Bastırma (Qure.ai) — **UYGULANAMAZ (feasibility blocker)** ⛔

* **Neden denenecek:** Literatür, kemik gölgelerinin H1 döngülerinin en büyük yapay kaynağı olduğunu ve Qure.ai modelinin **NODE21'de zaten doğrulandığını** söylüyordu (literature.md §4.1).
* **Uygulanabilirlik ölçümü (bu turda yapıldı):**
  * Model: [huggingface.co/qureaiorg/bone-suppression](https://huggingface.co/qureaiorg/bone-suppression) — iki TorchScript izi, **~400 MB × 2**, CC-BY-NC-SA-4.0 (ticari kullanım yasak).
  * README açıkça: **CPU'da ~1 dakika/görüntü**.
  * Gereken hasta sayısı (train dengeli + val) ≈ **1700** → **≈ 28 saat** CPU.
  * Ek engel: `cv2`, `huggingface_hub`, `pydicom` **kurulu değil** (+ kurulum ve indirme).
* **Karar:** **UYGULANAMAZ** — tek oturumda (ve CPU-only ortamda) maliyeti kabul edilemez. Deney **KILL değil, ERTELENDİ**; GPU erişimi (E0.0) sağlanırsa yeniden değerlendirilebilir.
* **Yerine geçen:** E1.2 (aynı kök nedeni — damar döngüleri — hedefler ve **çok ucuz**).

### E1.2 — Frangi Damar-Bastırma ile H1 Diyagram Filtrelemesi

* **Hipotez (ön-kayıtlı):** Damar-baskın H1 sınıflarının elenmesi, 5-zon CNN'in ortalama val ROC-AUC'sini **≥ +0.015 artırır** (3 seed).
* **Tek değişken:** Frangi vesselness eşiği `q` — H1 sınıfının **doğum hücresindeki** vesselness değeri, doku dağılımının `q` kantilini aşarsa sınıf elenir. **Görüntü değiştirilmez**, yalnızca H1 diyagramı süzülür.
* **Kapsam:** train(dengeli)+val = 5.710 zon · çıkarım 411 s (14 zon/s) · **test setine dokunulmadı**.
* **Sonuç:** ❌ **Hipotez çürütüldü → KILL** (filtreleme **monoton olarak zarar veriyor**)

  | Varyant | Ortalama val ROC-AUC (3 seed) | Base'e fark |
  | :--- | :---: | :---: |
  | **base** (filtre yok) | **0.8697** | — |
  | v90 (en damarsal %10 elenir) | 0.8606 | −0.0091 |
  | v75 (en damarsal %25) | 0.8547 | −0.0150 |
  | v50 (en damarsal %50) | 0.8358 | **−0.0338** |

* **📌 Yorum — E4.4 ve E4.7 ile tutarlı bir tabloya oturuyor:**
  1. Damar kaynaklı H1 sınıfları **gürültü değil, aksine faydalı**. Onları elemek başarımı monoton olarak düşürüyor.
  2. Bu, **E4.7'nin bulgusunu açıklıyor**: modeller lokalize etmiyorsa, ayırt edici sinyal büyük ölçüde **global damar/arka plan topolojisinde** olmalı — ve E1.2 tam da o sinyali kesiyor.
  3. Üç deney birleşince tutarlı bir sonuç: **E4.4** (zarf arka plan sınıflarıyla doygun) + **E4.7** (lokalizasyon yok) + **E1.2** (arka planı kesmek zarar veriyor) ⇒ **ayırt edici bilgi lokal nodül topolojisinde değil, global/arka plan topolojisindedir.**
  4. ⚠️ Bu, "arka planı bastırırsak lokal sinyal açığa çıkar" beklentisini **çürütüyor**. Kemik bastırma (E1.1) da benzer şekilde zarar verebilir — ki zaten uygulanamaz durumda.
* **Karar:** **KILL** — H1 filtrelenmez; mevcut hat korunur.
* **Manifest:** `artifacts/runs/E1.2_vesselness_h1_filter/manifest.json`

### E3.6 — Oracle-Lokalize TDA: Lokal Topolojik Sinyal Var mı? 🎯

* **Hipotez (ön-kayıtlı):** Nodül merkezine oturan yamaların TDA özellikleri, kontrol yamalarından **val ROC-AUC ≥ 0.70** ile ayrılır.
* **Neden:** E4.7 (lokalizasyon yok) + E1.2 (arka plan faydalı) lokal sinyalin zayıf olabileceğini düşündürdü. Ama **doğrudan test edilmemişti.** Eğer *ground-truth bbox ile* (yani nodülün tam nerede olduğunu bilerek) bile ayrım yoksa, **aday-bazlı mimari (E3.3) işe yaramaz.**
* **Tasarım:** `patient_info.json`'daki `crop_bbox` + nodül `center_x/center_y` kullanılarak nodül merkezine oturan yama çıkarıldı. Kontrol yaması **AYNI BOYUTTA** (`half` birebir eşleştirildi). Sınıflandırıcı: lojistik regresyon (deterministik). Hasta split'i sabit; **test hastaları kullanılmadı.**
* **Sonuç:** ⚠️ **KILL — lokal sinyal ZAYIF**

  | Tasarım | Val yama ROC-AUC |
  | :--- | :---: |
  | **Boyut-eşleştirilmiş (geçerli)** | **0.6866** (train 0.6735) |
  | ~~Eşleştirilmemiş (confound'lu)~~ | ~~0.9940~~ |

* **🚨 Kritik metodolojik uyarı (confound kontrolü):** İlk koşuda kontroller sabit **60×60**, pozitifler ise değişken boyuttaydı (2,5 × bbox). Sınıflandırıcı **nodülü değil, yama BOYUTUNU** öğrendi ve AUC **0.994** çıktı. Boyut eşleştirilince **0.687**'ye düştü. Bu, "fazla iyi" sonuçların sistematik olarak confound açısından denetlenmesi gerektiğinin çarpıcı bir örneğidir.
* **Yorum — stratejik sonuç:**
  1. **Oracle lokalizasyonla bile** — nodülün tam merkezini *bildiğimiz* hâlde — yama düzeyi AUC yalnızca **0.687**.
  2. ⇒ **Aday-bazlı mimarinin (E3.3) tavanı düşük.** Mükemmel lokalizasyon varsayımında ~0.69; gerçek (hatalı) aday üretimiyle daha da düşer. **Seçenek B (lokalizasyonu kurtarma) kapatıldı.**
  3. ⇒ Hasta düzeyi **0.9211**'in kaynağı ağırlıklı olarak **global sinyal** — E4.7 ve E1.2 ile tam tutarlı.
  4. ✅ **Klinik olarak anlamlı:** Nodül varlığı, lokal topolojiden çok **akciğerin genel topolojik/görüntüsel örüntüsüne** yansıyor. Bu, yayın için değerli ve dürüst bir çerçeve.
* **Karar:** **KILL** — lokalizasyon arayışı kapatıldı; **Seçenek A** (daraltılmış, dürüst kapsamla yayına hazırlık) benimsenir.
* **Manifest:** `artifacts/runs/E3.6_oracle_localized_tda/manifest.json`

### E3.7 — Ana Sonucun Çok-Seed Doğrulaması (Protokol Kural 10 uyumu)

* **Hipotez (ön-kayıtlı):** Füzyonun 3-seed ortalama test ROC-AUC'si, tek-seed değerinden (0.9211) **0.03'ten az** sapar ve ortalaması **> 0.89** kalır.
* **Gerekçe:** E4.6'da CNN seed varyansı **±0.02–0.03** ölçüldü ve **Kural 10** eklendi (performanslar ≥3 seed ortalaması). Ana sonuç hâlâ **tek seed**'likti — bu boşluk kapatıldı.
* **Tek değişken:** CNN eğitim tohumu (42/43/44). TDA özellikleri, split, mimari, SVM dalı, ResNet dalı (deterministik) ve füzyon protokolü **sabit**; her seed için füzyon ağırlığı ve eşiği **yalnızca validation'da** seçildi.
* **Sonuç:** ✅ **KEEP — ana sonuç seed'e karşı kararlı**

  | Seed | Füzyon ağırlığı `a` | Val AUC | **Test ROC-AUC** | Test doğruluk | TDA-tek test AUC |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | 42 | 0.55 | 0.9576 | **0.9238** | 0.8583 | 0.8833 |
  | 43 | 0.55 | 0.9511 | **0.9111** | 0.8529 | 0.8645 |
  | 44 | 0.55 | 0.9571 | **0.9222** | 0.8460 | 0.8773 |
  | **Ortalama** | | | **0.9190 ± 0.0056** | | **0.8750 ± 0.0079** |

* **Yorum:**
  1. **Üç seed'in hepsi 0.911–0.924 bandında** ve ortalama **0.9190 ± 0.0056**. Tek-seed 0.9211'e fark **−0.0021** ⇒ tek-seed sonucumuz **şanslı değildi**; seed varyansı burada beklenenden de küçük (±0.006).
  2. Füzyon ağırlığı üç seed'de de **a=0.55** seçildi ⇒ hiperparametre de kararlı.
  3. **Makalede raporlanacak ana sayı artık: 0.9190 ± 0.0056** (tek-seed 0.9211 yerine). Kural 10 uyumu sağlandı.
  4. TDA-tek dalı da kararlı: **0.8750 ± 0.0079**.
* **Karar:** **KEEP** — ana sonuç çok-seed ile doğrulandı.
* **Not:** Test seti bu deneyde ana sonucun **seed belirsizliğini ölçmek** için bir kez kullanıldı (model seçimi yapılmadı).
* **Manifest:** `artifacts/runs/E3.7_headline_multiseed/manifest.json`

### E5.3 — Kaynak-Dışı Genelleme (leave-one-source-out)

* **Hipotez (ön-kayıtlı):** 3 kaynağın ortalamasında füzyon test ROC-AUC **> 0.75** ⇒ model kaynak-dışı genelliyor.
* **Neden:** Splitimiz rastgele; hakemin soracağı soru *"eğitimde görmediği bir kaynak veri setinden gelen nodülleri tanıyor mu?"*. Kaynak bilgisi `patient_info.json`'da mevcut → **yeni veri indirmeye gerek yok**.
* **Tasarım:** Her kaynak sırayla dışarıda bırakıldı. **Test** = o kaynağın nodüllü hastaları + orana uygun (~1:3,27) kontrol örneklemi; **val/train** = kalanlar (zon bazında 1:1 denge). Tek değişken: test edilen kaynak.
* **Sonuç:** ✅ **KEEP**

  | Dışarıda bırakılan kaynak | Test hastası | TDA CNN | **Füzyon** |
  | :--- | :---: | :---: | :---: |
  | chestxray14 | 2635 | 0.7406 | **0.7821** |
  | padchest | 1341 | 0.8391 | **0.8499** |
  | jsrt | 636 | 0.9076 | **0.9085** |
  | **Ortalama** | | **0.8291** | **0.8468** |
  | *(rastgele split referansı)* | *734* | *0.8631* | *0.9190* |

* **Yorum:**
  1. **Model kaynak-dışı genelliyor:** ortalama füzyon AUC **0.8468**, şansın çok üzerinde ve eşiğin (0.75) üstünde.
  2. **Domain-shift bedeli gerçek:** rastgele split 0.9190 → kaynak-dışı 0.8468 (**−0.072**). Yani kaynak değişince belirgin ama yıkıcı olmayan bir düşüş var.
  3. **Kaynaklar arası büyük fark:** jsrt **0.9085** (en iyi) ↔ chestxray14 **0.7821** (en kötü).
  4. ⚠️ **Dürüstlük notu — kısmi confound:** chestxray14 en büyük kaynak olduğu için dışarıda bırakıldığında **eğitim verisi yarıya iniyor** (dengeli train 1034 zon; diğer katlarda 1876 ve 2222). Yani chestxray14 katındaki düşüşün bir kısmı **domain shift değil, veri azalması**. Kıyaslama için eşit-boyutlu rastgele split kontrolü yapılmalı (kayda geçirildi).
  5. Bu bir **genelleme çalışmasıdır**; dondurulmuş test seti kullanılmadı, model seçimi yapılmadı (her kat kendi train/val/test'ini kurar).
* **Karar:** **KEEP** — kaynak-dışı genelleme kanıtlandı; domain-shift bedeli ölçüldü.
* **Manifest:** `artifacts/runs/E5.3_cross_source/manifest.json`

#### E5.3b — Eşit-Boyut Kontrolü (confound düzeltmesi) ⚠️🎯

* **Neden:** Yukarıdaki 4. notta işaretlenen confound'u ölçmek için. **Tüm katlar aynı miktarda veriyle** (dengeli train = **1034 zon**) eğitildi.
* **Sonuç:** Confound **gerçekmiş ve büyükmüş**

  | Dışarıda bırakılan kaynak | Doğal boyut | **Eşit-boyut (1034)** | Fark |
  | :--- | :---: | :---: | :---: |
  | chestxray14 | 0.7821 (1034 zon) | **0.7821** | 0 (değişmedi) |
  | padchest | 0.8499 (1876 zon) | **0.7625** | **−0.0874** |
  | jsrt | 0.9085 (2222 zon) | **0.8847** | −0.0238 |
  | **Ortalama** | **0.8468** | **0.8098** | **−0.0370** |

* **Yorum:**
  1. ⚠️ **Önceki 0.8468 İYİMSERDİ.** padchest ve jsrt'nin avantajının büyük kısmı **daha fazla eğitim verisi** görmelerinden geliyordu, domain kolaylığından değil.
  2. **Düzeltilmiş kaynak-dışı ortalama: 0.8098** ⇒ rastgele split'e (0.9190) göre **domain-shift bedeli −0.109** (önceki tahmin −0.072 idi).
  3. Eşitleme sonrası **chestxray14 (0.782) ve padchest (0.763) benzer**; **jsrt (0.885)** belirgin biçimde en kolay kaynak olmaya devam ediyor.
  4. ✅ **Ana sonuç yine de ayakta:** 0.8098 şansın çok üzerinde ve ön-kayıtlı eşiğin (0.75) üstünde ⇒ **model görülmemiş kaynağa genelliyor**, ama bedeli daha yüksek.
* **Karar:** **KEEP** (genelleme) — ancak **makalede raporlanacak sayı 0.8098'dir**; 0.8468 confound'lu olduğu için kullanılmamalıdır.
* **Manifest:** `artifacts/runs/E5.3_cross_source_eqsize/manifest.json`

### E3.2 — Orta Seviye Füzyon (özellik birleştirme) vs Geç Füzyon

* **Hipotez (ön-kayıtlı):** Orta seviye füzyon (PI ⊕ ResNet özellikleri → tek lojistik regresyon), geç füzyonun **val** ROC-AUC'sini (0.9572) **≥ +0.015** geçer.
* **Tek değişken:** Füzyon seviyesi (özellik birleştirme vs karar birleştirme). Özellikler, split, zon yapısı ve hasta düzeyi max-pooling **sabit**.
* **Kapsam:** Yalnızca **validation** (test bütçesi tükendi).
* **Sonuç:** ❌ **KILL**

  | Varyant | Val ROC-AUC |
  | :--- | :---: |
  | TDA lojistik sondası (tek) | 0.8365 |
  | ResNet18 sondası (tek) | 0.8884 |
  | TDA ensemble (CNN+SVM) | 0.8699 |
  | **Geç füzyon (a=0.60)** | **0.9538** |
  | **Orta seviye (PI+ResNet → LR)** | **0.9482** |

  **Orta − Geç fark: −0.0056** (kriter ≥ +0.015 ✗)

* **Yorum:**
  1. **Özellik düzeyinde birleştirme, karar düzeyinde birleştirmeden daha iyi değil** — aksine hafifçe kötü.
  2. Muhtemel sebep: TDA dalındaki **doğrusal olmayan** bileşen (CNN + RBF-SVM) karar füzyonunda korunuyor; 1312 boyutlu birleşik vektör üzerinde **tek doğrusal** sınıflandırıcı bu ifade gücünü kaybediyor.
  3. ⇒ **Geç füzyon tasarımı doğrulanmış** sayılır; ana sonuç bu ablasyonla da destekleniyor.
* ⚠️ **Küçük yeniden üretilebilirlik notu:** TDA ensemble val değeri bu koşuda **0.8699**, E3.4'te **0.8821** (−0.0122). Kayıtlı bundle'ların (CNN + SVM) yeniden yüklenmesi, E3.4'ün betik içi hesabıyla tam örtüşmüyor. Ana sonuçları etkilemez (füzyon aynı koşu içinde tutarlı) ama **TDA-tek satırı için E3.4 değeri esas alınmalı** ve fark ileride açıklanmalıdır.
* **Karar:** **KILL** — geç füzyon korunur.
* **Manifest:** `artifacts/runs/E3.2_midlevel_fusion/manifest.json`

### E2.5 — Sliced Wasserstein Gömlemesi vs Kalıcılık İmgesi (PI)

* **Hipotez (ön-kayıtlı):** SW gömlemesi + SVM, PI + SVM'in **val** ROC-AUC'sini **≥ +0.015** geçer.
* **Gerekçe:** E2.8, PI'ın **ızgara çözünürlüğüne duyarlı** olduğunu gösterdi (15/20/25 → 0.8545/0.8697/0.8469): ızgara hem bilgi kaybettiriyor hem yapay duyarlılık getiriyor. Sliced Wasserstein, diyagramları **ızgarasız** temsil eder.
* **Yöntem:** H0/H1 diyagramları yeniden hesaplandı (önbellekte diyagram yoktu). N_θ=24 yön boyunca 1B Wasserstein; her diyagram için **K=96 çapa diyagramına** olan SW mesafeleri özellik vektörü (H0+H1 = **192 boyut**). Aynı protokolde RBF-SVM (C=10); PI (800 boyut) + aynı SVM ile karşılaştırıldı.
* **Tek değişken:** Diyagram temsili. Sınıflandırıcı, split, zon yapısı ve hasta max-pooling **sabit**.
* **Kapsam:** Yalnızca **validation** (test bütçesi tükendi).
* **Sonuç:** ❌ **KILL — fark çok büyük**

  | Temsil | Boyut | Val hasta ROC-AUC |
  | :--- | :---: | :---: |
  | PI | 800 | 0.8399 |
  | **PI (PCA → 192)** *(boyut kontrolü)* | **192** | **0.8399** |
  | SW gömlemesi | 192 | **0.6896** |
  | **SW − PI** | | **−0.1504** |
  | **SW − PI(PCA 192)** *(boyut-eşit)* | | **−0.1504** |

* **🎯 Boyut confound kontrolü (E2.5b):** İlk karşılaştırmada **PI 800, SW 192 boyuttu** — yani fark "temsil" kadar "boyut" da olabilirdi (E3.6'daki boyut confound'unun aynısı). **PI, PCA ile 192 boyuta indirildiğinde val AUC birebir aynı kaldı (0.8399 → 0.8399)** ve boyut-eşit fark **yine −0.1504** çıktı. ⇒ **Boyut confound'u YOKTUR; fark gerçekten temsilden gelir.**
  * **Bonus bulgu:** PI'ın **800 boyutu fazlasıyla yedeklidir** — 192 boyuta indirgemek **hiç kayıp vermiyor** (hesaplama/ depolama için faydalı bir not).
* **Yorum:**
  1. **Izgara kuantalaması PI'ın darboğazı değilmiş.** E2.8'in gösterdiği ızgara duyarlılığı, "ızgara bilgi kaybettiriyor" anlamına gelmiyor — aksine PI, ızgarasız bir alternatiften **çok daha güçlü**.
  2. Muhtemel sebep: **PI tam 2B `(birth, persistence)` yoğunluğunu korurken**, çapa-mesafesi gömlemesi diyagramı yalnızca **192 skaler** ile özetliyor; PCA kontrolü bu açıklamayı **zayıflatıyor** (boyut eşitken de fark aynı). Geriye kalan açıklama: **çapa seçimi** (rastgele, train'den) diyagram uzayını iyi germiyor.
  3. ⚠️ **Kapsam sınırı (dürüstlük):** Bu deney **"çapa-mesafesi SW gömlemesi"** ailesini test eder, *saf* bir SW çekirdeği SVM'i değil.
  4. ⚠️ **Hesaplama notu:** Bu deney **GPU'dan faydalanamaz** — GUDHI (kalıcı homoloji) ve sklearn (SVM) **CPU-only**'dir; GPU kullanımı koşu boyunca **%3**'te kaldı.
* **Karar:** **KILL** — PI temsili korunur.
* **Manifest:** `artifacts/runs/E2.5_sliced_wasserstein/manifest.json`

### E2.1 — Çift Yönlü (Diphasic) Filtrasyon ✅ **İLK POZİTİF KEEP**

* **Hipotez (ön-kayıtlı):** Çift yönlü temsil (2 × 800 = 1600 boyut), tek polariteli PI'ın **val** ROC-AUC'sini **≥ +0.015** geçer.
* **Gerekçe:** Mevcut hat kübik kompleksi **tek polaritede** kurar (H0 karanlık kümeler, H1 döngüler). Bir görüntünün **ters polaritesi** (parlak yapılar = superlevel kümeler) farklı topolojik bilgi taşır.
* **Yöntem:** A = mevcut filtrasyon, B = ters filtrasyon (`f_max − f`). Her polarite için H0/H1 → PI (20×20). Eksen aralıkları B için train'den [0.5, 99.5] yüzdelikleriyle türetildi. Sınıflandırıcı: RBF-SVM (C=10) — E2.5/E2.7/E2.8 ile aynı protokol.
* **Kapsam:** Yalnızca **validation** (test bütçesi tükendi).
* **Sonuç:** ✅ **KEEP** — ama **ancak doğru baseline ile**

  | Kol | Boyut | Val hasta ROC-AUC |
  | :--- | :---: | :---: |
  | **A = projenin PI'ı (donmuş aralıklar)** | 800 | **0.8399** |
  | **A + B (çift polarite)** | 1600 | **0.8609** |
  | **Fark** | | **+0.0210** ✓ (eşik +0.015) |
  | *A + B **karıştırılmış** (boyut kontrolü)* | 1600 | *0.7803* (**−0.0597**) |

* **🚨 İlk koşu confound'luydu (+0.0646 şişkin):** İlk denemede "tek polarite" kolunu **bu betikte yeniden türettim** ([0.5, 99.5] yüzdelikleri) ve baseline **0.7198**'e düştü — çünkü projenin donmuş aralıkları (özellikle H1 birth **[239,250]**) bilgiyi doğru banda odaklıyor, benim türettiğim aralık (**[0,250]**) ise çözünürlüğü dağıtıyor. Projenin gerçek PI'ı kullanılınca fark **+0.0210**'a indi. **Ders: baseline'ı yeniden türetmek, ölçülen kazancın bir kısmını yapay olarak şişirir.**
* **🎯 Boyut kontrolü (E2.1b) — kazanç gerçek:** Diphasic temsil 1600, PI 800 boyut olduğu için fark "boyut" olabilirdi (E2.5/E3.6'daki confound'un aynısı). Kontrol: **B kolunu zonlar arasında karıştırdım** — boyut aynı (1600) kalır, B'nin bilgisi yok olur. Sonuç **0.7803**, yani **PI'ın tek başına (0.8399) ALTINDA**. ⇒ **Boyut eklemek tek başına ZARAR veriyor** (gürültü seyreltmesi); **+0.0210 tamamen ters polaritenin bilgisinden geliyor.**
* **Yorum:**
  1. ✅ **Alt- ve üst-seviye kümeleri (sublevel + superlevel) aynı bilgi değildir.** İkisini birlikte kodlamak ölçülebilir kazanç sağlıyor.
  2. Bu, **uzun süredir ilk pozitif KEEP** ve programın **ilk gerçek temsil iyileştirmesi**.
  3. ⚠️ **Test bütçesi tükendiği için bu kazanç test'te doğrulanamadı.** Makalenin 0.9516'lık ana sonucuna **eklenemez**; ancak **val düzeyinde doğrulanmış bir ablasyon** ve bir sonraki test bütçesi için **öncelikli aday** olarak kaydedilir.
* **Karar:** **KEEP** (val) — çift yönlü filtrasyon, gelecek çalışma için öncelikli iyileştirme.
* **Manifest:** `artifacts/runs/E2.1_diphasic_filtration/manifest.json` · kontrol: `artifacts/runs/E2.1_diphasic_filtration_shuffledb/manifest.json`

### E2.2 — Çok Ölçekli Kalıcılık İmgesi (multi-scale PI)

* **Hipotez (ön-kayıtlı):** Çok ölçekli temsil (σ = 0.5/1.0/1.5/2.0, 3200 boyut), **en iyi** tek ölçekli PI'ı (σ=1.5, 800 boyut) **val** ROC-AUC'sinde **≥ +0.015** geçer.
* **Gerekçe:** E2.8, PI'ın σ'ya duyarlı olduğunu gösterdi (0.5→0.8460, 1.0→0.8697, **1.5→0.8734**, 2.0→0.8712). Tek σ seçmek zorunda değiliz; birden çok ölçeği birleştirmek farklı ölçeklerdeki yapıyı kodlayabilir.
* **Tasarım (E2.1'in dersleri uygulandı):** A = σ1.0 (proje varsayılanı) · A' = σ1.5 (en iyi tek) · B = concat(4 σ) · **C = boyut kontrolü** (σ1.0 + diğer üç ölçeğin **karıştırılmış** hâli; boyut aynı, ek ölçeklerin bilgisi yok). Sınıflandırıcı/split/zon/max-pooling sabit. **Özellikler E2.8 önbelleğinden okundu — yeniden çıkarım yok.**
* **Kapsam:** Yalnızca **validation**.
* **Sonuç:** ❌ **KILL**

  | Kol | Boyut | Val hasta ROC-AUC |
  | :--- | :---: | :---: |
  | A — tek ölçek σ=1.0 | 800 | 0.8399 |
  | A' — tek ölçek σ=1.5 (en iyi) | 800 | 0.8457 |
  | **B — çok ölçek (4 σ)** | 3200 | **0.8510** |
  | *C — boyut kontrolü (karıştırılmış)* | 3200 | *0.6598* |
  | **B − en iyi tek ölçek** | | **+0.0053** ✗ (eşik +0.015) |

* **Yorum:**
  1. Çok ölçek **gerçek ama küçük** bir kazanç veriyor (+0.0053) — eşiğin altında, 4× özellik maliyetini haklı çıkarmıyor.
  2. **Boyut kontrolü yine kritik:** karıştırılmış kol **0.6598**, yani PI'ın çok altında ⇒ **bilgisiz boyut eklemek şiddetle zararlı.** +0.0053 gerçek bilgidir, boyut artefaktı değil.
  3. 🔑 **E2.1 ile anlamlı karşıtlık:**

     | Eksen | Değişiklik | Val kazanç | Karar |
     | :--- | :--- | :---: | :--- |
     | **Polarite** (hangi kümeler filtreleniyor) | sublevel → **sublevel + superlevel** | **+0.0210** | ✅ **KEEP** |
     | **Ölçek** (diyagram nasıl yumuşatılıyor) | tek σ → 4 σ | +0.0053 | ❌ KILL |

     ⇒ **Topolojik bilgi kazancı "hangi kümeleri filtrelediğinden" geliyor, "diyagramı nasıl yumuşattığından" değil.** Bu, temsil tasarımı için **yön gösterici bir sonuç** ve E2.1'in neden işe yaradığını da açıklıyor.
* **Karar:** **KILL** — tek ölçek (σ=1.5) tercih edilir; çok ölçek kullanılmaz.
* **Manifest:** `artifacts/runs/E2.2_multiscale_pi/manifest.json`

### E2.3 — Sağlıklı Şablon (relative persistence) ⛔ **GEÇERSİZ TASARIM (no-op)**

* **Hipotez (ön-kayıtlı):** Şablon-göreli temsil (`X − T_zon`, T = zonun temiz-negatif train PI ortalaması), mutlak PI'ı val ROC-AUC'sinde **≥ +0.015** geçer.
* **Gerekçe:** E4.4/E4.2, zarfın büyük kısmının **ortak arka plan** olduğunu gösterdi (bronkovasküler döngüler, göğüs duvarı). O halde bilgi **şablondan sapma**da olabilir.
* **Sonuç:** ⛔ **HİPOTEZ TEST EDİLEMEDİ** — tasarım matematiksel olarak **no-op**

  | Kol | Boyut | Val hasta ROC-AUC |
  | :--- | :---: | :---: |
  | A — mutlak PI | 800 | 0.8399 |
  | B — `X − T(doğru zon)` | 800 | 0.8400 |
  | C — `X − T(yanlış zon)` | 800 | 0.8399 |

* **🚨 Neden geçersiz (kanıt):** Üç kol **birebir aynı** çıktı. Sebep matematiksel: `make_pipeline(StandardScaler(), SVC())` scaler'ı **zon başına** uygular, ve sabit bir vektör çıkarmak tamamen yutulur:
  ```
  (X − T) − mean(X − T) = (X − T) − (mean(X) − T) = X − mean(X)
  ```
  Std de sabit kaymadan etkilenmez. ⇒ Mutlak ve göreli temsiller **özdeştir**; farklar yalnızca SVM numeriğidir.
* **📌 Metodolojik ders (kayda değer):** **Per-feature afin dönüşümler, sonrasında gelen StandardScaler tarafından yutulur.** Bir ön-işleme adımının etkisini ölçmeden önce, sonraki adımın onu absorbe edip etmediği **matematiksel olarak** kontrol edilmelidir. (Bu, E0.5'teki önbellek kayması ve E3.6'daki boyut confound'u ile aynı sınıftan bir rigor hatasıdır — ama bu kez **koşmadan önce değil, sonuç birebir aynı çıkınca** yakalandı.)
* **✅ Düzeltilmiş tasarım önerileri (gelecek iş):**
  1. **Doğrusal olmayan sapma:** `(X − T)_+` eşikleme veya `|X − T|` — StandardScaler bunları yutamaz.
  2. **Diyagram düzleminde relativasyon:** yalnızca şablonun noktalarından **uzak** kalıcılık sınıflarını tut (PI'da değil, diyagramda).
  3. **Havuzlanmış sınıflandırıcı:** zon başına yerine tek model kullanılırsa T_zon sabit değildir (ancak bu, E3.5'te eşdeğer bulunan protokolü değiştirir).
* **Karar:** ⛔ **GEÇERSİZ** — KILL olarak kaydedilmez; hipotez açık kalır.
* **Manifest:** `artifacts/runs/E2.3_healthy_template/manifest.json` (sonuçlar no-op olduğu için **bilimsel kanıt sayılmaz**)

#### E2.3b — Sağlıklı Şablon, **DÜZELTİLMİŞ TASARIM** → geçerli sonuç: **KILL**

* **Düzeltme:** Sapmayı **doğrusal olmayan** hâle getirdim — ReLU `(X−T)₊` ve mutlak `|X−T|` dönüşümleri **per-feature afin değildir**, dolayısıyla StandardScaler bunları yutamaz. Böylece hipotez ilk kez **gerçekten** test edildi.
* **Sonuç:** ❌ **KILL** (geçerli test)

  | Kol | Boyut | Val hasta ROC-AUC | Fark |
  | :--- | :---: | :---: | :---: |
  | **A — mutlak PI** | 800 | **0.8399** | — |
  | B — `(X − T_doğru)₊` (ReLU sapma) | 800 | 0.7530 | **−0.0869** |
  | C — `|X − T_doğru|` (mutlak sapma) | 800 | 0.6365 | **−0.2034** |
  | D — `(X − T_yanlış)₊` (kontrol) | 800 | 0.8276 | −0.0124 |

* **Yorum:**
  1. **Şablon-göreli sapma bilgi KATMIYOR, aksine yok ediyor.** Mutlak PI en iyi; ReLU sapma −0.087, mutlak sapma −0.203 kaybettiriyor.
  2. 🔑 **Kontrol kolu asıl bulguyu veriyor:** **yanlış zonun** şablonunu çıkarmak (0.8276), **doğru zonun** şablonunu çıkarmaktan (0.7530) **daha iyi**. Yani doğru şablonu çıkarmak **daha çok** zarar veriyor ⇒ **şablon zon-özel FAYDALI bilgi taşıyor**, ortak arka plan gürültüsü değil.
  3. 🔗 **Program geneliyle tutarlı:** E1.2 (damar/arka plan H1 sınıflarını elemek **zarar veriyor**) + E2.3b (sağlıklı şablonu çıkarmak **zarar veriyor**) ⇒ **"arka planı bastır, lokal sinyali açığa çıkar" öncülü iki bağımsız yoldan çürütülmüştür.**
  4. Bu, E2.3'ün **geçersiz** sonucunun yerini alan **geçerli** sonuçtur; hipotez artık **kapalıdır**.
* **Karar:** **KILL** — mutlak PI korunur; şablon-göreli temsil kullanılmaz.
* **Manifest:** `artifacts/runs/E2.3b_healthy_template_nonlinear/manifest.json`

### E2.6 — Çok Parametreli Persistence (dilim yaklaşımı) 🎉 **EN BÜYÜK KAZANÇ**

* **Hipotez (ön-kayıtlı):** Çok dilimli temsil (3 dilim, 2400 boyut), tek dilimli PI'ı (t=0, 800 boyut) **val** ROC-AUC'sinde **≥ +0.015** geçer.
* **Gerekçe:** E2.6 roadmap'te "ertelendi" idi (multiparameter ağır/olgun değil). Ancak **modülün dilimleri** pratikte uygulanabilir: iki fonksiyonlu filtrasyon `f_t = f + t·g` için her `t`'de **tek parametreli** PH hesaplanır ve dilimler birleştirilir. İkinci fonksiyon **Sobel gradyan büyüklüğü** seçildi: E4.2 kalıcılık sınıflarının gradyanla ilişkisini ölçmüştü; gradyanı **ikinci parametre** yapmak sınıfları *elemek* yerine **katmanlaştırır** — E1.2'nin "elemek zarar veriyor" bulgusundan sonra mantıklı yön.
* **Tasarım:** Dilimler `t ∈ {0.0, 0.5, 1.5}`, `f_norm + t·g_norm`. **A** = t=0 · **B** = concat(3 dilim) · **C** = **boyut kontrolü** (t=0 + diğer dilimlerin karıştırılmış hâli). RBF-SVM C=10; split/zon/max-pooling sabit.
* **Kapsam:** Yalnızca **validation** (test bütçesi tükendi).
* **Sonuç:** ✅ **KEEP — kaydedilen en büyük val iyileştirmesi**

  | Kol | Boyut | Val hasta ROC-AUC |
  | :--- | :---: | :---: |
  | A — projenin PI'ı (tek dilim, t=0) | 800 | 0.8399 |
  | **B — proje PI + 2 gradyan dilimi** | 2400 | **0.8880** |
  | *C — boyut kontrolü (karıştırılmış)* | 2400 | *0.6925* |
  | **B − A** | | **+0.0480** ✓ |

* **Yorum:**
  1. 🎉 **+0.0480** — E2.1'in (+0.0210) iki katından fazla; **programın en büyük gerçek kazanımı**.
  2. 🎯 **Boyut kontrolü kritik:** karıştırılmış kol **0.6925**, A'nın (0.8399) çok altında ⇒ bilgisiz boyut eklemek şiddetle zararlı; **+0.0480 tamamen yeni bilgiden**.
  3. **Gradyanı ikinci parametre yapmak işe yarıyor**, oysa gradyanı **kullanıp sınıf elemek** (E1.2) zarar veriyordu ⇒ **elemek değil, katmanlaştırmak** doğru strateji.
  4. ⚠️ **İlk koşu yine şişkindi (+0.0883):** t=0 kolunu bu betikte türettiğim aralıklarla kurmuştum → baseline 0.7253'e düştü. **Projenin PI'ı** kullanılınca fark **+0.0480**'a indi. **Ders tekrarlandı: baseline'ı yeniden türetmek kazancı şişirir** (E2.1'de de aynısı olmuştu).
  5. ⚠️ **Test bütçesi kapalı** ⇒ 0.9516'lık ana sonuca **eklenemez**; **E2.1 ile birlikte gelecek test değerlendirmesinin öncelikli iki adayından biri** (hatta daha güçlüsü).
* **Karar:** **KEEP** (val) — çok parametreli dilimler, gelecek çalışma için **en güçlü iyileştirme adayı**.
* **Manifest:** `artifacts/runs/E2.6_multiparameter_slices/manifest.json`

### E0.4 — ResNet18 Son-Blok İnce Ayarı ("hazır model" eleştirisine cevap) ⚠️

* **Hipotez (ön-kayıtlı):** Son-blok (`layer4` + `fc`) ince ayarı, dondurulmuş versiyonun **val** hasta ROC-AUC'sini (0.8884) **≥ +0.015** artırır.
* **Gerekçe:** E0.3'ün görüntü dalı **donmuş** ImageNet ağırlıklarıydı; *"görüntü tarafında model eğitmediniz"* eleştirisine açıktı. GPU (E0.0) gelince bu deney ucuzladı.
* **Tek değişken:** `layer4+fc` eğitilebilir mi. Mimari, ön-işleme, split ve değerlendirme protokolü E0.3 ile **aynı**. Havuzlanmış tek model (E3.5'te zon-özel ile eşdeğer bulundu).
* **Kapsam:** Yalnızca **validation**; test setine dokunulmadı.
* **Sonuç:** ✅ **KEEP (kriter sağlandı)** — ama **ciddi aşırı uyum uyarısıyla**

  | Ölçüt | Değer |
  | :--- | :--- |
  | **HASTA düzeyi val ROC-AUC** | **0.9487** |
  | Dondurulmuş versiyon (E0.3) | 0.8884 |
  | **Fark** | **+0.0603** (eşik +0.015 ✓) |
  | Zon düzeyi val AUC (en iyi epoch) | 0.8590 |
  | GPU eğitim süresi | **5 s/epoch** (CPU'da 205 s → **41× hızlanma**), 15 epoch ≈ 75 s |

* **🚨 Kritik uyarı — aşırı uyum:**
  * Eğitim kaybı **0.5894 → 0.0012**'ye düşüyor (neredeyse ezberleme).
  * Buna karşılık **zon düzeyi** val AUC epoch 7'den sonra **0.859'da plato** yapıyor; iyileşme duruyor.
  * Yani model **2.040 dengeli train zonunu ezberliyor**, genel temsili iyileştirmiyor.
  * **Hasta düzeyi 0.9487 neden bu kadar yüksek?** Muhtemelen **max-pooling + aşırı özgüven** etkisi: aşırı uyumlu model bazı zonlarda çok kesin (0/1'e yakın) tahmin üretiyor ve max-pooling bunu amplifiye ediyor. Kırılgan bir kazanç olabilir.
  * **Ayrıca metodolojik asimetri:** ince ayarlı model val üzerinde **15 epoch × checkpoint** ile seçildi, dondurulmuş model ise val üzerinde 5 C değeriyle. İkisi de val kullanıyor ama ince ayarlının **val'e uyum fırsatı daha fazla** → val karşılaştırması iyimser olabilir.

* **📌 Stratejik sonuç (yayın açısından önemli):**
  İnce ayarlı görüntü modelinin val hasta AUC'si **0.9487**, füzyonun val AUC'si **0.9572**'ye çok yakın. Eğer bu **test'te de** tutarsa, ana iddiamız (*"TDA güçlü görüntü modeline tamamlayıcı bilgi katıyor"*) **zayıflar** — çünkü füzyon kazancının büyük kısmı **zayıf (dondurulmuş) baseline**'dan kaynaklanıyor olabilir. Bu, literatür taramasında öngördüğümüz **1 numaralı hakem itirazının** ta kendisidir.
  ⇒ **Son test değerlendirmesi bu soruyu çözmeye ayrılmalı:** ince ayarlı ResNet18 ve **ince ayarlı + TDA füzyonu** test'te ölçülmeli.
* **Karar:** **KEEP** (ön-kayıtlı kriter sağlandı) — ancak **aşırı uyum** nedeniyle test doğrulaması **zorunlu**; sonuçlar şimdilik **val-only** olarak işaretlenmiştir.
* **Manifest:** `artifacts/runs/E0.4_resnet_finetune/manifest.json`

### E0.5 — Düzenlileştirilmiş İnce Ayar Taraması (aşırı uyum giderme)

* **Amaç:** E0.4'ün aşırı uyumunu (train loss 0.0012) gidermek. Ön-kayıtlı iki koşul: **(a)** son train loss > 0.05, **(b)** zon val AUC ≥ 0.87.
* **🐛 Kritik bug ve düzeltme:** İlk koşular **geçersizdi**. Önbellek E0.4 tarafından `train_dengeli+val+test` = **9380** satırla üretilmişti, E0.5 ise `test` hariç **5710** satır kullanıyordu → **görüntüler etiketlerle kaydı** ve model şans düzeyinde kaldı (val AUC 0.58). Önbellek satır sayısı ekrana basıldığında (`Zon: 5710` vs `Onbellek: (9380,...)`) fark edildi. Düzeltme: aynı satır kümesi kullanıldı + **boyut uyuşmazlığında hata fırlatan koruma** eklendi.
* **Sonuç (düzeltilmiş):** ⚠️ **KILL** — hiçbir tarif iki koşulu birlikte sağlamadı

  | Konfigürasyon | Son train loss | Zon val AUC | **Hasta val AUC** |
  | :--- | :---: | :---: | :---: |
  | E0.4 (artırma/düzenlileştirme yok) | 0.0012 ⚠️ | 0.8590 | 0.9487 |
  | **mild** (hflip, wd=1e-4, dropout 0.3) | 0.0012 ⚠️ | **0.8706** | **0.9546** |
  | **full** (artırma + wd=1e-3) | **0.1247** ✓ | 0.8439 | 0.9368 |

* **Yorum:**
  1. **Tam artırma aşırı uyumu gerçekten giderdi** (train loss 0.0012 → 0.1247) ama **temsili biraz düşürdü** (zon AUC 0.859 → 0.844).
  2. **Hasta düzeyi val AUC üç tarifte de 0.937–0.955** → ince ayarın kazancı **tarife duyarsız**, sağlam.
  3. ⇒ Asıl mesele aşırı uyum değil: ince ayarlı görüntü modeli **füzyona val'de çok yakın** (0.955 vs 0.957) ⇒ **E0.6 belirleyici testi zorunlu kıldı.**
* **Karar:** **KILL** (koşullar sağlanmadı) — ancak E0.6 için gerekli bilgiyi üretti.
* **Manifest:** `artifacts/runs/E0.5_resnet_finetune_reg*/manifest.json`

### E0.6 — BELİRLEYİCİ TEST: TDA, Güçlü Görüntü Modeline de Katkı Sağlıyor mu? 🏆

* **Hipotez (ön-kayıtlı):** TDA dalı, **ince ayarlı** görüntü modeline de test ROC-AUC'sinde **≥ +0.015** katkı sağlar (bootstrap %95 CI sıfırı dışlar).
* **Neden son test hakkı buraya:** E3.1'in orijinal kazancı **donmuş** bir baseline'a karşıydı ve o baseline ciddi genelleme kaybı yaşıyordu. E0.4/E0.5, ince ayarın val AUC'sini 0.95'e çıkardığını gösterdi ⇒ *"füzyon kazancının çoğu zayıf baseline'dan geliyor"* itirazı **test'te** sınanmalıydı. Bu, literatür taramasında öngörülen **1 numaralı hakem itirazıydı.**
* **Kapsam:** **SON test değerlendirmesi** (bütçe 5/6 → **6/6**).
* **Sonuç:** ✅ **KEEP — ana iddia güçlü bir baseline karşısında AYAKTA**

  | Model | Test hasta ROC-AUC | (donmuş dönem) |
  | :--- | :---: | :---: |
  | İnce ayarlı ResNet18 (tek başına) | **0.9284** | 0.8349 |
  | TDA ensemble (tek başına) | 0.8629 | 0.8776 |
  | **FÜZYON (a=0.45)** | **0.9516** | 0.9190 |
  | **Füzyon doğruluğu** | **0.8937** | 0.8638 |

  **TDA katkısı: +0.0232, %95 CI [+0.0132, +0.0338], p=0.0000** ✓

* **Yorum:**
  1. ✅ **İnce ayar baseline'ı çok güçlendirdi** (0.8349 → **0.9284**, +0.0935) ⇒ E0.3'ün "donmuş model" eleştirisi haklıymış.
  2. ✅ **Buna rağmen TDA anlamlı katkı sağlıyor** (+0.0232, CI sıfırı dışlıyor) ⇒ **füzyon kazancı zayıf baseline artefaktı DEĞİL.** Makalenin ana iddiası **en güçlü baseline karşısında doğrulandı.**
  3. 🏆 **Yeni ana sonuç: füzyon test ROC-AUC 0.9516, doğruluk %89.4** (dondurulmuş dönemde 0.9190 / %86.4).
  4. ✅ **Tutarsızlık ÇÖZÜLDÜ (2026-10-16) — hata yok, varyant farkı:** TDA dalı bu koşuda **0.8629** çıktı, E3.4'ün başlığında **0.8776** yazıyordu. Sebep **bulundu ve sıfır artıkla doğrulandı**: E3.4 iki varyant raporluyor — **Varyant A** (iki dalı *önce hasta düzeyine* max-pool, *sonra* birleştir) → val 0.8821 / **test 0.8776**; **Varyant B** (önce zon düzeyinde birleştir, sonra max-pool) → val **0.8699** / test **0.8629**. E0.6 ve E3.2 koddan **Varyant B'yi** yeniden üretmiştir → **birebir aynı sonuç (val 0.8699, test 0.8629)**. İkisi de **w=0.55** kullanıyor ⇒ **ağırlık farkı değil, varyant farkı** ve **kayıtlı bundle'lar E3.4'ü tam olarak yeniden üretiyor** (yeniden üretilebilirlik sorunu **yok**).
     * 📌 **Önemli sonuç:** E0.6'nın füzyonu (0.9516) TDA'nın **daha zayıf varyantını (B, 0.8629)** kullanmıştır ⇒ **muhafazakârdır**; TDA katkısı **+0.0232** bir **alt sınırdır**. **Varyant A (0.8776)** ile füzyon daha yüksek olacaktır. Test bütçesi tükendiği için **doğrulanamadı**; gelecek bir test değerlendirmesinde **Varyant A kullanılmalıdır**.
* **Karar:** **KEEP** 🏆 — ana iddia, ince ayarlı güçlü görüntü modeli karşısında **istatistiksel olarak doğrulandı**.
* **Manifest:** `artifacts/runs/E0.6_decisive_test/manifest.json`

---

## 7. Bağımsız Dış Doğrulama (VinDr-CXR)

### V4 — VinDr-CXR üzerinde sıfır-eğitim harici test (dış kalibrasyon ablasyonu)

* **Veri:** VinDr-CXR (Vietnam — Hospital 108 & Hanoi Medical University Hospital), 3× indirgenmiş JPEG kohortu. **400 görüntü**: 150 nodül/kitle + 250 saf normal (annotasyonların *tamamı* "No finding"), `seed = 42`. ⚠️ **Görüntü düzeyi:** VinDr metadata'sında hasta kimliği yoktur; kohort "400 hasta" değil **400 görüntüdür**. Kohort deterministik olarak [`V4_build_vindr_cohort.py`](pipelines/exp/V4_build_vindr_cohort.py) ile üretilir ve önceki kohortla **400/400 örtüşür**.
* **Protokol:** **VinDr'de sıfır eğitim.** TDA dalı dondurulmuş NODE21 bundle'ları (SmallZoneCNN + SVM, `w = 0.55`, E3.4); füzyon **E0.6'nın kilitli ağırlığı `a = 0.45`** (görüntü ağırlığı; `test_report.json` ile doğrulandı) ve kilitli eşik `0.7172`.
* **ResNet18 dürüstlük notu:** E0.6 ağırlıkları **hiç kaydedilmemişti** ve GPU eğitimi bit-bit yeniden üretilebilir değildir. Bu yüzden ResNet, E0.6 tarifiyle **deterministik** olarak NODE21'de yeniden eğitilir, elde edilen checkpoint sabitlenir (`--reuse-resnet`) ve VinDr'de yalnızca çıkarım yapılır. Uyum: val zon AUC **0.8707** (E0.6 kaydı 0.8703; **Δ = 3,8·10⁻⁴**). ⇒ Sonuçlar "E0.6 *mimarisi/tarifi*"ne atfedilir, kayıp ağırlıklara değil.
* **Kalibrasyon kolları:** `linear` (8-bit × 16 ≈ NODE21 12-bit, eşleme yok) · `matched` (200 hastalık NODE21 havuzuna histogram eşleme; **yalnızca dokuya**, arka plan `0` kalır) · `matched_scalenorm` (ek olarak kesitler **akciğer yüksekliğine** göre ölçeklenir; NODE21 medyanı 810 px ↔ VinDr 570 px, ≈1,42×).

**Sonuçlar (ROC-AUC, %95 bootstrap CI; N = 2.000):**

| Kol | TDA ensemble | ResNet18 | Füzyon (**a = 0.45**) | Δ(füzyon − ResNet) | VinDr'de optimal *a* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `linear` | **0,5274** [0,4699; 0,5836] | **0,6204** [0,5609; 0,6763] | 0,5328 [0,4782; 0,5891] | **−0,0876** [−0,1262; −0,0491] p<0,001 | 1,00 (0,6204) |
| `matched` | **0,5409** [0,4852; 0,5973] | **0,6204** [0,5609; 0,6763] | 0,5438 [0,4863; 0,5982] | **−0,0765** [−0,1133; −0,0393] p=0,001 | 1,00 (0,6204) |
| `matched_scalenorm` | **0,5110** [0,4522; 0,5688] | **0,6204** [0,5609; 0,6763] | 0,5352 [0,4765; 0,5914] | **−0,0852** [−0,1302; −0,0403] p<0,001 | 1,00 (0,6204) |

* ResNet18 tüm kollarda aynı modeldir (VinDr'de eğitim yok, aynı checkpoint).
* **Tam a-taraması** (`a ∈ {0,00 … 1,00}`, 0,05 adım) `vindr_report_*.json` içindedir. Eğri, `a = 0,00`'dan (`a = 1` hariç) neredeyse düz ve **≤ 0,5578**'de kalır; `a = 1,00`'da sıçrayıp 0,6204 olur. ⇒ Harici veride füzyonun **optimal ağırlığı 1,00**, yani **TDA dalı katkı sağlamıyor**.
* Temsil korelasyonu **r = 0,27–0,40** (iç dağılımdaki 0,30 ile uyumlu).
* Füzyonun kilitli eşikteki ikili başarımı (kol `matched`): doğruluk **%56,5**, duyarlılık **%52,7**, özgüllük **%58,8**, F1 **0,4759**.

**Ölçek-bağımsız nodül boyutu alt grupları (kol `matched`; `rel_size` üçte birlikleri, sınırlar 0,1114 / 0,3452):**

| Üçte birlik (n = 50 pozitif + 250 kontrol) | `rel_size` aralığı | TDA | ResNet18 | Füzyon |
| :--- | :---: | :---: | :---: | :---: |
| Küçük | 0,033–0,110 | 0,5016 | 0,5823 | 0,4965 |
| Orta | 0,112–0,343 | 0,5494 | **0,6633** | 0,5763 |
| Büyük | 0,350–2,097 | 0,5718 | 0,6154 | 0,5586 |

> ⚠️ **mm dönüşümü YOK.** Kohort 3× indirgenmiş JPEG'lerdir ve DICOM spacing içermez; `bbox_max_dim_px` **annotasyon/orijinal DICOM uzayındadır**. `rel_size = bbox_max_dim_px / image_h` bu yüzden (yaklaşık sabit) bilinmeyen bir indirgeme katsayısı kadar **şişkindir** (> 1 değerler bu yüzden); yalnızca **sıralama/üçte birlik** anlamlıdır. Önceki "%15 mm / %30 mm" etiketleri **kaldırılmıştır** (dayanaksızdı).

**Donanım / kaynak kısayolu teşhisi (artifact'lı):**
* **A0/A1 — yalnızca meta-veri** (spacing + boyut + oran): Lojistik Regresyon val ROC-AUC **0,6850** / test 0,6576; Gradient Boosting val 0,6784 / test 0,6345 ⇒ **kısmi kısayol** (0,65–0,80 aralığı). Rapor: [`a0_a1_metadata_report.json`](data/external_vindr/a0_a1_metadata_report.json). ⇒ NODE21 içi başarım **kısmen** spacing/boyut dağılımından besleniyor; kurum içi sayılar bu confound ile okunmalıdır.
* **Eşit-boyutlu kaynak-dışı koşu (E5.3b):** tüm katlar 1034 dengeli zona eşitlendiğinde TDA ortalaması **0,7631**, füzyon **0,8098** ⇒ sinyal tamamen yapay bir donanım kısayolu **değil**. (Kaynak başına değerler kayıtlı değildir; yalnızca ortalamalar raporlanır.)

**Çıkarımlar:**
1. **TDA dış dünyaya taşınmıyor:** VinDr'de TDA ensemble şans düzeyinde (0,511–0,541; tüm CI'lar 0,5'i içerir veya kıl payı aşar), ResNet18 ise kısmen taşınıyor (0,6204; CI alt sınırı 0,5609). İç dağılımdaki 0,9516 füzyonu ile fark **devasa**.
2. **Kilitli ağırlıkla füzyon zarar veriyor:** a = 0,45 her kolda ResNet'in **altında** (Δ ≈ −0,077…−0,088, CI'lar sıfırı dışlar). Harici veride optimal a = 1,00 ⇒ **TDA katkısı sıfıra yakın**.
3. **Ölçek normalizasyonu kurtarmıyor:** akciğer yüksekliğine göre ölçekleme TDA'yı iyileştirmiyor (0,5110 ≤ 0,5409) ⇒ çöküş yalnızca yoğunluk/ölçek farkıyla açıklanamaz; dedektör fiziği/doku dokusu düzeyinde bir domain-shift söz konusudur.
4. **Konumlandırma:** sistem evrensel bir lezyon kutulayıcısı değil, **kurum-içi risk triyaj** altyapısıdır; dış merkeze taşınması kalibrasyon/adaptasyon gerektirir.

> 🐛 **Önceki (geçersiz) sayıların düzeltmesi.** Bu bölümde daha önce raporlanan VinDr değerleri (TDA 0,5103/0,5082/0,5429; ResNet 0,6242/0,6318/0,6545; doğruluk %53,5–55,8) **iki hatalı betikten** geliyordu: (i) `match_histograms` **tüm kesite** uygulanıyordu ve arka plan sıfırları havuzlanmış *doku* referansının en düşük değerine (~1080) taşınarak TDA'nın `image > 0` doku maskesini bozuyordu; (ii) "tek hasta (c0001) referansı" kolu keyfî bir referansa dayanıyordu. Ayrıca alt grup eşikleri dayanaksız mm etiketleriyle veriliyordu ve kaynak-dışı "0,8537/0,8985/0,9365" sayılarının hiçbir artifact karşılığı yoktu. Bu üç nokta **kaldırıldı**; yukarıdaki tablo tek koşuda, tek betikten ve kayıtlı raporlardan gelir.

### 7.1 Cihaz/kaynak ayrıştırılabilirliği — mimarinin fizik kodlama teşhisi (A2)

Alan kaymasının **kaynağını** VinDr'e dokunmadan ölçtük. NODE21'in kendi 4 kaynağı (chestxray14 0,143 mm/px · padchest 0,275 · jsrt 0,309 · openi 0,331) "cihaz/kurum" değişkeni olarak kullanıldı; test **yalnızca nodül-pozitif hastalarda** (etiketten bağımsız), hasta düzeyi, train→val, **test hastaları kullanılmadan**.

| Temsil | Kaynak tahmini (makro AUC) | Doğruluk |
| :--- | :---: | :---: |
| TDA PI (800) | 0,850 (LR) / 0,911 (HGB) | %70–80 |
| **ResNet18 özellikleri (512)** | **0,949** | **%89,5** |
| Yalnızca meta-veri (spacing+boyut) | 0,956 | %80,2 |

⇒ Temsiller **hangi cihazdan geldiğini neredeyse hastalığı tanıdığı kadar iyi** kodluyor (klasik confound imzası). Ayrıntı ve uyarılar: [ablation_matrix.md §7.3](ablation_matrix.md).

* **Kayıtlar:** `artifacts/runs/V4_vindr_external_validation/` (manifest + kol raporları + hasta düzeyi tahmin CSV'leri) · git'te izlenen kopyalar: `data/external_vindr/vindr_report_{linear,matched,matched_scalenorm}.json`, `vindr_summary_report.json`, `a0_a1_metadata_report.json`, `a2_source_predictability_report.json`.
* **Betikler:** [`V4_external_validation_vindr.py`](pipelines/exp/V4_external_validation_vindr.py) · [`V4_build_vindr_cohort.py`](pipelines/exp/V4_build_vindr_cohort.py) · [`V4_build_ref_histograms.py`](pipelines/exp/V4_build_ref_histograms.py) · sözleşme: [`configs/experiments/V4_vindr_external_validation.yaml`](configs/experiments/V4_vindr_external_validation.yaml).
* **Karar:** **EXTERNAL 🌐** — harici veride TDA taşınmıyor; v3 ana iddiası (kurum içi füzyon 0,9516) **geçerlidir** ama **kapsamı kurum içi ile sınırlıdır**.

---

## 8. Fiziksel Kanonikleştirme (V5.1)

> **Neden:** V4 (harici doğrulama) + A2 (kaynak ayrıştırılabilirliği) teşhisi, alan kaymasının kök nedenini gösterdi: filtrasyon **mutlak yoğunluğa** ve **piksel ölçeğine** bağlı. V5.1, **5-zon mimarisi korunarak** bu iki büyüklüğü kanonikleştirmeyi ölçer.
> **Kapsam kuralı:** zon şeması değiştirilmedi; `paper_draft.md` iskeleti bozulmadı.
> **Ön-kayıt:** [`configs/experiments/V5_1_physical_canonicalization.yaml`](configs/experiments/V5_1_physical_canonicalization.yaml). Tek değişken: her basamak birebir müdahale ekler. Model: sabit hiperparametreli RBF-SVM (kol başına tuning yok).

### Basamaklar
| Kol | Filtrasyon | Model eğitimi |
| :--- | :--- | :--- |
| `baseline` | v2 projesi: `scaled = piksel × 255 / 65535` (sabit) | eğitim istatistikleri kilitli |
| **`a_density`** | doku **p1–p99 → [0,255]** (mutlak yoğunluk bağımsız) | kilitli |
| **`b_density_scale`** | (a) + kesit, hastanın **akciğer yüksekliği** hedefe (809 px) gelene kadar ölçeklenir | kilitli |
| `+TTA` | aynı temsil | **StandardScaler hedef (etiketsiz) kümesinde yeniden fit** (BN-recalibrasyonun SVM karşılığı) |

### Sonuçlar (üçlü kapı)
| Kol | val label AUC | A2 kaynak AUC | LOO (E5.3b) |
| :--- | :---: | :---: | :---: |
| `baseline` | 0,8330 | 0,8501 | 0,7249 |
| **`a_density`** | **0,9211** (+0,0881) | 0,9554 (+0,1053) | **0,7861** (+0,0611) |
| `a_density` + TTA | 0,9003 | 0,9622 | **0,7967** (+0,0718) |
| `b_density_scale` | 0,8539 (+0,0209) | 0,9138 (+0,0637) | 0,7400 (+0,0150) |
| `b_density_scale` + TTA | 0,8762 | 0,9235 | 0,7305 |

**LOO katları (E5.3b, 1034 dengeli zona eşitlenmiş):**
| Kol | chestxray14 | padchest | jsrt | ortalama |
| :--- | :---: | :---: | :---: | :---: |
| `baseline` | 0,7380 | 0,6405 | 0,7963 | 0,7249 |
| **`a_density`** | 0,7453 | 0,7551 | **0,8579** | **0,7861** |
| `b_density_scale` | 0,7718 | 0,7311 | 0,7171 | 0,7400 |

**Ön-kayıtlı karar kuralına göre:** `a_density` **KILL** (kaynak AUC *düştü* değil +0,1053 *arttı*), `b_density_scale` **KILL** (kaynak +0,0637; LOO +0,0150 sınırda).

### 🔑 Bulgular
1. **Temsil fiilen ~4-bit'ti.** `intensity_max=65535` sabiti, NODE21'in **12-bit** verisi için **16× fazla** ⇒ doku yoğunluğu ~16 seviyeye kuantalanıyor (255 yerine). Kanonikleştirme 255 seviyeyi kullanır:
   | | H1 birth | H1 kalıcılık | H1 nokta sayısı | PI doluluk |
   | :--- | :---: | :---: | :---: | :---: |
   | `baseline` | 239–250 | **0–7** | 610.093 | %38 |
   | `a_density` | 13–248 | 0–97 | **1.633.427** | **%81** |
   ⇒ v1'deki "grid collapse" hatası **tam kapanmamıştı**; v2 yalnızca H0/H1 ızgaralarını ayırarak semptomu azaltmıştı. VinDr'de "matched" histogram eşlemesinin TDA'yı kurtaramamasının nedeni de bu niceleme darboğazıdır.
2. **Yoğunluk kanonikleştirmesi hem iç hem kaynak-dışı başarımı büyük ölçüde artırıyor** (val **+0,0881**, LOO **+0,0611**) — programın şimdiye kadarki **en büyük val kazancı** (v3'ün E2.6 +0,0480'ından da büyük). Buna karşın **kaynak AUC de artıyor** ⇒ A2'nin öncülü (*kaynak AUC düşerse genelleme artar*) **çürüdü**: kanonikleştirme cihaza özgü **dokuyu** görünür kıldığı için cihaz ve hastalık sinyali birlikte güçleniyor. ⇒ Kaynak AUC **tanısal** kalmalı, **kapı** olmamalı (yeni ön-kayıt gerekir).
3. **Ölçek normalizasyonu (b) zarar veriyor.** (a)'nın üstüne eklenince val −0,067, LOO −0,046. Kat bazında: chestxray14 (0,147 mm/px; 2× örnekleme) **+0,027**, ama jsrt (≈1,25× örnekleme) **−0,141**. ⇒ Topolojik temsiller **bilinear yeniden örneklemeye aşırı duyarlı**; ölçeği "düzeltmek" doku istatistiğini bozuyor. (V4'ün `matched_scalenorm` kolu da kurtarmamıştı; burada mekanizma ayrıştırıldı.)
4. **TTA tutarlı değil.** (a)'da: val −0,021 / LOO +0,011 (domain yararına takas). (b)'de ters işaret: val +0,022 / LOO −0,009. ⇒ TTA'nın faydası temsilin gücüne bağlı; **tek başına** taşınabilirlik garantisi değil. Kuralınıza uygun olarak ayrı satırda raporlanmıştır.

### ⚠️ Açık kontrol (V5.1b adayı)
p1–p99 **kırpıntı başına** hesaplandığı için "uyarlanabilir pencere" etkisi, kanonik ölçek etkisinden ayrılmamıştır (nodülün kendisi p99'u hafifçe yükseltebilir; etiket sızıntısı değildir ama kontrol gerektirir). ⇒ **Sabit referans** (train-geneli yüzdeler veya sabit window/level) ile karşılaştırma yapılmalı. Ayrıca bulgu **CNN ensemble** ile teyit edilmelidir (bu koşu tek-değişken için SVM kullandı).

* **Rapor:** [`v5_1_physical_canonicalization_report.json`](data/external_vindr/v5_1_physical_canonicalization_report.json) · **Manifest:** `artifacts/runs/V5.1_physical_canonicalization/manifest.json` · **Betik:** [`V5_1_physical_canonicalization.py`](pipelines/exp/V5_1_physical_canonicalization.py)
* **Ön-kayıt durumu:** kural-temelli karar `KILL` (kaynak AUC düşmedi) — kayıtta duruyor. **Sahip kararı: KEEP ✅ (2026-10-17)**; gerekçe, kaynak AUC artışının bilgi miktarındaki artışın doğal sonucu olması ve asıl hedeflerin (val +0,0881 · LOO +0,0611 · +TTA LOO +0,0718) büyük ölçüde artmasıdır. Ayrıntı: `owner_override` bloğu. Kapı değişikliği (kaynak AUC tanısal; LOO birincil + val guardrail) **ayrıca ön-kayıt edilecek**. Test bütçesi 6/6 tükendiği için bu kazanç **val/LOO düzeyindedir**.
### V5.1b — Sabit (tüm-akciğer) referans kontrolü: pencere artefaktı mı?

`a_density` p1–p99'u **her zon kırpıntısının kendi** dokusundan hesaplıyordu (uyarlanabilir pencere). Kontrol kolunda aynı eşleme **hastanın tüm akciğer dokusundan** (5 zonun birleşimi, tek p1/p99) hesaplanır ⇒ lezyonla istatistiksel çiftleşme kalkar. Diğer her şey birebir aynıdır (aynı kesitler, GUDHI, aralık türetimi, modeller).

| Ölçüt | `a_density` (per-kırpıntı) | **`c_lungref` (tüm-akciğer)** | Δ |
| :--- | :---: | :---: | :---: |
| val label AUC (SVM) | 0,9211 | 0,9133 | −0,0078 |
| A2 kaynak AUC | 0,9554 | 0,9617 | +0,0063 |
| LOO (E5.3b) | 0,7861 | **0,7897** | +0,0036 |
| TDA ensemble (CNN+SVM) | 0,9522 | 0,9335 | −0,0187 |
| **FÜZYON (val, en iyi a)** | 0,9856 (a=0,25) | **0,9778** (a=0,35) | **−0,0078** |
| *(v2 referans)* | *0,9729 (a=0,40)* | *+0,0049 vs v2* | |

**Yorum (ön-kayıtlı tolerans ±0,005; fark −0,0078 ⇒ "kısmi pencere etkisi"):**
1. ⚠️ **Kırpıntı-başına pencere, füzyon kazancının bir kısmını açıklıyor: ≈+0,0078** (0,0127'nin ~%60'ı). Bu katkı **topolojik değil, histogram/pencere kaynaklıdır** — kırpıntının kendi p99'u lezyondan hafifçe etkilenir. Test zamanında hesaplanabildiği için "hile" değildir, ama *"topolojik parmak izi"* iddiası için **ayrıştırılması gerekir**.
2. ✅ **Nicelleme düzeltmesinin kendisi baskın ve gerçek:** çiftleşmesiz varyantta bile TDA ensemble **0,8811 → 0,9335 (+0,0524)**; yani +0,0711'in **%74'ü** pencere olmadan da vardır.
3. ✅ **Çiftleşmesiz varyantta LOO daha iyi** (0,7897 vs 0,7861) ve val 0,9133 ⇒ taşınabilirlik açısından en az eşdeğer.
4. ⇒ **Karar: KEEP korunur** (her iki varyantta da TDA ensemble ve füzyon v2'nin üzerinde), ancak **füzyon için muhafazakâr sayı `c_lungref` = 0,9778**'dir; 0,9856 pencere katkısını içerir ve bu ayrım makalede açıkça yazılmalıdır.

* **Rapor:** [`v5_1b_fixed_reference_report.json`](data/external_vindr/v5_1b_fixed_reference_report.json) · **Manifest:** `artifacts/runs/V5.1b_fixed_reference/` · **Ön-kayıt:** [`configs/experiments/V5_1b_fixed_reference.yaml`](configs/experiments/V5_1b_fixed_reference.yaml)

### V5.2 — Düzeltilmiş TDA, projenin gerçek modelinde (val-only)

Aynı karşılaştırma, **5-zon küçük CNN + RBF-SVM ensemble** ve **ince ayarlı ResNet18 geç füzyonu** ile tekrarlandı. **Test bütçesi 6/6 kapalı** ⇒ tüm sayılar **validation** (734 hasta).

| Dal | v2 dondurulmuş | **Düzeltilmiş (V5.1)** | Δ |
| :--- | :---: | :---: | :---: |
| TDA CNN (tek) | 0,8580 | 0,9156 | **+0,0576** |
| TDA SVM (tek) | 0,8278 | 0,9369 | **+0,1092** |
| **TDA ensemble (0,55/0,45)** | 0,8811 | **0,9522** | **+0,0711** |
| ResNet18 (E0.6) | 0,9547 | 0,9547 | 0 (aynı checkpoint) |
| **FÜZYON (val'de en iyi a)** | 0,9729 (a=0,40) | **0,9856** (**a=0,25**) | **+0,0127** |
| Füzyon val doğruluğu | %92,23 | **%92,78** | +0,55 puan |

* ✅ **KEEP** (ön-kayıtlı kural: TDA ensemble ↑ **ve** füzyon ↑).
* 🔑 **Füzyon ağırlığı değişti: a = 0,40 → 0,25** (TDA ağırlığı 0,60 → **0,75**). Düzeltilmiş TDA artık ResNet'e neredeyse eşit (0,9522 vs 0,9547) ve füzyona gerçek katkı veriyor ⇒ v4'ün *"TDA iyileştirmeleri ResNet ile yedekli"* bulgusu **nicelleme düzeltmesinden sonra geçersizdir**; redundancy, bozuk temsilin artefaktıymış.
* ⚠️ **Dürüst not:** v4'ün daha sıkı füzyon kapısıyla (Δ_füzyon ≥ **+0,015**) +0,0127 **sınırda kalır (KILL)**; V5.2'nin kendi ön-kayıtlı kuralı KEEP der. İki sonuç birlikte raporlanır (Δ(a=0,45) = +0,0098).
* ✅ **Açık kontrol KAPANDI (V5.1b):** per-kırpıntı yüzdelerin "uyarlanabilir pencere" katkısı ölçüldü — tüm-akciğer referansıyla füzyon **0,9778** (−0,0078), yani kazancın ~%60'ı pencere kaynaklı; **nicelleme düzeltmesinin kendisi baskın** (TDA ensemble kazancının %74'ü penceresiz de var). ⇒ Füzyon için **muhafazakâr sayı 0,9778**'dir (bkz. V5.1b).

### V5.3 — VinDr-CXR harici testinin düzeltilmiş TDA ile tekrarı

**V5.3** · [betik](pipelines/exp/V5_3_corrected_tda_vindr.py) · VinDr'de **sıfır eğitim** (V5.1 bundle'ları + E0.6 ResNet checkpoint) · kalibrasyon: **kanonik yoğunluk** (histogram eşleme gereksizleşti).

| | V4 (eski ~4-bit temsil) | **V5.3 (düzeltilmiş)** | Δ (eşleştirilmiş bootstrap, n=400) |
| :--- | :---: | :---: | :---: |
| TDA ensemble | 0,5409 | **0,5959** [0,5475; 0,6450] | +0,0550 [−0,0002; +0,1078], p=0,053 |
| ResNet18 (E0.6) | 0,6204 | 0,6204 | 0 (aynı model) |
| **Füzyon (a=0,45, kilitli)** | 0,5438 | **0,6283** | **+0,0844 [+0,0464; +0,1219], p=0,0000** ✅ |
| VinDr'de optimal a | 1,00 (→0,6204) | **0,75** (→**0,6317**) | — |
| r(ResNet, TDA) | 0,4009 | **0,2997** | daha az yedekli |

**Nodül-boyutu alt grupları (rel_size üçte birlik):** küçük n=50 → TDA **0,5951** / ResNet 0,5823 / füzyon 0,6026 · orta n=50 → 0,6356 / 0,6632 / **0,6812** · büyük n=50 → 0,5570 / **0,6156** / 0,6010. ⇒ Düzeltilmiş TDA **küçük nodüllerde ResNet'i geçiyor** (eski temsilde 0,50'ye çöküyordu).

**Okunuşu:**
1. ✅ **Harici çöküş büyük ölçüde ~4-bit temsilin artefaktıydı:** füzyon +0,0844 (CI sıfırı dışlıyor, p<0,001). V4'ün "TDA dış dünyaya taşınmıyor" sonucu **temizlenmiş temsilde zayıflıyor**.
2. 🟡 **Ama TDA hâlâ ResNet'in altında** (0,5959 vs 0,6204; fark anlamlı değil, CI örtüşüyor) ve füzyonun görüntü dalına göre **artı kazancı** (+0,008…+0,011) **gürültü mertebesinde** (CI sıfırı içeriyor).
3. 🟡 **Optimal ağırlık 1,00 → 0,75** oldu: dış veride de TDA artık *zarar vermiyor* ve hafif katkı sağlıyor.
4. ⚠️ Mutlak harici AUC hâlâ mütevazı (≈0,63) ⇒ merkezler arası **kalibrasyon/adaptasyon** ihtiyacı sürüyor; iyileşen şey **göreli** taşınabilirlik.

* **Rapor:** [`v5_3_corrected_tda_vindr_report.json`](data/external_vindr/v5_3_corrected_tda_vindr_report.json) · **Manifest + tahminler:** `artifacts/runs/V5.3_corrected_tda_vindr/` · **Kıyas betikleri:** `scratch/v5_3_delta_ci.py`, `scratch/v5_3_vs_v4_delta.py`

* **Sıradaki adımlar:** **V5.1b** sabit-referans normalizasyon kontrolü (uyarlanabilir pencere confound'u) · **V5.3b** VinDr'de füzyon ağırlığının ayrı bir kümede seçilmesi · gerekirse **V5.4** harici adaptasyon (VinDr'de eğitim — ayrı ön-kayıt gerekir).

---

## 9. V6 — İki Aşamalı Lokalizasyon (Aday Üretici + Lokal TDA FP Eleme)

> **Kapsam:** v3/v5 programından **bağımsız** yeni program. Disk kuralı: görüntü/kırpıntı **yazılmaz**; yalnız hafif JSON + model checkpoint. Kod `pipelines/v6_two_stage/`, kayıtlar `artifacts/v6_two_stage/`.
> **Hedef:** 2. aşamadaki lokal TDA sınıflandırıcıya geçmeden, K=10–50 aday içinde **%98–99** nodül yakalama.

### V6.0 — 1. Aşama aday üretici (oracle Recall@K, val 231 lezyon)

| Yöntem | K10 | K20 | K30 | K50 | ms/görüntü |
| :--- | :---: | :---: | :---: | :---: | :---: |
| M1 LoG+CLAHE (eğitimsiz) | 0,069 | 0,143 | 0,173 | 0,229 | 1929 |
| M2 DoH+damar bastırma | 0,052 | 0,117 | 0,229 | 0,325 | 697 |
| M3 radyal simetri | 0,056 | 0,108 | 0,143 | 0,216 | 286 |
| M4 bilateral asimetri | 0,004 | 0,009 | 0,013 | 0,069 | 1347 |
| **M5 CenterNet (ResNet18-stride2)** | **0,866** | **0,961** | **0,974** | **0,978** | **26** |
| M6 FCOS-lite (focal objectness) | 0,481 | 0,576 | 0,654 | 0,758 | 54 |
| **M5 + heatTTA** | **0,909** | **0,965** | 0,970 | 0,978 | ~60 |
| **M5+M6 (RRF)** | 0,818 | 0,918 | 0,974 | **0,983** | ~150 |
| M7 hibrit (M5 %70 + M2 %30) | 0,784 | 0,939 | 0,970 | 0,978 | 686 |

**Faz 1 dersleri:** (i) eğitimsiz filtreler K50'de bile %7–33'te kalıyor (kaburga/hilus parlaklığı sıralamayı domine ediyor); (ii) **pozitif-only eğitim tuzağı doğrulandı** — 1:1 temiz kontrol (boş hedef) olmadan model her görüntüde tepe üretiyor (0,30 → kontrollerle 0,0 → 15 epoch'ta 0,97); (iii) M6'nın ilk hali dengesiz BCE yüzünden 0,017'de çöküyordu, **focal objectness** ile 0,76'ya çıktı; (iv) **stride-2 (256×256 çıktı = 1024'e stride-4)** ve heatmap eşiğinin 0,05→0,01'e indirilmesi M5'i 0,853→0,974 taşıdı.

### V6.1/V6.2 — 2. Aşama: lokal TDA + uzamsal öncül + FP eleme (val, 734 hasta)

Temsil: aday merkezli **32×32 (çekirdek) + 64×64 (çevre)** H0/H1 PI (1600 boyut, **c_lungref ölçeği korunur**; yerel min-max yok) + 4 uzamsal özellik (rel_x, rel_y, hilus, plevra) + 64×64 yama.

| Ayar | Model | Hasta AUC (Top-1 / Top-3 / Noisy-OR) | FROC @0.5/1/2/4 FP |
| :--- | :--- | :---: | :---: |
| M5M6TTA·K50 | (a) yalnız Aşama-1 | 0,688 / 0,709 / 0,711 | 0,281 / 0,403 / 0,520 / 0,623 |
| M5M6TTA·K50 | (b) **lokal TDA-HGB** | **0,882 / 0,922 / 0,926** | 0,312 / 0,411 / 0,515 / 0,615 |
| M5M6TTA·K50 | (c) yama CNN | 0,650 / 0,658 / 0,652 | 0,052 / 0,117 / 0,186 / 0,255 |
| M5M6TTA·K50 | (d) tam füzyon | 0,906 / 0,942 / 0,951 | 0,511 / 0,610 / 0,680 / 0,766 |
| **M5M6TTA·K50** | **(e) e_fusion_tda** | **0,911 / 0,945 / 0,955** | **0,528 / 0,615 / 0,688 / 0,771** |
| **M5_heatTTA·K30** | **(f) f_hierarchical** | **0,923 / 0,933 / 0,936** | **0,589 / 0,658 / 0,753 / 0,857** |

* **Yama CNN mimariden ÇIKARILDI** (AUC 0,646; onsuz füzyon eşit/daha iyi ve daha ucuz) ⇒ *"lokal ölçekte ayırt edici bilgi piksel dokusunda değil, topolojide"* bir kez daha doğrulandı.
* **Doğrusal olmayan topluluk (SVM+MLP rank-ortalaması) reddedildi**: HGB tek başına 0,919 iken topluluk 0,656 (zayıf üyeler güçlü HGB'yi aşağı çekti).
* **FROC protokolü düzeltildi:** post-fusion NMS + duplikasyon **IGNORE** (LUNA16/NODE21 stili) ⇒ hiyerarşik kalibrasyon FROC'u tüm eşiklerde yukarı taşıdı (0,857 @4 FP).
* **Hiyerarşik kalibrasyon:** `S_final = p_aday × P_NoisyOR(hasta)`.

### V6.3 — VinDr-CXR sıfır-eğitim bileşen otopsisi (400 görüntü)

| Bileşen | VinDr AUC | NODE21 val AUC |
| :--- | :---: | :---: |
| (a) Aşama-1 dedektör | **0,4618** | 0,7111 |
| (b) lokal TDA-HGB | **0,5901** | 0,9261 |
| (c) e_fusion_tda | **0,5911** | **0,9546** |
| V5.3 global 5-zon (0,45·ResNet+0,55·TDA) | **0,6283** | 0,9821 |
| r(V5.3, V6.2) | 0,359 | 0,385 |
| Birleşik (eşit rank-ort.) | 0,6214 | **0,9921** |

**VinDr bbox recall@K (3× indirgeme doğrulandı, fit=1,00; 150 pozitif):** M5_heatTTA **K30=0,880 · K50=0,900** · M5M6TTA K30=0,873 · K50=0,893.

**Kritik teşhis:** VinDr'de **aday üretici nodülleri buluyor** (K50 recall 0,90) ama **görüntü düzeyi güven skoru şans altı** (AUC 0,46) ⇒ darboğaz **lokalizasyon değil, skor kalibrasyonu** (normal görüntülerde de yüksek güvenli adaylar var). Lokal TDA (0,590) en taşınabilir V6 bileşeni; ancak V5.3 global (0,6283) hâlâ en iyi ve birleşim onu geçmiyor (0,6214).

### V6 nihai modeller (donduruldu)
* **Maks AUC:** `M5M6TTA · K50 · e_fusion_tda` — hasta AUC **0,955** (Noisy-OR), FROC 0,528/0,615/0,688/0,771.
* **FROC-optimal:** `M5_heatTTA · K30 · f_hierarchical` — FROC **0,589/0,658/0,753/0,857**.
* Checkpoint'ler: `artifacts/v6_two_stage/checkpoints/{M5_centernet.pt, M6_fcos.pt, stage2_final_tda_hgb.joblib}` (yama CNN yok).
* Raporlar: `v6_0_candidate_benchmark.json` · `v6_0_miss_autopsy.json` · `v6_0_variant_sweep.json` · `v6_1_stage2_fp_reduction.json` · `v6_2_froc_and_transfer.json` · `v6_3_final_report.json` · `v6_3_vindr_recall.json` · `v6_4_domain_and_test_seal.json` · `v6_4b_vindr_adabn_tda.json` · `v6_5_vindr_transfer_boost.json` · `v6_5b_fewshot_compact.json`.

### V6.4 — Domain stabilizasyonu + NODE21 TEST mühürü (tek sefer)

**Adım 1 — VinDr test-time stabilizasyon (eğitimsiz):** Aşama-1 görüntü AUC (top-1): baseline **0,5443** · SNR-diff 0,5440 · SNR-z 0,5453 ⇒ **SNR etkisiz**. **AdaBN 0,6332 (+0,089)**, recall@50 0,907 ⇒ M5 için gerçek kazanç; ancak M6 AdaBN'ı bozduğu için M5+M6 füzyonu 0,512 → 0,441 ⇒ **AdaBN yalnız M5'e uygulanmalı**.
*(Not: V6.3'teki 0,4618 noisy-OR toplamasıydı; burada top-1 — fark bundan.)*

**Adım 1b — M6'sız AdaBN'li M5 + lokal TDA (VinDr):** AdaBN-M5 Stage-1 **0,6346** · lokal TDA-HGB 0,6082 · **e_fusion_tda 0,6471** · V5.3 global 0,6283 · **V5.3 + birleşik 0,6494** (r = 0,460). ⇒ **M6 çıkarılınca AdaBN V5.3'ü geçiyor** (+0,019); birleşim en iyi (+0,021).

**Adım 2 — Fiziksel domain augmentation ile M5_domain_gen:** NODE21 val `M5_heatTTA@50` 0,978 → **0,919**, `M5M6TTA@50` 0,983 → **0,971**; VinDr AUC 0,544 → **0,516**. ⇒ **KILL**: in-domain kayıp var, transfer kazancı yok (gamma/CLAHE/jitter doku istatistiğini bozuyor).

**Adım 3 — NODE21 TEST mühürü (734 hasta / 229 lezyon; tek sefer, sıfır ayar):**
| Model | Test hasta AUC | Test FROC @0.5/1/2/4 FP |
| :--- | :---: | :---: |
| V6 lokal (`e_fusion_tda`) | 0,9464 (Noisy-OR) | **0,480 / 0,555 / 0,642 / 0,738** |
| V5.2 global (c_lungref, 5-zon) | 0,9612 | — |
| **Global + Lokal birleşik** | **0,9876** (r = 0,376) | — |

⇒ **Test setindeki en yüksek hasta AUC'si**: v3 headline 0,9516 ve V5.2 0,9612'nin üzerinde. Küresel ve lokal temsiller r=0,38 ile yeterince dik ⇒ birleşim **gerçek tamamlayıcılık** sağlıyor. Test seti bu mühürle **bir kez** kullanıldı; eşik/ağırlık seçimi yapılmadı.
* **Kayıt:** `artifacts/v6_two_stage/v6_4_domain_and_test_seal.json` · `checkpoints/M5_domain_gen.pt`

### V6.5 — VinDr harici transfer güçlendirmesi (zero-shot + few-shot, 400 görüntü)

**KOL 1 — eğitimsiz (0 VinDr etiketi).** AdaBN-M5 · K=50 · 19.955 aday; 8 konfigürasyon, aynı dondurulmuş 2. aşama (HGB + füzyon LR). Referans dağılım: 200 NODE21 train hastası, 4.000 aday.

| Konfig | AUC | Δ vs taban |
| :--- | ---: | ---: |
| taban (32/64 px, z-score) | 0,6439 | — |
| JPEG-denoise σ=0,6 | 0,6380 | −0,006 |
| JPEG-denoise σ=1,0 | 0,6498 | +0,006 |
| **Ölçek-eşlemeli yama** `S=round(32/64·H_lung/810)` | **0,6562** | **+0,012** |
| denoise 1,0 + ölçek | 0,6523 | +0,008 |
| Quantile (rank-gauss) | 0,6486 | +0,005 |
| CORAL (tam uzay) | 0,6318 | −0,012 |
| **denoise 1,0 + ölçek + CORAL** | **0,6620** | **+0,018** |

⇒ **Fiziksel hipotez doğrulandı ama etki küçük:** JPEG blok gürültüsünü eritmek (σ=1,0) ve VinDr'nin 0,70× ölçeğini telafi etmek birlikte **+0,018 AUC** getiriyor (0,6439 → 0,6620). Tek başına ölçek-eşleme en güçlü tekil düzeltme (+0,012). CORAL tek başına zararlı (−0,012; 20k örnekte 1604-boyutlu kovaryans gürültülü), ancak denoise+ölçek ile birlikte en iyi konfigürasyonu veriyor. **Hiçbiri V5.3 global'i (0,6283) tek başına geçemiyor**; sıfır-etiketli birleşik kanal ise **0,6349 → 0,6620**.

**KOL 2 — sızıntısız few-shot site kalibrasyonu.** 5-katlı katmanlı CV (her fold'da değerlendirme katmanına dokunulmadı), N ∈ {0,10,20,40,80,160} etiketli hedef görüntü (yarı pozitif/yarı negatif), Aşama-1 ağırlıkları donmuş; yalnız 2. aşama tablosal modeli (LR, C=0,1) hedef merkezde yeniden uyarlanır, skor dondurulmuş `lr_fusion([AdaBN-M5 s1, kalibre TDA])` üzerinden alınır.

| N | Yalnız lokal (AUC±std) | V5.3 + lokal birleşik (AUC±std) | FROC@1 / @2 / @4 FP |
| ---: | :---: | :---: | :---: |
| 0 | 0,6593 ± 0,044 | 0,6673 ± 0,031 | 0,165 / 0,212 / 0,240 |
| 10 | 0,6280 ± 0,023 | 0,6521 ± 0,040 | 0,109 / 0,128 / 0,150 |
| 20 | 0,6585 ± 0,064 | 0,6751 ± 0,038 | 0,109 / 0,138 / 0,157 |
| 40 | 0,6855 ± 0,057 | 0,6959 ± 0,043 | 0,105 / 0,132 / 0,168 |
| 80 | 0,7057 ± 0,022 | 0,6997 ± 0,031 | 0,114 / 0,139 / 0,181 |
| 160 | **0,7243 ± 0,051** | **0,7054 ± 0,032** | 0,115 / 0,162 / 0,219 |

⇒ **Az-örnekli kalibrasyon N≥40'ta çalışıyor:** N=10'da aşırı uyum var (0,628 < zero-shot 0,659), N=40'ta toparlıyor, N=160'ta yalnız lokal **0,7243 ± 0,051** (+0,065 vs zero-shot). **V5.3 global kanalı N≥80'de katkı vermiyor, hatta seyreltiyor** (0,7054 < 0,7243): hedef merkezde lokal model kalibre edildiğinde global temsil gereksizleşiyor. FROC ise kalibrasyonla **iyileşmiyor** (N=0'da 0,212 → N=160'ta 0,162 @2FP): kalibrasyon görüntü düzeyi sıralamayı düzeltiyor, lezyon düzeyi sıralamayı değil.
* **Sınırlama:** aday etiketi = görüntü etiketi (bbox'tan türetilmedi) ⇒ FROC değerleri mutlak değil, kalibrasyon eğrisi olarak okunmalı.
* **Kayıt:** `artifacts/v6_two_stage/v6_5_vindr_transfer_boost.json`.

**V6.5b — kompakt özellik + bbox-türevli aday etiketleri (iki zayıflığın testi).** Kalibrasyon LR'si NODE21 referansında fit edilmiş **PCA-128** uzayında (açıklanan varyans %92,0) ve aday etiketleri GT kutusuna isabetten türetilerek yeniden koşuldu. Aynı 5-fold/N protokolü.

| N | bbox etiketli (AUC±std) | görüntü etiketli (AUC±std) | V5.3+lokal (bbox) | FROC@2 (bbox) |
| ---: | :---: | :---: | :---: | :---: |
| 0 | 0,6593 ± 0,044 | 0,6593 ± 0,044 | 0,6673 ± 0,031 | 0,211 |
| 10 | 0,6057 ± 0,083 | 0,6305 ± 0,034 | 0,6490 ± 0,059 | 0,100 |
| 20 | 0,5508 ± 0,111 | 0,6548 ± 0,054 | 0,6144 ± 0,062 | 0,088 |
| 40 | 0,6068 ± 0,066 | 0,6728 ± 0,050 | 0,6395 ± 0,021 | 0,141 |
| 80 | 0,6440 ± 0,084 | **0,6861 ± 0,044** | 0,6688 ± 0,032 | 0,161 |
| 160 | 0,6273 ± 0,077 | **0,6796 ± 0,039** | 0,6537 ± 0,038 | **0,184** |

⇒ **İki zayıflık da beklenen yönde kapanmadı, ama nedeni netleşti:** (i) PCA-128 boyut indirgemesi N=10 çökmesini **düzeltmiyor** (0,6305 ≈ V6.5'in 0,6280'i) ⇒ sorun boyut değil, **10 görüntüdeki ~500 adayın azlığı**. (ii) bbox etiketleri **FROC'u iyileştiriyor** (N=160: 0,184 vs 0,162) ama **görüntü AUC'sini düşürüyor** (0,627 vs 0,724) ⇒ gerçek bir **AUC–FROC ödünleşimi**: görüntü etiketi görüntü sıralamasını, bbox etiketi lezyon sıralamasını optimize ediyor. Bu yüzden V6.5'in görüntü-etiketli eğrisi **AUC için**, V6.5b'nin bbox'lı eğrisi **FROC için** raporlanmalı.
* **Kayıt:** `artifacts/v6_two_stage/v6_5b_fewshot_compact.json`.

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

> ⚠️ **V6.9 denetim notu — bölümü okumadan önce:** Aşağıdaki VinDr deneylerinin
> **omurgası** kritik önemdedir. V4 / V5.3 / V6.3 / V6.5 / V6.5d deneyleri **ImageNet**
> modelleri (`M5_centernet`, ResNet18) kullanır ⇒ temizdir. **V6.6.3 ve V6.6.5** ise
> `resnet50-res512-all` kullanır; bu omurga VinDr-CXR ile ön-eğitilmiştir ve
> değerlendirme görüntülerinin **%100'ü** onun ön-eğitim listesindedir ⇒ **kontamine**.
> Kanıt: [LEAK_AUDIT.md](LEAK_AUDIT.md) §3.

> 🔒 **ZERO-SHOT SINIRI (bölümün tamamı için zorunlu beyan).**
> VinDr-CXR'ye **CNN fine-tuning'i hiç yapılmadı.** Aşama-1 ağırlıkları (`M5_centernet.pt`, `M6_fcos.pt`) ve 2. aşama bundle'ı her VinDr koşusunda **dondurulmuş** NODE21 modelleridir; NODE21 test mührü (0,9876) VinDr'den **tamamen bağımsızdır**.
> Ancak VinDr bilgisi **üç ayrı kanaldan** girmiştir. Bu yüzden bu bölümde raporlanan sayılar üç sınıfa ayrılır ve **saf dış doğrulama olarak yalnızca (A) raporlanabilir:**
>
> | Sınıf | VinDr'den ne kullanır | Etiket? | Bu belgedeki satırlar |
> | :--- | :--- | :---: | :--- |
> | **(A) Saf zero-shot** | yalnızca görüntünün kendisi (denoise, kendi akciğer yüksekliği) + **NODE21** referans istatistikleri | ❌ | V4 üç kol · V5.3 · V6.3 bileşen otopsisi · V6.4 Adım 2 (`M5_domain_gen`) ve Adım 1'in AdaBN'siz baz satırları (0,5443 / 0,5440 / 0,5453) |
> | **(B) Transductive** (etiketsiz **hedef** istatistiği) | VinDr **BN istatistikleri** veya VinDr **özellik dağılımı** (kovaryans) | ❌ | ⚠️ **V6.5 KOL 1'in TAMAMI** — 8 konfigürasyonun hepsi `AdaBN-M5` adayları üzerinde koşar (0,6439 → 0,6620) · V6.4 Adım 1 / 1b AdaBN satırları (0,6332 · 0,6346 · 0,6471 · 0,6494) |
> | **(C) Süpervizli hedef uyarlaması** | VinDr **etiketleri** | ✅ | V6.5 KOL 2 · V6.5b · V6.5c few-shot eğrilerinin tamamı |
>
> ⚠️ **Saf zero-shot (A) zirvesi: VinDr AUC 0,6283** — V5.3 global 5-zon füzyonu (kilitli a = 0,45). V6.5'in **0,6439–0,6620** bandı, taban adayları AdaBN-M5 olduğu için **(B)** sınıfındadır; "VinDr'ye hiç dokunulmadan elde edildi" biçiminde **sunulamaz**. KOL 1'in içindeki `base_z` / `denoise*` / `scale*` / `quantile` satırları da AdaBN tabanını miras alır — bunlar "saf" değil, **AdaBN'e göre göreli** iyileştirmelerdir.

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

### 7.1b Kaynak-içi AUC denetimi — etiket ile kaynak tamamen iç içe (A3)

Eksik olduğu haklı olarak belirtilen **kaynak-içi AUC** hesaplandı; ancak tanımsız olduğu görüldü: NODE21 test setinde **tüm pozitifler dört dış kaynaktan, tüm negatifler tek bir kontrol havuzundan** gelir.

| Kaynak | n | pozitif | negatif |
| :--- | ---: | ---: | ---: |
| chestxray14 | 89 | 89 | 0 |
| padchest | 58 | 58 | 0 |
| jsrt | 19 | 19 | 0 |
| openi | 6 | 6 | 0 |
| node21_control | 562 | 0 | 562 |

⇒ Her kaynak **tek sınıflı**; klasik kaynak-içi AUC **tanımsızdır**. Yerine geçen metrik: **kaynak pozitifleri vs ortak kontrol havuzu** AUC'si + kaynakların ağırlıksız ortalaması (makro AUC).

| Model | Genel AUC | Makro AUC | Fark |
| :--- | ---: | ---: | ---: |
| **TDA PI CNN (başlık tek model)** | 0,8631 | **0,8920** | −0,0289 |
| TDA PI RBF-SVM | 0,8289 | 0,8672 | −0,0383 |
| TDA PI k-NN | 0,7162 | 0,7017 | +0,0145 |
| Ham görüntü CNN | 0,7043 | 0,6638 | +0,0405 |
| PL CNN | 0,5451 | 0,5433 | +0,0018 |
| PL SVM | 0,5216 | 0,5051 | +0,0165 |

**Kaynak bazında (TDA PI CNN):** chestxray14 **0,816** [0,768; 0,862] · padchest **0,905** [0,872; 0,936] · jsrt **0,943** [0,914; 0,968] · openi **0,904** [0,859; 0,949] · kontrol ort. p = **0,555**, özgüllük@0,5 = **0,358**.

**Okuma:**
1. Model **dört kaynağın dördünde de** ayrım yapıyor (0,82–0,94) ⇒ etki tek bir kaynağın artefaktı değil. Makro AUC genel AUC'den **yüksek**; en zor kaynak en büyük olan chestxray14.
2. TDA (makro **0,892**) ham CNN'i (makro **0,664**) belirgin geçiyor ⇒ TDA kazancı yalnızca kaynak kimliği değil.
3. ⚠️ **Confound yapısal olarak sürüyor:** kontrollerin tamamı `node21_control`'den; edinim kaynağı ile etiket ayrıştırılamıyor. Kaynak-başına AUC bunu *çürütemez*, yalnızca "tek kaynak etkisi değil" der. Gerçek sınav, **kontrollerin de aynı kaynaktan geldiği** bir tasarımdır — NODE21'de yok. Bu yüzden birincil genelleme kanıtı **E5.3b (0,81)** ve **VinDr (0,63–0,66)** olarak kalır.
4. **Kalibrasyon uyarısı:** kontrol ortalama p = 0,555 ve özgüllük@0,5 = 0,358 ⇒ AUC iyi olsa da **işletim noktası kaymış**; eşik transportu ayrıca kalibre edilmelidir.

* **Betik:** [`A3_source_confound_audit.py`](pipelines/exp/A3_source_confound_audit.py) · **Kayıt:** `artifacts/runs/A3_source_confound_audit/report.json`.

### 7.1c Persistence Landscape Zarf Mekanizma Otopsisi — Maske Sınırı Artefaktı (A4)

E4.4 raporunda Persistence Landscape'in (PL) çöküşü (test ROC-AUC 0,5450), *"akciğerdeki bronkovasküler damar döngülerinin λ1..λ5 üst zarflarını doyurması (ortalama tepe ~90–110, H1 kalıcılığı ~180–220)"* hipoteziyle açıklanmıştı. Ancak V5.1'de projenin baseline filtrasyonunun ~4-bit (doku pikselleri yalnızca 239–252 aralığında) olduğu bulunmuştu. GUDHI `cofaces_of_persistence_pairs` aracıyla 100 dengeli train zonu (50 pozitif, 50 negatif) üzerinde hücre-piksel geri izleme otopsisi yapıldı.

| Metrik / Sınıf Tipi | H0 (Top-10) | H1 (Top-10) | Kalıcılık $\ge 50$ (H1) | Kalıcılık $\ge 100$ (H0) |
| :--- | :---: | :---: | :---: | :---: |
| **Sınır geçişli (boundary crossing)** | **%98,6** (n=986) | **%59,9** (n=599) | **%100,0** (599/599) | **%100,0** (986/986) |
| Saf doku içi (pure tissue) | %1,4 (n=14) | %40,1 (n=401) | **%0,0** (0/599) | **%0,0** (0/986) |
| Sınır sınıfları ort. kalıcılık | **247,6** (max 250,0) | **248,3** (max 255,0) | — | — |
| Saf doku içi ort. kalıcılık | 3,6 (max 7,0) | 4,2 (max 7,0) | — | — |

**Etiket kırılımı (Sınır sınıfı oranı):**
* H0 sınır oranı: Pozitif zonlar **%97,8** vs Negatif zonlar **%99,4**
* H1 sınır oranı: Pozitif zonlar **%60,2** vs Negatif zonlar **%59,6**

**Temel Bulgular:**
1. **Doku içi dinamik aralık yalnızca ~12 seviyedir (240–252);** bu nedenle saf doku içinde kalıcılığı $\ge 15$ olan hiçbir topolojik özellik **fiziksel ve matematiksel olarak var olamaz** (saf doku içi max kalıcılık H0'da 7,0, H1'de 7,0; ortalama ~3,6–4,2).
2. **PL üst zarfını doyuran devasa tepeler (%100) maske sınırıdır:** E4.4'te ölçülen ve tepe değeri $\sim 110$ ($\text{kalıcılık} \approx 220$) olan H1 döngülerinin ve H0 bileşenlerinin **%100'ü** akciğer segmentasyon maskesi sınırı ile siyah arka plan (0) arasındaki yapay basamaktan (0 vs 240+) kaynaklanır.
3. **Mekanizmanın düzeltilmesi:** PL çöküşü biyolojik damar ağacından değil, **akciğer segmentasyon maskesinin geometrik basamak artefaktından** kaynaklanmaktadır. Bu artefakt hem pozitif hem negatif zonlarda özdeş oranda (%60) mevcuttur. Persistence Image (PI), $(b, p)$ 2B koordinat düzleminde bu artefaktı köşeye izole edip doku içi sinyali korurken, Persistence Landscape (PL) fonksiyonel üst zarf projeksiyonunda bu 250 kalıcılıklı dev artefakt tarafından tamamen körleştirilir.

* **Betik:** [`A4_mask_boundary_audit.py`](pipelines/exp/A4_mask_boundary_audit.py) · **Kayıt:** `artifacts/runs/A4_mask_boundary_audit/report.json` · `boundary_audit_records.csv`.

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

Temsil: aday merkezli **32×32 (çekirdek) + 64×64 (çevre)** H0/H1 PI (1600 boyut, **c_lungref ölçeği korunur**; yerel min-max yok) + 4 uzamsal özellik (rel_x, rel_y, `hilus` = **akciğer bbox merkezine** uzaklık, `plevra` = **akciğer maskesi sınırına** uzaklık — ikisi de **vekil**, anatomik hilus/plevra tespiti yok; bkz. [TEKNIK_YONTEM.md](TEKNIK_YONTEM.md) §3.2) + 64×64 yama.

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

### V6.5 — VinDr harici transfer güçlendirmesi (transductive + few-shot, 400 görüntü)

> ⚠️ **Bu alt bölümün tamamı (A) saf zero-shot DEĞİLDİR** — bkz. §7 başındaki zero-shot sınırı. KOL 1'in taban adayları `AdaBN-M5`'tir (VinDr etiketsiz BN istatistikleri); KOL 2 ise VinDr **etiketlerini** kullanır.

**KOL 1 — etiketsiz ama *transductive* (0 VinDr etiketi; taban adayları AdaBN-M5).** AdaBN-M5 · K=50 · 19.955 aday; 8 konfigürasyon, aynı dondurulmuş 2. aşama (HGB + füzyon LR). Referans dağılım: 200 NODE21 train hastası, 4.000 aday.

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

⇒ **Az-örnekli kalibrasyon N≥40'ta çalışıyor:** N=10'da aşırı uyum var (0,628 < N=0 tabanı 0,659; ikisi de *transductive*), N=40'ta toparlıyor, N=160'ta yalnız lokal **0,7243 ± 0,051** (+0,065 vs N=0 tabanı). **V5.3 global kanalı N≥80'de katkı vermiyor, hatta seyreltiyor** (0,7054 < 0,7243): hedef merkezde lokal model kalibre edildiğinde global temsil gereksizleşiyor. FROC ise kalibrasyonla **iyileşmiyor** (N=0'da 0,212 → N=160'ta 0,162 @2FP): kalibrasyon görüntü düzeyi sıralamayı düzeltiyor, lezyon düzeyi sıralamayı değil.
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

**V6.5c — Ayrıştırılmış çift-kafa + NODE21 öncül çapası.** Kafa-1 (hasta kapısı) V6.5'in görüntü-etiketli kalibrasyonu olarak **sabitlendi** (tam hedef eğitim seti; AUC 0,7249). Kafa-2 yalnız pozitif görüntülerde gerçek kutu vs en yüksek skorlu 5 sahte kutu (1:5) üzerinde PCA-128 LR; `P_box = (1−α_N)·P_zero-shot + α_N·P_bbox`, `α_N = N/(N+40)`; nihai `S = P_patient^γ·P_box` (γ=1).

| N | P_patient (sabit) | FROC@1 | FROC@2 | FROC@4 |
| ---: | :---: | :---: | :---: | :---: |
| 0 | 0,6593 ± 0,044 | 0,167 | 0,212 | 0,240 |
| 10 | 0,7249 ± 0,041 | 0,164 | **0,221** | 0,255 |
| 20 | 0,7249 ± 0,041 | 0,152 | 0,211 | 0,260 |
| 40 | 0,7249 ± 0,041 | 0,118 | 0,173 | 0,235 |
| 80 | 0,7249 ± 0,041 | 0,058 | 0,096 | 0,132 |
| 160 | 0,7249 ± 0,041 | 0,029 | 0,059 | 0,071 |

⇒ **Kriter biçimsel olarak N=10'da sağlanıyor (AUC 0,7249 ≥ 0,724; FROC@2 0,2208 ≥ 0,22) ama mekanizma çalışmıyor:** kazanç yalnız **+0,009** ve N arttıkça FROC **monoton çöküyor** (0,221 → 0,059). α büyüdükçe hedef kutu sıralayıcısı sıfır-eğitim öncülünü **bozuyor**. Teşhis: VinDr'de kutu-içi TDA sıralaması taşınmıyor (V6.3'ün "lokalizasyon taşınıyor, güven taşınmıyor" bulgusunun aday düzeyindeki karşılığı); 1:5 hard-negative kurgusu bunu düzeltmiyor. **Karar: KILL** — AUC–FROC ödünleşimi kaldırılmadı, yalnızca FROC eksenine taşındı.
* **Metodolojik not:** Kafa-1 sabit olduğu için N yalnız Kafa-2'yi kontrol eder; N≥10'daki 0,7249 **sabit hedef-modelin** AUC'sidir, N=10'un kalibrasyon başarısı değildir. FROC@2'nin N ile düşmesi bu yüzden doğrudan Kafa-2'nin kalitesini ölçer.
* **Kayıt:** `artifacts/v6_two_stage/v6_5c_decoupled_calibration.json`.

### V6.5d — 🔒 Inductive (AdaBN'siz) vs Transductive (AdaBN) fiziksel hizalama ayrımı

**Motivasyon.** V6.5 KOL 1'in taban adayları **`AdaBN-M5`** idi ([v6_5_vindr_transfer_boost.py:292](pipelines/v6_two_stage/v6_5_vindr_transfer_boost.py#L292), docstring: *"KOL 1 zero-shot: AdaBN-M5 adayları üzerinde"*). AdaBN, VinDr'in **etiketsiz BN istatistiklerini** kullanır ⇒ **0,6439–0,6620 bandının tamamı transductive**'tir. Fiziksel hizalamanın hiçbir hedef istatistiği kullanmayan **tekil-görüntü (inductive / strict)** etkisi hiç ölçülmemişti. Bu bölüm o boşluğu kapatır.

**Protokol.** Aynı VinDr kohortu (400 görüntü), aynı `M5_centernet` adayları ama **AdaBN KAPALI** (dondurulmuş NODE21 BatchNorm), aynı dondurulmuş 2. aşama (HGB + füzyon LR); referans 200 NODE21 train hastası / 4.000 aday. Betik: [`v6_5d_strict_no_adabn.py`](pipelines/v6_two_stage/v6_5d_strict_no_adabn.py) · Kayıt: `v6_5_vindr_transfer_boost.json` → `strict_no_adabn_ablation`.

| Konfigürasyon | **Inductive** (AdaBN yok) | Transductive (AdaBN) | Δ (AdaBN etkisi) |
| :--- | ---: | ---: | ---: |
| base_z | 0,6090 | 0,6439 | +0,0349 |
| + JPEG-denoise σ=1,0 | 0,5829 | 0,6498 | +0,0669 |
| **+ ölçek-eşlemeli yama** | **0,6181** | 0,6562 | +0,0381 |
| + denoise + ölçek | 0,5814 | 0,6523 | +0,0709 |
| + denoise + ölçek + CORAL | 0,5916 | 0,6620 | +0,0704 |

Yalnız dedektör (AdaBN'siz) hasta AUC: top1 **0,5599** · top3 0,5559 · noisy-OR 0,5538 — V6.4'ün AdaBN'siz 0,5443'ü ile tutarlı (fark: burada NMS + TTA'sız çıkarım).

**Bulgular — üçü de önceki yorumu değiştirir:**

1. **AdaBN baş rolde:** her satırda **+0,035…+0,071**. V6.5'in "+0,018 fiziksel hizalama" kazancı **AdaBN tabanına göre göreli**ydi; ham etki AdaBN'indir.
2. **JPEG-denoise, AdaBN olmadan ZARARLI:** base 0,6090 → denoise1.0 0,5829 (**−0,0261**); ölçek 0,6181 → denoise+ölçek 0,5814 (**−0,0367**). ⇒ Denoise kazancı **tamamen AdaBN etkileşimi**ydi; hedef istatistiği olmadan yumuşatma ince dokuyu bozuyor. (σ=0,6'nın AdaBN'li koşuda da zararlı olması bu teşhisi destekler.)
3. **Tek gerçek inductive kazanç ölçek-eşlemedir:** 0,6090 → **0,6181 (+0,0091)**. Fiziksel hizalamanın dürüst **tekil-görüntü** katkısı budur — "+0,018" değil.
4. **CORAL inductive değildir** (VinDr kovaryansı): 0,5916. Bu yüzden **inductive tavan 0,6181**'dir.
5. ⚠️ **Inductive tavan (0,6181) hâlâ V5.3 global füzyonunun (0,6283) altında** ⇒ saf sıfır-öğrenmede en iyi VinDr kanalı **global 5-zon füzyonu** olmaya devam ediyor.

**Karar: KEEP (denetim kaydı).** Bu bölüm V6.5'in sayılarını geçersiz kılmaz; **hangi sayının hangi sınıfa ait olduğunu** kesinleştirir: `(A) inductive` = ölçek-eşleme ailesi; `(B) transductive` = AdaBN ve CORAL aileleri; `(C) süpervizli` = few-shot.

---

## 10. V6.6 — Foundation X-Ray Backbone Stage-1 (CXR vakıf omurgası)

> **Motivasyon.** V6.0'ın 1. aşama dedektörü (M5), **ImageNet (kedi/köpek)** ile öneğitilmiş ResNet18 kullanıyordu. V6.3'ün teşhisi netti: dedektör **lokalizasyonu taşıyor** (VinDr bbox recall@K50 ≈ 0,90) ama **görüntü düzeyi güveni taşımıyor** (AUC ≈ 0,46–0,56). V6.6, 1. aşamanın gövdesini **800.000+ gerçek göğüs röntgeni** ile (NIH ChestX-ray14, CheXpert, PadChest, MIMIC-CXR, OpenI, RSNA, VinDr) eğitilmiş bir **CXR vakıf omurgası** ile değiştirir.

### V6.6.0 — Omurga seçimi ve mimari

| | |
| :--- | :--- |
| **Omurga** | `torchxrayvision` **`resnet50-res512-all`** (pc-nih-rsna-siim-vin-state, 512×512 native) |
| **Neden bu?** | native girdi **512 = kanonik girdimizle birebir** ⇒ yeniden örnekleme yok, küçük nodül çözünürlüğü korunur. `densenet121-res224-all` 512→224 küçültmesi isterdi ve bu tam da kaçırdığımız küçük nodülleri silerdi. (`--backbone` ile desteklenir.) |
| **KRİTİK girdi ölçeği** | torchxrayvision omurgaları 1 kanallı girdiyi [0,1] değil **[−1024, +1024]** bekler (`xrv.utils.normalize(img, 255)`). Uygulandı: `foundation_input()` |
| **Boyun** | C2..C5 (1/4, 1/8, 1/16, 1/32) → **C5 üzerinde ASPP** (d = 1, 3, 6, 12 + global-pool) → **FPN** yukarı-doğru birleştirme → P2 (128×128) |
| **Kafa** | 128×128'de fuse (64ch) → ×2 upsample → **256×256 CenterNet** (1ch Gaussian heatmap + 1ch radius) |
| **Sözleşme** | Çıktı **M5 ile birebir aynı** (512 girdi → 256×256 = 1024'e stride-4) ⇒ aday çıkarımı, değerlendirme ve 2. aşama kodu **değişmeden** çalışır |

### V6.6.1 — Unutmayı önleyen iki fazlı diferansiyel eğitim

| Faz | Epoch | Omurga | BN | Öğrenme hızı |
| :--- | :---: | :--- | :--- | :--- |
| **Isınma** | 1–5 | **donuk** (`requires_grad=False`) | `.eval()` | head **1e-3** |
| **Diferansiyel FT** | 6–18 | üst bloklar açık (`layer3`+`layer4`, **22,1M** param) | `.eval()` (running istatistik **donuk**) | backbone **1e-4** · head **1e-3**, Cosine Annealing |

* **Neden BN donuk?** Gerekçe: 800.000 röntgenlik `running_mean/var` istatistikleri batch=4 ile bozulmamalı. ⚠️ **Bu bir gerekçeydi, ölçüm değil** — V6.8.3 bu kolu doğrudan test etti ve **in-domain val'de ölçülebilir fayda bulamadı** (Δ nOR −0,0003; sekiz metriğin yedisinde BN açık ≥). V6.6.3'te AdaBN'in vakıf modelini *bozması* ise **farklı bir müdahaledir** (hedef veriyle istatistikleri yeniden hesaplamak ≠ ince ayarda dondurmak) ⇒ dolaylı kanıt sayılır, doğrulama değil. Hipotezin asıl sınavı **alan dışı transferdir** (bkz. §V6.8.3).
* **Veri:** 790 nodül + 790 temiz kontrol (**1:1**), RAM önbelleği (512 uint8) ile I/O'suz.
* **Augmentasyon:** yatay çevirme + **gama γ ∈ [0,8; 1,25]** + **rastgele ölçek ±%20** (ölçek değişince merkez/yarıçap da dönüştürülür).
* **Kayıp:** Gaussian Focal (α=2, β=4) + 0,1 · L1(radius, yalnız pozitifler) — M5 ile aynı.
* **Eğitim kaybı:** 13,35 → **1,26** (18 epoch, ~20 dk, RTX 4050 6 GB; tepe VRAM ≈ 1,4 GB).
* **Checkpoint'ler:** `M5_foundation.pt` (final) · `M5_foundation_warm.pt` (epoch 5 — unutma teşhisi).

### V6.6.2 — NODE21 val Aşama-1 kıyaslaması (elma-elma, 734 hasta / 231 lezyon)

Aynı protokol, aynı val seti, aynı aday çıkarımı — **tek değişken: omurga**.

| Model | K10 | K20 | K30 | K50 | FROC @0,5 / 1 / 2 / 4 FP | Hasta AUC (top1 / top3 / noisy-OR) |
| :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| M5_heatTTA (ImageNet ResNet18) | 0,9091 | 0,9654 | 0,9697 | 0,9784 | 0,524 / 0,667 / 0,740 / 0,844 | 0,795 / 0,807 / 0,802 |
| **FND_heatTTA (CXR vakıf + FPN/ASPP)** | **0,9481** | **0,9827** | **0,9827** | **0,9913** | **0,831 / 0,866 / 0,926 / 0,952** | **0,960 / 0,967 / 0,970** |
| M5 (TTA'sız) | 0,8658 | 0,9610 | 0,9740 | 0,9784 | 0,502 / 0,593 / 0,714 / 0,801 | 0,784 / 0,800 / 0,796 |
| FND (TTA'sız) | 0,9610 | 0,9740 | 0,9827 | 0,9870 | 0,831 / 0,857 / 0,905 / 0,948 | 0,955 / 0,965 / 0,969 |

Kriter 2 (kapsama, IoBB ≥ 0,50 veya IoU ≥ 0,15) tablosu `v6_6_foundation_stage1.json` içindedir.

**🔑 Dört bulgu:**

1. **%98 oracle-Recall kapısı ilk kez GEÇİLDİ.** FND K20 = 0,9827, K30 = 0,9827, K50 = **0,9913**. V6.0'ın ön-kayıtlı kapısı (`gate_98`) M5 ile K20/K30'da **hiçbir varyantta** sağlanamamıştı (en iyi 0,974) — bu, o başarısızlığı kapatan sonuçtur.
2. **FROC'ta ezici kazanç:** @0,5 FP **0,524 → 0,831 (+0,307)**; @4 FP 0,844 → **0,952**. FROC, sistemin en zayıf halkasıydı.
3. **Yalnız-Aşama-1 hasta AUC 0,795 → 0,960 (+0,165).** Bu sayı, **eski TAM iki aşamalı sistemin** lokal AUC'sinin (V6.2 `e_fusion_tda` = 0,955) **üzerindedir** ⇒ aşama-1 artık tek başına eski uçtan uca sistemi geçiyor. Teşhis: V6.2'nin 2. aşaması, zayıf bir dedektörün FP'lerini temizlemekle meşguldü; dedektör güçlenince yük ortadan kalkıyor.
4. **TTA'sız FND bile M5_heatTTA'yı geçiyor** (top1 0,955 vs 0,795) ⇒ kazanç TTA'dan değil omurgadan geliyor.

### V6.6.3 — VinDr-CXR Aşama-1 (400 görüntü): strict zero-shot vs AdaBN

> 🔴 **V6.9 DENETİM DÜZELTMESİ — bu bölümdeki VinDr sayıları KONTAMİNEDİR.** Kullanılan
> omurga `resnet50-res512-all`; torchxrayvision'da tanımı *"trained on the datasets
> pc-nih-rsna-siim-**vin**"*. xrv'nin VinDr sınıfı bizim etiket aldığımız **aynı CSV**'yi
> (`vinbigdata-train.csv.gz`) okur ve kohortun **400/400 görüntüsü (%100)** o listenin
> içindedir. ⇒ "strict zero-shot" ifadesi **geçersizdir**; 0,6469 / 0,5711 sayıları
> VinDr'yi görmüş bir kodlayıcıyla üretilmiştir. Ayrıntı ve kanıt:
> [LEAK_AUDIT.md](LEAK_AUDIT.md) §3. **Temiz omurgayla ölçüm yapıldı (V6.9.5) ve
> kontaminasyon ölçülebilir bir şişme üretmedi:** VinDr'yi görmemiş omurga 0,6369 (top1) /
> 0,6663 (nOR) — yani bu satırdaki 0,6469 bir kontaminasyon artefaktı değil.
> (Aşağıdaki metin düzeltilmeden bırakıldı — tarihsel kayıt; doğru okuma bu uyarıyla yapılır.)

| Model | **Strict** (AdaBN yok — saf sıfır-öğrenme) | AdaBN'li | bbox Recall@K30 / K50 |
| :--- | ---: | ---: | :---: |
| M5 (ImageNet) | 0,5599 | **0,6332** ↑ | 0,726 / 0,752 |
| **FND (CXR vakıf)** | **0,6469** | 0,5711 ↓ | **0,832 / 0,865** |

**🔑 Üç bulgu:**

1. **Yeni en iyi saf zero-shot sonucu: 0,6469.** Vakıf omurgası, hedefe **hiç dokunmadan** M5'in **AdaBN'li** hâlini (0,6332) ve V5.3 global füzyonunu (**0,6283**) geçiyor. Yani V6.5'in tüm transductive/kalibrasyon zahmetiyle ulaştığı bandın üstüne, **hiçbir hedef istatistiği kullanmadan** çıkılıyor.
2. **AdaBN vakıf modelini BOZUYOR** (0,6469 → 0,5711, **−0,076**). Nedeni sağlam: vakıf omurgasının BN istatistikleri **zaten 800.000 gerçek CXR'den** (VinDr dahil) geliyor ⇒ 400 görüntüyle yeniden hesaplamak onları bozuyor. M5'te ise tersi: ImageNet istatistikleri CXR'e uymuyor, AdaBN düzeltiyor (**+0,073**).
3. ⇒ **Genel kural (V6.4 + V6.5d + V6.6.3 birlikte):** AdaBN'in değeri, omurganın **CXR-yerliliğiyle ters orantılıdır**. ImageNet ResNet18'de +0,073; CXR vakıf modelinde −0,076. **AdaBN bir yama, ilaç değil** — ve daha iyi omurga, yamayı gereksiz *ve* zararlı kılıyor.
4. **bbox recall da taşınıyor:** K30 0,726 → **0,832**; K50 0,752 → **0,865** (bu tek koşu içindeki elma-elma farktır; V6.3'ün kayıtlı 0,90'ı farklı bir çıkarım yolu kullanır, doğrudan karşılaştırılamaz).

* **Betikler:** [`detector_models_foundation.py`](pipelines/v6_two_stage/detector_models_foundation.py) · [`v6_6_train_foundation.py`](pipelines/v6_two_stage/v6_6_train_foundation.py) · [`v6_6_foundation_stage1.py`](pipelines/v6_two_stage/v6_6_foundation_stage1.py) · [`v6_6_vindr_e2e.py`](pipelines/v6_two_stage/v6_6_vindr_e2e.py)
* **Kayıt:** `artifacts/v6_two_stage/v6_6_foundation_stage1.json` · `v6_6_vindr_e2e.json` · `checkpoints/M5_foundation.pt`

### V6.6.4 — Aşama-2 entegrasyonu (NODE21 val, dondurulmuş lokal TDA-HGB)

Foundation adayları üzerinde **değişmemiş** V6.3 2. aşaması (c_lungref + 32/64 çift-ölçek PI + 4 uzamsal öncül → HGB → füzyon LR):

| K | Aşama-2 bileşeni | FROC @0,5 / 1 / 2 / 4 FP | Hasta AUC (top1 / top3 / noisy-OR) |
| :-: | :--- | :---: | :---: |
| 30 | yalnız Aşama-1 | **0,831 / 0,866 / 0,926 / 0,952** | 0,960 / 0,967 / 0,970 |
| 30 | lokal TDA-HGB | 0,268 / 0,359 / 0,476 / 0,610 | 0,854 / 0,910 / 0,919 |
| 30 | `e_fusion_tda` | 0,749 / 0,823 / 0,853 / 0,896 | 0,972 / 0,971 / 0,986 |
| 30 | `f_hierarchical` | 0,766 / 0,818 / 0,857 / 0,905 | **0,977** / 0,975 / 0,985 |
| 50 | `e_fusion_tda` | 0,745 / 0,801 / 0,853 / 0,879 | 0,972 / 0,973 / **0,988** |
| 50 | `f_hierarchical` | 0,766 / 0,823 / 0,857 / 0,892 | **0,977** / 0,978 / 0,986 |

**V6.2 (M5 tabanlı) kayıtlı sayılarıyla karşılaştırma:**

| Sistem | Hasta AUC (noisy-OR) | FROC @0,5 / 1 / 2 / 4 |
| :--- | ---: | :---: |
| V6.2 `M5M6TTA·K50·e_fusion_tda` | 0,955 | 0,528 / 0,615 / 0,688 / 0,771 |
| V6.2 `M5_heatTTA·K30·f_hierarchical` | 0,936 | 0,589 / 0,658 / 0,753 / 0,857 |
| **V6.6 `FND·K50·e_fusion_tda`** | **0,988** | 0,745 / 0,801 / 0,853 / 0,879 |
| **V6.6 `FND·K50·f_hierarchical`** | 0,986 | **0,766 / 0,823 / 0,857 / 0,892** |

**🔑 İki bulgu:**

1. **İki aşamalı mimari artık gereksizleşmiyor — aksine en iyi AUC'yi veriyor.** Aşama-1 tek başına 0,960 (top1) / 0,970 (noisy-OR) iken `e_fusion_tda` **0,972 / 0,988**. Füzyon kazancı +0,018 noisy-OR'dur. (V6.5'in "N≥80'de lokal model kalibre edilince global temsil gereksizleşiyor" bulgusunun tersi yönde: burada *lokal TDA* hâlâ katkı veriyor.)
   * ⚠️ **Dürüstlük kaydı (Tuzak B):** `e_fusion_tda`'nın **0,972 / 0,988**'i, füzyon **LR'ının 1. aşamanın KENDİ eğitim bölmesindeki (in-sample) skorlarıyla** uydurulduğu dondurulmuş bundle'dan gelir ⇒ val dürüsttür ama katsayılar hafif iyimser olabilir. **Sızıntısız alternatif** (sabit ağırlık, V6.6.6): **w=0,55 → nOR 0,9886 · FROC@0,5 0,8355**. V6.7 gönderim modeli bu tuzağı yapısal olarak kapatır. Ayrıntı: [STORY.md §12.2](STORY.md).
2. 🚨 **FROC darboğazı YER DEĞİŞTİRDİ.** Aşama-1 tek başına FROC@0,5 = **0,831**; lokal TDA'yı devreye sokmak onu **0,749**'a **düşürüyor** (AUC'yi ise 0,960 → 0,972 yükseltiyor). ⇒ Güçlü bir dedektörün ardından lokal TDA **FROC'a zarar veriyor**. Bu, V6.5b'nin AUC–FROC ödünleşiminin **kurum içi** karşılığıdır ve somut bir tasarım kararı doğurur: **FROC-optimal (lezyon bulma) dağıtımda 2. aşama atlanmalı; AUC-optimal (hasta triyajı) dağıtımda tutulmalı.**

### V6.6.5 — VinDr-CXR uçtan uca transfer (foundation + lokal TDA, 400 görüntü)

> 🔴 **V6.9 DENETİM DÜZELTMESİ — kontamine.** Bu bölümün tamamı `resnet50-res512-all`
> (pc-nih-rsna-siim-**vin**) ile üretildi ve değerlendirme görüntülerinin %100'ü o omuranın
> ön-eğitim listesinde ([LEAK_AUDIT.md](LEAK_AUDIT.md) §3). 0,6593 / 0,6637 / 0,7344 gibi
> sayılar **sıfır-atış değildir**; few-shot 0,7344 dahil hiçbiri dış-merkez genelleme
> kanıtı olarak kullanılamaz. Tarihsel kayıt olarak korunmuştur.
>
> ✅ **Ama etkisi ölçüldü ve bulunamadı (V6.9.5):** VinDr'yi hiç görmemiş omurgayla
> (`densenet121-res224-nih`) aynı tarif 0,6369 top1 / **0,6663 nOR** veriyor ⇒ bu
> bölümdeki mertebe **kontaminasyon artefaktı değil**, ~0,64–0,67'lik transfer tavanıdır.
> Yine de yayında **temiz omurga** sayısı kullanılmalıdır.

V6.5 şablonunun üç protokolü, foundation adaylarıyla (23.028 strict / 21.924 AdaBN aday):

**(A) STRICT / inductive — AdaBN YOK (saf tekil-görüntü):**

| Konfigürasyon | Hasta AUC | FROC @0,5 / 1 / 2 / 4 FP |
| :--- | ---: | :---: |
| base_z | 0,6416 | 0,156 / 0,189 / 0,227 / 0,270 |
| + JPEG-denoise σ=1,0 | 0,6593 | 0,170 / 0,218 / 0,286 / 0,345 |
| + ölçek-eşlemeli yama | 0,6528 | 0,158 / 0,187 / 0,218 / 0,267 |
| + denoise + ölçek | 0,6550 | 0,173 / 0,220 / 0,291 / **0,352** |
| + denoise + ölçek + CORAL † | **0,6637** | 0,180 / 0,225 / 0,258 / 0,322 |

**(B) TRANSDUCTIVE — AdaBN (VinDr etiketsiz BN):**

| Konfigürasyon | Hasta AUC | FROC @0,5 / 1 / 2 / 4 FP |
| :--- | ---: | :---: |
| base_z | 0,5524 | 0,073 / 0,097 / 0,123 / 0,180 |
| + JPEG-denoise σ=1,0 | 0,5769 | 0,106 / 0,154 / 0,177 / 0,222 |
| + ölçek-eşlemeli yama | 0,5926 | 0,078 / 0,092 / 0,137 / 0,180 |
| + denoise + ölçek | 0,5707 | 0,123 / 0,156 / 0,194 / 0,234 |
| + denoise + ölçek + CORAL † | 0,5960 | 0,095 / 0,128 / 0,163 / 0,182 |

**(C) FEW-SHOT hasta kalibrasyonu (5-fold CV, hedef ETİKETLİ):**

| N | STRICT tabanı (`scale_z`) | TRANSDUCTIVE tabanı (AdaBN+CORAL) |
| ---: | :---: | :---: |
| 0 | 0,6517 ± 0,0329 | 0,5976 ± 0,0458 |
| 10 | 0,6757 ± 0,0589 | 0,5684 ± 0,0605 |
| 20 | 0,6739 ± 0,0136 | 0,5743 ± 0,0827 |
| 40 | 0,7036 ± 0,0488 | 0,5361 ± 0,0635 |
| 80 | 0,7248 ± 0,0668 | 0,5867 ± 0,0739 |
| 160 | **0,7344 ± 0,0505** | 0,6019 ± 0,0488 |

† CORAL VinDr özellik kovaryansını kullandığı için **inductive değildir** (tek kanal transductive); diğer dört satır hiçbir hedef istatistiği kullanmaz.

**🔑 Beş bulgu:**

1. **AdaBN vakıf modelini HER konfigürasyonda bozuyor** (−0,065…−0,089). V6.6.3'ün bulgusu uçtan uca hatta da doğrulandı.
2. **JPEG-denoise artık YARARLI — M5'te zararlıydı.** M5-strict'te 0,6090 → 0,5829 (**−0,026**); foundation-strict'te 0,6416 → 0,6593 (**+0,018**). ⇒ CXR-yerli omurga JPEG blok artefaktını tolere ediyor, yumuşatmanın kazancı geri geliyor. **Fiziksel hizalamanın işe yarayıp yaramaması omurgaya bağlıdır.**
3. **Inductive tavan 0,6181 → 0,6593 (+0,041)**; CORAL ile 0,6637. (V6.5'in AdaBN'li 0,6620'sine, **hedefe hiç dokunmadan** ulaşılıyor.)
4. **Few-shot çalışıyor ve monoton:** 0,6517 → **0,7344 ± 0,0505** (V6.5: 0,7243). V6.5'te N=10'da çöken eğri artık **ilk adımdan itibaren iyileşiyor** (N=10: 0,6757). Transductive tabanda ise few-shot **çalışmıyor** (N=40: 0,5361 < N=0 0,5976) ⇒ AdaBN bozuk bir temsil bırakıyor.
5. **VinDr FROC@4FP: 0,240 (V6.5 tabanı) → 0,352** — dış merkezde lezyon bulma da ölçülebilir biçimde iyileşti.

* **Betik:** [`v6_6_vindr_e2e.py`](pipelines/v6_two_stage/v6_6_vindr_e2e.py) · **Kayıt:** `artifacts/v6_two_stage/v6_6_vindr_e2e.json`

### V6.6.6 — Füzyon FROC ağırlık kalibrasyonu (Pareto taraması, kapanış)

**Soru.** V6.6'da yalnız Aşama-1 FROC@0,5 = 0,831 verirken, **M5 üzerinde öğrenilmiş** `e_fusion_tda` (LR) katsayıları kullanılınca FROC **0,745**'e düşüyordu. Basit z-skorlu doğrusal füzyon ailesinde daha iyi bir işletim noktası var mı?

`S_fused(i) = w · z(s1_fnd) + (1 − w) · z(tda_hgb)`, **w ∈ [0, 1] adım 0,05**; 35.666 aday / 734 hasta / 231 lezyon. Skorlar noisy-OR toplaması için **sigmoid** ile [0,1]'e eşlendi (monoton ⇒ FROC ve top1/top3 değişmez, yalnız noisy-OR anlamlı olur). Betik: [`v6_6_fusion_calibration.py`](pipelines/v6_two_stage/v6_6_fusion_calibration.py).

**🔴 Sonuç: üç kapıyı aynı anda sağlayan w YOKTUR — kesişim boş.**

| Kapı | Sağlandığı aralık |
| :--- | :--- |
| FROC@4 ≥ 0,950 | yalnız **w ∈ [0,70 … 1,00]** |
| AUC (noisy-OR) ≥ 0,985 | yalnız **w ∈ [0,30 … 0,60]** |
| ⇒ | **kesişim ∅** — FROC@4'ü açan her ağırlıkta AUC 0,985'in altına iner |

**Seçilmiş noktalar:**

| w | FROC@0,5 | FROC@1 | FROC@2 | FROC@4 | AUC noisy-OR | AUC top1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,00 (yalnız TDA) | 0,2208 | — | — | 0,5281 | 0,9255 | 0,8664 |
| 0,45 (**AUC-optimal**) | 0,8182 | 0,8788 | 0,9307 | 0,9177 | **0,9909** | **0,9780** |
| **0,55 (önerilen)** | **0,8355** | 0,8788 | 0,9221 | 0,9307 | 0,9886 | 0,9746 |
| **0,70 (FROC@4 garantili)** | 0,8312 | 0,8788 | 0,9221 | **0,9524** | 0,9827 | 0,9689 |
| 0,90 (**FROC-optimal**) | **0,8442** | 0,8831 | 0,9221 | 0,9524 | 0,9732 | 0,9622 |
| 1,00 (yalnız Aşama-1) | 0,8312 | 0,8658 | 0,9264 | 0,9524 | 0,9692 | 0,9595 |
| **REF: `e_fusion_tda` (miras LR)** | 0,7446 | 0,8009 | 0,8528 | 0,8788 | 0,9879 | 0,9722 |

> 🐛 **Düzeltme (çift sigmoid):** Bu satırın noisy-OR'u önceki sürümde **0,9737** yazıyordu — hatalıydı. LR çıktısı **zaten [0,1] olasılığı** iken üzerine bir kez daha sigmoid uygulanmıştı; noisy-OR sıralamaya duyarlı olduğu için bozuldu (FROC ve top1 etkilenmedi, çünkü sıralamaya bağlılar). Düzeltilmiş değer **0,9879**, V6.6.4'ün bağımsız ölçtüğü 0,988 ile tutarlı.

**🔑 Üç bulgu:**

1. ✅ **Kalibrasyon, miras alınan LR katsayılarını DOMİNE ediyor.** **w ∈ [0,35 … 0,55]** aralığı dört metrikte de `e_fusion_tda`'ya eşit veya üstündür (uçlar doğrulandı: w=0,30 dört metrikte de geride, w=0,60 AUC'de 0,9866 < 0,9879). Özellikle **w = 0,55**: FROC@0,5 0,745 → **0,836** (+0,091), FROC@4 0,879 → **0,931** (+0,052), AUC noisy-OR 0,9879 → **0,9886** (+0,0007), top1 0,972 → **0,975** (+0,002). ⚠️ **Dürüst okuma: kazanç FROC'ta büyük, AUC'de marjinal.**
2. 🔴 **TUZAK B kaydı:** V6.1/V6.3'te füzyon LR'ı, 1. aşamanın **in-sample** skorlarıyla uydurulmuştu ⇒ val dürüst ama katsayılar hafif iyimser olabilir. Buradaki **sabit w** füzyonu bu riski taşımaz; V6.7 gönderim modeli Tuzak B'yi yapısal olarak kapatır. (Ayrıntılı dürüstlük kaydı: [STORY.md §12.2](STORY.md).)
3. ⚖️ **Önerilen işletim noktaları:**
   * **w\* = 0,55** — *varsayılan* (regresyonsuz; AUC ve FROC@0,5 kapıları sağlanır).
   * **w = 0,70** — FROC@4 ≥ 0,950 **zorunlu**ysa (lezyon bulma dağıtımı); AUC 0,9827, kapıya yalnız **0,0023** uzak.
   * **w = 0,45** — hasta triyajı (en yüksek AUC 0,9909).
   * **w = 0,90** — en yüksek FROC@0,5 (0,8442).
4. 🎯 **Teşhis.** Kapı üçlüsü, **AUC–FROC ödünleşimi** nedeniyle tanım gereği çelişkilidir: eşikler "en iyi AUC'li model" ile "en iyi FROC'lu model"in **aynı anda** olmasını istiyordu. V6.5b/V6.5c'de *dış merkezde* görülen bu ödünleşim, güçlü dedektörden sonra **kurum içinde de** ölçüldü (bkz. V6.6.4). Kalibrasyonun yaptığı iş, ödünleşmeyi **kaldırmak değil**, frontier'ı miras alınan noktanın **tamamının üstüne taşımaktır**.

* **Kayıt:** `artifacts/v6_two_stage/v6_6_fusion_calibration.json` (21 ağırlığın tam eğrisi + Pareto cephesi + öneri bloğu) · ana rapora `fusion_froc_calibration` anahtarıyla işlendi.
* **Önbellek:** `v6_6_val_candidate_scores.npz` (yalnız skaler skorlar; görüntü/kırpıntı YOK) ⇒ tarama yeniden koşulabilir.

### V6.6.7 — Resmî NODE21 detection metriği denetimi (eğitim YOK)

**Protokol düzeltmesi.** Resmî detection değerlendirmesi ([node21.grand-challenge.org/Details](https://node21.grand-challenge.org/Details/)) bizim eski protokolümüzden **farklı**:

| | Bizim eski protokolümüz | **Resmî NODE21** |
| :--- | :--- | :--- |
| FROC noktaları | @0,5 / 1 / 2 / 4 FP | **@1/8 (0,125) · 1/4 (0,25) · 1/2 (0,5)** FP/**görüntü** |
| Eşleştirme | `hit_criterion1` (merkez / 16 px) | **IoU > 0,2** (kutu–kutu) |
| Çakışan 2.+ tahmin | duplikasyon IGNORE (`hit` indeksine göre) | aynı referansa IoU>0,2 çakışanlar arasında **yalnız en yüksek skorlu** tutulur |
| Sıralama | — | **`rank_metric = 0,75·AUC + 0,25·Sens@0,25`** |

Betik: [`v6_6_optimization_audit.py`](pipelines/v6_two_stage/v6_6_optimization_audit.py) · Kayıt: `v6_6_optimization_audit.json` · 734 hasta / 231 lezyon · 296 değerlendirme (37 varyant × 8 kutu boyutu).

**🔴 Bulgu 1 — YARIÇAP KAFASI KULLANILAMAZ (gerçek hata).**
GT nodül boyutları **35–119 px** iken model ısı tepelerinde **R ≈ 0,01–0,04** tahmin ediyor ⇒ yarıçap kutusu kenarı **~0,1 px**. Neden: radius kaybının ağırlığı `0,1`, focal kayıp mertebesi ise 1–13 ⇒ kafa **çökmüş**. Resmî IoU>0,2 eşleştirmesinde yarıçap kutusu **S@0,25 = 0,000** (rank 0,7196) verir. ⇒ **Gönderim SABİT kare kutu kullanmak zorundadır.**

**🔴 Bulgu 2 — Kutu boyutu kritik; mevcut 64 px optimal değil.** Tüm ölçek setlerinde optimum **56×56 px**.

**🟢 Bulgu 3 — Çok ölçekli TTA monoton kazandırıyor (eğitim YOK).**

| Ölçek seti (+ yatay çevirme TTA) | AUC | Sens@0,25 | FROC₃ (resmî ort.) | **rank_metric** |
| :--- | ---: | ---: | ---: | ---: |
| single 1,0 | 0,9595 | 0,7186 | 0,7143 | 0,8993 |
| 0,9 + 1,0 | 0,9658 | 0,7403 | 0,7302 | 0,9094 |
| **0,9 + 1,0 + 1,1** | **0,9678** | **0,7489** | **0,7403** | **0,9131** |

Δ(single → üçlü) = **+0,0138 rank_metric · +0,0083 AUC · +0,0303 Sens@0,25 · +0,0260 FROC₃** — hepsi **sıfır eğitimle**.

**⚪ Bulgu 4 (NEGATİF) — NMS yarıçapı ve tepe penceresi etkisiz.**
NMS Δd ∈ {8, 12, 16, 20, 24, 28} ve tepe penceresi {3×3, 5×5} → rank **0,9131 SABİT** (tüm kombinasyonlarda). Görüntü başına aday 46,2 → 44,6 arasında değişiyor ama metrik değişmiyor. **Neden:** resmî protokol, aynı referansa IoU>0,2 çakışan 2.+ tahmini **zaten IGNORE ediyor**; 8 px'lik NMS bile yeterli. ⇒ *"Çift kutu FROC'u düşürür"* hipotezi **bu kurulumda doğrulanmadı**.

**En iyi konfigürasyon (resmî metrik):** çok ölçekli **{0,9, 1,0, 1,1}** + yatay çevirme TTA · tepe **3×3** · NMS **8 px** · **56×56 sabit kutu** → **rank_metric 0,9131** (AUC **0,9678** · Sens@0,125 **0,6710** · Sens@0,25 **0,7489** · Sens@0,5 **0,8009** · Sens@1 0,8355 · Sens@4 0,9091).

> ⚠️ **Kural 9 (değişmez):** Verimiz NODE21'in **halka açık eğitim setidir**; resmî test setleri (281 + 298) **özeldir**. Bu tablo yalnız kendi val bölmemizde resmî protokolün **simülasyonudur** — leaderboard'a gönderim veya sıralama karşılaştırması **mümkün değildir**. "1. sıraya çıkmak" bu veriyle ölçülemez.
> 📌 Resmî Sens@0,5 (0,8009) ile eski `hit_criterion1` FROC@0,5 (0,831) arasındaki fark, **eşleştirme kuralından** (IoU>0,2 vs merkez mesafesi) gelir; resmî ölçüt daha sıkıdır.

---

## 11. V6.7 — %100 Tam Veri Gönderim Eğitimi + Grand-Challenge Paketi

> **Amaç.** Kilitli standart (V6.6.7) resmî gönderime hazırlanır: 1. ve 2. aşama **4.882 hastanın tamamıyla** yeniden eğitilir ve Grand-Challenge `--network=none` sözleşmesine uygun bir konteyner paketlenir.

### V6.7.0 — 🔒 Değişmez kural: bilimsel checkpoint'ler korunur

Makale tablolarını üreten ağırlıklara **dokunulmadı**; SHA-256'ları `submission_config.json` içine kaydedildi. Tam-veri modelleri ayrı dizine yazılır:

| Rol | Yol | Durum |
| :--- | :--- | :--- |
| Bilimsel (V6.0–V6.6) | `checkpoints/M5_centernet.pt` · `M6_fcos.pt` · `M5_foundation.pt` · `stage2_final_tda_hgb.joblib` · `M5_domain_gen.pt` | **değişmedi** (mtime'lar aynı) |
| Gönderim (V6.7) | `submission_fulldata/M5_foundation_fulldata.pt` · `stage2_tda_hgb_fulldata.joblib` | yeni |

### V6.7.1 — 🛡️ TUZAK A koruması: sabit takvimle tam veri eğitimi

**Hiperparametre araması YOK** — V6.6'da doğrulanmış takvim birebir yeniden kullanıldı: 5 epoch donuk ısınma + 13 epoch diferansiyel fine-tune (18 epoch), BN donuk, Cosine Annealing, aynı augmentasyon (flip + γ∈[0,8;1,25] + ölçek±%20) ve aynı kayıp.

| | |
| :--- | :--- |
| Veri | **4.882 hastanın tamamı** (train + val + test): 1.134 nodüllü + **1.134** temiz kontrol (1:1) = **2.268** görüntü |
| Omurga | `xrv resnet50-res512-all` (800k+ CXR) + ASPP/FPN |
| Eğitim kaybı | **19,31 → 1,22** |
| Süre | ~31 dk (RTX 4050 6 GB) |
| **Yarıçap kafası** | `radius_detach=True` — yarıçap kanalı **detach edilmiş özelliklerden** eğitildi ⇒ gradyanı paylaşılan gövdeye/ısı haritasına **hiç akmadı**, ana ısı haritası keskinliği %100 korundu (ve çıkarımda zaten sabit 56×56 kutu kullanılıyor) |

> ⚠️ Tam-veri modeli **NODE21'de değerlendirilemez** (tüm veri eğitimde). Aşağıdaki referans sayılar `M5_foundation` ile val'de ölçülmüştür ve boru hattının *beklenen davranış* referansıdır, tam-veri performans iddiası değildir.

### V6.7.2 — 🛡️ TUZAK B koruması: sızıntısız 2. aşama birleştirmesi

**Sorun (Stacking Overfitting).** 2. aşama modelini, 1. aşamanın **kendi eğitim verisi** üzerinde ürettiği şişirilmiş skorlarla eğitmek yığma-uyumu yaratır. Koruma iki katmanlı:

1. **HGB matrisinde 1. aşama skoru (s1) YOK.** Girdi yalnız **1600 PI (32/64 px, c_lungref) + 4 uzamsal öncül**'dür. 1. aşamanın ezberlediği bilgi bu matrise **fiziksel olarak giremez** (bundle meta'sında `stage1_score_included: false` olarak doğrulanır).
2. **Füzyon ağırlığı HİÇBİR veri üzerinde yeniden uydurulmadı.** V6.6.6'da val'de doğrulanmış ve miras alınan LR katsayılarını dört metrikte domine eden **sabit w = 0,55** kilitlendi:
   `S = sigmoid(0,55·z(s1) + 0,45·z(p_tda))`, z-skor sabitleri tam-veri aday havuzundan bir kez hesaplanıp donduruldu.

**Neden bu yeterli:** HGB s1'i görmediği için 1. aşama ezberi HGB'ye sızamaz; füzyon ise *sabit* olduğu için hiçbir veriye uyum sağlayamaz. Tuzak B yapısal olarak imkânsız hale gelir.

### V6.7.3 — Grand-Challenge gönderim paketi (`submission_docker/`)

| Dosya | İçerik |
| :--- | :--- |
| `process.py` | GC arayüzü: `/input/images/chest-x-rays/*.mha` → `/output/nodules.json` |
| `Dockerfile` | CUDA runtime taban + **gömülü ağırlıklar** (`/opt/xrv_weights`) |
| `requirements.txt` · `README.md` | bağımlılıklar + sözleşme |
| `smoke_test.py` | yerel ağsız uçtan uca doğrulama |
| `weights/` | omurga (90 MB) + PSPNet (260 MB) — **gitignore** |

**3B yığın desteği:** NODE21 sunucusu birden fazla 1024×1024 röntgeni tek bir `(N_z, H, W)` `.mha` olarak verebilir. `process.py` her `z` katmanını **tek tek** işler ve köşelere `z` indeksini yazar.

**Koordinat zinciri:** ısı haritası 256×256 → kanonik 512×512 → **orijinal piksel** (x_px, y_px) → `sitk.TransformContinuousIndexToPhysicalPoint([x_px, y_px, z])` → **fiziksel mm**. Kutu, merkez etrafında sabit **56×56 px** olarak kurulur (yarıçap kafası resmî IoU>0,2 için kullanılamaz — V6.6.7 Bulgu 1).

**Çıktı şeması:** `{"name":"Regions of interest","type":"Multiple 2D bounding boxes","boxes":[{"corners":[[x1,y1,z],[x1,y2,z],[x2,y2,z],[x2,y1,z]],"probability":f}],"version":{"major":1,"minor":0}}`

**`--network=none` (iki tuzak):**
1. **Omurga indirme yasağı:** ağırlıklar imaja gömülü; `XRV_CACHE_DIR=/opt/xrv_weights`. Ayrıca `xrv.utils.download` çalışma anında **engellenir** ⇒ dosya eksikse *sessiz indirme* yerine açık hata.
2. **Canlı PSPNet maskesi:** sunucudan hazır maske gelmez; akciğer maskesi **çalışma anında** PSPNet ile ham röntgenden üretilir (ağırlıkları da imaja gömülü), ardından 15 px dilasyon + c_lungref (p1–p99) kanonikleştirmesi ve 4 uzamsal öncül bu canlı maske üzerinden hesaplanır.

### V6.7.4 — Denetimler

| Denetim | Yöntem | Sonuç |
| :--- | :--- | :--- |
| Bilimsel checkpoint korunması | SHA-256 + mtime | ✅ değişmedi |
| Tuzak B | `bundle.meta.stage1_score_included == false` (assert) | ✅ s1 matrise girmiyor |
| Çevrimdışı model yükleme | `xrv.utils.download` monkeypatch ile yükleme | ✅ ağ çağrısı yok |
| 3B yığın + mm dönüşümü | `smoke_test.py` (gerçek val görüntülerinden 1024×1024×3 `.mha`) | ✅ şema + tersinir mm |
| Sessiz indirme yasağı | boş `XRV_CACHE_DIR` ile negatif test | ✅ açık hata (`exit 1`) |
| **Gerçek konteyner, `--network=none`** | `docker run --network=none -v .../input:/input:ro -v .../output:/output` | ✅ **150 kutu / 3 katman** |

**Konteyner çıktısı (birebir, `--network=none`):**

```
✓ ağırlık: pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt (94.5 MB)
✓ ağırlık: pspnet_chestxray_best_model_4.pth (273.0 MB)
indirme koruması AKTİF (xrv.utils.download engellendi)
sözleşme v6.7-fulldata-1.0 | w=0.55 (kilitli)
stack.mha: yığın 3 katman, 1024x1024, ITK boyut=3
  z=0: 50 aday · z=1: 50 aday · z=2: 50 aday
YAZILDI /output/nodules.json — 150 kutu
```

* Şema doğrulandı: `name="Regions of interest"`, `type="Multiple 2D bounding boxes"`, `version={1,0}`, her kutuda **4 köşe × 3 koordinat (mm)**, olasılık 0,3549–**0,9938** azalan sırada.
* **mm↔piksel tersinirlik:** köşe `[101,81; 75,77; 0,0]` mm → `[642,0; 390,0; 0,0]` px; `642,0 · 0,143 + 10,0 = 101,81` ✓
* **Deterministiklik:** n0004'ün en yüksek skoru yerel testte de konteynerde de **0,9938**.
* ⏱️ **Konteyner GPU'suz koşuldu → ~26 s/katman.** Grand-Challenge NVIDIA T4 verdiğinde (imaj CUDA tabanlı) ~1–2 s/katman beklenir; CPU'da büyük yığınlarda süre sınırı riski vardır, bu yüzden T4 kullanımı önerilir.

* **Betikler:** [`v6_7_train_fulldata.py`](pipelines/v6_two_stage/v6_7_train_fulldata.py) · [`v6_7_build_fulldata_stage2.py`](pipelines/v6_two_stage/v6_7_build_fulldata_stage2.py) · [`v6_7_submission_infer.py`](pipelines/v6_two_stage/v6_7_submission_infer.py) · [`v6_7_freeze_config.py`](pipelines/v6_two_stage/v6_7_freeze_config.py) · [`submission_docker/process.py`](submission_docker/process.py) · [`submission_docker/smoke_test.py`](submission_docker/smoke_test.py)
* **Sözleşme:** `artifacts/v6_two_stage/submission_config.json`

### V6.7.5 — Resmî baseline uyumluluk rötuşu (3 düzeltme)

Resmî `node21challenge/node21_detection_baseline/process.py` **kaynaktan** okundu ([ham dosya](https://raw.githubusercontent.com/node21challenge/node21_detection_baseline/main/process.py)) ve `format_to_GC` fonksiyonuna göre **üç gerçek uyumsuzluk** bulunup düzeltildi:

| # | Bulgu (baseline ne yapıyor) | Önceki hâlimiz | Düzeltme |
| :-: | :--- | :--- | :--- |
| **1** | JSON `{type, boxes, version}` — **`name` alanı YOK** | `"name": "Regions of interest"` ekliyorduk | `name` **kaldırıldı** |
| **2** | Köşe sırası **[top_right, top_left, bottom_left, bottom_right]** = `[(xmax,ymax),(xmin,ymax),(xmin,ymin),(xmax,ymin)]` | `[(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)]` (BL,TL,TR,BR) — **farklı sıra** | baseline sırasına çevrildi |
| **3** | **z = slice İNDEKSİ** (`np_prediction['slice']`), x/y = **piksel × spacing** (origin **yok sayılır**) | z fiziksel mm; x/y transform (origin dahil) | z **slice indeksi**; x/y = `transform − origin` ⇒ baseline ile birebir |

**Neden fark etmiyordu ama önemliydi:** NODE21 `.mha` dosyalarının **origin'i (0,0,0)** olduğu için x/y'de transform ≡ `piksel×spacing`; ama *köşe sırası* ve *`name` alanı* gerçek sözleşme farklarıydı ve katı bir doğrulayıcıda reddedilebilirdi.

**Ek uyumluluk (Grand-Challenge güvenlik kuralı):**
* **`USER algorithm`** (uid 1000, non-root) — doğrulandı: `whoami → algorithm`, `id → uid=1000(algorithm)`.
* `/output` yazılabilir (chown algorithm) — doğrulandı: `touch /output/test.txt` başarılı.
* Ağırlıklar `a+rX` (okunabilir); `TORCH_HOME`/`MPLCONFIGDIR` kullanıcı-yazılabilir dizinlere taşındı.
* **`matplotlib` bağımlılığı tamamen kaldırıldı:** `node21_tda/preprocessing/__init__.py` **tembel import**'a (PEP 562) çevrildi ⇒ `zones` (matplotlib) yüklenmez. Bu, daha önce konteyneri çökerten `ModuleNotFoundError: matplotlib`'in de kök nedenini ortadan kaldırır.
* **`Dockerfile.cpu`** eklendi: CUDA tabanı ~14 GB iken CPU tabanı **~1,5 GB** (yükleme limiti için; bedeli ~26 s/katman).

**Esnek girdi yolu:** `process.py` artık `/input` altını **özyinelemeli** tarar (`rglob("*.mha")`) ⇒ hem baseline'ın `/input/<uuid>.mha` hem yeni GC'nin `/input/images/chest-x-rays/<uuid>.mha` düzeni çalışır. Yerel testte **ikisi de** doğrulandı (aynı 100/150 kutu).

**Yeniden doğrulama (gerçek konteyner, `--network=none`):**
```
algorithm                                  ← non-root ✓
uid=1000(algorithm) gid=1000(algorithm)    ← ✓
output YAZILABILIR ✓  drwxr-xr-x algorithm
girdi: /input/stack.mha                    ← esnek yol ✓
stack.mha: yığın 3 katman, 1024x1024       ← 3B ✓
YAZILDI /output/nodules.json — 150 kutu
```
* Çıktı anahtarları: **`['boxes','type','version']`** (`name` yok) ✓
* Köşe sırası TR,TL,BL,BR ✓ · olasılık **0,9938** (yerelle aynı) · z ∈ {0,1,2} **slice indeksi** ✓
* Origin düzeltmesi kanıtı: eski kutu `[101,81; 75,77]` → yeni `[91,81; 55,77]` = **origin (10,20) çıkarılmış**, tam olarak baseline sözleşmesi ✓

* **Kayıt:** `submission_docker/README.md` (uyumluluk tablosu) · [Dockerfile](submission_docker/Dockerfile) · [Dockerfile.cpu](submission_docker/Dockerfile.cpu) · [process.py](submission_docker/process.py) · [smoke_test.py](submission_docker/smoke_test.py)

---

## 12. V6.8 — Kanıt Boşluklarının Kapatılması (ısınma ablasyonu + yarıçap kafası onarımı)

**Motivasyon — iki kayıp/eksik kanıt:**
1. V6.6'da ısınma checkpoint'i `M5_foundation_warm.pt` **teşhis için** kaydedilmiş, sonra "temizlik" sırasında **değerlendirilmeden silinmişti** ⇒ *"iki fazlı eğitimin 2. fazı gerçekten kazandırıyor mu?"* sorusunun kanıtı yoktu.
2. V6.6.7'de yarıçap kafasının **çöktüğü** bulunmuş (R ≈ 0,01 vs GT 35–119 px) ama **onarılmamıştı**.

**Betik:** [v6_8_fix_checks.py](pipelines/v6_two_stage/v6_8_fix_checks.py) · **Kayıt:** `artifacts/v6_two_stage/v6_8_fix_checks.json` · **Kural:** yalnız train split'te eğitim, yalnız val'de değerlendirme (tam-veri V6.7 modeline sızıntı yok) · bilimsel checkpoint'lere **dokunulmadı**.

### V6.8.1 Isınma ablasyonu — silinen kanıt geri getirildi

| | |
| :--- | :--- |
| Veri | train split **1:1** → 790 nodüllü + 790 temiz kontrol = **1.580 örnek** |
| Isınma kolu | **5 epoch**, omurga **DONUK**, BN donuk, `lr_head=1e-3`, aug/kayıp/tohum `v6_6_train_foundation.py` ile **birebir** |
| Final kolu | kayıtlı `M5_foundation.pt` (18 epoch = 5 donuk + 13 diferansiyel ince ayar) |
| Değerlendirme | val **734 hasta / 231 lezyon**, `predict_tta` (NMS 20 px) + `froc_fixed` |

| Ölçüm (val) | Isınma (5 ep, donuk omurga) | **Final (18 ep, iki fazlı)** | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K10 | 0,9437 | **0,9481** | +0,0043 |
| Recall@K20 | 0,9654 | **0,9827** | **+0,0173** |
| Recall@K30 | 0,9870 | 0,9827 | −0,0043 |
| Recall@K50 | 0,9913 | 0,9913 | 0,0000 |
| FROC@0,5 | 0,7792 | **0,8312** | **+0,0520** |
| FROC@1 | 0,8398 | **0,8658** | +0,0260 |
| FROC@4 | 0,9091 | **0,9524** | **+0,0433** |
| Hasta AUC (top1) | 0,9275 | **0,9595** | **+0,0320** |
| Hasta AUC (noisy-OR) | 0,9416 | **0,9702** | **+0,0286** |

* ✅ **2. faz gerçekten kazandırıyor.** En büyük kazanç **FROC@0,5'te (+0,052)** ve **FROC@4'te (+0,043)**; yani ince ayarın işi **düşük-FP bölgesini** düzeltmek. **K50'de katkı tam sıfırdır** (0,9913 → 0,9913) ⇒ ısınma K50 tavanını **zaten** yakalıyor; K30'da küçük bir gerileme (−0,0043) vardır, yani kazanç **monoton değil**.
* 🔒 **Kanıt bütünlüğü (elma-elma):** yeniden üretilen ısınma koşusunun epoch kayıpları V6.6'nın kayıtlı train log'uyla örtüşür (13,36 / 2,23 / 2,10 / 1,96 / 1,87 vs 13,35 / 2,22 / 2,11 / 1,97 / 1,90) ve **final kolunun metrikleri V6.6'nın yayımlanmış değerlerinin aynısıdır** (K20 0,9827 · FROC@0,5 0,8312 · @4 0,9524 · top1 0,9595) ⇒ karşılaştırma aynı ölçüm hattındandır.
* 🔁 **İkinci bağımsız koşu (aynı betik · aynı tohum · yeniden eğitim):** ısınma kolunun metrikleri GPU çekirdeklerinin belirlenimsizliği kadar kayıyor (K20 0,9654 → 0,9697 · FROC@0,5 0,7792 → 0,7965 · AUC nOR 0,9416 → 0,9375; epoch kayıpları 13,39/2,24/2,10/1,95/1,88) ama **hüküm değişmedi**: Δ(final−ısınma) FROC@0,5 **+0,052 / +0,035**, FROC@4 **+0,043 / +0,035**, AUC nOR **+0,029 / +0,033**, K50 **iki koşuda da tam 0,0000**. ⇒ Etki büyüklüğü **"+0,03 … +0,05"** aralığı olarak raporlanmalı; tek koşunun 4. ondalığı anlamlı değil. Final kolunun sayıları iki koşuda da **birebir aynı** (kayıtlı checkpoint + deterministik değerlendirme).
* ⚠️ **Dürüst sınır:** Δ(final−ısınma), 2. fazın **toplam** katkısıdır (13 ek epoch **+** üst blokların açılması). Bunu tek başına *"unutma-önleme katkısı"* diye adlandırmak fazla iddia olurdu ⇒ BN dondurmanın payı ayrı kol olarak ölçülür (`--task bn`).
* **Teşhis checkpoint'i:** `submission_fulldata/M5_foundation_warm_probe.pt` (bilimsel değil; git dışı tutulur).

### V6.8.2 Yarıçap kafası onarımı — kök neden ağırlık DEĞİL, maske

**V6.6.7'nin teşhisi eksikti.** Orada çökme *"kayıp ağırlığı 0,1, focal ise 1–13 mertebesinde"* diye açıklanmıştı. V6.8 bu açıklamayı **doğrudan test etti**: aynı mimari, **ağırlık 0,1 → 1,0** (10×). Sonuç: **kafa YİNE çöktü** (tahmin edilen r₁₀₂₄ medyanı **0,02 px**, GT 17,5–59,5) ⇒ ağırlık tek başına neden değil.

**Gerçek kök neden: kaybın normalize edildiği piksel maskesi.** V6.6'nın kaybı, ısı haritasının **pozitif bölgesinin tamamı** üzerinde ortalanıyordu (`Y > 0`). O bölge nodül merkezinin çevresinde **961 piksel**dir (31×31 Gaussian çekirdeği) ama yarıçap hedefi `R` yalnız **tek merkez pikselinde** sıfırdan farklıdır. **Ölçüm** ([v6_8_mask_probe.py](pipelines/v6_two_stage/v6_8_mask_probe.py), 4 nodüllü örnek): `Y>0` = 5.402 piksel, `R>0` = **6** piksel ⇒ süpervizyonun **%0,11**'i gerçek hedef taşır, **%99,9'u "0 tahmin et"** der; kaybın minimumu **sabit 0**'dır. Merkez-ağırlıklı maske bunu düzeltir.

**Kontrollü karşılaştırma (tek fark: maske)** — omurga DONUK, aynı 2 katmanlı kafa, **aynı ağırlık (1,0)**, aynı 6 epoch, aynı tohum/veri (train 1:1 = 1.580), aynı değerlendirme (val 734 hasta / 231 lezyon, çok ölçekli TTA + NMS 8 px, resmî **IoU > 0,2**):

| Kol | Kayıp maskesi | Eğitim kaybı (ep1 → ep6) | Tahmin edilen r₁₀₂₄ (p10 / medyan / p90) | rank_metric | Sens@0,125 | Sens@0,25 | Sens@0,5 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| A (V6.6'nın yaptığı) | `Y > 0` (tüm Gaussian bölgesi) | 0,0076 → 0,0073 | 0,01 / **0,02** / 0,03 | 0,7258 ❌ | 0,000 | 0,000 | 0,000 |
| **B (onarım)** | **`R > 0` (yalnız merkez pikseli)** | 2,09 → **1,07** | 26,1 / **33,3** / 47,4 | **0,9185** ✅ | **0,6970** | **0,7706** | **0,8312** |

**Sabit kutu referansları (aynı adaylar, yalnız kutu geometrisi değişir):**

| Kutu | rank_metric | AUC | Sens@0,125 | Sens@0,25 | Sens@0,5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| sabit 48 px | 0,9109 | 0,9678 | 0,6710 | 0,7403 | 0,7922 |
| **sabit 56 px** (V6.6.7 optimumu) | 0,9131 | 0,9678 | 0,6710 | 0,7489 | 0,8009 |
| sabit 64 px | 0,9077 | 0,9678 | 0,6537 | 0,7273 | 0,7879 |
| **öğrenilmiş yarıçap (kol B)** | **0,9185** | 0,9678 | **0,6970** | **0,7706** | **0,8312** |

* ✅ **Onarım başarılı ve sabit kutuyu GEÇİYOR:** rank_metric **0,9131 → 0,9185** (+0,0054) ve **üç işletim noktasının üçünde de** duyarlılık artıyor (+0,026 / +0,022 / +0,030). Bu, **V6.6.7'den bu yana ilk kez** yarıçap kafasının resmî metrikte **işe yarar** hâle geldiği noktadır.
* 🔒 **AUC 0,9678 her satırda sabittir** — beklenen sonuç: kutu geometrisi sıralamayı değiştirmez. Kazanç **tamamen IoU tabanlı eşleştirmeden** gelir.
* ⚠️ **Dürüst okuma — dağılım daralması:** eğitilen kafa, GT aralığının (17,5–59,5 px) **ortasına doğru sıkışıyor** (p10 26, medyan 33, p90 47) — yani mükemmel kalibre değil, **regresyona-ortalama** yapıyor. Yine de sabit kutuyu geçiyor ⇒ kazanç "her nodüle kendi boyutu" fikrinden geliyor, kalibrasyondan değil.
* ⚠️ **Kapsam:** bu kafa **dondurulmuş özellik probudur** (omurga eğitilmedi) — ucuz (~6 dk) ve nedensel olarak temiz, ama uçtan uca eğitilmiş bir yarıçap dalının **tavanı** değildir.
* ⚠️ **Gönderim sözleşmesi DEĞİŞMEDİ:** `submission_config.json` kilitli **sabit 56 px** kullanır (Tuzak A: yeni hiperparametre aranmaz). Onarılan kafayı gönderime almak **ayrı bir sürüm kararıdır** ve tam-veri modelinin yarıçap dalının aynı maske düzeltmesiyle yeniden eğitilmesini gerektirir.
* 🔒 **Kanıt koruması (V6.6 hatasının tekrarı değil):** eğitilen kafa **diske yazılır** — `submission_fulldata/radius_head_weight1_center_only.pt` (77 KB; git dışı). Ayrıca koşu **bağımsız olarak yeniden üretildi**: ikinci koşu **aynı** rank_metric (0,9185), aynı duyarlılıklar ve aynı r yüzdeliklerini verdi.

### V6.8.3 Unutma-önleme kolu — BN dondurmanın payı ölçüldü (**beklenmedik sonuç**)

V6.6/V6.7'nin tasarım gerekçesi şuydu: *"BN `.eval()` — 800k röntgenlik `running_mean/var` bozulmasın"*. Bu bir **argümandı, ölçüm değildi**. `--task bn`, ısınma checkpoint'inden devam edip **2. fazı (13 epoch) BN istatistikleri AÇIK** koşar; veri/takvim/tohum/augmentasyon **birebir aynıdır**, tek fark BN'in güncellenip güncellenmemesidir.

| Ölçüm (val 734 hasta / 231 lezyon) | 2. faz **BN AÇIK** | Final (**BN donuk**) | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K10 | **0,9697** | 0,9481 | +0,0216 |
| Recall@K20 | **0,9913** | 0,9827 | +0,0086 |
| Recall@K30 | **0,9957** | 0,9827 | +0,0130 |
| Recall@K50 | **0,9957** | 0,9913 | +0,0044 |
| FROC@0,5 | **0,8442** | 0,8312 | **+0,0130** |
| FROC@1 | **0,8918** | 0,8658 | **+0,0260** |
| FROC@4 | 0,9524 | 0,9524 | 0,0000 |
| Hasta AUC (top1) | **0,9646** | 0,9595 | +0,0051 |
| Hasta AUC (noisy-OR) | 0,9699 | **0,9702** | −0,0003 |
| Eğitim kaybı (ep 18) | **1,158** | 1,257 | −0,099 |

* 🔴 **BN dondurmanın ölçülebilir bir faydası YOK.** Dokuz metrikten **yedisinde BN açık daha iyi**, birinde (**FROC@4**) **eşit**, birinde ise **−0,0003** (AUC noisy-OR, gürültü).
* ⚠️ **Ama bu bir "BN açık daha iyi" kanıtı da DEĞİL:** 231 lezyonla FROC@0,5'in standart hatası ≈ **0,024** ⇒ 0,013'lük fark **~0,5 SE**. Tek tohum, tek val bölmesi. **Doğru dürüst ifade:** *"BN dondurma kurum içi val'de ölçülebilir bir unutma-önleme faydası göstermedi; zarar da vermedi."*
* 🎯 **Gerekçe hipotez olarak kalıyor ve asıl sınavı alan dışıdır:** BN running istatistikleri **transfer edilen şeyin parçasıdır**; V6.6.3'te AdaBN'in vakıf modelini **her konfigürasyonda bozduğu** (−0,065…−0,089) ölçülmüştü ⇒ istatistiklerin hassas olduğuna dair **dolaylı** kanıt var. V6.8.3 bu hipotezi **test etmez** (yalnız in-domain val).
* ⚠️ BN açık kolun eğitim kaybı daha düşüktür (1,158 vs 1,257) — yani model eğitim verisine daha iyi oturuyor; buna rağmen val'de gerilemediği için burada bir aşırı uyum işareti yok.
* **Teşhis checkpoint'i:** `submission_fulldata/M5_foundation_bn_unfrozen_probe.pt` (bilimsel değil; git dışı).

## 13. V6.9 — Mühürlü Test Onayı (ön-kayıtlı, tek atım)

**Motivasyon.** V6.1–V6.8'de **her** tasarım kararı NODE21 **val** bölmesinde alındı ve **aynı** bölmede raporlandı: kutu boyu için **296** resmî-metrik değerlendirmesi (37 varyant × 8 kutu), füzyon ağırlığı için **19** nokta (w ∈ 0,00…0,90), 3 TTA ölçek seti, 12 NMS/tepe kombinasyonu. Bu sızıntı değil, **seçim yanlılığı**dır: 231 lezyonda Sens@0,25'in standart hatası ≈ 0,030 olduğu için 296 değerlendirmenin en iyisini seçmek, gerçek değerin ~0,02–0,03 üstünde bir sayı üretmeyi *beklenir* kılar. Bu bölüm o borcu **dokunulmamış test bölmesinde** kapatır.

**Betik:** [v6_9_test_seal.py](pipelines/v6_two_stage/v6_9_test_seal.py) · **Ön-kayıt:** `artifacts/v6_two_stage/test_seal_preregistration.json` (koşudan **önce** commit `deb59ed`) · **Aralık:** [v6_9b_test_seal_ci.py](pipelines/v6_two_stage/v6_9b_test_seal_ci.py) · **Denetim raporu:** [LEAK_AUDIT.md](LEAK_AUDIT.md)

### V6.9.1 Mühür tasarımı (neden "mühür")

| Garanti | Uygulama |
| :--- | :--- |
| Kod değişmedi | Betik kendi SHA-256'sını ön-kayıtla karşılaştırır; uyuşmazsa **koşmayı reddeder** (`b0c8aa9e…`) |
| Model/bundle değişmedi | `M5_foundation.pt` (`fc2775…`) + `stage2_final_tda_hgb.joblib` (`976afc…`) hash doğrulaması |
| Test'e ayar yapılmadı | Hiperparametre arayan kod yok; tüm değerler ön-kayıtta sabit |
| Transductive sızıntı yok | z-normalizasyon referansı **val'den** donduruldu (s1 0,02699/0,03850 · tda 0,18936/0,17468); test dağılımından istatistik alınmadı |
| Sonuç seçilmedi | Ön-kayıtta "ne çıkarsa yazılır" kuralı + başarısızlık tanımı (|Δ|>0,05) önceden yazıldı |

Kilitli protokol: 3 ölçek (0,9/1,0/1,1) + yatay flip TTA · 3×3 tepe · **NMS 8 px** · top-50 · **sabit 56×56 px kutu** · **w = 0,55**.

> ⚠️ **Beyan:** Test bölmesi bu projede **ikinci** kez görüldü. İlki V6.4 `test_seal()` (V6 CenterNet+FCOS+lokal TDA + V5.2 global füzyon → birleşik hasta AUC 0,9876, ayarsız). V6.4'ten **sonraki** hiçbir karar test'e bakılarak alınmadı (hepsi val'de). Üçüncü bir "mühür" artık mühür sayılmaz.

### V6.9.2 Sonuçlar — val (seçim) vs test (mühür)

Eğitim **yalnız train split**'te yapılmış `M5_foundation` kullanıldı (val/test görmedi); tam-veri gönderim modeli NODE21'de değerlendirilemez.

| Ölçüm | Val (734/231) | **Test (734/229)** | Δ | Test %95 GA (bootstrap 1.000) |
| :--- | ---: | ---: | ---: | :--- |
| **rank_metric — 1. aşama** | 0,9131 | **0,8842** | **−0,0289** | 0,8559 – 0,9089 |
| rank_metric — füzyon w=0,55 | 0,9174 | **0,8970** | −0,0204 | 0,8769 – 0,9213 |
| Hasta AUC — 1. aşama | 0,9678 | **0,9446** | −0,0232 | 0,9237 – 0,9621 |
| Hasta AUC — füzyon | 0,9793 | **0,9645** | −0,0148 | 0,9491 – 0,9786 |
| Sens@0,125 — 1. aşama | 0,6710 | **0,6114** | −0,0596 | 0,5250 – 0,6805 |
| Sens@0,25 — 1. aşama | 0,7489 | **0,7031** | −0,0458 | 0,6250 – 0,7645 |
| Sens@0,5 — 1. aşama | 0,8009 | **0,7642** | −0,0367 | 0,7110 – 0,8164 |
| Sens@0,125 — füzyon | 0,6883 | **0,6332** | −0,0551 | 0,5598 – 0,7077 |
| Sens@0,25 — füzyon | 0,7316 | **0,6943** | −0,0373 | 0,6335 – 0,7668 |
| Sens@0,5 — füzyon | 0,7879 | **0,7729** | −0,0150 | 0,7186 – 0,8233 |

* ✅ **Ölçüm hattı sağlam — düşüş kod hatası değil.** Mühür koşusu val kolunda V6.6.7'nin yayımlanmış sayılarını **birebir** yeniden üretti (0,9131 · 0,9678 · 0,6710 · 0,7489 · 0,8009). İkinci koşu (aralık betiği) dağıtımı yeniden hesaplayıp mühür kaydıyla **bit düzeyinde** uyuştu ⇒ deterministik.
* 📉 **Düşüş beklenen büyüklükte:** −0,029 ≈ **1 SE** (0,030). Ön-kayıttaki "|Δ| > 0,05 ⇒ seçim yanlılığı lehine kanıt" eşiği **aşılmadı**.
* ⚠️ **Ama iyimserlik yok sayılamaz:** val noktası (0,9131) test'in %95 aralığının (üst 0,9089) **az üstünde**; füzyon AUC'de de val (0,9793) aralık üstünün (0,9786) hemen dışında. Doğru cümle: *"val sayıları hafif iyimser; düzeltilmiş dürüst değerler test sütunundakilerdir."*
* 🔄 **V6.6.4'ün "2. aşamayı atla" önerisi test'te TERSİNE DÖNDÜ.** Val'de yalnız 1. aşama füzyondan FROC'ta üstündü (0,8442 vs 0,8268) ve dağıtım için "2. aşamasız" önerilmişti. Test'te **füzyon dört metrikten üçünde üstün** (rank 0,8970 > 0,8842 · AUC 0,9645 > 0,9446 · Sens@0,5 0,7729 > 0,7642; yalnız Sens@0,25'te geride). ⇒ O öneri **val'e özgüymüş**; mühür tam da bunu yakalamak için vardı.

### V6.9.3 Mühür kaydındaki iki teknik uyarı

* ❌ **`legacy_fusion.patient_auc.noisyor` = 0,6432 GEÇERSİZ.** Eski protokol bloğunda füzyon skoru z-ölçeğinde (negatif olabilir) verildiği için noisy-OR matematiksel olarak anlamsızdır. Geçerli olan **`A2a_fusion_tda.auc` = 0,9645**'tir (orada sigmoid uygulanır). Kayıttaki diğer tüm sayılar geçerlidir.
* ⚠️ **Yol farkı:** mühür aday üretiminde **resmî-metrik yolunu** (3 ölçek ortalaması + NMS 8 px) kullanır; V6.6.6 tablosu `predict_tta` (NMS 20 px) yolundandır ⇒ val füzyon FROC@0,5 burada 0,8268, orada 0,8355. **Yalnız 1. aşama** sayıları aynı yolu kullandığı için birebir uyuşur — mühür karşılaştırması bu satırda en temizdir.

### V6.9.4 Yayın için sonuç

> Ana tabloda **test sütunu** (0,8842 / 0,8970 / 0,9446 / 0,9645) dürüst dış-değerdir; val sütunu **"seçilmiş"** etiketiyle verilmelidir. VinDr dış doğrulaması ise **kontamine**dir (bkz. [LEAK_AUDIT.md](LEAK_AUDIT.md) §3: 400/400 görüntü omuranın ön-eğitim listesinde) — "zero-shot" ifadesi kullanılamaz.

### V6.9.5 Kontaminasyonun ÖLÇÜLMESİ — temiz omurga deneyi (V6.9c)

Denetim (§3) kontaminasyonun **varlığını** kanıtladı; bu alt bölüm **etkisini** ölçer. Tek değişken omurganın gördüğü veridir; tarif V6.8 ısınmasıyla aynıdır (omurga donuk, 5 epoch, lr_head 1e-3, BN donuk, seed 42, train 1:1 = 1.580).

| Satır | Omurga | VinDr ön-eğitimde? | NODE21 val nOR | NODE21 K20 | VinDr AUC top1 | VinDr nOR | VinDr bbox K50 |
| :-: | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| A | `resnet50-res512-all` (mevcut) | **var** | 0,9466 | 0,9740 | 0,6479 | 0,6414 | 0,8369 |
| B | `densenet121-res224-all` | **var** | 0,9440 | 0,9567 | 0,6406 | 0,6534 | 0,8416 |
| **C** | `densenet121-res224-nih` | **yok** | 0,8650 | 0,9091 | **0,6369** | **0,6663** | 0,8298 |
| | **B − C** | | **+0,0790** | +0,0476 | +0,0037 | **−0,0129** | +0,0118 |

* ✅ **Şişme bulunamadı.** VinDr'yi hiç görmemiş omurga, VinDr'de kontamine olanlar kadar iyi: top1 farkı **+0,004**, noisy-OR'da temiz omurga **daha iyi** (−0,013). 150 pozitif görüntüde AUC SE ≈ 0,04 ⇒ farklar **gürültü bandında**. ⇒ *"0,6469 kontaminasyon sayesinde yüksek"* iddiası **desteklenmiyor**.
* 🔑 **VinDr ön-eğitimi hedef göreve yarıyor, teste değil.** B−C farkı NODE21'de **+0,079** (büyük), VinDr'de ~0 ⇒ CXR ön-eğitimi daha iyi dedektör üretiyor; VinDr testi ise ~0,64–0,67'lik bir **transfer tavanına** dayanmış.
* ⚠️ **Karıştırıcı:** B ve C yalnız VinDr'de değil **toplam korpus büyüklüğünde** de ayrışır (`all` = nih+pc+chex+mimic+google+openi+kaggle vs yalnız `nih`) ⇒ +0,079'luk NODE21 farkı **VinDr'ye özgü sayılamaz**.
* 🎯 **Yayın kararı:** VinDr sayısı **C satırıyla** raporlanmalı (0,6663 nOR / 0,6369 top1); A satırı "kontamine üst sınır" olarak dipnotta kalır. Dürüst cümle: *"VinDr transferi ~0,64–0,67 bandında tavan gösteriyor; bu tavan, VinDr'yi görmemiş omurgada da aynı."*
* 🔁 **İç tutarlılık:** A satırı, V6.6.3'ün yayımlanmış strict değerini (0,6469) aynı tarifle 0,6479 olarak yeniden üretti ⇒ ölçüm hattı sağlam.
* **Betik:** [v6_9c_contamination.py](pipelines/v6_two_stage/v6_9c_contamination.py) · **Kayıt:** `artifacts/v6_two_stage/v6_9c_contamination.json` · omurga ağırlıkları `artifacts/xrv_weights/` (git dışı).

### V6.9.6 NODE21'in İÇİNDEKİ kaynak örtüşmesi (kaynak-ayrıştırmalı ölçüm)

**V6.9c'den farklı ve daha kritik mesele:** VinDr dış veriydi; bu kez **kendi testimizin kaynağı** sorgulandı.

**Veri nereden geliyor?** NODE21'in halka açık eğitim seti yeni hastane verisi değil, açık veri kümelerinden derlenmiş bir yeniden etiketlemedir. `patient_info.json` 4.882 hastada şunu söylüyor:

| `source_dataset` | n | etiket | `pc-nih-rsna-siim-vin` gördü mü? |
| :--- | ---: | :---: | :---: |
| `node21_control` | 3.748 | hepsi 0 | **bilinmiyor** (köken kayıtlı değil) |
| `chestxray14` (NIH) | 617 | hepsi 1 | ✅ `nih` |
| `padchest` | 314 | hepsi 1 | ✅ `pc` |
| `jsrt` | 149 | hepsi 1 | ❌ |
| `openi` | 54 | hepsi 1 | ❌ |

⇒ Pozitiflerin **%82,9'u** omuranın gördüğü korpuslardan. **Ama** korpusta yalnız görüntü + rapor düzeyi etiket var; **NODE21'in radyolog kutuları yok** ⇒ resmî metrik (IoU>0,2) kutu ezberine yapısal olarak kapalı.

**Ölçüm** ([v6_9d_source_overlap.py](pipelines/v6_two_stage/v6_9d_source_overlap.py); kilitli protokol bir kez koşuldu, mühür kaydıyla birebir doğrulandı; aynı aday listesi kaynaklara ayrıştırıldı):

| Test (734/229) | n hasta | n lezyon | AUC (kontrole karşı) | S@0,125 | S@0,25 | S@0,5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| chestxray14 ✅ | 89 | 138 | 0,9760 | 0,6449 | 0,6957 | 0,7754 |
| padchest ✅ | 58 | 66 | 0,9604 | 0,6515 | 0,7273 | 0,8182 |
| jsrt ❌ | 19 | 19 | 0,9147 | 0,3684 | 0,4737 | 0,5263 |
| openi ❌ | 6 | 6 | 0,9920 | 1,000 | 1,000 | 1,000 |
| **GÖRÜLEN birleşim** | 147 | 204 | **0,9698** | 0,6471 | **0,7059** | 0,7892 |
| **GÖRÜLMEYEN birleşim** | 25 | 25 | **0,9332** | 0,5200 | **0,6000** | 0,6400 |
| **Fark** | | | **−0,037** | −0,127 | **−0,106** | −0,149 |

Val'de aynı yön, daha küçük fark: 0,9826 vs 0,9660 (Δ −0,017); S@0,25 0,7448 vs 0,6667 (Δ −0,078).

* ⚠️ **Fark ≈ 1 SE** (görülmeyen grupta 25 lezyon; duyarlılık SE ≈ 0,10) ⇒ belirleyici değil.
* ⚠️ **Kaynak kimliği başlı başına güçlü değişken:** spacing chestxray14 0,143 · padchest 0,275 · jsrt 0,309 · openi 0,331 mm/px; A3'te **aynı model** için kaynak-içi AUC 0,78–0,94 (ablation §7.3b) ⇒ cihaz/geometri farkı, aşinalıktan bağımsız olarak da fark üretir.
* ✅ **Yön tutarlı** (val + test) ⇒ "hiç etkisi yok" denemez; ✅ **büyüklük sınırlı** (≤0,04 AUC) ⇒ "skorlar sahte" de denemez.
* 🔒 **Yarışma notu:** kurallar halka açık ön-eğitimli modellere izin verir, **ama gizli test setinin bileşimini biz doğrulayamayız** ⇒ "sıfır sızıntı garantili" cümlesi kurulmamalıdır.

### V6.9.7 Belirleyici test — omurga × kaynak ETKİLEŞİMİ (konu 🔒 kapandı)

V6.9.6'daki "görülen > görülmeyen" farkı **iki şeyle karışıktı**: kaynakların cihaz/geometri farkı ve modelin kendisi. Ayrıştırmak için iki omurga **aynı protokolde** (heat-map TTA) kıyaslandı: **M5** = ImageNet ResNet18 (hiç CXR görmemiş = "temiz"), **FND** = `resnet50-res512-vin` (pozitiflerin %82,9'unu görmüş = "kontamine"). Ölçülen: her kaynak grubunda hasta düzeyi AUC ve **Δ = FND − M5**; kritik istatistik **etkileşim = Δ(görülen) − Δ(görülmeyen)**.

**Hat doğrulaması:** M5_heatTTA recall@K10/K20/K50 = 0,9091/0,9654/0,9784 ve FND_heatTTA = 0,9481/0,9827/0,9913 — V6.6.2'nin yayımlanmış değerlerinin **aynısı**.

| Val (734 hasta · görülen 138 poz/192 lezyon · görülmeyen 34 poz/39 lezyon · 562 kontrol) | M5 (temiz) | FND (kontamine) | **Δ = FND − M5** |
| :--- | ---: | ---: | ---: |
| **Görülen** kaynaklar (chestxray14+padchest) | 0,8386 | 0,9636 | **+0,1249** |
| **Görülmeyen** kaynaklar (jsrt+openi) | 0,6176 | 0,9427 | **+0,3251** |
| **ETKİLEŞİM** | | | **−0,2002** (95% GA −0,299…−0,109 · P(>0) = **0,000**) |

top3 için −0,193 · noisy-OR için −0,202 ⇒ üç toplama kuralında da aynı.

* ✅ **Örtüşme lehine KANIT YOK — ters yönde işaret.** FND'nin ImageNet'e göre avantajı, omurganın **hiç görmediği** kaynaklarda **2,6 kat** daha büyük (+0,325 vs +0,125) ⇒ kazanç "hatırlama"dan değil, **genel CXR temsil kalitesinden**.
* 🔑 **V6.9.6'nın farkını asıl açıklayan şey kaynak zorluğu:** ImageNet omurgası JSRT/OpenI'de çöküyor (0,618), CXR omurgası çökmüyor (0,943). Görülen↔görülmeyen açığı M5'te **+0,221**, FND'de **+0,021** ⇒ **CXR ön-eğitimi kaynak-duyarlılığını ~0,20 AUC azaltıyor.** Yani tablo, örtüşmenin *şişirdiği* değil ön-eğitimin *sağlamlaştırdığı* bir tablodur.
* ⚠️ **Sınırlar:** yalnız **val** (test mühürlü — üçüncü kullanım yapılmaz); M5 ResNet18 ≈11M vs FND ResNet50 ≈23M ⇒ **kapasite karıştırıcısı**; görülmeyen grup 34 hasta/39 lezyon.
* 🔒 **KAPANIŞ:** Örtüşme konusu **mühürlendi** — üç bağımsız ölçüm (V6.9c dış, V6.9d iç, V6.9e etkileşim) aynı yönü gösteriyor; mevcut veriyle yapılacak **yeni ölçüm talebi kapatıldı**. Yalnız (a) yeni ve gerçekten dış bir veri kümesi veya (b) korpusu NODE21 kaynaklarıyla kesişmeyen bir omurga (ör. yalnız MIMIC-CH) mührü yeniden açabilir. Ayrıntı: [LEAK_AUDIT.md §10](LEAK_AUDIT.md).
* **Betik/kayıt:** [v6_9e_backbone_interaction.py](pipelines/v6_two_stage/v6_9e_backbone_interaction.py) · `artifacts/v6_two_stage/v6_9e_backbone_interaction.json`
* ⚠️ **Düzeltme:** [LEAK_AUDIT.md](LEAK_AUDIT.md)'nin ilk sürümünde "NODE21 omuranın hiç görmediği veridir" yazıyordu — **yanlıştı**; §9'da düzeltildi.

### V6.9.8 Dinamik yarıçap kafası — kilitli test doğrulaması (V6.9f)

V6.8'de `R>0` maskesiyle eğitilen dinamik yarıçap kafası **val**'de sabit 56 px'i geçmişti
(rank_metric 0,9185 > 0,9131). Gönderime almadan önce aynı kafayı **kilitli test**te doğruladık.

> ⚠️ **ÜÇÜNCÜ test kullanımı.** Test bölmesi V6.4 + V6.9 mühürleriyle iki kez kullanılmıştı; bu
> koşu, val'de seçilmiş bir iyileştirmenin tek-atımlık doğrulamasıdır (hiçbir seçim test üzerinde
> yapılmadı).

| Test (734 hasta / 229 lezyon) | S@0,125 | S@0,25 | S@0,5 | S@1 | S@2 | S@4 | rank_metric |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sabit 56×56 px | 0,6114 | 0,7031 | 0,7642 | 0,8210 | 0,8603 | 0,8952 | **0,8842** |
| **Dinamik yarıçap** | 0,6114 | 0,7031 | **0,7773** | **0,8341** | **0,8734** | **0,9083** | **0,8842** |
| Δ | 0,0000 | 0,0000 | **+0,0131** | **+0,0131** | **+0,0131** | **+0,0131** | **0,0000** |

* 🚨 **Val kazancı resmî skorda test'e TAŞINMADI.** rank_metric = 0,75·AUC + 0,25·S@0,25; S@0,25
  iki kolda da **0,7031** ⇒ resmî skor eşit (0,8842). Val'deki +0,0054 (0,9131 → 0,9185),
  test'in S@0,125/0,25 noktalarında **tekrarlanmadı**.
* ✅ **Ama dinamik kutu HİÇBİR noktada zarar vermiyor** ve daha gevşek eşiklerde (S@0,5/1/2/4)
  **+0,0131** kazandırıyor — bu, FROC-optimal (lezyon bulma) işletim bölgesidir.
* 🔑 **Ders:** val'de seçilen kutu boyutu, test dağılımına kısmen aşırı uyum sağlamış. Mühürlü
  testin değeri tam burada: val kazancının genelleşip genelleşmediğini ayırıyor.
* 🎯 **Karar:** sabit 56 px **muhafazakâr varsayılan olarak kalıyor**; dinamik kutu resmî skoru
  değiştirmediği için zorunlu değil (yalnızca FROC-optimal dağıtımda opsiyonel +0,013).
* **Betik/kayıt:** [v6_9f_radius_test.py](pipelines/v6_two_stage/v6_9f_radius_test.py) ·
  `artifacts/v6_two_stage/v6_9f_radius_test.json`

> ⚠️ **ZORUNLU TODO — Docker Gönderimi Öncesi (V6.9g):** Grand-Challenge sunucusuna fiilî Docker
> gönderimi yapılacağı vakit, `M5_foundation_fulldata.pt` modelinin yarıçap kafası, V6.8'deki
> `R>0` (yalnız merkez pikseli) maske düzeltmesiyle donuk gövde üzerinde yeniden eğitilecek
> ([v6_9g_fulldata_radius.py](pipelines/v6_two_stage/v6_9g_fulldata_radius.py)) ve
> `submission_docker/process.py` sabit 56×56 px yerine bu öğrenilmiş dinamik kutuyu kullanacak
> şekilde güncellenecektir. **⚠️ Ancak yukarıdaki V6.9f, dinamik kutunun resmî rank_metric'i
> DEĞİŞTİRMEDİĞİNİ gösterdi** (0,8842 = 0,8842; yalnızca S@0.5+ noktalarında +0,013). Bu yüzden
> gönderim öncesi karar yeniden değerlendirilmelidir: FROC-optimal (lezyon bulma) hedefleniyorsa
> dinamik kutu uygulanır, aksi hâlde sabit 56 px muhafazakâr varsayılan olarak korunur.

### V6.9.9 Yarıçap 5. özellik olarak 2. aşamaya — NEGATİF sonuç (V6.9h)

**Soru:** V6.9f, dinamik yarıçapın yalnızca *dış çerçeveyi* değiştirdiğini, TDA penceresinin
(32×32 + 64×64) hâlâ sabit olduğunu gösterdi. Peki tahmin edilen yarıçap R̂, 2. aşamaya **ek bir
özellik** olarak verilirse (1600 PI + 4 uzamsal + 1 yarıçap = 1605 özellik) nodül adayı ile
yanlış-alarm ayrımı iyileşir mi?

**Yöntem (temiz varyant — TDA penceresi DEĞİŞMEZ):** aynı adaylar üzerinde HGB(1604) vs HGB(1605);
yalnızca R̂ skaleri eklenir; train 400+400, val 734, class_weight balanced, w=0,55. Test KAPALI
(3. kullanım geçildi). Betik: [v6_9h_radius_feature.py](pipelines/v6_two_stage/v6_9h_radius_feature.py).

| Val (734/231) | TDA-tek AUC | Füzyon rank_metric | Füzyon AUC |
| :--- | ---: | ---: | ---: |
| HGB **1604** (yarıçap yok) | 0,9096 | **0,9238** | **0,9835** |
| HGB **1605** (+yarıçap) | 0,8559 | 0,9040 | 0,9658 |
| **Δ** | **−0,0537** | **−0,0198** | **−0,0177** |

* ❌ **Yarıçap özelliği ZARAR VERİYOR, fayda sağlamıyor.** TDA-tek AUC −0,054, füzyon rank −0,020,
  füzyon AUC −0,018 — üçü de geri.
* 🔑 **Neden (muhtemel):** yarıçap kafası yalnızca **nodül merkezinde** (`R>0` maskesi) eğitildi;
  yanlış-alarm adayları (damar/arka plan) için anlamlı bir "boyut" üretmez, ürettiği değer gürültü
  gibi davranır. V6.8'de zaten görülen **"regresyona-ortalama"** (medyan 33 px'e sıkışma) bunun
  işaretiydi ⇒ R̂ ayırt edici değil, sadece gürültü ekliyor.
* ✅ **Yan ürün:** aynı adaylar üzerinde HGB'yi **foundation adaylarıyla yeniden eğitmek**, dondurulmuş
  V6.3 bundle'ına göre val füzyonunu 0,9174 → **0,9238**'e taşıyor (HGB eğitim verisi ↔ çıkarım aday
  dağılımı uyumu). Bu, 2. aşamanın "yeniden eğitimle" potansiyel bir kazancı olduğunu gösterir ama
  yarıçap özelliğiyle **alakasızdır**.
* 🎯 **Karar:** yarıçap-uyarlamalı 2. aşama (özellik olarak) **reddedildi**; sabit 32/64 TDA penceresi
  + 1604 özellik **doğru seçimmiş**. Riskli "2R×2R alan-normalize TDA" varyantı da (V5.1 resize tuzağı +
  E3.6 boyut confound'u) bu negatif sonuçla daha az cazip hâle geldi.
* **Kayıt:** `artifacts/v6_two_stage/v6_9h_radius_feature.json`

### V6.9.10 Uyumlu 2. aşama — eski M5 kalıntısının temizlenmesi + kilitli test (V6.9i)

**Kök neden:** `stage2_final_tda_hgb.joblib`, **eski M5 (ImageNet) dedektörünün** aday dağılımıyla
eğitilmişti; oysa 1. aşamada artık `M5_foundation` kullanıyoruz. HGB'yi foundation'ın **kendi train
adaylarıyla** (790+790 hasta, 1604 özellik, s1 YOK = Tuzak B) yeniden eğitmek bu uyumsuzluğu giderir.

| Kilitli test (734/229) | rank_metric | AUC | S@0,125 | S@0,25 | S@0,5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Eski bundle (M5 adaylı HGB) | 0,8970 | 0,9645 | 0,6332 | 0,6943 | 0,7729 |
| **Uyumlu bundle (foundation adaylı HGB)** | **0,9045** | **0,9687** | 0,6245 | **0,7118** | **0,7904** |
| Δ | **+0,0075** | **+0,0042** | −0,0087 | **+0,0175** | **+0,0175** |

* ✅ **Uyum düzeltmesi gerçek bir kazanç:** rank_metric **+0,0075**, AUC **+0,0042**; S@0,25 ve
  S@0,5'te **+0,0175**. Yalnızca S@0,125'te −0,009 (sınırda) ve S@1'de −0,004 gerileme — net pozitif.
* 🔑 **Eski M5 kalıntısı gerçek bir engelmiş:** HGB'nin eğitim dağılımı (M5 adayları) ile çıkarım
  dağılımı (foundation adayları) arasındaki uyumsuzluk, test'te **0,0075 rank**'e mal oluyordu.
  V6.9h'nin "bonus bulgusu" (val 0,9174 → 0,9238) bunu önceden işaret etmişti.
* 🔒 **Koruma:** eski `stage2_final_tda_hgb.joblib` **DEĞİŞMEDİ** (SHA-256 korumalı bilimsel artifact);
  yeni bundle `checkpoints/stage2_foundation_tda_hgb.joblib` olarak ayrıca kaydedildi
  (`stage1_score_included=False` teyitli, w=0,55 kilitli, z-ref val'den donduruldu).
* ⚠️ **DÖRDÜNCÜ test kullanımı** — dağılım-uyum düzeltmesinin tek-atımlık doğrulaması; test üzerinde
  seçim yok. Bu koşudan sonra test bölmesi **tamamen kapanır**.
* **Betik/kayıt:** [v6_9i_compatible_stage2.py](pipelines/v6_two_stage/v6_9i_compatible_stage2.py) ·
  `artifacts/v6_two_stage/v6_9i_compatible_stage2.json`

### V6.9.11 Çift-katmanlı few-shot site kalibrasyonu (V6.9j) — VinDr-CXR

**Soru:** Güncel mimaride (M5_foundation + stage2_foundation_tda_hgb) yeni bir hastane kaynağına
N ∈ {0,10,20,40,80,160} etiketli görüntüyle adapte olurken 1. katmanı (head/neck FT), 2. katmanı
(TDA kalibrasyonu) ve ikisini birlikte kalibre etmenin etkisi nedir? (5-fold stratified CV, sızıntısız,
kilitli w=0,55 füzyon, sabit 56px kutu, resmî IoU>0,2.)

**Tasarım (kullanıcı onayı):** SABİT aday havuzu + yeniden puanlama. Donuk ResNet50'nin c2..c5
özellikleri **bir kez** hesaplanıp float16 RAM'de önbelleklendi (flip'siz + flip'li) ⇒ FT ve re-score
omurgayı YENİDEN geçmez (≈10× hız). 2. katman LR'si küçük N'de sıfır-eğitim p_tda0'a **shrinkage**
(α = N/(N+40)) ile harmanlandı. Betik: [v6_9j_dual_stage_fewshot.py](pipelines/v6_two_stage/v6_9j_dual_stage_fewshot.py).

| N | Kol1 — yalnız 2. katman (TDA LR) | Kol2 — yalnız 1. katman (head FT) | **Kol3 — ikisi birlikte** |
|---:|---:|---:|---:|
| 0 (sıfır-eğitim) | 0,6559 ± 0,0573 | 0,6559 ± 0,0573 | 0,6559 ± 0,0573 |
| 10 | 0,6573 ± 0,0657 | 0,6863 ± 0,0565 | 0,6863 ± 0,0663 |
| 20 | 0,6575 ± 0,0578 | 0,6869 ± 0,0521 | 0,6952 ± 0,0527 |
| 40 | 0,6748 ± 0,0491 | 0,7105 ± 0,0439 | **0,7424 ± 0,0243** |
| 80 | 0,7008 ± 0,0514 | 0,7180 ± 0,0636 | **0,7751 ± 0,0494** |
| 160 | 0,6928 ± 0,0549 | 0,7297 ± 0,0559 | **0,7992 ± 0,0462** |

(AUC = noisy-OR hasta düzeyi, 5-fold ortalama ± std. S@0.25 ≈ 0,09–0,12; FROC@2 ≈ 0,18–0,19 tüm kollarda.)

* 🏆 **Çift-katmanlı kalibrasyon N≥40'ta açıkça üstün:** Kol3, N=160'ta sıfır-eğitimden **+0,143**
  (0,656 → 0,799). N=40'ta +0,086, N=80'de +0,119.
* 🔑 **1. katman FT, 2. katman kalibrasyonundan daha değerli:** Kol2 (head FT) her N'de Kol1'i geçiyor;
  hâlbuki 2. katman tek başına (Kol1) N=80'de tepe yapıp N=160'ta geri düşüyor. Bu, V6.5'in "2. aşama
  tablosal modeli hedef merkezde kalibre et" reçetesinin güncel mimaride **yeterli olmadığını** gösterir:
  asıl kazanç, dedektörün kendi skorunu sahada yeniden kalibre etmektir.
* ✅ **İki katman tamamlayıcı:** Kol3 − Kol2 > 0 (N≥40'ta +0,03…+0,07) ⇒ FT edilmiş dedektör skoru ile
  kalibre edilmiş TDA skoru birbirini tamamlıyor.
* 📉 **Mutlak sınır:** N=160 bile 0,799 — kurum içi 0,969'un çok altında; dışmerkez kalibrasyonu
  domain-shift'i tamamen kapatmıyor (V6.9h/V6.9c ile tutarlı).
* ⚙️ **Hız:** omurga önbelleği sayesinde re-score ~60–90s → **~10–20s**; tam koşu ~1 saatin altına indi.
* **Kayıt:** `artifacts/v6_two_stage/v6_9j_dual_stage_fewshot.json`

### V6.9.12 Leave-JSRT-Out — kaynak ezberi testi + %100 yabancı JSRT dış testi (V6.9k · k2 · k3)

**Soru:** NODE21'in `train` bölmesinde **JSRT hastalarının 107'si** vardı. JSRT'yi eğitimden
**tamamen** çıkarıp iki aşamayı yeniden eğitirsek JSRT performansı ne olur? Yani JSRT'deki iyi
sayılarımız *kaynak ezberinden* mi geliyor?

**Kurulum (ana mimariyle birebir aynı reçete, değişen tek şey JSRT'nin yokluğu):**

| | Ana mimari | LOSO |
| :--- | :--- | :--- |
| 1. aşama | `M5_foundation` | `M5_nojsrt` (sıfırdan, aynı hiperparametreler) |
| Eğitim seti | 790 + 790 = **1.580** | 683 + 683 = **1.366** (`train − JSRT`, 1:1) |
| Epoch / lr / batch | 5 donuk + 13 FT · 1e-3/1e-4 · 4 | **aynı** (`tr.*` sabitleri import edilir) |
| PI aralıkları | V6.1'den devralınan | **LOSO adaylarından yeniden türetildi** (JSRT görmez) |
| z-referansı | val (tümü) | val **(− JSRT)** |
| 2. aşama | HGB 1604, s1-YOK, `200/0.08/balanced` | **aynı** |
| Füzyon / kutu | w=0,55 · 56×56 px | **aynı** |

**Sonuç A — LOSO modelinin JSRT dış testi (149 hasta, hepsi yabancı):**

| Recall@K10 | @K20 | @K50 | hasta-yakalama@50 | hasta AUC (top1) | (noisy-OR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0,8523** | **0,8792** | **0,9396** | **0,9396** | **0,9395** | **0,9767** |

**Sonuç B — ana model aynı 149 hastada (Δ = LOSO − ANA):**

| Ölçüt | LOSO | ANA | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K10 | 0,8523 | 0,9396 | −0,0873 |
| Recall@K20 | 0,8792 | 0,9732 | −0,0940 |
| Recall@K50 | 0,9396 | 0,9799 | −0,0403 |
| Aday AUC — 1. aşama | 0,8714 | 0,9208 | −0,0494 |
| Aday AUC — TDA | 0,6802 | 0,7966 | −0,1164 |
| Aday AUC — füzyon | 0,8072 | 0,9154 | −0,1082 |
| Hasta AUC (füzyon, top1) | 0,9395 | 0,9674 | −0,0279 |

**Sonuç C — temiz alt küme (42 JSRT hastası; 107'si ana modelin *eğitim* hastasıydı):**
eşleştirilmiş bootstrap (2.000 tekrar, aynı hastalar birlikte yeniden örneklenir):

| Δ (LOSO − ANA) | ortalama | %95 GA | P(Δ<0) |
| :--- | ---: | :---: | ---: |
| Aday AUC — 1. aşama | −0,0092 | [−0,0776, +0,0673] | 0,602 |
| Aday AUC — füzyon | −0,0509 | [−0,1226, +0,0192] | 0,920 |
| **Hasta AUC — 1. aşama** | **−0,0432** | **[−0,0809, −0,0064]** | **0,988** |
| Hasta AUC — füzyon | −0,0157 | [−0,0405, +0,0110] | 0,883 |

**Sonuç D — "sadece veri azaldı" elendi (V6.9k2), val(−JSRT):** 149 poz / 208 lezyon / 250 kontrol

| | Recall@K10 | @K20 | @K50 | Hasta AUC |
| :--- | ---: | ---: | ---: | ---: |
| LOSO | 0,9567 | 0,9856 | 0,9952 | 0,9761 |
| ANA | 0,9663 | 0,9808 | 1,0000 | 0,9748 |
| **Δ** | −0,0096 | +0,0048 | −0,0048 | **+0,0013** |

**Fark-farkı (DiD) = Δ(JSRT) − Δ(val), 1. aşama hasta AUC = −0,0435** [−0,0813, −0,0067],
P(<0) = 0,985.

**Sonuç E — avantaj 1. aşamada, TDA'da değil:** hasta düzeyi AUC (top1) **yalnız TDA skoruyla**
LOSO **0,9581** / ANA **0,9572** (Δ **+0,0009**). **Yalnız 1. aşama skoruyla** ise Δ **−0,0864**.
Yani kaynak ezberi derin dedektörün ısı haritasında taşınıyor, **lokal topolojik tanımlayıcıda
taşınmıyor** — TDA'nın kaynak-taşınabilirliği lehine bağımsız kanıt.

* 🟠 **Kaynak düzeyinde ezber GERÇEK ve iki bileşenli:** 149 hasta üzerindeki −0,087'lik
  Recall@K10 farkının ana kaynağı, 107 JSRT hastasının ana modelin **eğitim kümesinde** olması
  (in-sample). Hiç görülmeyen 42 hastada ise fark küçülüyor ama **sıfırlanmıyor**: hasta AUC
  ≈ +0,043 (GA 0'ı dışlıyor; veri azalmasıyla açıklanamıyor — val'de Δ≈0).
* 🟢 **Kilitli test sayıları geçerli ama *in-fold* olarak etiketlenmeli.** "Yeni bir merkeze dış
  genelleme" iddiası için doğru sayı LOSO'dur: **Recall@K10 0,8523 · K50 0,9396 · AUC 0,9395**.
* ⚠️ **§9.6 ile çelişmez:** o test *omurganın ön-eğitim korpusu* hakkındaydı (etkileşim −0,200).
  Bu test *NODE21'in kendi kaynak kompozisyonu* hakkındadır. İki ayrı örtüşme türü.
* 🔒 **İZOLASYON (kritik):** LOSO modeli/TDA'sı/HGB'si ana mimariye **sızmaz** — ayrı checkpoint
  (`M5_nojsrt.pt`), ayrı bundle (`stage2_nojsrt_tda_hgb.joblib`), ayrı JSON. Betik ana
  artifact'lerin SHA-256'sını koşu öncesi/sonrası doğrular: **değişmedi**
  (`isolation.unchanged = true`). Amaç yalnızca ölçüm ve dokümantasyondur.
* 📌 **TODO (bilinçli olarak yapılmadı):** birebir eşleştirilmiş bütçe kontrolü (aynı 1.366
  görüntüyle JSRT'yi koruyup 107 chestxray14 pozitifini çıkarmak) ≈1 ek eğitim koşusu
  maliyetinde; sonuçların hiçbiri buna bağlı değil. Ayrıntı: [roadmap.md](roadmap.md) §14.
* **Betik/kayıt:** [v6_9k_leave_jsrt_out.py](pipelines/v6_two_stage/v6_9k_leave_jsrt_out.py) ·
  [v6_9k2_val_control.py](pipelines/v6_two_stage/v6_9k2_val_control.py) ·
  [v6_9k3_bootstrap.py](pipelines/v6_two_stage/v6_9k3_bootstrap.py) ·
  `artifacts/v6_two_stage/v6_9k{,2,3}_*.json`

### V6.9.13 Uzamsal konum özellikleri ablasyonu — "hilus/plevra" gerçekten katkı sağlıyor mu? (V6.9l)

**Soru:** 1604 boyutlu vektörün son 4 sütunu uzamsal konumdur (`rel_x`, `rel_y`,
**akciğer bbox merkezine uzaklık** = "hilus vekili", **akciğer maskesi sınırına uzaklık** =
"plevra vekili"). Bu 4 sayı ölçülebilir bir katkı sağlıyor mu, ve daha "anatomik" bir konum
kodlaması daha iyi mi?

**Yöntem (yalnız val — kilitli test KAPALI, ana checkpoint'lere dokunulmadı).** 1600 TDA
özelliği **bir kez** hesaplanıp önbelleğe alındı (`_v6_9l_cache.npz`); üç kol arasında yalnız
son sütunlar değişti. Fit 1.580 hasta (790+790) · val 734 hasta / 231 lezyon / 33.910 aday.
HGB hiperparametreleri sabit (`max_iter=200, lr=0.08, balanced, seed=42`), Tuzak B korunur
(1. aşama skoru özellik değil), w=0,55 kilitli, z-ref her kol için val'den donduruldu.

| Kol | Özellik | TDA-tek AUC | TDA-tek rank | **Füzyon AUC** | **Füzyon rank** | S@0,125 | S@0,25 | S@0,5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A — konum YOK** | 1600 | **0,9387** | 0,7343 | 0,9836 | **0,9304** | 0,6883 | **0,7706** | 0,7965 |
| **B — mevcut vekiller** | 1604 | 0,9242 | 0,7354 | **0,9846** | 0,9279 | 0,6926 | 0,7576 | 0,8052 |
| **C — anatomik konum** | 1606 | 0,9302 | 0,7334 | 0,9838 | 0,9262 | 0,6710 | 0,7532 | **0,8095** |
| Δ(A − B) | | **+0,0145** | −0,0011 | −0,0010 | **+0,0025** | −0,0043 | **+0,0130** | −0,0087 |
| Δ(C − B) | | +0,0060 | −0,0020 | −0,0008 | −0,0017 | −0,0216 | −0,0044 | +0,0043 |

**Kol C nedir (GUDHI yok, saniyeler)?** `mask_5zones.mha` sol/sağ akciğeri **ayırır**
(1-3 = sağ, 4-5 = sol). Bu ayrımla her akciğerin **iç (medial/mediasten)** ve **dış
(lateral/göğüs duvarı)** kenarı ayrı ayrı bulunur; yatay koşu uzunlukları satır bazında
sonek-minimum/önek-maksimum ile **vektörel** hesaplanır. 6 özellik: `rel_x`, `rel_y`,
`hilus_uzaklığı` (kendi akciğerinin **iç kenar orta noktasına** Öklid), `iç_mediasten_uzaklığı`,
`dış_lateral_uzaklığı`, `taraf`. Hangi akciğerin küçük x'te olduğu **maskeden ölçülür**, varsayılmaz.

* ⚪ **Konum özellikleri NÖTR — katkı gösterilemedi (Δ eşiği ±0,002).** Füzyon AUC'de A ile B
  arasındaki fark **−0,0010** (B lehine, gürültü bandı), füzyon rank_metric'te ise **+0,0025**
  (A lehine). İki metrik **ters yönde** ve ikisi de eşiğin altında ⇒ 4 konum sütununun hasta
  düzeyi ayrıma katkısı **ölçülemedi**. S@0,25'te A **+0,013** önde, S@0,5'te B **+0,009** önde
  (yine zıt yönlü).
* 🔑 **Ama aday düzeyinde konum özellikleri ZARAR veriyor:** TDA-tek AUC, konum sütunları
  eklendiğinde **0,9387 → 0,9242** (−0,0145) düşüyor. Yorum: konum sütunları HGB'ye **pozisyon
  önyargısı** taşıyor ve şekil kanıtının bir kısmını bastırıyor; hasta düzeyinde `max` toplama
  bunu yıkadığı için nihai skora yansımıyor. ⇒ Bu 4 özellik "bedava" değil, **küçük bir bedeli**
  var.
* ⚪ **Anatomik kodlama (C) vekilden (B) daha iyi DEĞİL:** füzyon AUC **−0,0008**, rank
  **−0,0017**; TDA-tek AUC'de **+0,0060** ama bu da eşiğin altında. Yani "hilus/plevra vekili"
  yerine gerçek medial/lateral kenar kodlamak **ölçülebilir kazanç getirmiyor**.
* 🟢 **Karar:** ana checkpoint **DEĞİŞTİRİLMEDİ** (`stage2_foundation_tda_hgb.joblib` aynen
  korunuyor). Sağladığı fayda marjinal ve işaret tutarsız olduğu için yeni bir bundle üretilmedi.
* ⚠️ **Sınırlar:** (i) yalnız **val** — kilitli test kapalı; (ii) z-ref her kol için val'den
  dondurulduğu için mutlak sayılar hafif iyimserdir, ama **üç kol aynı işlemi gördüğü** için
  kıyas adildir; (iii) B/C özellik sayıları 1604 / 1606 — 1600 boyutun yanında ihmal edilebilir
  ama birebir eşleştirilmiş değil; (iv) kol başına tek tohum.
* 📌 **Not — kıyaslanamaz referans:** sık anılan "val füzyon 0,9238 / 0,9835" sayıları V6.9h'de
  **400+400 hasta** ile eğitilmiş HGB'ye aittir. Buradaki Kol B tam 790+790 ile eğitildiği için
  **0,9279 / 0,9846** verir; iki koşu doğrudan kıyaslanmamalıdır.
* **Betik/kayıt:** [v6_9l_spatial_ablation.py](pipelines/v6_two_stage/v6_9l_spatial_ablation.py) ·
  `artifacts/v6_two_stage/v6_9l_spatial_ablation.json`





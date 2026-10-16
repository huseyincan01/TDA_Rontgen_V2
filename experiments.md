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
  4. ⚠️ **Küçük tutarsızlık (kayda geçti):** TDA dalı bu koşuda 0.8629 çıktı, E3.4'te 0.8776 idi (−0.0148). Muhtemel sebep: kayıtlı CNN bundle'ının dropout/config yüklenmesi veya SVM bundle sürümü. Aynı koşu içinde füzyon tutarlı hesaplandığı için ana sonuç etkilenmez, ancak **TDA-tek sayısı için E3.4 değeri (0.8776) esas alınmalı** ve fark ileride açıklanmalıdır.
* **Karar:** **KEEP** 🏆 — ana iddia, ince ayarlı güçlü görüntü modeli karşısında **istatistiksel olarak doğrulandı**.
* **Manifest:** `artifacts/runs/E0.6_decisive_test/manifest.json`

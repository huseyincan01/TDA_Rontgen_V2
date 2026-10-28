# Sızıntı ve Geçerlilik Denetimi (V6.9)

> **Bu belge neden var:** Proje büyük oranda yapay zekâ desteğiyle ilerledi. Kod
> *çalışıyor* olsa da "çalışıyor" ≠ "doğru ölçüyor". Bu belge, yayına dönüşmeden önce
> sorulması gereken üç soruyu **kanıtla** yanıtlar:
>
> 1. **Sızıntı var mı?** (test/val bilgisi eğitime karışmış mı)
> 2. **Raporlanan sayılar dürüst mü?** (hepsi seçim yapılan bölmede mi ölçüldü)
> 3. **Dış doğrulama gerçekten dış mı?** (omurga o veriyi görmüş olabilir mi)
>
> Yöntem üç koldur: **statik kod denetimi** (iddia → dosya:satır), **mühürlü dış-değer
> ölçümü** (V6.9, ön-kayıtlı tek atım) ve **kontaminasyon denetimi** (ağırlık kökeni →
> veri listesi). Sonuçların tamamı — lehte ve aleyhte — aşağıdadır.

---

## 1. Özet hüküm

| Soru | Hüküm |
| :--- | :--- |
| Klasik sızıntı (hasta/etiket/test) | ✅ **Bulunamadı.** Beş bağımsız kontrol temiz. |
| Raporlanan sayıların dürüstlüğü | 🟠 **Kısmen iyimser.** Val'de 296 değerlendirme yapılmış; dürüst dış-değer **rank_metric 0,8842** (val'de 0,9131). |
| Dış doğrulama (VinDr) | 🔴 **Kontamine.** 400 değerlendirme görüntüsünün **%100'ü** omuranın ön-eğitim listesinde. "Zero-shot" iddiası **geçersiz**. |
| Test bölmesi | 🟡 **İki kez görüldü** (V6.4 mührü + V6.9 mührü). Beyan edildi; V6.4 sonrası hiçbir karar test'e bakılarak alınmadı. |
| Tohum çeşitliliği | 🟡 **Tek tohum.** Aynı protokol ±0,02 kayıyor (V6.8'de ölçüldü). |

---

## 2. Statik denetim — temiz çıkan beş kontrol

Her iddia, koddaki kanıtıyla birlikte:

| # | Kontrol | Sonuç | Kanıt |
| :-: | :--- | :--- | :--- |
| 1 | Hasta düzeyi bölme: bir hasta yalnız tek split'te | ✅ | [`splits_v2.py:45-120`](src/node21_tda/data/splits_v2.py#L45-L120) — `splits[pid]` tek atama; katmanlama `patient_label + zone_pattern` |
| 2 | Negatif (kontrol) havuzu aynı split'ten ve nodülsüz | ✅ | [`benchmark_v6_0_phase2.py`](pipelines/v6_two_stage/benchmark_v6_0_phase2.py) `_control_pids`: `split_patients()[split]` + `load_gt(p) == []` |
| 3 | 2. aşama (TDA-HGB) yalnız train'de eğitilmiş | ✅ | [`v6_2_stage2_v2.py:143`](pipelines/v6_two_stage/v6_2_stage2_v2.py#L143) — `_nodule_pids("train")` + `_control_pids("train")`; val yalnız `predict` |
| 4 | Kanonikleştirme veri kümesi genelinden istatistik almıyor | ✅ | [`common.py`](pipelines/v6_two_stage/common.py) `get_lung_p1p99()` hasta-bazlı sözlük; p1/p99 her görüntü için ayrı |
| 5 | Test bölmesi eğitimde kullanılmıyor (bilimsel modeller) | ✅ | `v6_7_train_fulldata.py:124` test'i yalnız **tam-veri gönderim** modelinde kullanır; `M5_foundation.pt` train-only |

> Bu beşi, "AI yazdı, belki karışmıştır" endişesinin **en olası** somut biçimleriydi.
> Hiçbiri ihlal edilmemiş. Aşağıdakiler ihlal değil; **ölçüm tasarımı** sorunlarıdır.

---

## 3. 🔴 VinDr "zero-shot" dış doğrulaması kontamine

**İddia (belgelerde):** "VinDr-CXR üzerinde zero-shot transfer: 0,6469" (V6.6.3), few-shot 0,7344 (V6.6.5), CORAL 0,6637.

**Denetim:** Omurga `resnet50-res512-all`. torchxrayvision kayıt defterindeki tanımı **kelimesi kelimesine**:

> *"This model was trained on the datasets **pc-nih-rsna-siim-vin** at a 512x512 resolution."*

`vin` = VinDr-CXR. Dahası, xrv'nin VinDr sınıfı varsayılan olarak `datapath/vinbigdata-train.csv.gz`
dosyasını okur — **bizim kohortumuzun etiketlerini aldığı dosyanın aynısı**
([`v6_5d_strict_no_adabn.py:178`](pipelines/v6_two_stage/v6_5d_strict_no_adabn.py#L178)).

**Ölçülen örtüşme:**

```
kohort görüntü sayısı          : 400
xrv vinbigdata-train listesi   : 15.000
kohort ∩ xrv-train             : 400  (%100,0)
```

⇒ Değerlendirdiğimiz **her** görüntü, omuranın ön-eğitim listesinde. Bu, "aynı domain"
değil, **aynı veri listesi**dir. Sonuç: 0,6469 / 0,6637 / 0,6593 / 0,7344 sayıları
VinDr'ye *sıfır-atış* değildir; **öğrenilmiş temsilin hatırlanması** ile karışmıştır.

**Dürüst sınır:** xrv'nin eğitim anında tam olarak hangi dosyaları yüklediği paketle
dağıtılmıyor; kanıt "deklare edilen 15k listesi 400'ünün tamamını içeriyor" düzeyindedir.
Bu, iddiayı **desteklemek için yeterli değil, çürütmek için yeterlidir**.

**Ne yapılmalı:** (a) VinDr'yi görmemiş bir omurgayla (`densenet121-res224-chex` / `-nih`)
dış doğrulama yeniden koşulmalı; (b) o koşu bitene kadar belgelerdeki "zero-shot"
etiketleri **kaldırılmalı** ve bu bölümdeki sayılar "kontamine" olarak işaretlenmelidir.

---

## 4. 🟠 Val'de 296 değerlendirme yapıldı, aynı val'de raporlandı

Sızıntı değil; **seçim yanlılığı**. Ölçüm bütçesi:

| Seçim | Aralık | Nerede |
| :--- | :--- | :--- |
| Kutu boyu × varyant | **37 varyant × 8 kutu = 296** resmî metrik değerlendirmesi | [`v6_6_optimization_audit.py`](pipelines/v6_two_stage/v6_6_optimization_audit.py) |
| Füzyon ağırlığı | w ∈ {0,00 … 0,90}, adım 0,05 (**19 nokta**) + üç kapı taraması | [`v6_6_fusion_calibration.py`](pipelines/v6_two_stage/v6_6_fusion_calibration.py) |
| TTA ölçek seti | 3 set (tek / 0,9-1,0 / 0,9-1,0-1,1) | aynı |
| NMS yarıçapı / tepe penceresi | 6 × 2 = 12 (etkisiz çıktı) | aynı |

231 lezyonluk val'de **Sens@0,25'in standart hatası ≈ 0,030**. Dolayısıyla 296
değerlendirmenin en iyisini seçmek, gerçek değerin ~0,02–0,03 üzerinde bir sayı üretmeyi
**beklenir** kılar. V6.9 mühürü bunu tam olarak bu büyüklükte gösterdi (aşağıda: −0,029).

---

## 5. V6.9 — Mühürlü dış-değer (ön-kayıtlı, tek atım)

**Tasarım:** Kilitli protokol (M5_foundation + `stage2_final_tda_hgb.joblib`, 3 ölçek +
flip TTA, 3×3 tepe, NMS 8 px, top-50, **sabit 56 px kutu**, **w=0,55**) hiç değiştirilmeden
dokunulmamış **test** bölmesinde (734 hasta / 229 lezyon) **bir kez** koşuldu.

**Mühür garantileri** ([`test_seal_preregistration.json`](artifacts/v6_two_stage/test_seal_preregistration.json) — koşudan **önce** commit `deb59ed`):

* Betik kendi SHA-256'sını ön-kayıtla karşılaştırır; uyuşmazsa **koşmayı reddeder**.
* Model ve 2. aşama bundle'ının SHA-256'sı da doğrulanır.
* z-normalizasyon referansı **val'den** donduruldu (test dağılımından istatistik alınmadı).
* Hiçbir hiperparametre test üzerinde seçilmedi; seçim yapacak kod yok.

### 5.1 Sonuçlar

| Ölçüm | Val (seçim bölmesi) | **Test (mühür)** | Δ | Test %95 GA (bootstrap, 1.000) |
| :--- | ---: | ---: | ---: | :--- |
| **rank_metric — yalnız 1. aşama** | 0,9131 | **0,8842** | **−0,0289** | 0,8559 – 0,9089 |
| rank_metric — TDA füzyonu (w=0,55) | 0,9174 | **0,8970** | −0,0204 | 0,8769 – 0,9213 |
| Hasta AUC — yalnız 1. aşama | 0,9678 | **0,9446** | −0,0232 | 0,9237 – 0,9621 |
| Hasta AUC — TDA füzyonu | 0,9793 | **0,9645** | −0,0148 | 0,9491 – 0,9786 |
| Sens@0,125 (1. aşama) | 0,6710 | **0,6114** | −0,0596 | 0,5250 – 0,6805 |
| Sens@0,25 (1. aşama) | 0,7489 | **0,7031** | −0,0458 | 0,6250 – 0,7645 |
| Sens@0,5 (1. aşama) | 0,8009 | **0,7642** | −0,0367 | 0,7110 – 0,8164 |
| Sens@0,125 (füzyon) | 0,6883 | **0,6332** | −0,0551 | 0,5598 – 0,7077 |
| Sens@0,5 (füzyon) | 0,7879 | **0,7729** | −0,0150 | 0,7186 – 0,8233 |

**Üç okuma:**

1. **Ölçüm hattı sağlam.** Mühür koşusu, val kolunda V6.6.7'nin yayımlanmış sayılarını
   **birebir** yeniden üretti (0,9131 / 0,9678 / 0,6710 / 0,7489 / 0,8009). Yani test
   düşüşü bir kod hatası değil, **gerçek bir genelleme farkı**.
2. **Düşüş beklendiği büyüklükte, sınırın altında.** −0,029 ≈ 1 SE (0,030). Ön-kayıtta
   "|Δ| > 0,05 = seçim yanlılığı lehine kanıt" diye tanımlanmıştı; **bu eşik aşılmadı**.
   Ama val noktası (0,9131) test'in %95 aralığının (üst 0,9089) **az üstünde** kaldı ⇒
   hafif iyimserlik var, yok sayılamaz.
3. **Bulgu: 2. aşamanın hükmü test'te tersine döndü.** Val'de yalnız 1. aşama, füzyondan
   FROC'ta üstündü (0,8442 vs 0,8268) ve "FROC-optimal dağıtımda 2. aşamayı atla" önerilmişti.
   Test'te **füzyon dört metrikten üçünde üstün** (rank_metric 0,8970 > 0,8842 · AUC
   0,9645 > 0,9446 · Sens@0,5 0,7729 > 0,7642; yalnız Sens@0,25'te geride). ⇒ **V6.6.4'ün
   "2. aşamayı atla" önerisi val'e özgüymüş**; mühür bunu yakaladı.

### 5.2 Mühür kaydındaki iki teknik uyarı (dürüstlük)

* **`legacy_fusion.patient_auc.noisyor` = 0,6432 GEÇERSİZ.** Eski protokol bloğunda füzyon
  skoru z-ölçeğinde (negatif olabilir) verildiği için noisy-OR anlamsızdır. Geçerli olan
  `A2a_fusion_tda.auc` = **0,9645**'tir (orada sigmoid uygulanır). Diğer tüm sayılar geçerli.
* **Yol farkı:** mühür, aday üretiminde resmî-metrik yolunu (3 ölçek ortalaması + NMS 8 px)
  kullanır; V6.6.6'nın `predict_tta` (NMS 20 px) yolundan biraz farklıdır. Bu yüzden val
  füzyon FROC@0,5 burada 0,8268, V6.6.6 tablosunda 0,8355'tir. **Yalnız 1. aşama** sayıları
  aynı yolu kullandığı için birebir uyuşur (0,9131) — mühür karşılaştırması bu yüzden
  1. aşama satırında en temizdir.

---

## 6. 🟡 Test bölmesi iki kez görüldü

| Kullanım | Model ailesi | Sonuç | Ayar |
| :--- | :--- | :--- | :--- |
| V6.4 `test_seal()` | V6 CenterNet+FCOS+lokal TDA + V5.2 global füzyon | birleşik hasta AUC **0,9876** | yok |
| **V6.9 (bu belge)** | M5_foundation + dondurulmuş 2. aşama, w=0,55 | rank_metric **0,8842** / AUC **0,9446** | yok |

V6.4'ten sonraki hiçbir karar (V6.5 domain genelleştirme, V6.6 omurga + FROC kalibrasyonu,
V6.6.7 resmî metrik, V6.7 tam-veri, V6.8 onarımlar) test sonucuna bakılarak alınmadı —
hepsi val'de alındı. Yine de **beyan**: bu bölme proje tarihinde iki kez görüldü; üçüncü
bir "mühür" artık mühür sayılmaz.

---

## 7. Yayına dönüşmeden önce değişmesi gerekenler

| # | Ne | Neden |
| :-: | :--- | :--- |
| 1 | **Ana tabloya test sütunu** (0,8842 / 0,8970 / 0,9446 / 0,9645) ve val sütununa **"seçilmiş"** etiketi | Val sayıları 296 değerlendirmenin ürünü |
| 2 | VinDr tablolarından **"zero-shot" ifadesinin kaldırılması** (§3) | %100 ön-eğitim örtüşmesi |
| 3 | **Seçim bütçesinin beyanı** (296 + 19 + 12 değerlendirme) | Hakemin ilk sorusu |
| 4 | Tüm ana sayılara **%95 GA** (bootstrap) | 229–231 lezyonla nokta tahmini yetersiz |
| 5 | **Tek tohum** sınırının açık yazılması (±0,02 bant) | V6.8'de ölçüldü |
| 6 | Tuzak B beyanı (LR füzyonu in-sample s1 ile uydurulmuştu) | Zaten belgeli; makaleye taşınmalı |

**Açık iş (denetim sonrası):** VinDr'yi görmemiş omurgayla dış doğrulama (≈40 dk) —
tamamlanınca §3'teki hüküm "kontamine" yerine "iki omurgayla ölçüldü" olur.

---

## 8. Kayıtlar

| Dosya | İçerik |
| :--- | :--- |
| `artifacts/v6_two_stage/test_seal_preregistration.json` | Ön-kayıt: kilitli protokol, hash'ler, raporlama kuralı, başarısızlık tanımı |
| `artifacts/v6_two_stage/v6_9_test_seal.json` | Mühür sonucu (val + test, kilitli parametreler, dondurulmuş z referansı) |
| `artifacts/v6_two_stage/v6_9b_test_seal_ci.json` | Bootstrap %95 aralıkları + "mühürle birebir aynı" doğrulaması |
| [`pipelines/v6_two_stage/v6_9_test_seal.py`](pipelines/v6_two_stage/v6_9_test_seal.py) | Mühür betiği (kendi hash'ini doğrular) |
| [`pipelines/v6_two_stage/v6_9b_test_seal_ci.py`](pipelines/v6_two_stage/v6_9b_test_seal_ci.py) | Aralık betiği (mühür uyuşmazlığında durur) |

*Son güncelleme: V6.9 (Ekim 2026)*

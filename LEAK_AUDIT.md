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
| Dış doğrulama (VinDr) | 🔴 Kontamine *ama* **ölçülebilir şişme yok**: 400/400 görüntü ön-eğitim listesinde; temiz omurgayla fark gürültü bandında (Δ top1 +0,004). "Zero-shot" etiketi yine de kullanılamaz — sayı **temiz omurgayla** raporlanmalı (bkz. §9) |
| **NODE21 iç kaynak örtüşmesi** | 🟠→✅ **Örtüşme gerçek, şişme yok.** Pozitiflerin %82,9'u omurganın gördüğü korpuslardan (NIH/PadChest) ama **kutular yeni** ⇒ lokalizasyon sızıntıya kapalı. Etkileşim testi **ters yönde**: FND−M5 farkı görülmeyen kaynaklarda daha büyük (**−0,200**, %95 GA 0'ı dışlıyor) ⇒ §9, §10 (🔒 kapalı) |
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

**Ne yapılmalıydı:** VinDr'yi görmemiş bir omurgayla dış doğrulamayı yeniden koşmak.
**Yapıldı** (§8): üç omurga, aynı tarif, yalnız ön-eğitim korpusu değişti. Sonuç
beklediğimizden **daha iyi**: kontaminasyon **ölçülebilir bir şişme üretmiyor**.

> 🔴 **DÜZELTME (bu belgenin ilk sürümünde yanlıştı):** İlk yazımda *"NODE21 omuranın
> ön-eğitim korpuslarının hiçbirinde yok"* demiştim. **Bu yanlıştı.** NODE21'in halka açık
> eğitim seti yeni hastane verisi değil, **dört açık veri kümesinden derlenmiş** bir
> yeniden etiketlemedir; pozitiflerin **%82,9'u** `chestxray14` (NIH) ve `padchest`'ten
> gelir — yani omurganın **gördüğü** korpuslardan. Ayrıntı ve ölçüm: **§9**.

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

> 🔄 **V6.9i güncellemesi (bu tablonun sonrası):** Mühürde kullanılan `stage2_final_tda_hgb.joblib`,
> eski M5 (ImageNet) adaylarıyla eğitilmişti. HGB'yi foundation'ın kendi adaylarıyla yeniden eğiten
> uyumlu bundle (`stage2_foundation_tda_hgb.joblib`), kilitli testte füzyonu **0,8970 → 0,9045**
> (rank) ve **0,9645 → 0,9687** (AUC) seviyesine taşıdı (4. test kullanımı, tek-atımlık). Ayrıntı:
> `experiments.md V6.9.10`. Bu, §5.1'in ana hükmünü (val iyimserliği ~0,02–0,03) değiştirmez; yalnızca
> dürüst dış-değeri hafifçe yukarı taşır.

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
| 7 | **Kaynak örtüşmesi beyanı** (§9): pozitiflerin %82,9'u NIH/PadChest'ten; kutular yeni; kaynak-ayrıştırmalı **ve** etkileşim tablosu eklenmeli | Hakem bunu mutlaka sorar; "sıfır sızıntı" iddiası kurulamaz |

**Uygulama durumu (Ekim 2026):** #2, #3, #4, #5, #6, #7 → **belgelerde uygulandı**
(`experiments.md §13`, `ablation_matrix.md §17`, `STORY.md §12.6`, `paper_draft.md §4.10 + §6`,
`README.md`, `progress.md §8`, `roadmap.md §14`, `v4_roadmap.md §7`, `literature.md §5.6`,
`data/external_vindr/README.md`, `submission_docker/README.md`). #1 → **kısmen**
(`paper_draft.md §4.10` test tablosu eklendi; ana tablonun tamamı yazım aşamasında).
**Kalan açık iş:** çok-seed koşusu (hata çubukları için) — §7'nin #5'i yazıldı ama
sayılar hâlâ tek tohumlu.

---

## 8. Kontaminasyonun ölçümü — temiz omurga deneyi (V6.9c)

Kontaminasyonun **varlığı** kanıtlandı (§3); şimdi **etkisini** ölçtük. Aynı tarif
(omurga donuk, 5 epoch, aynı veri/tohum/augmentasyon/kayıp), yalnız omurganın gördüğü
veri değişti:

| Satır | Omurga | VinDr ön-eğitimde? | NODE21 val nOR | NODE21 K20 | VinDr AUC top1 | VinDr nOR | VinDr bbox K50 |
| :-: | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| A | `resnet50-res512-all` | **var** | 0,9466 | 0,9740 | 0,6479 | 0,6414 | 0,8369 |
| B | `densenet121-res224-all` | **var** | 0,9440 | 0,9567 | 0,6406 | 0,6534 | 0,8416 |
| **C** | `densenet121-res224-nih` | **yok** | 0,8650 | 0,9091 | **0,6369** | **0,6663** | 0,8298 |
| | **B − C** | | **+0,0790** | +0,0476 | +0,0037 | **−0,0129** | +0,0118 |

**Üç okuma:**

1. ✅ **Şişme yok.** VinDr'yi görmemiş omurga, VinDr'de **en az** kontamine olanlar kadar
   iyi: top1 farkı +0,004, noisy-OR'da temiz omurga **daha iyi** (−0,013). 150 pozitif
   görüntüde AUC standart hatası ≈ 0,04 ⇒ bu farklar **gürültü bandında**. Yani
   "0,6469 kontaminasyon sayesinde yüksek" iddiası **desteklenmiyor**.
2. 🔑 **VinDr ön-eğitimi *hedef göreve* yarıyor, teste değil.** B−C farkı NODE21'de
   **+0,079** (büyük), VinDr'de ~0. Yorum: CXR ön-eğitimi daha iyi bir dedektör üretiyor;
   VinDr testi ise ~0,64–0,67'lik bir **domain-transfer tavanına** dayanmış — encoder'in
   o görüntüleri görmesi bu tavanı yükseltmiyor.
3. ⚠️ **Karıştırıcı (confound):** B ile C yalnız VinDr'de değil, **toplam korpus
   büyüklüğünde** de ayrışır (`all` = nih+pc+chex+mimic+google+openi+kaggle vs yalnız `nih`).
   Bu yüzden +0,079'luk NODE21 farkı **VinDr'ye özgü sayılamaz**; "daha çok CXR verisi
   daha iyi dedektör" okuması da aynı derecede geçerlidir.

**Yayın kararı:** VinDr dış doğrulaması **C satırıyla** (temiz omurga: 0,6663 nOR /
0,6369 top1) raporlanmalı; A satırı "kontamine üst sınır" olarak dipnotta durabilir —
çünkü fark gürültü bandında olsa da A satırının *hangi* görüntüleri gördüğü
kanıtlanamaz. Asıl dürüst cümle: *"VinDr transferi ~0,64–0,67 bandında bir tavan
gösteriyor; bu tavan, VinDr'yi görmemiş bir omurgada da aynı."*

**Kayıt:** `artifacts/v6_two_stage/v6_9c_contamination.json` ·
[`v6_9c_contamination.py`](pipelines/v6_two_stage/v6_9c_contamination.py)

---

## 9. 🟠 NODE21'in İÇİNDEKİ kaynak örtüşmesi (asıl mesele)

Bu bölüm, denetimin **en kritik** bulgusudur ve ilk sürümdeki bir hatamı düzeltir.

### 9.1 Veri nereden geliyor?

NODE21'in halka açık eğitim seti **yeni hastane verisi değildir**; dört açık veri
kümesinden gelen nodüllü röntgenlerin + bir kontrol havuzunun yeniden etiketlenmesidir.
Verinin kendi `patient_info.json` alanı bunu söylüyor (4.882 hasta, hepsi sayıldı):

| `source_dataset` | n | etiket | omurga gördü mü?<br>(`pc-nih-rsna-siim-vin`) |
| :--- | ---: | :---: | :---: |
| `node21_control` | 3.748 | hepsi **0** | **bilinmiyor** (kökeni kayıtta yok) |
| `chestxray14` (= NIH ChestX-ray14) | 617 | hepsi **1** | ✅ **evet** (`nih`) |
| `padchest` | 314 | hepsi **1** | ✅ **evet** (`pc`) |
| `jsrt` | 149 | hepsi **1** | ❌ hayır |
| `openi` | 54 | hepsi **1** | ❌ hayır |

⇒ Pozitiflerin **%82,9'u (931/1.134)** omuranın ön-eğitiminde **gördüğü** iki korpustan
geliyor. Yani *"NODE21 omuranın hiç görmediği veridir"* ifadesi **yanlıştır**; bu belgenin
ilk sürümünde bu hata yapılmıştı (§3'teki düzeltme kutusu).

> ⚠️ **Sizin verdiğiniz listeden iki sapma:** (i) NODE21 pozitiflerinde **CheXpert yok**,
> **JSRT** var; (ii) kontroller `node21_control` havuzundan geliyor ve bu havuzun özgün
> kaynağı bizim metadata'mızda **kayıtlı değil** — "kendi hastanelerinden" diyemeyiz.

### 9.2 Sızıntı ne düzeyde? (etiket mi, kutu mu)

| Bilgi türü | Omurga gördü mü? | Sonuç |
| :--- | :---: | :--- |
| Görüntü pikselleri | ✅ evet (%82,9 pozitif) | görüntü-düzeyi aşinalık mümkün |
| Kaba rapor etiketi ("Nodule=1") | ✅ evet (NIH/PadChest etiketleri) | sınıflandırma önyargısı mümkün |
| **NODE21'in radyolog kutuları** (x, y, w, h) | ❌ **hayır** | **lokalizasyon sızıntıya kapalı** |

NODE21'in ayırt edici katkısı tam olarak kutulardır: organizatörler açık veri
görüntülerini alıp **yeni radyolog anotasyonları** üretmiştir. Resmî metrik (IoU > 0,2)
bu yeni kutulara dayandığı için **kutu ezberi yapısal olarak imkânsızdır**; etkilenebilecek
tek şey dedektörün *duyarlılığı*dır.

### 9.3 Ölçüm — kaynak-ayrıştırmalı değerlendirme (V6.9d)

Yöntem: kilitli protokol **tek kez** koşuldu (mühür kaydıyla birebir doğrulandı), sonra
aynı aday listesi kaynaklara göre ayrıştırıldı. Test: omurganın gördüğü (chestxray14 +
padchest) ve görmediği (jsrt + openi) pozitiflerde performans ayrışıyor mu?

| Test (734 hasta / 229 lezyon) | n hasta | n lezyon | AUC (kontrole karşı) | S@0,125 | S@0,25 | S@0,5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| chestxray14 ✅ görüldü | 89 | 138 | 0,9760 | 0,6449 | 0,6957 | 0,7754 |
| padchest ✅ görüldü | 58 | 66 | 0,9604 | 0,6515 | 0,7273 | 0,8182 |
| jsrt ❌ görülmedi | 19 | 19 | 0,9147 | 0,3684 | 0,4737 | 0,5263 |
| openi ❌ görülmedi | 6 | 6 | 0,9920 | 1,000 | 1,000 | 1,000 |
| **GÖRÜLEN birleşim** | 147 | 204 | **0,9698** | 0,6471 | **0,7059** | 0,7892 |
| **GÖRÜLMEYEN birleşim** | 25 | 25 | **0,9332** | 0,5200 | **0,6000** | 0,6400 |
| **Fark** | | | **−0,037** | −0,127 | **−0,106** | −0,149 |

Val'de aynı yön, daha küçük fark: görülen **0,9826** / görülmeyen **0,9660** (Δ −0,017);
S@0,25 0,7448 → 0,6667 (Δ −0,078).

### 9.4 Okuma — dürüst sınırlar

1. ⚠️ **Fark ~1 SE.** Görülmeyen grupta yalnız **25 lezyon** (test) / 39 (val) var;
   duyarlılığın standart hatası ≈ 0,07–0,10. −0,106'lık fark **istatistiksel olarak
   belirleyici değildir**.
2. ⚠️ **Kaynak kimliği başlı başına güçlü bir değişken.** Bu dört kaynak farklı cihaz ve
   geometriye sahiptir (spacing: chestxray14 0,143 · padchest 0,275 · jsrt 0,309 ·
   openi 0,331 mm/px). A3 denetiminde **aynı model** için kaynak-içi AUC 0,78–0,94
   aralığındaydı (v3 tablosu, `ablation_matrix.md §7.3b`) ⇒ kaynaklar arası fark,
   ön-eğitim aşinalığı olmadan da beklenir.
3. ✅ **Yön tutarlı** (hem val hem testte görülen > görülmeyen) ⇒ *"ön-eğitim örtüşmesinin
   hiç etkisi yok"* denemez.
4. ✅ **Büyüklük sınırlı** ⇒ *"skorlar sahte"* de denemez: fark ≤0,04 AUC ve ≤0,11
   duyarlılık; toplam kazanç (ImageNet → CXR omurgası + TDA) bundan çok daha büyük.
5. 🔒 **Yarışma tarafı:** kurallar halka açık ön-eğitimli modellere izin verir; ancak
   **gizli test setinin bileşimini biz doğrulayamayız** — bu yüzden "sıfır sızıntı
   garantili" gibi bir cümle kurulmamalıdır. Aynı risk, halka açık ağırlık kullanan
   **tüm** katılımcılar için simetriktir.

### 9.5 Yayın için doğru cümle

> *"NODE21'in halka açık eğitim setindeki nodüllü olguların %82,9'u (ChestX-ray14 ve
> PadChest) kullandığımız ön-eğitimli kodlayıcının korpusuyla örtüşmektedir; ancak
> korpusta yalnız görüntü ve rapor düzeyi etiketler bulunur, NODE21'in radyolog kutuları
> **bulunmaz**. Kaynak-ayrıştırmalı değerlendirmede hasta AUC'si görülen kaynaklarda
> 0,970, görülmeyenlerde 0,933 (25 lezyon, ≈1 SE) ve lokalizasyon duyarlılığı 0,706 vs
> 0,600 olarak ölçüldü. Bu fark, kaynakların cihaz/geometri farklarıyla karışıktır;
> sonuçlarımızı örtüşmeden **tamamen arınmış** olarak sunmuyoruz."*

**Kayıt:** `artifacts/v6_two_stage/v6_9d_source_overlap.json` ·
[`v6_9d_source_overlap.py`](pipelines/v6_two_stage/v6_9d_source_overlap.py)

### 9.6 Belirleyici test — omurga × kaynak ETKİLEŞİMİ (V6.9e)

V6.9d'deki "görülen > görülmeyen" farkı **iki şeyle karışıktı**: kaynakların cihaz/geometri
farkı **ve** modelin kendisi. Ayrıştırmak için iki model **aynı protokolde** (heat-map TTA)
karşılaştırıldı:

* **M5** = ImageNet ResNet18 — hiç CXR görmemiş ⇒ **"temiz" omurga**
* **FND** = `resnet50-res512-vin` — pozitiflerin %82,9'unu görmüş ⇒ **"kontamine" omurga**

Ölçülen: her kaynak grubunda hasta düzeyi AUC ve **Δ = FND − M5**. Kritik istatistik
**etkileşim** = Δ(görülen) − Δ(görülmeyen).

**Üretim hattı doğrulandı:** M5_heatTTA recall@K10/K20/K50 = 0,9091 / 0,9654 / 0,9784 ve
FND_heatTTA = 0,9481 / 0,9827 / 0,9913 — yayımlanmış V6.6.2 değerlerinin **aynısı**.

| Val (734 hasta; görülen 138 poz/192 lezyon, görülmeyen 34 poz/39 lezyon, 562 kontrol) | M5 (temiz) | FND (kontamine) | **Δ = FND − M5** |
| :--- | ---: | ---: | ---: |
| **Görülen** kaynaklar (chestxray14+padchest) | 0,8386 | 0,9636 | **+0,1249** |
| **Görülmeyen** kaynaklar (jsrt+openi) | 0,6176 | 0,9427 | **+0,3251** |
| **ETKİLEŞİM** | | | **−0,2002** (95% GA −0,299 … −0,109; P(>0) = **0,000**) |

Aynı tablo top3 için −0,193 ve noisy-OR için −0,202 — üç toplama kuralında da aynı.

**Hüküm — örtüşme avantaj sağlıyor mu? HAYIR, ters yönde işaret:**

1. 🔴→✅ **Etkileşim tamamen negatif.** FND'nin ImageNet'e göre avantajı, omurganın **hiç
   görmediği** kaynaklarda **2,6 kat daha büyük** (+0,325 vs +0,125). Yani avantaj
   "hatırlama"dan değil, **genel CXR temsil kalitesinden** geliyor.
2. 🔑 **V6.9d'nin farkını asıl açıklayan şey kaynak zorluğu:** ImageNet omurgası JSRT/OpenI'de
   çöküyor (0,618), CXR omurgası çökmüyor (0,943). Görülen↔görülmeyen açığı M5'te **+0,221**,
   FND'de yalnız **+0,021** ⇒ **CXR ön-eğitimi kaynak-duyarlılığını ~0,20 AUC azaltıyor.**
   Bu, örtüşmenin *şişirdiği* değil, ön-eğitimin *sağlamlaştırdığı* bir tablodur.
3. ⚠️ **Sınırlar:** (i) yalnız **val** kullanıldı — test mühürlü, üçüncü kullanım yapılmaz;
   (ii) M5 ResNet18 ≈ 11M, FND ResNet50 ≈ 23M ⇒ **kapasite karıştırıcısı** var; (iii)
   görülmeyen grupta 34 hasta/39 lezyon.

**Kayıt:** `artifacts/v6_two_stage/v6_9e_backbone_interaction.json` ·
[`v6_9e_backbone_interaction.py`](pipelines/v6_two_stage/v6_9e_backbone_interaction.py)

---

## 10. 🔒 KAPANIŞ — ön-eğitim örtüşmesi konusu mühürlendi

**Soru:** Ön-eğitim korpusu NODE21'in kaynaklarıyla örtüştüğü için skorlarımız şişmiş olabilir mi?
**Cevap:** Örtüşme **gerçek**, ama **ölçülebilir bir şişme yok**; belirleyici test **ters yönde**.

| Ölçüm | Ne test edildi | Sonuç |
| :--- | :--- | :--- |
| **V6.9c** | Dış örtüşme (VinDr): 3 omurga, yalnız korpus değişti | Fark gürültü bandında (top1 +0,004; nOR'da temiz omurga **daha iyi**) |
| **V6.9d** | İç örtüşme: kilitli hattın görülen/görülmeyen kaynaklarda performansı | 0,9698 vs 0,9332 AUC (Δ −0,037, ≈1 SE, cihaz farkıyla karışık) |
| **V6.9e** | **Etkileşim:** FND−M5 farkı görülen vs görülmeyen kaynaklarda | **−0,200** [−0,299, −0,109], P(>0)=0 ⇒ avantaj görülmeyen kaynaklarda **daha büyük** |

**Mühürlü kararlar (yayın metni bunları kullanacak):**

1. **İddia edilmeyecek:** *"Sonuçlar ön-eğitim örtüşmesinden tamamen arınıktır"* — kanıtlanamaz.
2. **İddia edilecek:** *"Örtüşme vardır ve beyan edilmiştir; ancak kaynak-ayrıştırmalı ve
   etkileşim ölçümleri, skorların örtüşme kaynaklı şiştiğine dair kanıt göstermemiştir
   (etkileşim −0,200, %95 GA 0'ı dışlıyor, ters yönde)."*
3. **Raporlanacak tablolar:** §9.3 (kaynak-ayrıştırmalı) + §9.6 (etkileşim) birlikte.
4. **VinDr:** sayı **temiz omurgayla** (0,6663) verilecek; "zero-shot" etiketi kontamine
   satırlarda kullanılmayacak.
5. **Yarışma/sıralama:** gizli test setinin bileşimi doğrulanamadığı için **"sıfır sızıntı
   garantisi"** verilmeyecek.

**Bu mühür neyi kapatıyor?** Örtüşme konusunda **yeni ölçüm taleplerini** kapatıyor: mevcut
veriyle elde edilebilecek üç bağımsız test yapıldı ve üçü de aynı yönü gösteriyor.

**Mührü yeniden açacak tek şey** (exit criteria): (a) yeni ve **gerçekten dış** bir veri
kümesi, ya da (b) korpusu NODE21 kaynaklarıyla **kanıtlanabilir biçimde kesişmeyen** bir
omurga (ör. yalnız MIMIC-CH ile eğitilmiş DenseNet) ile aynı deneyin tekrarı.

**Mühür kaydı (SHA-256):**

| Dosya | SHA-256 |
| :--- | :--- |
| `pipelines/v6_two_stage/v6_9e_backbone_interaction.py` | `0baac12dc271e34197e4c686a241ce1d60d0901a0e68297d4b6d819f07a3a172` |
| `artifacts/v6_two_stage/v6_9e_backbone_interaction.json` | `2cb66f1f5e003c7c10b77b77355c538d2813a518c82e74eda5e28ddaffc4c2e9` |
| `artifacts/v6_two_stage/v6_9d_source_overlap.json` | `7c427d9208567ac8522a36c63e73eee55133f2b91c70dcad70a6c19e03b2adf5` |
| `artifacts/v6_two_stage/v6_9c_contamination.json` | `cbd3950236e48111f8783f01ddb0f37a6300a3c06b5d7f7495b53bf2e63b9746` |

**Mühür tarihi:** Ekim 2026 · **Durum:** 🔒 KAPALI

---

## 11. Kayıtlar

| Dosya | İçerik |
| :--- | :--- |
| `artifacts/v6_two_stage/test_seal_preregistration.json` | Ön-kayıt: kilitli protokol, hash'ler, raporlama kuralı, başarısızlık tanımı |
| `artifacts/v6_two_stage/v6_9_test_seal.json` | Mühür sonucu (val + test, kilitli parametreler, dondurulmuş z referansı) |
| `artifacts/v6_two_stage/v6_9b_test_seal_ci.json` | Bootstrap %95 aralıkları + "mühürle birebir aynı" doğrulaması |
| [`pipelines/v6_two_stage/v6_9_test_seal.py`](pipelines/v6_two_stage/v6_9_test_seal.py) | Mühür betiği (kendi hash'ini doğrular) |
| [`pipelines/v6_two_stage/v6_9b_test_seal_ci.py`](pipelines/v6_two_stage/v6_9b_test_seal_ci.py) | Aralık betiği (mühür uyuşmazlığında durur) |
| `artifacts/v6_two_stage/v6_9c_contamination.json` | VinDr kontaminasyon deneyi (3 omurga, §8) |
| `artifacts/v6_two_stage/v6_9d_source_overlap.json` | NODE21 iç kaynak örtüşmesi, kaynak-ayrıştırmalı (§9) |
| [`pipelines/v6_two_stage/v6_9d_source_overlap.py`](pipelines/v6_two_stage/v6_9d_source_overlap.py) | Kaynak-ayrıştırmalı değerlendirme betiği |

*Son güncelleme: V6.9 (Ekim 2026) — §9 kaynak örtüşmesi eklendi*


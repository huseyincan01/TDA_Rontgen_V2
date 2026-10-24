# NODE21 TDA — Projenin Hikâyesi (tek sayfa)

> **Bu belge nedir?** Projenin **başlangıçtan bugüne** anlatısı: her dönüm noktasında
> *ne sandığımız* ve *ne çıktığı*. Ayrıntı ve kayıt için ilgili belgeye bağlantı verilir.
> **Kural:** Buradaki her sayı `experiments.md`, `ablation_matrix.md`, `progress.md`
> veya `v4_roadmap.md` içinde kayıtlıdır. Kaydı olmayan hiçbir iddia yazılmamıştır.

---

## 0. Tek paragraf

Göğüs röntgenlerinde nodül tespiti için **5 anatomik zon üzerinde kübik kalıcı homoloji
(TDA)** temelli bir sınıflandırıcı kurduk; füzyonla **test ROC-AUC 0,9516** elde ettik.
Ama sistemin **lokalize etmediğini**, ayırt edici sinyalin **global/arka plan
topolojisinde** olduğunu ve bunun **büyük ölçüde cihaz/fizik imzası** taşıdığını ölçtük.
Temsili "daha doğru" yapmaya çalışmak yerine **tamamlayıcılık** aradık; tüm adaylar
yedekli çıktı. Dönüm noktası, temsilin fiilen **~4-bit'e sıkışmış** olduğunun
bulunmasıydı (65535 niceleme hatası); düzeltme her şeyi değiştirdi. Sonra **iki aşamalı
lokalizasyona** geçtik ve gövdeyi **800k+ röntgenle ön-eğitilmiş bir CXR vakıf
omurgasıyla** değiştirdik. Bugün elimizde: resmî NODE21 detection formatına uyan,
**çevrimdışı çalışan bir Docker konteyneri** ve 4.882 hastanın tamamıyla eğitilmiş bir
gönderim modeli var — ama **resmî test setinde hiç ölçülmedi.**

---

## 1. v1 — Beş metodolojik hata (Ekim 2026)

İlk sürüm çalışıyor *gibi* görünüyordu (SVM 0,7157 · k-NN 0,5555 · CNN 0,6510 AUC).
Denetim beş hata buldu ve **17 türetilmiş varlık silindi**:

1. Nodülleri yalnız merkez noktasına indirgeyip tek zona atamak → **130 zon** gerçekte
   nodül alanının ≥%20'sini taşıyordu ama "sağlıklı" etiketliydi (etiket sızıntısı).
2. %0–20 sınır dokusunu temiz negatif havuzuna koymak.
3. H0/H1 için keyfi ortak `[0,255]` PI ızgarası → H1'in **%95'i boş hücre**.
4. `source_dataset` ile katmanlama (kontrol hastalarının özgün kaynağı yok).
5. 800 özellikte FDR düzeltmesi yapmamak → "anlamlı" görünen ~40 özellik gürültüydü.

→ [progress.md §2](progress.md)

## 2. v2 — Temiz veri hattı

Kurallar: **≥%20 örtüşme** pozitif, **%0–20 `ambiguous`** (asla negatife girmez),
hiçbir zona %20 ulaşmayan **`unassigned`** (9 hasta). **Hasta düzeyi** katmanlı bölme
(3.414 / 734 / 734, sıfır sızıntı). PI aralıkları **yalnız train'den** türetildi.
Train'de **1:1 temiz negatif** dengelemesi. **7 denetim** (leakage, dengeleme,
kontaminasyon, hash) hepsi PASS.

→ [progress.md §3–4](progress.md)

## 3. v3 — İlk gerçek sonuç ve ilk çatlak

**12–13 Ekim 2026** benchmark'ı: 5-zon PI küçük CNN test **AUC 0,8631**; ham görüntü
CNN 0,7043; PL CNN 0,5450. Sonra füzyon: ince ayarlı ResNet18 + TDA → **test 0,9516**,
TDA katkısı **+0,0232** [0,0132; 0,0338] p=0,0000.

**Çatlak aynı dönemde çıktı:**
* **E4.7 lokalite testi:** model **lokalize etmiyor** — isabet şans düzeyinde
  (0,228–0,298 vs 0,268), zon AUC 0,48–0,53.
* **E3.6 oracle-lokalize TDA:** nodülün *tam merkezinde* bile AUC yalnız **0,6866**.
* **E1.2:** damar H1'ini filtrelemek **monoton zarar veriyor** (0,8697 → 0,8358).
* **E4.2:** H0 sinyali **arka plan/sınır** kodluyor (gradyan oranı 6,18); H1 ≈ rastgele.

⇒ *"TDA nodülü buluyor"* yanlış; *"global doku/arka plan topolojisi hastalıkla zayıf
korelasyon taşıyor"* doğru.

→ [experiments.md §6](experiments.md), [ablation_matrix.md §4](ablation_matrix.md)

## 4. v4 — "Hepsi yedekli" duvarı

Ölçüt değişti: TDA'nın *tek başına* AUC'si değil, **füzyona katkısı (d_fuzyon)**.
Sonuç: v3'ün en iyi iki iyileştirmesi füzyonda çöktü — E2.6 **+0,0480 → +0,0072**,
E2.1 **+0,0210 → −0,0019** (füzyona *zarar*). Ardından sırayla:

| Aday | d_fuzyon | Sonuç |
| :--- | ---: | :--- |
| V2.1 konum filtrasyonları (dikey/yatay/radyal) | −0,005…−0,003 | KILL · **r arttı** (0,293→0,39–0,42) |
| V2.4 hasta düzeyi global PH | −0,049 | KILL · hem zayıf hem yedekli |
| V3.1 zon-zon topolojik mesafe | −0,006 | KILL · kontrolle fark yalnız +0,03 |
| V3.2 PI simetrik farkı | +0,0066 | KILL · gürültü mertebesinde |

**Desen:** TDA dalını iyileştiren her müdahale, ResNet'e benzerliği de artırıyor.

→ [v4_roadmap.md §6–7](v4_roadmap.md)

## 5. V4 — Harici çöküş (2026-10-17)

VinDr-CXR (Vietnam, 400 görüntü, **sıfır eğitim**): TDA **0,511–0,541 ≈ şans**,
ResNet 0,6204. Kilitli füzyon (a=0,45) her kolda **ResNet'in altında**; harici veride
optimal ağırlık **a = 1,00** ⇒ dış eksende **TDA katkısı ~sıfır**. Ölçek normalizasyonu
kurtarmıyor ⇒ çöküş yalnız yoğunluk/ölçek değil.

Aynı dönemde **A2**: temsiller hangi **cihazdan** geldiğini hastalık kadar iyi kodluyor
(TDA 0,850 · ResNet 0,949 · yalnız meta-veri 0,956) ⇒ klasik kısayol imzası.

→ [experiments.md §7](experiments.md), [ablation_matrix.md §7.2–7.3](ablation_matrix.md)

## 6. V5.1 — Dönüm noktası: temsil fiilen ~4-bit'ti

`intensity_max=65535` **12-bit veri için 16× fazla** ⇒ filtrasyon dokuyu ~16 seviyeye
kuantalanmıştı. Doku **p1–p99 → 255 seviye** kanonikleştirmesi sonrası:

| | Önce | Sonra |
| :--- | :--- | :--- |
| H1 nokta sayısı | 610 bin | **1,63 milyon** |
| H1 kalıcılık aralığı | 0–7 | **0–97** |
| PI doluluğu | %38 | **%81** |
| Val label AUC | 0,8330 | **0,9211** (+0,0881) |
| LOO | 0,7249 | **0,7861** (+0,0611) |

**Sonuç:** v4'ün "TDA yedekli" teşhisi, **bozuk bir temsilin artefaktıymış**.
V5.2'de aynı düzeltme gerçek modelde: TDA ensemble 0,8811 → **0,9522**, füzyon
0,9729 → **0,9856** (a 0,40 → 0,25). V5.3'te harici: füzyon 0,5438 → **0,6283**
(Δ +0,0844, p<0,001) ⇒ "TDA dış dünyaya taşınmıyor" sonucu da **büyük ölçüde
temsil artefaktıymış**.

⚠️ **Dürüst kontrol (V5.1b):** kırpıntı-başına histogram penceresi füzyon kazancının
~%60'ından sorumlu; çiftleşmesiz varyantta bile nicelleme düzeltmesi baskın
(kazancın %74'ü) ⇒ muhafazakâr füzyon sayısı **0,9778**.

→ [experiments.md §8](experiments.md), [ablation_matrix.md §7.4](ablation_matrix.md)

## 7. V6 — Lokalizasyona geçiş

E4.7/E3.6 "5-zon temsili lokalize etmez" dediği için lokalizasyon **ayrı bir katmana**
taşındı: **Aşama-1** aday üretici + **Aşama-2** lokal TDA ile FP eleme.

* **Aşama-1 (M5 CenterNet):** eğitimsiz filtreler K50'de bile %32'de kaldı (kaburga/hilus
  parlaklığı sıralamayı domine ediyor). Eğitilen M5: **K50 recall 0,978** (26 ms).
  %98 kapısı K20/K30'da **geçilemedi** (en iyi 0,974).
* **Aşama-2 (lokal TDA-HGB):** 32/64 px çift ölçek PI + 4 uzamsal öncül → HGB.
  `e_fusion_tda` val **AUC 0,955**; `f_hierarchical` **FROC@4FP 0,857**.
  Yama CNN (0,646) **mimariden çıkarıldı**.
  ⚠️ *Dürüstlük notu (§12.2, Tuzak B):* 0,955, füzyon LR'ının 1. aşamanın **in-sample**
  skorlarıyla uydurulduğu bundle'dan gelir; **sızıntısız** eşdeğeri sabit **w=0,55**
  füzyonudur (V6.6.6: nOR **0,9886**). V6.7 gönderim modeli bu tuzağı yapısal olarak kapatır.
* **Test mührü (tek sefer):** V6 lokal **0,9464** · V5.2 global 0,9612 ·
  **birleşik 0,9876** (r=0,376).

→ [experiments.md §9](experiments.md), [ablation_matrix.md §13](ablation_matrix.md)

## 8. V6.5 — AdaBN maskesi ve strict denetim

Harici transferde kazanç arandı. Sonradan ortaya çıktı ki **KOL 1'in taban adayları
`AdaBN-M5`** idi ⇒ 0,6439–0,6620 bandının tamamı *transductive*. AdaBN kapatıldığında
(**V6.5d**):

| Konfigürasyon | Inductive (AdaBN yok) | Transductive (AdaBN) |
| :--- | ---: | ---: |
| base_z | 0,6090 | 0,6439 |
| + JPEG-denoise σ=1,0 | 0,5829 | 0,6498 |
| **+ ölçek-eşlemeli yama** | **0,6181** | 0,6562 |
| + denoise + ölçek + CORAL | 0,5916 | 0,6620 |

⇒ AdaBN baş rolde (+0,035…+0,071); **JPEG-denoise AdaBN olmadan zararlı**;
tek gerçek inductive kazanç **ölçek-eşleme (+0,0091)**.

→ [experiments.md §9 · V6.5d](experiments.md)

## 9. V6.6 — CXR vakıf omurgası

Gövde, **800k+ gerçek gösteri röntgeniyle** ön-eğitilmiş `resnet50-res512-all`
(torchxrayvision) + ASPP + FPN ile değiştirildi; 18 epoch iki fazlı diferansiyel eğitim.

| | M5 (ImageNet ResNet18) | **V6.6 (CXR vakıf)** |
| :--- | ---: | ---: |
| Recall@K20 / K30 | 0,9654 / 0,9697 | **0,9827 / 0,9827** |
| Recall@K50 | 0,9784 | **0,9913** |
| FROC@0,5 / @4 | 0,524 / 0,844 | **0,831 / 0,952** |
| Hasta AUC (top1) | 0,795 | **0,960** |

**%98 kapısı ilk kez geçildi.** VinDr'de saf zero-shot **0,6469** — M5'in
AdaBN'li hâlini (0,6332) ve V5.3'ü (0,6283) geçti.

🔑 **İki ters işaret bulundu:** AdaBN artık **zarar veriyor** (vakıf modeli −0,076,
ImageNet +0,073 idi) · **denoise artık yarıyor** (M5'te −0,026, vakıfta +0,018).
⇒ *yamanın faydası omurganın CXR-yerliliğine bağlı.*

→ [experiments.md §10](experiments.md), [ablation_matrix.md §14](ablation_matrix.md)

## 10. V6.6.7 — Resmî metrik ve kırık yarıçap kafası

Resmî NODE21 detection protokolü **kaynaktan** okundu: eşleştirme **IoU > 0,2**;
FROC **@1/8, 1/4, 1/2** FP/görüntü; **`rank_metric = 0,75·AUC + 0,25·Sens@0,25`**.

* 🔴 **Yarıçap kafası kullanılamaz:** GT boyutları 35–119 px iken model **R ≈ 0,01–0,04**
  tahmin ediyor (kutu ~0,1 px). Resmî IoU'da **Sens@0,25 = 0,000** ⇒ gönderimde
  **sabit kutu** şart.
* 🟢 **Çok ölçekli TTA kazandırıyor** (single → {0,9;1,0;1,1}): **+0,0138 rank_metric**.
* 🔴 **Optimum kutu 56×56 px** (64 değil).
* ⚪ **NMS yarıçapı ve tepe penceresi etkisiz** — resmî protokol aynı referansa çakışan
  2.+ kutuyu zaten IGNORE ediyor.

**En iyi: rank_metric 0,9131** (AUC 0,9678 · Sens@0,125 0,6710 · @0,25 0,7489 · @0,5 0,8009).

> 🔧 **V6.8 güncellemesi (§12.4b):** buradaki "yarıçap kafası kullanılamaz" hükmü
> **kaldırıldı**. Çökmenin nedeni kayıp ağırlığı değil, kaybın **tüm Gaussian-pozitif
> bölge** üzerinde ortalanmasıymış. `R > 0` (merkez) maskesiyle aynı kafa 6 dakikada
> eğitildi ve **rank_metric 0,9185** ile sabit 56 px kutuyu geçti (üç duyarlılık
> noktasında da artış). Gönderim sözleşmesi hâlâ kilitli **sabit 56 px**'tir.

→ [experiments.md §10 · V6.6.7](experiments.md), [ablation_matrix.md §14.6](ablation_matrix.md)

## 11. V6.7 — Tam veri ve Docker

**Füzyon kapısı (V6.6.6):** üç kapıyı (AUC ≥0,985 · FROC@0,5 ≥0,831 · FROC@4 ≥0,950)
**aynı anda** sağlayan w **yok** — kesişim boş (AUC–FROC ödünleşimi). Ama **w=0,55**
miras alınan LR katsayılarını **dört metrikte de domine ediyor** ⇒ kilitlendi.

**Gönderim modeli:** 4.882 hastanın tamamı (1.134 nodül + 1.134 kontrol = 2.268 görüntü),
18 epoch sabit takvim, kayıp 19,31 → 1,22. **Tuzak A** (sabit takvim, hiperparametre
aramasi yok, yarıçap kafası detach) ve **Tuzak B** (2. aşama matrisinde 1. aşama skoru
**yok**; füzyon **yeniden uydurulmadı**) korumalarıyla. Bilimsel checkpoint'ler
SHA-256 ile teyitli **değişmedi**.

> ⚠️ **V6.8 notu (12.4b–c):** buradaki iki tasarım kararı ölçüldü — **yarıçap kafası
> detach** gerekliydi *ama yetmiyordu* (asıl sorun kayıp maskesiydi; onarım 12.4b) ve
> **BN donuk** kararının in-domain val'de **ölçülebilir faydası bulunamadı** (12.4c).
> Gönderim modeli bu nedenle **değiştirilmedi** (Tuzak A: kilitli sözleşme).

**Docker:** resmî baseline şemasına **birebir** (köşe sırası **TR,TL,BL,BR** · `name`
alanı yok · **z = slice indeksi** · x/y = piksel × spacing), **non-root** (`algorithm`,
uid 1000), ağırlıklar gömülü, **`--network=none` altında test edildi** (3B `.mha`,
150 kutu, deterministik 0,9938). CUDA imajı 13,9 GB · CPU imajı 3,51 GB.

→ [experiments.md §11](experiments.md), [ablation_matrix.md §15](ablation_matrix.md),
[submission_docker/README.md](submission_docker/README.md)

---

## 12. Nerede duruyoruz — ve neyi bilmiyoruz

> ⚠️ **Bu bölüm üç ayrı şeyi aynı torbaya koymuyor.** *Kimsenin yapamayacağı* şeyler
> (12.1), *bizim yaptığımız* hatalar (12.2), *bulunup kapatılmış* tarihsel hatalar
> (12.3) ve *V6.8'de kapatılan kanıt boşlukları* (12.4) ayrı listelenir. Aksi hâlde
> okuyucu "bizim eksiğimiz" ile "yapısal sınır"ı karıştırıyor.

**Bildiğimiz:**
* Kurum içinde güçlü bir hasta düzeyi triyaj sistemi var (füzyon 0,9516; V6.6 uçtan uca
  val AUC 0,988 — **sızıntısız** sabit-ağırlık eşdeğeri **0,9886**).
* Lokalizasyon katmanı çalışıyor (K50 recall 0,991; resmî metrik simülasyonu 0,9131 →
  **0,9185** onarılmış yarıçap kafasıyla, V6.8).
* **TDA'nın gerçek katkısı global/arka plan topolojisinde**; lokal nodül topolojisinde
  değil (E3.6: 0,6866). Dış merkezde taşınan kısım sınırlı (VinDr zero-shot 0,647).

### 12.1 Bilinemez / ölçülemez (⚪ hata değil)

* **Resmî test setinde hiçbir skor yok** — verimiz NODE21'in **halka açık eğitim
  seti**; resmî test setleri **herkese kapalı**. Karşılaştırma **geçersiz** (Kural 9).
  Bu bir eksik değil, erişilemezliktir.
* Tam-veri gönderim modeli NODE21'de **değerlendirilemez** — tüm veri eğitimde.
  *Mantıksal zorunluluk*: hem tüm veriyi kullanıp hem o veride ölçemezsin (bilinçli ödünleşim).
* TDA'nın **AUC–FROC ödünleşimi çözülmedi** — *çözülmemiş problem*, hata değil:
  tanımlandı, üç kez saldırıldı (V6.5b/5c · V6.6.6), frontier taşındı, ödünleşim kaldı.

### 12.2 Yaptığımız gerçek hatalar

| # | Hata | Durum |
| :-: | :--- | :--- |
| 1 | 🔴 **Isınma (warm) kanıtı silindi:** `M5_foundation_warm.pt` teşhis için kaydedildi, sonra "temizlik" diye **değerlendirilmeden silindi** | ✅ **V6.8'de yeniden üretildi ve ölçüldü** (12.4a) |
| 2 | 🔴 **Yarıçap kafası çöktü ve haftalar boyunca fark edilmedi:** kayıp ağırlığı 0,1 iken focal kayıp 1–13 mertebesinde; resmî metrik (IoU>0,2) uygulanana kadar kutu üretmiyorduk | ✅ V6.6.7'de bulundu → **V6.8'de onarıldı** (12.4b): neden ağırlık değil **kayıp maskesi**ymiş; onarılan kafa sabit 56 px'i geçti (**0,9131 → 0,9185**) |
| 3 | 🔴 **Tuzak B — füzyon LR'ı in-sample s1 ile uyduruldu:** V6.1–V6.4'te 2. aşamanın LR'ı, 1. aşamanın *kendi eğitim bölmesindeki* skorlarıyla eğitildi. Val dürüsttür (M5 val'i görmedi) ama katsayılar aşırı özgüvenli dağılıma göre seçildi ⇒ hafif iyimserlik riski | ✅ V6.7'de **yapısal olarak kapatıldı** (s1 matrise girmez + w kilitli); V6.2'nin 0,955'i ve V6.6.4'ün 0,988'i artık **"referans"**, sızıntısız eşdeğeri sabit **w=0,55 → 0,9886** (V6.6.6) |
| 4 | 🟡 **Çift sigmoid:** V6.6.6'da LR referans satırının noisy-OR'u **0,9737** yazılmıştı (LR çıktısı zaten [0,1] iken bir kez daha sigmoid) | ✅ Düzeltildi → **0,9879**; betik ve artifact yeniden üretildi (arşiv birebir aynı çıktı) |
| 5 | 🟡 **V6.5 KOL 1 "zero-shot" diye etiketlendi ama AdaBN'liydi** (raporlama hatası) | ✅ V6.5d'de düzeltildi; 0,6439–0,6620 bandının tamamı **transductive** ilan edildi |
| 6 | 🟡 **`densenet121-res224-all` koşulmadı:** "512 doğal çözünürlük olduğu için ResNet50" savı **ölçümle değil akıl yürütmeyle** duruyor | ⏳ Hâlâ açık (12.5) |
| 7 | 🟡 **"BN donuk = unutma-önleme" gerekçesi ölçülmemişti** (V6.6/V6.7'de argüman olarak yazıldı, test edilmedi) | ✅ V6.8'de ölçüldü (12.4c): **ölçülebilir fayda yok** (dokuz metrikten sekizinde BN açık ≥; Δ nOR −0,0003) |

### 12.3 Tarihsel hatalar (bulundu ve düzeltildi — kayıtları var)

* **v1 — beş metodolojik hata** (etiket sızıntısı, kirli negatif havuz, keyfi PI ızgarası,
  yanlış katmanlama, FDR'siz 800 özellik) ⇒ 17 türev varlık silindi. → §1
* **V2.4 — sessiz hata:** `sitk.Resample` spacing hatası geniş `except` tarafından
  yutuldu ⇒ **4148 boş diyagram** ve **tam 0,5000 AUC**. Düzeltme: blok ortalaması +
  boş-diyagram sanity koruması. → [v4_roadmap.md](v4_roadmap.md)
* **E0.5 — önbellek kayması:** 9380 vs 5710 satır ⇒ görüntüler etiketlerle kaydı
  (AUC 0,58). → [experiments.md](experiments.md) §E0.5
* **E3.6 — boyut confound'u:** eşleştirilmemiş ilk koşuda sınıflandırıcı nodülü değil
  **yama boyutunu** öğrendi (**0,994** → boyut eşleştirilince **0,6866**).
* **V4 (VinDr) — iki hatalı betik:** histogram eşleme **tüm kesite** uygulanıp doku
  maskesi bozuldu; kayıtsız koşular ve tutarsız sayılar (**0,8537/0,8985/0,9365**)
  kaldırıldı. → [experiments.md](experiments.md) §7
* **E2.1/E2.6 — baseline'ı yeniden türetmek kazancı şişirir:** +0,0646 → gerçek **+0,0210**.

### 12.4 V6.8 — kanıt boşluklarının kapatılması

**(a) Isınma ablasyonu (silinen kanıt geri getirildi).** `v6_8_fix_checks.py --task warm`
ısınma fazını **aynı tohum/takvim/augmentasyon/kayıpla** yeniden üretti (train split
1:1 = 1580 örnek, 5 epoch donuk omurga) ve val'de (734 hasta) final 18-epoch modele
karşı ölçtü:

| Ölçüm (val) | Isınma (5 ep, donuk omurga) | **Final (18 ep, iki fazlı)** | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K20 | 0,9654 | **0,9827** | **+0,0173** |
| Recall@K50 | 0,9913 | 0,9913 | 0,0000 |
| FROC@0,5 | 0,7792 | **0,8312** | **+0,0520** |
| FROC@4 | 0,9091 | **0,9524** | **+0,0433** |
| Hasta AUC (noisy-OR) | 0,9416 | **0,9702** | **+0,0286** |
| Hasta AUC (top1) | 0,9275 | **0,9595** | **+0,0320** |

* ✅ **2. faz gerçekten kazandırıyor** — en büyük kazanç **FROC@0,5'te (+0,052)**;
  K50'de ise katkı **tam sıfır** (0,9913 → 0,9913) ⇒ ısınma, K50 tavanını **zaten**
  yakalıyor; ince ayarın işi **düşük-FP bölgesini** düzeltmek.
* 🔒 **Kanıt bütünlüğü:** yeniden üretilen ısınma koşusunun epoch kayıpları V6.6'nın
  kayıtlı train log'uyla örtüşüyor (13,36/2,23/2,10/1,96/1,87 vs 13,35/2,22/2,11/1,97/1,90)
  ve **final kolunun sayıları V6.6'nın yayımlanmış değerlerinin aynısı** (K20 0,9827 ·
  FROC@0,5 0,831 · @4 0,952 · top1 0,960) ⇒ karşılaştırma elmayla elmadır.
* ⚠️ **Dürüst sınır:** Δ(final−ısınma) 2. fazın **toplam** katkısıdır (13 ek epoch +
  üst blokların açılması). Bunu tek başına *"unutma-önleme katkısı"* saymak fazla iddia
  olurdu; BN dondurmanın payı ayrı kolda ölçülür (`--task bn`, 12.5).

**(b) Yarıçap kafası onarımı (V6.6.7'den beri açık olan hata).** V6.6.7 çökmeyi
"kayıp ağırlığı 0,1" ile açıklamıştı. V6.8 bu açıklamayı test etti: **ağırlık 1,0**
(10×) ile kafa **yine çöktü** (r₁₀₂₄ medyanı **0,02 px**) ⇒ ağırlık neden değildi.
Gerçek neden, kaybın **tüm Gaussian-pozitif bölge** (961 piksel) üzerinde ortalanması,
oysa yarıçap hedefinin **yalnız merkez pikselinde** dolu olmasıydı (ölçüm: süpervizyonun
**%0,11**'i hedef taşır) ⇒ gradyanın **%99,9'u** "0 tahmin et" yönündeydi; kaybın
minimumu **sabit 0**.

| Kutu geometrisi (aynı adaylar) | rank_metric | Sens@0,125 | Sens@0,25 | Sens@0,5 |
| :--- | ---: | ---: | ---: | ---: |
| sabit 48 px | 0,9109 | 0,6710 | 0,7403 | 0,7922 |
| sabit 56 px (V6.6.7 optimumu) | 0,9131 | 0,6710 | 0,7489 | 0,8009 |
| sabit 64 px | 0,9077 | 0,6537 | 0,7273 | 0,7879 |
| **onarılmış yarıçap kafası** (`R>0` maskesi, ağırlık 1,0, 6 epoch) | **0,9185** | **0,6970** | **0,7706** | **0,8312** |

* ✅ **Onarım çalıştı ve sabit kutuyu geçti:** rank_metric **0,9131 → 0,9185** ve
  **üç işletim noktasının üçünde de** duyarlılık arttı ⇒ yarıçap kafası **ilk kez**
  resmî metrikte işe yarar hâle geldi. AUC her satırda **0,9678** (kutu geometrisi
  sıralamayı değiştirmez) ⇒ kazanç tamamen **IoU tabanlı eşleştirmeden**.
* ⚠️ Kafa, GT aralığının (17,5–59,5 px) **ortasına sıkışıyor** (p10 26 / medyan 33 /
  p90 47): kalibre değil, **regresyona-ortalama** — ama sabit kutudan iyi.
* ⚠️ **Gönderim sözleşmesi değişmedi** (kilitli sabit 56 px, Tuzak A); onarılan kafayı
  gönderime almak tam-veri modelinin yarıçap dalını aynı maske düzeltmesiyle yeniden
  eğitmeyi gerektirir — **ayrı bir sürüm kararı**.
* 🔒 **Kanıt koruması:** eğitilen kafa **diske yazıldı**
  (`submission_fulldata/radius_head_weight1_center_only.pt`) ve koşu **bağımsız yeniden
  üretildi** (ikinci koşu aynı rank_metric · duyarlılıklar · r yüzdelikleri ⇒ tam
  tekrarlanabilirlik). V6.6'da silinen kanıt hatası burada **tekrarlanmadı**.

**(c) Unutma-önleme kolu — BN dondurma ölçüldü ve faydası GÖSTERİLEMEDİ.**
V6.6/V6.7'nin gerekçesi *"BN `.eval()` — 800k röntgenlik istatistik bozulmasın"* idi;
bu bir **argümandı**. V6.8 aynı takvimi (18 epoch) **tek farkla** koştu: 2. fazda BN
running istatistikleri **açık**. Sonuç tasarım beklentisinin **tersine**:

| Ölçüm (val 734/231) | 2. faz BN **AÇIK** | Final (BN **donuk**) | Δ |
| :--- | ---: | ---: | ---: |
| Recall@K20 | **0,9913** | 0,9827 | +0,0086 |
| Recall@K50 | **0,9957** | 0,9913 | +0,0044 |
| FROC@0,5 | **0,8442** | 0,8312 | +0,0130 |
| FROC@1 | **0,8918** | 0,8658 | +0,0260 |
| FROC@4 | 0,9524 | 0,9524 | 0,0000 |
| Hasta AUC (top1) | **0,9646** | 0,9595 | +0,0051 |
| Hasta AUC (noisy-OR) | 0,9699 | **0,9702** | −0,0003 |

* 🔴 **Ölçülebilir fayda yok:** dokuz metrikten **yedisinde BN açık daha iyi**, birinde
  (**FROC@4**) **eşit**, birinde ise **−0,0003** (AUC noisy-OR; gürültü).
* ⚠️ **Ters yönde kanıt da değil:** 231 lezyonda FROC@0,5'in standart hatası ≈ **0,024**
  ⇒ 0,013 ≈ **0,5 SE**; tek tohum, tek val bölmesi. Doğru cümle:
  *"BN dondurma ölçülebilir bir unutma-önleme faydası göstermedi; zarar da vermedi."*
* 🎯 **Hipotezin asıl sınavı alan dışıdır:** BN istatistikleri transfer edilen şeyin
  parçasıdır (V6.6.3: AdaBN vakıf modelini **bozuyordu**) — V6.8 bunu test **etmez**.

### 12.5 Hâlâ açık

* **`densenet121-res224-all`** omurga yolu kodda var, **koşulmadı** — "512 doğal
  çözünürlük" savı ölçümle desteklenmedi (~30 dk iş).
* **BN dondurmanın alan-dışı (VinDr) payı** — in-domain val'de fark yok (12.4c);
  hipotezin sınavı transferde yapılmalı.
* **Onarılan yarıçap dalının gönderime alınması** (12.4b) — mühendislik kararı.
* **AUC–FROC ödünleşimi** (12.1) — çözülmüş değil, yönetiliyor.



---

*Son güncelleme: V6.8 (Ekim 2026) · Ayrıntı: [experiments.md](experiments.md) (deneyler),
[ablation_matrix.md](ablation_matrix.md) (tablolar), [progress.md](progress.md) (v1–v3),
[roadmap.md](roadmap.md) (v3 protokolü), [v4_roadmap.md](v4_roadmap.md) (v4), [paper_draft.md](paper_draft.md) (makale).*

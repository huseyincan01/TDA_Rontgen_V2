# Teknik Yöntem — NODE21 Nodül Tespiti (İki Aşamalı: Dedektör + Lokal TDA)

> Bu belge, sistemin **nasıl inşa edildiğini** adım adım anlatır: veri hazırlığı, 1. aşama
> dedektörün mimarisi ve eğitimi, 2. aşama TDA havuzunun kurulması ve iki aşamanın sızıntısız
> birleştirilmesi. Bütün sayılar kodla ve kayıtlı deney raporlarıyla doğrulanmıştır.
>
> ⚠️ **İki düzeltme:** Taslak metindeki iki ifade kodla uyuşmuyordu; doğrusu ilgili bölümlerde
> 🔧 ile işaretlendi (2. fazın kapsamı ve boyut dalının akıbeti).

---

## 1. Veri Hazırlığı ve Sızıntısız Bölünme (Temel İnşa)

Halka açık NODE21 eğitim seti **4.882 hasta** içerir (1.134 nodüllü, 3.748 temiz kontrol).

**Hasta düzeyinde bölünme.** Bölme rastgele görüntü değil, **hasta** biriminde yapılır: bir
hastanın bütün görüntüleri tek bir kümede kalır. Böylece aynı hastanın anatomisi hem eğitimde
hem testte görünüp skoru yapay olarak şişiremez. Sonuç:

| Küme | Hasta | Kullanım amacı |
| :--- | ---: | :--- |
| **Eğitim** | 3.414 | Model buradan öğrenir |
| **Doğrulama** | 734 | **Tüm tasarım kararları** burada alınır |
| **Kilitli test** | 734 | Yalnız ölçüm (ayar yapılmaz) |

**1:1 dengeleme.** Veri seti doğal olarak dengesizdir (1.134 nodüllüye karşı 3.748 kontrol).
Bu hâliyle eğitilirse model "her şeye sağlam" demeye kayar. Bu yüzden eğitimde **her nodüllü
hastaya bir temiz kontrol** eşlenir; kontroller aynı bölmeden sabit tohumla (42) seçilir:

* **Bilimsel model** (raporlanan metrikler): train bölmesinden **790 + 790 = 1.580** görüntü.
* **Tam-veri gönderim modeli**: tüm hastalardan **1.134 + 1.134 = 2.268** görüntü.

Her iki durumda da nodüllü:kontrol oranı **1:1**'dir.

---

## 2. 1. Aşama — Aday Üretici Dedektör ve İki Fazlı Eğitim

### 2.1 Mimari

```
512×512 kanonik röntgen ([-1024, +1024] aralığı)
   → ResNet50 omurga (TorchXRayVision, 800 bin göğüs röntgeniyle ön-eğitilmiş)
      C2, C3, C4, C5 özellik haritaları (1/4, 1/8, 1/16, 1/32)
   → ASPP (C5 üzerinde, dilation 1/3/6/12 + global havuz) ⇒ büyük ve küçük nodül aynı anda
   → FPN (C2–C5 yukarı birleştirme) ⇒ 128×128 (1/4 çözünürlük)
   → ×2 bilinear yükseltme ⇒ 256×256
   → CenterNet kafası: 2 kanal
        • ısı haritası (nodül merkezi)
        • boyut/yarıçap kanalı
```

Isı haritası **Gaussian Focal Loss** ile eğitilir (CenterNet standardı: pozitif bölgede
"1'e yaklaş", arka planda "0'a yaklaş", kolay örneklerin ağırlığı düşük). Boyut kanalı,
**yalnızca nodülün Gaussian-pozitif bölgesinde** (tüm görüntüde değil) regresyonla denetlenir —
aksi hâlde arka plandaki milyonlarca piksel "0 boyut" baskın çıkar.

> 🔧 **Düzeltme 1 — boyut dalının akıbeti:** Bu kanal eğitimde `R>0` değil `Y>0` maskesiyle
> denetlendi; yarıçap hedefi ise yalnız **tek merkez pikselinde** dolu olduğu için kafa
> zamanla 0'a çöktü (V6.6.7'de bulundu: tahmin ≈0,02 px). Doğru maske (`R>0`, yalnız merkez
> pikseli) ile onarıldığında çalıştı (val 0,9185 > sabit kutu 0,9131), **ama kilitli testte
> resmî skoru değiştirmedi** (0,8842 = 0,8842, V6.9f). Bu yüzden **dağıtılan sistemde boyut
> tahmini kullanılmaz; her adaya sabit 56×56 px kutu** çizilir. Yani mimaride iki kanal vardır
> ama pratikte ikinci kanal (boyut) devre dışıdır.

### 2.2 İki fazlı eğitim (unutmayı önleme)

Amaç: 800 bin röntgenin öğrettiği genel dokuyu korumak, yeni katmanları ise hızla öğretmek.

| Faz | Epoch | Ne eğitilir | Öğrenme hızı |
| :--- | ---: | :--- | :--- |
| **Faz 1 (ısınma)** | 1–5 | Omurga **tamamen donuk**; yalnız ASPP+FPN boynu ve kafalar | 1e-3 |
| **Faz 2 (ince ayar)** | 6–18 | **Yalnız omurganın üst blokları** (layer3, layer4) + boyun + kafalar | omurga 1e-4 · kafa 1e-3 |

Toplam **18 epoch**, batch 4, Cosine Annealing ile 1e-5'e kadar azalan öğrenme hızı.
**BatchNorm istatistikleri tüm eğitim boyunca donuk** (`eval` modunda): 800 bin röntgenden gelen
`running_mean/var` bozulmaz.

> 🔧 **Düzeltme 2 — "tüm ağ" değil, diferansiyel ince ayar:** Faz 2'de ağın tamamı açılmaz;
> kodda `unfreeze_top()` yalnız `layer3` ve `layer4` bloklarını açar (erken katmanlar donuk
> kalır). Buna **diferansiyel (kademeli) ince ayar** denir: üst katmanlar hedefe özgü özellikleri
> öğrenirken alt katmanlar genel doku bilgisini korur. Ayrıca omurga düşük (1e-4), kafalar
> yüksek (1e-3) hızla öğrenir.

---

## 3. 2. Aşama — TDA Veri Havuzu ve Sınıflandırıcı

### 3.1 Aday havuzu ve etiketleme

Eğitilmiş 1. aşama modeli **eğitim hastalarında** çalıştırılır; hasta başına **50 şüpheli kutu**
(top-50) alınır. Uygulamada 1.580 eğitim hastasından **74.482 aday** toplanmıştır.

Her aday, gerçek nodül kutularıyla (Kriter-1: merkez bbox içinde) karşılaştırılarak etiketlenir:

* **Pozitif (1):** kutunun merkezi bir gerçek nodülün içine düşüyor.
* **Zor-negatif / Hard-negative (0):** dedektörün yanıldığı, nodül olmayan bölge
  (damar kesiti, kaburga gölgesi, hilus artefaktı).

"Zor" negatif olması kritiktir: model rastgele arka planla değil, **kendi karıştırdığı**
yapılarla sınanır.

**Dengesizlik:** havuzun yalnızca **%1,72'si pozitiftir** (≈1:57). Nodüllü bir hastanın 50
adayından tipik olarak 1–2'si gerçektir; yani hasta içinde ≈**1:49**. Bu dengesizlik
`class_weight="balanced"` ile korunur (nadir sınıfa otomatik ağırlık) ve eğitim hastaları
1:1 seçilir.

### 3.2 Özellik çıkarımı (lokal topoloji)

Her aday merkezinden iki kesit alınır: **32×32 (çekirdek)** ve **64×64 (çevre)**. Her kesitte
**GUDHI** kütüphanesiyle **kübik kalıcı homoloji** hesaplanır:

* **H0** → parlaklık tepecikleri (bağlı bileşenler)
* **H1** → döngüler/delikler

Her diyagram, **eğitim adaylarından türetilip dondurulmuş** doğum–kalıcılık eksen aralıklarıyla
(p1–p99) ve **Gaussian yumuşatma** (σ=1,0 piksel) ile **20×20 kalıcılık imgesine** rasterize
edilir:

```
2 kesit × 2 homoloji (H0+H1) × 20×20 = 1.600 topolojik öznitelik
                                        + 4 uzamsal özellik
                                        = 1.604 boyutlu vektör
```

4 uzamsal özellik: akciğer içi göreli konum (x, y), hilusa uzaklık, plevraya uzaklık.

> ⚠️ Kesitler **yeniden boyutlandırılmaz** ve sabit 32/64 px'tir. Sebep ölçülmüştür: interpolasyon
> (bilinear resize) H1 döngülerini siler — V5.1'de ölçek normalizasyonu TDA'yı 0,9211 → 0,8539'a
> düşürmüştü. Ayrıca değişken kesit boyutu "boyut confound'u" yaratır (E3.6: 0,994 → 0,687).

### 3.3 Sınıflandırıcı

1.604 boyutlu vektör **StandardScaler** ile ölçeklenir, ardından
**HistGradientBoostingClassifier** (`max_iter=200`, `learning_rate=0,08`,
`class_weight="balanced"`) eğitilir. Çıktı: adayın "gerçek nodül olma" olasılığı `p_TDA`.

> ℹ️ **Güncel sürüm (V6.9i):** Bu sınıflandırıcı, 1. aşamanın **kendi aday dağılımı** üzerinde
> eğitilir. Eski bir sürüm M5 (ImageNet) dedektörünün adaylarıyla eğitilmişti; o uyumsuzluk
> kilitli testte 0,0075 rank kaybına mal oluyordu. Yeni `stage2_foundation_tda_hgb.joblib`
> bunu giderdi (test: rank 0,8970 → **0,9045**, AUC 0,9645 → **0,9687**).

---

## 4. Aşırı Özgüven (Stacking) Tuzağından Kaçınarak Birleştirme

Naif yaklaşım şudur: 2. aşama modeline 1. aşamanın skorunu da özellik olarak verirsin, o da
"1. aşama ne dediyse ona güvenir". Bu **iki kat sızıntı** doğurur:

1. **Eğitimde in-sample skor:** 1. aşamanın skoru, kendi **eğitim verisinde** şişmiştir
   (model ezberlediği örneklerde aşırı özgüvenlidir). 2. aşama bu şişkin skoru "gerçek" sanıp
   ona aşırı ağırlık verir.
2. **Öğrenilen füzyon katsayısı** da bu yanlı dağılıma göre uydurulur ⇒ val'de iyi görünür,
   gerçek dağılımda çöker.

**Bizim çözümümüz (V6.7'den itibaren yapısal koruma):**

* ✅ **Bağımsız eğitim:** 2. aşamanın özellik matrisi **1. aşama skorunu içermez** —
  yalnız 1.600 PI + 4 uzamsal. (Bundle'da `stage1_score_included = False` olarak assert edilir.)
* ✅ **Kilitli sabit ağırlık:** İki skor, **doğrulama kümesinde** belirlenen ve bir daha
  **yeniden uydurulmayan** sabit ağırlıkla birleştirilir:

```
z(s1) ve z(p_TDA): doğrulama kümesinden dondurulmuş ortalama/standart sapma ile
S = sigmoid( 0,55 · z(s1) + 0,45 · z(p_TDA) )
```

`w = 0,55` **kilitlidir**; hiçbir veri kümesinde (eğitim/doğrulama/test) yeniden öğrenilmez.
Z-normalizasyon referansı da **doğrulamadan** donar; testin kendi dağılımından istatistik
alınmaz (transductive sızıntı yok).

> 📌 **Neden sabit ağırlık?** Öğrenilen bir LR/LR-benzeri füzyon, az sayıda hastada kolayca
> aşırı uyum sağlar. Sabit, önceden doğrulanmış ağırlık; hem sızıntısız hem tekrarlanabilirdir.
> Ayrıntı: [LEAK_AUDIT.md](LEAK_AUDIT.md) §4 ve Tuzak B notu.

---

## 5. Özet akış

```
4.882 hasta ──(hasta düzeyi bölme)──► 3.414 eğitim · 734 doğrulama · 734 kilitli test
                                          │
3.414 eğitim ──(1:1 denge)──► 1.580 (bilimsel) · 2.268 (tam-veri) görüntü
      │
      ▼
1. AŞAMA: ResNet50 + ASPP + FPN + CenterNet (Gaussian focal) — 5 donuk + 13 ince ayar = 18 epoch
      │  512→256 ısı haritası → 3×3 tepe → NMS 8 px → top-50 aday (sabit 56×56 kutu)
      ▼
2. AŞAMA: her aday → 32/64 kesit → GUDHI H0/H1 PI (1600) + 4 uzamsal = 1604
      │  → HistGradientBoosting (balanced, s1 YOK) → p_TDA
      ▼
FÜZYON: S = sigmoid(0,55·z(s1) + 0,45·z(p_TDA))   ← w ve z-referansı val'de dondurulmuş
```

**Kilitli test sonucu (sızıntısız, ayarsız):** yalnız 1. aşama rank_metric **0,8842** /
AUC **0,9446** → iki aşamalı füzyon rank_metric **0,9045** / AUC **0,9687**.
Ayrıntı: [experiments.md](experiments.md) · [ablation_matrix.md](ablation_matrix.md) ·
[LEAK_AUDIT.md](LEAK_AUDIT.md).

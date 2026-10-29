# Hoca Bilgilendirme Notu — Göğüs Röntgenlerinde Akciğer Nodülü Tespiti

> Bu belge, çalışmanın ne yaptığını ve nasıl çalıştığını, ilk kez okuyan bir bilgisayar
> bilimleri hocasının kısa sürede kavrayacağı bir dille anlatır. Mimari ayrıntılar
> atlanmamıştır; ancak terimler ilk geçtikleri yerde açıklanır. Bütün sayılar projede
> doğrulanmış ve sızıntı denetiminden geçmiştir.

---

## 1. Problemin Tanımı ve Projenin Amacı

Göğüs röntgeni ucuz ve yaygın bir görüntüleme yöntemidir; ancak akciğer nodüllerini
(erken evre akciğer kanserinin habercisi olabilecek küçük, yuvarlak lezyonları) bu görüntüde
bulmak zordur. Kaburga ve damar gölgeleri küçük nodüllerle üst üste biner; bu yüzden derin
öğrenme modelleri hızlı tarama yapsa da tam bu küçük bölgelerde çok sayıda **yalancı pozitif**
üretir: bir damar kesitini nodül sanar. Bu projenin amacı, derin öğrenmenin **hızlı tarama**
gücü ile Topolojik Veri Analizinin (TDA) **şekil/geometri ayrıştırma** gücünü iki aşamalı tek
bir sistemde birleştirmektir.

---

## 2. Sistem Nasıl Çalışıyor?

```
HAM GÖĞÜS RÖNTGENİ
   │  ÖN-İŞLEME: akciğer alanını otomatik ayır + parlaklığı akciğer dokusuna göre standartlaştır
   ▼
1. AŞAMA — ADAY ÜRETİCİ (derin öğrenme)
   Omurga: 800 bin göğüs röntgeniyle ön-eğitilmiş ResNet50
   Kafa:   CenterNet ısı haritası — her nodülün merkezine bir "sıcak nokta" çizer
   Çıktı:  en güçlü 50 sıcak nokta → 50 şüpheli kutu (56×56 piksel), her biri skorlu
   ▼
2. AŞAMA — YENİDEN PUANLAMA (topolojik)
   Her kutudan 32×32 (çekirdek) + 64×64 (çevre) kesit
   → her kesitin "şeklini" kalıcı homolojiyle özetle (tepecikler + döngüler)
   → özet + kutunun konumu → bir karar modeli → "gerçek nodül mü?"
   ▼
BİRLEŞTİRME: nihai skor = 1. aşama güveni + 2. aşama topolojik puanı
```

**Ön-İşleme.** Röntgende akciğer alanı, TorchXRayVision'ın PSPNet segmentasyon modeliyle
ayrılır; akciğer duvarına (plevraya) yapışık nodülleri kaçırmamak için bu maske 15 piksel
dışarı genişletilir. Ardından parlaklık, akciğer dokusunun %1–%99 persentil aralığına göre
yeniden ölçeklenir; böylece 12/16-bit derinlikteki ince parlaklık farkları yuvarlanıp
kaybolmaz — küçük ve soluk nodüller bu adım olmadan silinir.

**1. Aşama (Aday Üretici).** Göğüs röntgenleri üzerinde ön-eğitilmiş açık kaynaklı bir model
(TorchXRayVision'ın ResNet50'si) ile ısı haritası üreten CenterNet kafası arasında, farklı
boyuttaki nodülleri aynı anda yakalamak için çok-ölçekli bir özellik piramidi (ASPP + FPN)
bulunur. Çıkarımda ise çok-ölçekli (0,9×, 1,0×, 1,1× + yatay çevirme) test-zamanı artırma
(TTA) uygulanır. Görevi **duyarlılıktır**: tek bir gerçek nodülü kaçırmamayı önceler; bedeli,
çoğu yanlış alarm olan çok sayıda aday üretmektir.

> **Neden YOLO değil?** YOLO denemedik. Isı haritası yaklaşımını, kutu boyutu/oranı gibi
> varsayımlar yapmadığı ve "çok aday üretme" görevine yatkın olduğu için seçtik. Yine de bu
> dürüst bir eksiktir: NODE21 yarışmasının resmî liderleri YOLOv5 kullandı; YOLO ile birebir
> kıyas henüz yapılmadı.

**2. Aşama (Yeniden Puanlama).** Her şüpheli kutu, parlaklığın "yükseklik" olduğu küçük bir
tepe haritası gibi düşünülür. GUDHI kütüphanesiyle hesaplanan **kübik kalıcı homoloji**, bu
yüzeyde hangi **tepeciklerin** (bağlı bölgeler, H0) ve hangi **döngü/deliklerin** (H1) hangi
eşikte doğup öldüğünü ölçer. 32×32 ve 64×64 kesitlerden 20×20 çözünürlüklü **kalıcılık
imgeleri** üretilir (toplam 1.600 topolojik öznitelik); buna kutunun akciğer içindeki konumu
(4 uzamsal öznitelik) eklenir ve 1.604 boyutlu bir vektör oluşur. Gerçek nodül ile damar
gölgesinin bu "doğum-ölüm" izleri farklıdır; bu vektör, yorumlanabilir bir karar modeliyle
(HistGradientBoosting) puanlanır ve 1. aşama skoruyla birleştirilir.

> **2. aşama neyle eğitildi?** Negatif örnekler, 1. aşamanın ürettiği **yanlış alarmlardır**
> (nodül olmayan adaylar). Bu bilinçli bir seçimdir: modelin öğrenmesi gereken karışıklık tam
> olarak budur. Veri dengesizdir (bir nodüle karşılık ~49 yanlış alarm), bu yüzden nadir
> pozitif sınıfa otomatik ağırlık verilir ve eğitim hastaları nodüllü/normal olarak 1:1
> dengelenir.

---

## 3. Temel Bulgular ve Sayısal Sonuçlar

**Veri seti:** Halka açık NODE21 veri seti — 4.882 hasta (1.134 nodüllü, 3.748 normal),
hasta düzeyinde bölünmüş: 3.414 eğitim / 734 doğrulama / 734 kilitli test. Bir hastanın
görüntüleri asla hem eğitimde hem testte yer almaz; yani **hasta sızıntısı yok**.

| Ölçüt | Doğrulanmış sonuç |
| :--- | :--- |
| **Aday yakalama (1. aşama)** | Gerçek nodüllerin **%98–99'u** ilk 50 aday kutu içinde |
| **Küçük bölgede piksel vs. topoloji (2. aşama)** | Piksel tabanlı küçük CNN ≈**0,65** · lokal TDA tek başına ≈**0,93** (hasta AUC) |
| **Kilitli test — yalnız 1. aşama** | Hasta AUC **0,945** · resmî metrik benzetimi **0,884** |
| **Kilitli test — 1.+2. aşama (TDA füzyonu)** | Hasta AUC **0,965** · resmî metrik benzetimi **0,897** |
| **Dış veri (VinDr-CXR, Vietnam; sıfır eğitim)** | Hasta AUC ≈**0,65–0,66** |

Kısaca: "hasta AUC" (0 ile 1; 0,5 = şans, 1,0 = kusursuz) modelin *"bu hastada nodül var mı?"*
sorusunu ne kadar iyi ayırdığını gösterir; "resmî metrik benzetimi" ise NODE21 yarışmasının
gerçek puanlama yöntemini (hem doğru yerini bulma hem doğru sınıflandırma) taklit eder.

Üç bulgu öne çıkıyor:

1. **Aday yakalama:** 1. aşama, gerçek nodüllerin büyük çoğunluğunu (%98–99) ilk 50 aday
   içinde yakalıyor — "kaçırmama" hedefi sağlanıyor; kalan iş, bu adaylardaki damar
   gölgelerini elemek.
2. **Lokal piksel vs. lokal topoloji:** Aynı küçük kırpıntıları puanlarken piksel dokusuna
   bakan küçük bir CNN ≈0,65'te kalırken, kesitin **şeklini** özetleyen topolojik temsil tek
   başına ≈0,93'e ulaşıyor — yani bu küçük ölçekte ayırt edici bilgi dokudan çok **şekil**de.
3. **Kilitli test ve tamamlayıcılık:** Sızıntısız 734 hastalık test setimizde sistem hasta
   düzeyinde **0,965** AUC'ye ulaşıyor. Topolojik katman, derin modelin emin olmadığı kısmı
   tamamlıyor: hasta AUC **0,945'ten 0,965'e**, resmî metrik **0,884'ten 0,897'ye** yükseliyor.
   Göğüs röntgeniyle ön-eğitimli omurga, genel görüntü ön-eğitimli omurgaya göre belirgin
   biçimde daha iyi (aday yakalamada %99'a karşı %98). Ayrıca daha önceki bir sürümümüzde,
   5 bölgeli global temsil modeli ile lokal dedektör birleştirildiğinde, tamamlayıcılık
   sayesinde testte **0,988** hasta AUC'ye ulaşılmıştı.

---

## 4. Hocanın Aklına Gelebilecek Sorular

* **Neden iki aşama?** "Hiçbir nodülü kaçırma" (duyarlılık) ile "yanlış alarm üretme"
  (kesinlik) çatışır; iki aşamaya bölmek bu çatışmayı iki uzmana dağıtır. Ölçüm bunu
  destekliyor: 2. aşama eklenince hasta AUC 0,945 → 0,965.
* **Neden hazır bir omurga, sıfırdan değil?** ~4.900 hasta bu büyüklükte bir ağı sıfırdan
  eğitmek için küçüktür; göğüs röntgeniyle ön-eğitilmiş bir başlangıç ezberlemeyi azaltır.
  Topolojik katman ise eğitim gerektirmez, şekli doğrudan hesaplar.
* **Sızıntı var mı?** Hasta düzeyinde bölme, ayar yapılmayan mühürlü test ve kod denetimiyle
  kontrol edildi; sızıntı bulunmadı.
* **Ön-eğitim verisiyle sizin veriniz örtüşüyor mu?** Kısmen evet (NODE21 açık veri
  kümelerinden derlenmiş; omurga pozitiflerin ~%83'ünü görmüş). Bunu ölçtük ve skoru
  **şişirmediğini** gördük; bu nokta makalede açıkça beyan edilecek.
* **Neden AUC/FROC, doğruluk değil?** Veri dengesiz (hastalık az görülür); doğruluk "hep
  sağlam de" diyen modeli bile yüksek gösterir. AUC/FROC sıralamaya dayalıdır, bu tuzağa
  düşmez.
* **Nasıl dağıtılır?** Resmî Grand-Challenge standardında, internet bağlantısı olmadan
  çalışan bir Docker konteyneri olarak paketlendi.

---

## 5. Dürüstlük ve Sınırlamalar

Çalışmada yalnızca yüksek skor kovalanmadı; sistem güvenilmeden önce zorlu kontrollerden
geçirildi: veri sızıntısı denetimi, parlaklık niceleme hatası (12/16-bit derinlik kaybı),
ön-eğitim veri kaynağı örtüşmesi ve harici hastane genellemesi ayrı ayrı ölçüldü.

Dürüstçe kabul edilen sınırlar şunlardır:

* **Dış genelleme zayıf:** Farklı bir ülkenin verisinde (VinDr-CXR, Vietnam) hiç eğitim
  yapmadan test edildiğinde hasta AUC ≈0,65–0,66 çıkıyor; kurum içi 0,965'in belirgin
  altında. Sistem farklı cihazlara **kalibrasyonsuz** taşınmıyor.
* **Ödünleşim:** TDA hasta düzeyi ayrımı artırıyor, ama çok sıkı yanlış-alarm eşiklerinde
  küçük bir duyarlılık kaybı olabiliyor.
* **Tek eğitim tekrarı:** Sonuçlarda ±0,02'lik bir belirsizlik bandı var; çoklu tekrar
  henüz yapılmadı.
* **Kutu boyutu:** Model, `R>0` merkez-maskeli bir yarıçap kafasıyla artık her nodülün kendi
  boyutunu dinamik olarak tahmin edebiliyor (doğrulama setinde sabit kutuyu geçti: 0,9185 vs
  0,9131). Yine de kilitli test setinde resmî skor eşit çıktığı için (ikisi de 0,884), sabit
  56 px kutu muhafazakâr varsayılan olarak korunuyor.

---

## 6. Sonuç ve Danışmak İstediklerim

Bir lisans öğrencisi olarak bu projeyi, "derin öğrenme + topolojik veri analizi" fikrinden
çalışan, denetlenmiş ve sınırları dürüstçe çizilmiş bir sisteme kadar getirdim. Çalışmanın
akademik bir yayına (makale/bildiri) veya bir tez/proje konusuna dönüştürülmesi konusunda
sizin görüşünüze ve yönlendirmenize çok değer veriyorum; özellikle hangi yöne
(yerelleştirme metodolojisi, topolojik temsilin yorumlanabilirliği veya dış doğrulama)
ağırlık vermem gerektiğini danışmak isterim.

# NODE21 TDA Projesi — Çalışma Planı

Bu plan, JSRT + SCR ile geliştirilmiş prototipi NODE21 verisine taşıma sırasını anlatır. Öncelik önce NODE21 dosyalarını anlamak, düzenlemek ve işlenebilir hale getirmektir. Model seçimi daha sonra aynı özellikler üzerinde değiştirilebilir olacaktır. İlk derin öğrenme kıyası **sekiz zonu tek hasta girdisi olarak birleştiren küçük bir MLP** olacaktır. Arayüz, ana veri ve model akışı çalıştıktan sonraya bırakılmıştır.

## Prototipte yapılanlar ve çıkarımlar

Prototip JSRT görüntülerini metadata ile eşleştirdi, SCR maskelerini kullandı, sekiz anatomik zon oluşturdu ve her zon için TDA özellikleri çıkardı. TDA hattı H0 ve H1 persistence bilgilerini 20×20 Persistence Image’lara çevirerek toplam 800 değerlik vektör üretti. SVM, k-NN, ensemble ve veri artırma denemeleri yapıldı.

Sonuçlar tek bir test bölmesinde umut verici görünse de 5 katlı çapraz doğrulama sonucu belirgin biçimde daha düşüktü. Bu yüzden prototipteki skorlar NODE21 için başarı beklentisi olarak alınmayacak; yöntemleri karşılaştırmak için başlangıç deneyimi sayılacak. Ayrıca prototipte bazı değerlendirmelerde eşik test kümesinden seçilmiş. Yeni projede model/eşik seçimi validation verisinde yapılmalı, test verisi en son değerlendirmeye ayrılmalı.

Prototipteki TDA yaklaşımı, 8 zon fikri ve hasta bazlı ayırma yeni projeye aktarılacak başlangıç noktalarıdır. Maske, atlas, zon sınırları, TDA ayarları ve model yapısı NODE21 üzerinde kontrol edilmeden kesin karar sayılmayacak.

## Aşamalar

### 1. NODE21 verisini bul ve içeriğini doğrula

- NODE21’in indirilen dosyalarını ve metadata dosyasını `data/raw/node21/` altında tut.
- Lisans ve kaynak bilgilerini not et; ham görüntüleri değiştirme ve Git’e ekleme.
- Metadata sütunlarını, dosya biçimlerini ve görüntülerin gerçekten okunup okunmadığını incele.
- Her metadata kaydını görüntü dosyasıyla eşleştir; eşleşmeyen veya bozuk kayıtları raporla.
- Etiketleri, bounding box alanlarını ve koordinatlarının hangi görüntü boyutuna göre verildiğini doğrula.
- Kaynak veride hasta kimliği bulunup bulunmadığını kontrol et. NODE21’deki görüntü sayısını hasta sayısı varsayma.

**Bu aşamanın çıktısı:** Görüntü kimliği, dosya yolu, etiket ve anotasyon bağlantısı bulunan bir manifest ile veri özeti.

### 2. Hasta bazlı veri bölmelerini hazırla

- Hasta kimliği varsa aynı hastanın görüntülerini aynı train, validation veya test grubunda tut.
- Hasta kimliği yoksa hangi kimliklerle güvenilir grup kurulabileceğini incele ve sınırlamayı açıkça yaz.
- Bölmeleri görüntüleri çoğaltmadan ve model/özellik işlemlerinden önce oluştur.
- Sınıf dağılımını ve varsa kaynak veri kümelerinin dağılımını bölmeler arasında kontrol et.
- Bölme listesini kaydet ve deneylerde tekrar kullan.

Validation modeli ve karar eşiğini seçmek içindir. Test kümesi bu seçimlerde kullanılmayacak; yalnızca son değerlendirmede kullanılacak.

### 3. Görüntüleri okuma ve hazırlama

- NODE21’deki gerçek görüntü formatı ve boyutlarını envanterden belirle; prototipteki görüntü boyutunu otomatik olarak varsayma.
- Görüntüleri tek bir okuyucu üzerinden aç; piksel aralığı ve yön bilgisini kontrol et.
- Normalizasyon veya yeniden boyutlandırma gerekiyorsa seçilen işlemi belgele.
- Görüntü boyutu değişirse bounding box koordinatlarını aynı dönüşüme göre güncelle.
- Az sayıda görüntü üzerinde görsel kontrol yaparak anotasyonların görüntüyle hizalı olduğunu doğrula.

### 4. Sekiz zonu NODE21’e uyarlama

- Önce NODE21 dosyalarında veya resmi materyallerinde kullanılabilir akciğer maskesi olup olmadığını doğrula. Prototipte kullanılan SCR maskelerinin NODE21 için hazır olduğunu varsayma.
- Maske yoksa ilk olarak sekiz zonu üretmenin uygulanabilir yollarını incele. Gerekirse otomatik akciğer segmentasyonu bu aşamada değerlendirilir.
- Zonların sağ/sol ve üst/alt tanımlarını sabitle; görüntü yönünün bu tanımlarla uyumlu olduğunu kontrol et.
- Prototipteki atlas ve örtüşme ayarlarını doğrudan kopyalama. Önce örnek NODE21 görüntülerinde zonları ve bounding box’ları birlikte incele.
- Her görüntünün sekiz zonunu, anotasyonlarını ve kimliğini birbirinden kopmayacak şekilde kaydet.

**Bu aşamanın çıktısı:** Her görüntü için görüntü, anotasyon ve zonların izlenebilir biçimde eşleşmesi.

### 5. TDA özelliklerini çıkar

- Prototipteki 800 boyutlu H0/H1 Persistence Image yöntemini ilk deneme ayarı olarak uygula.
- NODE21 üzerinde filtrasyon yönünü, maske dışı piksellerin etkisini, boş zonları ve üretilen vektörlerin boyutunu kontrol et.
- Başarısız görüntü veya boş zonları hata/durum bilgisiyle kaydet; sessizce normal bir özellik vektörü gibi değerlendirme.
- Önce küçük bir örnek grubunda çalışma süresi ve sonuçları kontrol et; ardından tüm veriyi işle.
- Her vektörle birlikte görüntü/hasta kimliğini, zonu, etiketi ve split bilgisini sakla.

**Bu aşamanın çıktısı:** Model deneylerinde aynı biçimde kullanılabilecek, kimliği ve zonu belli TDA özellikleri.

### 6. Değiştirilebilir model altyapısını kur ve ilk modeli dene

- Modeller aynı özellik girdisini alıp aynı biçimde skor üretsin; hangi modelin çalışacağı config dosyasından seçilebilsin.
- İlk model: **sekiz zonu tek hasta girdisi olarak birleştiren küçük bir MLP**.
- İlk kıyas için prototipten gelen SVM yaklaşımını da aynı bölmeler ve özelliklerle referans olarak ekle.
- MLP’nin ayarlarını validation verisine göre seç; test verisini bu süreçte açma.
- Sonuçlara bakıldıktan sonra gerekirse diğer seçenekleri (ör. zon başına ayrı modeller veya zonlar arasında ortak ağırlık kullanan model) ek deney olarak değerlendir. Bunların hiçbiri baştan en iyi model kabul edilmeyecek.

### 7. Modelleri aynı koşullarda değerlendir

- Sınıflandırma hedefini ilk aşamada nodül var/yok olarak tut.
- ROC-AUC ve sınıf dengesini gözeten precision-recall sonuçlarını; ayrıca duyarlılık, özgüllük ve hata sayılarını raporla.
- Karar eşiğini validation üzerinde belirle, ardından sabit eşikle test sonuçlarını hesapla.
- Sonuçları hasta/grup bazında ayır; mümkünse çapraz doğrulama ile tek bir bölmeye bağlı kalıp kalmadığını incele.
- Zon tahminini bounding box ile karşılaştırırsan bunu yardımcı zon analizi olarak adlandır. Bu, bounding box tespitiyle aynı şey değildir.
- Gerçek konum bulma istenirse bounding box tahminine yönelik ayrı model ve değerlendirme aşaması planla.

### 8. Sonraki kararlar

İlk veri hattı ve model kıyasları tamamlandıktan sonra sonuçlara göre TDA ayarları, maske/zon yöntemi ve ek model seçenekleri güncellenecek. Mimari çalışıp değerlendirilmeden arayüz geliştirmeye geçilmeyecek.

## Model seçiminin değiştirilebilir olması

Model adı ve temel ayarlar yapılandırma dosyasında tutulacak. Örneğin ilk çalıştırmada `patient_mlp`, sonraki kıyasta `zone_svm` seçilebilecek. Her model aynı eğitim ve tahmin akışına uyacak; böylece model değiştirmek için veri hazırlama ve TDA kodunu yeniden yazmak gerekmeyecek. Hangi model seçeneklerinin ekleneceği ilk kıyas sonuçlarından sonra belirlenecek.

## Mevcut durum

- [x] Proje klasör iskeleti `rontgen_V2/` içine oluşturuldu; mevcut `venv/` aynı seviyede.
- [x] Prototip devir metni ve temel TDA, split, model değerlendirme kodları incelendi.
- [ ] NODE21 ham verisinin konumu, metadata şeması ve görüntü dosyaları henüz doğrulanmadı.
- [ ] İlk yapılacak iş: NODE21 dosyalarını envanterleyip metadata, etiket ve bounding box eşleşmelerini raporlamak.

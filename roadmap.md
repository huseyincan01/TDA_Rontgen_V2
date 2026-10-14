# NODE21 TDA — Araştırma Yol Haritası ve Deney Yönetimi (v3 Programı)

> Bu belge, kilitli **v2.1 hattı** (bkz. [progress.md](progress.md)) üzerine kurulacak **v3 araştırma programının** planıdır: iş kolları, deney backlog'u, bağımlılıklar, öldürme kriterleri ve yönetim protokolü.
> Sonuç kaydı: [experiments.md](experiments.md) — her deney burada ID'siyle raporlanır.

**Amaç:** Yayınlanabilir bir katkı üretmek. Katkı "SOTA nodül deteksiyonu" değil; **topolojik temsilin tamamlayıcı + yorumlanabilir bilgi taşıdığını kanıtlamak** ve bunu sağlam bir ablasyon matrisiyle desteklemek.

---

## 0. Yönetim Protokolü (Değişmez Kurallar)

Bu 8 kural, "her fikri denemek" ile "sonuçların güvenilir olması" arasındaki dengeyi kurar.

1. **Tek değişken kuralı.** Her deney yalnızca **bir** faktörü değiştirir (ör. yalnız kemik bastırma, yalnız çok ölçek). İki faktörü aynı anda değiştiren deney yalnızca "kombinasyon" olarak etiketlenir ve tek başına kanıt sayılmaz.
2. **Dondurulmuş test seti.** Model/eşik/hiperparametre seçimi **yalnızca validation** üzerinde yapılır. Test setine (734 hasta) yalnızca **faz sonlarında** dokunulur.
3. **Test seti bütçesi.** Program boyunca en fazla ~6 test değerlendirmesi yapılır (her fazın en iyi 1–2 modeli). Gerekçe: test setini tekrar tekrar kullanmak yayın bütünlüğünü bozar.
4. **Ön-kayıt (pre-registration).** Her deney başlamadan önce hipotezi ve **başarı eşiğini** backlog tablosuna yazılır. Deney bitince yalnızca "tuttu/tutmadı" kararı verilir; eşik sonradan değiştirilemez.
5. **Kayıt protokolü.** Her deney şu alanlarla kaydedilir: `ID, iş kolu, hipotez, tek değişken, bağımlılık, config, git commit, val metriği, test metriği, karar (KEEP/KILL), not`.
6. **Anlamlılık eşiği.** Bir iyileşme "gerçek" sayılır ancak **val ROC-AUC farkı ≥ +0.015** ve **N=2000 bootstrap %95 CI kesişmiyorsa**. Bunun altındaki farklar gürültü kabul edilir (mevcut bootstrap altyapısı: [07_compute_bootstrap_ci.py](pipelines/07_compute_bootstrap_ci.py)).
7. **Öldürme kriteri.** Yüksek riskli her deney için önceden "hangi durumda bırakırım" tanımlanır (bkz. §6). Kriter tetiklenirse deney **KILL** işaretlenir ve zaman harcanmaz.
8. **Karar günlüğü.** Her hafta sonunda roadmap.md durum tablosu + experiments.md güncellenir; verilen KEEP/KILL kararlarının **gerekçesi** yazılır. Bu, yazım aşamasında "neden bunu seçtiniz?" sorularının cevabıdır.

### Anlamlı "kanıt" ne demek?

| Kanıt seviyesi | Gereksinim |
| :--- | :--- |
| **Zayıf** | Tek seed, val farkı < +0.015 → gürültü, kayda değmez |
| **Orta** | En az 3 seed ortalaması, val farkı ≥ +0.015 |
| **Güçlü (yayın)** | 3+ seed, val ≥ +0.015, bootstrap CI kesişmiyor, faz sonunda testte doğrulandı |

---

## 1. Hesaplama Bütçesi ve Darboğazlar

| Kaynak | Durum |
| :--- | :--- |
| GPU | Tek **NVIDIA RTX 4050** → eğitim işleri serileştirilmeli |
| Özellik çıkarımı | 24.410 zon ≈ **36 dk** (10 worker) → her yeni ön-işleme tam yeniden çıkarım gerektirir |
| Veri | 4.882 hasta, 24.410 zon, 800 boyut PI (78 MB) |
| Kilitli test | 734 hasta (172 nodül / 562 kontrol) |

**Yönetim sonucu:** Ön-işleme varyantları (kemik/damar bastırma vb.) **toplu** çıkarılır (gece boyu tek seferde tüm varyantlar), sonra sınıflandırıcı deneyleri hızlıca koşturulur. GPU'yu beklerken bağımsız iş kolları (ör. değerlendirme/analiz) ilerletilir.

---

## 2. İş Kolları

| Kod | İş Kolu | Amaç |
| :--- | :--- | :--- |
| **W0** | Altyapı & Referans | Deney kaydı, referans yeniden üretim, pretrainli omurga altyapısı |
| **W1** | Görüntü Ön-İşleme | Kemik/damar bastırma, normalizasyon, atlas kaydı → **temiz topolojik sinyal** |
| **W2** | Topoloji (TDA) | Çift yön, çok ölçek, arka plan çıkarma, ağırlıklı filtrasyon, OT vektörleştirme |
| **W3** | Mimari & Füzyon | Pretrainli CNN + TDA füzyonu, iki aşamalı candidate-based, ensemble |
| **W4** | Değerlendirme & Yorumlanabilirlik | Robustness, hata analizi, radyolojik eşleme |
| **W5** | Yayın | Ablasyon matrisi, figürler, yazım |

---

## 3. Fazlar ve Çıkış Kriterleri

| Faz | İçerik | Süre (tahmini) | Çıkış kriteri |
| :--- | :--- | :--- | :--- |
| **Faz 0 — Altyapı** | E0.1, E0.2, E0.3 | ~1 hafta | Registry çalışıyor · 0.8631 referansı yeniden üretildi · pretrainli omurga smoke test geçti |
| **Faz 1 — Hızlı kazanç** | E3.4, E4.1, E1.1, E1.2 | ~1–2 hafta | En az 1 KEEP (ensemble **veya** kemik bastırma val'de ≥ +0.015) |
| **Faz 2 — Füzyon (ana iddia)** | E3.1, E3.2 | ~1–2 hafta | TDA'nın güçlü CNN'e **tamamlayıcı** katkısı val'de CI ile kanıtlı |
| **Faz 3 — Topoloji derinleşmesi** | E2.1, E2.2, E2.3, E2.4 | ~2–3 hafta | En az 1 topolojik novelty KEEP |
| **Faz 4 — Yüksek risk** | E3.3, E2.5, E1.4, E2.6 | ~3–4 hafta | En az 1 deney başarılı **veya** gerekçeli KILL (öğrenilmiş) |
| **Faz 5 — Yayın** | E4.2, E4.3, E5.1, E5.2 | ~2 hafta | Ablasyon matrisi tam · taslak metin + figürler |

---

## 4. Deney Backlog'u

Durum: `Planlandı` → `Devam` → `KEEP` / `KILL` / `Parkta`

| ID | İş Kolu | Hipotez (ön-kayıtlı) | Bağımlılık | Efor | Risk | Beklenen Etki | Durum |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **E0.1** | W0 | Deney kayıt protokolü + dizin sözleşmesi kurulabilir | — | 0.5g | Düşük | Altyapı | Planlandı |
| **E0.2** | W0 | PI CNN (0.8631) bit-bit yeniden üretilebilir | — | 0.5g | Düşük | Referans | Planlandı |
| **E0.3** | W0 | Pretrainli omurga (ResNet/EfficientNet) zon kesitlerinde çalışır | — | 1g | Düşük | Altyapı | Planlandı |
| **E3.4** | W3 | CNN + SVM soft-voting tek modellerden ≥ +0.015 AUC verir | E0.2 | 0.5g | Düşük | Orta | Planlandı |
| **E4.1** | W4 | %10/%20/%30 overlap kurallarında model sıralaması değişmez | E0.2 | 1g | Düşük | Yayın kanıtı | Planlandı |
| **E1.1** | W1 | Kemik bastırma H1 kemik gürültüsünü azaltır, val AUC ≥ +0.015 artar | — | 2g | Orta | **Yüksek** | Planlandı |
| **E1.2** | W1 | Frangi damar bastırma H1 damar döngülerini temizler, AUC ≥ +0.015 artar | — | 2g | Orta | **Yüksek** | Planlandı |
| **E3.1** | W3 | Pretrainli CNN + TDA geç füzyon > her ikisi tek başına (CI ile) | E0.3 | 1.5g | Düşük | **Çok Yüksek** (ana iddia) | Planlandı |
| **E3.2** | W3 | TDA özellikleri CNN özellik vektörüne eklenince AUC artar | E0.3 | 1.5g | Orta | **Çok Yüksek** | Planlandı |
| **E2.1** | W2 | Çift yön (sublevel+superlevel) filtrasyon tek yönden ≥ +0.015 iyi | — | 1.5g | Düşük | Orta | Planlandı |
| **E2.2** | W2 | Çok ölçekli PI tek ölçekten ≥ +0.015 iyi | E2.1 | 2g | Orta | Orta-Yüksek | Planlandı |
| **E2.3** | W2 | Sağlıklı şablon çıkarma (relative persistence) AUC artırır | E0.2 | 2.5g | Yüksek | **Yüksek** (novelty) | Planlandı |
| **E2.4** | W2 | Aday-olasılık ağırlıklı filtrasyon H1'i nodüle odaklar | E1.2 | 2.5g | Yüksek | Orta | Planlandı |
| **E4.2** | W4 | Kalıcılık özellikleri radyolojik yapılara eşlenebilir | E0.2 | 2g | Orta | **Yüksek** (yayın) | Planlandı |
| **E4.3** | W4 | Hangi zonlarda sistematik hata yapıyoruz? | E0.2 | 1g | Düşük | Orta | Planlandı |
| **E3.3** | W3 | İki aşamalı candidate-based TDA yanlış pozitifi azaltır | E1.2 | 4g | Yüksek | Yüksek | Planlandı |
| **E2.5** | W2 | Sliced Wasserstein / PersLay, PI'dan ≥ +0.015 iyi | E0.2 | 3g | Yüksek | Orta | Planlandı |
| **E1.4** | W1 | Atlas kaydı arka plan topolojisini tutarlılaştırır | E1.1 | 4g | Yüksek | Orta | Planlandı |
| **E2.6** | W2 | Çok parametreli persistence ek bilgi taşır | E2.2 | 5g+ | Çok Yüksek | Belirsiz | **Parkta** |
| **E5.1** | W5 | Tüm ablasyonlar tek matriste tutarlı sunulabilir | Faz 1–4 | 2g | Düşük | Yayın | Planlandı |
| **E5.2** | W5 | Metin + figürler hakem beklentilerini karşılar | E5.1 | 4g | Orta | Yayın | Planlandı |

---

## 5. Bağımlılık Grafiği

```
                        ┌──────────────────────────────────────────┐
                        │  FAZ 0 — ALTYAPI (paralel koşabilir)     │
                        │  E0.1 registry · E0.2 referans · E0.3 omurga │
                        └───────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              v                         v                          v
   ┌────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐
   │ W1 GÖRÜNTÜ         │   │ W3 FÜZYON (ana iddia)  │   │ W4 DEĞERLENDİRME │
   │ E1.1 kemik bastır  │   │ E3.1 geç füzyon        │   │ E3.4 ensemble    │
   │ E1.2 damar bastır  │   │ E3.2 orta füzyon       │   │ E4.1 robustness  │
   └─────────┬──────────┘   └────────────────────────┘   │ E4.3 hata analizi│
             │                                           └──────────────────┘
             v
   ┌────────────────────────────────────────┐
   │ W2 TOPOLOJİ (temiz sinyal üzerine)     │
   │ E2.1 çift yön → E2.2 çok ölçek         │
   │ E2.3 arka plan çıkarma (novelty)       │
   │ E2.4 ağırlıklı filtrasyon ← E1.2       │
   └─────────┬──────────────────────────────┘
             v
   ┌────────────────────────────────────────┐
   │ W5 / FAZ 5 — E4.2 yorumlanabilirlik    │
   │ E5.1 ablasyon matrisi → E5.2 yazım     │
   └────────────────────────────────────────┘
```

**Kritik yol:** `E0.3 (omurga) → E3.1 (füzyon)`. Bu, yayının ana iddiasını taşır; öncelik burada.

---

## 6. Öldürme Kriterleri (Kill Criteria)

| ID | Bırakma koşulu |
| :--- | :--- |
| **E1.1 / E1.2** | Yeniden çıkarım sonrası val ROC-AUC artışı < +0.015 **veya** H1 doluluk oranı değişmiyorsa → KILL (36 dk extraction maliyeti haklı değil) |
| **E2.2** | Çok ölçek, tek ölçeğe val'de < +0.015 → KILL |
| **E2.3** | Şablon çıkarma val'de < +0.015 **veya** şablon kontrollerde kararsızsa (yüksek varyans) → KILL |
| **E2.4** | Aday haritası zaten AUC'yi domine ediyorsa (yani TDA katkısı marjinal) → KILL |
| **E2.5** | 3 gün içinde çalışan bir eğitim prototipi yoksa **veya** val < +0.015 → KILL |
| **E2.6** | Hesaplama > 1 saat/zon **veya** 2 haftada prototip yok → Parkta kalır (KILL) |
| **E3.3** | Aday üretici recall < %80 ise (tavan düşük) → KILL |
| **E1.4** | Kayıt sonrası zon hizalaması görsel olarak bozuluyorsa → KILL |

---

## 7. Dizin ve Adlandırma Sözleşmesi

Çok sayıda deneyi karıştırmamak için tek bir sözleşme uygulanır:

```
configs/experiments/<ID>_<slug>.yaml          # deney konfigürasyonu (tek değişken burada)
pipelines/exp/<ID>_<slug>.py                  # deneye özgü betik (gerekirse)
artifacts/runs/<ID>_<slug>/                   # çıktılar (gitignored — kalıcı kayıt experiments.md'de)
    ├── manifest.json                         # git commit + config hash + seed + bağımlılıklar
    ├── val_report.json
    └── test_report.json                      # YALNIZCA faz sonunda
data/processed/features_v3_<slug>/            # yeni ön-işleme varyantlarının özellikleri
```

> **Not:** `artifacts/` ve `data/processed/` git tarafından yok sayılır. Bu yüzden **kalıcı kanıt kaydı `experiments.md` ve bu belgedeki durum tablosudur.** Manifest'ler yerelde tutulur; kritik sayısal sonuçlar mutlaka `experiments.md`'ye yazılır.

---

## 8. Haftalık Ritim

| Gün | İş |
| :--- | :--- |
| Pazartesi | Haftanın deneyini seç, hipotezi + başarı eşiğini backlog'a **ön-kaydet** |
| Salı–Perşembe | Deneyi koştur; ön-işleme varyantlarını gece toplu çıkar |
| Cuma | Sonuçları `experiments.md`'ye yaz, KEEP/KILL kararı ver, roadmap durum tablosunu güncelle |
| Faz sonu | En iyi modeli **bir kez** test setinde değerlendir (test bütçesi §0.3) |

---

## 9. Durum Tablosu (Canlı)

| Faz | Deneyler | Tamamlanan | Aktif | KEEP | KILL |
| :--- | :--- | :---: | :--- | :---: | :---: |
| Faz 0 | E0.1, E0.2, E0.3 | 0/3 | — | 0 | 0 |
| Faz 1 | E3.4, E4.1, E1.1, E1.2 | 0/4 | — | 0 | 0 |
| Faz 2 | E3.1, E3.2 | 0/2 | — | 0 | 0 |
| Faz 3 | E2.1, E2.2, E2.3, E2.4 | 0/4 | — | 0 | 0 |
| Faz 4 | E3.3, E2.5, E1.4, E2.6 | 0/4 | — | 0 | 0 |
| Faz 5 | E4.2, E4.3, E5.1, E5.2 | 0/4 | — | 0 | 0 |

**Referans noktası (v2.1, kilitli):** PI CNN test ROC-AUC **0.8631** · RBF-SVM **0.8289** · k-NN **0.7162** · Ham CNN **0.7043** · PL CNN **0.5450** · PL SVM **0.5217**

---

## 10. Karar Günlüğü

| Tarih | ID | Karar | Gerekçe |
| :--- | :--- | :--- | :--- |
| — | — | — | Henüz karar yok |

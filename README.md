# node21_tda

NODE21 göğüs röntgenlerinde nodül tespiti üzerine bir araştırma projesi: hasta bazında
veri hazırlama, **kübik kalıcı homoloji (TDA)** özellikleri ve derin öğrenme dedektörleriyle
**iki aşamalı lokalizasyon**, ve resmî NODE21 detection formatına uyan bir
**Grand-Challenge gönderim konteyneri**.

> **Hikâyenin tamamı tek sayfada: [STORY.md](STORY.md)** — v1'den V6.7'ye, her dönüm
> noktasında "ne sandığımız / ne çıktığı".

---

## Belgeler — hangisi neyi barındırıyor

| Belge | İçerik | Ne zaman okunur |
| :--- | :--- | :--- |
| **[STORY.md](STORY.md)** | **Anlatı**: v1 → v2 → v3 → v4 → V5.1 → V6 → V6.6 → V6.7, dönüm noktaları ve dürüst sınırlar | **İlk buradan başla** |
| [progress.md](progress.md) | Veri hattının kuruluşu: v1'in 5 metodolojik hatası, v2 temiz hat (≥%20 kuralı, hasta düzeyi split, train'den türetilen TDA aralıkları, 1:1 dengeleme, 7 denetim) + v2.1/v3 sonuçları · **§8: V6.9 sonrası hangi ifadeler hâlâ geçerli** | Veri nasıl kuruldu |
| [roadmap.md](roadmap.md) | **v3** protokolü: 15 yönetim kuralı, test bütçesi, faz kapıları, kill kriterleri, karar günlüğü, §13 kapanış · **§14: bütçe muhasebesi (2 mühür) + yöntemsel ders** | Yöntem disiplini |
| [v4_roadmap.md](v4_roadmap.md) | **v4** programı: V1.1/V1.2, V2.x, V3.x, A2, V4/V5 kararları ve gerekçeleri · **§7: dış doğrulama hükmünün V6.9 düzeltmesi** | Tamamlayıcılık arayışı |
| [experiments.md](experiments.md) | **Tüm deneyler, kronolojik** (§1–§13): v2.1 benchmark → v3 (E0–E5) → VinDr harici → V5.x → V6 → V6.6 → V6.7 → V6.8 → V6.9 mührü | Belirli bir sayının kaynağı |
| [ablation_matrix.md](ablation_matrix.md) | **Makale tabloları**: liderlik, 5 ablasyon, istatistik, V6 (§13), V6.6 (§14), V6.7 (§15), V6.8 (§16), V6.9 (§17) | Yayın sayıları |
| **[LEAK_AUDIT.md](LEAK_AUDIT.md)** | **Sızıntı ve geçerlilik denetimi**: temiz çıkan 5 kontrol, VinDr kontaminasyonu (%100 örtüşme), val seçim yanlılığı, mühürlü test onayı, yayın öncesi 6 düzeltme | **Yayına girmeden önce zorunlu** |
| [paper_draft.md](paper_draft.md) | Makale taslağı (v3 iskeleti + §4.8 harici + §4.10 V6 + **§4.10 V6.9 mühür tablosu**, §6'da 3 yeni sınırlama) | Yazım |
| [literature.md](literature.md) | Literatür taraması, yenilik konumlandırması, hakem riski · **§5.6: denetimin doğurduğu 3 yeni hakem itirazı ve cevapları** | Konumlandırma |
| **[submission_docker/README.md](submission_docker/README.md)** | Gönderim konteyneri: boru hattı, resmî baseline uyumluluğu, `--network=none` | Gönderim/dağıtım |

**Okuma sırası (yeni okuyucu):** `STORY.md` → `progress.md` (veri) → `roadmap.md`
(protokol) → `ablation_matrix.md` (nihai sayılar) → `experiments.md` (ayrıntı).

---

## Mevcut durum (kısa)

| | |
| :--- | :--- |
| **Dürüst (mühürlü test) sonuç** | Hasta AUC **0,9645** (füzyon) / **0,9446** (1. aşama) · rank_metric **0,8970** / **0,8842** (V6.9, ön-kayıtlı tek atım) |
| **Val'de ölçülen (seçilmiş) sonuç** | Füzyon AUC **0,9879** · rank_metric **0,9131** (56 px) · **0,9185** (onarılmış yarıçap, V6.8) — val'de 296 değerlendirme yapıldı ⇒ **hafif iyimser** |
| **Lokalizasyon (V6.6, val)** | Yalnız Aşama-1: Recall@K20/K30 **0,9827**, K50 **0,9913** · FROC@0,5 **0,831**, @4 **0,952** |
| **Harici (VinDr-CXR)** | ✅ **Temiz omurgayla** (`densenet121-res224-nih`, VinDr'yi hiç görmemiş): **0,6663 AUC (nOR)** / 0,6369 (top1) · kontaminasyonun etkisi **ölçüldü ve bulunamadı** ([LEAK_AUDIT.md](LEAK_AUDIT.md) §8) |
| **Gönderim modeli** | 4.882 hastanın tamamıyla eğitildi → **NODE21'de değerlendirilemez** |
| **Konteyner** | `submission_docker/` · çevrimdışı (`--network=none`) test edildi · non-root · CUDA 13,9 GB / CPU 3,51 GB |

---

## Değişmez kurallar

1. **Kural 9 — leaderboard karşılaştırması yok.** Verimiz NODE21'in **halka açık eğitim
   setidir** (4.882 hasta); resmî test setleri (281 + 298) **özeldir**. Kendi iç
   bölmemiz resmî test değildir ⇒ sıralama karşılaştırması **geçersizdir**.
2. **Test bütçesi: 2/2 kullanıldı ve kapandı.** İlki V6.4 mührü (0,9876), ikincisi V6.9
   mührü (0,8842 / 0,9645) — ikisi de **ayarsız, tek atım**. Üçüncü bir "mühür" mühür
   sayılmaz; yeni bir test sonucu ancak yeni bir veri kümesiyle üretilebilir.
3. **Bilimsel checkpoint'ler korunur.** `M5_centernet.pt`, `M6_fcos.pt`,
   `M5_foundation.pt`, `stage2_final_tda_hgb.joblib` SHA-256 ile teyitlidir; tam-veri
   gönderim modelleri ayrı dizinde (`submission_fulldata/`) tutulur.
4. **Sayı çelişkisi:** öncelik sırası → `ablation_matrix.md` › `roadmap.md §13` ›
   `experiments.md`. En güncel çalışma `exp/v4-directional-radial` dalındadır.
5. **"Seçilmiş" ≠ "dürüst".** Val'de alınan sayılar **seçilmiş**tir (296 değerlendirme +
   19 w noktası aynı bölmede); **dürüst** dış-değer yalnız mühürlü test koşusundan gelir.
   Her tablo hangisi olduğunu **yazmak zorundadır**. Ayrıntı: [LEAK_AUDIT.md](LEAK_AUDIT.md).
6. **Dış doğrulama iddiası, omuranın ön-eğitim listesiyle çapraz kontrol edilmeden
   yazılmaz.** VinDr dersi: 400/400 örtüşme ancak denetimle ortaya çıktı.

---

## Dallar

| Dal | Anlamı |
| :--- | :--- |
| `main` | **v3 — kapandı, yayına hazır** (`v3-publication-ready`) |
| `exp/v4-directional-radial` | **aktif dal** — v4'ten başlayıp **V6.9'a** kadar tüm çalışma (V5 fiziksel kanonikleştirme, V6 iki aşamalı lokalizasyon, V6.6 CXR vakıf omurgası, V6.7 tam veri + Docker, V6.8 kanıt boşlukları, V6.9 mühürlü test onayı + sızıntı denetimi) |

---

## Depo yapısı (özet)

```
src/node21_tda/            veri hattı, TDA çıkarıcı, segmenter, modeller
pipelines/                 v2 hattı (01–07) + exp/ (E*, V*) + v6_two_stage/ (V6.0–V6.9)
artifacts/v6_two_stage/    V6 raporları (JSON) + submission_config.json
artifacts/v6_two_stage/test_seal_preregistration.json   V6.9 ön-kaydı (hash kilitli)
artifacts/v6_two_stage/submission_fulldata/   tam-veri gönderim modelleri (git dışı)
submission_docker/         process.py · Dockerfile · Dockerfile.cpu · smoke_test.py · weights/
LEAK_AUDIT.md              sızıntı/geçerlilik denetimi (yayın öncesi zorunlu okuma)
```

**Git dışı tutulanlar:** model checkpoint'leri, gömülü xrv ağırlıkları (~350 MB),
türetilmiş önbellekler (`*.npz`), `scratch/`.

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
| [progress.md](progress.md) | Veri hattının kuruluşu: v1'in 5 metodolojik hatası, v2 temiz hat (≥%20 kuralı, hasta düzeyi split, train'den türetilen TDA aralıkları, 1:1 dengeleme, 7 denetim) + v2.1/v3 sonuçları | Veri nasıl kuruldu |
| [roadmap.md](roadmap.md) | **v3** protokolü: 15 yönetim kuralı, test bütçesi, faz kapıları, kill kriterleri, karar günlüğü, §13 kapanış | Yöntem disiplini |
| [v4_roadmap.md](v4_roadmap.md) | **v4** programı: V1.1/V1.2, V2.x, V3.x, A2, V4/V5 kararları ve gerekçeleri | Tamamlayıcılık arayışı |
| [experiments.md](experiments.md) | **Tüm deneyler, kronolojik** (§1–§11): v2.1 benchmark → v3 (E0–E5) → VinDr harici → V5.x → V6 → V6.6 → V6.7 | Belirli bir sayının kaynağı |
| [ablation_matrix.md](ablation_matrix.md) | **Makale tabloları**: liderlik, 5 ablasyon, istatistik, V6 (§13), V6.6 (§14), V6.7 (§15) | Yayın sayıları |
| [paper_draft.md](paper_draft.md) | Makale taslağı (v3 iskeleti + §4.8 harici + §4.10 V6) | Yazım |
| [literature.md](literature.md) | Literatür taraması, yenilik konumlandırması, hakem riski | Konumlandırma |
| **[submission_docker/README.md](submission_docker/README.md)** | Gönderim konteyneri: boru hattı, resmî baseline uyumluluğu, `--network=none` | Gönderim/dağıtım |

**Okuma sırası (yeni okuyucu):** `STORY.md` → `progress.md` (veri) → `roadmap.md`
(protokol) → `ablation_matrix.md` (nihai sayılar) → `experiments.md` (ayrıntı).

---

## Mevcut durum (kısa)

| | |
| :--- | :--- |
| **En iyi kurum içi sonuç** | Füzyon test ROC-AUC **0,9516** (v3 headline; V6.6 uçtan uca val AUC **0,988** · **sızıntısız** sabit-ağırlık eşdeğeri **0,9886**) |
| **Lokalizasyon (V6.6)** | Yalnız Aşama-1: Recall@K20/K30 **0,9827**, K50 **0,9913** · FROC@0,5 **0,831**, @4 **0,952** |
| **Resmî metrik simülasyonu** | `rank_metric = 0,75·AUC + 0,25·Sens@0,25` → **0,9131** (IoU>0,2, 56 px sabit kutu) · **onarılmış yarıçap kafasıyla 0,9185** (V6.8) |
| **Harici (VinDr-CXR)** | Saf zero-shot en iyi **0,6469**; 160 hedef etiketle few-shot **0,7344** |
| **Gönderim modeli** | 4.882 hastanın tamamıyla eğitildi → **NODE21'de değerlendirilemez** |
| **Konteyner** | `submission_docker/` · çevrimdışı (`--network=none`) test edildi · non-root · CUDA 13,9 GB / CPU 3,51 GB |

---

## Değişmez kurallar

1. **Kural 9 — leaderboard karşılaştırması yok.** Verimiz NODE21'in **halka açık eğitim
   setidir** (4.882 hasta); resmî test setleri (281 + 298) **özeldir**. Kendi iç
   bölmemiz resmî test değildir ⇒ sıralama karşılaştırması **geçersizdir**.
2. **Test bütçesi 6/6 tükendi** (v3 için kapalı). V6/V6.6/V6.7 sayıları **val düzeyindedir**;
   V6'nın tek seferlik test mührü (0,9876) ayrı ve kapalıdır.
3. **Bilimsel checkpoint'ler korunur.** `M5_centernet.pt`, `M6_fcos.pt`,
   `M5_foundation.pt`, `stage2_final_tda_hgb.joblib` SHA-256 ile teyitlidir; tam-veri
   gönderim modelleri ayrı dizinde (`submission_fulldata/`) tutulur.
4. **Sayı çelişkisi:** öncelik sırası → `ablation_matrix.md` › `roadmap.md §13` ›
   `experiments.md`. En güncel çalışma `exp/v4-directional-radial` dalındadır.

---

## Dallar

| Dal | Anlamı |
| :--- | :--- |
| `main` | **v3 — kapandı, yayına hazır** (`v3-publication-ready`) |
| `exp/v4-directional-radial` | **aktif dal** — v4'ten başlayıp **V6.8'e** kadar tüm çalışma (V5 fiziksel kanonikleştirme, V6 iki aşamalı lokalizasyon, V6.6 CXR vakıf omurgası, V6.7 tam veri + Docker, V6.8 kanıt boşluklarının kapatılması) |

---

## Depo yapısı (özet)

```
src/node21_tda/            veri hattı, TDA çıkarıcı, segmenter, modeller
pipelines/                 v2 hattı (01–07) + exp/ (E*, V*) + v6_two_stage/ (V6.0–V6.7)
artifacts/v6_two_stage/    V6 raporları (JSON) + submission_config.json
artifacts/v6_two_stage/submission_fulldata/   tam-veri gönderim modelleri (git dışı)
submission_docker/         process.py · Dockerfile · Dockerfile.cpu · smoke_test.py · weights/
```

**Git dışı tutulanlar:** model checkpoint'leri, gömülü xrv ağırlıkları (~350 MB),
türetilmiş önbellekler (`*.npz`), `scratch/`.

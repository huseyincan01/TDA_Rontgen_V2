# VinDr-CXR Harici Doğrulama Verisi (V4)

> 🔴 **ÖN-EĞİTİM ÇAKIŞMASI UYARISI (V6.9 denetimi).** Bu kohortun **400/400 görüntüsü
> (%100)**, torchxrayvision `resnet50-res512-all` ağırlığının ön-eğitim listesinde
> (`vinbigdata-train.csv.gz`, 15.000 görüntü) bulunmaktadır. Etiketlerimizi aldığımız
> CSV, xrv'nin VinDr sınıfının varsayılan CSV'sinin **aynısıdır** (bkz.
> `v6_5d_strict_no_adabn.py:178`). ⇒ Bu veri kümesiyle yapılan değerlendirmeler
> **VinDr'yi görmemiş bir omurga** ile yapılmalıdır. Etkisi ölçüldü:
> `densenet121-res224-nih` (temiz) **0,6663** nOR / 0,6369 top1 ≈ kontamine omurgalar
> (0,6414 / 0,6479) ⇒ **şişme yok**, ama "zero-shot" etiketi kontamine satırlar için
> kullanılamaz. Ayrıntı: [LEAK_AUDIT.md](../../LEAK_AUDIT.md) §3 ve §8.
>
> ✅ Temiz sayılan kullanımlar: v3/v4/V5.3 ve V6.5d — bunlar **ImageNet** tabanlı
> `M5_centernet` / ResNet18 kullanır, xrv ağırlığı yoktur.

Bu klasör, v3 (E0.6) mimarisinin **bağımsız dış veri** üzerinde sıfır-eğitim
değerlendirmesinin girdilerini ve çıktılarını barındırır.

## İçerik

| Dosya | Açıklama | Git |
| :--- | :--- | :--- |
| `images/` | 400 VinDr-CXR JPEG (3× indirgenmiş; ~80 MB) | **takip edilmez** (`.gitignore`) |
| `cohort_metadata.csv` | `image_id, patient_label, group` | takip edilir |
| `cohort_with_diameters.csv` | + `bbox_max_dim_px`, `image_h/w`, `rel_size` | takip edilir |
| `v4_cohort_build_report.json` | Kohort üretim raporu (seed, mantık, aralıklar) | takip edilir |
| `v4_ref_histograms_report.json` | NODE21 200-hasta referans histogram raporu | takip edilir |
| `a0_a1_metadata_report.json` | "Sadece meta-veri" kısayol testi (A0/A1) | takip edilir |
| `vindr_report_*.json` | Kol başına değerlendirme (a-sweep, CI, alt gruplar) | takip edilir |
| `vindr_summary_report.json` | Tüm kolların özeti + protokol | takip edilir |

## Yeniden üretim sırası

```powershell
# 1) NODE21 referans histogramları (200 hasta, train-only)
python pipelines/exp/V4_build_ref_histograms.py

# 2) VinDr kohortu (150 nodül/kitle + 250 saf normal, seed 42)
python pipelines/exp/V4_build_vindr_cohort.py

# 3) Kohortun JPEG'leri (3x indirgenmiş VinDr arşivi) images/ altına açılmalı
#    (bkz. scratch/extract_vindr_cohort.py)

# 4) Değerlendirme (3 kalibrasyon kolu; VinDr'de eğitim YOK)
python pipelines/exp/V4_external_validation_vindr.py --reuse-resnet
```

## Önemli kısıtlar (belgelerle senkron)

* **Görüntü düzeyi:** VinDr metadata'sında hasta kimliği yoktur; kohort 400
  **görüntüdür**, 400 "hasta" değil.
* **mm dönüşümü yoktur:** kohort 3× indirgenmiş JPEG'lerden oluşur ve DICOM
  spacing içermez. Bu yüzden nodül boyutu alt grupları px/mm yerine
  **ölçek-bağımsız** `rel_size = bbox_max_dim_px / image_h` üçte birlikleri ile
  tanımlanır (`bbox_max_dim_px` annotasyon/orijinal DICOM uzayındadır).
* **Ölçek confound'u:** NODE21 kesitleri 1024×1024 ve spacing kaynağa göre
  0.147–0.615 mm/px; VinDr görüntüleri 682–1056 px ve spacing bilinmiyor. Kübik
  persistence ölçeğe duyarlıdır. Bu nedenle `matched_scalenorm` kolu, kesitleri
  **akciğer yüksekliğine göre** ölçekleyerek bu confound'u sınar.
* **ResNet ağırlıkları:** E0.6 ağırlıkları hiç kaydedilmediği için bit-bit
  yeniden üretim imkânsızdır; ResNet, E0.6 tarifiyle **deterministik** olarak
  NODE21'de yeniden eğitilir, val zon AUC uyumu (Δ<5e-3) doğrulanır ve elde
  edilen sabit checkpoint VinDr çıkarımında kullanılır. VinDr'de eğitim yoktur.
* **Histogram eşleme yalnızca dokuya uygulanır**; arka plan `0` kalır (aksi
  hâlde TDA'nın `image > 0` doku maskesi bozulur).

# NODE21 Detection — Grand-Challenge Gönderim Paketi (V6.7)

Kilitli çıkarım standardı `artifacts/v6_two_stage/submission_config.json` içinde
dondurulmuştur. Bu paket o sözleşmeyi sunucuda birebir uygular.

## Boru hattı (tek görüntü)

```
ham CXR
  → PSPNet akciğer segmentasyonu (CANLI; sunucu maske vermez)
  → 15 px dilasyon + c_lungref (tüm-akciğer p1–p99) kanonikleştirme
  → M5_foundation_fulldata (xrv ResNet50-res512 + ASPP/FPN + CenterNet)
      çok ölçekli {0.9, 1.0, 1.1} × yatay çevirme TTA  (6 ileri geçiş)
  → 3×3 yerel tepe + 8 px mesafe-NMS + top-50 aday
  → 56×56 sabit kutu                       (yarıçap kafası KULLANILMAZ)
  → her aday için 32/64 px çift ölçekli H0/H1 PI (1600) + 4 uzamsal
  → TDA-HGB_fulldata  →  p_tda
  → KİLİTLİ füzyon:  S = sigmoid(0.55·z(s1) + 0.45·z(p_tda))
  → /output/nodules.json  ("Multiple 2D bounding boxes")
```

## Girdi / Çıktı

| | |
| :--- | :--- |
| Girdi | `/input/images/chest-x-rays/<ad>.mha` — 2B `(H,W)` **veya** 3B `(N_z,H,W)` yığın |
| 3B kural | Her `z` katmanı **tek tek** işlenir; köşelere `z` indeksi yazılır |
| Çıktı | `/output/nodules.json` |
| Şema | `{"name":"Regions of interest","type":"Multiple 2D bounding boxes","boxes":[{"corners":[[x1,y1,z],[x1,y2,z],[x2,y2,z],[x2,y1,z]],"probability":f}],"version":{"major":1,"minor":0}}` |
| Koordinat | `sitk_image.TransformContinuousIndexToPhysicalPoint([x_px, y_px, z])` → **fiziksel mm** |

## `--network=none` güvenliği

1. **İndirme yasağı:** omurga (`pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt`,
   90 MB) ve PSPNet (`pspnet_chestxray_best_model_4.pth`, 260 MB) ağırlıkları
   imaja **gömülüdür** (`submission_docker/weights/` → `/opt/xrv_weights`,
   `XRV_CACHE_DIR`). Dosyalar mevcutsa torchxrayvision indirme yapmaz; ayrıca
   `xrv.utils.download` çalışma anında **engellenir** (sessiz indirme imkânsız).
2. **Canlı maske:** akciğer maskesi çalışma anında PSPNet ile ham röntgenden
   üretilir (sunucudan hazır maske gelmez).

## Build & test

```bash
# 1) İmaj
docker build -f submission_docker/Dockerfile -t node21-detection:v6.7 .

# 2) Ağsız uçtan uca (3B yığın)
docker run --rm --network=none \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  node21-detection:v6.7
```

**Yerel (Docker'sız) doğrulama:**

```bash
python submission_docker/smoke_test.py --n 3
```

Bu betik: gerçek val görüntülerinden bilinen spacing/origin ile 3B `.mha` yığını
üretir, `process.py`'yi koşar, JSON şemasını + mm dönüşümünün tersinirliğini
doğrular ve **negatif test** yapar (boş ağırlık önbelleği → sessiz indirme değil,
açık hata).

## Sınırlar (dürüstçe)

* **Rule 9:** eğitim verimiz NODE21'in **halka açık eğitim setidir**; resmî test
  setleri özeldir ⇒ leaderboard sıralaması bu depodan **ölçülemez**.
* `submission_config.json` içindeki `metric.validated_values`, **M5_foundation**
  (tam-veri değil) ile NODE21 val'de ölçülmüştür; tam-veri modeli NODE21'de
  değerlendirilemez (tüm veri eğitimde kullanıldı). Bu sayılar boru hattının
  **beklenen davranış referansıdır**, tam-veri modelinin performans iddiası değil.
* Zaman bütçesi: ≈2 s / görüntü (PSPNet ~0.1 s + omurga ~0.3 s + 50 aday TDA ~1.6 s).

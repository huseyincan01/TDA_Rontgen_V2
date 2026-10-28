# NODE21 Detection — Grand-Challenge Gönderim Paketi (V6.7)

Kilitli çıkarım standardı `artifacts/v6_two_stage/submission_config.json` içinde
dondurulmuştur. Bu paket o sözleşmeyi sunucuda birebir uygular.

## Resmî uyumluluk (`node21challenge/node21_detection_baseline`)

Resmî baseline `process.py`'nin `format_to_GC` fonksiyonu **kaynaktan** incelendi ve
şu üç noktada birebir eşitlendi:

| Konu | Resmî baseline | Bu paket |
| :--- | :--- | :--- |
| JSON anahtarları | `{type, boxes, version}` — **`name` YOK** | ✅ aynı |
| **Köşe sırası** | `[top_right, top_left, bottom_left, bottom_right]` = `[(xmax,ymax), (xmin,ymax), (xmin,ymin), (xmax,ymin)]` | ✅ aynı |
| **z koordinatı** | **slice indeksi** (`np_prediction['slice']`), mm değil | ✅ aynı |
| x/y birimi | **mm = piksel × spacing** (origin **yok sayılır**) | ✅ aynı (transform − origin) |
| Girdi yolu | `/input/<uuid>.mha` (evalutils) | ✅ **özyinelemeli** `rglob("*.mha")` — `/input/*.mha` **ve** `/input/images/chest-x-rays/*.mha` |
| Konteyner kullanıcısı | — (GC güvenlik kuralı: non-root) | ✅ `USER algorithm` (uid 1000), `/output` yazılabilir |

> Baseline kaynağı: <https://raw.githubusercontent.com/node21challenge/node21_detection_baseline/main/process.py>
> ⚠️ Baseline `probability`'yi 2 ondalığa yuvarlar; burada **4 ondalık** kullanılır
> (eşitlik/AUC sıralamasında tie üretmemek için). Koordinatlar baseline gibi **2 ondalık**.

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
| Girdi | `/input/images/chest-x-rays/<ad>.mha` **veya** `/input/<ad>.mha` — 2B `(H,W)` ya da 3B `(N_z,H,W)` |
| 3B kural | Her `z` katmanı **tek tek** işlenir; köşelere **`z` slice indeksi** yazılır |
| Çıktı | `/output/nodules.json` |
| Şema | `{"type":"Multiple 2D bounding boxes","boxes":[{"corners":[[xmax,ymax,z],[xmin,ymax,z],[xmin,ymin,z],[xmax,ymin,z]],"probability":f}],"version":{"major":1,"minor":0}}` |
| Koordinat | x/y **mm** (`piksel × spacing`, 2 ondalık) · z **slice indeksi** |

## `--network=none` güvenliği

1. **İndirme yasağı:** omurga (`…resnet50-test512-e400-state.pt`, 90 MB) ve PSPNet
   (`pspnet_chestxray_best_model_4.pth`, 260 MB) imaja **gömülüdür**
   (`/opt/xrv_weights`, `XRV_CACHE_DIR`). Dosya eksikse `xrv.utils.download`
   **engellenir** ⇒ sessiz indirme imkânsız, açık hata.
2. **Canlı maske:** akciğer maskesi çalışma anında PSPNet ile ham röntgenden üretilir.

## Build & test

```bash
# CUDA (T4 için, ~14 GB açılmış)
docker build -f submission_docker/Dockerfile -t node21-detection:v6.7 .

# CPU (küçük imaj ~1.5 GB, yavaş)
docker build -f submission_docker/Dockerfile.cpu -t node21-detection:v6.7-cpu .

# Ağsız uçtan uca
docker run --rm --network=none -v <girdi>:/input:ro -v <cikti>:/output node21-detection:v6.7
```

**Yerel (Docker'sız) doğrulama — 5 denetim:**

```bash
python submission_docker/smoke_test.py --n 3
```
1. JSON şeması (`name` yok, `type`/`version` doğru)
2. **Köşe sırası** [TR, TL, BL, BR]
3. x/y mm ↔ piksel tersinirlik + 56 px → mm dönüşümü
4. **Negatif test:** boş ağırlık önbelleği → açık hata (sessiz indirme yasak)
5. **Her iki girdi düzeni** (`/input/*.mha` ve `/input/images/chest-x-rays/*.mha`)

## Sınırlar (dürüstçe)

* **Rule 9:** eğitim verimiz NODE21'in **halka açık eğitim setidir**; resmî test
  setleri özeldir ⇒ leaderboard sıralaması bu depodan **ölçülemez**.
* NODE21 yarışması "Now Closed"; paket ileriye dönük **Algoritma** yüklemesi /
  tekrar üretilebilirlik için hazırdır.
* `submission_config.json` içindeki `metric.validated_values`, **M5_foundation**
  (tam-veri değil) ile NODE21 val'de ölçülmüştür; tam-veri modeli NODE21'de
  değerlendirilemez (tüm veri eğitimde kullanıldı).
* ⚠️ **Bu değerler "seçilmiş"tir:** kutu boyu/füzyon/TTA/tepe penceresi aynı val
  bölmesinde **296 + 19 + 12** değerlendirmeyle seçildi. Aynı kilitli protokolün
  ön-kayıtlı ikinci mührü (V6.9) dürüst dış-değeri verir: **rank_metric 0,8842**
  (1. aşama) / **0,8970** (füzyon) · hasta AUC **0,9446** / **0,9645** — yani val
  sayılarından **0,015–0,029** düşük. Konteynerin çıktısı değişmez (aynı kilitli
  sözleşme); değişen, o sözleşmenin **beklenen performansıdır**.
  Ayrıntı: [../LEAK_AUDIT.md](../LEAK_AUDIT.md) §5.
* Zaman: GPU'da ~1–2 s/katman; **GPU'suz ~26 s/katman** (CPU varyantı da aynı).

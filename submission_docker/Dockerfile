# NODE21 Detection Track — Grand-Challenge gönderim imajı
#
# Tasarım kararları
# -----------------
# * CUDA runtime tabanı: sunucu NVIDIA T4 verir; ResNet50 6 ileri geçiş + PSPNet
#   CPU'da zaman sınırını zorlar.
# * AĞIRLIKLAR İMAJA GÖMÜLÜDÜR (submission_docker/weights/ → /opt/xrv_weights).
#   XRV_CACHE_DIR bu dizine işaret eder ⇒ `--network=none` altında hiçbir indirme
#   denenmez (dosyalar mevcutsa torchxrayvision indirme yapmaz).
# * Gönderim modelleri (M5_foundation_fulldata.pt + stage2_tda_hgb_fulldata.joblib)
#   ve kilitli submission_config.json imaja kopyalanır.
#
# Build (depo kökünden):
#   docker build -f submission_docker/Dockerfile -t node21-detection:v6.7 .
#
# Ağsız uçtan uca test:
#   docker run --rm --network=none \
#     -v <yerel_girdi>:/input:ro -v <yerel_cikti>:/output \
#     node21-detection:v6.7

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XRV_CACHE_DIR=/opt/xrv_weights \
    HOME=/root \
    MPLBACKEND=Agg

WORKDIR /opt/app

# --- 1) Python bağımlılıkları (torch tabandan gelir)
COPY submission_docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- 2) Kaynak kod + kilitli sözleşme + gönderim modelleri
COPY src/ /opt/app/src/
COPY pipelines/ /opt/app/pipelines/
COPY artifacts/v6_two_stage/submission_config.json /opt/app/artifacts/v6_two_stage/submission_config.json
COPY artifacts/v6_two_stage/submission_fulldata/ /opt/app/artifacts/v6_two_stage/submission_fulldata/

# --- 3) Gömülü torchxrayvision ağırlıkları (omurga + PSPNet) ⇒ AĞSIZ
COPY submission_docker/weights/ /opt/xrv_weights/

# --- 4) Giriş noktası
COPY submission_docker/process.py /opt/app/process.py

# Ağırlık dosyalarının imajda olduğunu build sırasında doğrula (erken hata)
RUN python - <<'PY'
import os, pathlib
need = ["pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt",
        "pspnet_chestxray_best_model_4.pth"]
for n in need:
    p = pathlib.Path(os.environ["XRV_CACHE_DIR"]) / n
    assert p.exists(), f"EKSİK AĞIRLIK: {p}"
    print("OK", n, round(p.stat().st_size / 1e6, 1), "MB")
PY

CMD ["python", "/opt/app/process.py"]

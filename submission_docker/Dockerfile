# NODE21 Detection Track — Grand-Challenge gönderim imajı
#
# Resmî uyumluluk (node21challenge/node21_detection_baseline'a göre)
# ------------------------------------------------------------------
# * NON-ROOT kullanıcı (`USER algorithm`, uid 1000) — Grand Challenge güvenlik kuralı.
#   Gömülü ağırlıklar ve kod okunabilir (a+rX), /output yazılabilir.
# * AĞIRLIKLAR İMAJA GÖMÜLÜ (submission_docker/weights/ → /opt/xrv_weights).
#   XRV_CACHE_DIR bu dizine işaret eder ⇒ `--network=none` altında indirme YOK.
# * Girdi yolu esnektir: process.py /input altını ÖZYİNELEMELİ tarar
#   (/input/<uuid>.mha ve /input/images/chest-x-rays/<uuid>.mha ikisi de).
#
# Build (depo kökünden):
#   docker build -f submission_docker/Dockerfile -t node21-detection:v6.7 .
#
# Ağsız test:
#   docker run --rm --network=none -v <girdi>:/input:ro -v <cikti>:/output \
#     node21-detection:v6.7
#
# NOT — imaj boyutu: CUDA runtime tabanı büyüktür (~14 GB açılmış). Grand Challenge
# yükleme sınırını aşıyorsa `submission_docker/Dockerfile.cpu` (CPU taban, ~3 GB)
# kullanılabilir; o varyant daha yavaştır (~26 s/katman GPU'suz).

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XRV_CACHE_DIR=/opt/xrv_weights \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch_home \
    MPLBACKEND=Agg

WORKDIR /opt/app

# --- 1) Python bağımlılıkları (torch tabandan gelir; matplotlib GEREKMEZ — tembel import)
COPY submission_docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -rf /root/.cache /tmp/requirements.txt

# --- 2) Kaynak kod + kilitli sözleşme + gönderim modelleri
COPY src/ /opt/app/src/
COPY pipelines/ /opt/app/pipelines/
COPY artifacts/v6_two_stage/submission_config.json /opt/app/artifacts/v6_two_stage/submission_config.json
COPY artifacts/v6_two_stage/submission_fulldata/ /opt/app/artifacts/v6_two_stage/submission_fulldata/

# --- 3) Gömülü torchxrayvision ağırlıkları (omurga + PSPNet) ⇒ AĞSIZ
COPY submission_docker/weights/ /opt/xrv_weights/

# --- 4) Giriş noktası
COPY submission_docker/process.py /opt/app/process.py

# --- 5) NON-ROOT kullanıcı + izinler (Grand Challenge kuralı)
RUN useradd -m -u 1000 -s /bin/bash algorithm \
 && mkdir -p /output /tmp/matplotlib /tmp/torch_home \
 && chmod -R a+rX /opt/xrv_weights /opt/app \
 && chown -R algorithm:algorithm /output /tmp/matplotlib /tmp/torch_home

# Ağırlıkların imajda ve OKUNABİLİR olduğunu build sırasında doğrula
RUN python - <<'PY'
import os, pathlib
need = ["pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt",
        "pspnet_chestxray_best_model_4.pth"]
for n in need:
    p = pathlib.Path(os.environ["XRV_CACHE_DIR"]) / n
    assert p.exists(), f"EKSİK AĞIRLIK: {p}"
    assert os.access(p, os.R_OK), f"OKUNAMIYOR: {p}"
    print("OK", n, round(p.stat().st_size / 1e6, 1), "MB")
PY

USER algorithm

CMD ["python", "/opt/app/process.py"]

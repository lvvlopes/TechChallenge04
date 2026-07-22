# =====================================================================
# Monitoramento Multimodal de Pacientes — imagem de runtime
#
# Build leve (padrão): núcleo + Azure Speech/Language (RF02, RF03).
# Build completo (RF01 com visão computacional):
#     docker build --build-arg INSTALL_VISION=true -t mmonitor .
# =====================================================================
FROM python:3.11-slim

# Ativa OpenCV + MediaPipe + YOLO. Mantido desligado por padrão: a imagem
# salta de ~700 MB para ~3 GB e exige contêiner com mais memória.
ARG INSTALL_VISION=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# ffmpeg é chamado via subprocess por audio_analysis/speech_to_text.py para
# converter uploads em WAV 16 kHz mono antes de enviar ao Azure Speech.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Coorte sintética (20 pacientes) lida por cohort.py em runtime — CSVs, .txt e
# .wav pequenos (~500 KB cada). Necessária para a página /patients funcionar.
COPY data/patients ./data/patients

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[audio]" \
    && if [ "$INSTALL_VISION" = "true" ]; then \
    apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir ".[vision]"; \
    fi

# Uploads de áudio/vídeo são gravados em disco pelo endpoint /intake.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn multimodal_monitor.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

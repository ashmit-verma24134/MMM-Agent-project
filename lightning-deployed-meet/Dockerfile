FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIPELINE_WORKDIR=/data/deployed-meet-runs

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

RUN mkdir -p /data/deployed-meet-runs

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]


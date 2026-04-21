# Base: PyTorch 2.5.1 + CUDA 12.1 + cuDNN 9, matches VoxCPM2 requirements
# (Python >=3.10 <3.13, PyTorch >=2.5.0, CUDA >=12.0, ~8GB VRAM)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    MODELSCOPE_CACHE=/runpod-volume/modelscope \
    VOXCPM_MODEL_ID=openbmb/VoxCPM2 \
    VOXCPM_LOAD_DENOISER=false \
    CC=gcc \
    CXX=g++

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
        git \
        ca-certificates \
        gcc \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY handler.py .
COPY test_input.json .

# Optional: pre-download model into the image so cold starts don't block on HF.
# Comment out if you prefer lazy download onto a RunPod network volume.
ARG PREFETCH_MODEL=true
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        python -c "from huggingface_hub import snapshot_download; \
snapshot_download('openbmb/VoxCPM2', cache_dir='/opt/hf-cache')" && \
        echo "Prefetched model into /opt/hf-cache"; \
    fi
ENV HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    MODELSCOPE_CACHE=/opt/hf-cache/modelscope

CMD ["python", "-u", "handler.py"]

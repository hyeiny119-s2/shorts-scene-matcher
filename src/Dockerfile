FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DINOv2-Base 모델은 첫 실행 시 자동 다운로드 (~330MB)
# 캐시 유지하려면: -v ~/.cache/huggingface:/root/.cache/huggingface

ENTRYPOINT ["python", "main.py"]

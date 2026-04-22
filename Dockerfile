# PyTorch + CUDA 12.1 베이스 (torch/torchvision 이미 포함)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CLIP ViT-B/32 가중치 빌드 시 미리 다운로드 (실행마다 재다운로드 방지)
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"

CMD ["python", "main.py"]

FROM --platform=linux/amd64 python:3.7.6-slim

# 필수 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# PyTorch 설치 (CUDA 10.2 버전)
RUN pip install torch==1.10.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# 나머지 패키지 설치
RUN pip install numpy==1.19.5 \
    scipy==1.5.2 \
    tensorboard==2.11.2 \
    six

WORKDIR /workspace

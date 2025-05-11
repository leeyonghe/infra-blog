---
title: "Audiocraft Docker 빌드 설치"
date: 2024-04-03 12:30:00 +0900
categories: [audiocraft]
tags: [docker, python, pip, audiocraft, module-not-found]
---

Audiocraft를 Docker 환경에서 실행하기 위한 빌드 환경 설정 방법을 안내해드리겠습니다.

## 사전 요구사항

- Docker가 설치되어 있어야 합니다.
- Git이 설치되어 있어야 합니다.
- 최소 8GB 이상의 RAM이 필요합니다.
- CUDA 지원 GPU가 권장됩니다.

## Docker 이미지 빌드

1. Audiocraft 저장소 클론:
```bash
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
```

2. Dockerfile 생성:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# Audiocraft 설치
COPY . .

# 작업 디렉토리 설정
WORKDIR /app

# 기본 포트 설정
EXPOSE 8000

# 실행 명령
CMD ["python", "app.py"]
```

3. Docker 이미지 빌드:
```bash
docker build -t audiocraft:latest .
```

## Docker 컨테이너 실행

```bash
docker run -it --gpus all -p 8000:8000 audiocraft:latest
```

## 주의사항

1. GPU 사용을 위해서는 NVIDIA Container Toolkit이 설치되어 있어야 합니다.
2. 메모리 사용량이 많으므로 충분한 시스템 리소스가 필요합니다.
3. 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다.

## 문제 해결

### 일반적인 오류

1. CUDA 관련 오류:
   - NVIDIA 드라이버가 최신 버전인지 확인
   - Docker의 GPU 지원이 활성화되어 있는지 확인

2. 메모리 부족 오류:
   - Docker 컨테이너의 메모리 제한을 늘리기
   - `--memory` 옵션 사용

3. 모듈을 찾을 수 없는 오류:
   - requirements.txt가 올바르게 설치되었는지 확인
   - Python 경로가 올바르게 설정되어 있는지 확인

## 추가 설정

### 환경 변수 설정

```bash
docker run -it --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MODEL_PATH=/app/models \
  -p 8000:8000 \
  audiocraft:latest
```

### 볼륨 마운트

```bash
docker run -it --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  -p 8000:8000 \
  audiocraft:latest
```

이렇게 설정하면 Audiocraft를 Docker 환경에서 안정적으로 실행할 수 있습니다. 필요한 경우 추가적인 설정이나 문제 해결 방법을 문의해 주세요.


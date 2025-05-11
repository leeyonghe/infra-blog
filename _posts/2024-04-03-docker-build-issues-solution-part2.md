---
layout: post
title: "Docker 빌드 문제 해결하기 - Python 패키지 설치와 모듈 경로"
date: 2024-04-03 12:30:00 +0900
categories: [audiocraft]
tags: [docker, python, pip, audiocraft, module-not-found]
---

# Docker 빌드 문제 해결하기 - Python 패키지 설치와 모듈 경로

오늘은 Docker 이미지 빌드 과정에서 발생한 Python 패키지 설치와 모듈 경로 관련 문제를 해결한 경험을 공유하고자 합니다.

## 1. Python 가상환경 설정

### 문제 상황
```
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager.
```

### 원인
Docker 컨테이너에서 root 사용자로 pip를 실행하면 권한 문제가 발생할 수 있습니다.

### 해결 방법
Python 가상환경을 생성하고 활성화하여 패키지를 설치하도록 했습니다:

```dockerfile
# Python 가상환경 생성 및 활성화
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
```

## 2. 패키지 설치 순서 최적화

### 문제 상황
```
ModuleNotFoundError: No module named 'audiocraft.serve'
```

### 원인
1. 파일 복사와 패키지 설치 순서가 최적화되지 않았습니다.
2. `audiocraft.serve` 모듈이 존재하지 않았습니다.

### 해결 방법
Dockerfile의 순서를 다음과 같이 변경했습니다:

```dockerfile
# 애플리케이션 파일 복사
COPY . .

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# audiocraft 패키지 설치
RUN pip install -e .
```

이렇게 하면:
1. 먼저 모든 소스 코드를 복사
2. requirements.txt의 의존성 설치
3. audiocraft 패키지를 개발 모드로 설치

## 3. 모듈 경로 문제

### 문제 상황
`audiocraft.serve.api` 모듈을 찾을 수 없었습니다.

### 원인
프로젝트 구조를 확인해보니 `audiocraft/serve` 디렉토리가 존재하지 않았습니다.

### 해결 방법
1. `audiocraft/serve` 디렉토리와 `api.py` 파일을 생성하거나
2. `audiocraft.serve.api`를 사용하지 않는 다른 방식으로 실행

## 최종 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 의존성 설치
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 가상환경 생성 및 활성화
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 애플리케이션 파일 복사
COPY . .

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# audiocraft 패키지 설치
RUN pip install -e .

# 환경 변수 설정
ENV PYTHONPATH=/workspace
```

## 결론

Docker 빌드 과정에서 발생하는 Python 관련 문제들은 다음과 같은 원인들로 인해 발생할 수 있습니다:
1. 패키지 설치 권한 문제
2. 파일 복사와 패키지 설치 순서
3. 모듈 경로와 프로젝트 구조

이러한 문제들은 가상환경 사용, 적절한 설치 순서, 그리고 프로젝트 구조 확인을 통해 해결할 수 있습니다. 
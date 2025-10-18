---
layout: post
title: "Docker Compose 오케스트레이션 심화 분석 - AudioCraft Custom 프로젝트"
date: 2024-12-21
categories: [DevOps, Docker, Container Orchestration]
tags: [AudioCraft, Docker Compose, GPU Resources, Volume Management, Service Definition]
author: "AI Blog"
---

# Docker Compose 오케스트레이션 심화 분석

AudioCraft Custom 프로젝트의 Docker Compose 구성을 통해 컨테이너 오케스트레이션의 고급 기법들을 심층 분석해보겠습니다. GPU 리소스 관리, 볼륨 마운트 전략, 환경 변수 관리 등 실제 프로덕션 환경에서 중요한 구성 요소들을 상세히 살펴보겠습니다.

## 📋 목차
1. [서비스 정의 아키텍처](#서비스-정의-아키텍처)
2. [볼륨 마운트 전략](#볼륨-마운트-전략)
3. [GPU 리소스 관리](#gpu-리소스-관리)
4. [환경 변수 구성](#환경-변수-구성)
5. [네트워크 및 포트 관리](#네트워크-및-포트-관리)
6. [프로덕션 최적화 전략](#프로덕션-최적화-전략)

## 서비스 정의 아키텍처

### 기본 서비스 구성

```yaml
services:
  audiocraft:
    tty: true
    build:
      context: .
      dockerfile: Dockerfile
```

#### 🏗️ 서비스 정의 분석
- **tty: true**: 터미널 할당으로 대화형 셸 지원
- **build 컨텍스트**: 현재 디렉토리를 빌드 컨텍스트로 설정
- **Dockerfile 지정**: 명시적 Dockerfile 경로 설정

#### 💡 TTY 할당의 중요성
```yaml
tty: true
```
- **대화형 디버깅**: 컨테이너 내부에서 직접 명령 실행 가능
- **프로세스 안정성**: 장기 실행 프로세스의 안정적 동작
- **로그 출력**: 실시간 로그 확인과 디버깅 지원

### 빌드 컨텍스트 최적화

```yaml
build:
  context: .
  dockerfile: Dockerfile
  args:
    - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
  target: production  # 다단계 빌드에서 특정 스테이지 선택
```

#### 🔧 고급 빌드 옵션
- **빌드 인수**: 런타임 변수를 빌드 타임에 전달
- **타겟 스테이지**: 다단계 빌드에서 특정 단계만 빌드
- **캐시 전략**: 빌드 성능 최적화

## 볼륨 마운트 전략

### 계층화된 볼륨 구조

```yaml
volumes:
  - .:/workspace  # 현재 디렉토리를 컨테이너의 /workspace에 마운트
  - ./dataset:/workspace/dataset  # 데이터셋 디렉토리 마운트
  - ./api:/workspace/api  # API 디렉토리 마운트
```

#### 📁 볼륨 마운트 전략 분석

##### 1. 전체 프로젝트 마운트
```yaml
- .:/workspace
```
- **개발 효율성**: 코드 변경 시 즉시 반영
- **실시간 테스트**: 컨테이너 재빌드 없이 테스트 가능
- **디버깅 지원**: 호스트에서 직접 파일 수정 가능

##### 2. 데이터셋 분리 마운트
```yaml
- ./dataset:/workspace/dataset
```
- **데이터 격리**: 대용량 데이터의 독립적 관리
- **성능 최적화**: 데이터 로딩 성능 향상
- **백업 전략**: 데이터와 코드의 분리된 백업

##### 3. API 서비스 마운트
```yaml
- ./api:/workspace/api
```
- **서비스 분리**: API 코드의 독립적 개발
- **핫 리로드**: FastAPI 서버의 자동 재시작
- **모듈화**: 마이크로서비스 아키텍처 지원

### 고급 볼륨 옵션

```yaml
volumes:
  - type: bind
    source: ./models
    target: /workspace/models
    read_only: true  # 읽기 전용 마운트
  - type: tmpfs
    target: /tmp/audio_processing
    tmpfs:
      size: 1G  # 임시 파일시스템 크기 제한
  - type: volume
    source: model_cache
    target: /root/.cache/huggingface
```

#### 🛡️ 볼륨 보안 및 성능
- **읽기 전용**: 중요한 모델 파일 보호
- **tmpfs**: 메모리 기반 임시 저장소로 I/O 성능 향상
- **명명된 볼륨**: 영구 캐시 데이터 관리

## GPU 리소스 관리

### NVIDIA GPU 할당

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia  # NVIDIA GPU 사용 설정
          count: 1
          capabilities: [gpu]
```

#### 🎯 GPU 리소스 예약 전략

##### 리소스 예약 vs 제한
```yaml
deploy:
  resources:
    reservations:  # 최소 보장 리소스
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:  # 최대 사용 가능 리소스
      memory: 8G
      cpus: '4.0'
```

##### GPU 세부 설정
```yaml
devices:
  - driver: nvidia
    device_ids: ['0', '1']  # 특정 GPU 지정
    capabilities: [gpu, utility]  # 추가 기능 활성화
    options:
      - "compute-capability=8.0"  # CUDA Compute Capability
```

#### ⚡ GPU 메모리 최적화

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all  # 모든 GPU 사용 가능하도록 설정
  - CUDA_VISIBLE_DEVICES=0,1   # 특정 GPU만 사용
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

##### GPU 메모리 관리 옵션
- **NVIDIA_VISIBLE_DEVICES**: Docker에서 보이는 GPU 제어
- **CUDA_VISIBLE_DEVICES**: CUDA 애플리케이션에서 사용할 GPU
- **PyTorch 메모리 설정**: 메모리 할당 전략 최적화

### 다중 GPU 환경 구성

```yaml
services:
  audiocraft-worker-1:
    extends:
      service: audiocraft
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_ID=1

  audiocraft-worker-2:
    extends:
      service: audiocraft
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - WORKER_ID=2
```

#### 🔀 로드 밸런싱 전략
- **GPU별 워커**: 각 GPU에 전용 워커 할당
- **작업 분산**: 여러 GPU 간 작업 분배
- **장애 복구**: GPU 장애 시 다른 워커로 자동 전환

## 환경 변수 구성

### 기본 환경 변수

```yaml
environment:
  - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}
  - NVIDIA_VISIBLE_DEVICES=all
```

#### 🔐 보안 환경 변수 관리

##### .env 파일 활용
```bash
# .env 파일
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MODEL_CACHE_DIR=/models
DEBUG_MODE=false
```

##### 환경별 설정 분리
```yaml
# docker-compose.override.yml (개발환경)
services:
  audiocraft:
    environment:
      - DEBUG_MODE=true
      - LOG_LEVEL=debug
      - RELOAD=true

# docker-compose.prod.yml (프로덕션)
services:
  audiocraft:
    environment:
      - DEBUG_MODE=false
      - LOG_LEVEL=warning
      - WORKERS=4
```

### 고급 환경 변수 패턴

```yaml
environment:
  # 모델 설정
  - MODEL_CACHE_DIR=/root/.cache/huggingface
  - TRANSFORMERS_CACHE=/root/.cache/transformers
  - TORCH_HOME=/root/.cache/torch
  
  # 성능 튜닝
  - OMP_NUM_THREADS=4
  - MKL_NUM_THREADS=4
  - CUDA_LAUNCH_BLOCKING=0
  
  # 메모리 관리
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
  
  # 로깅 및 모니터링
  - PYTHONUNBUFFERED=1
  - CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-all}
```

#### 🎛️ 환경 변수 카테고리
- **인증 정보**: API 키, 토큰 등 보안 정보
- **경로 설정**: 캐시, 모델, 데이터 경로
- **성능 튜닝**: 스레드 수, 메모리 설정
- **디버깅**: 로그 레벨, 버퍼링 설정

## 네트워크 및 포트 관리

### 포트 매핑 전략

```yaml
ports:
  - "8000:8000"  # FastAPI 서버 포트
  - "7860:7860"  # Gradio UI 포트 (옵션)
  - "6006:6006"  # TensorBoard 포트 (옵션)
```

#### 🌐 네트워크 구성 최적화

##### 사용자 정의 네트워크
```yaml
networks:
  audiocraft-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  audiocraft:
    networks:
      - audiocraft-network
    ports:
      - "8000:8000"
```

##### 서비스 간 통신
```yaml
services:
  audiocraft-api:
    ports:
      - "8000:8000"
    networks:
      - audiocraft-network

  audiocraft-worker:
    expose:
      - "8001"  # 내부 통신용 포트
    networks:
      - audiocraft-network
```

### 로드 밸런서 통합

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - audiocraft-api
    networks:
      - audiocraft-network

  audiocraft-api:
    deploy:
      replicas: 3
    expose:
      - "8000"
    networks:
      - audiocraft-network
```

## 프로덕션 최적화 전략

### 헬스 체크 구성

```yaml
services:
  audiocraft:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

#### 🏥 헬스 체크 패턴
- **HTTP 체크**: API 엔드포인트 상태 확인
- **간격 설정**: 30초마다 상태 점검
- **재시도 정책**: 3회 실패 시 unhealthy 상태

### 리소스 제한 및 모니터링

```yaml
services:
  audiocraft:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

#### 📊 리소스 관리 전략
- **CPU 제한**: 시스템 자원 보호
- **메모리 제한**: OOM 방지
- **재시작 정책**: 장애 복구 자동화

### 로깅 및 모니터링

```yaml
services:
  audiocraft:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.audiocraft.rule=Host(`audiocraft.example.com`)"
```

#### 📝 로깅 구성
- **로그 드라이버**: JSON 파일 형식
- **로그 로테이션**: 크기 및 파일 수 제한
- **라벨링**: 서비스 디스커버리 지원

### 시크릿 관리

```yaml
secrets:
  hf_token:
    external: true
  ssl_cert:
    file: ./certs/cert.pem

services:
  audiocraft:
    secrets:
      - source: hf_token
        target: /run/secrets/hugging_face_token
        mode: 0400
    environment:
      - HUGGING_FACE_HUB_TOKEN_FILE=/run/secrets/hugging_face_token
```

#### 🔒 보안 강화
- **외부 시크릿**: Docker Swarm/Kubernetes 통합
- **파일 권한**: 읽기 전용 권한 설정
- **환경 분리**: 시크릿과 환경 변수 분리

## 🔍 핵심 인사이트

### 1. 계층화된 볼륨 전략
- **개발 효율성**: 실시간 코드 반영으로 개발 속도 향상
- **데이터 분리**: 코드와 데이터의 독립적 관리
- **성능 최적화**: 용도별 최적화된 볼륨 타입 선택

### 2. 정교한 GPU 관리
- **리소스 격리**: GPU별 워커 할당으로 안정성 확보
- **메모리 최적화**: CUDA 설정을 통한 메모리 효율성
- **확장성**: 다중 GPU 환경에서의 수평 확장

### 3. 환경별 설정 분리
- **개발/프로덕션**: 환경별 최적화된 설정
- **보안 관리**: 시크릿과 환경 변수의 체계적 관리
- **유연성**: 동적 환경 변수로 런타임 설정 변경

### 4. 프로덕션 준비도
- **헬스 체크**: 자동 장애 감지 및 복구
- **리소스 제한**: 시스템 안정성 보장
- **모니터링**: 종합적인 로깅 및 메트릭 수집

## 🎯 결론

AudioCraft Custom 프로젝트의 Docker Compose 구성은 개발 편의성과 프로덕션 안정성을 모두 고려한 균형잡힌 설계를 보여줍니다. GPU 리소스의 효율적 관리, 계층화된 볼륨 전략, 그리고 환경별 설정 분리를 통해 AI 워크로드에 특화된 컨테이너 오케스트레이션 모범 사례를 제시합니다.

특히 GPU 메모리 최적화와 다중 워커 구성은 대용량 AI 모델 서빙에서 중요한 성능 향상을 가져다주며, 체계적인 환경 변수 관리는 다양한 배포 환경에서의 유연성을 확보합니다.

---

*이 분석은 AudioCraft Custom 프로젝트의 실제 Docker Compose 구성을 기반으로 작성되었습니다. 더 자세한 구현 내용은 [AudioCraft 공식 저장소](https://github.com/facebookresearch/audiocraft)에서 확인할 수 있습니다.*
---
title: "Stable Diffusion WebUI Docker 프로젝트 구조 상세 설명"
date: 2024-03-25 12:00:00 +0900
categories: [stable-diffusion]
tags: [docker, stable-diffusion, webui]
---

Stable Diffusion WebUI Docker 프로젝트 구조 상세 설명

![NVIDIA Logo](https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-d@2x.png){: width="500" height="300"}

이 문서에서는 Stable Diffusion WebUI Docker 프로젝트의 모든 폴더와 파일에 대한 상세한 설명을 제공합니다.

## 1. 핵심 디렉토리 구조

### models/
Stable Diffusion 모델 파일들이 저장되는 핵심 디렉토리입니다.

#### models/Stable-diffusion/
- **체크포인트 모델**: `.ckpt`, `.safetensors` 확장자를 가진 모델 파일들
  - 기본 모델: `v1-5-pruned.ckpt`, `v2-1_768-ema-pruned.safetensors` 등
  - 커스텀 모델: 사용자가 추가한 체크포인트 파일들
- **VAE 모델**: `vae-ft-mse-840000.pt` 등 VAE 관련 파일들
- **LoRA 모델**: `.safetensors` 확장자를 가진 LoRA 파일들
  - 각 LoRA 파일은 특정 스타일이나 개념을 학습한 모델

#### models/ControlNet/
- ControlNet 모델 파일들
- 각 모델은 특정 제어 기능을 담당 (예: pose, depth, canny 등)

#### models/ESRGAN/
- 이미지 업스케일링을 위한 모델 파일들
- `RealESRGAN_x4plus.pth` 등

### outputs/
생성된 이미지와 관련 메타데이터가 저장되는 디렉토리입니다.

#### outputs/txt2img-images/
- 텍스트로 이미지를 생성한 결과물
- 각 이미지와 함께 생성 파라미터가 포함된 텍스트 파일
- 타임스탬프가 포함된 폴더 구조

#### outputs/img2img-images/
- 이미지 변환 결과물
- 원본 이미지와 변환된 이미지 쌍
- 변환 파라미터 정보

#### outputs/grids/
- 그리드 형태로 생성된 이미지들
- 여러 이미지를 한 번에 비교할 수 있는 결과물

### extensions/
커스텀 확장 기능들이 저장되는 디렉토리입니다.

#### extensions/installed/
- 설치된 확장 기능들의 소스 코드
- 각 확장 기능별 독립적인 디렉토리
- `requirements.txt`와 `install.py` 포함

#### extensions/disabled/
- 비활성화된 확장 기능들
- 필요시 다시 활성화 가능

### repositories/
외부 저장소에서 가져온 코드들이 저장되는 디렉토리입니다.

#### repositories/stable-diffusion-stability-ai/
- Stable Diffusion의 핵심 구현체
- 모델 아키텍처와 관련 코드

#### repositories/CodeFormer/
- 얼굴 복원을 위한 코드
- 모델 가중치와 구현체

## 2. 설정 및 스크립트 디렉토리

### configs/
설정 파일들이 저장되는 디렉토리입니다.

#### configs/v1-inference.yaml
- v1 모델의 기본 설정
- 모델 아키텍처 파라미터

#### configs/v2-inference.yaml
- v2 모델의 기본 설정
- 고해상도 생성 관련 설정

### scripts/
실행 스크립트들이 저장되는 디렉토리입니다.

#### scripts/onload/
- 웹UI 시작 시 자동으로 실행되는 스크립트
- 초기 설정 및 환경 구성

#### scripts/postprocessing/
- 이미지 후처리 관련 스크립트
- 업스케일링, 필터링 등

### localizations/
다국어 지원을 위한 번역 파일들이 저장되는 디렉토리입니다.

#### localizations/ko.json
- 한국어 번역 파일
- UI 요소들의 번역

#### localizations/en.json
- 영어 번역 파일
- 기본 언어 설정

## 3. 웹 인터페이스 관련 디렉토리

### html/
웹 인터페이스의 HTML 파일들이 저장되는 디렉토리입니다.

#### html/index.html
- 메인 웹 인터페이스
- UI 레이아웃 정의

#### html/extra-networks.html
- 추가 네트워크 선택 인터페이스
- LoRA, 임베딩 등 선택 UI

### javascript/
웹 인터페이스의 JavaScript 파일들이 저장되는 디렉토리입니다.

#### javascript/ui.js
- UI 상호작용 처리
- 이벤트 핸들러

#### javascript/imageviewer.js
- 이미지 뷰어 기능
- 확대/축소, 회전 등

## 4. 주요 설정 파일

### Dockerfile
Docker 이미지 빌드를 위한 설정 파일입니다.
- 기본 이미지 설정
- 의존성 설치 명령
- 환경 변수 설정
- 작업 디렉토리 설정

### docker-compose.yml
Docker 컨테이너 실행을 위한 설정 파일입니다.
- 서비스 정의
- 볼륨 마운트 설정
- 포트 매핑
- 환경 변수 설정

### requirements.txt
Python 패키지 의존성 목록입니다.
- 핵심 라이브러리 버전
- 선택적 의존성
- 호환성 정보

### webui.py
웹UI 실행을 위한 메인 Python 스크립트입니다.
- 서버 초기화
- 모델 로딩
- API 엔드포인트 정의
- 설정 관리

## 5. 실행 스크립트

### webui.sh
Linux/macOS용 실행 스크립트입니다.
- 환경 변수 설정
- 의존성 확인
- Python 가상환경 설정
- 서버 시작

### webui.bat
Windows용 실행 스크립트입니다.
- Windows 환경 설정
- 경로 설정
- 서비스 시작

### webui-macos-env.sh
macOS 환경 설정 스크립트입니다.
- macOS 특화 설정
- M1/M2 칩 지원
- 환경 변수 설정
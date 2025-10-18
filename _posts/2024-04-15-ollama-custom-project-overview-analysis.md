---
layout: post
title: "Ollama Custom 프로젝트 개요 분석 - 로컬 LLM 실행 플랫폼의 아키텍처"
date: 2024-04-15
categories: [AI, Go, LLM]
tags: [Ollama, Go, Language Models, Local AI, Custom Implementation]
---

# Ollama Custom 프로젝트 개요 분석

Ollama Custom은 로컬 환경에서 대규모 언어 모델(LLM)을 실행하기 위한 플랫폼입니다. 이 프로젝트는 원본 Ollama를 개인적으로 커스터마이징한 버전으로, Go 언어로 구현된 고성능 LLM 실행 환경을 제공합니다.

## 1. 프로젝트 구조 개요

### 1.1 핵심 디렉토리 구성

```
ollama-custom/
├── main.go                 # 애플리케이션 진입점
├── cmd/                    # CLI 명령어 구현
├── server/                 # HTTP 서버 및 API 라우팅
├── api/                    # 클라이언트-서버 통신 프로토콜
├── llm/                    # LLM 런타임 엔진
├── model/                  # 모델 관리 시스템
├── template/               # 프롬프트 템플릿 엔진
├── fs/                     # 파일 시스템 추상화
├── envconfig/              # 환경 설정 관리
├── types/                  # 커스텀 타입 정의
└── runner/                 # 모델 실행 환경
```

### 1.2 프로젝트 진입점

```go
// main.go
package main

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}
```

**분석 포인트:**
- **단순한 진입점**: 복잡한 초기화 로직 없이 Cobra CLI 프레임워크로 위임
- **컨텍스트 활용**: `context.Background()`로 전체 애플리케이션 생명주기 관리
- **에러 처리**: `cobra.CheckErr()`로 CLI 레벨에서 통합 에러 처리

## 2. Go 모듈 의존성 분석

### 2.1 핵심 의존성

```go
// go.mod 주요 의존성
require (
    github.com/containerd/console v1.0.3      // 터미널 제어
    github.com/gin-gonic/gin v1.10.0          // HTTP 웹 프레임워크
    github.com/spf13/cobra v1.7.0             // CLI 프레임워크
    github.com/google/uuid v1.6.0             // UUID 생성
    golang.org/x/sync v0.12.0                 // 고급 동기화 프리미티브
)
```

**의존성 특징:**
- **웹 서버**: Gin을 활용한 고성능 HTTP 서버
- **CLI**: Cobra를 통한 풍부한 명령행 인터페이스
- **동시성**: Go의 고급 동시성 패턴 활용
- **시스템**: 크로스 플랫폼 시스템 통합

### 2.2 머신러닝 특화 의존성

```go
require (
    github.com/pdevine/tensor v0.0.0-20240510204454-f88f4562727c
    github.com/nlpodyssey/gopickle v0.3.0
    github.com/x448/float16 v0.8.4
    github.com/d4l3k/go-bfloat16 v0.0.0-20211005043715-690c3bdd05f1
)
```

**ML 최적화:**
- **텐서 연산**: 효율적인 다차원 배열 처리
- **모델 직렬화**: Python Pickle 파일 파싱
- **수치 정밀도**: Float16, BFloat16 지원으로 메모리 최적화

## 3. 커스터마이징 특징

### 3.1 개인화 목적

README.md에서 명시된 바와 같이:
```markdown
[English] This project is for my personal customization and use.
[한국어] 이 프로젝트는 나 나름대로 커스터마이징 하여 사용하기 위함임.
```

**커스터마이징 방향:**
- 개인 워크플로우에 최적화된 기능
- 한국어 지원 강화
- 특정 사용 사례에 맞춘 성능 튜닝

### 3.2 지원 모델 스펙트럼

| 모델 | 파라미터 | 크기 | 실행 명령 |
|------|----------|------|-----------|
| Gemma 3 | 1B | 815MB | `ollama run gemma3:1b` |
| Gemma 3 | 4B | 3.3GB | `ollama run gemma3` |
| Gemma 3 | 12B | 8.1GB | `ollama run gemma3:12b` |
| Gemma 3 | 27B | 17GB | `ollama run gemma3:27b` |

## 4. 아키텍처 설계 원칙

### 4.1 모듈화 구조

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     CLI     │────│   Server    │────│     LLM     │
│   (cmd/)    │    │ (server/)   │    │   (llm/)    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                   ┌─────────────┐
                   │    Model    │
                   │  (model/)   │
                   └─────────────┘
```

**설계 특징:**
- **관심사 분리**: CLI, 서버, LLM 엔진의 명확한 역할 구분
- **플러그인 아키텍처**: 모듈 간 느슨한 결합
- **확장성**: 새로운 모델이나 기능 추가 용이

### 4.2 플랫폼 지원 전략

```dockerfile
# Dockerfile의 멀티 플랫폼 지원
ARG FLAVOR=${TARGETARCH}
FROM base-${TARGETARCH} AS base

# CPU 최적화
FROM base AS cpu
RUN dnf install -y gcc-toolset-11-gcc gcc-toolset-11-gcc-c++

# GPU 가속 (CUDA)
FROM base AS cuda-12
RUN dnf install -y cuda-toolkit-${CUDA12VERSION//./-}

# AMD GPU (ROCm)
FROM base AS rocm-6
ENV PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:$PATH
```

**플랫폼 최적화:**
- **CPU**: GCC 툴체인 최적화
- **NVIDIA GPU**: CUDA 12.x 지원
- **AMD GPU**: ROCm 6.x 지원
- **ARM64**: ARM 프로세서 네이티브 지원

## 5. 성능 최적화 전략

### 5.1 메모리 효율성

```go
// types/syncmap을 활용한 동시성 제어
type SyncMap struct {
    mu    sync.RWMutex
    items map[string]interface{}
}
```

**메모리 관리:**
- **동시성 안전**: sync.RWMutex를 통한 읽기/쓰기 분리
- **타입 안전성**: 제네릭을 활용한 타입 보장
- **메모리 풀링**: 대용량 모델 로딩 최적화

### 5.2 I/O 최적화

```go
// envconfig를 통한 설정 최적화
type Config struct {
    ModelPath    string `env:"OLLAMA_MODELS"`
    Host         string `env:"OLLAMA_HOST"`
    Port         int    `env:"OLLAMA_PORT"`
    Debug        bool   `env:"OLLAMA_DEBUG"`
}
```

**I/O 전략:**
- **환경 변수 기반**: 런타임 설정 변경
- **캐싱 시스템**: 모델 로딩 시간 단축
- **스트리밍**: 대용량 파일 처리 최적화

## 6. 보안 및 안정성

### 6.1 타입 안전성

```go
// types/errtypes을 통한 에러 타입 시스템
type ModelError struct {
    Model   string
    Message string
    Code    int
}

func (e ModelError) Error() string {
    return fmt.Sprintf("model %s: %s (code: %d)", e.Model, e.Message, e.Code)
}
```

**안전성 보장:**
- **강타입 에러**: 에러 상황별 구체적인 타입 정의
- **컨텍스트 보존**: 에러 발생 시점의 상세 정보 유지
- **복구 가능성**: 에러 종류별 적절한 복구 전략

### 6.2 암호화 지원

```go
// crypto/ed25519를 활용한 키 관리
privateKey, publicKey, err := ed25519.GenerateKey(rand.Reader)
if err != nil {
    return fmt.Errorf("failed to generate key: %w", err)
}
```

## 7. 개발 도구 및 빌드 시스템

### 7.1 빌드 자동화

```bash
# scripts/build_linux.sh
#!/bin/bash
set -e

# 플랫폼별 최적화 빌드
CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -tags cuda
CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -tags rocm
```

### 7.2 테스트 전략

```
server/
├── routes_test.go          # API 라우팅 테스트
├── model_test.go           # 모델 로딩 테스트
├── sched_test.go           # 스케줄링 테스트
└── testdata/               # 테스트 데이터
```

## 8. 결론

Ollama Custom은 현대적인 Go 개발 패턴을 적용한 고품질 LLM 플랫폼입니다:

### 8.1 주요 강점

- **모듈화된 아키텍처**: 유지보수 및 확장 용이성
- **크로스 플랫폼**: CPU, GPU, ARM 모든 플랫폼 지원
- **성능 최적화**: 메모리, I/O, 동시성 최적화
- **개발자 친화적**: 풍부한 CLI와 API 인터페이스

### 8.2 커스터마이징 가치

- **개인화**: 특정 워크플로우에 맞춘 최적화
- **현지화**: 한국어 지원 강화
- **실험적 기능**: 원본에 없는 혁신적 기능 테스트

다음 포스트에서는 CLI 명령어 시스템의 상세 구현을 분석해보겠습니다.
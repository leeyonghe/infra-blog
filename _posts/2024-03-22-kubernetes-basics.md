---
layout: post
title: "Kubernetes 기초부터 실전까지 | Kubernetes from Basics to Production"
date: 2024-03-22 10:00:00 +0900
categories: [Kubernetes, Container]
tags: [kubernetes, k8s, container, orchestration, devops, docker]
---

Kubernetes는 현대 인프라의 핵심 기술입니다. 이 포스트에서는 Kubernetes의 기본 개념부터 실제 운영 환경에서의 활용법까지 자세히 알아보겠습니다.

## Kubernetes란? | What is Kubernetes?

Kubernetes(K8s)는 컨테이너화된 애플리케이션의 배포, 확장, 관리를 자동화하는 오픈소스 플랫폼입니다.

### 주요 특징 | Key Features
- **자동 스케일링** | Auto-scaling
- **서비스 디스커버리** | Service Discovery  
- **롤링 업데이트** | Rolling Updates
- **자동 복구** | Self-healing
- **비밀 정보 관리** | Secret Management

## 핵심 컴포넌트 | Core Components

### 1. Master Node Components
```yaml
# Control Plane 구성요소
- API Server: 클러스터 관리 API 제공
- etcd: 클러스터 데이터 저장소
- Controller Manager: 클러스터 상태 관리
- Scheduler: 파드 스케줄링
```

### 2. Worker Node Components
```yaml
# Worker Node 구성요소
- kubelet: 노드 에이전트
- kube-proxy: 네트워크 프록시
- Container Runtime: Docker, containerd 등
```

## 기본 리소스 타입 | Basic Resource Types

### Pod
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    ports:
    - containerPort: 80
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
```

## 실전 운영 가이드 | Production Operations Guide

### 모니터링 설정
```bash
# Prometheus + Grafana 설치
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack
```

### 로깅 구성
```bash
# ELK Stack 배포
kubectl apply -f elasticsearch.yaml
kubectl apply -f kibana.yaml
kubectl apply -f filebeat.yaml
```

### 보안 설정
```yaml
# Network Policy 예제
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

다음 포스트에서는 Kubernetes 클러스터 구축 실습을 다뤄보겠습니다!
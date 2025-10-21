---
layout: post
title: "배포 전략 완전 가이드 | Complete Deployment Strategies Guide"
date: 2024-11-04 10:00:00 +0900
categories: [DevOps, Deployment]
tags: [deployment, ci-cd, blue-green, canary, rolling-update, kubernetes, aws, automation]
---

현대적인 소프트웨어 개발에서 배포는 단순히 코드를 서버에 올리는 것이 아닙니다. 안정적이고 예측 가능한 배포 전략을 통해 서비스 중단을 최소화하고 사용자 경험을 보장하는 것이 핵심입니다.

## 3-1 배포 | Deployment

### 3-1-1 배포 자동화 | Deployment Automation

배포 자동화는 소프트웨어 릴리스 과정을 체계화하고 인간의 실수를 최소화하여 안정적인 서비스 운영을 보장합니다.

#### 🚀 배포 자동화의 핵심 개념

```bash
# 배포 자동화 구성 요소
1. 소스 코드 관리 (SCM)
   - Git, SVN
   - 브랜치 전략 (Git Flow, GitHub Flow)

2. 빌드 자동화
   - 컴파일, 패키징
   - 테스트 실행
   - 아티팩트 생성

3. 배포 파이프라인
   - 단계별 배포 (Dev → Staging → Production)
   - 승인 프로세스
   - 롤백 메커니즘

4. 인프라 자동화
   - IaC (Infrastructure as Code)
   - 컨테이너화
   - 오케스트레이션
```

#### CI/CD 파이프라인 설계

```yaml
# .gitlab-ci.yml - GitLab CI/CD 파이프라인
stages:
  - build
  - test
  - security
  - deploy-staging
  - deploy-production

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_HOST: tcp://docker:2376
  DOCKER_TLS_CERTDIR: "/certs"

before_script:
  - echo "Pipeline started at $(date)"
  - docker info

# 빌드 단계
build:
  stage: build
  image: node:16-alpine
  script:
    - npm ci --only=production
    - npm run build
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  artifacts:
    paths:
      - dist/
    expire_in: 1 hour
  only:
    - main
    - develop

# 테스트 단계
unit-test:
  stage: test
  image: node:16-alpine
  script:
    - npm ci
    - npm run test:unit
    - npm run test:coverage
  artifacts:
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'

integration-test:
  stage: test
  services:
    - postgres:13
    - redis:6
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
  script:
    - npm run test:integration
  only:
    - main

# 보안 스캔
security-scan:
  stage: security
  image: owasp/zap2docker-stable
  script:
    - zap-baseline.py -t http://localhost:3000 -J zap-report.json
    - npm audit --audit-level high
  artifacts:
    reports:
      sast: zap-report.json
  allow_failure: true

# 스테이징 배포
deploy-staging:
  stage: deploy-staging
  image: alpine/helm:3.8.0
  script:
    - helm upgrade --install myapp-staging ./helm-chart
      --namespace staging
      --set image.tag=$CI_COMMIT_SHA
      --set ingress.host=staging.myapp.com
      --wait
  environment:
    name: staging
    url: https://staging.myapp.com
  only:
    - develop

# 프로덕션 배포 (수동 승인 필요)
deploy-production:
  stage: deploy-production
  image: alpine/helm:3.8.0
  script:
    - helm upgrade --install myapp-prod ./helm-chart
      --namespace production
      --set image.tag=$CI_COMMIT_SHA
      --set ingress.host=myapp.com
      --set replicas=3
      --wait
  environment:
    name: production
    url: https://myapp.com
  when: manual
  only:
    - main
```

#### Jenkins 파이프라인 구성

```groovy
// Jenkinsfile - Jenkins 선언적 파이프라인
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        APP_NAME = 'myapp'
        KUBECONFIG = credentials('kubeconfig-prod')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Build & Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm ci'
                        sh 'npm run test:unit'
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results.xml'
                            publishCoverage adapters: [
                                coberturaAdapter('coverage/cobertura-coverage.xml')
                            ]
                        }
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        sh 'npm audit'
                        sh 'docker run --rm -v $(pwd):/app clair-scanner --ip $(hostname -i) myapp:latest'
                    }
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${APP_NAME}:${env.GIT_COMMIT_SHORT}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh """
                    helm upgrade --install ${APP_NAME}-staging ./helm-chart \\
                        --namespace staging \\
                        --set image.tag=${env.GIT_COMMIT_SHORT} \\
                        --wait --timeout=600s
                """
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh 'npm run test:integration -- --env=staging'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
                ok "Deploy"
                parameters {
                    choice(name: 'DEPLOYMENT_STRATEGY', choices: ['rolling', 'blue-green', 'canary'])
                }
            }
            steps {
                script {
                    switch(params.DEPLOYMENT_STRATEGY) {
                        case 'blue-green':
                            sh './scripts/blue-green-deploy.sh ${env.GIT_COMMIT_SHORT}'
                            break
                        case 'canary':
                            sh './scripts/canary-deploy.sh ${env.GIT_COMMIT_SHORT}'
                            break
                        default:
                            sh """
                                helm upgrade --install ${APP_NAME}-prod ./helm-chart \\
                                    --namespace production \\
                                    --set image.tag=${env.GIT_COMMIT_SHORT} \\
                                    --wait --timeout=600s
                            """
                    }
                }
            }
        }
    }
    
    post {
        failure {
            script {
                slackSend(
                    channel: '#deployments',
                    color: 'danger',
                    message: "❌ Deployment failed: ${env.JOB_NAME} ${env.BUILD_NUMBER}"
                )
            }
        }
        success {
            script {
                slackSend(
                    channel: '#deployments',
                    color: 'good',
                    message: "✅ Deployment successful: ${env.JOB_NAME} ${env.BUILD_NUMBER}"
                )
            }
        }
    }
}
```

### 3-1-2 배포 자동화 도구 선택 | Deployment Automation Tools

#### 🛠️ 도구 비교 매트릭스

```bash
# CI/CD 플랫폼 비교
                GitLab CI    Jenkins     GitHub Actions    Azure DevOps    AWS CodePipeline
설정 복잡도        낮음        중간           낮음            중간              높음
클라우드 통합      우수        보통           우수            우수              우수  
온프레미스 지원    우수        우수           제한적          우수              불가
비용              무료/유료    오픈소스       무료/유료        유료              사용량기반
확장성            우수        우수           우수            우수              우수
커뮤니티 지원      우수        최고           우수            보통              보통

# 컨테이너 오케스트레이션
                Kubernetes   Docker Swarm   OpenShift      Nomad           ECS/Fargate
학습 곡선          가파름       완만          중간           완만             완만
기능 풍부함        최고        기본           우수           기본             기본
클라우드 지원      우수        보통           우수           보통             AWS전용
커뮤니티          최고        감소           보통           성장             AWS생태계
```

#### GitHub Actions 워크플로우

```yaml
# .github/workflows/deploy.yml
name: Deploy Application

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16.x, 18.x]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run tests
      run: |
        npm run test:unit
        npm run test:integration
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage/lcov.info

  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.image.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
    
    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - name: Deploy to staging
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/staging/deployment.yaml
          k8s/staging/service.yaml
        images: ${{ needs.build.outputs.image }}@${{ needs.build.outputs.digest }}
        namespace: staging

  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Deploy to production
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/production/deployment.yaml
          k8s/production/service.yaml
        images: ${{ needs.build.outputs.image }}@${{ needs.build.outputs.digest }}
        namespace: production
        strategy: blue-green
```

#### ArgoCD GitOps 워크플로우

```yaml
# argocd-app.yaml - ArgoCD Application 정의
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io

spec:
  project: default
  
  source:
    repoURL: https://github.com/company/myapp-k8s-manifests
    targetRevision: HEAD
    path: overlays/production
    
  destination:
    server: https://kubernetes.default.svc
    namespace: production
    
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
      
  revisionHistoryLimit: 10
  
  # Health checks
  ignoreDifferences:
  - group: apps
    kind: Deployment
    jsonPointers:
    - /spec/replicas

---
# kustomization.yaml - Kustomize 설정
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: production

resources:
- ../../base

patchesStrategicMerge:
- deployment-patch.yaml
- service-patch.yaml

images:
- name: myapp
  newTag: v1.2.3

replicas:
- name: myapp-deployment
  count: 3

configMapGenerator:
- name: myapp-config
  files:
  - config/production.yaml

secretGenerator:
- name: myapp-secrets
  files:
  - secrets/database.env
```

## 3-2 배포 전략 | Deployment Strategies

### 3-2-1 가장 간단한 배포 | Recreate Deployment

Recreate 배포는 기존 버전을 완전히 중단한 후 새 버전을 시작하는 가장 단순한 배포 방식입니다.

#### 📋 Recreate 배포 특징

```bash
# 장점
- 구현이 간단함
- 리소스 사용량이 적음
- 버전 간 충돌 없음
- 데이터베이스 마이그레이션에 유리

# 단점  
- 서비스 중단 시간 발생
- 사용자 경험 저하
- 롤백 시간이 김
```

#### Kubernetes Recreate 배포

```yaml
# recreate-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-recreate
  namespace: default

spec:
  replicas: 3
  strategy:
    type: Recreate  # Recreate 전략 사용
    
  selector:
    matchLabels:
      app: myapp
      
  template:
    metadata:
      labels:
        app: myapp
        version: v2.0.0
        
    spec:
      containers:
      - name: myapp
        image: myapp:v2.0.0
        ports:
        - containerPort: 8080
        
        # 헬스체크 설정
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

#### Recreate 배포 스크립트

```bash
#!/bin/bash
# recreate-deploy.sh

set -e

APP_NAME="myapp"
NEW_VERSION="$1"
NAMESPACE="production"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new-version>"
    exit 1
fi

echo "=== Recreate Deployment Started ==="
echo "App: $APP_NAME"
echo "New Version: $NEW_VERSION"
echo "Namespace: $NAMESPACE"
echo "Time: $(date)"

# 1. 현재 배포 상태 확인
echo "Current deployment status:"
kubectl get deployment $APP_NAME -n $NAMESPACE

# 2. 헬스체크 엔드포인트 확인
echo "Checking current health status..."
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

# 3. 트래픽 중단 (로드밸런서에서 제거)
echo "Removing from load balancer..."
kubectl patch service $APP_NAME -n $NAMESPACE -p '{"spec":{"selector":{"app":"maintenance"}}}'

# 4. 기존 파드 종료 대기
echo "Stopping existing pods..."
kubectl delete pods -n $NAMESPACE -l app=$APP_NAME --grace-period=30

# 5. 새 버전으로 업데이트
echo "Updating to new version: $NEW_VERSION"
kubectl set image deployment/$APP_NAME -n $NAMESPACE container=$APP_NAME:$NEW_VERSION

# 6. 배포 완료 대기
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=300s

# 7. 헬스체크 확인
echo "Performing health checks..."
sleep 30

READY_REPLICAS=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.replicas}')

if [ "$READY_REPLICAS" = "$DESIRED_REPLICAS" ]; then
    echo "Health check passed: $READY_REPLICAS/$DESIRED_REPLICAS pods ready"
else
    echo "Health check failed: $READY_REPLICAS/$DESIRED_REPLICAS pods ready"
    exit 1
fi

# 8. 트래픽 복원
echo "Restoring traffic..."
kubectl patch service $APP_NAME -n $NAMESPACE -p '{"spec":{"selector":{"app":"'$APP_NAME'"}}}'

echo "=== Recreate Deployment Completed Successfully ==="
```

### 3-2-2 롤링 업데이트 | Rolling Update

롤링 업데이트는 서비스 중단 없이 점진적으로 새 버전을 배포하는 방식입니다.

#### 🔄 롤링 업데이트 원리

```yaml
# rolling-update-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-rolling

spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1      # 최대 1개 파드 중단 허용
      maxSurge: 2           # 최대 2개 추가 파드 생성 허용
      
  selector:
    matchLabels:
      app: myapp
      
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:v2.1.0
        ports:
        - containerPort: 8080
        
        # 점진적 종료를 위한 설정
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
              
        # 빠른 시작을 위한 헬스체크
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 2
          timeoutSeconds: 1
          successThreshold: 1
          failureThreshold: 3
          
        livenessProbe:
          httpGet:
            path: /health  
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### 고급 롤링 업데이트 스크립트

```bash
#!/bin/bash
# advanced-rolling-update.sh

set -e

APP_NAME="myapp"
NEW_VERSION="$1"
NAMESPACE="production"
MAX_ROLLBACK_REVISION=5

# 컬러 출력을 위한 함수
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 사전 점검
pre_deployment_checks() {
    log "Performing pre-deployment checks..."
    
    # 1. 새 이미지 존재 확인
    if ! docker pull $APP_NAME:$NEW_VERSION > /dev/null 2>&1; then
        error "Image $APP_NAME:$NEW_VERSION not found"
        exit 1
    fi
    
    # 2. 클러스터 상태 확인
    if ! kubectl cluster-info > /dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # 3. 네임스페이스 존재 확인
    if ! kubectl get namespace $NAMESPACE > /dev/null 2>&1; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # 4. 현재 배포 상태 확인
    CURRENT_REPLICAS=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    DESIRED_REPLICAS=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    if [ "$CURRENT_REPLICAS" != "$DESIRED_REPLICAS" ]; then
        warn "Current deployment is not healthy: $CURRENT_REPLICAS/$DESIRED_REPLICAS pods ready"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log "Pre-deployment checks completed successfully"
}

# 배포 실행
perform_rolling_update() {
    log "Starting rolling update to version $NEW_VERSION"
    
    # 현재 이미지 태그 저장 (롤백용)
    CURRENT_IMAGE=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
    
    # 배포 히스토리 기록
    kubectl annotate deployment $APP_NAME -n $NAMESPACE \
        deployment.kubernetes.io/revision-history="Previous: $CURRENT_IMAGE, New: $APP_NAME:$NEW_VERSION, Time: $(date)"
    
    # 새 버전 배포
    kubectl set image deployment/$APP_NAME -n $NAMESPACE $APP_NAME=$APP_NAME:$NEW_VERSION
    
    # 배포 상태 모니터링
    log "Monitoring rollout progress..."
    
    # 타임아웃과 함께 롤아웃 상태 확인
    if timeout 600 kubectl rollout status deployment/$APP_NAME -n $NAMESPACE; then
        log "Rolling update completed successfully"
    else
        error "Rolling update timed out"
        return 1
    fi
}

# 배포 후 검증
post_deployment_validation() {
    log "Performing post-deployment validation..."
    
    # 1. 파드 상태 확인
    sleep 30
    
    READY_REPLICAS=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
    DESIRED_REPLICAS=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.replicas}')
    
    if [ "$READY_REPLICAS" != "$DESIRED_REPLICAS" ]; then
        error "Pod readiness check failed: $READY_REPLICAS/$DESIRED_REPLICAS"
        return 1
    fi
    
    # 2. 애플리케이션 헬스체크
    SERVICE_IP=$(kubectl get service $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    
    for i in {1..10}; do
        if curl -f -s http://$SERVICE_IP:8080/health > /dev/null; then
            log "Application health check passed"
            break
        else
            warn "Health check attempt $i failed, retrying..."
            sleep 5
        fi
        
        if [ $i -eq 10 ]; then
            error "Application health check failed after 10 attempts"
            return 1
        fi
    done
    
    # 3. 메모리 및 CPU 사용률 확인
    log "Checking resource usage..."
    kubectl top pods -n $NAMESPACE -l app=$APP_NAME
    
    log "Post-deployment validation completed successfully"
}

# 롤백 함수
rollback_deployment() {
    local rollback_reason="$1"
    
    error "Rolling back deployment due to: $rollback_reason"
    
    kubectl rollout undo deployment/$APP_NAME -n $NAMESPACE
    
    log "Waiting for rollback to complete..."
    kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=300s
    
    error "Rollback completed"
}

# 메인 실행 로직
main() {
    if [ -z "$NEW_VERSION" ]; then
        echo "Usage: $0 <new-version>"
        exit 1
    fi
    
    log "=== Rolling Update Started ==="
    log "App: $APP_NAME"
    log "New Version: $NEW_VERSION"
    log "Namespace: $NAMESPACE"
    
    # 단계별 실행
    if ! pre_deployment_checks; then
        error "Pre-deployment checks failed"
        exit 1
    fi
    
    if ! perform_rolling_update; then
        rollback_deployment "Rolling update failed"
        exit 1
    fi
    
    if ! post_deployment_validation; then
        rollback_deployment "Post-deployment validation failed"
        exit 1
    fi
    
    log "=== Rolling Update Completed Successfully ==="
}

main "$@"
```

### 3-2-3 블루그린 배포 | Blue-Green Deployment

블루그린 배포는 두 개의 동일한 환경을 운영하여 무중단 배포와 즉시 롤백을 가능하게 하는 전략입니다.

#### 🔵🟢 블루그린 배포 개념

```bash
# 블루그린 배포 환경 구성
Production Traffic → Load Balancer → Blue Environment (현재 버전)
                                  → Green Environment (새 버전, 대기)

# 배포 과정
1. Green 환경에 새 버전 배포
2. Green 환경 테스트 및 검증  
3. 로드밸런서 트래픽을 Blue → Green으로 전환
4. Blue 환경을 새로운 Green으로 준비
```

#### Kubernetes 블루그린 배포

```yaml
# blue-green-service.yaml - 트래픽 라우팅 서비스
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
  namespace: production

spec:
  selector:
    app: myapp
    version: blue    # blue 또는 green으로 전환
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  type: LoadBalancer

---
# blue-deployment.yaml - Blue 환경
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-blue
  namespace: production
  labels:
    app: myapp
    version: blue

spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
      
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: myapp
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

---
# green-deployment.yaml - Green 환경
apiVersion: apps/v1  
kind: Deployment
metadata:
  name: myapp-green
  namespace: production
  labels:
    app: myapp
    version: green

spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
      
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
      - name: myapp
        image: myapp:v2.0.0  # 새 버전
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 3-2-4 카나리 배포 | Canary Deployment

카나리 배포는 새 버전을 일부 트래픽에만 점진적으로 노출하여 리스크를 최소화하는 배포 전략입니다.

#### 🐤 카나리 배포 구현

```yaml
# canary-istio.yaml - Istio를 이용한 카나리 배포
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: myapp-canary
  namespace: production

spec:
  hosts:
  - myapp.company.com
  
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: myapp-service
        subset: v2
      weight: 100
      
  - route:
    - destination:
        host: myapp-service
        subset: v1
      weight: 90    # 90% 기존 버전
    - destination:
        host: myapp-service  
        subset: v2
      weight: 10    # 10% 새 버전

---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: myapp-destination
  namespace: production

spec:
  host: myapp-service
  subsets:
  - name: v1
    labels:
      version: v1.0.0
  - name: v2
    labels:
      version: v2.0.0
```

#### 점진적 카나리 배포 스크립트

```bash
#!/bin/bash
# progressive-canary.sh

set -e

APP_NAME="myapp"
NEW_VERSION="$1"
NAMESPACE="production"

# 트래픽 분배 단계 (%)
CANARY_STAGES=(5 10 25 50 75 100)

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# 메트릭 수집 함수
collect_metrics() {
    local version=$1
    local duration=$2
    
    log "Collecting metrics for version $version over $duration seconds"
    
    # Prometheus 메트릭 쿼리
    local error_rate=$(curl -s "http://prometheus:9090/api/v1/query" \
        --data-urlencode "query=rate(http_requests_total{version=\"$version\",status=~\"5..\"}[${duration}s]) / rate(http_requests_total{version=\"$version\"}[${duration}s]) * 100")
    
    local response_time=$(curl -s "http://prometheus:9090/api/v1/query" \
        --data-urlencode "query=histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{version=\"$version\"}[${duration}s]))")
    
    local cpu_usage=$(curl -s "http://prometheus:9090/api/v1/query" \
        --data-urlencode "query=avg(rate(container_cpu_usage_seconds_total{pod=~\"$APP_NAME-.*\",container=\"$APP_NAME\"}[${duration}s])) by (version)")
    
    # JSON 파싱 (jq 필요)
    local error_rate_value=$(echo $error_rate | jq -r '.data.result[0].value[1] // "0"')
    local response_time_value=$(echo $response_time | jq -r '.data.result[0].value[1] // "0"')
    local cpu_usage_value=$(echo $cpu_usage | jq -r '.data.result[0].value[1] // "0"')
    
    echo "$error_rate_value,$response_time_value,$cpu_usage_value"
}

# SLI/SLO 체크
check_slo() {
    local metrics=$1
    local version=$2
    
    IFS=',' read -r error_rate response_time cpu_usage <<< "$metrics"
    
    log "SLO Check for $version - Error Rate: $error_rate%, Response Time: ${response_time}s, CPU: $cpu_usage"
    
    # SLO 임계값
    local max_error_rate=1.0
    local max_response_time=0.5
    local max_cpu_usage=0.8
    
    if (( $(echo "$error_rate > $max_error_rate" | bc -l) )); then
        log "❌ Error rate SLO violated: $error_rate% > $max_error_rate%"
        return 1
    fi
    
    if (( $(echo "$response_time > $max_response_time" | bc -l) )); then
        log "❌ Response time SLO violated: ${response_time}s > ${max_response_time}s"
        return 1
    fi
    
    if (( $(echo "$cpu_usage > $max_cpu_usage" | bc -l) )); then
        log "❌ CPU usage SLO violated: $cpu_usage > $max_cpu_usage"
        return 1
    fi
    
    log "✅ All SLOs passed for version $version"
    return 0
}

# 트래픽 가중치 업데이트
update_traffic_weight() {
    local canary_weight=$1
    local stable_weight=$((100 - canary_weight))
    
    log "Updating traffic weights - Stable: $stable_weight%, Canary: $canary_weight%"
    
    kubectl patch virtualservice $APP_NAME-canary -n $NAMESPACE --type='json' -p="[
        {\"op\": \"replace\", \"path\": \"/spec/http/1/route/0/weight\", \"value\": $stable_weight},
        {\"op\": \"replace\", \"path\": \"/spec/http/1/route/1/weight\", \"value\": $canary_weight}
    ]"
}

# 카나리 배포 실행
perform_canary_deployment() {
    log "Starting canary deployment for version $NEW_VERSION"
    
    # 1. 카나리 버전 배포 (0% 트래픽)
    kubectl set image deployment/$APP_NAME-canary -n $NAMESPACE $APP_NAME=$APP_NAME:$NEW_VERSION
    kubectl rollout status deployment/$APP_NAME-canary -n $NAMESPACE --timeout=300s
    
    # 2. 단계별 트래픽 증가
    for stage in "${CANARY_STAGES[@]}"; do
        log "=== Canary Stage: $stage% traffic ==="
        
        # 트래픽 가중치 업데이트
        update_traffic_weight $stage
        
        # 안정화 대기
        log "Waiting for traffic stabilization..."
        sleep 60
        
        # 메트릭 수집 및 분석 (5분간)
        local canary_metrics=$(collect_metrics "v$NEW_VERSION" 300)
        local stable_metrics=$(collect_metrics "v$(get_current_version)" 300)
        
        # SLO 체크
        if ! check_slo "$canary_metrics" "canary"; then
            log "❌ Canary version failed SLO check, rolling back..."
            rollback_canary
            return 1
        fi
        
        # 비교 분석
        log "Comparing canary vs stable metrics..."
        if ! compare_versions "$canary_metrics" "$stable_metrics"; then
            log "❌ Canary version performance degraded, rolling back..."
            rollback_canary  
            return 1
        fi
        
        log "✅ Stage $stage% completed successfully"
        
        # 마지막 단계가 아니면 대기
        if [ "$stage" != "100" ]; then
            log "Waiting before next stage..."
            sleep 300  # 5분 대기
        fi
    done
    
    log "🎉 Canary deployment completed successfully"
}

# 롤백 함수
rollback_canary() {
    log "Rolling back canary deployment..."
    update_traffic_weight 0
    log "Traffic rolled back to stable version"
}

main() {
    if [ -z "$NEW_VERSION" ]; then
        echo "Usage: $0 <new-version>"
        exit 1
    fi
    
    log "=== Progressive Canary Deployment Started ==="
    log "App: $APP_NAME, New Version: $NEW_VERSION"
    
    if ! perform_canary_deployment; then
        log "❌ Canary deployment failed"
        exit 1
    fi
    
    log "=== Canary Deployment Completed Successfully ==="
}

main "$@"
```

## 실습: AWS EC2 기반의 블루그린 배포

### 🏗️ AWS 인프라 구성

```bash
# 1. VPC 및 서브넷 생성
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=BlueGreen-VPC}]'

VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=BlueGreen-VPC" --query 'Vpcs[0].VpcId' --output text)

# 퍼블릭 서브넷 생성 (2개 AZ)
aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.1.0/24 --availability-zone us-west-2a --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-1}]'

aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.2.0/24 --availability-zone us-west-2b --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-2}]'

# 프라이빗 서브넷 생성 (2개 AZ)
aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.11.0/24 --availability-zone us-west-2a --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-1}]'

aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.12.0/24 --availability-zone us-west-2b --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-2}]'

# 인터넷 게이트웨이 생성 및 연결
aws ec2 create-internet-gateway --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=BlueGreen-IGW}]'

IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=tag:Name,Values=BlueGreen-IGW" --query 'InternetGateways[0].InternetGatewayId' --output text)

aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID
```

#### Application Load Balancer 구성

```json
{
  "LoadBalancerName": "myapp-bluegreen-alb",
  "Scheme": "internet-facing",
  "Type": "application",
  "IpAddressType": "ipv4",
  "Subnets": [
    "subnet-12345678",
    "subnet-87654321"
  ],
  "SecurityGroups": [
    "sg-12345678"
  ],
  "Tags": [
    {
      "Key": "Name", 
      "Value": "MyApp-BlueGreen-ALB"
    },
    {
      "Key": "Environment",
      "Value": "Production"
    }
  ]
}
```

#### 블루그린 배포 자동화 스크립트

```bash
#!/bin/bash
# aws-bluegreen-deploy.sh

set -e

# 설정 변수
APP_NAME="myapp"
NEW_VERSION="$1"
AWS_REGION="us-west-2"
VPC_ID="vpc-12345678"
ALB_ARN="arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/myapp-alb/1234567890123456"
BLUE_TG_ARN="arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/myapp-blue/1234567890123456"
GREEN_TG_ARN="arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/myapp-green/1234567890123456"
LISTENER_ARN="arn:aws:elasticloadbalancing:us-west-2:123456789012:listener/app/myapp-alb/1234567890123456/1234567890123456"

# 현재 활성 환경 확인
get_active_environment() {
    local listener_rules=$(aws elbv2 describe-rules --listener-arn $LISTENER_ARN --region $AWS_REGION)
    local active_target_group=$(echo $listener_rules | jq -r '.Rules[] | select(.Priority == "100") | .Actions[0].TargetGroupArn')
    
    if [[ "$active_target_group" == "$BLUE_TG_ARN" ]]; then
        echo "blue"
    else
        echo "green"  
    fi
}

# Auto Scaling Group에 새 Launch Template 적용
update_asg_launch_template() {
    local environment=$1
    local new_ami=$2
    
    local asg_name="${APP_NAME}-${environment}-asg"
    local lt_name="${APP_NAME}-${environment}-lt"
    
    # 새 Launch Template 버전 생성
    aws ec2 create-launch-template-version \
        --launch-template-name $lt_name \
        --source-version '$Latest' \
        --launch-template-data "{\"ImageId\":\"$new_ami\"}" \
        --region $AWS_REGION
    
    # ASG에 새 Launch Template 버전 적용
    local latest_version=$(aws ec2 describe-launch-template-versions \
        --launch-template-name $lt_name \
        --region $AWS_REGION \
        --query 'LaunchTemplateVersions[0].VersionNumber' \
        --output text)
    
    aws autoscaling update-auto-scaling-group \
        --auto-scaling-group-name $asg_name \
        --launch-template LaunchTemplateName=$lt_name,Version=$latest_version \
        --region $AWS_REGION
    
    log "Updated ASG $asg_name with Launch Template version $latest_version"
}

# 인스턴스 교체 (Rolling Replacement)
perform_instance_refresh() {
    local environment=$1
    local asg_name="${APP_NAME}-${environment}-asg"
    
    log "Starting instance refresh for $asg_name"
    
    local refresh_id=$(aws autoscaling start-instance-refresh \
        --auto-scaling-group-name $asg_name \
        --preferences '{
            "InstanceWarmup": 300,
            "MinHealthyPercentage": 50,
            "CheckpointPercentages": [20, 50, 100],
            "CheckpointDelay": 600
        }' \
        --region $AWS_REGION \
        --query 'InstanceRefreshId' \
        --output text)
    
    log "Instance refresh started with ID: $refresh_id"
    
    # 인스턴스 교체 완료 대기
    while true; do
        local status=$(aws autoscaling describe-instance-refreshes \
            --auto-scaling-group-name $asg_name \
            --instance-refresh-ids $refresh_id \
            --region $AWS_REGION \
            --query 'InstanceRefreshes[0].Status' \
            --output text)
        
        log "Instance refresh status: $status"
        
        case $status in
            "Successful")
                log "✅ Instance refresh completed successfully"
                break
                ;;
            "Failed"|"Cancelled")
                error "❌ Instance refresh failed with status: $status"
                return 1
                ;;
            "InProgress"|"Pending")
                log "Instance refresh in progress, waiting..."
                sleep 60
                ;;
        esac
    done
}

# 헬스체크 수행
health_check_environment() {
    local environment=$1
    local target_group_arn
    
    if [ "$environment" = "blue" ]; then
        target_group_arn=$BLUE_TG_ARN
    else
        target_group_arn=$GREEN_TG_ARN
    fi
    
    log "Performing health check for $environment environment"
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        local healthy_targets=$(aws elbv2 describe-target-health \
            --target-group-arn $target_group_arn \
            --region $AWS_REGION \
            --query 'TargetHealthDescriptions[?TargetHealth.State==`healthy`] | length(@)')
        
        local total_targets=$(aws elbv2 describe-target-health \
            --target-group-arn $target_group_arn \
            --region $AWS_REGION \
            --query 'TargetHealthDescriptions | length(@)')
        
        log "Health check attempt $attempt: $healthy_targets/$total_targets targets healthy"
        
        if [ "$healthy_targets" -gt 0 ] && [ "$healthy_targets" -eq "$total_targets" ]; then
            log "✅ All targets are healthy in $environment environment"
            return 0
        fi
        
        sleep 30
        ((attempt++))
    done
    
    error "❌ Health check failed for $environment environment after $max_attempts attempts"
    return 1
}

# 트래픽 전환
switch_traffic() {
    local target_environment=$1
    local target_group_arn
    
    if [ "$target_environment" = "blue" ]; then
        target_group_arn=$BLUE_TG_ARN
    else
        target_group_arn=$GREEN_TG_ARN
    fi
    
    log "Switching traffic to $target_environment environment"
    
    aws elbv2 modify-rule \
        --rule-arn $(aws elbv2 describe-rules --listener-arn $LISTENER_ARN --region $AWS_REGION --query 'Rules[?Priority==`100`].RuleArn' --output text) \
        --actions Type=forward,TargetGroupArn=$target_group_arn \
        --region $AWS_REGION
    
    log "✅ Traffic switched to $target_environment environment"
}

# 배포 검증
validate_deployment() {
    local environment=$1
    
    log "Validating deployment in $environment environment"
    
    # ALB DNS 이름 가져오기
    local alb_dns=$(aws elbv2 describe-load-balancers \
        --load-balancer-arns $ALB_ARN \
        --region $AWS_REGION \
        --query 'LoadBalancers[0].DNSName' \
        --output text)
    
    # 애플리케이션 버전 확인
    local app_version=$(curl -s -f http://$alb_dns/version 2>/dev/null || echo "unknown")
    
    if [ "$app_version" = "$NEW_VERSION" ]; then
        log "✅ Application version validated: $app_version"
    else
        error "❌ Version mismatch. Expected: $NEW_VERSION, Got: $app_version"
        return 1
    fi
    
    # 기본 기능 테스트
    local health_status=$(curl -s -o /dev/null -w "%{http_code}" http://$alb_dns/health)
    
    if [ "$health_status" = "200" ]; then
        log "✅ Application health check passed"
    else
        error "❌ Application health check failed. Status code: $health_status"
        return 1
    fi
    
    log "✅ Deployment validation completed successfully"
}

# 롤백 수행
rollback_deployment() {
    local current_environment=$(get_active_environment)
    local rollback_environment
    
    if [ "$current_environment" = "blue" ]; then
        rollback_environment="green"
    else
        rollback_environment="blue"
    fi
    
    log "Rolling back from $current_environment to $rollback_environment"
    
    switch_traffic $rollback_environment
    
    log "✅ Rollback completed"
}

# 메인 배포 로직
main() {
    if [ -z "$NEW_VERSION" ]; then
        echo "Usage: $0 <new-version>"
        exit 1
    fi
    
    log "=== AWS EC2 Blue-Green Deployment Started ==="
    log "Application: $APP_NAME"
    log "New Version: $NEW_VERSION"
    log "Region: $AWS_REGION"
    
    # 1. 현재 활성 환경 확인
    local active_env=$(get_active_environment)
    local inactive_env
    
    if [ "$active_env" = "blue" ]; then
        inactive_env="green"
    else
        inactive_env="blue"
    fi
    
    log "Current active environment: $active_env"
    log "Deploying to inactive environment: $inactive_env"
    
    # 2. 새 AMI ID 가져오기 (이미 빌드된 AMI 사용)
    local new_ami=$(aws ec2 describe-images \
        --filters "Name=tag:Version,Values=$NEW_VERSION" "Name=tag:Application,Values=$APP_NAME" \
        --region $AWS_REGION \
        --query 'Images[0].ImageId' \
        --output text)
    
    if [ "$new_ami" = "None" ] || [ -z "$new_ami" ]; then
        error "❌ AMI for version $NEW_VERSION not found"
        exit 1
    fi
    
    log "Using AMI: $new_ami for version $NEW_VERSION"
    
    # 3. 비활성 환경에 새 버전 배포
    if ! update_asg_launch_template $inactive_env $new_ami; then
        error "❌ Failed to update launch template"
        exit 1
    fi
    
    if ! perform_instance_refresh $inactive_env; then
        error "❌ Failed to refresh instances"
        exit 1
    fi
    
    # 4. 비활성 환경 헬스체크
    if ! health_check_environment $inactive_env; then
        error "❌ Health check failed for $inactive_env environment"
        exit 1
    fi
    
    # 5. 트래픽 전환
    if ! switch_traffic $inactive_env; then
        error "❌ Failed to switch traffic"
        rollback_deployment
        exit 1
    fi
    
    # 6. 배포 검증
    sleep 60  # 트래픽 안정화 대기
    
    if ! validate_deployment $inactive_env; then
        error "❌ Deployment validation failed"
        rollback_deployment
        exit 1
    fi
    
    log "🎉 Blue-Green deployment completed successfully!"
    log "New active environment: $inactive_env"
    log "Previous environment ($active_env) is now inactive and ready for next deployment"
}

# 유틸리티 함수
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# 스크립트 실행
main "$@"
```

## 정리 및 모범 사례

### 📊 배포 전략 선택 가이드

```bash
# 배포 전략 선택 매트릭스
상황                     | 권장 전략        | 이유
====================|===============|==============================
개발/테스트 환경         | Recreate      | 단순함, 리소스 절약
스테이지 환경           | Rolling       | 프로덕션과 유사한 환경
프로덕션 (위험 높음)     | Blue-Green    | 즉시 롤백, 완전한 검증
프로덕션 (위험 보통)     | Canary        | 점진적 검증, 리스크 분산
프로덕션 (위험 낮음)     | Rolling       | 효율적, 리소스 최적화
레거시 시스템           | Blue-Green    | 안전성 우선
마이크로서비스          | Canary        | 서비스별 독립 배포
```

### 🔧 모범 사례

#### 1. 배포 준비
```bash
# 배포 전 체크리스트
✅ 백업 완료
✅ 롤백 계획 수립
✅ 모니터링 준비
✅ 팀 커뮤니케이션
✅ 헬스체크 엔드포인트 구현
✅ 설정 검증
```

#### 2. 모니터링 및 알람
```bash
# 핵심 메트릭
- 에러율 (< 1%)
- 응답 시간 (< 500ms)
- 처리량 (baseline 대비)
- 리소스 사용률 (< 80%)
- 가용성 (> 99.9%)
```

#### 3. 자동화 원칙
```bash
# 모든 것을 자동화
1. 빌드 프로세스
2. 테스트 실행
3. 배포 프로세스
4. 헬스체크
5. 롤백 절차
6. 알림 및 로깅
```

현대적인 배포 전략을 마스터하셨나요? 안정적이고 효율적인 배포로 서비스 품질을 한 단계 높여보세요! 🚀
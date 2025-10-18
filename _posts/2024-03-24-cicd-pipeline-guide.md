---
layout: post
title: "DevOps CI/CD 파이프라인 구축 가이드 | DevOps CI/CD Pipeline Implementation Guide"
date: 2024-03-24 16:30:00 +0900
categories: [DevOps, CI/CD]
tags: [devops, cicd, jenkins, gitlab, github-actions, automation, deployment]
---

효과적인 DevOps CI/CD 파이프라인 구축을 위한 완전 가이드입니다. 다양한 도구별 구현 방법을 알아보겠습니다.

## CI/CD란? | What is CI/CD?

### 지속적 통합 (Continuous Integration)
- **코드 변경사항 자동 빌드**
- **자동화된 테스트 실행**
- **코드 품질 검증**

### 지속적 배포 (Continuous Deployment)
- **자동화된 배포 프로세스**
- **무중단 배포**
- **롤백 기능**

## Jenkins 파이프라인 | Jenkins Pipeline

### Jenkinsfile 예제
```groovy
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/your-repo/app.git'
            }
        }
        
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
        
        stage('Test') {
            steps {
                sh 'npm test'
                publishTestResults testResultsPattern: 'test-results.xml'
            }
        }
        
        stage('Security Scan') {
            steps {
                sh 'npm audit'
                sh 'sonar-scanner'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("myapp:${env.BUILD_NUMBER}")
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh 'kubectl apply -f k8s-staging/'
                sh 'kubectl set image deployment/myapp myapp=myapp:${env.BUILD_NUMBER}'
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh 'npm run test:integration'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input 'Deploy to production?'
                sh 'kubectl apply -f k8s-production/'
                sh 'kubectl set image deployment/myapp myapp=myapp:${env.BUILD_NUMBER}'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "dev-team@company.com"
            )
        }
    }
}
```

## GitHub Actions | GitHub Actions

### Workflow 예제
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run tests
      run: npm test
    
    - name: Run security audit
      run: npm audit --audit-level high

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.DOCKER_REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # kubectl 명령어로 배포

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # kubectl 명령어로 프로덕션 배포
```

## GitLab CI/CD | GitLab CI/CD

### .gitlab-ci.yml 예제
```yaml
stages:
  - test
  - build
  - deploy-staging
  - deploy-production

variables:
  DOCKER_REGISTRY: $CI_REGISTRY
  IMAGE_NAME: $CI_PROJECT_PATH
  DOCKER_DRIVER: overlay2

before_script:
  - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY

test:
  stage: test
  image: node:18
  script:
    - npm ci
    - npm test
    - npm audit --audit-level high
  coverage: '/Coverage: \d+\.\d+%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

build:
  stage: build
  script:
    - docker build -t $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA .
    - docker push $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA
  only:
    - main
    - develop

deploy-staging:
  stage: deploy-staging
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/myapp myapp=$DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA
    - kubectl rollout status deployment/myapp
  environment:
    name: staging
    url: https://staging.myapp.com
  only:
    - develop

deploy-production:
  stage: deploy-production
  script:
    - kubectl config use-context production
    - kubectl set image deployment/myapp myapp=$DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA
    - kubectl rollout status deployment/myapp
  environment:
    name: production
    url: https://myapp.com
  when: manual
  only:
    - main
```

## Terraform을 활용한 Infrastructure as Code

### main.tf 예제
```hcl
# VPC 구성
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "main-vpc"
  }
}

# EKS 클러스터
resource "aws_eks_cluster" "main" {
  name     = "main-cluster"
  role_arn = aws_iam_role.cluster.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
  ]
}

# Jenkins 배포를 위한 Helm
resource "helm_release" "jenkins" {
  name       = "jenkins"
  repository = "https://charts.jenkins.io"
  chart      = "jenkins"
  namespace  = "jenkins"

  values = [
    file("${path.module}/jenkins-values.yaml")
  ]
}
```

## 모니터링 및 알림 | Monitoring & Alerting

### Prometheus 설정
```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

rule_files:
  - "alert-rules.yml"

scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 알림 규칙
```yaml
# alert-rules.yml
groups:
- name: deployment
  rules:
  - alert: DeploymentFailed
    expr: increase(jenkins_builds_failed_total[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Deployment failed"
      description: "Jenkins build {{ $labels.job }} has failed"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
```

## 보안 모범 사례 | Security Best Practices

### 1. 시크릿 관리
```yaml
# Kubernetes Secret
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: <base64-encoded-value>
  api-key: <base64-encoded-value>
```

### 2. RBAC 설정
```yaml
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: jenkins-deploy
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "create", "update", "patch"]
```

### 3. 이미지 스캐닝
```bash
# Trivy를 이용한 이미지 취약점 스캔
trivy image --severity HIGH,CRITICAL myapp:latest
```

다음 포스트에서는 Kubernetes 기반 무중단 배포 전략을 다뤄보겠습니다!
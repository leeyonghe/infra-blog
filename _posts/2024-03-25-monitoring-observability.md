---
layout: post
title: "모니터링 & 관찰성 완전 가이드 | Complete Monitoring & Observability Guide"
date: 2024-03-25 11:00:00 +0900
categories: [Monitoring, Observability]
tags: [prometheus, grafana, elk, observability, monitoring, alerting, logging, metrics]
---

현대적인 인프라 모니터링과 관찰성을 위한 완전 가이드입니다. Prometheus, Grafana, ELK Stack 등을 활용한 종합적인 모니터링 환경을 구축해보겠습니다.

## 관찰성의 3가지 기둥 | Three Pillars of Observability

### 1. 메트릭 (Metrics) 📊
- **시스템 성능 지표**
- **비즈니스 메트릭**
- **SLA/SLI 추적**

### 2. 로그 (Logs) 📝
- **구조화된 로깅**
- **중앙집중식 로그 관리**
- **로그 분석 및 검색**

### 3. 트레이스 (Traces) 🔍
- **분산 추적**
- **요청 흐름 추적**
- **성능 병목 지점 식별**

## Prometheus + Grafana 모니터링 스택

### Prometheus 설치 및 설정
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "/etc/prometheus/rules/*.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
      
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        relabel_configs:
          - source_labels: [__address__]
            regex: '(.*):10250'
            target_label: __address__
            replacement: '${1}:9100'
      
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
      
      - job_name: 'kubernetes-services'
        kubernetes_sd_configs:
          - role: endpoints
        relabel_configs:
          - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
            action: keep
            regex: true
```

### Helm으로 Prometheus Stack 설치
```bash
# Prometheus Community Helm Repository 추가
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# kube-prometheus-stack 설치
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values prometheus-values.yaml
```

### prometheus-values.yaml
```yaml
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi

grafana:
  adminPassword: "admin123"
  ingress:
    enabled: true
    annotations:
      kubernetes.io/ingress.class: nginx
      cert-manager.io/cluster-issuer: letsencrypt-prod
    hosts:
      - grafana.yourdomain.com
    tls:
      - secretName: grafana-tls
        hosts:
          - grafana.yourdomain.com

alertmanager:
  config:
    global:
      smtp_smarthost: 'smtp.gmail.com:587'
      smtp_from: 'alerts@yourdomain.com'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
    
    receivers:
      - name: 'web.hook'
        email_configs:
          - to: 'admin@yourdomain.com'
            subject: 'Alert: {{ .GroupLabels.alertname }}'
            body: |
              {{ range .Alerts }}
              Alert: {{ .Annotations.summary }}
              Description: {{ .Annotations.description }}
              {{ end }}
```

## ELK Stack 로그 관리

### Elasticsearch 설치
```yaml
# elasticsearch.yaml
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: elasticsearch
spec:
  version: 8.8.0
  nodeSets:
  - name: default
    count: 3
    config:
      node.store.allow_mmap: false
      xpack.security.enabled: true
      xpack.security.transport.ssl.enabled: true
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          resources:
            requests:
              memory: 4Gi
              cpu: 1
            limits:
              memory: 8Gi
              cpu: 2
    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 100Gi
        storageClassName: gp3
```

### Kibana 설정
```yaml
# kibana.yaml
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: kibana
spec:
  version: 8.8.0
  count: 1
  elasticsearchRef:
    name: elasticsearch
  http:
    tls:
      selfSignedCertificate:
        disabled: true
  podTemplate:
    spec:
      containers:
      - name: kibana
        resources:
          requests:
            memory: 1Gi
            cpu: 500m
          limits:
            memory: 2Gi
            cpu: 1
```

### Filebeat 로그 수집
```yaml
# filebeat.yaml
apiVersion: beat.k8s.elastic.co/v1beta1
kind: Beat
metadata:
  name: filebeat
spec:
  type: filebeat
  version: 8.8.0
  elasticsearchRef:
    name: elasticsearch
  config:
    filebeat.inputs:
    - type: kubernetes
      node: ${NODE_NAME}
      hints.enabled: true
      hints.default_config:
        type: container
        paths:
          - /var/log/containers/*${data.kubernetes.container.id}.log
    
    processors:
    - add_kubernetes_metadata:
        host: ${NODE_NAME}
        matchers:
        - logs_path:
            logs_path: "/var/log/containers/"
    
    output.elasticsearch:
      hosts: ["elasticsearch-es-http:9200"]
  
  daemonSet:
    podTemplate:
      spec:
        serviceAccountName: filebeat
        terminationGracePeriodSeconds: 30
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        containers:
        - name: filebeat
          securityContext:
            runAsUser: 0
          volumeMounts:
          - name: varlogcontainers
            mountPath: /var/log/containers
          - name: varlogpods
            mountPath: /var/log/pods
          env:
          - name: NODE_NAME
            valueFrom:
              fieldRef:
                fieldPath: spec.nodeName
        volumes:
        - name: varlogcontainers
          hostPath:
            path: /var/log/containers
        - name: varlogpods
          hostPath:
            path: /var/log/pods
```

## 알림 규칙 설정

### Prometheus Alert Rules
```yaml
# alert-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: infrastructure-alerts
spec:
  groups:
  - name: infrastructure
    rules:
    - alert: NodeDown
      expr: up{job="node-exporter"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.instance }} is down"
        description: "Node {{ $labels.instance }} has been down for more than 5 minutes"
    
    - alert: HighCPUUsage
      expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[2m])) * 100) > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage on {{ $labels.instance }}"
        description: "CPU usage is above 80% on {{ $labels.instance }}"
    
    - alert: HighMemoryUsage
      expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage on {{ $labels.instance }}"
        description: "Memory usage is above 85% on {{ $labels.instance }}"
    
    - alert: DiskSpaceLow
      expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Low disk space on {{ $labels.instance }}"
        description: "Disk space is below 10% on {{ $labels.instance }}"
    
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Pod {{ $labels.pod }} is crash looping"
        description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is restarting frequently"
```

## Grafana 대시보드

### 인프라 대시보드 JSON
```json
{
  "dashboard": {
    "id": null,
    "title": "Infrastructure Overview",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[2m])) * 100)"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "stat", 
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
          }
        ]
      },
      {
        "title": "Disk Usage",
        "type": "table",
        "targets": [
          {
            "expr": "100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)"
          }
        ]
      }
    ]
  }
}
```

## Jaeger 분산 추적

### Jaeger Operator 설치
```bash
kubectl create namespace observability
kubectl create -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.46.0/jaeger-operator.yaml -n observability
```

### Jaeger 인스턴스 생성
```yaml
# jaeger.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
spec:
  strategy: production
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      redundancyPolicy: SingleRedundancy
      storage:
        storageClassName: gp3
        size: 100Gi
  collector:
    maxReplicas: 5
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
```

## 애플리케이션 메트릭 수집

### Spring Boot 애플리케이션
```java
// build.gradle
implementation 'org.springframework.boot:spring-boot-starter-actuator'
implementation 'io.micrometer:micrometer-registry-prometheus'

// application.yml
management:
  endpoints:
    web:
      exposure:
        include: health,metrics,prometheus
  metrics:
    export:
      prometheus:
        enabled: true
```

### Node.js 애플리케이션
```javascript
const promClient = require('prom-client');
const express = require('express');
const app = express();

// 기본 메트릭 수집
promClient.collectDefaultMetrics();

// 커스텀 메트릭
const httpRequestsTotal = new promClient.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

app.get('/metrics', (req, res) => {
  res.set('Content-Type', promClient.register.contentType);
  res.end(promClient.register.metrics());
});
```

## SLI/SLO 모니터링

### SLI 정의
```yaml
# sli-slo.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sli-slo-config
data:
  sli.yaml: |
    slis:
      - name: api_availability
        query: |
          (
            sum(rate(http_requests_total{job="api-server",code!~"5.."}[5m])) /
            sum(rate(http_requests_total{job="api-server"}[5m]))
          ) * 100
        target: 99.9
      
      - name: api_latency
        query: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{job="api-server"}[5m])) by (le)
          )
        target: 0.5
```

다음 포스트에서는 보안 모니터링 및 위협 탐지 시스템 구축을 다뤄보겠습니다!
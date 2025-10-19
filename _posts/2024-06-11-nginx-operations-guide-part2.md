---
layout: post
title: "Nginx 운영 가이드 Part 2 - 성능 모니터링, 메트릭 수집, 상태 확인"
date: 2024-06-11
categories: [Infrastructure, Nginx, Monitoring]
tags: [nginx, monitoring, metrics, performance, prometheus, grafana]
---

# Nginx 운영 가이드 Part 2 - 성능 모니터링, 메트릭 수집, 상태 확인

실무 운영 환경에서 Nginx의 성능을 효과적으로 모니터링하고 메트릭을 수집하는 방법을 다룹니다. Prometheus, Grafana와의 연동부터 실시간 알림 시스템까지 포괄적인 모니터링 전략을 학습합니다.

<!--more-->

## 목차
1. [성능 메트릭 기초](#성능-메트릭-기초)
2. [Nginx 상태 모듈](#nginx-상태-모듈)
3. [Prometheus 연동](#prometheus-연동)
4. [Grafana 대시보드](#grafana-대시보드)
5. [로그 기반 모니터링](#로그-기반-모니터링)
6. [알림 시스템](#알림-시스템)

## 성능 메트릭 기초

### 핵심 성능 지표
```bash
# 기본 성능 메트릭 수집 스크립트
#!/bin/bash
# /usr/local/bin/nginx-metrics.sh

NGINX_STATUS_URL="http://localhost/nginx_status"
LOG_FILE="/var/log/nginx/access.log"

collect_basic_metrics() {
    echo "=== Nginx Performance Metrics $(date) ==="
    
    # 연결 통계
    if curl -s $NGINX_STATUS_URL > /dev/null 2>&1; then
        local stats=$(curl -s $NGINX_STATUS_URL)
        echo "Connection Statistics:"
        echo "$stats" | while read line; do
            echo "  $line"
        done
    else
        echo "Error: Cannot access nginx status page"
    fi
    
    # 프로세스 리소스 사용량
    echo -e "\nProcess Resource Usage:"
    ps aux | grep '[n]ginx' | awk '{
        cpu += $3
        mem += $4
        vsz += $5
        rss += $6
        processes++
    }
    END {
        printf "  Processes: %d\n", processes
        printf "  Total CPU: %.2f%%\n", cpu
        printf "  Total Memory: %.2f%%\n", mem
        printf "  Total VSZ: %.2f MB\n", vsz/1024
        printf "  Total RSS: %.2f MB\n", rss/1024
    }'
    
    # 네트워크 연결 수
    echo -e "\nNetwork Connections:"
    local http_conn=$(ss -tuln | grep -c ":80 ")
    local https_conn=$(ss -tuln | grep -c ":443 ")
    echo "  HTTP connections: $http_conn"
    echo "  HTTPS connections: $https_conn"
    
    # 최근 요청 통계 (지난 1분)
    echo -e "\nRecent Request Statistics:"
    local one_min_ago=$(date -d '1 minute ago' '+%d/%b/%Y:%H:%M')
    local recent_requests=$(grep "$one_min_ago" $LOG_FILE 2>/dev/null | wc -l)
    echo "  Requests in last minute: $recent_requests"
    echo "  Requests per second: $((recent_requests / 60))"
}

# 응답 시간 분석
analyze_response_times() {
    echo -e "\n=== Response Time Analysis ==="
    
    # 최근 1000개 요청의 응답 시간 분석
    tail -1000 $LOG_FILE | awk '{
        # 로그 형식에 따라 응답 시간 필드 조정 필요
        if (NF >= 12 && $12 ~ /^[0-9]/) {
            response_time = $12
            total += response_time
            count++
            
            if (response_time < 0.1) fast++
            else if (response_time < 0.5) medium++
            else if (response_time < 1.0) slow++
            else very_slow++
            
            if (response_time > max_time) max_time = response_time
            if (min_time == 0 || response_time < min_time) min_time = response_time
        }
    }
    END {
        if (count > 0) {
            printf "  Total requests analyzed: %d\n", count
            printf "  Average response time: %.3fs\n", total/count
            printf "  Min response time: %.3fs\n", min_time
            printf "  Max response time: %.3fs\n", max_time
            printf "  Fast (< 0.1s): %d (%.1f%%)\n", fast, fast/count*100
            printf "  Medium (0.1-0.5s): %d (%.1f%%)\n", medium, medium/count*100
            printf "  Slow (0.5-1.0s): %d (%.1f%%)\n", slow, slow/count*100
            printf "  Very slow (> 1.0s): %d (%.1f%%)\n", very_slow, very_slow/count*100
        }
    }'
}

# 에러율 분석
analyze_error_rates() {
    echo -e "\n=== Error Rate Analysis ==="
    
    tail -10000 $LOG_FILE | awk '{
        status = $9
        total++
        
        if (status ~ /^2/) success++
        else if (status ~ /^3/) redirect++
        else if (status ~ /^4/) client_error++
        else if (status ~ /^5/) server_error++
    }
    END {
        if (total > 0) {
            printf "  Total requests: %d\n", total
            printf "  Success (2xx): %d (%.2f%%)\n", success, success/total*100
            printf "  Redirects (3xx): %d (%.2f%%)\n", redirect, redirect/total*100
            printf "  Client errors (4xx): %d (%.2f%%)\n", client_error, client_error/total*100
            printf "  Server errors (5xx): %d (%.2f%%)\n", server_error, server_error/total*100
            printf "  Overall error rate: %.2f%%\n", (client_error+server_error)/total*100
        }
    }'
}

collect_basic_metrics
analyze_response_times
analyze_error_rates
```

## Nginx 상태 모듈

### stub_status 모듈 설정
```nginx
# nginx.conf에 상태 페이지 추가
server {
    listen 80;
    server_name localhost;
    
    # 상태 페이지
    location /nginx_status {
        stub_status on;
        
        # 접근 제한
        allow 127.0.0.1;
        allow 192.168.0.0/16;
        allow 10.0.0.0/8;
        deny all;
        
        # 로그 비활성화
        access_log off;
    }
    
    # 확장 상태 정보 (nginx-plus)
    location /status {
        status;
        status_format json;
        
        allow 127.0.0.1;
        deny all;
        access_log off;
    }
}
```

### 상태 정보 파싱 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-status-parser.sh

NGINX_STATUS_URL="http://localhost/nginx_status"
OUTPUT_FILE="/var/log/nginx/status_metrics.log"

parse_stub_status() {
    local status_output=$(curl -s $NGINX_STATUS_URL 2>/dev/null)
    
    if [ -z "$status_output" ]; then
        echo "Error: Could not fetch nginx status"
        return 1
    fi
    
    # stub_status 출력 파싱
    # Active connections: 2
    # server accepts handled requests
    #  16630948 16630948 31070465
    # Reading: 0 Writing: 2 Waiting: 0
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local active_conn=$(echo "$status_output" | grep "Active connections" | awk '{print $3}')
    local accepts=$(echo "$status_output" | sed -n '3p' | awk '{print $1}')
    local handled=$(echo "$status_output" | sed -n '3p' | awk '{print $2}')
    local requests=$(echo "$status_output" | sed -n '3p' | awk '{print $3}')
    local reading=$(echo "$status_output" | grep "Reading" | awk '{print $2}')
    local writing=$(echo "$status_output" | grep "Writing" | awk '{print $4}')
    local waiting=$(echo "$status_output" | grep "Waiting" | awk '{print $6}')
    
    # Prometheus 형식으로 출력
    cat << EOF
# HELP nginx_connections_active Active connections
# TYPE nginx_connections_active gauge
nginx_connections_active $active_conn

# HELP nginx_connections_reading Reading connections
# TYPE nginx_connections_reading gauge
nginx_connections_reading $reading

# HELP nginx_connections_writing Writing connections
# TYPE nginx_connections_writing gauge
nginx_connections_writing $writing

# HELP nginx_connections_waiting Waiting connections
# TYPE nginx_connections_waiting gauge
nginx_connections_waiting $waiting

# HELP nginx_accepts_total Total accepted connections
# TYPE nginx_accepts_total counter
nginx_accepts_total $accepts

# HELP nginx_handled_total Total handled connections
# TYPE nginx_handled_total counter
nginx_handled_total $handled

# HELP nginx_requests_total Total requests
# TYPE nginx_requests_total counter
nginx_requests_total $requests
EOF
    
    # 로그 파일에도 기록
    echo "$timestamp,$active_conn,$reading,$writing,$waiting,$accepts,$handled,$requests" >> $OUTPUT_FILE
}

# JSON 형식으로 출력 (API용)
output_json() {
    local status_output=$(curl -s $NGINX_STATUS_URL 2>/dev/null)
    
    if [ -z "$status_output" ]; then
        echo '{"error": "Could not fetch nginx status"}'
        return 1
    fi
    
    local active_conn=$(echo "$status_output" | grep "Active connections" | awk '{print $3}')
    local accepts=$(echo "$status_output" | sed -n '3p' | awk '{print $1}')
    local handled=$(echo "$status_output" | sed -n '3p' | awk '{print $2}')
    local requests=$(echo "$status_output" | sed -n '3p' | awk '{print $3}')
    local reading=$(echo "$status_output" | grep "Reading" | awk '{print $2}')
    local writing=$(echo "$status_output" | grep "Writing" | awk '{print $4}')
    local waiting=$(echo "$status_output" | grep "Waiting" | awk '{print $6}')
    
    cat << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "connections": {
    "active": $active_conn,
    "reading": $reading,
    "writing": $writing,
    "waiting": $waiting
  },
  "requests": {
    "accepts": $accepts,
    "handled": $handled,
    "total": $requests
  }
}
EOF
}

case "$1" in
    "prometheus")
        parse_stub_status
        ;;
    "json")
        output_json
        ;;
    *)
        parse_stub_status
        ;;
esac
```

## Prometheus 연동

### nginx-prometheus-exporter 설정
```yaml
# docker-compose.yml for nginx-prometheus-exporter
version: '3.8'
services:
  nginx-exporter:
    image: nginx/nginx-prometheus-exporter:0.10.0
    ports:
      - "9113:9113"
    command:
      - '-nginx.scrape-uri=http://nginx:80/nginx_status'
    depends_on:
      - nginx
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
```

### Prometheus 설정
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "nginx_rules.yml"

scrape_configs:
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Nginx 관련 알림 규칙
```yaml
# nginx_rules.yml
groups:
- name: nginx
  rules:
  
  # 높은 에러율 알림
  - alert: NginxHighErrorRate
    expr: |
      (
        sum(rate(nginx_http_requests_total{status=~"5.."}[5m]))
        /
        sum(rate(nginx_http_requests_total[5m]))
      ) * 100 > 5
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Nginx high error rate"
      description: "Nginx error rate is {{ $value }}% over the last 5 minutes"

  # 높은 응답 시간 알림
  - alert: NginxHighResponseTime
    expr: nginx_http_request_duration_seconds{quantile="0.95"} > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Nginx high response time"
      description: "95th percentile response time is {{ $value }}s"

  # 낮은 성공률 알림
  - alert: NginxLowSuccessRate
    expr: |
      (
        sum(rate(nginx_http_requests_total{status=~"2.."}[5m]))
        /
        sum(rate(nginx_http_requests_total[5m]))
      ) * 100 < 95
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Nginx low success rate"
      description: "Success rate is {{ $value }}% over the last 10 minutes"

  # 연결 수 급증 알림
  - alert: NginxHighConnections
    expr: nginx_connections_active > 1000
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Nginx high active connections"
      description: "Active connections: {{ $value }}"

  # Nginx 서비스 다운 알림
  - alert: NginxDown
    expr: up{job="nginx"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Nginx service is down"
      description: "Nginx service has been down for more than 1 minute"
```

## Grafana 대시보드

### Nginx 대시보드 JSON
```json
{
  "dashboard": {
    "id": null,
    "title": "Nginx Performance Dashboard",
    "tags": ["nginx"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Requests per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(nginx_requests_total[5m])",
            "legendFormat": "RPS"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Active Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "nginx_connections_active",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(nginx_http_requests_total{status=~\"4..\"}[5m])) by (status)",
            "legendFormat": "4xx Errors"
          },
          {
            "expr": "sum(rate(nginx_http_requests_total{status=~\"5..\"}[5m])) by (status)",
            "legendFormat": "5xx Errors"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

### 대시보드 프로비저닝
```yaml
# grafana/provisioning/dashboards/nginx.yml
apiVersion: 1

providers:
  - name: 'nginx'
    orgId: 1
    folder: 'Nginx'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards/nginx
```

## 로그 기반 모니터링

### 실시간 로그 분석기
```bash
#!/bin/bash
# /usr/local/bin/nginx-log-monitor.sh

LOG_FILE="/var/log/nginx/access.log"
METRICS_FILE="/var/log/nginx/realtime_metrics.txt"
ALERT_THRESHOLD_ERROR_RATE=5  # 5% 이상 에러율
ALERT_THRESHOLD_RESPONSE_TIME=2  # 2초 이상 응답시간

# 실시간 메트릭 계산
calculate_realtime_metrics() {
    # 최근 1분간의 로그 분석
    local one_min_ago=$(date -d '1 minute ago' '+%d/%b/%Y:%H:%M')
    local current_min=$(date '+%d/%b/%Y:%H:%M')
    
    tail -10000 $LOG_FILE | awk -v start="$one_min_ago" -v end="$current_min" '
    BEGIN {
        total = 0
        success = 0
        client_error = 0
        server_error = 0
        total_response_time = 0
        max_response_time = 0
    }
    {
        # 시간 필터링 (로그 형식에 따라 조정 필요)
        log_time = $4
        gsub(/\[/, "", log_time)
        
        if (log_time >= start && log_time <= end) {
            total++
            status = $9
            
            # 응답 시간 (로그 형식에 따라 필드 번호 조정)
            if (NF >= 12 && $12 ~ /^[0-9]/) {
                response_time = $12
                total_response_time += response_time
                if (response_time > max_response_time) {
                    max_response_time = response_time
                }
            }
            
            # 상태 코드 분류
            if (status ~ /^2/) success++
            else if (status ~ /^4/) client_error++
            else if (status ~ /^5/) server_error++
        }
    }
    END {
        if (total > 0) {
            error_rate = (client_error + server_error) / total * 100
            avg_response_time = total_response_time / total
            
            printf "timestamp=%s\n", strftime("%Y-%m-%d %H:%M:%S")
            printf "total_requests=%d\n", total
            printf "requests_per_second=%.2f\n", total/60
            printf "success_rate=%.2f\n", success/total*100
            printf "error_rate=%.2f\n", error_rate
            printf "avg_response_time=%.3f\n", avg_response_time
            printf "max_response_time=%.3f\n", max_response_time
        }
    }' > $METRICS_FILE
}

# 알림 확인 및 발송
check_alerts() {
    if [ ! -f "$METRICS_FILE" ]; then
        return
    fi
    
    local error_rate=$(grep "error_rate=" $METRICS_FILE | cut -d'=' -f2)
    local max_response_time=$(grep "max_response_time=" $METRICS_FILE | cut -d'=' -f2)
    
    # 에러율 알림
    if (( $(echo "$error_rate > $ALERT_THRESHOLD_ERROR_RATE" | bc -l) )); then
        send_alert "HIGH_ERROR_RATE" "Error rate: ${error_rate}% (threshold: ${ALERT_THRESHOLD_ERROR_RATE}%)"
    fi
    
    # 응답시간 알림
    if (( $(echo "$max_response_time > $ALERT_THRESHOLD_RESPONSE_TIME" | bc -l) )); then
        send_alert "HIGH_RESPONSE_TIME" "Max response time: ${max_response_time}s (threshold: ${ALERT_THRESHOLD_RESPONSE_TIME}s)"
    fi
}

# 알림 발송 함수
send_alert() {
    local alert_type=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] ALERT: $alert_type - $message"
    
    # Slack 알림 (webhook URL 필요)
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Nginx Alert: $alert_type\\n$message\"}" \
            $SLACK_WEBHOOK_URL
    fi
    
    # 이메일 알림
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "Nginx Alert: $alert_type" admin@example.com
    fi
    
    # 로그 기록
    echo "[$timestamp] $alert_type: $message" >> /var/log/nginx/alerts.log
}

# 메인 실행
calculate_realtime_metrics
check_alerts

# Prometheus 형식으로 메트릭 노출
if [ "$1" = "prometheus" ]; then
    if [ -f "$METRICS_FILE" ]; then
        while IFS='=' read -r key value; do
            case $key in
                total_requests)
                    echo "# HELP nginx_realtime_requests_total Total requests in last minute"
                    echo "# TYPE nginx_realtime_requests_total gauge"
                    echo "nginx_realtime_requests_total $value"
                    ;;
                requests_per_second)
                    echo "# HELP nginx_realtime_rps Requests per second"
                    echo "# TYPE nginx_realtime_rps gauge"
                    echo "nginx_realtime_rps $value"
                    ;;
                error_rate)
                    echo "# HELP nginx_realtime_error_rate Error rate percentage"
                    echo "# TYPE nginx_realtime_error_rate gauge"
                    echo "nginx_realtime_error_rate $value"
                    ;;
                avg_response_time)
                    echo "# HELP nginx_realtime_avg_response_time Average response time"
                    echo "# TYPE nginx_realtime_avg_response_time gauge"
                    echo "nginx_realtime_avg_response_time $value"
                    ;;
            esac
        done < $METRICS_FILE
    fi
fi
```

## 알림 시스템

### Alertmanager 설정
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@example.com'
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://127.0.0.1:5001/'

- name: 'critical-alerts'
  email_configs:
  - to: 'admin@example.com'
    subject: 'Critical Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  slack_configs:
  - channel: '#alerts'
    color: danger
    title: 'Critical Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  slack_configs:
  - channel: '#monitoring'
    color: warning
    title: 'Warning Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
```

### 커스텀 웹훅 알림 서버
```python
#!/usr/bin/env python3
# /usr/local/bin/nginx-alert-webhook.py

from flask import Flask, request, jsonify
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 설정
SMTP_SERVER = 'smtp.example.com'
SMTP_PORT = 587
EMAIL_USERNAME = 'alerts@example.com'
EMAIL_PASSWORD = 'your_password'
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

def send_email(subject, body, to_email):
    """이메일 알림 발송"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logging.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False

def send_slack_notification(message, color='danger'):
    """Slack 알림 발송"""
    try:
        payload = {
            'attachments': [{
                'color': color,
                'text': message,
                'ts': int(time.time())
            }]
        }
        
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            logging.info("Slack notification sent")
            return True
        else:
            logging.error(f"Slack notification failed: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Failed to send Slack notification: {e}")
        return False

@app.route('/webhook', methods=['POST'])
def webhook():
    """Alertmanager 웹훅 엔드포인트"""
    data = request.json
    
    for alert in data.get('alerts', []):
        alert_name = alert.get('labels', {}).get('alertname', 'Unknown')
        severity = alert.get('labels', {}).get('severity', 'unknown')
        description = alert.get('annotations', {}).get('description', 'No description')
        summary = alert.get('annotations', {}).get('summary', 'No summary')
        
        # 알림 메시지 생성
        message = f"""
        🚨 *{alert_name}*
        
        *Severity:* {severity.upper()}
        *Summary:* {summary}
        *Description:* {description}
        *Time:* {alert.get('startsAt', 'Unknown')}
        """
        
        # 심각도에 따른 알림 발송
        if severity == 'critical':
            send_email(
                subject=f"CRITICAL: {alert_name}",
                body=message.replace('*', '<b>').replace('*', '</b>'),
                to_email='admin@example.com'
            )
            send_slack_notification(message, 'danger')
        elif severity == 'warning':
            send_slack_notification(message, 'warning')
        
        logging.info(f"Processed alert: {alert_name} ({severity})")
    
    return jsonify({'status': 'success'})

@app.route('/health', methods=['GET'])
def health():
    """헬스체크 엔드포인트"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
```

### 모니터링 자동화 Cron 작업
```bash
# /etc/cron.d/nginx-monitoring

# 실시간 메트릭 수집 (1분마다)
* * * * * root /usr/local/bin/nginx-log-monitor.sh >> /var/log/nginx/monitoring.log 2>&1

# 상세 성능 분석 (5분마다)
*/5 * * * * root /usr/local/bin/nginx-metrics.sh >> /var/log/nginx/performance.log 2>&1

# 상태 정보 수집 (1분마다)
* * * * * root /usr/local/bin/nginx-status-parser.sh prometheus > /var/lib/node_exporter/textfile_collector/nginx.prom

# 일일 리포트 생성 (매일 오전 9시)
0 9 * * * root /usr/local/bin/generate-daily-report.sh | mail -s "Daily Nginx Report" admin@example.com

# 로그 정리 (매주 일요일)
0 2 * * 0 root find /var/log/nginx -name "*.log" -mtime +30 -delete
```

## 실습 예제

### 종합 모니터링 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-monitoring-suite.sh

# 설정
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
ALERT_EMAIL="admin@example.com"

# 모든 모니터링 도구 상태 확인
check_monitoring_stack() {
    echo "=== Monitoring Stack Health Check ==="
    
    # Prometheus 상태
    if curl -s "$PROMETHEUS_URL/api/v1/query?query=up" > /dev/null; then
        echo "✓ Prometheus is running"
    else
        echo "✗ Prometheus is not accessible"
    fi
    
    # Grafana 상태
    if curl -s "$GRAFANA_URL/api/health" > /dev/null; then
        echo "✓ Grafana is running"
    else
        echo "✗ Grafana is not accessible"
    fi
    
    # Nginx Exporter 상태
    if curl -s "http://localhost:9113/metrics" > /dev/null; then
        echo "✓ Nginx Exporter is running"
    else
        echo "✗ Nginx Exporter is not accessible"
    fi
}

# 현재 알림 상태 확인
check_active_alerts() {
    echo -e "\n=== Active Alerts ==="
    
    local alerts=$(curl -s "$PROMETHEUS_URL/api/v1/alerts" | jq -r '.data.alerts[] | select(.state == "firing") | .labels.alertname')
    
    if [ -z "$alerts" ]; then
        echo "No active alerts"
    else
        echo "Active alerts:"
        echo "$alerts" | while read alert; do
            echo "  - $alert"
        done
    fi
}

# 성능 요약
performance_summary() {
    echo -e "\n=== Performance Summary ==="
    
    # Prometheus 쿼리를 통한 메트릭 수집
    local rps=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(nginx_requests_total[5m])" | jq -r '.data.result[0].value[1] // 0')
    local error_rate=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(nginx_http_requests_total{status=~\"5..\"}[5m])/rate(nginx_http_requests_total[5m])*100" | jq -r '.data.result[0].value[1] // 0')
    local active_conn=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=nginx_connections_active" | jq -r '.data.result[0].value[1] // 0')
    
    printf "Requests per second: %.2f\n" $rps
    printf "Error rate: %.2f%%\n" $error_rate
    printf "Active connections: %.0f\n" $active_conn
}

check_monitoring_stack
check_active_alerts
performance_summary
```

## 다음 단계

다음 포스트에서는 다음 내용을 다루겠습니다:

- 백업 및 복구 전략
- 무중단 업그레이드 방법
- 장애 대응 및 복구 절차
- 성능 튜닝 고급 기법

## 참고 자료

- [Prometheus Nginx Exporter](https://github.com/nginxinc/nginx-prometheus-exporter)
- [Grafana Nginx Dashboard](https://grafana.com/grafana/dashboards/12708)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/alertmanager/)
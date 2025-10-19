---
layout: post
title: "Nginx 운영 가이드 Part 1 - 프로세스 관리, 서비스 제어, 로그 관리"
date: 2024-06-10
categories: [Infrastructure, Nginx, Operations]
tags: [nginx, operations, process-management, logging, service-control]
---

# Nginx 운영 가이드 Part 1 - 프로세스 관리, 서비스 제어, 로그 관리

실무에서 Nginx를 안정적으로 운영하기 위한 핵심 기술들을 다룹니다. 프로세스 관리부터 서비스 제어, 로그 관리까지 운영자가 반드시 알아야 할 내용들을 실습 중심으로 학습합니다.

<!--more-->

## 목차
1. [프로세스 관리](#프로세스-관리)
2. [서비스 제어](#서비스-제어)
3. [로그 관리](#로그-관리)
4. [설정 테스트 및 리로드](#설정-테스트-및-리로드)
5. [시그널 처리](#시그널-처리)
6. [자동화 스크립트](#자동화-스크립트)

## 프로세스 관리

### Nginx 프로세스 구조
```bash
# 프로세스 트리 확인
ps aux --forest | grep nginx

# 프로세스 상세 정보
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | grep nginx

# 메모리 사용량 상세 분석
cat /proc/$(pgrep -f "nginx: master")/status
```

### 마스터/워커 프로세스 관리
```nginx
# nginx.conf
user www-data;
worker_processes auto;  # CPU 코어 수만큼 자동 생성
worker_cpu_affinity auto;  # CPU 코어에 워커 바인딩
worker_rlimit_nofile 65535;  # 파일 디스크립터 한계

# 워커 프로세스 우선순위
worker_priority -10;  # -20(highest) to 20(lowest)

# 워커 종료 타임아웃
worker_shutdown_timeout 30s;

# 워커별 메모리 제한
worker_rlimit_core 2G;
```

### 프로세스 모니터링 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-monitor.sh

NGINX_PID_FILE="/var/run/nginx.pid"
THRESHOLD_CPU=80
THRESHOLD_MEM=80

# 프로세스 상태 확인
check_nginx_status() {
    if [ -f "$NGINX_PID_FILE" ]; then
        local pid=$(cat $NGINX_PID_FILE)
        if ps -p $pid > /dev/null 2>&1; then
            echo "Nginx is running (PID: $pid)"
            return 0
        else
            echo "Nginx PID file exists but process is dead"
            return 1
        fi
    else
        echo "Nginx is not running"
        return 1
    fi
}

# 리소스 사용량 확인
check_resource_usage() {
    local nginx_pids=$(pgrep nginx)
    local total_cpu=0
    local total_mem=0
    
    for pid in $nginx_pids; do
        if [ -d "/proc/$pid" ]; then
            local cpu=$(ps -p $pid -o %cpu --no-headers)
            local mem=$(ps -p $pid -o %mem --no-headers)
            total_cpu=$(echo "$total_cpu + $cpu" | bc -l)
            total_mem=$(echo "$total_mem + $mem" | bc -l)
        fi
    done
    
    echo "Total CPU usage: ${total_cpu}%"
    echo "Total Memory usage: ${total_mem}%"
    
    # 임계값 확인
    if (( $(echo "$total_cpu > $THRESHOLD_CPU" | bc -l) )); then
        echo "WARNING: CPU usage exceeds threshold!"
        # 알림 발송 로직 추가
    fi
    
    if (( $(echo "$total_mem > $THRESHOLD_MEM" | bc -l) )); then
        echo "WARNING: Memory usage exceeds threshold!"
        # 알림 발송 로직 추가
    fi
}

# 워커 프로세스 개수 확인
check_worker_count() {
    local master_pid=$(cat $NGINX_PID_FILE 2>/dev/null)
    local worker_count=$(pgrep -P $master_pid 2>/dev/null | wc -l)
    local expected_workers=$(nproc)
    
    echo "Worker processes: $worker_count (expected: $expected_workers)"
    
    if [ $worker_count -ne $expected_workers ]; then
        echo "WARNING: Worker process count mismatch!"
    fi
}

# 메인 실행
check_nginx_status
check_resource_usage
check_worker_count
```

## 서비스 제어

### Systemd 서비스 관리
```bash
# 서비스 상태 확인
systemctl status nginx

# 상세 상태 정보
systemctl show nginx

# 서비스 시작/중지/재시작
systemctl start nginx
systemctl stop nginx
systemctl restart nginx
systemctl reload nginx

# 서비스 활성화/비활성화
systemctl enable nginx
systemctl disable nginx

# 서비스 로그 확인
journalctl -u nginx -f
journalctl -u nginx --since "1 hour ago"
```

### 커스텀 Systemd 서비스 파일
```ini
# /etc/systemd/system/nginx.service
[Unit]
Description=A high performance web server and a reverse proxy server
Documentation=man:nginx(8)
After=network.target nss-lookup.target

[Service]
Type=forking
PIDFile=/run/nginx.pid
ExecStartPre=/usr/sbin/nginx -t -q -g 'daemon on; master_process on;'
ExecStart=/usr/sbin/nginx -g 'daemon on; master_process on;'
ExecReload=/bin/sh -c "/bin/kill -s HUP $(/bin/cat /run/nginx.pid)"
ExecStop=/bin/sh -c "/bin/kill -s TERM $(/bin/cat /run/nginx.pid)"
TimeoutStopSec=5
KillMode=mixed
PrivateTmp=true
LimitNOFILE=65535
LimitNPROC=4096

# 보안 강화
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/nginx /var/cache/nginx /run

[Install]
WantedBy=multi-user.target
```

### 그레이스풀 셧다운 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-graceful-shutdown.sh

NGINX_PID_FILE="/var/run/nginx.pid"
SHUTDOWN_TIMEOUT=30

graceful_shutdown() {
    if [ ! -f "$NGINX_PID_FILE" ]; then
        echo "Nginx is not running"
        return 0
    fi
    
    local pid=$(cat $NGINX_PID_FILE)
    echo "Starting graceful shutdown of Nginx (PID: $pid)"
    
    # 새 연결 차단
    kill -QUIT $pid
    
    # 워커 프로세스들이 종료될 때까지 대기
    local count=0
    while kill -0 $pid 2>/dev/null; do
        if [ $count -ge $SHUTDOWN_TIMEOUT ]; then
            echo "Timeout reached, forcing shutdown"
            kill -TERM $pid
            sleep 2
            kill -KILL $pid 2>/dev/null
            break
        fi
        
        echo "Waiting for graceful shutdown... ($count/$SHUTDOWN_TIMEOUT)"
        sleep 1
        ((count++))
    done
    
    echo "Nginx shutdown complete"
}

graceful_shutdown
```

## 로그 관리

### 로그 설정 최적화
```nginx
http {
    # 로그 형식 정의
    log_format main '$remote_addr - $remote_user [$time_local] '
                   '"$request" $status $body_bytes_sent '
                   '"$http_referer" "$http_user_agent"';
    
    log_format detailed '$remote_addr - $remote_user [$time_local] '
                       '"$request" $status $body_bytes_sent '
                       '"$http_referer" "$http_user_agent" '
                       '$request_time $upstream_response_time '
                       '$pipe $connection_requests';
    
    log_format json escape=json '{'
                   '"timestamp":"$time_iso8601",'
                   '"remote_addr":"$remote_addr",'
                   '"method":"$request_method",'
                   '"uri":"$request_uri",'
                   '"status":$status,'
                   '"bytes_sent":$body_bytes_sent,'
                   '"request_time":$request_time,'
                   '"user_agent":"$http_user_agent",'
                   '"referer":"$http_referer"'
                   '}';
    
    # 조건부 로깅
    map $status $loggable {
        ~^[23]  0;  # 2xx, 3xx는 로그하지 않음
        default 1;
    }
    
    # 전역 로그 설정
    access_log /var/log/nginx/access.log main buffer=32k flush=5m;
    error_log /var/log/nginx/error.log warn;
}

server {
    listen 80;
    server_name example.com;
    
    # 사이트별 로그
    access_log /var/log/nginx/example.access.log detailed if=$loggable;
    error_log /var/log/nginx/example.error.log;
    
    # API 로그 분리
    location /api/ {
        access_log /var/log/nginx/api.log json;
        proxy_pass http://backend;
    }
    
    # 정적 파일 로그 비활성화
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        access_log off;
        expires 1y;
    }
    
    # 로봇 로그 분리
    location = /robots.txt {
        access_log /var/log/nginx/robots.log;
    }
}
```

### 로그 로테이션 설정
```bash
# /etc/logrotate.d/nginx
/var/log/nginx/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data adm
    sharedscripts
    
    prerotate
        if [ -d /etc/logrotate.d/httpd-prerotate ]; then
            run-parts /etc/logrotate.d/httpd-prerotate
        fi
    endscript
    
    postrotate
        # USR1 시그널로 로그 파일 재오픈
        if [ -f /var/run/nginx.pid ]; then
            kill -USR1 `cat /var/run/nginx.pid`
        fi
    endscript
}
```

### 실시간 로그 분석 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-log-analyzer.sh

ACCESS_LOG="/var/log/nginx/access.log"
ERROR_LOG="/var/log/nginx/error.log"

# 실시간 에러 모니터링
monitor_errors() {
    echo "Monitoring Nginx errors (Ctrl+C to stop)..."
    tail -f $ERROR_LOG | while read line; do
        if echo "$line" | grep -q "error"; then
            echo "[ERROR] $(date): $line"
            # 심각한 에러 시 알림 발송
            if echo "$line" | grep -qE "(emerg|alert|crit)"; then
                echo "CRITICAL ERROR DETECTED!" | wall
            fi
        fi
    done
}

# 트래픽 통계
traffic_stats() {
    echo "=== Traffic Statistics (Last 1000 requests) ==="
    tail -1000 $ACCESS_LOG | awk '{
        status_code = $9
        response_size += $10
        request_count++
        
        if (status_code ~ /^2/) { success++ }
        else if (status_code ~ /^3/) { redirect++ }
        else if (status_code ~ /^4/) { client_error++ }
        else if (status_code ~ /^5/) { server_error++ }
    }
    END {
        print "Total requests:", request_count
        print "Success (2xx):", success
        print "Redirects (3xx):", redirect
        print "Client errors (4xx):", client_error
        print "Server errors (5xx):", server_error
        print "Average response size:", response_size/request_count " bytes"
        print "Error rate:", (client_error+server_error)/request_count*100 "%"
    }'
}

# 상위 IP 주소
top_ips() {
    echo "=== Top 10 IP Addresses ==="
    tail -10000 $ACCESS_log | awk '{print $1}' | sort | uniq -c | sort -nr | head -10
}

# 상위 User-Agent
top_user_agents() {
    echo "=== Top 10 User Agents ==="
    tail -1000 $ACCESS_LOG | awk -F'"' '{print $6}' | sort | uniq -c | sort -nr | head -10
}

# 응답 시간 분석
response_time_analysis() {
    echo "=== Response Time Analysis ==="
    tail -1000 $ACCESS_LOG | awk '{
        if (NF >= 11 && $11 ~ /^[0-9]/) {
            response_time = $11
            total_time += response_time
            count++
            
            if (response_time < 0.1) slow_requests[1]++
            else if (response_time < 0.5) slow_requests[2]++
            else if (response_time < 1.0) slow_requests[3]++
            else slow_requests[4]++
        }
    }
    END {
        if (count > 0) {
            print "Average response time:", total_time/count "s"
            print "< 0.1s:", slow_requests[1]
            print "0.1s - 0.5s:", slow_requests[2]
            print "0.5s - 1.0s:", slow_requests[3]
            print "> 1.0s:", slow_requests[4]
        }
    }'
}

# 메인 메뉴
case "$1" in
    "monitor")
        monitor_errors
        ;;
    "stats")
        traffic_stats
        ;;
    "ips")
        top_ips
        ;;
    "agents")
        top_user_agents
        ;;
    "response-time")
        response_time_analysis
        ;;
    "all")
        traffic_stats
        echo ""
        top_ips
        echo ""
        response_time_analysis
        ;;
    *)
        echo "Usage: $0 {monitor|stats|ips|agents|response-time|all}"
        exit 1
        ;;
esac
```

## 설정 테스트 및 리로드

### 설정 검증 자동화
```bash
#!/bin/bash
# /usr/local/bin/nginx-config-test.sh

CONFIG_DIR="/etc/nginx"
BACKUP_DIR="/etc/nginx/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 설정 백업
backup_config() {
    echo "Backing up current configuration..."
    mkdir -p $BACKUP_DIR
    tar czf "$BACKUP_DIR/nginx_config_$TIMESTAMP.tar.gz" -C /etc nginx/
    echo "Configuration backed up to: $BACKUP_DIR/nginx_config_$TIMESTAMP.tar.gz"
}

# 설정 문법 검사
test_config() {
    echo "Testing Nginx configuration..."
    if nginx -t -q; then
        echo "✓ Configuration test passed"
        return 0
    else
        echo "✗ Configuration test failed"
        nginx -t
        return 1
    fi
}

# 안전한 리로드
safe_reload() {
    echo "Performing safe reload..."
    
    # 설정 백업
    backup_config
    
    # 설정 테스트
    if ! test_config; then
        echo "Configuration test failed. Aborting reload."
        return 1
    fi
    
    # 리로드 수행
    if systemctl reload nginx; then
        echo "✓ Nginx reloaded successfully"
        
        # 리로드 후 상태 확인
        sleep 2
        if systemctl is-active nginx >/dev/null; then
            echo "✓ Nginx is running properly after reload"
        else
            echo "✗ Nginx failed after reload, attempting rollback..."
            systemctl restart nginx
        fi
    else
        echo "✗ Nginx reload failed"
        return 1
    fi
}

# 설정 비교
compare_configs() {
    if [ -n "$1" ]; then
        echo "Comparing current config with backup: $1"
        diff -u "$1" "$CONFIG_DIR/nginx.conf"
    else
        echo "Usage: $0 compare <backup_file>"
    fi
}

# 메인 로직
case "$1" in
    "test")
        test_config
        ;;
    "backup")
        backup_config
        ;;
    "reload")
        safe_reload
        ;;
    "compare")
        compare_configs "$2"
        ;;
    *)
        echo "Usage: $0 {test|backup|reload|compare}"
        echo "  test    - Test configuration syntax"
        echo "  backup  - Backup current configuration"
        echo "  reload  - Safe configuration reload with backup"
        echo "  compare - Compare configurations"
        exit 1
        ;;
esac
```

## 시그널 처리

### Nginx 시그널 정리
```bash
#!/bin/bash
# /usr/local/bin/nginx-signals.sh

NGINX_PID_FILE="/var/run/nginx.pid"

get_nginx_pid() {
    if [ -f "$NGINX_PID_FILE" ]; then
        cat $NGINX_PID_FILE
    else
        echo "Nginx PID file not found" >&2
        return 1
    fi
}

# 시그널 함수들
nginx_quit() {
    local pid=$(get_nginx_pid)
    echo "Sending QUIT signal to Nginx (graceful shutdown)"
    kill -QUIT $pid
}

nginx_reload() {
    local pid=$(get_nginx_pid)
    echo "Sending HUP signal to Nginx (reload configuration)"
    kill -HUP $pid
}

nginx_reopen_logs() {
    local pid=$(get_nginx_pid)
    echo "Sending USR1 signal to Nginx (reopen log files)"
    kill -USR1 $pid
}

nginx_upgrade() {
    local pid=$(get_nginx_pid)
    echo "Sending USR2 signal to Nginx (upgrade binary)"
    kill -USR2 $pid
    
    # 새 마스터 프로세스 확인
    sleep 2
    local new_pid=$(pgrep -f "nginx: master process" | grep -v $pid)
    if [ -n "$new_pid" ]; then
        echo "New master process started (PID: $new_pid)"
        echo "Send WINCH to old master to shutdown old workers: kill -WINCH $pid"
        echo "Send QUIT to old master when ready: kill -QUIT $pid"
    else
        echo "Failed to start new master process"
    fi
}

nginx_worker_shutdown() {
    local pid=$(get_nginx_pid)
    echo "Sending WINCH signal to Nginx (graceful worker shutdown)"
    kill -WINCH $pid
}

# 메인 메뉴
case "$1" in
    "quit")
        nginx_quit
        ;;
    "reload")
        nginx_reload
        ;;
    "reopen")
        nginx_reopen_logs
        ;;
    "upgrade")
        nginx_upgrade
        ;;
    "winch")
        nginx_worker_shutdown
        ;;
    *)
        echo "Nginx Signal Manager"
        echo "Usage: $0 {quit|reload|reopen|upgrade|winch}"
        echo ""
        echo "  quit    - QUIT (graceful shutdown)"
        echo "  reload  - HUP (reload configuration)"
        echo "  reopen  - USR1 (reopen log files)"
        echo "  upgrade - USR2 (upgrade binary)"
        echo "  winch   - WINCH (graceful worker shutdown)"
        exit 1
        ;;
esac
```

## 자동화 스크립트

### 종합 관리 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-manager.sh

SCRIPT_DIR="/usr/local/bin"
LOG_DIR="/var/log/nginx"
CONFIG_DIR="/etc/nginx"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 상태 확인
status_check() {
    echo "=== Nginx Status Check ==="
    
    # 프로세스 상태
    if systemctl is-active nginx >/dev/null; then
        log_info "Nginx service is active"
    else
        log_error "Nginx service is not active"
    fi
    
    # 설정 테스트
    if nginx -t -q 2>/dev/null; then
        log_info "Configuration is valid"
    else
        log_error "Configuration has errors"
        nginx -t
    fi
    
    # 포트 확인
    if netstat -tlpn | grep -q ":80.*nginx" 2>/dev/null; then
        log_info "HTTP port (80) is listening"
    else
        log_warn "HTTP port (80) is not listening"
    fi
    
    if netstat -tlpn | grep -q ":443.*nginx" 2>/dev/null; then
        log_info "HTTPS port (443) is listening"
    else
        log_warn "HTTPS port (443) is not listening"
    fi
    
    # 로그 파일 확인
    local log_files=("$LOG_DIR/access.log" "$LOG_DIR/error.log")
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            local size=$(du -h "$log_file" | cut -f1)
            log_info "Log file $log_file exists (size: $size)"
        else
            log_warn "Log file $log_file not found"
        fi
    done
}

# 성능 체크
performance_check() {
    echo "=== Performance Check ==="
    
    # CPU 및 메모리 사용량
    local cpu_usage=$(ps -C nginx -o %cpu --no-headers | awk '{sum += $1} END {print sum}')
    local mem_usage=$(ps -C nginx -o %mem --no-headers | awk '{sum += $1} END {print sum}')
    
    echo "CPU Usage: ${cpu_usage}%"
    echo "Memory Usage: ${mem_usage}%"
    
    # 연결 수 확인
    local connections=$(ss -tuln | grep -E ':80|:443' | wc -l)
    echo "Active connections: $connections"
    
    # 워커 프로세스 수
    local workers=$(pgrep -c "nginx: worker")
    local cpu_cores=$(nproc)
    echo "Worker processes: $workers (CPU cores: $cpu_cores)"
}

# 헬스체크
health_check() {
    echo "=== Health Check ==="
    
    # HTTP 응답 확인
    local http_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/ 2>/dev/null)
    if [ "$http_status" = "200" ]; then
        log_info "HTTP health check passed (status: $http_status)"
    else
        log_error "HTTP health check failed (status: $http_status)"
    fi
    
    # 최근 에러 로그 확인
    local recent_errors=$(tail -100 "$LOG_DIR/error.log" 2>/dev/null | grep -c "$(date '+%Y/%m/%d')")
    if [ "$recent_errors" -gt 10 ]; then
        log_warn "High error count today: $recent_errors"
    else
        log_info "Error count today: $recent_errors"
    fi
}

# 메인 메뉴
main_menu() {
    echo "================================"
    echo "     Nginx Management Tool      "
    echo "================================"
    echo "1. Status Check"
    echo "2. Performance Check"
    echo "3. Health Check"
    echo "4. All Checks"
    echo "5. Configuration Test"
    echo "6. Safe Reload"
    echo "7. View Recent Logs"
    echo "8. Exit"
    echo "================================"
    read -p "Select option: " choice
    
    case $choice in
        1) status_check ;;
        2) performance_check ;;
        3) health_check ;;
        4) status_check; echo ""; performance_check; echo ""; health_check ;;
        5) nginx -t ;;
        6) $SCRIPT_DIR/nginx-config-test.sh reload ;;
        7) $SCRIPT_DIR/nginx-log-analyzer.sh all ;;
        8) exit 0 ;;
        *) echo "Invalid option"; main_menu ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    main_menu
}

# 스크립트 실행
if [ "$1" = "status" ]; then
    status_check
elif [ "$1" = "performance" ]; then
    performance_check
elif [ "$1" = "health" ]; then
    health_check
elif [ "$1" = "all" ]; then
    status_check
    echo ""
    performance_check
    echo ""
    health_check
else
    main_menu
fi
```

## Cron 작업 설정

```bash
# /etc/cron.d/nginx-maintenance
# Nginx 로그 분석 (매시간)
0 * * * * root /usr/local/bin/nginx-log-analyzer.sh stats >> /var/log/nginx/hourly-stats.log 2>&1

# Nginx 상태 체크 (5분마다)
*/5 * * * * root /usr/local/bin/nginx-monitor.sh >> /var/log/nginx/monitor.log 2>&1

# 로그 로테이션 후 통계 (매일 자정)
5 0 * * * root /usr/local/bin/nginx-log-analyzer.sh all > /var/log/nginx/daily-report.log 2>&1

# 설정 백업 (매일)
30 2 * * * root /usr/local/bin/nginx-config-test.sh backup >> /var/log/nginx/backup.log 2>&1
```

## 다음 단계

다음 포스트에서는 다음 내용들을 다루겠습니다:

- 성능 모니터링 및 메트릭 수집
- Prometheus/Grafana 연동
- 알림 시스템 구축
- 트래픽 분석 및 최적화

## 참고 자료

- [Nginx Admin Guide](https://docs.nginx.com/nginx/admin-guide/)
- [Nginx Logging Guide](https://docs.nginx.com/nginx/admin-guide/monitoring/logging/)
- [Systemd Service Management](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
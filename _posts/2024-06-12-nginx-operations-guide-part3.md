---
layout: post
title: "Nginx 운영 가이드 Part 3 - 백업/복구, 업그레이드, 장애 대응"
date: 2024-06-12
categories: [Infrastructure, Nginx, Operations]
tags: [nginx, backup, recovery, upgrade, disaster-response, optimization]
---

# Nginx 운영 가이드 Part 3 - 백업/복구, 업그레이드, 장애 대응

실제 프로덕션 환경에서 Nginx를 안정적으로 운영하기 위한 핵심 운영 기술들을 다룹니다. 백업 및 복구 전략부터 무중단 업그레이드, 장애 상황 대응까지 운영자가 마주할 수 있는 모든 상황에 대한 실무 가이드를 제공합니다.

<!--more-->

## 목차
1. [백업 및 복구 전략](#백업-및-복구-전략)
2. [무중단 업그레이드](#무중단-업그레이드)
3. [장애 대응 절차](#장애-대응-절차)
4. [성능 튜닝](#성능-튜닝)
5. [고가용성 구성](#고가용성-구성)
6. [자동화 및 최적화](#자동화-및-최적화)

## 백업 및 복구 전략

### 백업 대상 및 정책
```bash
#!/bin/bash
# /usr/local/bin/nginx-backup.sh

# 백업 설정
BACKUP_ROOT="/backup/nginx"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30
COMPRESSION_LEVEL=6

# 백업 대상 디렉토리
CONFIG_DIRS=(
    "/etc/nginx"
    "/etc/ssl/nginx"
    "/var/www"
)

LOG_DIRS=(
    "/var/log/nginx"
)

# 백업 함수들
create_backup_structure() {
    local backup_date=$(date +%Y%m%d)
    local backup_dir="$BACKUP_ROOT/$backup_date"
    
    mkdir -p "$backup_dir"/{config,logs,data}
    echo "$backup_dir"
}

backup_configurations() {
    local backup_dir=$1
    echo "Backing up Nginx configurations..."
    
    # 설정 파일 백업
    for dir in "${CONFIG_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            local dirname=$(basename "$dir")
            tar -czf "$backup_dir/config/${dirname}_${TIMESTAMP}.tar.gz" \
                -C "$(dirname "$dir")" "$(basename "$dir")"
            
            # 백업 검증
            if tar -tzf "$backup_dir/config/${dirname}_${TIMESTAMP}.tar.gz" > /dev/null 2>&1; then
                echo "✓ Configuration backup successful: $dirname"
            else
                echo "✗ Configuration backup failed: $dirname"
                return 1
            fi
        fi
    done
}

backup_logs() {
    local backup_dir=$1
    echo "Backing up Nginx logs..."
    
    # 로그 파일 백업 (압축률을 높이기 위해 별도 처리)
    for dir in "${LOG_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            local dirname=$(basename "$dir")
            
            # 현재 로그 파일들 (실시간 사용 중)
            find "$dir" -name "*.log" -type f | while read logfile; do
                local filename=$(basename "$logfile")
                gzip -c "$logfile" > "$backup_dir/logs/${filename}_${TIMESTAMP}.gz"
            done
            
            # 압축된 오래된 로그들
            find "$dir" -name "*.log.gz" -type f -exec cp {} "$backup_dir/logs/" \;
            
            echo "✓ Log backup completed: $dirname"
        fi
    done
}

backup_ssl_certificates() {
    local backup_dir=$1
    echo "Backing up SSL certificates..."
    
    # Let's Encrypt 인증서
    if [ -d "/etc/letsencrypt" ]; then
        tar -czf "$backup_dir/config/letsencrypt_${TIMESTAMP}.tar.gz" \
            -C /etc letsencrypt/
        echo "✓ Let's Encrypt certificates backed up"
    fi
    
    # 커스텀 SSL 인증서
    if [ -d "/etc/ssl/private" ]; then
        tar -czf "$backup_dir/config/ssl_private_${TIMESTAMP}.tar.gz" \
            -C /etc/ssl private/ --warning=no-file-changed
        chmod 600 "$backup_dir/config/ssl_private_${TIMESTAMP}.tar.gz"
        echo "✓ Private SSL certificates backed up"
    fi
}

create_manifest() {
    local backup_dir=$1
    local manifest_file="$backup_dir/backup_manifest_${TIMESTAMP}.txt"
    
    cat > "$manifest_file" << EOF
Nginx Backup Manifest
=====================
Backup Date: $(date)
Hostname: $(hostname)
Nginx Version: $(nginx -v 2>&1)
OS Version: $(lsb_release -d 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME)

Backup Contents:
EOF
    
    # 백업 파일 목록 및 체크섬
    find "$backup_dir" -type f -name "*.tar.gz" -o -name "*.gz" | while read file; do
        local size=$(du -h "$file" | cut -f1)
        local checksum=$(md5sum "$file" | cut -d' ' -f1)
        echo "File: $(basename "$file") | Size: $size | MD5: $checksum" >> "$manifest_file"
    done
    
    echo "✓ Backup manifest created"
}

cleanup_old_backups() {
    echo "Cleaning up old backups (retention: ${RETENTION_DAYS} days)..."
    
    find "$BACKUP_ROOT" -type d -name "????????" -mtime +$RETENTION_DAYS | while read old_backup; do
        echo "Removing old backup: $old_backup"
        rm -rf "$old_backup"
    done
}

# 메인 백업 실행
perform_backup() {
    echo "=== Starting Nginx Backup Process ==="
    
    # 백업 디렉토리 생성
    local backup_dir=$(create_backup_structure)
    
    # 설정 테스트 (백업 전 확인)
    if ! nginx -t; then
        echo "Warning: Current Nginx configuration has errors"
    fi
    
    # 백업 실행
    backup_configurations "$backup_dir"
    backup_logs "$backup_dir"
    backup_ssl_certificates "$backup_dir"
    
    # 메타데이터 생성
    create_manifest "$backup_dir"
    
    # 정리
    cleanup_old_backups
    
    echo "=== Backup Process Completed ==="
    echo "Backup location: $backup_dir"
    
    # 백업 결과 알림
    local backup_size=$(du -sh "$backup_dir" | cut -f1)
    echo "Total backup size: $backup_size"
}

# 백업 실행
perform_backup
```

### 복구 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-restore.sh

BACKUP_ROOT="/backup/nginx"
RESTORE_LOG="/var/log/nginx/restore.log"

list_available_backups() {
    echo "Available backups:"
    ls -la "$BACKUP_ROOT" | grep "^d" | awk '{print $9}' | grep -E "^[0-9]{8}$" | sort -r
}

restore_from_backup() {
    local backup_date=$1
    local backup_dir="$BACKUP_ROOT/$backup_date"
    
    if [ ! -d "$backup_dir" ]; then
        echo "Error: Backup directory not found: $backup_dir"
        return 1
    fi
    
    echo "Starting restore from backup: $backup_date" | tee -a "$RESTORE_LOG"
    
    # 현재 설정 백업
    local current_backup="/tmp/nginx_current_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$current_backup"
    cp -r /etc/nginx "$current_backup/"
    echo "Current configuration backed up to: $current_backup"
    
    # Nginx 중지
    echo "Stopping Nginx service..."
    systemctl stop nginx
    
    # 설정 파일 복구
    echo "Restoring configuration files..."
    cd "$backup_dir/config"
    
    for backup_file in nginx_*.tar.gz; do
        if [ -f "$backup_file" ]; then
            echo "Extracting: $backup_file"
            tar -xzf "$backup_file" -C /etc/
        fi
    done
    
    # SSL 인증서 복구
    for ssl_backup in letsencrypt_*.tar.gz ssl_private_*.tar.gz; do
        if [ -f "$ssl_backup" ]; then
            echo "Restoring SSL certificates: $ssl_backup"
            if [[ "$ssl_backup" == *"private"* ]]; then
                tar -xzf "$ssl_backup" -C /etc/ssl/
            else
                tar -xzf "$ssl_backup" -C /etc/
            fi
        fi
    done
    
    # 설정 테스트
    echo "Testing restored configuration..."
    if nginx -t; then
        echo "✓ Configuration test passed"
        
        # Nginx 시작
        systemctl start nginx
        
        if systemctl is-active nginx >/dev/null; then
            echo "✓ Nginx service started successfully"
            echo "Restore completed successfully" | tee -a "$RESTORE_LOG"
        else
            echo "✗ Failed to start Nginx service"
            echo "Restoring previous configuration..."
            cp -r "$current_backup/nginx" /etc/
            systemctl start nginx
            return 1
        fi
    else
        echo "✗ Configuration test failed"
        echo "Restoring previous configuration..."
        cp -r "$current_backup/nginx" /etc/
        systemctl start nginx
        return 1
    fi
}

# 메인 실행
case "$1" in
    "list")
        list_available_backups
        ;;
    "restore")
        if [ -z "$2" ]; then
            echo "Usage: $0 restore <backup_date>"
            echo "Available backups:"
            list_available_backups
            exit 1
        fi
        restore_from_backup "$2"
        ;;
    *)
        echo "Usage: $0 {list|restore <backup_date>}"
        exit 1
        ;;
esac
```

## 무중단 업그레이드

### Binary 업그레이드 프로세스
```bash
#!/bin/bash
# /usr/local/bin/nginx-upgrade.sh

NGINX_VERSION_TARGET="1.24.0"
NGINX_USER="nginx"
NGINX_GROUP="nginx"
BUILD_DIR="/tmp/nginx-build"
BACKUP_DIR="/backup/nginx-upgrade"

# 현재 설정 확인
check_current_setup() {
    echo "=== Current Nginx Setup ==="
    echo "Version: $(nginx -V 2>&1 | head -n1)"
    echo "Configuration test:"
    nginx -t
    echo "Process info:"
    ps aux | grep nginx | grep -v grep
    echo "Compiled modules:"
    nginx -V 2>&1 | tr ' ' '\n' | grep -E '^--'
}

# 의존성 확인
check_dependencies() {
    echo "Checking build dependencies..."
    
    local required_packages=("build-essential" "libpcre3-dev" "libssl-dev" "zlib1g-dev")
    
    for package in "${required_packages[@]}"; do
        if ! dpkg -l | grep -q "^ii.*$package"; then
            echo "Installing missing dependency: $package"
            apt-get update && apt-get install -y "$package"
        fi
    done
}

# 새 버전 컴파일
compile_new_version() {
    echo "Compiling Nginx $NGINX_VERSION_TARGET..."
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # 소스 다운로드
    wget "http://nginx.org/download/nginx-${NGINX_VERSION_TARGET}.tar.gz"
    tar -xzf "nginx-${NGINX_VERSION_TARGET}.tar.gz"
    cd "nginx-${NGINX_VERSION_TARGET}"
    
    # 현재 컴파일 옵션 가져오기
    local current_config=$(nginx -V 2>&1 | grep "configure arguments:" | cut -d: -f2-)
    
    # 컴파일
    echo "Using configuration: $current_config"
    ./configure $current_config
    
    make -j$(nproc)
    
    if [ $? -eq 0 ]; then
        echo "✓ Compilation successful"
        return 0
    else
        echo "✗ Compilation failed"
        return 1
    fi
}

# 무중단 업그레이드 실행
perform_hot_upgrade() {
    echo "Starting hot upgrade process..."
    
    # 기존 바이너리 백업
    cp $(which nginx) "$BACKUP_DIR/nginx.old.$(date +%Y%m%d_%H%M%S)"
    
    # 새 바이너리 설치
    local new_binary="$BUILD_DIR/nginx-${NGINX_VERSION_TARGET}/objs/nginx"
    cp "$new_binary" $(which nginx)
    
    # 마스터 프로세스에 USR2 시그널 전송
    local old_pid=$(cat /var/run/nginx.pid)
    echo "Sending USR2 signal to old master process (PID: $old_pid)"
    kill -USR2 $old_pid
    
    # 새 마스터 프로세스 확인
    sleep 2
    local new_pid_file="/var/run/nginx.pid.oldbin"
    
    if [ -f "$new_pid_file" ]; then
        echo "✓ New master process started"
        
        # 새 프로세스 확인
        local new_pid=$(cat /var/run/nginx.pid)
        echo "New master PID: $new_pid"
        
        # 기존 워커 프로세스 종료
        echo "Shutting down old worker processes..."
        kill -WINCH $old_pid
        
        # 잠시 대기 후 상태 확인
        sleep 5
        
        # 새 프로세스가 정상 작동하는지 확인
        if curl -s http://localhost/nginx_status > /dev/null 2>&1; then
            echo "✓ New processes are working correctly"
            
            # 기존 마스터 프로세스 종료
            echo "Terminating old master process..."
            kill -QUIT $old_pid
            
            echo "✓ Hot upgrade completed successfully"
            
            # 새 버전 확인
            echo "New version: $(nginx -V 2>&1 | head -n1)"
        else
            echo "✗ New processes not responding correctly"
            echo "Rolling back..."
            kill -HUP $old_pid  # 기존 워커들 재시작
            kill -TERM $new_pid # 새 마스터 종료
            return 1
        fi
    else
        echo "✗ New master process failed to start"
        return 1
    fi
}

# 업그레이드 검증
verify_upgrade() {
    echo "Verifying upgrade..."
    
    # 버전 확인
    local current_version=$(nginx -v 2>&1 | cut -d/ -f2)
    if [ "$current_version" = "$NGINX_VERSION_TARGET" ]; then
        echo "✓ Version upgrade verified: $current_version"
    else
        echo "✗ Version mismatch: expected $NGINX_VERSION_TARGET, got $current_version"
        return 1
    fi
    
    # 설정 테스트
    if nginx -t; then
        echo "✓ Configuration test passed"
    else
        echo "✗ Configuration test failed"
        return 1
    fi
    
    # 서비스 상태 확인
    if systemctl is-active nginx >/dev/null; then
        echo "✓ Nginx service is active"
    else
        echo "✗ Nginx service is not active"
        return 1
    fi
    
    # HTTP 응답 확인
    local http_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/)
    if [ "$http_status" = "200" ] || [ "$http_status" = "301" ] || [ "$http_status" = "302" ]; then
        echo "✓ HTTP response check passed (status: $http_status)"
    else
        echo "✗ HTTP response check failed (status: $http_status)"
        return 1
    fi
}

# 롤백 함수
rollback_upgrade() {
    echo "Performing rollback..."
    
    local backup_binary=$(ls -t "$BACKUP_DIR"/nginx.old.* | head -n1)
    
    if [ -f "$backup_binary" ]; then
        systemctl stop nginx
        cp "$backup_binary" $(which nginx)
        systemctl start nginx
        
        if systemctl is-active nginx >/dev/null; then
            echo "✓ Rollback completed successfully"
        else
            echo "✗ Rollback failed"
        fi
    else
        echo "✗ No backup binary found for rollback"
    fi
}

# 메인 실행
main() {
    case "$1" in
        "check")
            check_current_setup
            ;;
        "prepare")
            check_dependencies
            compile_new_version
            ;;
        "upgrade")
            mkdir -p "$BACKUP_DIR"
            check_current_setup
            check_dependencies
            compile_new_version && perform_hot_upgrade && verify_upgrade
            ;;
        "verify")
            verify_upgrade
            ;;
        "rollback")
            rollback_upgrade
            ;;
        *)
            echo "Usage: $0 {check|prepare|upgrade|verify|rollback}"
            echo "  check    - Show current setup"
            echo "  prepare  - Prepare new version (compile)"
            echo "  upgrade  - Perform hot upgrade"
            echo "  verify   - Verify upgrade success"
            echo "  rollback - Rollback to previous version"
            exit 1
            ;;
    esac
}

main "$@"
```

## 장애 대응 절차

### 자동 장애 감지 및 복구
```bash
#!/bin/bash
# /usr/local/bin/nginx-health-monitor.sh

HEALTH_CHECK_URL="http://localhost/nginx_status"
ALERT_EMAIL="admin@example.com"
LOG_FILE="/var/log/nginx/health-monitor.log"
MAX_FAILURES=3
FAILURE_COUNT=0
CHECK_INTERVAL=30

# 건강 상태 체크 함수들
check_nginx_process() {
    if pgrep nginx > /dev/null; then
        return 0
    else
        return 1
    fi
}

check_nginx_response() {
    local response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "$HEALTH_CHECK_URL" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        return 0
    else
        echo "HTTP response error: $response" | tee -a "$LOG_FILE"
        return 1
    fi
}

check_configuration() {
    if nginx -t 2>/dev/null; then
        return 0
    else
        echo "Configuration test failed" | tee -a "$LOG_FILE"
        nginx -t 2>&1 | tee -a "$LOG_FILE"
        return 1
    fi
}

check_disk_space() {
    local usage=$(df /var/log/nginx | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$usage" -gt 90 ]; then
        echo "Disk space critical: ${usage}% used" | tee -a "$LOG_FILE"
        return 1
    fi
    return 0
}

check_memory_usage() {
    local nginx_memory=$(ps -C nginx -o %mem --no-headers | awk '{sum += $1} END {print sum}')
    
    if (( $(echo "$nginx_memory > 80" | bc -l) )); then
        echo "Memory usage high: ${nginx_memory}%" | tee -a "$LOG_FILE"
        return 1
    fi
    return 0
}

# 자동 복구 함수들
attempt_nginx_restart() {
    echo "Attempting to restart Nginx..." | tee -a "$LOG_FILE"
    
    systemctl restart nginx
    sleep 5
    
    if check_nginx_response; then
        echo "✓ Nginx restart successful" | tee -a "$LOG_FILE"
        return 0
    else
        echo "✗ Nginx restart failed" | tee -a "$LOG_FILE"
        return 1
    fi
}

attempt_config_reload() {
    echo "Attempting configuration reload..." | tee -a "$LOG_FILE"
    
    if check_configuration; then
        systemctl reload nginx
        sleep 2
        
        if check_nginx_response; then
            echo "✓ Configuration reload successful" | tee -a "$LOG_FILE"
            return 0
        fi
    fi
    
    echo "✗ Configuration reload failed" | tee -a "$LOG_FILE"
    return 1
}

cleanup_logs() {
    echo "Cleaning up old log files..." | tee -a "$LOG_FILE"
    
    # 30일 이상된 로그 파일 삭제
    find /var/log/nginx -name "*.log" -mtime +30 -delete
    
    # 압축된 로그 파일 중 90일 이상된 것 삭제
    find /var/log/nginx -name "*.gz" -mtime +90 -delete
    
    echo "✓ Log cleanup completed" | tee -a "$LOG_FILE"
}

# 알림 발송
send_alert() {
    local alert_type=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 로그 기록
    echo "[$timestamp] ALERT: $alert_type - $message" | tee -a "$LOG_FILE"
    
    # 이메일 알림
    if command -v mail >/dev/null 2>&1; then
        echo "Nginx Health Monitor Alert

Time: $timestamp
Type: $alert_type
Message: $message

Server: $(hostname)
" | mail -s "Nginx Alert: $alert_type" "$ALERT_EMAIL"
    fi
    
    # Slack 알림 (웹훅 URL이 설정된 경우)
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Nginx Alert: $alert_type\\n$message\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null
    fi
}

# 메인 건강 체크 루프
main_health_check() {
    local all_checks_passed=true
    
    # 프로세스 체크
    if ! check_nginx_process; then
        echo "Nginx process not running" | tee -a "$LOG_FILE"
        all_checks_passed=false
    fi
    
    # HTTP 응답 체크
    if ! check_nginx_response; then
        all_checks_passed=false
    fi
    
    # 설정 체크
    if ! check_configuration; then
        all_checks_passed=false
    fi
    
    # 디스크 공간 체크
    if ! check_disk_space; then
        cleanup_logs
        all_checks_passed=false
    fi
    
    # 메모리 사용량 체크
    if ! check_memory_usage; then
        all_checks_passed=false
    fi
    
    if [ "$all_checks_passed" = true ]; then
        if [ $FAILURE_COUNT -gt 0 ]; then
            echo "Health check recovered" | tee -a "$LOG_FILE"
            send_alert "RECOVERY" "All health checks are now passing"
        fi
        FAILURE_COUNT=0
        return 0
    else
        ((FAILURE_COUNT++))
        echo "Health check failed (failure count: $FAILURE_COUNT)" | tee -a "$LOG_FILE"
        
        # 자동 복구 시도
        if [ $FAILURE_COUNT -le $MAX_FAILURES ]; then
            echo "Attempting automatic recovery..." | tee -a "$LOG_FILE"
            
            # 설정 리로드 시도
            if attempt_config_reload; then
                FAILURE_COUNT=0
                return 0
            fi
            
            # Nginx 재시작 시도
            if attempt_nginx_restart; then
                FAILURE_COUNT=0
                return 0
            fi
        fi
        
        # 최대 실패 횟수 도달 시 알림
        if [ $FAILURE_COUNT -ge $MAX_FAILURES ]; then
            send_alert "CRITICAL" "Nginx health check failed $FAILURE_COUNT times. Manual intervention required."
        fi
        
        return 1
    fi
}

# 데몬 모드로 실행
if [ "$1" = "daemon" ]; then
    echo "Starting Nginx health monitor daemon..." | tee -a "$LOG_FILE"
    
    while true; do
        main_health_check
        sleep $CHECK_INTERVAL
    done
else
    # 단일 실행 모드
    main_health_check
fi
```

### 장애 상황별 대응 가이드
```bash
#!/bin/bash
# /usr/local/bin/nginx-troubleshoot.sh

# 일반적인 문제 진단 및 해결
diagnose_connection_issues() {
    echo "=== Connection Issues Diagnosis ==="
    
    # 포트 확인
    echo "Checking port bindings..."
    netstat -tlpn | grep -E ':80|:443'
    
    # 방화벽 확인
    echo "Checking firewall rules..."
    if command -v ufw >/dev/null 2>&1; then
        ufw status
    elif command -v firewall-cmd >/dev/null 2>&1; then
        firewall-cmd --list-ports
        firewall-cmd --list-services
    fi
    
    # 프로세스 상태 확인
    echo "Checking Nginx processes..."
    ps aux | grep nginx
    
    # 로그에서 에러 확인
    echo "Recent error log entries..."
    tail -20 /var/log/nginx/error.log
}

diagnose_performance_issues() {
    echo "=== Performance Issues Diagnosis ==="
    
    # 리소스 사용량
    echo "System resource usage:"
    top -bn1 | grep -E "Cpu|Mem|nginx"
    
    # 연결 통계
    echo "Connection statistics:"
    if curl -s http://localhost/nginx_status > /dev/null 2>&1; then
        curl -s http://localhost/nginx_status
    fi
    
    # 네트워크 연결 수
    echo "Network connections:"
    ss -tuln | grep -E ':80|:443' | wc -l
    
    # 디스크 I/O
    echo "Disk I/O statistics:"
    iostat -x 1 3 2>/dev/null || echo "iostat not available"
}

diagnose_ssl_issues() {
    echo "=== SSL Issues Diagnosis ==="
    
    # 인증서 유효성 확인
    echo "Checking SSL certificates..."
    
    for domain in $(nginx -T 2>/dev/null | grep "server_name" | awk '{print $2}' | grep -v ";" | sort -u); do
        if [ "$domain" != "_" ] && [ -n "$domain" ]; then
            echo "Checking certificate for: $domain"
            
            # 로컬 인증서 정보
            echo | openssl s_client -connect "$domain:443" -servername "$domain" 2>/dev/null | \
            openssl x509 -noout -dates 2>/dev/null || echo "Cannot connect to $domain"
            
            # 만료일 확인
            local cert_file=$(nginx -T 2>/dev/null | grep -A 5 "server_name.*$domain" | \
                             grep "ssl_certificate " | head -1 | awk '{print $2}' | sed 's/;//')
            
            if [ -f "$cert_file" ]; then
                echo "Certificate file: $cert_file"
                openssl x509 -in "$cert_file" -noout -dates
            fi
        fi
    done
}

fix_common_issues() {
    echo "=== Attempting Common Fixes ==="
    
    # 권한 문제 해결
    echo "Fixing file permissions..."
    chown -R www-data:www-data /var/www/
    chmod -R 755 /var/www/
    
    # 로그 디렉토리 권한
    chown -R www-data:adm /var/log/nginx/
    chmod -R 755 /var/log/nginx/
    
    # PID 파일 문제 해결
    if [ ! -f /var/run/nginx.pid ]; then
        echo "Recreating PID file..."
        rm -f /var/run/nginx.pid
        systemctl restart nginx
    fi
    
    # 설정 구문 오류 수정 시도
    echo "Checking configuration syntax..."
    if ! nginx -t; then
        echo "Configuration has errors. Please fix manually."
        nginx -t 2>&1
    fi
}

generate_diagnostic_report() {
    local report_file="/tmp/nginx_diagnostic_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "Generating diagnostic report: $report_file"
    
    cat > "$report_file" << EOF
Nginx Diagnostic Report
=====================
Generated: $(date)
Hostname: $(hostname)
Nginx Version: $(nginx -v 2>&1)

=== System Information ===
$(uname -a)
$(lsb_release -a 2>/dev/null || cat /etc/os-release)

=== Nginx Status ===
Service Status: $(systemctl is-active nginx)
Process Count: $(pgrep nginx | wc -l)

=== Configuration Test ===
$(nginx -t 2>&1)

=== Process Information ===
$(ps aux | grep nginx | grep -v grep)

=== Network Status ===
$(netstat -tlpn | grep nginx)

=== Recent Error Logs ===
$(tail -50 /var/log/nginx/error.log)

=== Resource Usage ===
$(free -h)
$(df -h)

=== Configuration Summary ===
$(nginx -T 2>/dev/null | grep -E "(server_name|listen|root|proxy_pass)" | head -20)
EOF

    echo "Report generated: $report_file"
    echo "You can send this report to support team for analysis"
}

# 메인 메뉴
case "$1" in
    "connection")
        diagnose_connection_issues
        ;;
    "performance")
        diagnose_performance_issues
        ;;
    "ssl")
        diagnose_ssl_issues
        ;;
    "fix")
        fix_common_issues
        ;;
    "report")
        generate_diagnostic_report
        ;;
    "all")
        diagnose_connection_issues
        echo ""
        diagnose_performance_issues
        echo ""
        diagnose_ssl_issues
        echo ""
        generate_diagnostic_report
        ;;
    *)
        echo "Nginx Troubleshooting Tool"
        echo "Usage: $0 {connection|performance|ssl|fix|report|all}"
        echo ""
        echo "  connection   - Diagnose connection issues"
        echo "  performance  - Diagnose performance issues"
        echo "  ssl         - Diagnose SSL/TLS issues"
        echo "  fix         - Attempt common fixes"
        echo "  report      - Generate diagnostic report"
        echo "  all         - Run all diagnostics"
        exit 1
        ;;
esac
```

## 성능 튜닝

### 시스템 레벨 최적화
```bash
#!/bin/bash
# /usr/local/bin/nginx-performance-tuning.sh

# 시스템 한계값 최적화
optimize_system_limits() {
    echo "Optimizing system limits for Nginx..."
    
    # /etc/security/limits.conf 설정
    cat >> /etc/security/limits.conf << 'EOF'
# Nginx optimization
nginx soft nofile 65536
nginx hard nofile 65536
nginx soft nproc 32768
nginx hard nproc 32768
www-data soft nofile 65536
www-data hard nofile 65536
EOF

    # systemd 서비스 한계값
    mkdir -p /etc/systemd/system/nginx.service.d
    cat > /etc/systemd/system/nginx.service.d/limits.conf << 'EOF'
[Service]
LimitNOFILE=65536
LimitNPROC=32768
EOF

    systemctl daemon-reload
    
    echo "✓ System limits optimized"
}

# 커널 파라미터 튜닝
optimize_kernel_parameters() {
    echo "Optimizing kernel parameters..."
    
    cat >> /etc/sysctl.conf << 'EOF'
# Nginx performance tuning
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1800
net.ipv4.tcp_keepalive_probes = 7
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.ip_local_port_range = 1024 65535
vm.swappiness = 10
EOF

    sysctl -p
    echo "✓ Kernel parameters optimized"
}

# Nginx 설정 최적화
generate_optimized_config() {
    echo "Generating optimized Nginx configuration..."
    
    local cpu_cores=$(nproc)
    local total_memory=$(free -m | awk 'NR==2{print $2}')
    
    cat > /etc/nginx/conf.d/performance.conf << EOF
# Performance optimized configuration
# Generated on $(date)

# Worker process optimization
worker_processes $cpu_cores;
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;

# Event processing optimization
events {
    worker_connections 8192;
    use epoll;
    multi_accept on;
    accept_mutex off;
}

# HTTP optimization
http {
    # Basic optimization
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    
    # Timeout optimization
    keepalive_timeout 65;
    keepalive_requests 1000;
    client_header_timeout 60;
    client_body_timeout 60;
    send_timeout 60;
    
    # Buffer optimization
    client_body_buffer_size 128k;
    client_header_buffer_size 4k;
    large_client_header_buffers 4 8k;
    client_max_body_size 50m;
    
    # Hash table optimization
    server_names_hash_bucket_size 128;
    server_names_hash_max_size 2048;
    types_hash_max_size 2048;
    
    # Gzip optimization
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;
    
    # Rate limiting zones
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=1r/s;
    
    # Connection limiting
    limit_conn_zone \$binary_remote_addr zone=perip:10m;
    limit_conn_zone \$server_name zone=perserver:10m;
}
EOF

    echo "✓ Optimized configuration generated"
}

# 프록시 최적화
optimize_proxy_settings() {
    echo "Optimizing proxy settings..."
    
    cat > /etc/nginx/conf.d/proxy_optimization.conf << 'EOF'
# Proxy optimization
proxy_buffering on;
proxy_buffer_size 128k;
proxy_buffers 4 256k;
proxy_busy_buffers_size 256k;
proxy_temp_file_write_size 256k;
proxy_max_temp_file_size 1024m;

# Proxy timeouts
proxy_connect_timeout 30s;
proxy_send_timeout 30s;
proxy_read_timeout 30s;

# Proxy headers
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# Connection reuse
proxy_http_version 1.1;
proxy_set_header Connection "";

# Cache optimization
proxy_cache_path /var/cache/nginx/proxy 
                 levels=1:2 
                 keys_zone=proxy_cache:10m 
                 max_size=1g 
                 inactive=60m 
                 use_temp_path=off;

proxy_cache_key "$scheme$request_method$host$request_uri$is_args$args";
proxy_cache_valid 200 302 10m;
proxy_cache_valid 301 1h;
proxy_cache_valid any 1m;
EOF

    mkdir -p /var/cache/nginx/proxy
    chown -R www-data:www-data /var/cache/nginx
    
    echo "✓ Proxy settings optimized"
}

# 성능 테스트
run_performance_test() {
    echo "Running performance tests..."
    
    # 기본 응답 시간 테스트
    echo "Basic response time test:"
    for i in {1..10}; do
        curl -s -o /dev/null -w "Request $i: %{time_total}s\n" http://localhost/
    done
    
    # 동시 연결 테스트 (ab가 설치된 경우)
    if command -v ab >/dev/null 2>&1; then
        echo "Concurrent connection test:"
        ab -n 1000 -c 10 http://localhost/ | grep -E "Requests per second|Time per request"
    fi
    
    # wrk 테스트 (wrk가 설치된 경우)
    if command -v wrk >/dev/null 2>&1; then
        echo "Load test with wrk:"
        wrk -t4 -c100 -d30s http://localhost/
    fi
}

# 메인 실행
case "$1" in
    "system")
        optimize_system_limits
        optimize_kernel_parameters
        ;;
    "nginx")
        generate_optimized_config
        optimize_proxy_settings
        nginx -t && systemctl reload nginx
        ;;
    "test")
        run_performance_test
        ;;
    "all")
        optimize_system_limits
        optimize_kernel_parameters
        generate_optimized_config
        optimize_proxy_settings
        nginx -t && systemctl reload nginx
        echo "Optimization completed. Reboot recommended for kernel changes."
        ;;
    *)
        echo "Nginx Performance Tuning Tool"
        echo "Usage: $0 {system|nginx|test|all}"
        echo ""
        echo "  system - Optimize system limits and kernel parameters"
        echo "  nginx  - Generate optimized Nginx configuration"
        echo "  test   - Run performance tests"
        echo "  all    - Apply all optimizations"
        exit 1
        ;;
esac
```

## 고가용성 구성

### Keepalived를 이용한 HA 구성
```bash
#!/bin/bash
# /usr/local/bin/setup-nginx-ha.sh

# HA 구성 설정
setup_keepalived() {
    echo "Setting up Keepalived for Nginx HA..."
    
    # Keepalived 설치
    apt-get update && apt-get install -y keepalived
    
    # VIP와 우선순위 설정
    read -p "Enter Virtual IP address: " VIRTUAL_IP
    read -p "Enter interface name (e.g., eth0): " INTERFACE
    read -p "Enter priority (100 for master, 90 for backup): " PRIORITY
    
    # Keepalived 설정 파일 생성
    cat > /etc/keepalived/keepalived.conf << EOF
global_defs {
    router_id LVS_$(hostname)
    vrrp_skip_check_adv_addr
    vrrp_strict
    vrrp_garp_interval 0
    vrrp_gna_interval 0
}

vrrp_script chk_nginx {
    script "/usr/local/bin/check_nginx.sh"
    interval 2
    weight -2
    fall 3
    rise 2
}

vrrp_instance VI_1 {
    state $([ $PRIORITY -eq 100 ] && echo "MASTER" || echo "BACKUP")
    interface $INTERFACE
    virtual_router_id 51
    priority $PRIORITY
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass nginx_ha_pass
    }
    virtual_ipaddress {
        $VIRTUAL_IP
    }
    track_script {
        chk_nginx
    }
    notify_master "/usr/local/bin/nginx_master.sh"
    notify_backup "/usr/local/bin/nginx_backup.sh"
    notify_fault "/usr/local/bin/nginx_fault.sh"
}
EOF

    # Nginx 상태 체크 스크립트
    cat > /usr/local/bin/check_nginx.sh << 'EOF'
#!/bin/bash
curl -f http://localhost/nginx_status >/dev/null 2>&1
exit $?
EOF

    # Master 상태 스크립트
    cat > /usr/local/bin/nginx_master.sh << 'EOF'
#!/bin/bash
echo "$(date): Becoming MASTER" >> /var/log/keepalived-nginx.log
# 마스터로 전환 시 수행할 작업
systemctl restart nginx
EOF

    # Backup 상태 스크립트
    cat > /usr/local/bin/nginx_backup.sh << 'EOF'
#!/bin/bash
echo "$(date): Becoming BACKUP" >> /var/log/keepalived-nginx.log
# 백업으로 전환 시 수행할 작업
EOF

    # Fault 상태 스크립트
    cat > /usr/local/bin/nginx_fault.sh << 'EOF'
#!/bin/bash
echo "$(date): FAULT detected" >> /var/log/keepalived-nginx.log
# 장애 발생 시 알림
mail -s "Nginx HA FAULT on $(hostname)" admin@example.com < /dev/null
EOF

    # 스크립트 실행 권한 부여
    chmod +x /usr/local/bin/check_nginx.sh
    chmod +x /usr/local/bin/nginx_master.sh
    chmod +x /usr/local/bin/nginx_backup.sh
    chmod +x /usr/local/bin/nginx_fault.sh
    
    # Keepalived 시작
    systemctl enable keepalived
    systemctl start keepalived
    
    echo "✓ Keepalived setup completed"
    echo "Virtual IP: $VIRTUAL_IP"
    echo "Priority: $PRIORITY"
}

# 로드 밸런서 풀 구성
setup_upstream_pool() {
    echo "Setting up upstream server pool..."
    
    read -p "Enter backend servers (comma-separated, e.g., 192.168.1.10:8080,192.168.1.11:8080): " BACKEND_SERVERS
    
    cat > /etc/nginx/conf.d/upstream.conf << EOF
upstream backend_pool {
    least_conn;
    
    # Backend servers
$(echo "$BACKEND_SERVERS" | tr ',' '\n' | while read server; do
    echo "    server $server max_fails=3 fail_timeout=30s;"
done)
    
    # Health check (nginx-plus feature)
    # health_check;
    
    # Session persistence
    # ip_hash;
    
    # Keepalive connections
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://backend_pool;
        
        # Health check endpoint
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
        proxy_next_upstream_timeout 30s;
        
        # Headers
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Connection reuse
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # Status page for health checking
    location /nginx_status {
        stub_status on;
        allow 127.0.0.1;
        allow 192.168.0.0/16;
        deny all;
        access_log off;
    }
}
EOF

    nginx -t && systemctl reload nginx
    echo "✓ Upstream pool configured"
}

# 상태 모니터링
setup_monitoring() {
    echo "Setting up HA monitoring..."
    
    cat > /usr/local/bin/ha_monitor.sh << 'EOF'
#!/bin/bash

LOG_FILE="/var/log/nginx-ha-monitor.log"

check_vip() {
    local vip=$(ip addr show | grep "inet.*scope global secondary" | awk '{print $2}' | cut -d'/' -f1)
    if [ -n "$vip" ]; then
        echo "$(date): VIP active - $vip" >> $LOG_FILE
        return 0
    else
        echo "$(date): VIP not active" >> $LOG_FILE
        return 1
    fi
}

check_keepalived() {
    if systemctl is-active keepalived >/dev/null; then
        return 0
    else
        echo "$(date): Keepalived service down" >> $LOG_FILE
        systemctl start keepalived
        return 1
    fi
}

check_backend_health() {
    echo "$(date): Checking backend health" >> $LOG_FILE
    
    # Parse upstream configuration
    nginx -T 2>/dev/null | grep "server.*:" | while read line; do
        local server=$(echo $line | awk '{print $2}' | sed 's/;//')
        local host=$(echo $server | cut -d':' -f1)
        local port=$(echo $server | cut -d':' -f2)
        
        if nc -z $host $port; then
            echo "$(date): Backend $server - OK" >> $LOG_FILE
        else
            echo "$(date): Backend $server - FAILED" >> $LOG_FILE
        fi
    done
}

# Main monitoring loop
check_keepalived
check_vip
check_backend_health
EOF

    chmod +x /usr/local/bin/ha_monitor.sh
    
    # Cron 작업 추가
    echo "*/1 * * * * root /usr/local/bin/ha_monitor.sh" >> /etc/crontab
    
    echo "✓ HA monitoring setup completed"
}

# 메인 실행
case "$1" in
    "keepalived")
        setup_keepalived
        ;;
    "upstream")
        setup_upstream_pool
        ;;
    "monitoring")
        setup_monitoring
        ;;
    "all")
        setup_keepalived
        setup_upstream_pool
        setup_monitoring
        ;;
    *)
        echo "Nginx High Availability Setup"
        echo "Usage: $0 {keepalived|upstream|monitoring|all}"
        echo ""
        echo "  keepalived  - Setup Keepalived for VIP management"
        echo "  upstream    - Configure upstream server pool"
        echo "  monitoring  - Setup HA monitoring"
        echo "  all         - Setup complete HA solution"
        exit 1
        ;;
esac
```

## 자동화 및 최적화

### 종합 운영 자동화 스크립트
```bash
#!/bin/bash
# /usr/local/bin/nginx-ops-automation.sh

SCRIPT_DIR="/usr/local/bin"
CONFIG_DIR="/etc/nginx"
LOG_DIR="/var/log/nginx"
BACKUP_DIR="/backup/nginx"

# 일일 운영 작업
daily_operations() {
    echo "=== Daily Nginx Operations $(date) ===" | tee -a "$LOG_DIR/operations.log"
    
    # 1. 백업 실행
    echo "Performing daily backup..."
    $SCRIPT_DIR/nginx-backup.sh
    
    # 2. 로그 분석
    echo "Analyzing logs..."
    $SCRIPT_DIR/nginx-log-analyzer.sh all >> "$LOG_DIR/daily-analysis.log"
    
    # 3. 성능 메트릭 수집
    echo "Collecting performance metrics..."
    $SCRIPT_DIR/nginx-metrics.sh >> "$LOG_DIR/daily-metrics.log"
    
    # 4. 건강 상태 체크
    echo "Running health checks..."
    $SCRIPT_DIR/nginx-health-monitor.sh >> "$LOG_DIR/health-check.log"
    
    # 5. 설정 검증
    echo "Validating configuration..."
    if nginx -t; then
        echo "✓ Configuration valid"
    else
        echo "✗ Configuration has errors" | tee -a "$LOG_DIR/operations.log"
        nginx -t 2>&1 | tee -a "$LOG_DIR/operations.log"
    fi
    
    # 6. 디스크 공간 정리
    echo "Cleaning up disk space..."
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete
    find "$BACKUP_DIR" -type f -mtime +90 -delete
    
    # 7. 보고서 생성
    generate_daily_report
    
    echo "Daily operations completed" | tee -a "$LOG_DIR/operations.log"
}

generate_daily_report() {
    local report_file="$LOG_DIR/daily-report-$(date +%Y%m%d).txt"
    
    cat > "$report_file" << EOF
Nginx Daily Operations Report
===========================
Date: $(date)
Server: $(hostname)

=== System Status ===
Nginx Version: $(nginx -v 2>&1)
Service Status: $(systemctl is-active nginx)
Uptime: $(uptime)
Load Average: $(cat /proc/loadavg)

=== Traffic Statistics ===
$(tail -10000 "$LOG_DIR/access.log" | awk '{
    total++
    if ($9 ~ /^2/) success++
    else if ($9 ~ /^4/) client_error++
    else if ($9 ~ /^5/) server_error++
}
END {
    print "Total Requests: " total
    print "Success Rate: " (success/total*100) "%"
    print "Client Errors: " client_error " (" (client_error/total*100) "%)"
    print "Server Errors: " server_error " (" (server_error/total*100) "%)"
}')

=== Performance Metrics ===
$(curl -s http://localhost/nginx_status 2>/dev/null || echo "Status page not available")

=== Recent Errors ===
$(tail -10 "$LOG_DIR/error.log")

=== Disk Usage ===
$(df -h | grep -E "/$|/var")

=== Memory Usage ===
$(free -h)
EOF

    echo "Daily report generated: $report_file"
    
    # 이메일로 보고서 전송
    if command -v mail >/dev/null 2>&1; then
        mail -s "Nginx Daily Report - $(hostname)" admin@example.com < "$report_file"
    fi
}

# 주간 운영 작업
weekly_operations() {
    echo "=== Weekly Nginx Operations $(date) ===" | tee -a "$LOG_DIR/operations.log"
    
    # 1. 전체 시스템 백업
    echo "Performing full system backup..."
    tar -czf "$BACKUP_DIR/weekly/full-backup-$(date +%Y%m%d).tar.gz" \
        "$CONFIG_DIR" "$LOG_DIR" /var/www 2>/dev/null
    
    # 2. 보안 업데이트 확인
    echo "Checking for security updates..."
    apt list --upgradable 2>/dev/null | grep nginx || echo "No nginx updates available"
    
    # 3. SSL 인증서 만료 확인
    echo "Checking SSL certificate expiration..."
    find /etc/letsencrypt/live -name "cert.pem" -exec openssl x509 -in {} -noout -dates \; 2>/dev/null
    
    # 4. 성능 최적화 검토
    echo "Performance optimization review..."
    $SCRIPT_DIR/nginx-performance-tuning.sh test
    
    # 5. 로그 아카이브
    echo "Archiving old logs..."
    find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \;
    
    echo "Weekly operations completed" | tee -a "$LOG_DIR/operations.log"
}

# 응급 상황 대응
emergency_response() {
    echo "=== Emergency Response Activated $(date) ===" | tee -a "$LOG_DIR/emergency.log"
    
    # 1. 현재 상태 스냅샷
    echo "Taking system snapshot..."
    $SCRIPT_DIR/nginx-troubleshoot.sh report
    
    # 2. 자동 복구 시도
    echo "Attempting automatic recovery..."
    $SCRIPT_DIR/nginx-health-monitor.sh
    
    # 3. 긴급 알림 발송
    echo "Sending emergency alerts..."
    echo "Emergency response activated on $(hostname) at $(date)" | \
    mail -s "EMERGENCY: Nginx Issue on $(hostname)" admin@example.com
    
    # 4. 백업 설정으로 전환
    if [ -f "$BACKUP_DIR/last_known_good/nginx.conf" ]; then
        echo "Reverting to last known good configuration..."
        cp "$BACKUP_DIR/last_known_good/nginx.conf" "$CONFIG_DIR/nginx.conf"
        nginx -t && systemctl reload nginx
    fi
    
    echo "Emergency response completed" | tee -a "$LOG_DIR/emergency.log"
}

# 설정 변경 시 자동 검증
validate_and_deploy() {
    local config_file=$1
    
    echo "Validating and deploying configuration changes..."
    
    # 백업 생성
    cp "$CONFIG_DIR/nginx.conf" "$BACKUP_DIR/last_known_good/nginx.conf.$(date +%Y%m%d_%H%M%S)"
    
    # 새 설정 적용
    if [ -f "$config_file" ]; then
        cp "$config_file" "$CONFIG_DIR/nginx.conf"
    fi
    
    # 검증
    if nginx -t; then
        echo "✓ Configuration validation passed"
        
        # 리로드
        systemctl reload nginx
        
        if systemctl is-active nginx >/dev/null; then
            echo "✓ Configuration deployed successfully"
            
            # 최신 정상 설정으로 업데이트
            cp "$CONFIG_DIR/nginx.conf" "$BACKUP_DIR/last_known_good/nginx.conf"
        else
            echo "✗ Service failed to reload, reverting..."
            cp "$BACKUP_DIR/last_known_good/nginx.conf" "$CONFIG_DIR/nginx.conf"
            systemctl reload nginx
        fi
    else
        echo "✗ Configuration validation failed, reverting..."
        cp "$BACKUP_DIR/last_known_good/nginx.conf" "$CONFIG_DIR/nginx.conf"
    fi
}

# 메인 실행
case "$1" in
    "daily")
        daily_operations
        ;;
    "weekly")
        weekly_operations
        ;;
    "emergency")
        emergency_response
        ;;
    "deploy")
        validate_and_deploy "$2"
        ;;
    "status")
        echo "=== Current Nginx Status ==="
        systemctl status nginx
        echo ""
        nginx -t
        echo ""
        curl -s http://localhost/nginx_status || echo "Status page not available"
        ;;
    *)
        echo "Nginx Operations Automation"
        echo "Usage: $0 {daily|weekly|emergency|deploy <config_file>|status}"
        echo ""
        echo "  daily     - Run daily maintenance operations"
        echo "  weekly    - Run weekly maintenance operations"
        echo "  emergency - Run emergency response procedures"
        echo "  deploy    - Deploy and validate new configuration"
        echo "  status    - Show current nginx status"
        exit 1
        ;;
esac
```

## 다음 단계

이제 운영 가이드가 완료되었습니다. 다음 포스트에서는 보안 가이드를 시작하겠습니다:

- 기본 보안 설정 및 헤더 보안
- 접근 제어 및 인증 시스템
- DDoS 방어 및 Rate Limiting
- SSL/TLS 보안 강화

## 참고 자료

- [Nginx Admin Guide - High Availability](https://docs.nginx.com/nginx/admin-guide/high-availability/)
- [Keepalived Documentation](https://keepalived.readthedocs.io/)
- [Nginx Performance Tuning](https://www.nginx.com/blog/tuning-nginx/)
---
layout: post
title: "리눅스 보안 완전 가이드 4편 - 침입 탐지와 실시간 모니터링 | Linux Security Guide Part 4 - Intrusion Detection & Real-time Monitoring"
date: 2025-04-12 09:00:00 +0900
categories: [Security, Linux]
tags: [intrusion-detection, aide, fail2ban, wazuh, siem, monitoring, threat-detection, log-analysis]
---

리눅스 시스템의 완벽한 보안을 위한 마지막 퍼즐 조각인 침입 탐지와 실시간 모니터링을 완전히 마스터해보겠습니다. AIDE부터 Wazuh SIEM까지, 모든 위협을 실시간으로 탐지하고 대응하는 최고급 보안 시스템을 구축합니다.

## AIDE 파일 무결성 모니터링 | AIDE File Integrity Monitoring

### 🔍 AIDE 완전 설정 및 운영

#### AIDE 설치 및 기본 설정
```bash
# AIDE 설치
# CentOS/RHEL
yum install aide
# Ubuntu/Debian  
apt-get install aide aide-common

# 초기 데이터베이스 생성
aide --init
mv /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz

# 기본 무결성 검사 실행
aide --check

# 설정 파일 위치
# CentOS/RHEL: /etc/aide.conf
# Ubuntu/Debian: /etc/aide/aide.conf
```

#### 고급 AIDE 설정 파일 작성
```bash
# /etc/aide/aide.conf (Ubuntu) 또는 /etc/aide.conf (CentOS)
# 종합적인 AIDE 설정

# 데이터베이스 경로
database=file:/var/lib/aide/aide.db.gz
database_out=file:/var/lib/aide/aide.db.new.gz

# 로그 설정
verbose=5
report_url=file:/var/log/aide/aide.log
report_url=stdout

# 검사 규칙 정의
# R = p+i+n+u+g+s+m+c+md5
# L = p+i+n+u+g
# E = 빈 그룹 (존재만 확인)
# > = 로그 파일 (크기 증가만 허용)
# X = 제외

# 기본 규칙들
Binlib = p+i+n+u+g+s+b+m+c+md5+sha1+sha256+sha512
ConfFiles = p+i+n+u+g+s+m+c+md5+sha1+sha256
LogFiles = p+u+g+i+n+S
DatabaseFiles = p+i+n+u+g+s+m+c+md5+sha1+sha256
StaticFiles = p+i+n+u+g+s+m+c+md5+sha1+sha256+acl+selinux+xattrs
DeviceFiles = p+i+n+u+g+s+b+c+md5+sha1

# 고급 규칙들
CriticalFiles = p+i+n+u+g+s+m+c+md5+sha1+sha256+sha512+acl+selinux+xattrs
WebFiles = p+i+n+u+g+s+m+c+md5+sha1+sha256
TempFiles = n+u+g+i
UserFiles = p+i+n+u+g+s+m+c+md5+sha1

# 시스템 중요 디렉토리
/boot                   CriticalFiles
/bin                    Binlib
/sbin                   Binlib
/usr/bin                Binlib
/usr/sbin               Binlib
/usr/local/bin          Binlib
/usr/local/sbin         Binlib
/lib                    Binlib
/lib64                  Binlib
/usr/lib                Binlib
/usr/lib64              Binlib

# 설정 파일들
/etc                    ConfFiles
!/etc/mtab
!/etc/.*~
!/etc/\..*
!/etc/passwd-
!/etc/shadow-
!/etc/group-
!/etc/gshadow-
!/etc/security/opasswd
!/etc/mail/statistics
!/etc/prelink\.cache
!/etc/crontab
!/etc/cron\.d
!/etc/cron\.daily
!/etc/cron\.hourly
!/etc/cron\.monthly
!/etc/cron\.weekly

# 시스템 특수 파일들
/etc/passwd             CriticalFiles
/etc/shadow             CriticalFiles
/etc/group              CriticalFiles
/etc/gshadow            CriticalFiles
/etc/sudoers            CriticalFiles
/etc/ssh/sshd_config    CriticalFiles
/etc/hosts              CriticalFiles
/etc/hosts.allow        CriticalFiles
/etc/hosts.deny         CriticalFiles

# 웹 서버 파일들 (해당하는 경우)
/var/www                WebFiles
!/var/www/html/.*\.log$
!/var/www/.*cache.*

# 로그 디렉토리 (증가만 허용)
/var/log                LogFiles
!/var/log/.*\.[0-9]+\.gz
!/var/log/.*\.old
!/var/log/.*\.log\.[0-9]+
!/var/log/lastlog
!/var/log/wtmp
!/var/log/btmp
!/var/log/utmp

# 사용자 홈 디렉토리 (중요 파일만)
/home                   UserFiles
!/home/[^/]+/\.bash_history
!/home/[^/]+/\.viminfo
!/home/[^/]+/\.cache
!/home/[^/]+/\.local
!/home/[^/]+/\.mozilla
!/home/[^/]+/\.gnome
!/home/[^/]+/\.config

# 특별 관심 파일들
/root                   CriticalFiles
!/root/\.bash_history
!/root/\.viminfo

# 데이터베이스 파일들
/var/lib/mysql          DatabaseFiles
/var/lib/postgresql     DatabaseFiles

# 임시 디렉토리 (존재만 확인)
/tmp                    TempFiles
/var/tmp                TempFiles

# 제외할 디렉토리들
!/proc
!/sys
!/dev
!/run
!/var/run
!/var/lock
!/var/cache
!/var/spool
!/media
!/mnt
!/tmp/.*
!/var/tmp/.*
!/lost\+found

# 고급 제외 규칙들
!/\.journal
!/\.updated
!/var/lib/dhcp/dhcpd\.leases.*
!/var/lib/logrotate\.status
!/var/lib/random-seed
!/var/lib/systemd
!/var/lib/dbus
```

#### AIDE 자동화 및 모니터링 스크립트
```bash
#!/bin/bash
# /usr/local/bin/aide-monitor.sh
# AIDE 자동 모니터링 및 알림 시스템

AIDE_CONFIG="/etc/aide/aide.conf"
AIDE_DB="/var/lib/aide/aide.db.gz"
AIDE_NEW_DB="/var/lib/aide/aide.db.new.gz"
LOG_FILE="/var/log/aide/aide-monitor.log"
ALERT_LOG="/var/log/aide/aide-alerts.log"
EMAIL_TO="admin@example.com"
LOCKFILE="/var/run/aide-monitor.lock"

# 함수 정의
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

send_alert() {
    local subject="$1"
    local message="$2"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ALERT: $subject" >> "$ALERT_LOG"
    echo "$message" >> "$ALERT_LOG"
    
    # 이메일 발송 (sendmail 또는 mail 명령 사용)
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "AIDE Alert: $subject" "$EMAIL_TO"
    fi
    
    # Slack 웹훅 (설정된 경우)
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"🚨 AIDE Alert: $subject\n$message\"}" \
             "$SLACK_WEBHOOK"
    fi
    
    # syslog에도 기록
    logger -p local0.alert "AIDE Alert: $subject"
}

check_prerequisites() {
    # AIDE 설치 확인
    if ! command -v aide >/dev/null 2>&1; then
        log_message "ERROR: AIDE is not installed"
        exit 1
    fi
    
    # 데이터베이스 존재 확인
    if [ ! -f "$AIDE_DB" ]; then
        log_message "ERROR: AIDE database not found at $AIDE_DB"
        log_message "Run: aide --init && mv $AIDE_NEW_DB $AIDE_DB"
        exit 1
    fi
    
    # 로그 디렉토리 생성
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$ALERT_LOG")"
}

# 락 파일 확인 (중복 실행 방지)
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        log_message "Another instance is running (PID: $PID)"
        exit 1
    else
        rm -f "$LOCKFILE"
    fi
fi

# 락 파일 생성
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# 액션 파라미터
ACTION=${1:-check}

case $ACTION in
    "init")
        log_message "Initializing AIDE database..."
        check_prerequisites
        
        aide --init
        if [ $? -eq 0 ]; then
            mv "$AIDE_NEW_DB" "$AIDE_DB"
            log_message "AIDE database initialized successfully"
        else
            log_message "ERROR: AIDE database initialization failed"
            exit 1
        fi
        ;;
        
    "check")
        log_message "Starting AIDE integrity check..."
        check_prerequisites
        
        # 임시 파일에 결과 저장
        TEMP_RESULT="/tmp/aide-result-$$"
        
        # AIDE 실행
        aide --check > "$TEMP_RESULT" 2>&1
        AIDE_EXIT_CODE=$?
        
        case $AIDE_EXIT_CODE in
            0)
                log_message "AIDE check completed - No changes detected"
                ;;
            1)
                log_message "WARNING: AIDE detected file changes"
                
                # 변경 사항 분석
                CHANGES=$(grep -E "(added|removed|changed)" "$TEMP_RESULT" | wc -l)
                
                # 상세 변경 내역
                CHANGE_DETAILS=$(grep -A 5 -B 5 -E "(added|removed|changed)" "$TEMP_RESULT")
                
                # 알림 발송
                ALERT_MESSAGE="AIDE detected $CHANGES file system changes:

$CHANGE_DETAILS

Full report saved to: $LOG_FILE
Time: $(date)
Hostname: $(hostname)"
                
                send_alert "File System Changes Detected" "$ALERT_MESSAGE"
                
                # 전체 결과를 로그에 저장
                cat "$TEMP_RESULT" >> "$LOG_FILE"
                ;;
            2|3|4|5|6|7|14|15|16|17)
                log_message "ERROR: AIDE check failed with exit code $AIDE_EXIT_CODE"
                send_alert "AIDE Check Failed" "AIDE integrity check failed with exit code: $AIDE_EXIT_CODE"
                ;;
        esac
        
        rm -f "$TEMP_RESULT"
        ;;
        
    "update")
        log_message "Updating AIDE database..."
        check_prerequisites
        
        # 백업 생성
        BACKUP_DB="/var/lib/aide/aide.db.backup.$(date +%Y%m%d-%H%M%S).gz"
        cp "$AIDE_DB" "$BACKUP_DB"
        log_message "Database backed up to $BACKUP_DB"
        
        # 새 데이터베이스 생성
        aide --init
        if [ $? -eq 0 ]; then
            mv "$AIDE_NEW_DB" "$AIDE_DB"
            log_message "AIDE database updated successfully"
        else
            log_message "ERROR: AIDE database update failed"
            exit 1
        fi
        ;;
        
    "config-test")
        log_message "Testing AIDE configuration..."
        aide --config-check
        if [ $? -eq 0 ]; then
            log_message "AIDE configuration is valid"
        else
            log_message "ERROR: AIDE configuration has errors"
            exit 1
        fi
        ;;
        
    "stats")
        log_message "AIDE Statistics:"
        echo "Database size: $(du -h $AIDE_DB 2>/dev/null || echo 'N/A')"
        echo "Database date: $(stat -c %y $AIDE_DB 2>/dev/null || echo 'N/A')"
        echo "Config file: $AIDE_CONFIG"
        echo "Log file: $LOG_FILE"
        
        # 최근 체크 이력
        echo "Recent checks:"
        tail -n 10 "$LOG_FILE" 2>/dev/null || echo "No recent checks found"
        ;;
        
    "report")
        DAYS=${2:-7}
        log_message "Generating AIDE report for last $DAYS days..."
        
        echo "=== AIDE Activity Report (Last $DAYS days) ==="
        echo "Generated: $(date)"
        echo ""
        
        # 최근 체크 결과
        echo "Recent integrity checks:"
        grep -E "AIDE check completed|AIDE detected|ERROR:" "$LOG_FILE" 2>/dev/null | \
        awk -v days="$DAYS" '
        BEGIN {
            cutoff = systime() - (days * 24 * 60 * 60)
        }
        {
            # 날짜 파싱 (간단한 방법)
            print $0
        }' | tail -n 20
        
        echo ""
        echo "Alert summary:"
        grep "ALERT:" "$ALERT_LOG" 2>/dev/null | tail -n 10 || echo "No alerts in period"
        ;;
        
    *)
        echo "Usage: $0 {init|check|update|config-test|stats|report} [days]"
        echo ""
        echo "Actions:"
        echo "  init        - Initialize AIDE database"
        echo "  check       - Perform integrity check"
        echo "  update      - Update AIDE database"
        echo "  config-test - Test configuration"
        echo "  stats       - Show statistics"
        echo "  report      - Generate activity report"
        echo ""
        echo "Examples:"
        echo "  $0 check              # Perform integrity check"
        echo "  $0 report 30          # Generate 30-day report"
        exit 1
        ;;
esac

log_message "AIDE monitor completed successfully"
```

#### AIDE 자동화 크론잡 설정
```bash
# /etc/cron.d/aide-monitoring
# AIDE 자동 모니터링 크론 설정

# 매일 02:00에 무결성 검사 실행
0 2 * * * root /usr/local/bin/aide-monitor.sh check

# 매주 일요일 03:00에 주간 리포트 생성
0 3 * * 0 root /usr/local/bin/aide-monitor.sh report 7

# 매월 1일 04:00에 월간 리포트 생성
0 4 1 * * root /usr/local/bin/aide-monitor.sh report 30

# logrotate 설정
# /etc/logrotate.d/aide
/var/log/aide/*.log {
    daily
    missingok
    rotate 365
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        # AIDE 로그 순환 후 처리 (필요시)
    endscript
}
```

## Fail2Ban 고급 설정 | Advanced Fail2Ban Configuration

### 🚫 포괄적인 침입 차단 시스템

#### Fail2Ban 고급 메인 설정
```bash
# /etc/fail2ban/jail.local
# 종합적인 Fail2Ban 설정

[DEFAULT]
# 기본 설정
ignorelist = 127.0.0.1/8 ::1 192.168.1.0/24 10.0.0.0/8
bantime = 3600
findtime = 600
maxretry = 3
backend = auto

# 백엔드 우선순위: systemd > pyinotify > gamin > polling
backend = systemd

# 액션 설정
banaction = iptables-multiport
banaction_allports = iptables-allports
action_ = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s", chain="%(chain)s"]

# 이메일 액션
action_mw = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s", chain="%(chain)s"]
           %(mta)s-whois[name=%(__name__)s, sender="%(sender)s", dest="%(destemail)s", protocol="%(protocol)s", chain="%(chain)s"]

# 이메일 + 로그 액션  
action_mwl = %(banaction)s[name=%(__name__)s, bantime="%(bantime)s", port="%(port)s", protocol="%(protocol)s", chain="%(chain)s"]
            %(mta)s-whois-lines[name=%(__name__)s, sender="%(sender)s", dest="%(destemail)s", logpath="%(logpath)s", chain="%(chain)s"]

# 기본 액션
action = %(action_mwl)s

# 이메일 설정
destemail = admin@example.com
sender = fail2ban@example.com
mta = sendmail

# 로그 레벨
loglevel = INFO
logtarget = /var/log/fail2ban.log

# 소켓 설정
socket = /var/run/fail2ban/fail2ban.sock
pidfile = /var/run/fail2ban/fail2ban.pid

# 데이터베이스 설정
dbfile = /var/lib/fail2ban/fail2ban.sqlite3
dbpurgeage = 86400

#
# SSH 보안 강화
#
[sshd]
enabled = true
port = ssh,2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
findtime = 600

[sshd-ddos]
enabled = true
port = ssh,2222
filter = sshd-ddos
logpath = /var/log/auth.log
maxretry = 6
bantime = 1800
findtime = 120

#
# 웹 서버 보안
#
[apache-auth]
enabled = true
port = http,https
filter = apache-auth
logpath = /var/log/apache2/error.log
maxretry = 3
bantime = 3600

[apache-badbots]
enabled = true
port = http,https
filter = apache-badbots
logpath = /var/log/apache2/access.log
maxretry = 2
bantime = 7200

[apache-noscript]
enabled = true
port = http,https
filter = apache-noscript
logpath = /var/log/apache2/access.log
maxretry = 6
bantime = 1800

[apache-overflows]
enabled = true
port = http,https
filter = apache-overflows
logpath = /var/log/apache2/error.log
maxretry = 2
bantime = 7200

[apache-nohome]
enabled = true
port = http,https
filter = apache-nohome
logpath = /var/log/apache2/access.log
maxretry = 2
bantime = 3600

[apache-botsearch]
enabled = true
port = http,https
filter = apache-botsearch
logpath = /var/log/apache2/access.log
maxretry = 2
bantime = 7200

#
# Nginx 보안
#
[nginx-http-auth]
enabled = true
port = http,https
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3
bantime = 3600

[nginx-limit-req]
enabled = true
port = http,https
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 10
bantime = 600
findtime = 60

[nginx-botsearch]
enabled = true
port = http,https
filter = nginx-botsearch
logpath = /var/log/nginx/access.log
maxretry = 2
bantime = 7200

#
# 메일 서버 보안
#
[postfix]
enabled = true
port = smtp,465,submission
filter = postfix
logpath = /var/log/mail.log
maxretry = 3
bantime = 3600

[dovecot]
enabled = true
port = pop3,pop3s,imap,imaps,submission,465,sieve
filter = dovecot
logpath = /var/log/mail.log
maxretry = 3
bantime = 3600

#
# FTP 보안
#
[vsftpd]
enabled = true
port = ftp,ftp-data,ftps,ftps-data
filter = vsftpd
logpath = /var/log/vsftpd.log
maxretry = 3
bantime = 3600

#
# 데이터베이스 보안
#
[mysqld-auth]
enabled = true
port = 3306
filter = mysqld-auth
logpath = /var/log/mysql/error.log
maxretry = 3
bantime = 3600

#
# 커스텀 애플리케이션 보안
#
[custom-app]
enabled = true
port = 8080,8443
filter = custom-app
logpath = /var/log/custom-app/security.log
maxretry = 5
bantime = 7200
findtime = 300

#
# 시스템 로그 모니터링
#
[pam-generic]
enabled = true
filter = pam-generic
logpath = /var/log/auth.log
maxretry = 6
bantime = 1800

[sudo]
enabled = true
filter = sudo
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
```

#### 고급 커스텀 필터 작성
```bash
# /etc/fail2ban/filter.d/custom-app.conf
# 커스텀 애플리케이션용 필터

[Definition]
# 실패한 로그인 시도
failregex = ^<HOST> .* "POST /api/login HTTP.*" 401 .*$
            ^<HOST> .* "POST /admin/login HTTP.*" 403 .*$
            ^<HOST> .* "Failed login attempt for user .* from <HOST>"$
            ^<HOST> .* "Invalid API key from <HOST>"$
            ^<HOST> .* "Suspicious activity detected from <HOST>"$
            ^<HOST> .* "Rate limit exceeded for <HOST>"$
            ^<HOST> .* "Brute force attempt detected from <HOST>"$

# 무시할 패턴
ignoreregex = ^<HOST> .* "GET /health HTTP.*" 200 .*$
              ^<HOST> .* "GET /status HTTP.*" 200 .*$

# /etc/fail2ban/filter.d/nginx-botsearch.conf
# Nginx 봇 검색 패턴 필터

[Definition]
failregex = ^<HOST> -.*GET.*(\.php|\.asp|\.exe|\.pl|\.cgi|\.scgi)
            ^<HOST> -.*GET.*admin
            ^<HOST> -.*GET.*phpMyAdmin
            ^<HOST> -.*GET.*wp-admin
            ^<HOST> -.*GET.*wp-login
            ^<HOST> -.*GET.*/etc/passwd
            ^<HOST> -.*GET.*\.\./
            ^<HOST> -.*GET.*(proc/self/environ|etc/shadow|etc/passwd)
            ^<HOST> -.*GET.*(cmd\.exe|command\.com)
            ^<HOST> -.*GET.*sql.*dump

ignoreregex =

# /etc/fail2ban/filter.d/wordpress-security.conf
# WordPress 보안 전용 필터

[Definition]
failregex = ^<HOST> .*POST.*/wp-login\.php.* 200
            ^<HOST> .*POST.*/wp-admin/admin-ajax\.php.* 400
            ^<HOST> .*GET.*/wp-admin.*
            ^<HOST> .*GET.*/wp-content/.*\.php
            ^<HOST> .*GET.*\?author=\d+
            ^<HOST> .*GET.*/xmlrpc\.php

ignoreregex = ^<HOST> .*POST.*/wp-login\.php.* "WordPress/.*"

# /etc/fail2ban/filter.d/ddos.conf
# DDoS 공격 감지 필터

[Definition]
failregex = ^<HOST> -.*GET.*
ignoreregex =

# /etc/fail2ban/filter.d/port-scan.conf
# 포트 스캔 감지 필터

[Definition]
failregex = ^.*kernel:.*IN=.*SRC=<HOST>.*DPT=(1|7|9|11|15|70|79|80|109|110|143|443|993|995).*
            ^.*kernel:.*SRC=<HOST>.*DPT=(23|53|111|137|139|445|513|514|515|993|995|1433|1521|3389).*

ignoreregex =
```

#### Fail2Ban 고급 관리 스크립트
```bash
#!/bin/bash
# /usr/local/bin/fail2ban-manager.sh
# Fail2Ban 고급 관리 도구

ACTION=$1
JAIL=$2
IP=$3

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_color() {
    echo -e "${2}${1}${NC}"
}

check_fail2ban() {
    if ! systemctl is-active --quiet fail2ban; then
        print_color "ERROR: Fail2Ban is not running" "$RED"
        exit 1
    fi
}

case $ACTION in
    "status")
        check_fail2ban
        echo "=== Fail2Ban Status ==="
        fail2ban-client status
        
        if [ -n "$JAIL" ]; then
            echo ""
            echo "=== Jail: $JAIL ==="
            fail2ban-client status "$JAIL"
        fi
        ;;
        
    "ban")
        if [ -z "$JAIL" ] || [ -z "$IP" ]; then
            echo "Usage: $0 ban <jail> <ip>"
            exit 1
        fi
        
        check_fail2ban
        print_color "Banning IP $IP in jail $JAIL..." "$YELLOW"
        fail2ban-client set "$JAIL" banip "$IP"
        print_color "IP $IP banned successfully" "$GREEN"
        ;;
        
    "unban")
        if [ -z "$IP" ]; then
            echo "Usage: $0 unban <jail|all> <ip>"
            exit 1
        fi
        
        check_fail2ban
        if [ "$JAIL" = "all" ]; then
            print_color "Unbanning IP $IP from all jails..." "$YELLOW"
            for jail in $(fail2ban-client status | grep "Jail list:" | cut -d: -f2 | tr ',' ' '); do
                jail=$(echo $jail | xargs)  # trim whitespace
                fail2ban-client set "$jail" unbanip "$IP" 2>/dev/null || true
            done
        else
            print_color "Unbanning IP $IP from jail $JAIL..." "$YELLOW"
            fail2ban-client set "$JAIL" unbanip "$IP"
        fi
        print_color "IP $IP unbanned successfully" "$GREEN"
        ;;
        
    "list-banned")
        check_fail2ban
        if [ -n "$JAIL" ]; then
            echo "=== Banned IPs in jail: $JAIL ==="
            fail2ban-client get "$JAIL" banip
        else
            echo "=== All Banned IPs ==="
            for jail in $(fail2ban-client status | grep "Jail list:" | cut -d: -f2 | tr ',' ' '); do
                jail=$(echo $jail | xargs)
                banned=$(fail2ban-client get "$jail" banip 2>/dev/null)
                if [ -n "$banned" ] && [ "$banned" != "[]" ]; then
                    echo "Jail: $jail"
                    echo "$banned"
                    echo ""
                fi
            done
        fi
        ;;
        
    "stats")
        check_fail2ban
        echo "=== Fail2Ban Statistics ==="
        echo ""
        
        # 전체 통계
        echo "Active jails:"
        fail2ban-client status | grep "Jail list:" | cut -d: -f2
        
        echo ""
        echo "Jail statistics:"
        for jail in $(fail2ban-client status | grep "Jail list:" | cut -d: -f2 | tr ',' ' '); do
            jail=$(echo $jail | xargs)
            status=$(fail2ban-client status "$jail" 2>/dev/null)
            if [ $? -eq 0 ]; then
                currently_failed=$(echo "$status" | grep "Currently failed:" | awk '{print $NF}')
                total_failed=$(echo "$status" | grep "Total failed:" | awk '{print $NF}')
                currently_banned=$(echo "$status" | grep "Currently banned:" | awk '{print $NF}')
                total_banned=$(echo "$status" | grep "Total banned:" | awk '{print $NF}')
                
                echo "  $jail: Failed=$currently_failed/$total_failed, Banned=$currently_banned/$total_banned"
            fi
        done
        
        echo ""
        echo "Top banned IPs:"
        grep "Ban " /var/log/fail2ban.log 2>/dev/null | \
        awk '{print $(NF-1)}' | sort | uniq -c | sort -nr | head -10
        ;;
        
    "top-attackers")
        DAYS=${JAIL:-7}
        echo "=== Top Attackers (Last $DAYS days) ==="
        
        # 지정된 날짜부터의 로그 분석
        if [ $DAYS -gt 0 ]; then
            DATE_FILTER="-since \"$DAYS days ago\""
        else
            DATE_FILTER=""
        fi
        
        # 로그에서 공격 IP 추출 및 분석
        eval journalctl -u fail2ban $DATE_FILTER 2>/dev/null | \
        grep "Ban " | \
        awk '{print $(NF-1)}' | \
        sort | uniq -c | sort -nr | head -20 | \
        while read count ip; do
            # IP 지역 정보 조회 (geoip 사용, 설치된 경우)
            if command -v geoiplookup >/dev/null 2>&1; then
                country=$(geoiplookup "$ip" 2>/dev/null | cut -d: -f2 | xargs)
            else
                country="Unknown"
            fi
            printf "%-6s %-15s %s\n" "$count" "$ip" "$country"
        done
        ;;
        
    "analyze")
        HOURS=${JAIL:-24}
        echo "=== Fail2Ban Analysis (Last $HOURS hours) ==="
        
        # 최근 활동 분석
        eval journalctl -u fail2ban --since "$HOURS hours ago" 2>/dev/null | \
        grep -E "(Ban|Unban)" | \
        awk '{
            if ($0 ~ /Ban /) bans++
            if ($0 ~ /Unban/) unbans++
        }
        END {
            print "Total bans: " (bans ? bans : 0)
            print "Total unbans: " (unbans ? unbans : 0)
            print "Net banned: " ((bans ? bans : 0) - (unbans ? unbans : 0))
        }'
        
        echo ""
        echo "Activity by jail:"
        eval journalctl -u fail2ban --since "$HOURS hours ago" 2>/dev/null | \
        grep "Ban " | \
        awk '{
            # jail 이름 추출 (로그 형식에 따라 조정 필요)
            match($0, /\[([^\]]+)\]/, arr)
            if (arr[1]) jails[arr[1]]++
        }
        END {
            for (jail in jails) {
                print "  " jail ": " jails[jail]
            }
        }'
        ;;
        
    "reload")
        print_color "Reloading Fail2Ban configuration..." "$YELLOW"
        systemctl reload fail2ban
        print_color "Fail2Ban reloaded successfully" "$GREEN"
        ;;
        
    "test-filter")
        if [ -z "$JAIL" ]; then
            echo "Usage: $0 test-filter <filter-name> [log-file]"
            exit 1
        fi
        
        FILTER=$JAIL
        LOGFILE=${IP:-/var/log/auth.log}
        
        echo "=== Testing filter: $FILTER ==="
        echo "Log file: $LOGFILE"
        echo ""
        
        fail2ban-regex "$LOGFILE" "/etc/fail2ban/filter.d/${FILTER}.conf"
        ;;
        
    "whitelist")
        if [ -z "$JAIL" ] || [ -z "$IP" ]; then
            echo "Usage: $0 whitelist <add|remove|list> <ip|network>"
            exit 1
        fi
        
        WHITELIST_ACTION=$JAIL
        TARGET_IP=$IP
        
        case $WHITELIST_ACTION in
            "add")
                print_color "Adding $TARGET_IP to whitelist..." "$YELLOW"
                # jail.local의 ignoreip에 추가
                if grep -q "ignoreip.*$TARGET_IP" /etc/fail2ban/jail.local; then
                    print_color "IP $TARGET_IP is already whitelisted" "$BLUE"
                else
                    sed -i "/^ignoreip = / s/$/ $TARGET_IP/" /etc/fail2ban/jail.local
                    systemctl reload fail2ban
                    print_color "IP $TARGET_IP added to whitelist" "$GREEN"
                fi
                ;;
            "remove")
                print_color "Removing $TARGET_IP from whitelist..." "$YELLOW"
                sed -i "s/ $TARGET_IP//g" /etc/fail2ban/jail.local
                systemctl reload fail2ban
                print_color "IP $TARGET_IP removed from whitelist" "$GREEN"
                ;;
            "list")
                echo "=== Current Whitelist ==="
                grep "^ignoreip" /etc/fail2ban/jail.local
                ;;
        esac
        ;;
        
    "backup")
        BACKUP_DIR="/etc/fail2ban/backup/$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        cp -r /etc/fail2ban/jail.local "$BACKUP_DIR/" 2>/dev/null || true
        cp -r /etc/fail2ban/filter.d/* "$BACKUP_DIR/" 2>/dev/null || true
        cp -r /etc/fail2ban/action.d/* "$BACKUP_DIR/" 2>/dev/null || true
        
        print_color "Fail2Ban configuration backed up to: $BACKUP_DIR" "$GREEN"
        ;;
        
    *)
        echo "Usage: $0 {status|ban|unban|list-banned|stats|top-attackers|analyze|reload|test-filter|whitelist|backup} [options]"
        echo ""
        echo "Commands:"
        echo "  status [jail]                    - Show Fail2Ban status"
        echo "  ban <jail> <ip>                 - Ban IP in specific jail"
        echo "  unban <jail|all> <ip>           - Unban IP from jail or all jails"
        echo "  list-banned [jail]              - List banned IPs"
        echo "  stats                           - Show comprehensive statistics"
        echo "  top-attackers [days]            - Show top attacking IPs"
        echo "  analyze [hours]                 - Analyze recent activity"
        echo "  reload                          - Reload configuration"
        echo "  test-filter <filter> [logfile]  - Test filter regex"
        echo "  whitelist <add|remove|list> <ip> - Manage whitelist"
        echo "  backup                          - Backup configuration"
        echo ""
        echo "Examples:"
        echo "  $0 status sshd"
        echo "  $0 ban sshd 192.168.1.100"
        echo "  $0 unban all 192.168.1.100"
        echo "  $0 top-attackers 30"
        echo "  $0 test-filter sshd"
        exit 1
        ;;
esac
```

## 다음 편 예고

다음 포스트에서는 **Wazuh SIEM과 컨테이너 보안**을 상세히 다룰 예정입니다:
- Wazuh 완전 구축 및 설정
- 실시간 위협 탐지 룰 작성
- Docker/Kubernetes 보안 강화
- 컴플라이언스 및 감사 로그 관리

AIDE와 Fail2Ban으로 강력한 침입 탐지 시스템을 완성하셨나요? 🔍🛡️
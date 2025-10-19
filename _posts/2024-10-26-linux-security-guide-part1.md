---
layout: post
title: "리눅스 보안 완전 가이드 1편 - 기초 보안 설정 | Linux Security Guide Part 1 - Basic Security Configuration"
date: 2024-10-26 09:00:00 +0900
categories: [Security, Linux]
tags: [linux-security, user-management, password-policy, ssh, authentication]
---

리눅스 시스템 보안의 첫 번째 단계인 기초 보안 설정을 완벽하게 마스터해보겠습니다. 사용자 계정 관리부터 SSH 보안 강화까지 실무에서 바로 적용할 수 있는 내용으로 구성했습니다.

## 사용자 계정 보안 강화 | User Account Security

### 👤 사용자 계정 관리 기초

#### 안전한 사용자 생성 및 관리
```bash
# 보안을 고려한 사용자 생성
# 시스템 사용자 (서비스용)
useradd -r -s /usr/sbin/nologin -d /var/lib/myservice -c "MyService System User" myservice

# 일반 사용자 생성 (홈 디렉토리 권한 강화)
useradd -m -s /bin/bash -c "John Doe" -G users john
chmod 750 /home/john
chown john:john /home/john

# 사용자 정보 확인
id john
groups john
finger john
last john

# 사용자 계정 잠금/해제
usermod -L john    # 계정 잠금
usermod -U john    # 계정 해제
passwd -l john     # 패스워드 잠금
passwd -u john     # 패스워드 해제

# 계정 만료 설정
usermod -e 2024-12-31 john    # 계정 만료일 설정
chage -E 2024-12-31 john      # 동일한 기능

# 비활성 사용자 계정 찾기
lastlog | awk '$2 == "Never" || $2 < "'$(date -d '90 days ago' '+%Y-%m-%d')'" {print $1}'

# 불필요한 계정 제거
userdel -r olduser    # 홈 디렉토리도 함께 삭제
```

#### 강력한 패스워드 정책 구현
```bash
# PAM 기반 패스워드 복잡성 설정
# /etc/pam.d/common-password (Ubuntu/Debian)
password requisite pam_pwquality.so retry=3 minlen=14 minclass=4 maxrepeat=2 ucredit=-2 lcredit=-2 dcredit=-2 ocredit=-2 difok=4 gecoscheck=1 dictcheck=1

# /etc/security/pwquality.conf 상세 설정
# 패스워드 길이 및 복잡성
minlen = 14              # 최소 14자
minclass = 4             # 4개 문자 클래스 모두 포함
maxrepeat = 2            # 연속 동일 문자 2개 이하
maxclasssrepeat = 3      # 동일 클래스 연속 3개 이하

# 문자별 최소 개수 (음수는 필수)
ucredit = -2             # 대문자 최소 2개
lcredit = -2             # 소문자 최소 2개  
dcredit = -2             # 숫자 최소 2개
ocredit = -2             # 특수문자 최소 2개

# 패스워드 품질 검사
difok = 4                # 이전 패스워드와 최소 4글자 차이
gecoscheck = 1           # GECOS 필드(이름 등) 포함 금지
dictcheck = 1            # 사전 단어 사용 금지
usercheck = 1            # 사용자명 포함 금지
enforcing = 1            # 엄격한 정책 적용

# 금지 패스워드 목록
badwords = password 123456 qwerty admin root

# 패스워드 에이징 정책 (고급)
# /etc/login.defs
PASS_MAX_DAYS   60       # 최대 60일 유효
PASS_MIN_DAYS   7        # 최소 7일 후 변경 가능
PASS_MIN_LEN    14       # 최소 14자
PASS_WARN_AGE   7        # 만료 7일 전 경고
ENCRYPT_METHOD  SHA512   # 강력한 해시 알고리즘

# 기존 사용자에게 정책 적용 스크립트
#!/bin/bash
for user in $(cut -d: -f1 /etc/passwd | grep -v "^#" | sort); do
    # 시스템 계정은 제외 (UID 1000 미만)
    uid=$(id -u "$user" 2>/dev/null)
    if [[ $uid -ge 1000 && $uid -le 60000 ]]; then
        echo "Applying password policy to $user..."
        chage -M 60 -m 7 -W 7 "$user"
        # 다음 로그인 시 패스워드 변경 강제
        chage -d 0 "$user"
    fi
done
```

#### 계정 잠금 및 브루트포스 방지
```bash
# PAM 기반 계정 잠금 설정 (최신 방식)
# /etc/pam.d/common-auth
auth required pam_faillock.so preauth silent audit deny=5 unlock_time=900 fail_interval=900
auth [default=die] pam_faillock.so authfail audit deny=5 unlock_time=900 fail_interval=900
auth sufficient pam_unix.so nullok_secure
auth [default=die] pam_faillock.so authsucc audit deny=5 unlock_time=900 fail_interval=900

# /etc/pam.d/common-account에 추가
account required pam_faillock.so

# faillock 설정 파일
# /etc/security/faillock.conf
dir = /var/run/faillock
audit
silent
deny = 5
fail_interval = 900      # 15분 간격
unlock_time = 1800       # 30분 잠금
even_deny_root          # 루트도 잠금 적용
root_unlock_time = 60    # 루트는 1분만 잠금

# faillock 관리 명령
faillock --user john     # 사용자 실패 횟수 확인
faillock --user john --reset   # 사용자 잠금 해제
faillock --reset        # 모든 사용자 잠금 해제

# 실시간 모니터링 스크립트
#!/bin/bash
# /usr/local/bin/monitor-auth-failures.sh
LOG_FILE="/var/log/auth.log"
ALERT_THRESHOLD=3

tail -f "$LOG_FILE" | while read line; do
    if echo "$line" | grep -q "authentication failure"; then
        user=$(echo "$line" | grep -o "user=[^ ]*" | cut -d= -f2)
        ip=$(echo "$line" | grep -o "rhost=[^ ]*" | cut -d= -f2)
        
        # 최근 5분간 실패 횟수 계산
        failures=$(grep -c "authentication failure.*user=$user.*rhost=$ip" \
                  <(tail -n 1000 "$LOG_FILE" | \
                    awk -v since="$(date -d '5 minutes ago' '+%b %d %H:%M')" \
                    '$0 >= since'))
        
        if [[ $failures -ge $ALERT_THRESHOLD ]]; then
            echo "$(date): ALERT - Multiple auth failures for user $user from $ip ($failures attempts)"
            # 알림 발송 (선택사항)
            # echo "Authentication attack detected: $user from $ip" | \
            # mail -s "Security Alert" admin@company.com
        fi
    fi
done
```

### 🔐 루트 계정 보안 강화

#### 루트 접근 제한 및 sudo 구성
```bash
# 루트 직접 로그인 완전 차단
# /etc/ssh/sshd_config
PermitRootLogin no

# 콘솔 루트 로그인 제한
# /etc/securetty (비어있게 하면 콘솔 로그인 차단)
> /etc/securetty

# 또는 특정 터미널만 허용
cat > /etc/securetty << 'EOF'
console
tty1
EOF

# sudo 권한 세밀한 제어
# /etc/sudoers.d/custom-rules

# 1. 그룹 기반 권한 설정
%wheel ALL=(ALL:ALL) ALL
%admin ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart *, /usr/bin/systemctl reload *

# 2. 사용자별 세부 권한
# 웹 서버 관리자
webadmin ALL=(ALL) /usr/sbin/service apache2 *, /usr/sbin/service nginx *, \
              /usr/bin/systemctl restart apache2, /usr/bin/systemctl reload nginx, \
              /usr/bin/tail -f /var/log/apache2/*, /usr/bin/tail -f /var/log/nginx/*

# 데이터베이스 관리자  
dbadmin ALL=(postgres) NOPASSWD: /usr/bin/psql, /usr/bin/pg_dump, /usr/bin/pg_restore
dbadmin ALL=(mysql) NOPASSWD: /usr/bin/mysql, /usr/bin/mysqldump

# 백업 관리자
backup ALL=(ALL) NOPASSWD: /usr/bin/rsync, /bin/tar, /bin/gzip, /usr/bin/find /home -name "*"

# 3. 명령 제한 및 인수 제한
developer ALL=(ALL) /usr/bin/systemctl status *, !/usr/bin/systemctl * --force

# 4. 시간 제한
nightshift ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart * \
    # 야간 근무 시간에만 허용 (예시)

# 5. 호스트 기반 제한
john server1=(ALL) /bin/ls, /bin/cat /var/log/*

# sudo 로깅 강화
# /etc/sudoers에 추가
Defaults    logfile="/var/log/sudo.log"
Defaults    log_input, log_output      # 입출력 로깅
Defaults    iolog_dir="/var/log/sudo-io/%{user}/%{time}"
Defaults    timestamp_timeout=0        # 캐시 비활성화
Defaults    passwd_tries=3            # 3회 시도 후 실패
Defaults    passwd_timeout=5          # 5분 타임아웃
Defaults    env_reset                 # 환경변수 초기화
Defaults    mail_badpass              # 잘못된 패스워드 시 메일
Defaults    secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# sudo 세션 모니터링 스크립트
#!/bin/bash
# /usr/local/bin/sudo-monitor.sh
SUDO_LOG="/var/log/sudo.log"
ALERT_EMAIL="admin@company.com"

# 위험한 명령어 패턴
DANGEROUS_PATTERNS=(
    "rm -rf /"
    "dd if="
    "mkfs\."
    "fdisk"
    "parted"
    "chmod 777"
    "chown root"
    "/bin/bash"
    "/bin/sh"
    "su -"
    "passwd root"
    "userdel"
    "usermod -u 0"
)

tail -f "$SUDO_LOG" | while read line; do
    for pattern in "${DANGEROUS_PATTERNS[@]}"; do
        if echo "$line" | grep -i "$pattern" >/dev/null; then
            echo "$(date): DANGEROUS SUDO COMMAND DETECTED: $line"
            echo "Dangerous sudo command executed: $line" | \
            mail -s "SECURITY ALERT: Dangerous Command" "$ALERT_EMAIL"
        fi
    done
done
```

### 🔑 고급 인증 메커니즘

#### 다중 인증 요소 (MFA) 구현
```bash
# Google Authenticator (TOTP) 설정
# 1. 패키지 설치
apt-get install libpam-google-authenticator

# 2. 사용자별 설정 (각 사용자가 실행)
google-authenticator
# - 시간 기반 토큰 사용: y
# - QR 코드 표시 후 앱에 등록
# - 긴급 스크래치 코드 저장
# - 토큰 재사용 방지: y
# - 시간 허용 오차: y

# 3. SSH에서 MFA 활성화
# /etc/pam.d/sshd에 추가 (password 줄 위에)
auth required pam_google_authenticator.so

# /etc/ssh/sshd_config 수정
ChallengeResponseAuthentication yes
AuthenticationMethods publickey,keyboard-interactive

# 4. sudo에서 MFA 적용
# /etc/pam.d/sudo에 추가
auth required pam_google_authenticator.so

# 5. MFA 백업 및 복구
# 사용자별 설정 백업
cp ~/.google_authenticator ~/.google_authenticator.backup

# 관리자용 긴급 복구 스크립트
#!/bin/bash
# /usr/local/bin/mfa-recovery.sh
read -p "Username: " username
read -p "Emergency scratch code: " scratch_code

if grep -q "$scratch_code" "/home/$username/.google_authenticator"; then
    echo "Valid scratch code. Temporarily disabling MFA..."
    mv "/home/$username/.google_authenticator" "/home/$username/.google_authenticator.disabled"
    echo "MFA disabled for $username. Re-enable after password reset."
else
    echo "Invalid scratch code."
fi
```

#### LDAP/Active Directory 통합 인증
```bash
# SSSD를 이용한 AD 통합 (Ubuntu/CentOS)
# 1. 필요 패키지 설치
apt-get install sssd sssd-tools realmd adcli packagekit

# 2. 도메인 발견 및 가입
realm discover company.local
realm join -U administrator company.local

# 3. SSSD 구성
# /etc/sssd/sssd.conf
[sssd]
domains = company.local
config_file_version = 2
services = nss, pam

[domain/company.local]
default_shell = /bin/bash
krb5_store_password_if_offline = True
cache_credentials = True
krb5_realm = COMPANY.LOCAL
realmd_tags = manages-system joined-with-adcli
id_provider = ad
fallback_homedir = /home/%u@%d
ad_domain = company.local
use_fully_qualified_names = True
ldap_id_mapping = True
access_provider = ad

# 특정 그룹만 허용
ad_access_filter = (memberOf=CN=Linux-Users,OU=Groups,DC=company,DC=local)

chmod 600 /etc/sssd/sssd.conf
systemctl enable sssd
systemctl start sssd

# 4. 홈 디렉토리 자동 생성
# /etc/pam.d/common-session에 추가
session required pam_mkhomedir.so skel=/etc/skel umask=0022

# 5. sudo 권한 설정 (AD 그룹 기반)
# /etc/sudoers.d/ad-groups
%domain\ admins@company.local ALL=(ALL:ALL) ALL
%linux-administrators@company.local ALL=(ALL) NOPASSWD: /usr/bin/systemctl *, /usr/sbin/service *

# 6. 연결 테스트
getent passwd user@company.local
id user@company.local
su - user@company.local
```

### 🛂 세션 및 로그인 제어

#### 고급 로그인 제어 설정
```bash
# /etc/security/access.conf - 액세스 제어
# 형식: 권한:사용자/그룹:터미널/호스트

# 루트는 로컬에서만 로그인 허용
+ : root : LOCAL
- : root : ALL

# admin 그룹은 특정 IP에서만
+ : @admin : 192.168.1.0/24
- : @admin : ALL

# 특정 사용자는 특정 시간에만
+ : nightshift : ALL EXCEPT HOLIDAYS
- : nightshift : Wl0800-1800

# 시간 기반 접근 제어
# /etc/security/time.conf
login;*;users;Al0800-1800
sshd;*;developers;MoTuWeThFr0900-1800

# 로그인 시도 제한
# /etc/security/limits.conf
# 사용자별 동시 로그인 세션 제한
@users hard maxlogins 2
john hard maxlogins 1

# 프로세스 수 제한 (포크 폭탄 방지)
@users hard nproc 1024
@developers hard nproc 2048

# 메모리 사용량 제한
@users hard as 1048576    # 1GB

# 파일 디스크립터 제한
@users hard nofile 4096

# 코어 덤프 비활성화
* hard core 0

# 로그인 배너 설정
# /etc/issue (콘솔 로그인)
cat > /etc/issue << 'EOF'
**********************************************************************
*                        WARNING NOTICE                             *
*                                                                    *
* This system is for authorized users only. All activities may be   *
* monitored and recorded. Unauthorized access is prohibited and     *
* will be prosecuted to the full extent of the law.                *
*                                                                    *
**********************************************************************

EOF

# /etc/issue.net (네트워크 로그인)
cp /etc/issue /etc/issue.net

# /etc/motd (로그인 후 메시지)
cat > /etc/motd << 'EOF'
System Information:
- Last system update: $(date)
- Security policy: https://company.com/security-policy
- Report security incidents: security@company.com

EOF

# 동적 MOTD 생성 스크립트
#!/bin/bash
# /etc/update-motd.d/10-sysinfo
echo "System Status as of $(date)"
echo "======================================"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime -p)"
echo "Load: $(cat /proc/loadavg | awk '{print $1", "$2", "$3}')"
echo "Memory: $(free -h | awk 'NR==2{printf "%.1fG/%.1fG (%.1f%%)", $3/1024, $2/1024, $3*100/$2}')"
echo "Disk: $(df -h / | awk 'NR==2{printf "%s/%s (%s)", $3, $2, $5}')"
echo ""

# 보안 상태 확인
if systemctl is-active --quiet fail2ban; then
    echo "✓ Fail2ban: Active"
else
    echo "✗ Fail2ban: Inactive"
fi

if systemctl is-active --quiet sshd; then
    echo "✓ SSH: Active"
else
    echo "✗ SSH: Inactive"  
fi

echo "======================================"
echo ""

chmod +x /etc/update-motd.d/10-sysinfo
```

### 📊 인증 로그 모니터링 및 분석

#### 실시간 로그인 모니터링 시스템
```python
#!/usr/bin/env python3
# /usr/local/bin/auth-monitor.py
import re
import time
import subprocess
import smtplib
from email.mime.text import MIMEText
from collections import defaultdict, deque
from datetime import datetime, timedelta

class AuthenticationMonitor:
    def __init__(self):
        self.failed_attempts = defaultdict(deque)
        self.successful_logins = defaultdict(deque)
        self.suspicious_ips = set()
        self.alert_threshold = 5
        self.time_window = 300  # 5 minutes
        
    def parse_log_line(self, line):
        """로그 라인 파싱"""
        patterns = {
            'failed_password': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)',
            'accepted_password': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*Accepted password for (\w+) from (\d+\.\d+\.\d+\.\d+)',
            'invalid_user': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*Invalid user (\w+) from (\d+\.\d+\.\d+\.\d+)',
            'sudo_command': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*sudo.*USER=(\w+).*COMMAND=(.*)'
        }
        
        for event_type, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                return event_type, match.groups()
        return None, None
    
    def is_suspicious_ip(self, ip):
        """의심스러운 IP 판별"""
        # 내부 네트워크 IP는 제외
        internal_ranges = [
            '192.168.', '10.', '172.16.', '172.17.', '172.18.',
            '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
            '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
            '172.29.', '172.30.', '172.31.', '127.'
        ]
        
        return not any(ip.startswith(prefix) for prefix in internal_ranges)
    
    def check_brute_force(self, ip):
        """브루트포스 공격 확인"""
        now = time.time()
        
        # 오래된 시도 제거
        while (self.failed_attempts[ip] and 
               now - self.failed_attempts[ip][0] > self.time_window):
            self.failed_attempts[ip].popleft()
        
        return len(self.failed_attempts[ip]) >= self.alert_threshold
    
    def send_alert(self, alert_type, details):
        """알림 발송"""
        message = f"""
Security Alert: {alert_type}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Details: {details}

System: {subprocess.check_output('hostname', shell=True).decode().strip()}

Please investigate immediately.
        """
        
        print(f"ALERT: {alert_type} - {details}")
        
        # 이메일 발송 (선택사항)
        # self.send_email("Security Alert", message)
        
        # 로그 기록
        with open('/var/log/security-alerts.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {alert_type}: {details}\n")
    
    def monitor_log(self, log_file='/var/log/auth.log'):
        """로그 모니터링 메인 루프"""
        print(f"Starting authentication monitoring on {log_file}")
        
        # 기존 로그 처리 (마지막 100줄)
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-100:]:
                    self.process_line(line.strip())
        except FileNotFoundError:
            print(f"Log file {log_file} not found")
            return
        
        # 실시간 모니터링
        proc = subprocess.Popen(['tail', '-f', log_file], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        
        try:
            for line in iter(proc.stdout.readline, ''):
                self.process_line(line.strip())
        except KeyboardInterrupt:
            proc.terminate()
            print("\nMonitoring stopped")
    
    def process_line(self, line):
        """로그 라인 처리"""
        event_type, data = self.parse_log_line(line)
        
        if not event_type:
            return
            
        current_time = time.time()
        
        if event_type == 'failed_password':
            timestamp, user, ip = data
            self.failed_attempts[ip].append(current_time)
            
            if self.is_suspicious_ip(ip) and self.check_brute_force(ip):
                self.send_alert(
                    "Brute Force Attack Detected",
                    f"IP {ip} failed {len(self.failed_attempts[ip])} login attempts for user {user}"
                )
                self.suspicious_ips.add(ip)
        
        elif event_type == 'accepted_password':
            timestamp, user, ip = data
            
            if ip in self.suspicious_ips:
                self.send_alert(
                    "Successful Login from Suspicious IP",
                    f"User {user} logged in from previously suspicious IP {ip}"
                )
            
            # 비정상 시간대 로그인 확인
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # 오전 6시 이전, 오후 10시 이후
                self.send_alert(
                    "Off-Hours Login",
                    f"User {user} logged in from {ip} at unusual time: {datetime.now().strftime('%H:%M')}"
                )
        
        elif event_type == 'invalid_user':
            timestamp, user, ip = data
            self.send_alert(
                "Invalid User Login Attempt",
                f"Attempt to login as non-existent user '{user}' from {ip}"
            )
        
        elif event_type == 'sudo_command':
            timestamp, user, command = data
            
            # 위험한 명령어 감지
            dangerous_patterns = ['rm -rf', 'dd if=', 'mkfs', 'fdisk', 'passwd root']
            for pattern in dangerous_patterns:
                if pattern in command:
                    self.send_alert(
                        "Dangerous Sudo Command",
                        f"User {user} executed: {command}"
                    )
                    break

if __name__ == '__main__':
    monitor = AuthenticationMonitor()
    monitor.monitor_log()
```

#### 로그인 통계 및 분석 도구
```bash
#!/bin/bash
# /usr/local/bin/auth-stats.sh
# 인증 로그 통계 분석 스크립트

echo "=== Linux Authentication Statistics Report ==="
echo "Generated on: $(date)"
echo "=============================================="
echo

# 1. 최근 성공한 로그인
echo "🔑 Recent Successful Logins (Last 24 hours):"
last -s yesterday | head -20
echo

# 2. 실패한 로그인 시도 통계
echo "❌ Failed Login Attempts (Last 7 days):"
grep "Failed password" /var/log/auth.log* | \
awk '{print $1, $2, $11}' | \
sort | uniq -c | sort -nr | head -10
echo

# 3. 의심스러운 사용자명 시도
echo "👤 Invalid User Attempts:"
grep "Invalid user" /var/log/auth.log* | \
awk '{print $8}' | sort | uniq -c | sort -nr | head -10
echo

# 4. IP별 접근 시도 통계
echo "🌐 Access Attempts by IP:"
grep -E "(Failed password|Accepted password)" /var/log/auth.log* | \
awk '{print $NF}' | sort | uniq -c | sort -nr | head -10
echo

# 5. 시간대별 로그인 패턴
echo "⏰ Login Pattern by Hour:"
grep "Accepted password" /var/log/auth.log* | \
awk '{print $3}' | cut -d: -f1 | sort -n | uniq -c | \
while read count hour; do
    printf "%02d:00 %s %s\n" "$hour" "$count" "$(printf "%*s" $((count/2)) "" | tr " " "▇")"
done
echo

# 6. sudo 사용 통계
echo "🔧 Sudo Command Usage:"
grep "sudo" /var/log/auth.log* | \
grep "COMMAND=" | \
awk -F'COMMAND=' '{print $2}' | \
awk '{print $1}' | sort | uniq -c | sort -nr | head -10
echo

# 7. 보안 이벤트 요약
echo "🚨 Security Events Summary:"
total_failed=$(grep -c "Failed password" /var/log/auth.log*)
total_invalid=$(grep -c "Invalid user" /var/log/auth.log*)
total_sudo=$(grep -c "sudo.*COMMAND=" /var/log/auth.log*)

echo "   Failed password attempts: $total_failed"
echo "   Invalid user attempts: $total_invalid"
echo "   Sudo commands executed: $total_sudo"

# 8. 추천 보안 조치
echo
echo "💡 Security Recommendations:"

# 높은 실패 시도가 있는 IP 확인
high_fail_ips=$(grep "Failed password" /var/log/auth.log* | \
awk '{print $NF}' | sort | uniq -c | \
awk '$1 > 50 {print $2}' | wc -l)

if [ $high_fail_ips -gt 0 ]; then
    echo "   - Consider blocking IPs with high failure rates"
fi

# 비정상적인 시간대 접근 확인
night_logins=$(grep "Accepted password" /var/log/auth.log* | \
awk '{print $3}' | cut -d: -f1 | \
awk '$1 < 6 || $1 > 22' | wc -l)

if [ $night_logins -gt 0 ]; then
    echo "   - Review off-hours login activity"
fi

echo "=============================================="
```

## 다음 편 예고

다음 포스트에서는 **SSH 고급 보안 설정과 키 관리**를 상세히 다룰 예정입니다:
- SSH 키 기반 인증 구축
- SSH Certificate Authority 설정  
- 포트 포워딩 및 터널링 보안
- SSH 접근 제어 고급 기법

리눅스 기초 보안 설정을 완벽하게 마스터하셨나요? 🔐✨
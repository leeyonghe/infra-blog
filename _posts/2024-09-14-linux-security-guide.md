---
layout: post
title: "리눅스 보안 완전 가이드 | Complete Linux Security Guide - 시스템 보안부터 고급 보안 기법까지"
date: 2024-09-14 12:00:00 +0900
categories: [Security, Linux]
tags: [linux-security, firewall, selinux, hardening, intrusion-detection, container-security, compliance]
---

리눅스 시스템의 보안을 체계적으로 구축하고 관리하는 완전한 가이드입니다. 기초부터 고급 보안 기법까지 실무에 바로 적용할 수 있는 내용으로 구성했습니다.

## 리눅스 보안 기초 | Linux Security Fundamentals

### 🔐 사용자 계정 보안

#### 패스워드 정책 강화
```bash
# 패스워드 복잡성 정책 설정
# /etc/pam.d/common-password (Ubuntu/Debian)
password requisite pam_pwquality.so retry=3 minlen=12 minclass=3 maxrepeat=2 ucredit=-1 lcredit=-1 dcredit=-1 ocredit=-1

# /etc/security/pwquality.conf
minlen = 12
minclass = 3
maxrepeat = 2
maxclasssrepeat = 4
ucredit = -1    # 최소 대문자 1개
lcredit = -1    # 최소 소문자 1개  
dcredit = -1    # 최소 숫자 1개
ocredit = -1    # 최소 특수문자 1개
difok = 3       # 이전 패스워드와 3글자 이상 달라야 함

# 패스워드 에이징 정책
# /etc/login.defs
PASS_MAX_DAYS   90     # 최대 유효기간
PASS_MIN_DAYS   1      # 최소 변경 주기
PASS_MIN_LEN    12     # 최소 길이
PASS_WARN_AGE   7      # 만료 경고 기간

# 기존 사용자에게 정책 적용
chage -M 90 -m 1 -W 7 username

# 계정 잠금 정책 (실패 시도 제한)
# /etc/pam.d/common-auth
auth required pam_tally2.so deny=5 unlock_time=1800 onerr=fail

# 잠긴 계정 확인 및 해제
pam_tally2 --user username
pam_tally2 --user username --reset
```

#### 루트 계정 보안
```bash
# 루트 직접 로그인 차단
# /etc/ssh/sshd_config
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes

# 콘솔 루트 로그인 제한
# /etc/securetty (허용할 터미널만 남기기)
console
tty1

# sudo 권한 세밀하게 제어
# /etc/sudoers
# 특정 명령만 허용
webadmin ALL=(ALL) /usr/sbin/service apache2 *, /usr/sbin/service nginx *
# 패스워드 없이 특정 명령 실행
backup ALL=(ALL) NOPASSWD: /usr/bin/rsync, /bin/tar

# sudo 사용 로그 강화
# /etc/rsyslog.conf 또는 /etc/rsyslog.d/50-default.conf
local2.*                        /var/log/sudo.log

# /etc/sudoers에 추가
Defaults    logfile="/var/log/sudo.log"
Defaults    log_input, log_output
Defaults    iolog_dir="/var/log/sudo-io"
```

### 🔒 SSH 보안 강화

#### SSH 서버 보안 설정
```bash
# /etc/ssh/sshd_config 보안 설정
Protocol 2
Port 2222                    # 기본 포트 변경
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
MaxAuthTries 3
MaxSessions 3
MaxStartups 3
LoginGraceTime 60
ClientAliveInterval 300
ClientAliveCountMax 2
UsePAM yes
X11Forwarding no
AllowTcpForwarding no
GatewayPorts no
PermitTunnel no

# 특정 사용자/그룹만 SSH 접근 허용
AllowUsers admin developer
AllowGroups sshusers
DenyUsers guest anonymous
DenyGroups wheel

# 특정 IP에서만 접근 허용
Match Address 192.168.1.0/24,10.0.0.0/8
    PasswordAuthentication yes
    
Match Address *,!192.168.1.0/24,!10.0.0.0/8
    DenyUsers *

# 서비스 재시작
systemctl restart sshd
```

#### SSH 키 기반 인증 구축
```bash
# 클라이언트에서 키 생성 (ED25519 권장)
ssh-keygen -t ed25519 -b 4096 -C "your-email@domain.com"
# 또는 RSA 키 (최소 4096비트)
ssh-keygen -t rsa -b 4096 -C "your-email@domain.com"

# 공개키 서버에 복사
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@server-ip

# 수동으로 공개키 설정
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/authorized_keys << 'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG... your-email@domain.com
EOF
chmod 600 ~/.ssh/authorized_keys
chown username:username ~/.ssh -R

# SSH 키 관리 - 키별 제한 설정
# ~/.ssh/authorized_keys
command="/usr/local/bin/backup-script",no-port-forwarding,no-X11-forwarding,no-agent-forwarding ssh-rsa AAAAB3... backup@server
from="192.168.1.100",no-port-forwarding ssh-rsa AAAAB3... admin@workstation

# SSH Certificate Authority 구축
# CA 키 생성
ssh-keygen -t rsa -b 4096 -f /etc/ssh/ca_key

# 사용자 인증서 발급
ssh-keygen -s /etc/ssh/ca_key -I "user-certificate" -n username -V +1w ~/.ssh/id_rsa.pub

# 서버 설정에서 CA 신뢰
# /etc/ssh/sshd_config
TrustedUserCAKeys /etc/ssh/ca_key.pub
```

### 🛡️ 방화벽 설정

#### iptables 기본 보안 설정
```bash
#!/bin/bash
# 강력한 iptables 보안 설정

# 모든 기존 규칙 초기화
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# 기본 정책: 모든 트래픽 차단
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# Loopback 허용
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 기존 연결 유지
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT

# SSH 접근 허용 (브루트포스 방지)
iptables -A INPUT -p tcp --dport 2222 -m conntrack --ctstate NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 2222 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport 2222 -j ACCEPT
iptables -A OUTPUT -p tcp --sport 2222 -j ACCEPT

# DNS 허용 (필수)
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# NTP 허용
iptables -A OUTPUT -p udp --dport 123 -j ACCEPT

# HTTP/HTTPS 아웃바운드 허용
iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

# 웹 서버가 있다면 인바운드도 허용
# iptables -A INPUT -p tcp --dport 80 -j ACCEPT
# iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# ICMP 제한적 허용 (ping)
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/second -j ACCEPT
iptables -A OUTPUT -p icmp --icmp-type echo-reply -j ACCEPT
iptables -A OUTPUT -p icmp --icmp-type echo-request -j ACCEPT

# 로깅 설정 (DDoS 방지)
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables INPUT denied: " --log-level 7

# 설정 저장
iptables-save > /etc/iptables/rules.v4

# 부팅 시 자동 로드
echo '#!/bin/bash' > /etc/network/if-pre-up.d/iptables
echo 'iptables-restore < /etc/iptables/rules.v4' >> /etc/network/if-pre-up.d/iptables
chmod +x /etc/network/if-pre-up.d/iptables
```

#### UFW (Uncomplicated Firewall) 활용
```bash
# UFW 초기화 및 기본 설정
ufw --force reset
ufw default deny incoming
ufw default deny outgoing
ufw default deny forward

# 필수 아웃바운드 허용
ufw allow out 53      # DNS
ufw allow out 80      # HTTP
ufw allow out 443     # HTTPS
ufw allow out 123     # NTP

# SSH 접근 허용 (포트 변경했다면)
ufw allow from 192.168.1.0/24 to any port 2222

# 애플리케이션별 허용
ufw allow "Apache Full"
ufw allow "Nginx Full"

# 고급 규칙
ufw allow from 10.0.0.0/8 to any port 3306  # MySQL
ufw deny from 192.168.1.100                 # 특정 IP 차단

# 로깅 활성화
ufw logging on

# 방화벽 활성화
ufw enable

# 상태 확인
ufw status verbose
ufw status numbered
```

## SELinux/AppArmor 보안 | Mandatory Access Control

### 🔐 SELinux 구성 및 관리

#### SELinux 기본 설정
```bash
# SELinux 상태 확인
sestatus
getenforce

# SELinux 모드 변경
# /etc/selinux/config
SELINUX=enforcing     # enforcing, permissive, disabled
SELINUXTYPE=targeted  # targeted, minimum, mls

# 임시 모드 변경
setenforce 1  # enforcing
setenforce 0  # permissive

# 컨텍스트 확인
ls -lZ /var/www/html/
ps auxZ | grep httpd
id -Z

# 파일 컨텍스트 복원
restorecon -Rv /var/www/html/
restorecon -Rv /home/username/

# 컨텍스트 수동 설정
chcon -t httpd_exec_t /usr/local/apache2/bin/httpd
semanage fcontext -a -t httpd_exec_t "/usr/local/apache2/bin/httpd"
```

#### SELinux 정책 관리
```bash
# 불린 값 확인 및 설정
getsebool -a | grep httpd
setsebool -P httpd_can_network_connect on
setsebool -P httpd_can_sendmail on

# 포트 라벨 관리
semanage port -l | grep http
semanage port -a -t http_port_t -p tcp 8080

# 사용자 매핑
semanage login -l
semanage user -l
semanage login -a -s user_u regularuser

# 커스텀 정책 모듈 생성
# audit.log에서 정책 생성
grep httpd /var/log/audit/audit.log | audit2allow -m myhttpd
grep httpd /var/log/audit/audit.log | audit2allow -M myhttpd
semodule -i myhttpd.pp

# 정책 모듈 관리
semodule -l | grep my
semodule -r myhttpd
```

### 🛡️ AppArmor 프로파일 작성

#### AppArmor 기본 관리
```bash
# AppArmor 상태 확인
aa-status
aa-enabled

# 프로파일 모드 확인
aa-status | grep profiles

# 프로파일 모드 변경
aa-enforce /etc/apparmor.d/usr.bin.firefox
aa-complain /etc/apparmor.d/usr.bin.firefox
aa-disable /etc/apparmor.d/usr.bin.firefox

# 프로파일 재로드
apparmor_parser -r /etc/apparmor.d/usr.bin.firefox
```

#### 커스텀 AppArmor 프로파일 작성
```bash
# 새로운 애플리케이션 프로파일 생성
# /etc/apparmor.d/usr.local.bin.myapp
#include <tunables/global>

/usr/local/bin/myapp {
  #include <abstractions/base>
  #include <abstractions/nameservice>
  
  # 실행 파일
  /usr/local/bin/myapp mr,
  
  # 라이브러리
  /lib{,32,64}/** mr,
  /usr/lib{,32,64}/** mr,
  
  # 설정 파일 (읽기 전용)
  /etc/myapp/** r,
  owner /home/*/.myapp/** rw,
  
  # 데이터 디렉토리
  /var/lib/myapp/** rw,
  /var/log/myapp/** w,
  
  # 네트워크 접근
  network inet stream,
  network inet6 stream,
  
  # 프로세스 제어
  capability setuid,
  capability setgid,
  
  # 임시 파일
  /tmp/myapp.** rw,
  owner /tmp/myapp-** rw,
  
  # 거부할 접근
  deny /etc/passwd r,
  deny /etc/shadow r,
  deny owner /home/*/.ssh/** rw,
  
  # 하위 프로세스 실행
  /bin/dash ix,
  /usr/bin/python3 ix,
}

# 프로파일 로드 및 활성화
apparmor_parser -r /etc/apparmor.d/usr.local.bin.myapp
aa-enforce /etc/apparmor.d/usr.local.bin.myapp

# 프로파일 개발 모드 (학습)
aa-genprof /usr/local/bin/myapp
# 애플리케이션 실행하며 동작 확인
aa-logprof
```

## 시스템 하드닝 | System Hardening

### 🔧 커널 보안 매개변수 조정

#### sysctl 보안 설정
```bash
# /etc/sysctl.d/99-security.conf
# 네트워크 보안
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# SYN 플러드 방지
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# ICMP 보안
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.icmp_echo_ignore_all = 0

# IP 스푸핑 방지
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# IPv6 보안
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# 메모리 보안
kernel.randomize_va_space = 2
kernel.exec-shield = 1
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2

# 프로세스 보안  
fs.suid_dumpable = 0
kernel.core_uses_pid = 1
kernel.ctrl-alt-del = 0

# 설정 적용
sysctl -p /etc/sysctl.d/99-security.conf
```

#### 파일 시스템 보안
```bash
# 중요 디렉토리 마운트 옵션 강화
# /etc/fstab
/tmp        /tmp        tmpfs   defaults,nodev,nosuid,noexec    0 0
/var/tmp    /var/tmp    tmpfs   defaults,nodev,nosuid,noexec    0 0
/dev/shm    /dev/shm    tmpfs   defaults,nodev,nosuid,noexec    0 0

# 파일 권한 강화
chmod 700 /root
chmod 600 /etc/ssh/sshd_config
chmod 600 /etc/passwd-
chmod 600 /etc/shadow
chmod 600 /etc/gshadow
chmod 644 /etc/group

# 불필요한 SUID/SGID 제거
find / -type f \( -perm -4000 -o -perm -2000 \) -print > /tmp/suid_sgid_files
# 검토 후 필요없는 것들 제거
chmod u-s /usr/bin/unnecessary-suid-program

# 숨겨진 파일 및 디렉토리 검사
find / -name ".*" -type f -exec ls -la {} \; 2>/dev/null
find / -name ".*" -type d -exec ls -lad {} \; 2>/dev/null

# 대용량 파일 검사 (backdoor 가능성)
find / -size +10M -type f -exec ls -la {} \; 2>/dev/null

# 최근 수정된 파일 검사
find / -mtime -7 -type f -exec ls -la {} \; 2>/dev/null
```

### 🔍 서비스 및 데몬 보안

#### 불필요한 서비스 제거
```bash
# 실행 중인 서비스 확인
systemctl list-units --type=service --state=running
netstat -tulpn
ss -tulpn

# 불필요한 서비스 중지 및 비활성화
systemctl stop cups
systemctl disable cups
systemctl mask cups

# 위험한 서비스들 (일반적으로 비활성화)
services_to_disable=(
    "telnet"
    "rsh"
    "rlogin"
    "tftp"
    "xinetd"
    "sendmail"
    "postfix"
    "dovecot"
    "cups"
    "avahi-daemon"
    "bluetooth"
)

for service in "${services_to_disable[@]}"; do
    if systemctl is-enabled "$service" >/dev/null 2>&1; then
        echo "Disabling $service..."
        systemctl stop "$service"
        systemctl disable "$service"
        systemctl mask "$service"
    fi
done

# 네트워크 서비스 점검
lsof -i
netstat -anp | grep LISTEN
```

#### 애플리케이션별 보안 설정
```bash
# Apache 보안 설정
# /etc/apache2/conf-available/security.conf
ServerTokens Prod
ServerSignature Off
TraceEnable Off
Header always set X-Content-Type-Options nosniff
Header always set X-Frame-Options DENY
Header always set X-XSS-Protection "1; mode=block"
Header always set Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"

# Nginx 보안 설정
# /etc/nginx/nginx.conf
server_tokens off;
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

# MySQL/MariaDB 보안
mysql_secure_installation

# PostgreSQL 보안 설정
# /etc/postgresql/*/main/postgresql.conf
ssl = on
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
ssl_prefer_server_ciphers = on

# /etc/postgresql/*/main/pg_hba.conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
```

## 침입 탐지 시스템 | Intrusion Detection Systems

### 🚨 AIDE (Advanced Intrusion Detection Environment)

#### AIDE 설치 및 구성
```bash
# AIDE 설치
apt-get install aide aide-common

# 설정 파일 수정
# /etc/aide/aide.conf
database=file:/var/lib/aide/aide.db
database_out=file:/var/lib/aide/aide.db.new
gzip_dbout=yes
report_url=file:/var/log/aide/aide.log
report_url=stdout

# 규칙 정의
/boot   f+p+u+g+s+b+m+c+md5+sha1
/bin    f+p+u+g+s+b+m+c+md5+sha1
/sbin   f+p+u+g+s+b+m+c+md5+sha1
/lib    f+p+u+g+s+b+m+c+md5+sha1
/opt    f+p+u+g+s+b+m+c+md5+sha1
/usr    f+p+u+g+s+b+m+c+md5+sha1
/root   f+p+u+g+s+b+m+c+md5+sha1
/etc    f+p+u+g+s+b+m+c+md5+sha1

# 제외할 디렉토리
!/var/log/.*
!/var/spool/.*
!/var/run/.*
!/var/lock/.*
!/proc/.*
!/sys/.*
!/dev/.*
!/tmp/.*

# 초기 데이터베이스 생성
aideinit

# 데이터베이스 업데이트
cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# 무결성 검사 실행
aide --check

# 자동화 스크립트
#!/bin/bash
# /usr/local/bin/aide-check.sh
AIDE_LOG="/var/log/aide/aide-$(date +%Y%m%d).log"

aide --check > "$AIDE_LOG" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "AIDE detected changes on $(hostname)" | \
    mail -s "AIDE Alert - File Integrity Check Failed" admin@company.com \
    -A "$AIDE_LOG"
fi

# cron 설정
# /etc/cron.d/aide
0 2 * * * root /usr/local/bin/aide-check.sh
```

### 🔍 Fail2Ban 설정

#### Fail2Ban 구성 및 커스터마이징
```bash
# Fail2Ban 설치
apt-get install fail2ban

# 기본 설정 파일 복사
cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# /etc/fail2ban/jail.local 설정
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd
banaction = iptables-multiport
banaction_allports = iptables-allports
ignoreip = 127.0.0.1/8 192.168.1.0/24 10.0.0.0/8

# SSH 보호
[sshd]
enabled = true
port = 2222
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

# Apache 보호
[apache-auth]
enabled = true
port = http,https
logpath = /var/log/apache2/error.log

[apache-badbots]
enabled = true
port = http,https
logpath = /var/log/apache2/access.log

[apache-noscript]
enabled = true
port = http,https
logpath = /var/log/apache2/access.log

# 커스텀 필터 생성
# /etc/fail2ban/filter.d/custom-app.conf
[Definition]
failregex = ^.*Failed login attempt from <HOST>.*$
            ^.*Invalid user .* from <HOST>.*$
            ^.*authentication failure.*rhost=<HOST>.*$
ignoreregex =

# 커스텀 jail 설정
# /etc/fail2ban/jail.local에 추가
[custom-app]
enabled = true
port = 8080
logpath = /var/log/custom-app.log
filter = custom-app
maxretry = 5
bantime = 7200

# Fail2Ban 관리 명령
fail2ban-client status
fail2ban-client status sshd
fail2ban-client unban 192.168.1.100
fail2ban-client reload
```

### 🔐 OSSEC/Wazuh 설치

#### Wazuh Agent 설치 및 구성
```bash
# Wazuh 저장소 추가
curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | apt-key add -
echo "deb https://packages.wazuh.com/4.x/apt/ stable main" > /etc/apt/sources.list.d/wazuh.list
apt-get update

# Wazuh Agent 설치
apt-get install wazuh-agent

# 에이전트 설정
# /var/ossec/etc/ossec.conf
<ossec_config>
  <client>
    <server>
      <address>MANAGER_IP</address>
      <port>1514</port>
      <protocol>tcp</protocol>
    </server>
    <config-profile>linux, ubuntu, ubuntu20,</config-profile>
    <notify_time>10</notify_time>
    <time-reconnect>60</time-reconnect>
    <auto_restart>yes</auto_restart>
    <crypto_method>aes</crypto_method>
  </client>

  <!-- 로그 모니터링 -->
  <localfile>
    <log_format>syslog</log_format>
    <location>/var/log/auth.log</location>
  </localfile>

  <localfile>
    <log_format>syslog</log_format>
    <location>/var/log/syslog</location>
  </localfile>

  <localfile>
    <log_format>apache</log_format>
    <location>/var/log/apache2/access.log</location>
  </localfile>

  <!-- 파일 무결성 모니터링 -->
  <syscheck>
    <disabled>no</disabled>
    <frequency>43200</frequency>
    <scan_on_start>yes</scan_on_start>
    
    <directories check_all="yes" realtime="yes">/etc</directories>
    <directories check_all="yes" realtime="yes">/usr/bin</directories>
    <directories check_all="yes" realtime="yes">/usr/sbin</directories>
    <directories check_all="yes" realtime="yes">/bin</directories>
    <directories check_all="yes" realtime="yes">/sbin</directories>
    
    <ignore>/etc/mtab</ignore>
    <ignore>/etc/hosts.deny</ignore>
    <ignore>/etc/mail/statistics</ignore>
    <ignore>/etc/random-seed</ignore>
    <ignore>/etc/adjtime</ignore>
  </syscheck>

  <!-- 루트킷 탐지 -->
  <rootcheck>
    <disabled>no</disabled>
    <check_files>yes</check_files>
    <check_trojans>yes</check_trojans>
    <check_dev>yes</check_dev>
    <check_sys>yes</check_sys>
    <check_pids>yes</check_pids>
    <check_ports>yes</check_ports>
    <check_if>yes</check_if>
  </rootcheck>

  <!-- 활성 응답 -->
  <active-response>
    <disabled>no</disabled>
    <ca_store>/var/ossec/etc/wpk_root.pem</ca_store>
    <ca_verification>yes</ca_verification>
  </active-response>
</ossec_config>

# 서비스 시작
systemctl enable wazuh-agent
systemctl start wazuh-agent

# 상태 확인
systemctl status wazuh-agent
/var/ossec/bin/ossec-control status
```

## 컨테이너 보안 | Container Security

### 🐳 Docker 보안 설정

#### Docker 데몬 보안 강화
```bash
# Docker 데몬 설정
# /etc/docker/daemon.json
{
    "icc": false,
    "userns-remap": "default",
    "no-new-privileges": true,
    "seccomp-profile": "/etc/docker/seccomp.json",
    "selinux-enabled": true,
    "disable-legacy-registry": true,
    "live-restore": true,
    "userland-proxy": false,
    "experimental": false,
    "metrics-addr": "127.0.0.1:9323",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "default-ulimits": {
        "nofile": {
            "name": "nofile",
            "hard": 64000,
            "soft": 64000
        }
    }
}

# 시스템 재시작
systemctl restart docker

# 보안 컨테이너 실행 예시
docker run -d \
  --name secure-app \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  --tmpfs /var/lock \
  --user 1000:1000 \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --security-opt no-new-privileges:true \
  --security-opt apparmor:docker-default \
  --memory 512m \
  --cpus="0.5" \
  --pids-limit 100 \
  --restart unless-stopped \
  nginx:alpine
```

#### Dockerfile 보안 모범 사례
```dockerfile
# 보안 강화 Dockerfile 예시
FROM alpine:3.18

# 보안 업데이트 적용
RUN apk update && apk upgrade && \
    apk add --no-cache tini && \
    rm -rf /var/cache/apk/*

# 비특권 사용자 생성
RUN addgroup -g 1000 -S appgroup && \
    adduser -u 1000 -S appuser -G appgroup

# 애플리케이션 디렉토리 생성 및 권한 설정
WORKDIR /app
COPY --chown=appuser:appgroup . .

# 실행 파일 권한만 부여
RUN chmod 755 /app/entrypoint.sh && \
    chmod 644 /app/*.conf

# 비특권 사용자로 전환
USER appuser:appgroup

# 시그널 처리를 위한 tini 사용
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["./entrypoint.sh"]

# 불필요한 네트워크 포트 노출 금지
# EXPOSE 80 (필요한 경우만)

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

### ⚓ Kubernetes 보안

#### Pod Security Standards 적용
```yaml
# Pod Security Policy (deprecated, use Pod Security Standards)
apiVersion: v1
kind: Namespace
metadata:
  name: secure-namespace
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

---
# 보안 강화 Pod 예시
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  namespace: secure-namespace
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: nginx:alpine
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE
    resources:
      limits:
        memory: "256Mi"
        cpu: "200m"
      requests:
        memory: "128Mi"
        cpu: "100m"
    volumeMounts:
    - name: tmp-volume
      mountPath: /tmp
    - name: var-cache-nginx
      mountPath: /var/cache/nginx
    - name: var-run
      mountPath: /var/run
  volumes:
  - name: tmp-volume
    emptyDir: {}
  - name: var-cache-nginx
    emptyDir: {}
  - name: var-run
    emptyDir: {}

---
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: secure-namespace
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-web-traffic
  namespace: secure-namespace
spec:
  podSelector:
    matchLabels:
      app: web
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 80
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  - to: {}
    ports:
    - protocol: UDP
      port: 53
```

## 컴플라이언스와 감사 | Compliance and Auditing

### 📋 CIS Benchmark 적용

#### CIS Controls 자동화 스크립트
```bash
#!/bin/bash
# CIS Ubuntu 20.04 LTS Benchmark 자동 적용 스크립트

echo "CIS Benchmark 자동 적용을 시작합니다..."

# 1. 파일 시스템 구성
echo "1. 파일 시스템 보안 설정..."

# 1.1 임시 파일 시스템 보안
if ! grep -q "/tmp.*nodev" /etc/fstab; then
    echo "/tmp /tmp tmpfs defaults,rw,nosuid,nodev,noexec,relatime 0 0" >> /etc/fstab
fi

# 1.2 불필요한 파일 시스템 비활성화
cat >> /etc/modprobe.d/blacklist-rare-filesystems.conf << 'EOF'
install cramfs /bin/true
install freevxfs /bin/true
install jffs2 /bin/true
install hfs /bin/true
install hfsplus /bin/true
install squashfs /bin/true
install udf /bin/true
install fat /bin/true
install vfat /bin/true
install usb-storage /bin/true
EOF

# 2. 서비스 설정
echo "2. 서비스 보안 설정..."

# 2.1 시간 동기화
systemctl enable ntp
systemctl start ntp

# 2.2 불필요한 서비스 비활성화
services_to_disable=(
    "autofs"
    "avahi-daemon" 
    "cups"
    "dhcpd"
    "slapd"
    "nfs"
    "rpcbind"
    "bind9"
    "vsftpd"
    "apache2"
    "dovecot"
    "smbd"
    "squid"
    "snmpd"
    "rsync"
)

for service in "${services_to_disable[@]}"; do
    if systemctl is-enabled "$service" >/dev/null 2>&1; then
        systemctl disable "$service"
        systemctl stop "$service"
    fi
done

# 3. 네트워크 매개변수 설정
echo "3. 네트워크 보안 설정..."
cat > /etc/sysctl.d/99-cis.conf << 'EOF'
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_syncookies = 1
net.ipv6.conf.all.accept_ra = 0
net.ipv6.conf.default.accept_ra = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv6.conf.all.disable_ipv6 = 1
EOF

sysctl -p /etc/sysctl.d/99-cis.conf

# 4. 로깅 및 감사 설정
echo "4. 로깅 및 감사 설정..."

# 4.1 auditd 설치 및 구성
apt-get install -y auditd audispd-plugins

cat > /etc/audit/rules.d/cis.rules << 'EOF'
# 시간 변경 감사
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change
-a always,exit -F arch=b64 -S clock_settime -k time-change
-a always,exit -F arch=b32 -S clock_settime -k time-change
-w /etc/localtime -p wa -k time-change

# 사용자/그룹 정보 감사
-w /etc/group -p wa -k identity
-w /etc/passwd -p wa -k identity
-w /etc/gshadow -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/security/opasswd -p wa -k identity

# 네트워크 환경 감사
-a always,exit -F arch=b64 -S sethostname -S setdomainname -k system-locale
-a always,exit -F arch=b32 -S sethostname -S setdomainname -k system-locale
-w /etc/issue -p wa -k system-locale
-w /etc/issue.net -p wa -k system-locale
-w /etc/hosts -p wa -k system-locale
-w /etc/network -p wa -k system-locale

# MAC 정책 변경 감사
-w /etc/selinux/ -p wa -k MAC-policy
-w /usr/share/selinux/ -p wa -k MAC-policy

# 로그인/로그아웃 감사
-w /var/log/faillog -p wa -k logins
-w /var/log/lastlog -p wa -k logins
-w /var/log/tallylog -p wa -k logins

# 세션 시작 정보 감사
-w /var/run/utmp -p wa -k session
-w /var/log/wtmp -p wa -k logins
-w /var/log/btmp -p wa -k logins

# 권한 변경 감사
-a always,exit -F arch=b64 -S chmod -S fchmod -S fchmodat -F auid>=1000 -F auid!=4294967295 -k perm_mod
-a always,exit -F arch=b32 -S chmod -S fchmod -S fchmodat -F auid>=1000 -F auid!=4294967295 -k perm_mod
-a always,exit -F arch=b64 -S chown -S fchown -S fchownat -S lchown -F auid>=1000 -F auid!=4294967295 -k perm_mod
-a always,exit -F arch=b32 -S chown -S fchown -S fchownat -S lchown -F auid>=1000 -F auid!=4294967295 -k perm_mod

# 관리자 액세스 감사
-w /etc/sudoers -p wa -k scope
-w /etc/sudoers.d/ -p wa -k scope

# 커널 모듈 로딩/언로딩 감사
-w /sbin/insmod -p x -k modules
-w /sbin/rmmod -p x -k modules
-w /sbin/modprobe -p x -k modules
-a always,exit -F arch=b64 -S init_module -S delete_module -k modules

# 파일 삭제 감사
-a always,exit -F arch=b64 -S unlink -S unlinkat -S rename -S renameat -F auid>=1000 -F auid!=4294967295 -k delete
-a always,exit -F arch=b32 -S unlink -S unlinkat -S rename -S renameat -F auid>=1000 -F auid!=4294967295 -k delete

# 설정 불변성
-e 2
EOF

systemctl enable auditd
systemctl start auditd

# 5. 액세스 제어 설정
echo "5. 액세스 제어 설정..."

# 5.1 cron 접근 제한
echo "root" > /etc/cron.allow
chmod 600 /etc/cron.allow
rm -f /etc/cron.deny

# 5.2 SSH 보안 강화 (이미 앞에서 다룸)
# 5.3 PAM 설정 강화 (이미 앞에서 다룸)

# 6. 시스템 유지보수
echo "6. 시스템 유지보수 설정..."

# 6.1 파일 권한 점검
find /etc -type f -perm /g+w,o+w -exec chmod go-w {} \;

# 6.2 SUID/SGID 점검 스크립트 생성
cat > /usr/local/bin/check-suid-sgid.sh << 'EOF'
#!/bin/bash
find / \( -perm -4000 -o -perm -2000 \) -type f -exec ls -ld {} \; 2>/dev/null | \
while read line; do
    echo "$(date): $line" >> /var/log/suid-sgid.log
done
EOF

chmod +x /usr/local/bin/check-suid-sgid.sh

# 6.3 정기 보안 점검 cron 설정
cat > /etc/cron.daily/security-check << 'EOF'
#!/bin/bash
/usr/local/bin/check-suid-sgid.sh
/usr/bin/aide --check 2>&1 | logger -t aide
EOF

chmod +x /etc/cron.daily/security-check

echo "CIS Benchmark 적용 완료!"
echo "시스템을 재부팅하여 모든 설정을 적용하세요."
```

### 📊 보안 모니터링 대시보드

#### Prometheus + Grafana 보안 메트릭
```yaml
# prometheus-rules.yml
groups:
- name: security_rules
  rules:
  - alert: HighFailedLoginRate
    expr: rate(node_auth_failed_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High failed login rate detected"
      description: "{% raw %}{{ $labels.instance }}{% endraw %} has a high failed login rate of {% raw %}{{ $value }}{% endraw %} per second"

  - alert: RootLoginDetected
    expr: increase(node_auth_success_total{user="root"}[1m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Root login detected"
      description: "Root user login detected on {% raw %}{{ $labels.instance }}{% endraw %}"

  - alert: SudoCommandExecuted
    expr: increase(node_sudo_commands_total[1m]) > 0
    for: 0m
    labels:
      severity: info
    annotations:
      summary: "Sudo command executed"
      description: "Sudo command executed on {% raw %}{{ $labels.instance }}{% endraw %}"

  - alert: FileSystemModification
    expr: rate(node_filesystem_files_free[5m]) < -0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Rapid file system changes detected"
      description: "Rapid file system changes on {% raw %}{{ $labels.instance }}{% endraw %} at {% raw %}{{ $labels.mountpoint }}{% endraw %}"
```

#### 보안 로그 분석 스크립트
```python
#!/usr/bin/env python3
import re
import json
import datetime
from collections import defaultdict, Counter
import argparse

class SecurityLogAnalyzer:
    def __init__(self):
        self.failed_logins = defaultdict(list)
        self.successful_logins = defaultdict(list)
        self.sudo_commands = []
        self.suspicious_activities = []
        
    def parse_auth_log(self, log_file):
        """인증 로그 파싱"""
        patterns = {
            'failed_login': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*Failed password for (\w+) from (\d+\.\d+\.\d+\.\d+)',
            'successful_login': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*Accepted password for (\w+) from (\d+\.\d+\.\d+\.\d+)',
            'sudo_command': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*sudo.*USER=(\w+).*COMMAND=(.*)',
            'invalid_user': r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}).*Invalid user (\w+) from (\d+\.\d+\.\d+\.\d+)'
        }
        
        with open(log_file, 'r') as f:
            for line in f:
                # 실패한 로그인 시도
                match = re.search(patterns['failed_login'], line)
                if match:
                    timestamp, user, ip = match.groups()
                    self.failed_logins[ip].append({
                        'timestamp': timestamp,
                        'user': user,
                        'ip': ip
                    })
                
                # 성공한 로그인
                match = re.search(patterns['successful_login'], line)
                if match:
                    timestamp, user, ip = match.groups()
                    self.successful_logins[ip].append({
                        'timestamp': timestamp,
                        'user': user,
                        'ip': ip
                    })
                
                # sudo 명령
                match = re.search(patterns['sudo_command'], line)
                if match:
                    timestamp, user, command = match.groups()
                    self.sudo_commands.append({
                        'timestamp': timestamp,
                        'user': user,
                        'command': command
                    })
                
                # 잘못된 사용자
                match = re.search(patterns['invalid_user'], line)
                if match:
                    timestamp, user, ip = match.groups()
                    self.suspicious_activities.append({
                        'type': 'invalid_user',
                        'timestamp': timestamp,
                        'user': user,
                        'ip': ip
                    })
    
    def detect_brute_force(self, threshold=10):
        """브루트포스 공격 탐지"""
        brute_force_ips = []
        
        for ip, attempts in self.failed_logins.items():
            if len(attempts) >= threshold:
                brute_force_ips.append({
                    'ip': ip,
                    'attempts': len(attempts),
                    'users_targeted': list(set([attempt['user'] for attempt in attempts])),
                    'first_attempt': attempts[0]['timestamp'],
                    'last_attempt': attempts[-1]['timestamp']
                })
        
        return brute_force_ips
    
    def detect_privilege_escalation(self):
        """권한 상승 탐지"""
        privilege_escalations = []
        
        dangerous_commands = [
            'passwd', 'useradd', 'usermod', 'userdel',
            'chmod 777', 'chmod 4755', 'chown root',
            'systemctl', 'service', 'crontab -e',
            '/bin/bash', '/bin/sh', 'su -'
        ]
        
        for sudo_cmd in self.sudo_commands:
            for dangerous_cmd in dangerous_commands:
                if dangerous_cmd in sudo_cmd['command']:
                    privilege_escalations.append({
                        'timestamp': sudo_cmd['timestamp'],
                        'user': sudo_cmd['user'],
                        'command': sudo_cmd['command'],
                        'risk_level': 'high' if dangerous_cmd in ['passwd', 'useradd', '/bin/bash'] else 'medium'
                    })
                    break
        
        return privilege_escalations
    
    def analyze_login_patterns(self):
        """로그인 패턴 분석"""
        patterns = {
            'geographic_anomalies': [],
            'time_anomalies': [],
            'user_anomalies': []
        }
        
        # 사용자별 로그인 빈도 분석
        user_login_count = Counter()
        for ip, logins in self.successful_logins.items():
            for login in logins:
                user_login_count[login['user']] += 1
        
        # 비정상적으로 많은 로그인
        avg_logins = sum(user_login_count.values()) / len(user_login_count) if user_login_count else 0
        for user, count in user_login_count.items():
            if count > avg_logins * 3:  # 평균의 3배 이상
                patterns['user_anomalies'].append({
                    'user': user,
                    'login_count': count,
                    'anomaly_type': 'excessive_logins'
                })
        
        return patterns
    
    def generate_report(self):
        """보안 분석 리포트 생성"""
        report = {
            'analysis_date': datetime.datetime.now().isoformat(),
            'summary': {
                'total_failed_logins': sum(len(attempts) for attempts in self.failed_logins.values()),
                'total_successful_logins': sum(len(logins) for logins in self.successful_logins.values()),
                'total_sudo_commands': len(self.sudo_commands),
                'suspicious_activities': len(self.suspicious_activities)
            },
            'brute_force_attacks': self.detect_brute_force(),
            'privilege_escalations': self.detect_privilege_escalation(),
            'login_patterns': self.analyze_login_patterns(),
            'suspicious_activities': self.suspicious_activities,
            'recommendations': []
        }
        
        # 추천사항 생성
        if report['brute_force_attacks']:
            report['recommendations'].append(
                "브루트포스 공격이 탐지되었습니다. fail2ban 설정을 검토하고 IP 차단을 고려하세요."
            )
        
        if report['privilege_escalations']:
            report['recommendations'].append(
                "권한 상승 활동이 탐지되었습니다. sudo 사용을 검토하고 필요시 권한을 제한하세요."
            )
        
        if report['summary']['total_failed_logins'] > 100:
            report['recommendations'].append(
                "과도한 실패 로그인 시도가 있습니다. 패스워드 정책을 강화하세요."
            )
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Security Log Analyzer')
    parser.add_argument('--log-file', default='/var/log/auth.log', 
                       help='Path to authentication log file')
    parser.add_argument('--output', '-o', help='Output file for JSON report')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Brute force detection threshold')
    
    args = parser.parse_args()
    
    analyzer = SecurityLogAnalyzer()
    analyzer.parse_auth_log(args.log_file)
    
    report = analyzer.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
```

## 마무리와 다음 단계

이 종합적인 리눅스 보안 가이드를 통해 다음과 같은 내용을 다뤘습니다:

### ✅ 다룬 주요 내용
- **기초 보안**: 사용자 계정, SSH, 방화벽 설정
- **고급 보안**: SELinux/AppArmor, 시스템 하드닝
- **침입 탐지**: AIDE, Fail2Ban, Wazuh 구축
- **컨테이너 보안**: Docker/Kubernetes 보안 설정
- **컴플라이언스**: CIS Benchmark 적용
- **모니터링**: 보안 로그 분석 및 대시보드

### 🚀 다음 단계 권장사항

1. **정기적인 보안 점검** - 월간 보안 체크리스트 운영
2. **침투 테스트** - 분기별 모의 해킹 테스트 수행  
3. **보안 교육** - 팀원 대상 보안 인식 교육
4. **인시던트 대응** - 보안 사고 대응 절차 수립
5. **백업 및 복구** - 정기적인 백업 및 복구 테스트

### 📚 추가 학습 자료
- **인증**: CompTIA Security+, CISSP, CEH
- **도구**: Metasploit, Nessus, OpenVAS
- **표준**: NIST Cybersecurity Framework, ISO 27001

리눅스 보안은 지속적인 과정입니다. 항상 최신 위협 동향을 파악하고 보안 설정을 업데이트하세요! 🔒🛡️
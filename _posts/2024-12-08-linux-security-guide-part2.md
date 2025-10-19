---
layout: post
title: "리눅스 보안 완전 가이드 2편 - SSH 고급 보안과 방화벽 | Linux Security Guide Part 2 - Advanced SSH Security & Firewall"
date: 2024-12-08 09:00:00 +0900
categories: [Security, Linux]
tags: [ssh, firewall, iptables, ufw, ssh-keys, port-forwarding]
---

SSH 보안을 한 단계 높이고 강력한 방화벽 정책을 구축하는 방법을 완벽하게 마스터해보겠습니다. SSH 키 관리부터 고급 방화벽 설정까지 실무에서 바로 활용할 수 있는 내용으로 구성했습니다.

## SSH 고급 보안 설정 | Advanced SSH Security

### 🔑 SSH 키 기반 인증 완벽 구축

#### 강화된 SSH 키 생성 및 관리
```bash
# 최신 암호화 알고리즘을 사용한 키 생성
# ED25519 (권장 - 빠르고 안전)
ssh-keygen -t ed25519 -b 4096 -C "admin@company.com-$(date +%Y%m%d)" -f ~/.ssh/id_ed25519_admin

# RSA 키 (호환성이 필요한 경우)
ssh-keygen -t rsa -b 4096 -C "admin@company.com-$(date +%Y%m%d)" -f ~/.ssh/id_rsa_admin

# ECDSA 키 (대안)
ssh-keygen -t ecdsa -b 521 -C "admin@company.com-$(date +%Y%m%d)" -f ~/.ssh/id_ecdsa_admin

# 키 생성 시 보안 옵션
ssh-keygen -t ed25519 -b 4096 \
    -C "admin@company.com-$(date +%Y%m%d)" \
    -f ~/.ssh/id_ed25519_admin \
    -N "$(openssl rand -base64 32)" \  # 강력한 패스프레이즈 자동 생성
    -o \                               # OpenSSH 형식 사용
    -a 100                            # KDF 라운드 수 증가

# 키 권한 설정 (중요!)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_*
chmod 644 ~/.ssh/id_*.pub
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/config

# 키 핑거프린트 확인
ssh-keygen -lf ~/.ssh/id_ed25519_admin.pub
ssh-keygen -E sha256 -lf ~/.ssh/id_ed25519_admin.pub  # SHA256 해시

# 키 만료일 설정 (OpenSSH 8.2+)
ssh-keygen -t ed25519 -V +365d -C "expires-$(date -d '+1 year' +%Y%m%d)"
```

#### SSH 클라이언트 설정 최적화
```bash
# ~/.ssh/config - 클라이언트 설정
# 기본 설정
Host *
    # 보안 설정
    Protocol 2
    HashKnownHosts yes
    VisualHostKey yes
    StrictHostKeyChecking ask
    UserKnownHostsFile ~/.ssh/known_hosts
    
    # 연결 설정
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ConnectTimeout 10
    TCPKeepAlive no
    
    # 암호화 설정
    Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com
    KexAlgorithms curve25519-sha256@libssh.org,diffie-hellman-group16-sha512
    MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com
    HostKeyAlgorithms ssh-ed25519,ecdsa-sha2-nistp256,ecdsa-sha2-nistp384,ecdsa-sha2-nistp521
    
    # 인증 설정
    PreferredAuthentications publickey,keyboard-interactive,password
    PubkeyAuthentication yes
    PasswordAuthentication no
    GSSAPIAuthentication no
    
    # 포워딩 설정 (기본적으로 비활성화)
    ForwardAgent no
    ForwardX11 no
    ForwardX11Trusted no

# 서버별 개별 설정
Host production-server
    HostName 192.168.1.100
    Port 2222
    User admin
    IdentityFile ~/.ssh/id_ed25519_admin
    IdentitiesOnly yes
    RequestTTY yes
    RemoteForward 9000 localhost:9000
    
Host development-*
    Port 22
    User developer
    IdentityFile ~/.ssh/id_ed25519_dev
    StrictHostKeyChecking no        # 개발 서버용
    UserKnownHostsFile /dev/null
    LogLevel QUIET

Host bastion
    HostName bastion.company.com
    Port 2222
    User jumpuser
    IdentityFile ~/.ssh/id_ed25519_jump
    ControlMaster auto
    ControlPath ~/.ssh/control-%h-%p-%r
    ControlPersist 600

# 베스천 호스트를 통한 접근
Host internal-*
    ProxyJump bastion
    User admin
    IdentityFile ~/.ssh/id_ed25519_admin

# 키 에이전트 설정
# ~/.bashrc에 추가
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval $(ssh-agent -s)
    ssh-add ~/.ssh/id_ed25519_admin
    ssh-add ~/.ssh/id_ed25519_dev
fi

# 키 만료 확인 스크립트
#!/bin/bash
# /usr/local/bin/check-ssh-keys.sh
for keyfile in ~/.ssh/id_*; do
    if [[ -f "$keyfile" && ! "$keyfile" == *.pub ]]; then
        echo "Checking $keyfile..."
        ssh-keygen -l -f "$keyfile" 2>/dev/null || echo "  Invalid or encrypted key"
    fi
done
```

#### SSH Certificate Authority (CA) 구축
```bash
# 1. CA 키 생성 (보안이 중요한 별도 시스템에서)
ssh-keygen -t ed25519 -f /etc/ssh/ca_key -C "SSH-CA-$(date +%Y%m%d)"
chmod 600 /etc/ssh/ca_key
chmod 644 /etc/ssh/ca_key.pub

# 2. 사용자 인증서 발급
# 단기간 유효한 사용자 인증서 (1주일)
ssh-keygen -s /etc/ssh/ca_key \
    -I "john-doe-$(date +%Y%m%d)" \
    -n john,admin \
    -V +7d \
    -z 1001 \
    ~/.ssh/id_ed25519.pub

# 호스트별 제한된 인증서
ssh-keygen -s /etc/ssh/ca_key \
    -I "backup-service" \
    -n backup \
    -V +1d \
    -O clear \
    -O source-address="192.168.1.100/32" \
    -O force-command="/usr/local/bin/backup-script" \
    ~/.ssh/id_ed25519_backup.pub

# 권한 제한 인증서
ssh-keygen -s /etc/ssh/ca_key \
    -I "readonly-access" \
    -n readonly \
    -V +1h \
    -O clear \
    -O no-agent-forwarding \
    -O no-port-forwarding \
    -O no-pty \
    -O no-user-rc \
    ~/.ssh/id_ed25519_readonly.pub

# 3. 서버 설정에서 CA 신뢰
# /etc/ssh/sshd_config
TrustedUserCAKeys /etc/ssh/ca_key.pub
AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u
PubkeyAuthentication yes
CertificateAuthentication yes

# 4. 사용자별 주체(principal) 설정
# /etc/ssh/auth_principals/john
john
admin
developer

# /etc/ssh/auth_principals/backup
backup

# 5. 호스트 인증서도 구축
# 호스트 키에 대한 인증서 발급
ssh-keygen -s /etc/ssh/ca_key \
    -I "server1.company.com" \
    -h \
    -n server1.company.com,server1,192.168.1.100 \
    -V +365d \
    /etc/ssh/ssh_host_ed25519_key.pub

# 클라이언트에서 호스트 CA 신뢰
# ~/.ssh/known_hosts에 추가
@cert-authority *.company.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICAKeyFromCA...

# 6. 인증서 관리 스크립트
#!/bin/bash
# /usr/local/bin/ssh-cert-manager.sh

ACTION=$1
USER=$2
DAYS=${3:-7}

case $ACTION in
    "issue")
        if [ -z "$USER" ]; then
            echo "Usage: $0 issue <username> [days]"
            exit 1
        fi
        
        USER_KEY="/home/$USER/.ssh/id_ed25519.pub"
        if [ ! -f "$USER_KEY" ]; then
            echo "User key not found: $USER_KEY"
            exit 1
        fi
        
        ssh-keygen -s /etc/ssh/ca_key \
            -I "$USER-$(date +%Y%m%d-%H%M%S)" \
            -n "$USER" \
            -V "+${DAYS}d" \
            "$USER_KEY"
        
        echo "Certificate issued for $USER, valid for $DAYS days"
        ;;
        
    "revoke")
        # 인증서 폐기 목록 관리
        CERT_SERIAL=$(ssh-keygen -L -f "/home/$USER/.ssh/id_ed25519-cert.pub" | grep Serial | awk '{print $2}')
        echo "$CERT_SERIAL" >> /etc/ssh/revoked_keys
        echo "Certificate revoked for $USER"
        ;;
        
    "list")
        echo "Active certificates:"
        for cert in /home/*/.ssh/*-cert.pub; do
            if [ -f "$cert" ]; then
                echo "$(basename $(dirname $cert)): $(ssh-keygen -L -f $cert | grep Valid)"
            fi
        done
        ;;
        
    *)
        echo "Usage: $0 {issue|revoke|list} <username> [days]"
        exit 1
        ;;
esac
```

### 🛡️ SSH 서버 고급 보안 설정

#### 강화된 sshd_config 설정
```bash
# /etc/ssh/sshd_config - 최고 보안 수준 설정
# 기본 설정
Protocol 2
Port 2222                           # 기본 포트 변경
AddressFamily inet                  # IPv4만 사용 (필요시)
ListenAddress 192.168.1.100        # 특정 IP만 바인딩

# 암호화 및 키 교환
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,hmac-sha2-256,hmac-sha2-512
KexAlgorithms curve25519-sha256@libssh.org,ecdh-sha2-nistp521,ecdh-sha2-nistp384,ecdh-sha2-nistp256,diffie-hellman-group16-sha512

# 호스트 키 설정 (약한 키 제거)
HostKey /etc/ssh/ssh_host_ed25519_key
HostKey /etc/ssh/ssh_host_ecdsa_key
# RSA 키는 4096비트 이상만 사용
HostKey /etc/ssh/ssh_host_rsa_key

# 인증 설정
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
GSSAPIAuthentication no
HostbasedAuthentication no
IgnoreUserKnownHosts yes
PermitRootLogin no

# 로그인 제한
LoginGraceTime 30
MaxAuthTries 3
MaxSessions 3
MaxStartups 3:30:10
ClientAliveInterval 300
ClientAliveCountMax 2

# 사용자/그룹 제한
AllowUsers admin developer
AllowGroups sshusers
DenyUsers guest anonymous backup
DenyGroups wheel nogroup

# 기능 제한
X11Forwarding no
AllowTcpForwarding no
GatewayPorts no
PermitTunnel no
PermitUserEnvironment no
PermitUserRC no
PrintMotd yes
PrintLastLog yes
TCPKeepAlive no
Compression no                      # 압축 비활성화 (보안)

# 로깅
SyslogFacility AUTH
LogLevel VERBOSE                    # 상세 로깅

# Chroot 설정 (SFTP 전용 사용자)
Subsystem sftp /usr/lib/openssh/sftp-server

# 조건부 설정
Match User sftpuser
    ChrootDirectory /var/sftp/%u
    ForceCommand internal-sftp
    AllowTcpForwarding no
    X11Forwarding no
    PermitTunnel no

Match Address 192.168.1.0/24
    PasswordAuthentication yes
    MaxAuthTries 5

Match Address 10.0.0.0/8
    AllowTcpForwarding local
    GatewayPorts no

Match Group developers
    AllowTcpForwarding yes
    PermitOpen localhost:3000 localhost:8080

# 설정 검증
sshd -t
sshd -T | grep -i cipher          # 암호화 설정 확인
sshd -T | grep -i mac             # MAC 설정 확인
```

#### SSH 접근 제어 및 모니터링
```bash
# 1. TCP Wrapper 설정
# /etc/hosts.allow
sshd: 192.168.1.0/24 : ALLOW
sshd: 10.0.0.0/8 : ALLOW
sshd: ALL : spawn (/usr/local/bin/log-ssh-attempt %a %d) : DENY

# /etc/hosts.deny
sshd: ALL

# 2. SSH 접근 로깅 스크립트
#!/bin/bash
# /usr/local/bin/log-ssh-attempt
CLIENT_IP=$1
DAEMON=$2
TIMESTAMP=$(date)

echo "$TIMESTAMP: Blocked SSH attempt from $CLIENT_IP to $DAEMON" >> /var/log/ssh-blocked.log

# 지리적 위치 확인 (선택사항)
# LOCATION=$(geoiplookup $CLIENT_IP 2>/dev/null | cut -d: -f2)
# echo "$TIMESTAMP: $CLIENT_IP ($LOCATION) blocked" >> /var/log/ssh-geo.log

# 3. SSH 세션 모니터링
#!/bin/bash
# /usr/local/bin/ssh-monitor.sh

# 활성 SSH 세션 모니터링
while true; do
    ACTIVE_SESSIONS=$(who | grep -c "pts/")
    SSH_PROCESSES=$(pgrep -c sshd)
    
    if [ $ACTIVE_SESSIONS -gt 10 ]; then
        echo "$(date): High SSH session count: $ACTIVE_SESSIONS" >> /var/log/ssh-monitor.log
        # 알림 발송
        echo "High SSH session count detected: $ACTIVE_SESSIONS active sessions" | \
        mail -s "SSH Monitor Alert" admin@company.com
    fi
    
    # 장시간 유지되는 세션 확인
    who | while read user tty time rest; do
        # 12시간 이상 유지된 세션 확인
        if [[ "$time" < "$(date -d '12 hours ago' '+%H:%M')" ]]; then
            echo "$(date): Long running session: $user on $tty since $time" >> /var/log/ssh-monitor.log
        fi
    done
    
    sleep 60
done

# 4. SSH 키 로테이션 스크립트
#!/bin/bash
# /usr/local/bin/ssh-key-rotation.sh

# 90일마다 호스트 키 로테이션
HOSTKEY_AGE=$(find /etc/ssh/ssh_host_*_key -mtime +90 2>/dev/null | wc -l)

if [ $HOSTKEY_AGE -gt 0 ]; then
    echo "$(date): Host keys are older than 90 days. Rotation recommended." >> /var/log/ssh-key-rotation.log
    
    # 백업
    cp -r /etc/ssh /etc/ssh.backup.$(date +%Y%m%d)
    
    # 새 키 생성
    ssh-keygen -A
    
    # 서비스 재시작
    systemctl restart sshd
    
    echo "$(date): Host keys rotated successfully" >> /var/log/ssh-key-rotation.log
fi

# 사용자 키 만료 확인
for user_home in /home/*; do
    username=$(basename $user_home)
    auth_keys="$user_home/.ssh/authorized_keys"
    
    if [ -f "$auth_keys" ]; then
        while read -r key; do
            if [[ $key =~ ^ssh- ]]; then
                # 키 생성일 확인 (코멘트에서 날짜 추출)
                key_date=$(echo $key | grep -o '[0-9]\{8\}' | head -1)
                if [ -n "$key_date" ]; then
                    key_age=$(( ($(date +%s) - $(date -d "$key_date" +%s)) / 86400 ))
                    if [ $key_age -gt 365 ]; then
                        echo "$(date): User $username has key older than 1 year ($key_age days)" >> /var/log/ssh-key-rotation.log
                    fi
                fi
            fi
        done < "$auth_keys"
    fi
done
```

### 🔥 방화벽 보안 설정

#### iptables 고급 보안 규칙
```bash
#!/bin/bash
# /usr/local/bin/setup-iptables-advanced.sh
# 고급 iptables 보안 설정 스크립트

# 기존 규칙 초기화
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
iptables -t raw -F
iptables -t raw -X

# 기본 정책 설정 (모든 트래픽 차단)
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# 1. 기본 허용 규칙
# Loopback 인터페이스 허용
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 기존 연결 유지
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# 2. SSH 보안 규칙 (고급)
# SSH 브루트포스 방지 (복합적 접근)
iptables -N SSH_BRUTEFORCE
iptables -A SSH_BRUTEFORCE -m recent --set --name SSH_ATTACK
iptables -A SSH_BRUTEFORCE -m recent --update --seconds 3600 --hitcount 3 --name SSH_ATTACK -j LOG --log-prefix "SSH Bruteforce: "
iptables -A SSH_BRUTEFORCE -m recent --update --seconds 3600 --hitcount 3 --name SSH_ATTACK -j DROP
iptables -A SSH_BRUTEFORCE -j ACCEPT

# SSH 접근 제한
iptables -A INPUT -p tcp --dport 2222 -s 192.168.1.0/24 -j SSH_BRUTEFORCE
iptables -A INPUT -p tcp --dport 2222 -s 10.0.0.0/8 -j SSH_BRUTEFORCE
iptables -A INPUT -p tcp --dport 2222 -j LOG --log-prefix "SSH Unauthorized: "
iptables -A INPUT -p tcp --dport 2222 -j DROP

# 3. 웹 서버 보안 (있는 경우)
# HTTP/HTTPS with rate limiting
iptables -N WEB_RATE_LIMIT
iptables -A WEB_RATE_LIMIT -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A WEB_RATE_LIMIT -j LOG --log-prefix "HTTP Rate Limit: "
iptables -A WEB_RATE_LIMIT -j DROP

iptables -A INPUT -p tcp --dport 80 -j WEB_RATE_LIMIT
iptables -A INPUT -p tcp --dport 443 -j WEB_RATE_LIMIT

# 4. DDoS 방지 규칙
# SYN Flood 방지
iptables -A INPUT -p tcp --syn -m limit --limit 1/second --limit-burst 3 -j ACCEPT
iptables -A INPUT -p tcp --syn -j LOG --log-prefix "SYN Flood: "
iptables -A INPUT -p tcp --syn -j DROP

# Ping Flood 방지
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/second --limit-burst 2 -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j LOG --log-prefix "Ping Flood: "
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# Port Scan 방지
iptables -N PORT_SCAN
iptables -A PORT_SCAN -p tcp --tcp-flags SYN,ACK,FIN,RST RST -m limit --limit 1/s --limit-burst 2 -j RETURN
iptables -A PORT_SCAN -j LOG --log-prefix "Port Scan: "
iptables -A PORT_SCAN -j DROP
iptables -A INPUT -j PORT_SCAN

# 5. 아웃바운드 트래픽 제어
# DNS 허용 (필수)
iptables -A OUTPUT -p udp --dport 53 -d 8.8.8.8 -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -d 1.1.1.1 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# NTP 허용
iptables -A OUTPUT -p udp --dport 123 -j ACCEPT

# HTTP/HTTPS 아웃바운드 (업데이트용)
iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

# SMTP 아웃바운드 (이메일)
iptables -A OUTPUT -p tcp --dport 587 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 25 -j ACCEPT

# SSH 아웃바운드 (관리용)
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 2222 -j ACCEPT

# 6. 지리적 IP 차단 (GeoIP 사용)
# 특정 국가 IP 차단 (예: 러시아, 중국, 북한)
# iptables -A INPUT -m geoip --src-cc RU,CN,KP -j LOG --log-prefix "GeoIP Block: "
# iptables -A INPUT -m geoip --src-cc RU,CN,KP -j DROP

# 7. 애플리케이션별 보안 규칙
# 데이터베이스 접근 제한 (내부 네트워크만)
iptables -A INPUT -p tcp --dport 3306 -s 192.168.1.0/24 -j ACCEPT  # MySQL
iptables -A INPUT -p tcp --dport 5432 -s 192.168.1.0/24 -j ACCEPT  # PostgreSQL
iptables -A INPUT -p tcp --dport 3306 -j LOG --log-prefix "MySQL Unauthorized: "
iptables -A INPUT -p tcp --dport 3306 -j DROP
iptables -A INPUT -p tcp --dport 5432 -j LOG --log-prefix "PostgreSQL Unauthorized: "
iptables -A INPUT -p tcp --dport 5432 -j DROP

# 8. 악성 트래픽 차단
# Invalid 패킷 차단
iptables -A INPUT -m conntrack --ctstate INVALID -j LOG --log-prefix "Invalid Packet: "
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# NULL 스캔 차단
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j LOG --log-prefix "NULL Scan: "
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP

# XMAS 스캔 차단
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j LOG --log-prefix "XMAS Scan: "
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP

# 9. 로깅 설정 (최종 단계)
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "INPUT DROP: " --log-level 7
iptables -A OUTPUT -m limit --limit 5/min -j LOG --log-prefix "OUTPUT DROP: " --log-level 7

# 10. 설정 저장
iptables-save > /etc/iptables/rules.v4

# 11. 부팅 시 자동 로드 설정
cat > /etc/systemd/system/iptables-restore.service << 'EOF'
[Unit]
Description=Restore iptables rules
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables/rules.v4
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable iptables-restore.service

echo "Advanced iptables configuration completed!"
echo "Rules saved to /etc/iptables/rules.v4"
echo "Service enabled for boot: iptables-restore.service"

# 규칙 확인
iptables -L -n -v --line-numbers
```

#### UFW 고급 설정 및 자동화
```bash
#!/bin/bash
# /usr/local/bin/setup-ufw-advanced.sh
# UFW 고급 보안 설정

# UFW 초기화 및 기본 설정
ufw --force reset
ufw default deny incoming
ufw default deny outgoing
ufw default deny forward

# 기본 서비스 허용
echo "Setting up basic services..."

# SSH (보안 포트)
ufw allow from 192.168.1.0/24 to any port 2222 comment 'SSH from LAN'
ufw allow from 10.0.0.0/8 to any port 2222 comment 'SSH from VPN'

# 아웃바운드 필수 서비스
ufw allow out 53 comment 'DNS'
ufw allow out 123 comment 'NTP'
ufw allow out 80 comment 'HTTP'
ufw allow out 443 comment 'HTTPS'

# 메일 발송
ufw allow out 587 comment 'SMTP-TLS'
ufw allow out 25 comment 'SMTP'

# 웹 서버 (있는 경우)
# ufw allow 80 comment 'HTTP Server'
# ufw allow 443 comment 'HTTPS Server'

# 고급 규칙 설정
echo "Setting up advanced rules..."

# 1. 애플리케이션 프로파일 생성
cat > /etc/ufw/applications.d/custom-apps << 'EOF'
[SSH-Custom]
title=SSH Custom Port
description=SSH on custom port 2222
ports=2222/tcp

[Database-Internal]
title=Database Internal Access
description=MySQL/PostgreSQL for internal network
ports=3306,5432/tcp

[Monitoring]
title=Monitoring Services
description=Prometheus, Grafana, etc.
ports=9090,3000,9100/tcp
EOF

ufw app update SSH-Custom
ufw allow SSH-Custom

# 2. Rate limiting 설정
ufw limit ssh comment 'Rate limit SSH'
ufw limit 80/tcp comment 'Rate limit HTTP'
ufw limit 443/tcp comment 'Rate limit HTTPS'

# 3. 특정 IP 차단 (예시)
# 알려진 악성 IP 차단
MALICIOUS_IPS=(
    "192.0.2.100"
    "203.0.113.50"
)

for ip in "${MALICIOUS_IPS[@]}"; do
    ufw deny from "$ip" comment "Known malicious IP"
done

# 4. 지역별 접근 제한 (예시)
# 관리 서비스는 국내 IP만 허용
ufw allow from 220.0.0.0/8 to any port 22 comment 'Korea Telecom range'
ufw allow from 121.0.0.0/8 to any port 22 comment 'SK Broadband range'

# 5. 로깅 설정
ufw logging on

# 6. UFW 활성화
ufw --force enable

# 7. 상태 출력
ufw status verbose

# 8. UFW 로그 모니터링 스크립트 생성
cat > /usr/local/bin/ufw-log-monitor.sh << 'EOF'
#!/bin/bash
# UFW 로그 모니터링 및 분석

UFW_LOG="/var/log/ufw.log"
ALERT_EMAIL="admin@company.com"

# 로그 패턴 분석
analyze_ufw_logs() {
    echo "=== UFW Log Analysis Report $(date) ==="
    
    # 최근 1시간 동안의 차단된 연결
    echo "Top blocked IPs (last hour):"
    grep "$(date -d '1 hour ago' '+%b %d %H')" "$UFW_LOG" 2>/dev/null | \
    grep "BLOCK" | awk '{print $(NF-1)}' | cut -d= -f2 | \
    sort | uniq -c | sort -nr | head -10
    
    echo ""
    
    # 포트별 공격 통계
    echo "Top attacked ports (last hour):"
    grep "$(date -d '1 hour ago' '+%b %d %H')" "$UFW_LOG" 2>/dev/null | \
    grep "BLOCK" | grep -o "DPT=[0-9]*" | cut -d= -f2 | \
    sort | uniq -c | sort -nr | head -10
    
    echo ""
    
    # 프로토콜별 통계
    echo "Protocol statistics (last hour):"
    grep "$(date -d '1 hour ago' '+%b %d %H')" "$UFW_LOG" 2>/dev/null | \
    grep "BLOCK" | grep -o "PROTO=[A-Z]*" | cut -d= -f2 | \
    sort | uniq -c | sort -nr
    
    echo "=================================="
}

# 실시간 모니터링
monitor_realtime() {
    tail -f "$UFW_LOG" | while read line; do
        if echo "$line" | grep -q "BLOCK"; then
            src_ip=$(echo "$line" | grep -o "SRC=[0-9.]*" | cut -d= -f2)
            dst_port=$(echo "$line" | grep -o "DPT=[0-9]*" | cut -d= -f2)
            
            # 높은 빈도 공격 감지
            recent_blocks=$(grep -c "$src_ip" <(tail -100 "$UFW_LOG"))
            
            if [ "$recent_blocks" -gt 10 ]; then
                echo "$(date): High frequency attack from $src_ip (port $dst_port)"
                # 자동 차단 강화 (선택사항)
                # ufw insert 1 deny from "$src_ip" comment "Auto-blocked: high frequency"
            fi
        fi
    done
}

case "$1" in
    "analyze")
        analyze_ufw_logs
        ;;
    "monitor")
        echo "Starting real-time UFW monitoring... (Ctrl+C to stop)"
        monitor_realtime
        ;;
    "report")
        analyze_ufw_logs | mail -s "UFW Security Report" "$ALERT_EMAIL"
        ;;
    *)
        echo "Usage: $0 {analyze|monitor|report}"
        exit 1
        ;;
esac
EOF

chmod +x /usr/local/bin/ufw-log-monitor.sh

# 9. 정기적인 로그 분석 설정
cat > /etc/cron.hourly/ufw-analysis << 'EOF'
#!/bin/bash
/usr/local/bin/ufw-log-monitor.sh analyze >> /var/log/ufw-analysis.log
EOF

chmod +x /etc/cron.hourly/ufw-analysis

echo "UFW advanced configuration completed!"
echo "Monitor logs with: /usr/local/bin/ufw-log-monitor.sh monitor"
echo "View status with: ufw status verbose"
```

## 다음 편 예고

다음 포스트에서는 **SELinux/AppArmor와 시스템 하드닝**을 자세히 다룰 예정입니다:
- SELinux 정책 작성 및 관리
- AppArmor 프로파일 커스터마이징
- 커널 보안 매개변수 최적화
- 파일 시스템 보안 강화

SSH와 방화벽 보안을 완벽하게 마스터하셨나요? 🔐🛡️
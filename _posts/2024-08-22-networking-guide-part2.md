---
layout: post
title: "네트워킹 완전 가이드 2편 - IP 주소와 서브네팅 마스터하기 | Complete Network Guide Part 2 - IP Addressing & Subnetting Mastery"
date: 2024-08-22 10:00:00 +0900
categories: [Networking, IP-Addressing]
tags: [networking, ip-addressing, subnetting, vlsm, cidr, ipv4, ipv6]
---

IP 주소 체계와 서브네팅을 완전히 마스터하여 효율적인 네트워크 설계를 할 수 있도록 상세히 알아보겠습니다.

## IP 주소 체계 | IP Addressing System

### 📡 IPv4 주소 구조

IPv4 주소는 32비트로 구성되며, 8비트씩 4개의 옥텟으로 나뉩니다.

```
예시: 192.168.1.100
이진수: 11000000.10101000.00000001.01100100
```

### 🏷️ IPv4 주소 클래스

#### 클래스 A (Class A)
```
범위: 1.0.0.0 ~ 126.255.255.255
기본 서브넷 마스크: 255.0.0.0 (/8)
네트워크 비트: 8비트 (126개 네트워크)
호스트 비트: 24비트 (16,777,214개 호스트)

사용 예: 대기업, ISP
10.0.0.0/8 (사설 IP)
```

#### 클래스 B (Class B)
```
범위: 128.0.0.0 ~ 191.255.255.255
기본 서브넷 마스크: 255.255.0.0 (/16)
네트워크 비트: 16비트 (16,384개 네트워크)
호스트 비트: 16비트 (65,534개 호스트)

사용 예: 중간 규모 기업
172.16.0.0/12 (사설 IP)
```

#### 클래스 C (Class C)
```
범위: 192.0.0.0 ~ 223.255.255.255
기본 서브넷 마스크: 255.255.255.0 (/24)
네트워크 비트: 24비트 (2,097,152개 네트워크)
호스트 비트: 8비트 (254개 호스트)

사용 예: 소규모 기업, 가정
192.168.0.0/16 (사설 IP)
```

#### 클래스 D & E
```
클래스 D: 224.0.0.0 ~ 239.255.255.255 (멀티캐스트)
클래스 E: 240.0.0.0 ~ 255.255.255.255 (실험적 용도)
```

### 🔒 사설 IP 주소 (Private IP)

```bash
# RFC 1918에서 정의된 사설 IP 대역
클래스 A: 10.0.0.0/8        (10.0.0.0 ~ 10.255.255.255)
클래스 B: 172.16.0.0/12     (172.16.0.0 ~ 172.31.255.255)  
클래스 C: 192.168.0.0/16    (192.168.0.0 ~ 192.168.255.255)

# 특수 목적 IP 주소
루프백: 127.0.0.0/8         (127.0.0.1 - localhost)
링크로컬: 169.254.0.0/16     (APIPA - 자동 사설 IP)
브로드캐스트: x.x.x.255      (네트워크 내 모든 호스트)
```

## 서브네팅 (Subnetting) | Network Subnetting

### 🔧 서브네팅 기본 개념

서브네팅은 하나의 큰 네트워크를 여러 개의 작은 네트워크로 나누는 기법입니다.

#### 서브넷 마스크의 역할
```
IP 주소:        192.168.1.100
서브넷 마스크:   255.255.255.0
네트워크 부분:   192.168.1.0
호스트 부분:     0.0.0.100
```

### 📊 CIDR 표기법 (Classless Inter-Domain Routing)

```bash
# CIDR 표기법 예시
192.168.1.0/24 = 255.255.255.0
192.168.1.0/25 = 255.255.255.128
192.168.1.0/26 = 255.255.255.192
192.168.1.0/27 = 255.255.255.224
192.168.1.0/28 = 255.255.255.240
192.168.1.0/29 = 255.255.255.248
192.168.1.0/30 = 255.255.255.252

# 호스트 개수 계산 공식
호스트 개수 = 2^(32-서브넷비트수) - 2
예: /24 → 2^(32-24) - 2 = 2^8 - 2 = 254개
```

### 🧮 서브네팅 계산 실습

#### 실습 1: 192.168.1.0/24를 4개 서브넷으로 나누기

```bash
# 원본 네트워크: 192.168.1.0/24 (254개 호스트)
# 목표: 4개 서브넷 (2^2 = 4) → 2비트 차용
# 새로운 서브넷 마스크: /26 (255.255.255.192)

서브넷 1: 192.168.1.0/26    (192.168.1.1 ~ 192.168.1.62)
서브넷 2: 192.168.1.64/26   (192.168.1.65 ~ 192.168.1.126)
서브넷 3: 192.168.1.128/26  (192.168.1.129 ~ 192.168.1.190)
서브넷 4: 192.168.1.192/26  (192.168.1.193 ~ 192.168.1.254)
```

#### 실습 2: 10.0.0.0/8을 부서별로 나누기

```bash
# 본사 네트워크 설계
# IT부서: 100대 필요 → /25 (126개 호스트)
# 영업부서: 50대 필요 → /26 (62개 호스트)  
# 총무부서: 30대 필요 → /27 (30개 호스트)
# 게스트: 10대 필요 → /28 (14개 호스트)

IT부서:     10.0.1.0/25    (10.0.1.1 ~ 10.0.1.126)
영업부서:   10.0.2.0/26    (10.0.2.1 ~ 10.0.2.62)
총무부서:   10.0.3.0/27    (10.0.3.1 ~ 10.0.3.30)
게스트:     10.0.4.0/28    (10.0.4.1 ~ 10.0.4.14)
```

### 🔀 VLSM (Variable Length Subnet Masking)

효율적인 IP 주소 할당을 위한 가변 길이 서브넷 마스킹

```bash
# 네트워크 요구사항
지점 A: 50대 호스트 필요
지점 B: 25대 호스트 필요  
지점 C: 10대 호스트 필요
WAN 링크: 2대 호스트 필요 (Point-to-Point)

# VLSM 할당 (큰 서브넷부터)
지점 A: 192.168.1.0/26   (62개 호스트 - 50대 수용 가능)
지점 B: 192.168.1.64/27  (30개 호스트 - 25대 수용 가능)
지점 C: 192.168.1.96/28  (14개 호스트 - 10대 수용 가능)
WAN 1:  192.168.1.112/30 (2개 호스트)
WAN 2:  192.168.1.116/30 (2개 호스트)
```

## IPv6 주소 체계 | IPv6 Addressing

### 🌐 IPv6 주소 구조

IPv6는 128비트 주소로 구성되며, 16진수로 표현됩니다.

```bash
# IPv6 주소 형식
전체 형식: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
압축 형식: 2001:db8:85a3::8a2e:370:7334

# 주소 구조
├─────────────┬─────────────┐
│ 네트워크 (64비트) │ 인터페이스 ID (64비트) │
└─────────────┴─────────────┘

# 서브넷 구조  
├───────────┬─────┬─────────────┐
│ 글로벌(48) │서브넷(16)│ 인터페이스(64) │
└───────────┴─────┴─────────────┘
```

### 🏠 IPv6 주소 유형

```bash
# 유니캐스트 (Unicast)
글로벌: 2001::/3                # 인터넷 라우팅 가능
링크로컬: fe80::/10             # 같은 링크 내에서만
고유로컬: fc00::/7              # 사설 네트워크 (IPv4의 사설 IP와 유사)

# 멀티캐스트 (Multicast)  
멀티캐스트: ff00::/8            # 그룹 통신

# 애니캐스트 (Anycast)
글로벌 유니캐스트 범위에서 할당  # 가장 가까운 노드로 전송

# 특수 주소
루프백: ::1                     # IPv4의 127.0.0.1
언스펙: ::                      # IPv4의 0.0.0.0
IPv4 매핑: ::ffff:192.168.1.1   # IPv4-in-IPv6
```

### ⚙️ IPv6 설정 및 관리

```bash
# Linux IPv6 설정
# 임시 설정
ip -6 addr add 2001:db8::1/64 dev eth0
ip -6 route add default via 2001:db8::1

# 영구 설정 (/etc/netplan/01-netcfg.yaml)
network:
  version: 2
  ethernets:
    eth0:
      addresses:
        - 192.168.1.100/24
        - 2001:db8::100/64
      gateway4: 192.168.1.1
      gateway6: 2001:db8::1
      nameservers:
        addresses: [8.8.8.8, 2001:4860:4860::8888]

# IPv6 연결 확인
ping6 google.com
ping6 2001:4860:4860::8888
traceroute6 google.com

# IPv6 주소 확인
ip -6 addr show
ip -6 route show
```

## NAT (Network Address Translation) | 네트워크 주소 변환

### 🔄 NAT의 종류

#### Static NAT (정적 NAT)
```bash
# 1:1 매핑
사설 IP 192.168.1.10 ↔ 공인 IP 203.0.113.10

# Cisco 라우터 설정
Router(config)# ip nat inside source static 192.168.1.10 203.0.113.10
Router(config)# interface fastethernet 0/0
Router(config-if)# ip nat inside
Router(config)# interface serial 0/0  
Router(config-if)# ip nat outside
```

#### Dynamic NAT (동적 NAT)
```bash
# 풀에서 동적 할당
사설 IP Pool: 192.168.1.0/24
공인 IP Pool: 203.0.113.10 ~ 203.0.113.20

# Cisco 라우터 설정
Router(config)# access-list 1 permit 192.168.1.0 0.0.0.255
Router(config)# ip nat pool OUTSIDE 203.0.113.10 203.0.113.20 netmask 255.255.255.0
Router(config)# ip nat inside source list 1 pool OUTSIDE
```

#### PAT/NAT Overload (포트 주소 변환)
```bash
# 다대일 매핑 (포트 번호 이용)
192.168.1.10:1234 → 203.0.113.1:5000
192.168.1.20:1234 → 203.0.113.1:5001

# Cisco 라우터 설정  
Router(config)# access-list 1 permit 192.168.1.0 0.0.0.255
Router(config)# ip nat inside source list 1 interface serial 0/0 overload
```

### 🐧 Linux NAT 설정

```bash
# iptables를 이용한 NAT 설정
# IP 포워딩 활성화
echo 1 > /proc/sys/net/ipv4/ip_forward
echo 'net.ipv4.ip_forward = 1' >> /etc/sysctl.conf

# MASQUERADE (동적 IP용)
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o eth1 -m state --state RELATED,ESTABLISHED -j ACCEPT

# SNAT (정적 IP용)
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.1

# 포트 포워딩 (DNAT)
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.10:80
iptables -t nat -A PREROUTING -p tcp --dport 443 -j DNAT --to-destination 192.168.1.10:443

# 설정 저장
iptables-save > /etc/iptables/rules.v4
```

## DHCP (Dynamic Host Configuration Protocol) | 동적 주소 할당

### 📋 DHCP 동작 과정

```bash
# DHCP 4-Way Handshake
1. DHCP Discover  (Client → Server, Broadcast)
2. DHCP Offer     (Server → Client, Unicast)  
3. DHCP Request   (Client → Server, Broadcast)
4. DHCP ACK       (Server → Client, Unicast)

# 패킷 캡처로 확인
tcpdump -i eth0 port 67 or port 68 -v
```

### 🖥️ DHCP 서버 설정

#### Linux DHCP 서버 (ISC DHCP)
```bash
# 설치
sudo apt install isc-dhcp-server

# 설정 파일 (/etc/dhcp/dhcpd.conf)
option domain-name "company.local";
option domain-name-servers 8.8.8.8, 8.8.4.4;

default-lease-time 600;
max-lease-time 7200;

subnet 192.168.1.0 netmask 255.255.255.0 {
  range 192.168.1.100 192.168.1.200;
  option routers 192.168.1.1;
  option broadcast-address 192.168.1.255;
  
  # 정적 할당 (MAC 기반)
  host server1 {
    hardware ethernet 00:11:22:33:44:55;
    fixed-address 192.168.1.10;
  }
}

# 서비스 시작
sudo systemctl start isc-dhcp-server
sudo systemctl enable isc-dhcp-server

# 임대 현황 확인
cat /var/lib/dhcp/dhcpd.leases
```

#### Windows DHCP 서버
```powershell
# DHCP 서버 역할 설치
Install-WindowsFeature -Name DHCP -IncludeManagementTools

# DHCP 스코프 생성
Add-DhcpServerv4Scope -Name "LAN Scope" -StartRange 192.168.1.100 -EndRange 192.168.1.200 -SubnetMask 255.255.255.0

# 옵션 설정
Set-DhcpServerv4OptionValue -ScopeId 192.168.1.0 -OptionId 3 -Value 192.168.1.1  # 게이트웨이
Set-DhcpServerv4OptionValue -ScopeId 192.168.1.0 -OptionId 6 -Value 8.8.8.8, 8.8.4.4  # DNS

# 예약 주소 설정
Add-DhcpServerv4Reservation -ScopeId 192.168.1.0 -IPAddress 192.168.1.10 -ClientId "00-11-22-33-44-55"
```

## DNS (Domain Name System) | 도메인 네임 시스템

### 🌐 DNS 계층 구조

```
                    Root (.)
                       │
          ┌─────────────┼─────────────┐
        .com          .org         .net
          │             │            │
      google.com    wikipedia.org  example.net
          │
    www.google.com
```

### 🔍 DNS 쿼리 과정

```bash
# 반복적 쿼리 (Iterative Query)
1. 클라이언트 → 로컬 DNS: www.google.com?
2. 로컬 DNS → 루트 DNS: www.google.com?
3. 루트 DNS → 로컬 DNS: .com 네임서버 주소
4. 로컬 DNS → .com DNS: www.google.com?
5. .com DNS → 로컬 DNS: google.com 네임서버 주소
6. 로컬 DNS → google.com DNS: www.google.com?
7. google.com DNS → 로컬 DNS: 142.250.191.4
8. 로컬 DNS → 클라이언트: 142.250.191.4
```

### ⚙️ DNS 서버 설정

#### BIND9 DNS 서버 (Linux)
```bash
# 설치
sudo apt install bind9 bind9utils bind9-doc

# 주 설정 파일 (/etc/bind/named.conf.local)
zone "company.local" {
    type master;
    file "/etc/bind/db.company.local";
};

zone "1.168.192.in-addr.arpa" {
    type master;  
    file "/etc/bind/db.192.168.1";
};

# 정방향 조회 존 파일 (/etc/bind/db.company.local)
$TTL    604800
@       IN      SOA     ns1.company.local. admin.company.local. (
                              2         ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL

        IN      NS      ns1.company.local.
        IN      A       192.168.1.1

ns1     IN      A       192.168.1.1
www     IN      A       192.168.1.10
mail    IN      A       192.168.1.20
ftp     IN      A       192.168.1.30

# 역방향 조회 존 파일 (/etc/bind/db.192.168.1)
$TTL    604800
@       IN      SOA     ns1.company.local. admin.company.local. (
                              1         ; Serial
                         604800         ; Refresh
                          86400         ; Retry
                        2419200         ; Expire
                         604800 )       ; Negative Cache TTL

        IN      NS      ns1.company.local.
1       IN      PTR     company.local.
10      IN      PTR     www.company.local.
20      IN      PTR     mail.company.local.

# 서비스 재시작
sudo systemctl restart bind9
sudo systemctl enable bind9

# DNS 테스트
nslookup www.company.local localhost
dig @localhost www.company.local
dig @localhost -x 192.168.1.10
```

## 네트워크 보안 | Network Security

### 🛡️ 방화벽 구성

```bash
# iptables 기본 정책
iptables -P INPUT DROP
iptables -P FORWARD DROP  
iptables -P OUTPUT ACCEPT

# 기본 허용 규칙
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 서비스별 허용
iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT    # SSH
iptables -A INPUT -p tcp --dport 80 -j ACCEPT                      # HTTP
iptables -A INPUT -p tcp --dport 443 -j ACCEPT                     # HTTPS
iptables -A INPUT -p tcp --dport 53 -j ACCEPT                      # DNS
iptables -A INPUT -p udp --dport 53 -j ACCEPT                      # DNS

# DDoS 방어
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# 설정 저장 및 복원
iptables-save > /etc/iptables.rules
iptables-restore < /etc/iptables.rules
```

### 🔐 VPN 설정

#### OpenVPN 서버 설정
```bash
# 설치 및 설정
sudo apt install openvpn easy-rsa

# CA 인증서 생성
make-cadir ~/openvpn-ca
cd ~/openvpn-ca
./easyrsa init-pki
./easyrsa build-ca
./easyrsa gen-req server nopass
./easyrsa sign-req server server
./easyrsa gen-dh

# 서버 설정 파일 (/etc/openvpn/server.conf)
port 1194
proto udp
dev tun
ca ca.crt
cert server.crt
key server.key
dh dh.pem
server 10.8.0.0 255.255.255.0
ifconfig-pool-persist ipp.txt
push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 8.8.8.8"
keepalive 10 120
comp-lzo
user nobody
group nogroup
persist-key
persist-tun
```

## 다음 편 예고

다음 포스트에서는 **네트워크 라우팅과 스위칭**에 대해 다룰 예정입니다:
- 정적/동적 라우팅 프로토콜
- VLAN과 트렁킹
- STP와 이중화
- 로드 밸런싱

네트워크 주소 체계를 완전히 마스터하셨나요? 🌐
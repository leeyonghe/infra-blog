---
layout: post
title: "네트워킹 완전 가이드 1편 - OSI 7계층과 TCP/IP 모델 | Complete Network Guide Part 1 - OSI 7 Layers & TCP/IP Model"
date: 2024-07-15 09:00:00 +0900
categories: [Networking, Fundamentals]
tags: [networking, osi, tcp-ip, protocols, network-fundamentals, infrastructure]
---

네트워킹의 기초가 되는 OSI 7계층 모델과 TCP/IP 모델을 깊이 있게 알아보겠습니다.

## OSI 7계층 모델 | OSI 7-Layer Model

### 📚 OSI 모델 개요

OSI(Open Systems Interconnection) 모델은 네트워크 통신을 7개 계층으로 나누어 표준화한 참조 모델입니다.

### 🔗 계층별 상세 설명

#### 1️⃣ 물리 계층 (Physical Layer)
**역할**: 실제 전기적, 기계적 신호 전송
```
• 비트 스트림을 전기 신호로 변환
• 케이블, 허브, 리피터, 모뎀
• 전송 매체: 동축케이블, 광섬유, 무선
• 신호의 증폭 및 재생
```

**주요 장비**:
- **허브 (Hub)**: 모든 포트로 데이터 브로드캐스트
- **리피터**: 신호 증폭 및 재생
- **케이블**: UTP, STP, 광섬유, 동축케이블

#### 2️⃣ 데이터 링크 계층 (Data Link Layer)
**역할**: 같은 네트워크 내 노드 간 신뢰성 있는 전송
```
• 프레임 단위로 데이터 처리
• MAC 주소를 이용한 주소 지정
• 오류 검출 및 수정
• 흐름 제어 및 접근 제어
```

**주요 프로토콜**:
- **이더넷 (Ethernet)**: CSMA/CD 방식
- **Wi-Fi (802.11)**: CSMA/CA 방식
- **PPP (Point-to-Point Protocol)**

**주요 장비**:
```bash
# 스위치 설정 예시
Switch(config)# interface fastethernet 0/1
Switch(config-if)# switchport mode access
Switch(config-if)# switchport access vlan 10
Switch(config-if)# spanning-tree portfast
```

#### 3️⃣ 네트워크 계층 (Network Layer)
**역할**: 서로 다른 네트워크 간 경로 설정 및 라우팅
```
• IP 주소를 이용한 논리적 주소 지정
• 패킷 라우팅 및 경로 결정
• 서브네팅 및 VLSM
• 네트워크 간 연결성 제공
```

**주요 프로토콜**:
- **IPv4/IPv6**: 인터넷 프로토콜
- **ICMP**: 제어 메시지 프로토콜
- **ARP/RARP**: 주소 해석 프로토콜
- **라우팅 프로토콜**: RIP, OSPF, BGP

**라우팅 설정 예시**:
```bash
# 정적 라우팅
Router(config)# ip route 192.168.2.0 255.255.255.0 192.168.1.2

# OSPF 동적 라우팅
Router(config)# router ospf 1
Router(config-router)# network 192.168.1.0 0.0.0.255 area 0
Router(config-router)# network 10.0.0.0 0.0.0.255 area 0
```

#### 4️⃣ 전송 계층 (Transport Layer)
**역할**: 종단 간 신뢰성 있는 데이터 전송
```
• 포트 번호를 이용한 프로세스 식별
• 연결 지향/비연결 지향 서비스
• 오류 검출 및 복구
• 흐름 제어 및 혼잡 제어
```

**주요 프로토콜**:

**TCP (Transmission Control Protocol)**:
```
• 연결 지향형 프로토콜
• 신뢰성 보장 (순서, 중복, 오류 검사)
• 3-way handshake로 연결 설정
• 슬라이딩 윈도우 흐름 제어
• 혼잡 제어 알고리즘
```

**UDP (User Datagram Protocol)**:
```
• 비연결 지향형 프로토콜
• 빠른 전송, 오버헤드 적음
• 신뢰성 보장하지 않음
• 실시간 스트리밍에 적합
```

#### 5️⃣ 세션 계층 (Session Layer)
**역할**: 애플리케이션 간 세션 관리
```
• 세션 설정, 관리, 종료
• 동기화 체크포인트 설정
• 전이중/반이중 통신 제어
• 세션 복구 기능
```

**주요 프로토콜**:
- **NetBIOS**: 네트워크 기본 입출력 시스템
- **RPC**: 원격 프로시저 호출
- **SQL 세션**: 데이터베이스 연결

#### 6️⃣ 표현 계층 (Presentation Layer)
**역할**: 데이터 표현 방식 변환
```
• 데이터 암호화/복호화
• 데이터 압축/압축 해제
• 문자 인코딩 변환
• 바이트 순서 변환
```

**주요 기능**:
- **SSL/TLS**: 암호화 프로토콜
- **JPEG, GIF, PNG**: 이미지 압축
- **ASCII, EBCDIC**: 문자 인코딩

#### 7️⃣ 응용 계층 (Application Layer)
**역할**: 사용자와 직접 상호작용하는 애플리케이션 서비스
```
• 네트워크 서비스 제공
• 사용자 인터페이스
• 애플리케이션 프로세스
```

**주요 프로토콜**:
```bash
# HTTP/HTTPS (웹)
GET /index.html HTTP/1.1
Host: example.com

# FTP (파일 전송)
USER username
PASS password
LIST
RETR filename

# SMTP (메일 전송)
MAIL FROM: <sender@example.com>
RCPT TO: <recipient@example.com>
DATA

# DNS (도메인 해석)
nslookup example.com
dig example.com MX
```

## TCP/IP 모델 | TCP/IP Model

### 🏗️ 4계층 구조

#### 1️⃣ 네트워크 접근 계층 (Network Access Layer)
**OSI 1-2계층에 해당**
```
• 물리적 네트워크 접근
• 이더넷, Wi-Fi, PPP 등
• MAC 주소 기반 통신
```

#### 2️⃣ 인터넷 계층 (Internet Layer)
**OSI 3계층에 해당**
```
• IP 패킷 라우팅
• IPv4/IPv6, ICMP, ARP
• 논리적 주소 지정
```

#### 3️⃣ 전송 계층 (Transport Layer)
**OSI 4계층에 해당**
```
• TCP/UDP 프로토콜
• 포트 번호 기반 통신
• 종단 간 데이터 전송
```

#### 4️⃣ 응용 계층 (Application Layer)
**OSI 5-7계층에 해당**
```
• HTTP, FTP, SMTP, DNS 등
• 애플리케이션 서비스
• 사용자 인터페이스
```

## 실습: 패킷 분석 | Packet Analysis Practice

### Wireshark를 이용한 패킷 캡처

```bash
# 네트워크 인터페이스 확인
sudo tcpdump -i eth0 -n

# HTTP 트래픽 캡처
sudo tcpdump -i eth0 port 80 -A

# DNS 쿼리 캡처
sudo tcpdump -i eth0 port 53 -n
```

### 계층별 헤더 분석

```
Ethernet Header (L2):
┌─────────────────┬─────────────────┬──────────┐
│ Dest MAC (6B)   │ Src MAC (6B)    │ Type(2B) │
└─────────────────┴─────────────────┴──────────┘

IP Header (L3):
┌─────┬─────┬────────┬───────────┬─────────────┐
│Ver/L│ ToS │ Length │ ID/Flags  │ TTL/Proto   │
├─────┴─────┴────────┴───────────┴─────────────┤
│ Checksum          │ Source IP Address        │
├───────────────────┴──────────────────────────┤
│ Destination IP Address                       │
└──────────────────────────────────────────────┘

TCP Header (L4):
┌─────────────────┬─────────────────┬─────────────┐
│ Source Port     │ Dest Port       │ Seq Number  │
├─────────────────┴─────────────────┴─────────────┤
│ Ack Number                      │ Flags/Window │
└─────────────────────────────────┴─────────────┘
```

## 네트워크 문제 해결 | Network Troubleshooting

### 계층별 문제 진단

```bash
# 1계층 (물리) 문제 확인
ethtool eth0                    # 링크 상태 확인
cat /proc/net/dev              # 네트워크 통계

# 2계층 (데이터링크) 문제 확인
arp -a                         # ARP 테이블 확인
ip neighbor show               # 이웃 테이블 확인

# 3계층 (네트워크) 문제 확인
ping 8.8.8.8                   # 연결성 테스트
traceroute google.com          # 경로 추적
ip route show                  # 라우팅 테이블

# 4계층 (전송) 문제 확인
netstat -tuln                  # 포트 상태 확인
ss -tuln                       # 소켓 상태 확인
telnet google.com 80           # 포트 연결 테스트

# 7계층 (응용) 문제 확인
nslookup google.com            # DNS 해석 확인
curl -I http://google.com      # HTTP 응답 확인
```

## 보안 고려사항 | Security Considerations

### 각 계층별 보안 위협

```
계층 7 (응용): SQL Injection, XSS, DDoS
계층 6 (표현): 암호화 약점, 인증서 위조
계층 5 (세션): 세션 하이재킹, 중간자 공격
계층 4 (전송): 포트 스캔, SYN Flooding
계층 3 (네트워크): IP Spoofing, 라우팅 공격
계층 2 (데이터링크): MAC Flooding, VLAN Hopping
계층 1 (물리): 도청, 물리적 접근
```

### 보안 대책

```bash
# 방화벽 설정 (iptables)
iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j DROP

# 네트워크 모니터링
# Intrusion Detection System
sudo snort -A console -q -c /etc/snort/snort.conf -i eth0

# 네트워크 스캔 탐지
sudo nmap -sS -O target_ip
```

## 성능 최적화 | Performance Optimization

### 네트워크 튜닝

```bash
# TCP 버퍼 크기 조정
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf

# TCP 연결 타임아웃 조정
echo 'net.ipv4.tcp_fin_timeout = 15' >> /etc/sysctl.conf

# TCP 윈도우 스케일링 활성화
echo 'net.ipv4.tcp_window_scaling = 1' >> /etc/sysctl.conf

sysctl -p  # 설정 적용
```

## 다음 편 예고

다음 포스트에서는 **IP 주소 체계와 서브네팅**에 대해 자세히 다룰 예정입니다:
- IPv4/IPv6 주소 구조
- 서브넷 마스크와 CIDR
- VLSM과 서브네팅 실습
- NAT와 포트 포워딩

네트워킹의 기초를 탄탄히 다져보세요! 🚀
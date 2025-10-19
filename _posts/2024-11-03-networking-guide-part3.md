---
layout: post
title: "네트워킹 완전 가이드 3편 - 라우팅과 스위칭 심화 | Complete Network Guide Part 3 - Advanced Routing & Switching"
date: 2024-11-03 11:00:00 +0900
categories: [Networking, Routing-Switching]
tags: [routing, switching, vlan, stp, ospf, bgp, vxlan, network-protocols]
---

네트워크의 핵심인 라우팅과 스위칭 기술을 심화 학습하여 대규모 네트워크 운영 능력을 갖춰보겠습니다.

## 라우팅 프로토콜 | Routing Protocols

### 🗺️ 라우팅 기본 개념

라우팅은 패킷이 목적지까지 가는 최적의 경로를 찾는 과정입니다.

```bash
# 라우팅 테이블 구성 요소
목적지 네트워크 | 서브넷 마스크 | 게이트웨이 | 인터페이스 | 메트릭
192.168.1.0    | /24          | 직접연결    | eth0      | 0
10.0.0.0       | /8           | 192.168.1.1| eth0      | 1
0.0.0.0        | /0           | 192.168.1.1| eth0      | 1 (기본경로)
```

### 📊 라우팅 프로토콜 분류

#### Distance Vector vs Link State
```
Distance Vector (거리 벡터):
- RIP (Routing Information Protocol)
- 벨만-포드 알고리즘 사용
- 홉 카운트 기반 메트릭
- 단순하지만 수렴 속도 느림

Link State (링크 상태):  
- OSPF (Open Shortest Path First)
- 다익스트라 알고리즘 사용
- 대역폭 기반 메트릭
- 빠른 수렴, 확장성 좋음
```

#### IGP vs EGP
```
IGP (Interior Gateway Protocol):
- 같은 자율 시스템(AS) 내부
- RIP, OSPF, EIGRP

EGP (Exterior Gateway Protocol):
- 서로 다른 AS 간
- BGP (Border Gateway Protocol)
```

### 🔄 RIP (Routing Information Protocol)

```bash
# Cisco 라우터 RIP 설정
Router(config)# router rip
Router(config-router)# version 2
Router(config-router)# network 192.168.1.0
Router(config-router)# network 10.0.0.0
Router(config-router)# no auto-summary
Router(config-router)# passive-interface fastethernet 0/0

# Linux Quagga/FRR RIP 설정
router rip
 version 2
 network 192.168.1.0/24
 network 10.0.0.0/8
 redistribute connected
 redistribute static

# RIP 정보 확인
show ip route rip
show ip rip database
debug ip rip
```

#### RIP의 특징과 한계
```
장점:
- 구성이 간단함
- 소규모 네트워크에 적합
- 표준 프로토콜

단점:
- 홉 카운트 제한 (15홉)
- 느린 수렴 속도
- 대역폭 고려하지 않음
- 루프 방지 메커니즘 제한적
```

### 🌐 OSPF (Open Shortest Path First)

OSPF는 대규모 네트워크에서 가장 널리 사용되는 링크 상태 라우팅 프로토콜입니다.

#### OSPF 기본 개념
```bash
# OSPF 용어
Area: 라우터들의 논리적 그룹
ABR (Area Border Router): Area 경계 라우터
ASBR (Autonomous System Boundary Router): AS 경계 라우터  
DR (Designated Router): 대표 라우터
BDR (Backup Designated Router): 백업 대표 라우터

# LSA (Link State Advertisement) 유형
LSA Type 1: Router LSA (라우터 정보)
LSA Type 2: Network LSA (네트워크 정보)
LSA Type 3: Summary LSA (Area 간 요약)
LSA Type 4: ASBR Summary LSA
LSA Type 5: External LSA (외부 라우트)
```

#### OSPF 설정 실습
```bash
# Cisco 라우터 OSPF 설정
Router(config)# router ospf 1
Router(config-router)# router-id 1.1.1.1
Router(config-router)# network 192.168.1.0 0.0.0.255 area 0
Router(config-router)# network 10.0.0.0 0.255.255.255 area 1
Router(config-router)# area 1 stub

# 인터페이스별 OSPF 설정
Router(config)# interface fastethernet 0/0
Router(config-if)# ip ospf cost 100
Router(config-if)# ip ospf priority 255
Router(config-if)# ip ospf hello-interval 5
Router(config-if)# ip ospf dead-interval 20

# Linux FRR OSPF 설정
router ospf
 ospf router-id 1.1.1.1
 network 192.168.1.0/24 area 0.0.0.0
 network 10.0.0.0/8 area 0.0.0.1
 area 0.0.0.1 stub

# OSPF 정보 확인
show ip ospf neighbor
show ip ospf database  
show ip ospf interface
show ip route ospf
```

#### OSPF Area 설계
```bash
# Multi-Area OSPF 설계 예시
Area 0 (Backbone): 10.0.0.0/24
├── Area 1 (Sales): 10.1.0.0/16
├── Area 2 (Engineering): 10.2.0.0/16  
└── Area 3 (Guest): 10.3.0.0/16

# Area 유형
Standard Area: 모든 LSA 허용
Stub Area: Type 5 LSA 차단, 기본 경로 주입
Totally Stub: Type 3,4,5 LSA 차단
NSSA: Type 5 LSA를 Type 7로 변환
```

### 🌍 BGP (Border Gateway Protocol)

BGP는 인터넷의 라우팅을 담당하는 경로 벡터 프로토콜입니다.

#### BGP 기본 개념
```bash
# BGP 속성
AS-Path: AS 번호 경로
Next-Hop: 다음 홉 주소  
Local Preference: 로컬 우선순위 (높을수록 선호)
MED: Multi-Exit Discriminator (낮을수록 선호)
Origin: 경로 발생지 (IGP > EGP > Incomplete)
Community: 라우팅 정책 태그
```

#### BGP 설정 예시
```bash
# Cisco 라우터 BGP 설정
Router(config)# router bgp 65001
Router(config-router)# bgp router-id 1.1.1.1
Router(config-router)# neighbor 203.0.113.1 remote-as 65002
Router(config-router)# neighbor 203.0.113.1 description "ISP-A Connection"
Router(config-router)# network 192.168.0.0 mask 255.255.0.0
Router(config-router)# aggregate-address 192.168.0.0 255.255.0.0 summary-only

# BGP 경로 정책 설정
Router(config)# ip prefix-list ALLOW-CUSTOMERS seq 10 permit 192.168.0.0/16 le 24
Router(config)# route-map CUSTOMER-IN permit 10
Router(config-route-map)# match ip address prefix-list ALLOW-CUSTOMERS
Router(config-route-map)# set local-preference 200

Router(config)# router bgp 65001  
Router(config-router)# neighbor 203.0.113.1 route-map CUSTOMER-IN in

# Linux FRR BGP 설정
router bgp 65001
 bgp router-id 1.1.1.1
 neighbor 203.0.113.1 remote-as 65002
 neighbor 203.0.113.1 description ISP-A
 address-family ipv4 unicast
  network 192.168.0.0/16
  neighbor 203.0.113.1 activate
```

## 스위칭 기술 | Switching Technology

### 🔌 이더넷 스위칭 기본

#### MAC 주소 학습 과정
```bash
# 1단계: MAC 주소 테이블이 비어있음
Switch# show mac address-table
          Mac Address Table
-------------------------------------------
Vlan    Mac Address       Type        Ports
----    -----------       --------    -----

# 2단계: PC-A(00:11:22:33:44:AA)에서 PC-B로 프레임 전송
# 스위치가 포트 1에서 수신, MAC 주소 학습

# 3단계: MAC 주소 테이블 업데이트  
Switch# show mac address-table
          Mac Address Table
-------------------------------------------
Vlan    Mac Address       Type        Ports
----    -----------       --------    -----
   1    0011.2233.44aa    DYNAMIC     Fa0/1

# 4단계: 목적지 MAC 주소를 모르므로 플러딩
# 5단계: PC-B가 응답하면 포트 2에서 학습
```

#### 스위치 포트 설정
```bash
# 액세스 포트 설정 (단일 VLAN)
Switch(config)# interface fastethernet 0/1
Switch(config-if)# switchport mode access
Switch(config-if)# switchport access vlan 10
Switch(config-if)# switchport port-security
Switch(config-if)# switchport port-security maximum 2
Switch(config-if)# switchport port-security violation shutdown

# 트렁크 포트 설정 (다중 VLAN)
Switch(config)# interface fastethernet 0/24
Switch(config-if)# switchport mode trunk
Switch(config-if)# switchport trunk encapsulation dot1q
Switch(config-if)# switchport trunk allowed vlan 10,20,30
Switch(config-if)# switchport trunk native vlan 1
```

### 🏷️ VLAN (Virtual Local Area Network)

VLAN은 물리적 위치와 관계없이 논리적으로 네트워크를 분할하는 기술입니다.

#### VLAN 설정 및 관리
```bash
# VLAN 생성
Switch(config)# vlan 10
Switch(config-vlan)# name SALES
Switch(config-vlan)# vlan 20  
Switch(config-vlan)# name ENGINEERING
Switch(config-vlan)# vlan 30
Switch(config-vlan)# name GUEST

# VLAN 정보 확인
Switch# show vlan brief
VLAN Name                             Status    Ports
---- -------------------------------- --------- -------------------------------
1    default                          active    Fa0/5, Fa0/6, Fa0/7, Fa0/8
10   SALES                           active    Fa0/1, Fa0/2
20   ENGINEERING                     active    Fa0/3, Fa0/4  
30   GUEST                           active    
999  UNUSED                          active    

# 동적 VLAN 할당 (VMPS)
Switch(config)# vmps server 192.168.1.100
Switch(config)# interface range fastethernet 0/1-20
Switch(config-if-range)# switchport mode dynamic desirable
```

#### Inter-VLAN 라우팅
```bash
# 라우터의 서브인터페이스 설정 (Router-on-a-Stick)
Router(config)# interface fastethernet 0/0
Router(config-if)# no shutdown
Router(config-if)# interface fastethernet 0/0.10
Router(config-subif)# encapsulation dot1Q 10
Router(config-subif)# ip address 192.168.10.1 255.255.255.0
Router(config-subif)# interface fastethernet 0/0.20
Router(config-subif)# encapsulation dot1Q 20  
Router(config-subif)# ip address 192.168.20.1 255.255.255.0

# SVI (Switched Virtual Interface) 설정
Switch(config)# ip routing
Switch(config)# interface vlan 10
Switch(config-if)# ip address 192.168.10.1 255.255.255.0
Switch(config-if)# no shutdown
Switch(config-if)# interface vlan 20
Switch(config-if)# ip address 192.168.20.1 255.255.255.0
Switch(config-if)# no shutdown
```

### 🌳 STP (Spanning Tree Protocol)

STP는 스위치 네트워크에서 루프를 방지하고 이중화를 제공하는 프로토콜입니다.

#### STP 기본 개념
```bash
# STP 포트 상태
Disabled: 포트 비활성화
Blocking: 데이터 전송 차단, BPDU 수신만
Listening: BPDU 송수신, MAC 주소 학습하지 않음
Learning: MAC 주소 학습, 데이터 전송하지 않음  
Forwarding: 정상 데이터 전송

# STP 포트 역할
Root Port: 루트 브리지로 가는 최단 경로
Designated Port: 세그먼트의 대표 포트
Alternate Port: 루트 포트의 백업
Backup Port: 같은 스위치의 다른 포트 백업
```

#### STP 설정 및 최적화
```bash
# 루트 브리지 설정
Switch(config)# spanning-tree vlan 1 root primary
Switch(config)# spanning-tree vlan 1 priority 4096

# 포트 우선순위 및 비용 설정
Switch(config)# interface fastethernet 0/1  
Switch(config-if)# spanning-tree vlan 1 port-priority 128
Switch(config-if)# spanning-tree vlan 1 cost 19

# RSTP (Rapid Spanning Tree) 설정
Switch(config)# spanning-tree mode rapid-pvst

# 포트 최적화
Switch(config)# interface range fastethernet 0/1-20
Switch(config-if-range)# spanning-tree portfast
Switch(config-if-range)# spanning-tree bpduguard enable

# STP 정보 확인
Switch# show spanning-tree
Switch# show spanning-tree vlan 1
Switch# show spanning-tree interface fastethernet 0/1
```

#### MST (Multiple Spanning Tree)
```bash
# MST 설정
Switch(config)# spanning-tree mode mst
Switch(config)# spanning-tree mst configuration
Switch(config-mst)# name COMPANY  
Switch(config-mst)# revision 1
Switch(config-mst)# instance 1 vlan 10,20
Switch(config-mst)# instance 2 vlan 30,40
Switch(config-mst)# exit

# MST 루트 설정
Switch(config)# spanning-tree mst 1 root primary
Switch(config)# spanning-tree mst 2 root secondary
```

## 고급 네트워킹 기술 | Advanced Networking

### 🔗 Link Aggregation (EtherChannel)

여러 물리적 링크를 논리적으로 묶어 대역폭을 증가시키고 이중화를 제공합니다.

```bash
# LACP (Link Aggregation Control Protocol) 설정
Switch(config)# interface range fastethernet 0/1-2
Switch(config-if-range)# channel-group 1 mode active
Switch(config-if-range)# exit
Switch(config)# interface port-channel 1
Switch(config-if)# switchport mode trunk
Switch(config-if)# switchport trunk allowed vlan 10,20,30

# PAgP (Port Aggregation Protocol) 설정  
Switch(config)# interface range fastethernet 0/3-4
Switch(config-if-range)# channel-group 2 mode desirable

# 정적 EtherChannel 설정
Switch(config)# interface range fastethernet 0/5-6  
Switch(config-if-range)# channel-group 3 mode on

# EtherChannel 확인
Switch# show etherchannel summary
Switch# show etherchannel port-channel
```

### 🌐 VXLAN (Virtual Extensible LAN)

클라우드 환경에서 L2 오버레이 네트워크를 구현하는 기술입니다.

```bash
# Linux에서 VXLAN 설정
# VXLAN 인터페이스 생성
ip link add vxlan10 type vxlan id 10 remote 192.168.1.2 local 192.168.1.1 dev eth0 dstport 4789

# VXLAN을 브리지에 연결
ip link add br0 type bridge
ip link set vxlan10 master br0
ip link set eth1 master br0

# 인터페이스 활성화
ip link set vxlan10 up
ip link set br0 up

# 멀티캐스트 VXLAN
ip link add vxlan20 type vxlan id 20 group 239.1.1.1 dev eth0 dstport 4789

# VXLAN 정보 확인
bridge fdb show dev vxlan10
ip -d link show vxlan10
```

### ⚖️ 로드 밸런싱

#### HAProxy 설정
```bash
# HAProxy 설정 파일 (/etc/haproxy/haproxy.cfg)
global
    daemon
    maxconn 4096
    log 127.0.0.1:514 local0

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

frontend web_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/website.pem
    redirect scheme https if !{ ssl_fc }
    default_backend web_servers

backend web_servers
    balance roundrobin
    option httpchk GET /health
    server web1 192.168.1.10:80 check
    server web2 192.168.1.11:80 check
    server web3 192.168.1.12:80 check backup

# 통계 페이지
listen stats
    bind *:8080
    stats enable
    stats uri /stats
    stats refresh 30s
```

#### NGINX 로드 밸런싱
```nginx
# /etc/nginx/nginx.conf
upstream backend {
    least_conn;  # 로드 밸런싱 방법
    server 192.168.1.10:80 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:80 max_fails=3 fail_timeout=30s;
    server 192.168.1.12:80 backup;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 헬스 체크
        proxy_connect_timeout 1s;
        proxy_send_timeout 1s;
        proxy_read_timeout 1s;
    }
}
```

### 🔐 네트워크 보안 심화

#### 포트 보안 (Port Security)
```bash
# MAC 주소 기반 포트 보안
Switch(config)# interface fastethernet 0/1
Switch(config-if)# switchport port-security
Switch(config-if)# switchport port-security maximum 2
Switch(config-if)# switchport port-security mac-address sticky
Switch(config-if)# switchport port-security violation restrict

# 802.1X 인증 설정
Switch(config)# aaa new-model
Switch(config)# aaa authentication dot1x default group radius
Switch(config)# dot1x system-auth-control
Switch(config)# interface fastethernet 0/1
Switch(config-if)# authentication port-control auto
Switch(config-if)# dot1x pae authenticator
```

#### ACL (Access Control List) 심화
```bash
# 확장 ACL
Router(config)# ip access-list extended BLOCK_SOCIAL
Router(config-ext-nacl)# deny tcp any host 23.35.67.140 eq 80
Router(config-ext-nacl)# deny tcp any host 23.35.67.141 eq 443  
Router(config-ext-nacl)# permit ip any any
Router(config-ext-nacl)# exit
Router(config)# interface fastethernet 0/1
Router(config-if)# ip access-group BLOCK_SOCIAL out

# 시간 기반 ACL
Router(config)# time-range WORK_HOURS
Router(config-time-range)# periodic weekdays 09:00 to 18:00
Router(config)# ip access-list extended TIME_BASED
Router(config-ext-nacl)# permit tcp 192.168.1.0 0.0.0.255 any eq 80 time-range WORK_HOURS
Router(config-ext-nacl)# deny tcp 192.168.1.0 0.0.0.255 any eq 80

# 반사형 ACL (Reflexive ACL)
Router(config)# ip access-list extended OUTBOUND
Router(config-ext-nacl)# permit tcp 192.168.1.0 0.0.0.255 any reflect TCP_TRAFFIC
Router(config-ext-nacl)# permit icmp 192.168.1.0 0.0.0.255 any reflect ICMP_TRAFFIC

Router(config)# ip access-list extended INBOUND  
Router(config-ext-nacl)# evaluate TCP_TRAFFIC
Router(config-ext-nacl)# evaluate ICMP_TRAFFIC
Router(config-ext-nacl)# deny ip any any
```

## 네트워크 모니터링 및 문제 해결 | Network Monitoring & Troubleshooting

### 📊 SNMP (Simple Network Management Protocol)

```bash
# SNMP v3 설정 (Cisco)
Router(config)# snmp-server view READONLY iso included
Router(config)# snmp-server group ADMIN v3 auth read READONLY  
Router(config)# snmp-server user admin ADMIN v3 auth sha password123 priv aes 128 password456
Router(config)# snmp-server host 192.168.1.100 version 3 auth admin

# Linux SNMP 클라이언트
# OID를 이용한 정보 조회
snmpwalk -v3 -u admin -a SHA -A password123 -x AES -X password456 -l authPriv 192.168.1.1 1.3.6.1.2.1.1

# 인터페이스 통계 조회
snmpwalk -v3 -u admin -a SHA -A password123 -x AES -X password456 -l authPriv 192.168.1.1 1.3.6.1.2.1.2.2.1.10
```

### 🔍 네트워크 분석 도구

#### Wireshark 고급 필터
```bash
# 프로토콜별 필터
tcp.port == 80                    # HTTP 트래픽
tcp.flags.syn == 1 and tcp.flags.ack == 0  # TCP SYN 패킷
icmp.type == 8                    # ICMP Echo Request
dns.qry.name contains "google"    # DNS 쿼리

# 네트워크별 필터
ip.src == 192.168.1.0/24          # 소스 네트워크
ip.dst == 10.0.0.0/8             # 목적지 네트워크
eth.addr == 00:11:22:33:44:55     # MAC 주소

# 성능 분석
tcp.analysis.retransmission       # TCP 재전송
tcp.analysis.duplicate_ack        # 중복 ACK
tcp.analysis.zero_window          # 제로 윈도우
```

#### 네트워크 성능 측정
```bash
# 대역폭 측정 (iperf3)
# 서버 모드
iperf3 -s -p 5201

# 클라이언트 모드  
iperf3 -c 192.168.1.100 -p 5201 -t 60 -P 4

# UDP 측정
iperf3 -c 192.168.1.100 -u -b 100M

# 지연 시간 측정 (hping3)
hping3 -S -p 80 -c 10 google.com
hping3 -1 -c 100 -i u1000 192.168.1.1  # 마이크로초 간격

# MTU 경로 발견
tracepath google.com
ping -M do -s 1472 google.com
```

## 다음 편 예고

다음 포스트에서는 **무선 네트워킹과 최신 기술**을 다룰 예정입니다:
- Wi-Fi 6/6E 기술
- SD-WAN과 네트워크 가상화
- 5G와 엣지 컴퓨팅
- 네트워크 자동화 (Ansible, Python)

라우팅과 스위칭 기술을 완전히 마스터하셨나요? 🚀
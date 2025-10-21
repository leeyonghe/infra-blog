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

## 2-3 IaC (Infrastructure as Code) | 코드형 인프라

### 2-3-1 IaC 종류 | Types of IaC

IaC는 인프라를 코드로 정의하고 관리하는 방법론으로, 네트워크 인프라 구축과 관리를 자동화합니다.

#### 🛠️ 주요 IaC 도구 분류

```bash
# 선언적 vs 명령적
선언적 (Declarative):
- Terraform, CloudFormation, Ansible (일부)
- 최종 상태를 정의
- 더 안정적이고 예측 가능

명령적 (Imperative):  
- Shell Scripts, Python Scripts
- 실행 단계를 정의
- 더 유연하지만 복잡함

# 에이전트 기반 vs 에이전트리스
에이전트 기반:
- Puppet, Chef
- 대상 시스템에 에이전트 설치 필요
- 지속적인 상태 관리

에이전트리스:
- Ansible, Terraform
- SSH/WinRM 등을 통한 원격 실행
- 설치 부담 없음
```

#### Terraform
```hcl
# Provider 설정
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    cisco = {
      source  = "CiscoDevNet/aci"
      version = "~> 2.0"
    }
  }
}

# 변수 정의
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}
```

#### Ansible
```yaml
# 네트워크 장비 인벤토리 (/etc/ansible/hosts)
[switches]
switch1 ansible_host=192.168.1.10 ansible_network_os=ios
switch2 ansible_host=192.168.1.11 ansible_network_os=ios

[routers]  
router1 ansible_host=192.168.1.1 ansible_network_os=ios
router2 ansible_host=192.168.1.2 ansible_network_os=ios

[network:children]
switches
routers

[network:vars]
ansible_user=admin
ansible_password=password
ansible_connection=network_cli
ansible_become=yes
ansible_become_method=enable
```

#### CloudFormation (AWS)
```yaml
# AWS 네트워크 인프라 템플릿
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Network Infrastructure Template'

Parameters:
  EnvironmentName:
    Description: Environment name prefix
    Type: String
    Default: Production

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-VPC
```

#### Pulumi
```python
# Python을 이용한 IaC
import pulumi
import pulumi_aws as aws

# VPC 생성
vpc = aws.ec2.Vpc("main-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={
        "Name": "main-vpc",
        "Environment": "production"
    }
)

# 서브넷 생성
public_subnet = aws.ec2.Subnet("public-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone="us-west-2a",
    map_public_ip_on_launch=True,
    tags={
        "Name": "public-subnet",
        "Type": "Public"
    }
)
```

#### CDK (Cloud Development Kit)
```typescript
// TypeScript를 이용한 AWS CDK
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export class NetworkStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // VPC 생성
    const vpc = new ec2.Vpc(this, 'MainVpc', {
      maxAzs: 3,
      cidr: '10.0.0.0/16',
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        }
      ]
    });
  }
}
```

### 2-3-2 테라폼으로 환경 구성 | Environment Setup with Terraform

Terraform을 사용하여 완전한 네트워크 환경을 구성해보겠습니다.

#### 🏗️ 프로젝트 구조 설계

```bash
# 디렉토리 구조
terraform-network/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── production/
├── modules/
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── security-groups/
│   └── load-balancer/
├── global/
│   └── iam/
└── shared/
    └── data.tf
```

#### VPC 모듈 구현
```hcl
# modules/vpc/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = var.enable_dns_hostnames
  enable_dns_support   = var.enable_dns_support

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-vpc"
  })
}

# 인터넷 게이트웨이
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-igw"
  })
}

# 가용 영역별 퍼블릭 서브넷
resource "aws_subnet" "public" {
  count = length(var.public_subnets)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnets[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-public-${count.index + 1}"
    Type = "Public"
  })
}

# 프라이빗 서브넷
resource "aws_subnet" "private" {
  count = length(var.private_subnets)

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnets[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-private-${count.index + 1}"
    Type = "Private"
  })
}

# NAT 게이트웨이를 위한 EIP
resource "aws_eip" "nat" {
  count = var.enable_nat_gateway ? length(var.public_subnets) : 0

  domain = "vpc"
  depends_on = [aws_internet_gateway.main]

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-nat-eip-${count.index + 1}"
  })
}

# NAT 게이트웨이
resource "aws_nat_gateway" "main" {
  count = var.enable_nat_gateway ? length(var.public_subnets) : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-nat-gw-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

# 퍼블릭 라우트 테이블
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-public-rt"
  })
}

# 프라이빗 라우트 테이블
resource "aws_route_table" "private" {
  count = var.enable_nat_gateway ? length(var.private_subnets) : 1

  vpc_id = aws_vpc.main.id

  dynamic "route" {
    for_each = var.enable_nat_gateway ? [1] : []
    content {
      cidr_block     = "0.0.0.0/0"
      nat_gateway_id = aws_nat_gateway.main[count.index].id
    }
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-private-rt-${count.index + 1}"
  })
}

# 라우트 테이블 연결
resource "aws_route_table_association" "public" {
  count = length(var.public_subnets)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(var.private_subnets)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = var.enable_nat_gateway ? aws_route_table.private[count.index].id : aws_route_table.private[0].id
}
```

#### VPC 모듈 변수
```hcl
# modules/vpc/variables.tf
variable "name_prefix" {
  description = "Name prefix for all resources"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "public_subnets" {
  description = "List of public subnet CIDR blocks"
  type        = list(string)
}

variable "private_subnets" {
  description = "List of private subnet CIDR blocks"
  type        = list(string)
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC"
  type        = bool
  default     = true
}

variable "enable_dns_support" {
  description = "Enable DNS support in the VPC"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
```

#### VPC 모듈 출력
```hcl
# modules/vpc/outputs.tf
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].id
}
```

#### 보안 그룹 모듈
```hcl
# modules/security-groups/main.tf
# 웹 서버 보안 그룹
resource "aws_security_group" "web" {
  name_prefix = "${var.name_prefix}-web-"
  vpc_id      = var.vpc_id
  description = "Security group for web servers"

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description     = "SSH from management"
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.management.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-web-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# 데이터베이스 보안 그룹
resource "aws_security_group" "database" {
  name_prefix = "${var.name_prefix}-db-"
  vpc_id      = var.vpc_id
  description = "Security group for database servers"

  ingress {
    description     = "MySQL/MariaDB"
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }

  ingress {
    description     = "PostgreSQL"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-db-sg"
  })
}

# 관리 보안 그룹
resource "aws_security_group" "management" {
  name_prefix = "${var.name_prefix}-mgmt-"
  vpc_id      = var.vpc_id
  description = "Security group for management access"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.management_cidr_blocks
  }

  ingress {
    description = "RDP"
    from_port   = 3389
    to_port     = 3389
    protocol    = "tcp"
    cidr_blocks = var.management_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-mgmt-sg"
  })
}
```

#### 메인 환경 설정
```hcl
# environments/production/main.tf
terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket         = "company-terraform-state"
    key            = "network/production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment   = var.environment
      Project       = var.project_name
      ManagedBy     = "Terraform"
      CreatedBy     = var.created_by
      CostCenter    = var.cost_center
    }
  }
}

# 데이터 소스
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC 모듈 호출
module "vpc" {
  source = "../../modules/vpc"

  name_prefix        = var.name_prefix
  cidr_block         = var.vpc_cidr
  public_subnets     = var.public_subnet_cidrs
  private_subnets    = var.private_subnet_cidrs
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
  enable_nat_gateway = var.enable_nat_gateway

  tags = local.common_tags
}

# 보안 그룹 모듈 호출
module "security_groups" {
  source = "../../modules/security-groups"

  name_prefix             = var.name_prefix
  vpc_id                  = module.vpc.vpc_id
  management_cidr_blocks  = var.management_cidr_blocks

  tags = local.common_tags
}

# 로컬 값
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
  }
}
```

#### 환경별 변수 파일
```hcl
# environments/production/variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "name_prefix" {
  description = "Name prefix for resources"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway"
  type        = bool
  default     = true
}

variable "management_cidr_blocks" {
  description = "CIDR blocks allowed for management access"
  type        = list(string)
}

variable "created_by" {
  description = "Creator of the infrastructure"
  type        = string
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
}
```

#### Terraform 실행 값
```hcl
# environments/production/terraform.tfvars
project_name = "company-web-app"
name_prefix  = "prod-webapp"

vpc_cidr = "10.0.0.0/16"

public_subnet_cidrs = [
  "10.0.1.0/24",
  "10.0.2.0/24", 
  "10.0.3.0/24"
]

private_subnet_cidrs = [
  "10.0.11.0/24",
  "10.0.12.0/24",
  "10.0.13.0/24"
]

enable_nat_gateway = true

management_cidr_blocks = [
  "203.0.113.0/24",  # 회사 사무실
  "198.51.100.0/24"  # VPN 네트워크
]

created_by   = "devops-team"
cost_center  = "engineering"
```

#### Terraform 실행 및 관리
```bash
# 초기화
terraform init

# 계획 확인  
terraform plan -var-file="terraform.tfvars"

# 적용
terraform apply -var-file="terraform.tfvars"

# 상태 확인
terraform show
terraform state list

# 리소스 확인
terraform state show module.vpc.aws_vpc.main

# 출력 값 확인
terraform output

# 리소스 삭제
terraform destroy -var-file="terraform.tfvars"

# 워크스페이스 관리
terraform workspace new staging
terraform workspace select production
terraform workspace list
```

#### 고급 Terraform 기능
```hcl
# 조건부 리소스 생성
resource "aws_instance" "web" {
  count = var.environment == "production" ? 3 : 1
  
  ami           = data.aws_ami.amazon_linux.id
  instance_type = var.environment == "production" ? "t3.medium" : "t3.micro"
  
  tags = {
    Name = "${var.name_prefix}-web-${count.index + 1}"
  }
}

# 동적 블록
resource "aws_security_group" "web" {
  name_prefix = "${var.name_prefix}-web-"
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}

# 데이터 소스 활용
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# 원격 상태 참조
data "terraform_remote_state" "vpc" {
  backend = "s3"
  config = {
    bucket = "company-terraform-state"
    key    = "network/production/terraform.tfstate"
    region = "us-west-2"
  }
}
```

## 2-4 단일 장애 지점 제거 | Eliminating Single Points of Failure

### 2-4-1 SPOF 찾기 | Identifying SPOF

단일 장애 지점(SPOF)은 시스템의 한 구성 요소가 실패할 때 전체 시스템이 작동을 멈추게 되는 지점을 말합니다. 네트워크 인프라에서 SPOF를 식별하고 제거하는 것은 고가용성 확보의 핵심입니다.

#### 🔍 SPOF 식별 방법론

```bash
# SPOF 분석 체크리스트
1. 네트워크 토폴로지 분석
   - 단일 연결점 확인
   - 백업 경로 존재 여부
   - 대역폭 병목 지점

2. 하드웨어 구성 요소 점검
   - 단일 스위치/라우터 의존성
   - 전원 공급 장치
   - 케이블링 경로

3. 서비스 의존성 분석
   - DNS 서버
   - DHCP 서버  
   - 인증 서버 (RADIUS/LDAP)

4. 외부 연결 검토
   - ISP 연결
   - WAN 회선
   - 클라우드 연결
```

#### 네트워크 토폴로지 SPOF 분석

```bash
# 현재 네트워크 상태 확인
# Linux 환경에서 네트워크 경로 분석
ip route show table main
traceroute -n 8.8.8.8
mtr --report --report-cycles=100 google.com

# 라우팅 테이블 분석
netstat -rn
route -n

# 인터페이스 상태 확인
ip link show
ip addr show
ethtool eth0  # 링크 상태, 속도 확인

# Cisco 장비에서 SPOF 분석
show ip route
show interface summary  
show spanning-tree root
show etherchannel summary
show redundancy

# 대역폭 사용률 모니터링
show interface fastethernet 0/1 | include rate
show processes cpu sorted
show memory summary
```

#### 일반적인 네트워크 SPOF 패턴

```bash
# 1. 단일 업링크 (Single Uplink)
문제: 하나의 업스트림 연결만 존재
영향: 해당 연결 실패 시 전체 네트워크 단절
해결: 이중 업링크 + 로드밸런싱

# 2. 단일 코어 스위치 (Single Core Switch)  
문제: 모든 액세스 스위치가 하나의 코어에 연결
영향: 코어 스위치 실패 시 전체 네트워크 마비
해결: 이중 코어 스위치 + HSRP/VRRP

# 3. 단일 VLAN (Single VLAN)
문제: 브로드캐스트 도메인 과부하
영향: 네트워크 성능 저하, 장애 전파
해결: VLAN 분할 + Inter-VLAN 라우팅

# 4. 단일 ISP (Single ISP)
문제: 하나의 인터넷 서비스 제공업체
영향: ISP 장애 시 인터넷 연결 불가
해결: 멀티 ISP + BGP 라우팅
```

#### SPOF 탐지 도구와 스크립트

```python
#!/usr/bin/env python3
# spof_detector.py - 네트워크 SPOF 탐지 도구

import subprocess
import json
import sys
from collections import defaultdict
import networkx as nx

class SPOFDetector:
    def __init__(self):
        self.network_graph = nx.Graph()
        self.devices = {}
        self.links = {}
        
    def scan_network_topology(self, network_range):
        """네트워크 토폴로지 스캔"""
        print(f"Scanning network range: {network_range}")
        
        # SNMP을 통한 네트워크 디스커버리
        devices = self.discover_devices(network_range)
        
        for device in devices:
            self.analyze_device(device)
            
        return self.identify_spof()
    
    def discover_devices(self, network_range):
        """SNMP를 통한 네트워크 장비 발견"""
        cmd = f"nmap -sn {network_range}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        devices = []
        for line in result.stdout.split('\n'):
            if 'Nmap scan report' in line:
                ip = line.split()[-1]
                devices.append(ip)
        
        return devices
    
    def analyze_device(self, device_ip):
        """개별 장비 분석"""
        device_info = {
            'ip': device_ip,
            'type': self.detect_device_type(device_ip),
            'interfaces': self.get_interfaces(device_ip),
            'neighbors': self.get_neighbors(device_ip),
            'redundancy': self.check_redundancy(device_ip)
        }
        
        self.devices[device_ip] = device_info
        self.network_graph.add_node(device_ip, **device_info)
        
    def detect_device_type(self, device_ip):
        """장비 유형 감지 (라우터/스위치/방화벽)"""
        # SNMP OID를 통한 장비 유형 확인
        snmp_cmd = f"snmpget -v2c -c public {device_ip} 1.3.6.1.2.1.1.1.0"
        result = subprocess.run(snmp_cmd.split(), capture_output=True, text=True)
        
        if 'cisco' in result.stdout.lower():
            if 'router' in result.stdout.lower():
                return 'router'
            elif 'switch' in result.stdout.lower():
                return 'switch'
        
        return 'unknown'
    
    def get_interfaces(self, device_ip):
        """인터페이스 정보 수집"""
        # SNMP를 통한 인터페이스 상태 확인
        interfaces = {}
        
        # 인터페이스 이름 (1.3.6.1.2.1.2.2.1.2)
        # 인터페이스 상태 (1.3.6.1.2.1.2.2.1.8)
        # 인터페이스 속도 (1.3.6.1.2.1.2.2.1.5)
        
        return interfaces
    
    def get_neighbors(self, device_ip):
        """인접 장비 정보 수집 (CDP/LLDP)"""
        neighbors = []
        
        # CDP 정보 수집 (1.3.6.1.4.1.9.9.23.1.2.1.1.6)
        # LLDP 정보 수집 (1.0.8802.1.1.2.1.4.1.1.9)
        
        return neighbors
    
    def check_redundancy(self, device_ip):
        """이중화 구성 확인"""
        redundancy_info = {
            'power_supplies': self.check_power_supplies(device_ip),
            'uplinks': self.check_uplinks(device_ip),
            'protocols': self.check_redundancy_protocols(device_ip)
        }
        
        return redundancy_info
    
    def identify_spof(self):
        """SPOF 식별"""
        spof_list = []
        
        # 1. 연결성 분석 - 단일 연결점 찾기
        for node in self.network_graph.nodes():
            if self.network_graph.degree(node) == 1:
                spof_list.append({
                    'type': 'single_connection',
                    'device': node,
                    'risk': 'high',
                    'description': f'Device {node} has only one connection'
                })
        
        # 2. 브리지 포인트 분석
        bridges = list(nx.bridges(self.network_graph))
        for bridge in bridges:
            spof_list.append({
                'type': 'bridge_link',
                'link': bridge,
                'risk': 'critical',
                'description': f'Link between {bridge[0]} and {bridge[1]} is critical'
            })
        
        # 3. 절단점 분석 (Articulation Points)
        articulation_points = list(nx.articulation_points(self.network_graph))
        for point in articulation_points:
            spof_list.append({
                'type': 'articulation_point',
                'device': point,
                'risk': 'critical',
                'description': f'Device {point} is a critical junction point'
            })
        
        return spof_list
    
    def generate_report(self, spof_list):
        """SPOF 분석 보고서 생성"""
        report = {
            'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            'total_devices': len(self.devices),
            'total_spof_issues': len(spof_list),
            'critical_issues': len([s for s in spof_list if s['risk'] == 'critical']),
            'high_issues': len([s for s in spof_list if s['risk'] == 'high']),
            'spof_details': spof_list,
            'recommendations': self.generate_recommendations(spof_list)
        }
        
        return report

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 spof_detector.py <network_range>")
        print("Example: python3 spof_detector.py 192.168.1.0/24")
        sys.exit(1)
    
    network_range = sys.argv[1]
    detector = SPOFDetector()
    
    print("Starting SPOF detection...")
    spof_list = detector.scan_network_topology(network_range)
    
    report = detector.generate_report(spof_list)
    
    # JSON 형태로 결과 출력
    print(json.dumps(report, indent=2))
    
    # 요약 출력
    print(f"\n=== SPOF Detection Summary ===")
    print(f"Total devices scanned: {report['total_devices']}")
    print(f"SPOF issues found: {report['total_spof_issues']}")
    print(f"Critical issues: {report['critical_issues']}")
    print(f"High risk issues: {report['high_issues']}")

if __name__ == "__main__":
    main()
```

#### Ansible을 이용한 SPOF 점검 자동화

```yaml
---
# spof_check.yml - SPOF 점검 플레이북
- name: Network SPOF Assessment
  hosts: network_devices
  gather_facts: no
  vars:
    spof_report: []
    
  tasks:
    - name: Check device redundancy status
      ios_command:
        commands:
          - show redundancy
          - show environment power
          - show interface summary
          - show spanning-tree root
          - show etherchannel summary
      register: device_status
      
    - name: Analyze redundancy configuration
      set_fact:
        redundancy_analysis: "{{ device_status.stdout | analyze_redundancy }}"
        
    - name: Check for single uplinks
      ios_command:
        commands:
          - show cdp neighbors
          - show interface trunk
      register: uplink_status
      
    - name: Identify SPOF risks
      set_fact:
        spof_risks: "{{ spof_risks | default([]) + [item] }}"
      loop:
        - "{{ 'Single uplink detected' if uplink_count == 1 else 'Multiple uplinks OK' }}"
        - "{{ 'No redundant power' if power_supplies < 2 else 'Redundant power OK' }}"
        - "{{ 'No EtherChannel' if etherchannel_count == 0 else 'EtherChannel configured' }}"
      vars:
        uplink_count: "{{ uplink_status.stdout[1] | regex_findall('trunking') | length }}"
        power_supplies: "{{ device_status.stdout[1] | regex_findall('PS[0-9]+') | length }}"
        etherchannel_count: "{{ device_status.stdout[4] | regex_findall('Po[0-9]+') | length }}"
        
    - name: Generate SPOF report
      template:
        src: spof_report.j2
        dest: "/tmp/spof_report_{{ inventory_hostname }}.html"
      vars:
        device_name: "{{ inventory_hostname }}"
        check_date: "{{ ansible_date_time.iso8601 }}"
        
    - name: Send SPOF alert if critical issues found
      mail:
        to: "{{ network_admin_email }}"
        subject: "CRITICAL: SPOF detected on {{ inventory_hostname }}"
        body: "Critical single points of failure detected. See attached report."
        attach: "/tmp/spof_report_{{ inventory_hostname }}.html"
      when: "'Single uplink detected' in spof_risks or 'No redundant power' in spof_risks"
```

#### 네트워크 토폴로지 시각화

```bash
# Graphviz를 이용한 네트워크 토폴로지 생성
#!/bin/bash
# generate_network_topology.sh

# SNMP를 통한 네트워크 맵 생성
generate_dot_file() {
    echo "digraph network {" > network.dot
    echo "  rankdir=TB;" >> network.dot
    echo "  node [shape=box];" >> network.dot
    
    # 각 장비 정보 수집
    for ip in $(nmap -sn 192.168.1.0/24 | grep "Nmap scan report" | awk '{print $5}'); do
        device_name=$(snmpget -v2c -c public $ip 1.3.6.1.2.1.1.5.0 2>/dev/null | cut -d'"' -f2)
        if [ ! -z "$device_name" ]; then
            echo "  \"$device_name\" [label=\"$device_name\\n$ip\"];" >> network.dot
            
            # CDP 정보로 연결 관계 확인
            cdp_neighbors=$(snmpwalk -v2c -c public $ip 1.3.6.1.4.1.9.9.23.1.2.1.1.6 2>/dev/null)
            while read -r neighbor; do
                neighbor_name=$(echo $neighbor | cut -d'"' -f2)
                if [ ! -z "$neighbor_name" ]; then
                    echo "  \"$device_name\" -> \"$neighbor_name\";" >> network.dot
                fi
            done <<< "$cdp_neighbors"
        fi
    done
    
    echo "}" >> network.dot
}

# DOT 파일을 이미지로 변환
generate_dot_file
dot -Tpng network.dot -o network_topology.png
dot -Tsvg network.dot -o network_topology.svg

echo "Network topology generated: network_topology.png"

# SPOF 하이라이트 버전 생성
sed 's/node \[shape=box\];/node [shape=box]; edge [color=red, penwidth=3];/' network.dot > network_spof.dot
dot -Tpng network_spof.dot -o network_spof.png

echo "SPOF highlighted topology: network_spof.png"
```

#### 실시간 SPOF 모니터링

```python
# spof_monitor.py - 실시간 SPOF 모니터링
import time
import threading
import logging
from prometheus_client import start_http_server, Gauge, Counter

class SPOFMonitor:
    def __init__(self):
        self.spof_gauge = Gauge('network_spof_count', 'Number of detected SPOFs')
        self.link_status_gauge = Gauge('network_link_status', 'Link status', ['device', 'interface'])
        self.spof_alert_counter = Counter('network_spof_alerts_total', 'Total SPOF alerts')
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def monitor_links(self):
        """링크 상태 모니터링"""
        while True:
            try:
                devices = self.get_monitored_devices()
                
                for device in devices:
                    interfaces = self.get_device_interfaces(device)
                    
                    for interface in interfaces:
                        status = self.check_interface_status(device, interface)
                        self.link_status_gauge.labels(
                            device=device['name'], 
                            interface=interface
                        ).set(1 if status == 'up' else 0)
                        
                        if status == 'down':
                            self.check_spof_impact(device, interface)
                
                time.sleep(30)  # 30초마다 체크
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def check_spof_impact(self, device, failed_interface):
        """장애 발생 시 SPOF 영향 분석"""
        remaining_paths = self.calculate_remaining_paths(device, failed_interface)
        
        if remaining_paths == 0:
            self.logger.critical(f"SPOF ALERT: {device['name']} {failed_interface} failure causes network partition")
            self.spof_alert_counter.inc()
            self.send_alert(device, failed_interface, 'SPOF_CRITICAL')
        elif remaining_paths == 1:
            self.logger.warning(f"SPOF WARNING: {device['name']} {failed_interface} creates single path")
            self.send_alert(device, failed_interface, 'SPOF_WARNING')
    
    def start_monitoring(self):
        """모니터링 시작"""
        # Prometheus 메트릭 서버 시작
        start_http_server(8000)
        
        # 링크 모니터링 쓰레드 시작
        monitor_thread = threading.Thread(target=self.monitor_links)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.logger.info("SPOF monitoring started on port 8000")
        
        # 메인 루프
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped")

if __name__ == "__main__":
    monitor = SPOFMonitor()
    monitor.start_monitoring()
```

#### SPOF 제거 우선순위 매트릭스

```bash
# SPOF 위험도 평가 매트릭스
위험도 = (영향도 × 발생가능성) / 대응시간

영향도 점수:
- 전체 네트워크 단절: 10
- 부분 네트워크 단절: 7  
- 성능 저하: 4
- 일부 서비스 영향: 2

발생가능성 점수:
- 매우 높음 (월 1회 이상): 10
- 높음 (분기 1회): 7
- 보통 (연 2-3회): 4  
- 낮음 (연 1회 이하): 2

대응시간 점수:
- 4시간 이상: 1
- 2-4시간: 2
- 1-2시간: 4
- 1시간 이하: 7

# 우선순위 결정
우선순위 1 (즉시): 위험도 > 35
우선순위 2 (1개월): 위험도 20-35
우선순위 3 (3개월): 위험도 10-20  
우선순위 4 (1년): 위험도 < 10
```
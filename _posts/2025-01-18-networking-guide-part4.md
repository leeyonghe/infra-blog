---
layout: post
title: "네트워킹 완전 가이드 4편 - 무선 네트워킹과 최신 기술 | Complete Network Guide Part 4 - Wireless Networking & Modern Technologies"
date: 2025-01-18 12:00:00 +0900
categories: [Networking, Wireless]
tags: [wifi, wireless, 5g, sd-wan, network-automation, edge-computing, wifi6]
---

무선 네트워킹 기술과 최신 네트워크 기술 동향을 깊이 있게 알아보고 실무에 적용해보겠습니다.

## 무선 네트워킹 기초 | Wireless Networking Fundamentals

### 📡 무선 통신 기본 개념

#### 전파의 특성
```bash
# 주파수 대역별 특성
2.4GHz 대역:
- 전파 도달 거리: 길음 (벽 투과력 좋음)
- 대역폭: 제한적 (최대 3개 비간섭 채널)
- 간섭: 많음 (전자레인지, 블루투스 등)

5GHz 대역:  
- 전파 도달 거리: 짧음 (직진성 강함)
- 대역폭: 풍부 (최대 25개 비간섭 채널)
- 간섭: 적음

6GHz 대역 (Wi-Fi 6E):
- 전파 도달 거리: 매우 짧음
- 대역폭: 매우 풍부 (59개 20MHz 채널)
- 간섭: 거의 없음
```

#### dBm과 신호 강도
```bash
# dBm 참조표
-30 dBm: 매우 강함 (AP 바로 옆)
-50 dBm: 강함 (우수한 연결)
-60 dBm: 좋음 (안정적 연결)
-70 dBm: 약함 (최소 연결 가능)
-80 dBm: 매우 약함 (연결 불안정)
-90 dBm: 연결 불가

# 신호 강도 측정 (Linux)
iwconfig wlan0
iw dev wlan0 scan | grep -E "SSID|signal|freq"
wavemon  # 실시간 무선 모니터링
```

### 🔐 Wi-Fi 보안 기술

#### 보안 프로토콜 발전사
```bash
WEP (Wired Equivalent Privacy):
- 1997년 도입, 현재 사용 금지
- 40/104비트 키 길이
- RC4 암호화, 쉽게 크랙됨

WPA (Wi-Fi Protected Access):
- 2003년 도입, WEP의 임시 대안
- TKIP 암호화, RC4 기반
- 동적 키 생성

WPA2 (802.11i):
- 2004년 도입, 현재 표준
- AES-CCMP 암호화
- PSK(개인용) / Enterprise(기업용)

WPA3 (2018년 도입):
- SAE (Simultaneous Authentication of Equals)
- 개선된 오픈 네트워크 보안
- 더 강력한 암호화 (192비트)
```

#### WPA2/WPA3 Enterprise 설정

```bash
# FreeRADIUS 서버 설정 (/etc/freeradius/clients.conf)
client wireless-controller {
    ipaddr = 192.168.1.10
    secret = radiusSecret123
    shortname = wlc
}

# 사용자 인증 설정 (/etc/freeradius/users)
john    Cleartext-Password := "password123"
        Tunnel-Type = VLAN,
        Tunnel-Medium-Type = IEEE-802,
        Tunnel-Private-Group-Id = 10

# 인증서 기반 설정 (802.1X EAP-TLS)
alice   TLS-Cert-Serial := "1234567890"
        Tunnel-Type = VLAN,
        Tunnel-Medium-Type = IEEE-802,
        Tunnel-Private-Group-Id = 20

# Cisco WLC RADIUS 설정
(Cisco Controller) > configure terminal
(Cisco Controller) > radius auth add 1 192.168.1.100 1812 radiusSecret123
(Cisco Controller) > radius acct add 1 192.168.1.100 1813 radiusSecret123
(Cisco Controller) > wlan create 10 CORPORATE
(Cisco Controller) > wlan security 802.1x enable 10
(Cisco Controller) > wlan security wpa akm 802.1x enable 10
```

### 📶 Wi-Fi 6/6E 기술

#### Wi-Fi 6 (802.11ax) 주요 특징
```bash
OFDMA (Orthogonal Frequency Division Multiple Access):
- 다중 사용자 동시 전송
- 효율적인 스펙트럼 사용
- 지연 시간 감소

MU-MIMO (Multi-User MIMO):
- 최대 8x8 안테나 지원
- 다운링크/업링크 모두 지원
- 동시 다중 사용자 서비스

1024-QAM:
- 25% 높은 데이터 전송률
- 신호 품질이 좋은 환경에서 효과적

TWT (Target Wake Time):
- IoT 디바이스 배터리 수명 연장
- 스케줄링된 통신
```

#### Wi-Fi 6 AP 설정 실습
```bash
# Cisco Wi-Fi 6 AP 설정
# 802.11ax 활성화
ap dot11 24ghz radio 1
 station-role root
 power local maximum 20
 channel width 80
 txpower auto

# OFDMA 활성화  
ap dot11 5ghz radio 2
 station-role root
 power local maximum 23
 channel width 160
 ofdma
 mu-mimo

# BSS Coloring 설정 (간섭 감소)
wlan 10
 bss-color 1-63
 
# TWT 설정
ap name AP-WIFI6-01
 twt enable
```

## 5G와 셀룰러 기술 | 5G and Cellular Technology

### 📱 5G 기술 개요

#### 5G 네트워크 아키텍처
```bash
# 5G 핵심 구성요소
RAN (Radio Access Network):
- gNB (5G Base Station)
- CU (Centralized Unit)  
- DU (Distributed Unit)
- RU (Radio Unit)

Core Network (5GC):
- AMF (Access and Mobility Management Function)
- SMF (Session Management Function)
- UPF (User Plane Function)
- AUSF (Authentication Server Function)
- UDM (Unified Data Management)

# 5G 주파수 대역
Sub-6GHz (FR1):
- 저주파: 600MHz ~ 6GHz
- 넓은 커버리지, 건물 투과력 좋음
- 최대 속도: 1Gbps

mmWave (FR2):
- 고주파: 24GHz ~ 100GHz  
- 초고속, 초저지연
- 커버리지 제한적
- 최대 속도: 10Gbps+
```

#### 5G 네트워크 슬라이싱
```bash
# 슬라이스 유형별 특성
eMBB (Enhanced Mobile Broadband):
- 높은 대역폭 (20Gbps+)  
- 4K/8K 비디오, AR/VR
- 지연시간: 4ms 이하

URLLC (Ultra-Reliable Low-Latency Communications):
- 초저지연 (1ms)
- 99.999% 신뢰성
- 자율주행, 산업 자동화

mMTC (Massive Machine Type Communications):  
- 대량 IoT 디바이스 (1M/km²)
- 저전력, 저비용
- 스마트시티, 농업 IoT
```

### 🌐 SD-WAN 기술

SD-WAN은 소프트웨어 정의 방식으로 WAN을 관리하는 기술입니다.

#### SD-WAN 아키텍처
```bash
# SD-WAN 구성요소
Edge Device (vCPE/uCPE):
- 지사에 설치되는 SD-WAN 장비
- 트래픽 라우팅 및 정책 적용

Orchestrator:  
- 중앙 집중식 관리 플랫폼
- 정책 설정 및 배포

Controller:
- 네트워크 토폴로지 관리
- 경로 최적화 결정

# 전송 방식
MPLS: 높은 품질, 높은 비용
Internet: 낮은 비용, 가변적 품질  
LTE/5G: 이동성, 백업용
```

#### SD-WAN 구현 (VeloCloud/VMware)
```bash
# Edge 설정 예시
# WAN 인터페이스 설정
configure
interface GE1
 ip dhcp
 wan-interface internet1
 exit

interface GE2  
 ip address 10.1.1.1/30
 wan-interface mpls1
 exit

# 비즈니스 정책 설정
business-policy VOICE
 application-classification voice
 sla latency 150
 sla jitter 30
 sla loss 1
 path-preference mpls1
 path-preference internet1

# 애플리케이션 기반 라우팅
application-map
 application office365 path internet1
 application salesforce path internet1
 application voice path mpls1
 application video path mpls1
```

#### Open Source SD-WAN (FlexiWAN)
```bash
# FlexiWAN Agent 설치
curl -s https://get.flexiwan.com | sudo bash

# 디바이스 등록
flexiwan-mgmt device register --token <registration-token>

# 인터페이스 설정
{
  "interfaces": [
    {
      "name": "eth0",
      "type": "WAN", 
      "dhcp": "yes",
      "metric": 100
    },
    {
      "name": "eth1",
      "type": "LAN",
      "addr": "192.168.1.1/24"
    }
  ]
}

# 터널 설정
{
  "tunnels": [
    {
      "num": 1,
      "peer": "203.0.113.2",
      "peer_port": 4789,
      "local_port": 4789,
      "encryption": "yes"
    }
  ]
}
```

## 네트워크 가상화 | Network Virtualization

### 🔀 SDN (Software Defined Networking)

#### OpenFlow 기본 개념
```bash
# OpenFlow 스위치 구조
Flow Table → Group Table → Meter Table

# Flow Entry 구성
Match Fields: 패킷 매칭 조건
Instructions: 수행할 동작  
Counters: 통계 정보
Timeouts: 타임아웃 설정
Priority: 우선순위

# OpenFlow 메시지 유형
Controller-to-Switch:
- Flow-Mod: 플로우 테이블 수정
- Group-Mod: 그룹 테이블 수정
- Port-Mod: 포트 설정 수정

Switch-to-Controller:
- Packet-In: 매칭되지 않은 패킷 전송
- Flow-Removed: 플로우 제거 알림
```

#### OpenDaylight 컨트롤러 설정
```bash
# OpenDaylight 설치
wget https://nexus.opendaylight.org/content/repositories/opendaylight.release/org/opendaylight/integration/karaf/0.13.3/karaf-0.13.3.tar.gz
tar -xzf karaf-0.13.3.tar.gz
cd karaf-0.13.3

# 컨트롤러 시작
./bin/karaf

# 필수 피처 설치
opendaylight-user@root> feature:install odl-restconf odl-l2switch-switch-ui odl-mdsal-apidocs

# REST API를 통한 플로우 추가
curl -u admin:admin -H "Content-Type: application/json" -X PUT \
http://localhost:8181/restconf/config/opendaylight-inventory:nodes/node/openflow:1/flow-node-inventory:table/0/flow/1 \
-d '{
  "flow": [
    {
      "id": "1",
      "match": {
        "ethernet-match": {
          "ethernet-type": {
            "type": 2048
          }
        },
        "ipv4-destination": "192.168.1.100/32"
      },
      "instructions": {
        "instruction": [
          {
            "order": 0,
            "apply-actions": {
              "action": [
                {
                  "order": 0,
                  "output-action": {
                    "output-node-connector": "2"
                  }
                }
              ]
            }
          }
        ]
      }
    }
  ]
}'
```

### ☁️ 네트워크 함수 가상화 (NFV)

#### VNF (Virtual Network Function) 구현
```bash
# 가상 라우터 (VyOS) 배포
# Docker 컨테이너로 배포
docker run -d --name vyos-router \
  --privileged \
  --cap-add=NET_ADMIN \
  -v /lib/modules:/lib/modules:ro \
  vyos/vyos:1.3

# 네트워크 네임스페이스 설정
ip netns add router-ns
ip link add veth0 type veth peer name veth1
ip link set veth1 netns router-ns
ip netns exec router-ns ip addr add 10.0.1.1/24 dev veth1
ip netns exec router-ns ip link set veth1 up

# 가상 방화벽 (iptables 기반)
#!/bin/bash
# Virtual Firewall Function
iptables -F
iptables -P INPUT DROP
iptables -P FORWARD DROP  
iptables -P OUTPUT ACCEPT

# 허용 규칙
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT

# 포워딩 규칙
iptables -A FORWARD -i eth0 -o eth1 -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT

# NAT 설정
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
```

## 엣지 컴퓨팅 | Edge Computing

### 🏗️ MEC (Multi-access Edge Computing)

#### MEC 플랫폼 구조
```bash
# MEC 구성요소
MEC Platform:
- 애플리케이션 라이프사이클 관리
- 서비스 레지스트리 및 발견
- 트래픽 라우팅

MEC Apps:
- 실시간 비디오 분석
- AR/VR 애플리케이션  
- IoT 데이터 처리

# MEC 배포 시나리오
Corporate Edge: 기업 캠퍼스
Service Provider Edge: 통신사 CO
Access Edge: 기지국, Wi-Fi AP
Device Edge: CPE, 게이트웨이
```

#### Kubernetes 기반 엣지 컴퓨팅
```yaml
# Edge Node 설정
apiVersion: v1
kind: Node
metadata:
  name: edge-node-01
  labels:
    node-type: edge
    location: branch-office
    kubernetes.io/arch: arm64

# Edge 애플리케이션 배포
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video-analytics
spec:
  replicas: 1
  selector:
    matchLabels:
      app: video-analytics
  template:
    metadata:
      labels:
        app: video-analytics
    spec:
      nodeSelector:
        node-type: edge
      containers:
      - name: analytics
        image: video-analytics:latest
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi" 
            cpu: "250m"
        env:
        - name: CAMERA_URL
          value: "rtsp://192.168.1.100:554/stream"
```

## 네트워크 자동화 | Network Automation

### 🤖 Ansible 네트워크 자동화

#### 네트워크 인벤토리 설정
```yaml
# inventory.yml
all:
  children:
    routers:
      hosts:
        router01:
          ansible_host: 192.168.1.1
          ansible_network_os: ios
        router02:
          ansible_host: 192.168.1.2
          ansible_network_os: ios
    switches:
      hosts:
        switch01:
          ansible_host: 192.168.1.10
          ansible_network_os: ios
        switch02:
          ansible_host: 192.168.1.11
          ansible_network_os: ios

  vars:
    ansible_connection: network_cli
    ansible_user: admin
    ansible_ssh_pass: "{{ vault_ssh_password }}"
    ansible_become: yes
    ansible_become_method: enable
    ansible_become_pass: "{{ vault_enable_password }}"
```

#### VLAN 자동 배포 플레이북
```yaml
# deploy-vlans.yml
---
- name: Configure VLANs across switches
  hosts: switches
  gather_facts: no
  vars:
    vlans:
      - id: 10
        name: SALES
        ports: [1, 2, 3, 4, 5]
      - id: 20
        name: ENGINEERING  
        ports: [6, 7, 8, 9, 10]
      - id: 30
        name: GUEST
        ports: [11, 12]

  tasks:
    - name: Create VLANs
      ios_vlan:
        vlan_id: "{{ item.id }}"
        name: "{{ item.name }}"
        state: present
      loop: "{{ vlans }}"

    - name: Configure access ports
      ios_l2_interfaces:
        config:
          - name: "FastEthernet0/{{ item[1] }}"
            access:
              vlan: "{{ item[0].id }}"
      with_subelements:
        - "{{ vlans }}"
        - ports

    - name: Configure trunk port
      ios_l2_interfaces:
        config:
          - name: FastEthernet0/24
            trunk:
              allowed_vlans: "10,20,30"
              native_vlan: 1
```

### 🐍 Python 네트워크 자동화

#### NAPALM을 이용한 네트워크 관리
```python
from napalm import get_network_driver
import json

# 디바이스 연결 설정
driver = get_network_driver('ios')
device = driver('192.168.1.1', 'admin', 'password')

try:
    # 디바이스 연결
    device.open()
    
    # 인터페이스 정보 수집
    interfaces = device.get_interfaces()
    print(json.dumps(interfaces, indent=2))
    
    # 라우팅 테이블 확인
    routes = device.get_route_to('8.8.8.8')
    print(json.dumps(routes, indent=2))
    
    # 설정 변경 (ACL 추가)
    config = """
    ip access-list extended BLOCK_SOCIAL
    deny tcp any host 23.35.67.140 eq 80
    permit ip any any
    """
    
    # 설정 로드 (아직 적용하지 않음)
    device.load_merge_candidate(config=config)
    
    # 설정 차이 확인
    diff = device.compare_config()
    if diff:
        print("Configuration diff:")
        print(diff)
        
        # 사용자 확인 후 적용
        confirm = input("Apply configuration? (y/n): ")
        if confirm.lower() == 'y':
            device.commit_config()
            print("Configuration applied successfully")
        else:
            device.discard_config()
            print("Configuration discarded")
    
finally:
    device.close()
```

#### Netmiko를 이용한 대량 설정
```python
from netmiko import ConnectHandler
from concurrent.futures import ThreadPoolExecutor
import threading

# 디바이스 리스트
devices = [
    {
        'device_type': 'cisco_ios',
        'host': '192.168.1.1',
        'username': 'admin',
        'password': 'password',
        'hostname': 'Router01'
    },
    {
        'device_type': 'cisco_ios', 
        'host': '192.168.1.2',
        'username': 'admin',
        'password': 'password',
        'hostname': 'Router02'
    }
]

def configure_device(device):
    """개별 디바이스 설정 함수"""
    try:
        # SSH 연결
        connection = ConnectHandler(**device)
        
        # NTP 서버 설정
        ntp_config = [
            'ntp server 192.168.1.100',
            'ntp server 8.8.8.8',
            'clock timezone KST 9 0'
        ]
        
        # 설정 적용
        output = connection.send_config_set(ntp_config)
        
        # SNMP 설정
        snmp_config = [
            'snmp-server community public ro',
            'snmp-server host 192.168.1.200 version 2c public'
        ]
        
        output += connection.send_config_set(snmp_config)
        
        # 설정 저장
        save_output = connection.save_config()
        
        connection.disconnect()
        
        print(f"✅ {device['hostname']}: Configuration completed")
        return True
        
    except Exception as e:
        print(f"❌ {device['hostname']}: Error - {str(e)}")
        return False

# 병렬 실행
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(configure_device, devices))

success_count = sum(results)
print(f"\n완료: {success_count}/{len(devices)} 디바이스")
```

### 📊 네트워크 모니터링 자동화

#### Prometheus + Grafana 모니터링
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'snmp-devices'
    static_configs:
      - targets:
        - 192.168.1.1  # Router01
        - 192.168.1.2  # Router02
        - 192.168.1.10 # Switch01
    metrics_path: /snmp
    params:
      module: [if_mib]
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: 192.168.1.100:9116  # SNMP Exporter

  - job_name: 'snmp-exporter'
    static_configs:
      - targets: ['192.168.1.100:9116']
```

#### 네트워크 장애 알림 자동화
```python
import requests
import smtplib
from email.mime.text import MIMEText
import time

class NetworkMonitor:
    def __init__(self):
        self.devices = [
            {'name': 'Router01', 'ip': '192.168.1.1'},
            {'name': 'Switch01', 'ip': '192.168.1.10'},
            {'name': 'AP01', 'ip': '192.168.1.20'}
        ]
        self.previous_status = {}
        
    def check_device(self, device):
        """SNMP 또는 PING으로 디바이스 상태 확인"""
        import subprocess
        
        try:
            # PING 테스트
            result = subprocess.run(
                ['ping', '-c', '3', '-W', '3', device['ip']], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                return 'UP'
            else:
                return 'DOWN'
                
        except Exception as e:
            print(f"Error checking {device['name']}: {e}")
            return 'ERROR'
    
    def send_alert(self, device, status):
        """Slack 또는 이메일로 알림 전송"""
        # Slack 알림
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        
        message = {
            "text": f"🚨 Network Alert: {device['name']} is {status}",
            "attachments": [
                {
                    "color": "danger" if status == 'DOWN' else "good",
                    "fields": [
                        {"title": "Device", "value": device['name'], "short": True},
                        {"title": "IP", "value": device['ip'], "short": True},
                        {"title": "Status", "value": status, "short": True},
                        {"title": "Time", "value": time.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                print(f"Alert sent for {device['name']}")
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def monitor_loop(self):
        """메인 모니터링 루프"""
        while True:
            for device in self.devices:
                current_status = self.check_device(device)
                previous_status = self.previous_status.get(device['name'], 'UP')
                
                # 상태 변화 감지
                if current_status != previous_status:
                    print(f"{device['name']} status changed: {previous_status} -> {current_status}")
                    self.send_alert(device, current_status)
                    
                self.previous_status[device['name']] = current_status
                
            time.sleep(60)  # 1분마다 체크

if __name__ == "__main__":
    monitor = NetworkMonitor()
    monitor.monitor_loop()
```

## 다음 편 예고

다음 포스트에서는 **네트워크 보안과 트러블슈팅**의 고급 주제들을 다룰 예정입니다:
- 차세대 방화벽과 IDS/IPS
- 네트워크 포렌식과 위협 헌팅  
- 제로 트러스트 네트워크 아키텍처
- 고급 네트워크 문제 해결 기법

무선 네트워킹과 최신 기술들을 마스터하셨나요? 🌐✨
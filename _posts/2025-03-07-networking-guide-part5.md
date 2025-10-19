---
layout: post
title: "네트워킹 완전 가이드 5편 - 네트워크 보안과 트러블슈팅 | Complete Network Guide Part 5 - Network Security & Troubleshooting"
date: 2025-03-07 12:00:00 +0900
categories: [Networking, Security]
tags: [network-security, firewall, ids, ips, zero-trust, troubleshooting, forensics]
---

네트워크 보안의 최신 기술과 고급 트러블슈팅 기법을 실무 중심으로 완전히 마스터해보겠습니다.

## 차세대 방화벽 (NGFW) | Next-Generation Firewall

### 🔥 NGFW 기능과 특징

#### 전통 방화벽 vs NGFW 비교
```bash
Traditional Firewall:
- L3/L4 패킷 필터링 (IP, Port)
- Stateful 연결 추적
- NAT/PAT 기능

Next-Generation Firewall (NGFW):
- Application Awareness (L7)
- Intrusion Prevention (IPS)
- SSL/TLS Decryption
- User Identity Integration
- Advanced Threat Protection
- Sandboxing
```

#### Palo Alto NGFW 설정 실습
```bash
# 보안 정책 설정
# 애플리케이션 기반 정책
configure
set rulebase security rules "Allow-Office365" from any
set rulebase security rules "Allow-Office365" to any
set rulebase security rules "Allow-Office365" source any
set rulebase security rules "Allow-Office365" destination any
set rulebase security rules "Allow-Office365" application [ ms-office365 outlook-web-access ]
set rulebase security rules "Allow-Office365" service application-default
set rulebase security rules "Allow-Office365" action allow

# 사용자 기반 정책
set rulebase security rules "Block-Social-Media" from trust
set rulebase security rules "Block-Social-Media" to untrust
set rulebase security rules "Block-Social-Media" source-user [ "DOMAIN\sales-team" ]
set rulebase security rules "Block-Social-Media" application [ facebook twitter instagram ]
set rulebase security rules "Block-Social-Media" action deny

# SSL Decryption 정책
set shared ssl-decrypt ssl-decrypt-policy "Decrypt-Inbound" rules "web-traffic"
set shared ssl-decrypt ssl-decrypt-policy "Decrypt-Inbound" rules "web-traffic" category [ "business-and-economy" "computer-and-internet-info" ]
set shared ssl-decrypt ssl-decrypt-policy "Decrypt-Inbound" rules "web-traffic" action decrypt
```

#### pfSense 오픈소스 방화벽 설정
```bash
# pfSense 패키지 설치
pkg install pfSense-pkg-suricata
pkg install pfSense-pkg-pfBlockerNG

# Suricata IPS 설정
# /usr/local/etc/suricata/suricata.yaml
HOME_NET: "[192.168.1.0/24,10.0.0.0/8]"
EXTERNAL_NET: "!$HOME_NET"

rule-files:
  - emerging-threats.rules
  - botcc.rules
  - emerging-malware.rules

# 사용자 정의 룰
# /usr/local/etc/suricata/rules/local.rules
alert tcp $HOME_NET any -> $EXTERNAL_NET 80 (msg:"HTTP outbound connection"; sid:1000001; rev:1;)
alert tcp any any -> $HOME_NET 22 (msg:"SSH connection attempt"; threshold: type both, track by_src, count 5, seconds 60; sid:1000002; rev:1;)

# pfBlockerNG GeoIP 차단
# 러시아, 중국, 북한 IP 대역 차단
Asia_Russia: Deny_Inbound, Deny_Outbound
Asia_China: Deny_Inbound, Permit_Outbound
Asia_North_Korea: Deny_Both
```

### 🛡️ IDS/IPS 시스템

#### Suricata IDS/IPS 배포
```bash
# Suricata 설치 (Ubuntu)
sudo add-apt-repository ppa:oisf/suricata-stable
sudo apt update
sudo apt install suricata

# 네트워크 인터페이스 설정
# /etc/suricata/suricata.yaml
af-packet:
  - interface: eth0
    cluster-id: 99
    cluster-type: cluster_flow
    defrag: yes
    use-mmap: yes
    mmap-locked: yes

# 룰셋 업데이트
sudo suricata-update
sudo suricata-update list-sources
sudo suricata-update enable-source et/pro  # Emerging Threats Pro
sudo suricata-update enable-source ptresearch/attackdetection

# 커스텀 룰 작성
# /etc/suricata/rules/local.rules
# DDoS 공격 탐지
alert tcp any any -> $HOME_NET any (msg:"Possible DDoS attack"; flags:S; threshold: type both, track by_src, count 100, seconds 10; sid:1000003; rev:1;)

# SQL 인젝션 탐지  
alert http any any -> $HOME_NET any (msg:"SQL Injection Attack"; content:"union select"; nocase; http_uri; sid:1000004; rev:1;)

# 크리덴셜 스터핑 탐지
alert http any any -> $HOME_NET any (msg:"Credential Stuffing Attack"; content:"POST"; http_method; threshold: type both, track by_src, count 50, seconds 60; sid:1000005; rev:1;)
```

#### Zeek (Bro) 네트워크 분석
```bash
# Zeek 설치
sudo apt install zeek

# 네트워크 인터페이스 설정
# /opt/zeek/etc/node.cfg
[zeek]
type=standalone
host=localhost
interface=eth0

# 커스텀 스크립트 작성
# /opt/zeek/share/zeek/site/local.zeek
@load base/protocols/http
@load base/protocols/dns
@load base/protocols/ssl

# HTTP 트래픽 모니터링
event http_request(c: connection, method: string, original_URI: string, unescaped_URI: string, version: string)
{
    if ( /\.(exe|zip|rar|7z)$/ in unescaped_URI )
        print fmt("%s 다운로드: %s -> %s%s", 
                 strftime("%Y-%m-%d %H:%M:%S", network_time()), 
                 c$id$orig_h, c$id$resp_h, unescaped_URI);
}

# DNS 모니터링
event dns_request(c: connection, msg: dns_msg, query: string, qtype: count, qclass: count)
{
    if ( /[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.in-addr\.arpa/ !in query )
    {
        if ( /\.tk$|\.ml$|\.ga$|\.cf$/ in query )
            print fmt("의심스러운 도메인 질의: %s -> %s", c$id$orig_h, query);
    }
}

# 서비스 시작
sudo zeekctl deploy
sudo zeekctl status
```

## 제로 트러스트 네트워크 | Zero Trust Network

### 🔒 제로 트러스트 아키텍처

#### 제로 트러스트 원칙
```bash
Never Trust, Always Verify:
- 기본적으로 모든 트래픽 차단
- 명시적 검증과 인가
- 최소 권한 원칙

Verify Explicitly:
- 사용자 신원 확인
- 디바이스 상태 검증
- 네트워크 위치 무관

Least Privileged Access:
- Just-in-Time (JIT) 액세스
- Just-Enough-Access (JEA)
- 위험 기반 적응형 정책

Assume Breach:
- 세그멘테이션과 격리
- 횡적 이동 차단
- 지속적 모니터링
```

#### 마이크로 세그멘테이션 구현
```bash
# Cisco ACI 마이크로 세그멘테이션
# 애플리케이션 프로파일 생성
apic1# configure
apic1(config)# tenant production
apic1(config-tenant)# application app-web
apic1(config-tenant-app)# epg web-tier
apic1(config-tenant-app-epg)# bridge-domain web-bd
apic1(config-tenant-app-epg)# exit

# 계약 (Contract) 정의
apic1(config-tenant-app)# contract web-to-db
apic1(config-tenant-app-contract)# subject db-access
apic1(config-tenant-app-contract-subject)# filter mysql-filter
apic1(config-tenant-app-contract-subject-filter)# entry mysql
apic1(config-tenant-app-contract-subject-filter-entry)# ether-type ip
apic1(config-tenant-app-contract-subject-filter-entry)# ip-protocol tcp
apic1(config-tenant-app-contract-subject-filter-entry)# destination-port-range from 3306 to 3306

# VMware NSX 마이크로 세그멘테이션
# 분산 방화벽 정책
nsxcli> configure firewall
nsxcli(firewall)> add rule web-tier-protection
nsxcli(firewall-rule)> set source security-group web-servers
nsxcli(firewall-rule)> set destination security-group db-servers
nsxcli(firewall-rule)> set service MYSQL
nsxcli(firewall-rule)> set action allow
nsxcli(firewall-rule)> set applied-to security-group web-servers
nsxcli(firewall-rule)> commit
```

#### Kubernetes 네트워크 정책
```yaml
# 네임스페이스 격리 정책
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# 웹 애플리케이션 정책
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: web-app-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web-frontend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: api-backend
    ports:
    - protocol: TCP
      port: 3000
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53  # DNS

---
# 데이터베이스 정책
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: mysql-db
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-backend
    ports:
    - protocol: TCP
      port: 3306
```

## 네트워크 포렌식 | Network Forensics

### 🔍 패킷 분석과 증거 수집

#### Wireshark 고급 분석 기법
```bash
# 의심스러운 트래픽 필터링
# 대량 데이터 전송 탐지
tcp.len > 1400 and ip.src == 192.168.1.100

# DNS 터널링 탐지
dns.qry.name contains "."
dns.qry.name matches "^[a-f0-9]{20,}\\."

# HTTP POST 데이터 exfiltration
http.request.method == "POST" and http.content_length > 10000

# 비정상적인 포트 사용
tcp.port == 443 and not ssl.handshake.type
tcp.port == 80 and ssl.handshake.type

# 커맨드 앤 컨트롤 통신 패턴
(tcp.flags.push == 1) and (tcp.len < 100) and (tcp.len > 10)

# 통계 분석을 위한 tshark 사용
# 상위 통신 호스트
tshark -r capture.pcap -q -z conv,ip | sort -k7 -nr | head -20

# 프로토콜 분포
tshark -r capture.pcap -q -z prot,colinfo

# 시간별 트래픽 분석
tshark -r capture.pcap -T fields -e frame.time -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport | 
awk '{print $1 " " $2 ":" $4 " -> " $3 ":" $5}' | sort | uniq -c | sort -nr
```

#### 네트워크 플로우 분석 (nfcapd)
```bash
# nfcapd 설정 및 시작
nfcapd -w -D -p 9995 -l /var/cache/nfcapd

# 라우터에서 NetFlow 활성화 (Cisco)
interface FastEthernet0/1
 ip flow ingress
 ip flow egress

ip flow-export source FastEthernet0/1
ip flow-export version 9
ip flow-export destination 192.168.1.100 9995

# 플로우 데이터 분석
# 상위 통신량 호스트
nfdump -r /var/cache/nfcapd/nfcapd.* -s srcip/bytes -n 20

# 특정 시간대 분석
nfdump -r /var/cache/nfcapd/nfcapd.* -t 2024-03-30.10:00:00-2024-03-30.11:00:00 -s dstip/packets

# 의심스러운 포트 사용 
nfdump -r /var/cache/nfcapd/nfcapd.* 'port > 10000 and port < 65000' -s srcip/flows

# 대용량 전송 탐지
nfdump -r /var/cache/nfcapd/nfcapd.* 'bytes > 100000000' -o extended
```

### 🕵️ 위협 헌팅 (Threat Hunting)

#### ELK 스택 기반 위협 헌팅
```yaml
# Logstash 설정 - 네트워크 로그 파싱
# /etc/logstash/conf.d/network-logs.conf
input {
  beats {
    port => 5044
  }
  syslog {
    port => 514
    type => "firewall"
  }
}

filter {
  if [type] == "firewall" {
    grok {
      match => { 
        "message" => "%{TIMESTAMP_ISO8601:timestamp} %{WORD:device} %{WORD:action} %{IP:src_ip}:%{INT:src_port} -> %{IP:dst_ip}:%{INT:dst_port} %{WORD:protocol}"
      }
    }
    
    geoip {
      source => "src_ip"
      target => "src_geoip"
    }
    
    if [src_geoip][country_name] in ["Russia", "China", "North Korea"] {
      mutate {
        add_tag => [ "suspicious_geo" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "network-logs-%{+YYYY.MM.dd}"
  }
}
```

#### Kibana 위협 헌팅 쿼리
```json
{
  "query": {
    "bool": {
      "must": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-1h"
            }
          }
        },
        {
          "terms": {
            "tags": ["suspicious_geo"]
          }
        }
      ],
      "should": [
        {
          "range": {
            "bytes": {
              "gte": 1000000
            }
          }
        },
        {
          "script": {
            "script": {
              "source": "doc['dst_port'].value > 10000 && doc['dst_port'].value < 65000"
            }
          }
        }
      ]
    }
  },
  "aggs": {
    "suspicious_ips": {
      "terms": {
        "field": "src_ip",
        "size": 100
      },
      "aggs": {
        "unique_ports": {
          "cardinality": {
            "field": "dst_port"
          }
        },
        "total_bytes": {
          "sum": {
            "field": "bytes"
          }
        }
      }
    }
  }
}
```

#### YARA 룰 기반 네트워크 탐지
```bash
# YARA 룰 예시 - 악성코드 통신 패턴
rule APT_Communication_Pattern
{
    meta:
        description = "APT 그룹 통신 패턴 탐지"
        author = "Security Team"
        date = "2024-03-30"
        
    strings:
        $user_agent = "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"
        $uri_pattern = /\/[a-f0-9]{8}\/[a-f0-9]{8}/
        $post_data = { 48 54 54 50 2F 31 2E 31 20 32 30 30 20 4F 4B }  // "HTTP/1.1 200 OK"
        
    condition:
        $user_agent and $uri_pattern and $post_data
}

rule DNS_Tunneling
{
    meta:
        description = "DNS 터널링 패턴 탐지"
        
    strings:
        $long_subdomain = /[a-zA-Z0-9]{50,}\..*\.(com|net|org)/
        $base64_pattern = /[A-Za-z0-9+\/]{20,}={0,2}/
        
    condition:
        $long_subdomain or $base64_pattern
}

# 네트워크 트래픽에 YARA 적용
suricata -c /etc/suricata/suricata.yaml -i eth0 --runmode single
```

## 고급 트러블슈팅 | Advanced Troubleshooting

### 🔧 네트워크 성능 문제 해결

#### 대역폭과 지연시간 분석
```bash
# iperf3를 이용한 성능 측정
# 서버 측
iperf3 -s -p 5201

# 클라이언트 측 - TCP 테스트
iperf3 -c 192.168.1.100 -t 30 -i 1 -P 4
# -t: 테스트 시간, -i: 인터벌, -P: 병렬 연결

# UDP 테스트 (패킷 손실 확인)
iperf3 -c 192.168.1.100 -u -b 100M -t 30

# 양방향 테스트
iperf3 -c 192.168.1.100 --bidir -t 30

# MTU 크기 최적화 테스트
ping -M do -s 1472 192.168.1.100  # Linux
ping -f -l 1472 192.168.1.100     # Windows

# 점진적 MTU 테스트 스크립트
#!/bin/bash
for size in {1200..1500..50}; do
    echo "Testing MTU size: $size"
    ping -M do -s $size -c 1 192.168.1.100 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ MTU $size: Success"
    else
        echo "❌ MTU $size: Failed"
    fi
done
```

#### 네트워크 지연시간 상세 분석
```bash
# hping3를 이용한 다양한 테스트
# TCP SYN 플러드 테스트
hping3 -S -p 80 -i u1000 192.168.1.100

# UDP 포트 스캔
hping3 --udp -p 53 -c 3 192.168.1.100

# ICMP 타임스탬프 요청
hping3 --icmp-ts 192.168.1.100

# traceroute 고급 옵션
# TCP traceroute
traceroute -T -p 443 google.com

# UDP traceroute  
traceroute -U -p 53 8.8.8.8

# Paris traceroute (로드 밸런싱 고려)
paris-traceroute google.com

# mtr - 실시간 traceroute
mtr --report --report-cycles 100 --no-dns google.com
```

### 📊 네트워크 모니터링과 분석

#### SNMP 기반 모니터링
```python
from pysnmp.hlapi import *
import time
import matplotlib.pyplot as plt

class NetworkMonitor:
    def __init__(self, host, community='public'):
        self.host = host
        self.community = community
        
    def get_interface_stats(self, interface_index=2):
        """인터페이스 통계 수집"""
        oids = {
            'ifInOctets': f'1.3.6.1.2.1.2.2.1.10.{interface_index}',
            'ifOutOctets': f'1.3.6.1.2.1.2.2.1.16.{interface_index}',
            'ifSpeed': f'1.3.6.1.2.1.2.2.1.5.{interface_index}',
            'ifOperStatus': f'1.3.6.1.2.1.2.2.1.8.{interface_index}'
        }
        
        results = {}
        for name, oid in oids.items():
            for (errorIndication, errorStatus, errorIndex, varBinds) in nextCmd(
                SnmpEngine(),
                CommunityData(self.community),
                UdpTransportTarget((self.host, 161)),
                ContextData(),
                ObjectType(ObjectIdentity(oid)),
                lexicographicMode=False):
                
                if errorIndication:
                    print(errorIndication)
                    break
                elif errorStatus:
                    print('%s at %s' % (errorStatus.prettyPrint(),
                                        errorIndex and varBinds[int(errorIndex) - 1][0] or '?'))
                    break
                else:
                    for varBind in varBinds:
                        results[name] = int(varBind[1])
                    break
                    
        return results
    
    def calculate_bandwidth_utilization(self, interval=5, samples=60):
        """대역폭 사용률 계산"""
        prev_stats = self.get_interface_stats()
        time.sleep(interval)
        
        utilizations = []
        timestamps = []
        
        for i in range(samples):
            current_stats = self.get_interface_stats()
            
            # 바이트 증가량 계산
            in_bytes = current_stats['ifInOctets'] - prev_stats['ifInOctets']
            out_bytes = current_stats['ifOutOctets'] - prev_stats['ifOutOctets']
            
            # 비트로 변환하고 초당 계산
            in_bps = (in_bytes * 8) / interval
            out_bps = (out_bytes * 8) / interval
            
            # 사용률 계산 (%)
            interface_speed = current_stats['ifSpeed']
            in_utilization = (in_bps / interface_speed) * 100
            out_utilization = (out_bps / interface_speed) * 100
            
            utilizations.append({
                'timestamp': time.time(),
                'in_utilization': in_utilization,
                'out_utilization': out_utilization,
                'in_bps': in_bps,
                'out_bps': out_bps
            })
            
            print(f"시간: {time.strftime('%H:%M:%S')}, "
                  f"In: {in_utilization:.2f}% ({in_bps/1000000:.2f}Mbps), "
                  f"Out: {out_utilization:.2f}% ({out_bps/1000000:.2f}Mbps)")
            
            prev_stats = current_stats
            time.sleep(interval)
            
        return utilizations

# 사용 예시
if __name__ == "__main__":
    monitor = NetworkMonitor('192.168.1.1')
    data = monitor.calculate_bandwidth_utilization()
    
    # 그래프 생성
    timestamps = [d['timestamp'] for d in data]
    in_util = [d['in_utilization'] for d in data]
    out_util = [d['out_utilization'] for d in data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, in_util, label='Inbound', color='blue')
    plt.plot(timestamps, out_util, label='Outbound', color='red')
    plt.xlabel('Time')
    plt.ylabel('Bandwidth Utilization (%)')
    plt.title('Network Interface Utilization')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 🚨 네트워크 장애 대응 프로세스

#### 자동화된 장애 대응 스크립트
```bash
#!/bin/bash
# 네트워크 장애 자동 진단 및 대응 스크립트

LOG_FILE="/var/log/network_troubleshoot.log"
ALERT_THRESHOLD=5  # 5% 패킷 손실 임계값

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
    echo "$1"
}

# 1단계: 기본 연결성 확인
check_connectivity() {
    log_message "=== 1단계: 기본 연결성 확인 ==="
    
    # 게이트웨이 핑 테스트
    gateway=$(ip route | grep default | awk '{print $3}' | head -1)
    log_message "기본 게이트웨이: $gateway"
    
    packet_loss=$(ping -c 5 $gateway | grep "packet loss" | awk '{print $6}' | sed 's/%//')
    
    if (( $(echo "$packet_loss > $ALERT_THRESHOLD" | bc -l) )); then
        log_message "❌ 게이트웨이 연결 문제 감지: ${packet_loss}% 패킷 손실"
        return 1
    else
        log_message "✅ 게이트웨이 연결 정상: ${packet_loss}% 패킷 손실"
        return 0
    fi
}

# 2단계: DNS 해상도 확인  
check_dns() {
    log_message "=== 2단계: DNS 해상도 확인 ==="
    
    dns_servers=("8.8.8.8" "1.1.1.1" "168.126.63.1")
    test_domains=("google.com" "naver.com" "github.com")
    
    for server in "${dns_servers[@]}"; do
        for domain in "${test_domains[@]}"; do
            response_time=$(dig @$server $domain +short +time=3 2>&1)
            if [ $? -eq 0 ] && [ -n "$response_time" ]; then
                log_message "✅ DNS 서버 $server: $domain 해상도 성공"
            else
                log_message "❌ DNS 서버 $server: $domain 해상도 실패"
            fi
        done
    done
}

# 3단계: 인터페이스 상태 확인
check_interfaces() {
    log_message "=== 3단계: 네트워크 인터페이스 확인 ==="
    
    ip link show | grep -E "^[0-9]+:" | while read line; do
        interface=$(echo $line | cut -d: -f2 | sed 's/ //')
        status=$(echo $line | grep -o "state [A-Z]*" | cut -d' ' -f2)
        
        if [ "$status" = "UP" ]; then
            log_message "✅ 인터페이스 $interface: $status"
        else
            log_message "❌ 인터페이스 $interface: $status"
            
            # 인터페이스 재시작 시도
            log_message "인터페이스 $interface 재시작 시도중..."
            sudo ip link set $interface down
            sleep 2
            sudo ip link set $interface up
            sleep 5
            
            # 재확인
            new_status=$(ip link show $interface | grep -o "state [A-Z]*" | cut -d' ' -f2)
            log_message "재시작 후 상태: $new_status"
        fi
    done
}

# 4단계: 라우팅 테이블 확인
check_routing() {
    log_message "=== 4단계: 라우팅 테이블 확인 ==="
    
    # 기본 게이트웨이 확인
    default_routes=$(ip route | grep default | wc -l)
    if [ $default_routes -eq 0 ]; then
        log_message "❌ 기본 게이트웨이가 설정되지 않음"
        
        # DHCP 갱신 시도
        log_message "DHCP 갱신 시도중..."
        sudo dhclient -r
        sudo dhclient
    elif [ $default_routes -gt 1 ]; then
        log_message "⚠️  여러 개의 기본 게이트웨이 감지"
        ip route | grep default
    else
        log_message "✅ 기본 게이트웨이 정상"
    fi
    
    # 라우팅 테이블 출력
    log_message "현재 라우팅 테이블:"
    ip route >> $LOG_FILE
}

# 5단계: 포트 스캔 (서비스 가용성 확인)
check_services() {
    log_message "=== 5단계: 핵심 서비스 확인 ==="
    
    services=(
        "8.8.8.8:53:DNS"
        "google.com:80:HTTP"
        "google.com:443:HTTPS"
        "github.com:22:SSH"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r host port name <<< "$service"
        
        if timeout 5 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
            log_message "✅ $name ($host:$port): 접근 가능"
        else
            log_message "❌ $name ($host:$port): 접근 불가"
        fi
    done
}

# 6단계: 자동 복구 시도
auto_recovery() {
    log_message "=== 6단계: 자동 복구 시도 ==="
    
    # 네트워크 관리자 재시작
    if systemctl is-active NetworkManager >/dev/null 2>&1; then
        log_message "NetworkManager 재시작 중..."
        sudo systemctl restart NetworkManager
        sleep 10
    fi
    
    # 방화벽 규칙 초기화 (임시)
    log_message "방화벽 규칙 임시 허용..."
    sudo iptables -P INPUT ACCEPT
    sudo iptables -P FORWARD ACCEPT
    sudo iptables -P OUTPUT ACCEPT
    
    # DNS 캐시 플러시
    log_message "DNS 캐시 플러시..."
    sudo systemctl restart systemd-resolved
    
    log_message "자동 복구 완료. 5분 후 재테스트 예정..."
}

# 메인 실행 함수
main() {
    log_message "==================== 네트워크 진단 시작 ===================="
    
    failure_count=0
    
    check_connectivity || ((failure_count++))
    check_dns || ((failure_count++))  
    check_interfaces || ((failure_count++))
    check_routing || ((failure_count++))
    check_services || ((failure_count++))
    
    if [ $failure_count -gt 2 ]; then
        log_message "⚠️  심각한 네트워크 문제 감지 ($failure_count 개 실패)"
        auto_recovery
        
        # 5분 후 재테스트
        sleep 300
        log_message "==================== 복구 후 재진단 ===================="
        main
    else
        log_message "✅ 네트워크 상태 정상 (실패: $failure_count 개)"
    fi
    
    log_message "==================== 진단 완료 ===================="
}

# 스크립트 실행
main
```

## 최신 보안 위협과 대응

### 🎯 AI/ML 기반 위협 탐지

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class NetworkAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, network_data):
        """네트워크 데이터에서 피처 추출"""
        features = pd.DataFrame()
        
        # 시간대별 트래픽 패턴
        features['hour'] = pd.to_datetime(network_data['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(network_data['timestamp']).dt.dayofweek
        
        # 트래픽 볼륨
        features['bytes_per_second'] = network_data['bytes'] / network_data['duration']
        features['packets_per_second'] = network_data['packets'] / network_data['duration']
        
        # 연결 패턴
        features['unique_src_ports'] = network_data.groupby('src_ip')['src_port'].nunique()
        features['unique_dst_ports'] = network_data.groupby('src_ip')['dst_port'].nunique() 
        features['connection_count'] = network_data.groupby('src_ip').size()
        
        # 프로토콜 분포
        protocol_counts = pd.get_dummies(network_data['protocol'])
        features = pd.concat([features.reset_index(drop=True), 
                            protocol_counts.reset_index(drop=True)], axis=1)
        
        return features.fillna(0)
    
    def train(self, training_data):
        """정상 트래픽으로 모델 학습"""
        features = self.prepare_features(training_data)
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        
    def detect_anomalies(self, test_data):
        """이상 트래픽 탐지"""
        features = self.prepare_features(test_data)
        scaled_features = self.scaler.transform(features)
        
        # 이상 점수 계산 (-1: 이상, 1: 정상)
        predictions = self.model.predict(scaled_features)
        anomaly_scores = self.model.score_samples(scaled_features)
        
        # 결과 데이터프레임 생성
        results = test_data.copy()
        results['is_anomaly'] = predictions == -1
        results['anomaly_score'] = anomaly_scores
        
        return results
    
    def visualize_anomalies(self, results):
        """이상 탐지 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 시간별 이상 트래픽
        hourly_anomalies = results.groupby(
            pd.to_datetime(results['timestamp']).dt.hour
        )['is_anomaly'].sum()
        
        axes[0,0].bar(hourly_anomalies.index, hourly_anomalies.values)
        axes[0,0].set_title('시간대별 이상 트래픽')
        axes[0,0].set_xlabel('시간')
        axes[0,0].set_ylabel('이상 트래픽 수')
        
        # IP별 이상 점수 분포
        ip_scores = results.groupby('src_ip')['anomaly_score'].mean().sort_values()
        axes[0,1].barh(range(len(ip_scores.tail(10))), ip_scores.tail(10).values)
        axes[0,1].set_yticks(range(len(ip_scores.tail(10))))
        axes[0,1].set_yticklabels(ip_scores.tail(10).index)
        axes[0,1].set_title('상위 10개 의심 IP')
        
        # 포트 스캔 패턴
        port_scan_ips = results[results['unique_dst_ports'] > 100]['src_ip'].value_counts()
        if len(port_scan_ips) > 0:
            axes[1,0].bar(range(len(port_scan_ips.head(10))), port_scan_ips.head(10).values)
            axes[1,0].set_xticks(range(len(port_scan_ips.head(10))))
            axes[1,0].set_xticklabels(port_scan_ips.head(10).index, rotation=45)
            axes[1,0].set_title('포트 스캔 의심 IP')
        
        # 데이터 전송량 분포
        axes[1,1].scatter(results['bytes'], results['anomaly_score'], 
                         c=results['is_anomaly'], alpha=0.6)
        axes[1,1].set_xlabel('전송 바이트')
        axes[1,1].set_ylabel('이상 점수')
        axes[1,1].set_title('데이터 전송량 vs 이상 점수')
        
        plt.tight_layout()
        plt.show()

# 사용 예시
detector = NetworkAnomalyDetector()

# 모의 네트워크 데이터 생성
training_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-03-01', periods=10000, freq='1min'),
    'src_ip': np.random.choice(['192.168.1.100', '192.168.1.101', '192.168.1.102'], 10000),
    'dst_ip': np.random.choice(['8.8.8.8', '1.1.1.1', '208.67.222.222'], 10000),
    'src_port': np.random.randint(1024, 65535, 10000),
    'dst_port': np.random.choice([80, 443, 53, 22], 10000),
    'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], 10000),
    'bytes': np.random.lognormal(8, 2, 10000),
    'packets': np.random.poisson(50, 10000),
    'duration': np.random.exponential(30, 10000)
})

detector.train(training_data)

# 테스트 데이터에 일부 이상 트래픽 포함
test_data = training_data.sample(1000).copy()
# 포트 스캔 시뮬레이션
test_data.loc[0:50, 'dst_port'] = range(1, 51)
test_data.loc[0:50, 'src_ip'] = '10.0.0.100'  # 외부 IP

results = detector.detect_anomalies(test_data)
detector.visualize_anomalies(results)
```

## 마무리

이 시리즈를 통해 네트워킹의 모든 측면을 다뤘습니다:

1. **Part 1**: OSI 7계층과 TCP/IP 프로토콜 기초
2. **Part 2**: IP 주소 체계와 서브네팅 
3. **Part 3**: 라우팅과 스위칭 고급 기술
4. **Part 4**: 무선 네트워킹과 최신 기술 동향
5. **Part 5**: 네트워크 보안과 고급 트러블슈팅

실무에서 이 지식들을 적극 활용하여 안전하고 효율적인 네트워크 인프라를 구축하시기 바랍니다! 🌐🔒

## 추천 학습 자료

- **서적**: "Computer Networking: A Top-Down Approach" - Kurose & Ross
- **실습**: Packet Tracer, GNS3, EVE-NG
- **인증**: CCNA, CCNP, CCIE (Cisco), JNCIA, JNCIP (Juniper)
- **오픈소스**: Wireshark, pfSense, OpenWrt, Suricata

네트워크 마스터의 길, 함께 걸어가요! 🚀
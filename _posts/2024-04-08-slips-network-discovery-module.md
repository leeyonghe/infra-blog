---
layout: post
title: "Slips 네트워크 발견 모듈 상세 분석"
date: 2024-04-08 16:30:00 +0900
categories: [network-analysis]
tags: [network-discovery, network-security, system-architecture]
---

# Slips 네트워크 발견 모듈 상세 분석

네트워크 발견 모듈은 Slips의 네트워크 인프라 발견과 모니터링을 담당하는 핵심 컴포넌트입니다. 이 글에서는 네트워크 발견 모듈의 구현과 주요 기능에 대해 살펴보겠습니다.

## 1. 네트워크 발견 모듈 개요

네트워크 발견 모듈은 네트워크 내의 호스트와 서비스 및 설정을 자동으로 발견하고 모니터링하는 역할을 합니다. 주요 기능은 다음과 같습니다:

- 네트워크 호스트 발견
- 서비스 및 포트 스캔
- 네트워크 토폴로지 매핑
- 네트워크 변경사항 감지

## 2. 주요 기능

### 2.1 호스트 발견
```python
class NetworkDiscovery(Module):
    def __init__(self):
        super().__init__()
        self.discovered_hosts = {}
        self.active_services = {}
        self.network_topology = {}

    def discover_hosts(self, network):
        """
        네트워크 내의 호스트들을 발견합니다.

        Args:
            network (str): 탐색할 네트워크 범위
        """
        try:
            # ARP 스캔
            arp_hosts = self.arp_scan(network)

            # ICMP 스캔
            icmp_hosts = self.icmp_scan(network)

            # 호스트 정보 통합
            for host in set(arp_hosts + icmp_hosts):
                host_info = self.get_host_info(host)
                if host_info:
                    self.discovered_hosts[host] = host_info

            self.update_network_topology()
        except Exception as e:
            self.logger.error(f"호스트 발견 실패: {str(e)}")
```

### 2.2 서비스 발견
```python
def discover_services(self, host):
    """
    호스트의 실행 중인 서비스들을 발견합니다.

    Args:
        host (str): 대상 호스트의 IP

    Returns:
        dict: 서비스 정보
    """
    try:
        services = {
            'tcp': {},
            'udp': {},
            'metadata': {}
        }

        # TCP 포트 스캔
        tcp_ports = self.scan_tcp_ports(host)
        for port in tcp_ports:
            service = self.identify_service(host, port, 'tcp')
            if service:
                services['tcp'][port] = service

        # UDP 포트 스캔
        udp_ports = self.scan_udp_ports(host)
        for port in udp_ports:
            service = self.identify_service(host, port, 'udp')
            if service:
                services['udp'][port] = service

        # 서비스 메타데이터 수집
        services['metadata'] = self.collect_service_metadata(host, services)

        return services
    except Exception as e:
        self.logger.error(f"서비스 발견 실패: {str(e)}")
        return None
```

### 2.3 네트워크 토폴로지 매핑
```python
def map_network_topology(self):
    """
    네트워크 토폴로지를 매핑합니다.
    """
    try:
        topology = {
            'hosts': {},
            'connections': [],
            'routers': [],
            'switches': []
        }

        # 호스트 정보 수집
        for host, info in self.discovered_hosts.items():
            topology['hosts'][host] = {
                'ip': host,
                'mac': info.get('mac'),
                'os': info.get('os'),
                'services': info.get('services', {})
            }

        # 네트워크 연결 분석
        for host in topology['hosts']:
            connections = self.analyze_host_connections(host)
            topology['connections'].extend(connections)

        # 라우터 및 스위치 식별
        topology['routers'] = self.identify_routers()
        topology['switches'] = self.identify_switches()

        self.network_topology = topology
        return topology
    except Exception as e:
        self.logger.error(f"네트워크 토폴로지 매핑 실패: {str(e)}")
        return None
```

## 3. 변경사항 감지

### 3.1 호스트 변경 감지
```python
def detect_host_changes(self):
    """
    네트워크 호스트의 변경사항을 감지합니다.
    """
    try:
        changes = {
            'new_hosts': [],
            'removed_hosts': [],
            'modified_hosts': []
        }

        # 현재 호스트 재스캔
        current_hosts = set(self.discover_hosts(self.network))
        previous_hosts = set(self.discovered_hosts.keys())

        # 새로운 호스트 감지
        changes['new_hosts'] = list(current_hosts - previous_hosts)

        # 제거된 호스트 감지
        changes['removed_hosts'] = list(previous_hosts - current_hosts)

        # 변경된 호스트 감지
        for host in current_hosts & previous_hosts:
            if self.has_host_changed(host):
                changes['modified_hosts'].append(host)

        return changes
    except Exception as e:
        self.logger.error(f"호스트 변경 감지 실패: {str(e)}")
        return None
```

### 3.2 서비스 변경 감지
```python
def detect_service_changes(self, host):
    """
    호스트의 서비스 변경사항을 감지합니다.

    Args:
        host (str): 대상 호스트의 IP

    Returns:
        dict: 서비스 변경사항
    """
    try:
        changes = {
            'new_services': [],
            'removed_services': [],
            'modified_services': []
        }

        # 현재 서비스 재스캔
        current_services = self.discover_services(host)
        previous_services = self.active_services.get(host, {})

        # 새로운 서비스 감지
        for port, service in current_services.get('tcp', {}).items():
            if port not in previous_services.get('tcp', {}):
                changes['new_services'].append(service)

        # 제거된 서비스 감지
        for port, service in previous_services.get('tcp', {}).items():
            if port not in current_services.get('tcp', {}):
                changes['removed_services'].append(service)

        # 변경된 서비스 감지
        for port, service in current_services.get('tcp', {}).items():
            if port in previous_services.get('tcp', {}) and \
               service != previous_services['tcp'][port]:
                changes['modified_services'].append(service)

        return changes
    except Exception as e:
        self.logger.error(f"서비스 변경 감지 실패: {str(e)}")
        return None
```

## 4. 데이터 관리

### 4.1 네트워크 정보 저장
```python
def store_network_info(self, network_info):
    """
    네트워크 정보를 저장합니다.

    Args:
        network_info (dict): 저장할 네트워크 정보
    """
    try:
        self.db.set('network_info', json.dumps(network_info))
    except Exception as e:
        self.logger.error(f"네트워크 정보 저장 실패: {str(e)}")
```

### 4.2 네트워크 정보 검색
```python
def search_network_info(self, query):
    """
    네트워크 정보를 검색합니다.

    Args:
        query (dict): 검색 쿼리

    Returns:
        list: 검색 결과
    """
    try:
        results = []
        network_info = json.loads(self.db.get('network_info'))

        for host, info in network_info.get('hosts', {}).items():
            if self._matches_query(info, query):
                results.append(info)

        return results
    except Exception as e:
        self.logger.error(f"네트워크 정보 검색 실패: {str(e)}")
        return []
```

## 5. 성능 최적화

### 5.1 캐싱
```python
def cache_network_info(self, network_info, ttl=3600):
    """
    네트워크 정보를 캐시합니다.

    Args:
        network_info (dict): 캐시할 네트워크 정보
        ttl (int): 캐시 유효 기간(초)
    """
    try:
        self.redis.setex('network_cache', ttl, json.dumps(network_info))
    except Exception as e:
        self.logger.error(f"네트워크 정보 캐싱 실패: {str(e)}")
```

## 6. 결론

네트워크 발견 모듈은 Slips의 네트워크 인프라 관리와 보안 모니터링의 중요한 역할을 합니다. 주요 장점은 다음과 같습니다:

- 자동화된 네트워크 인프라 발견
- 실시간 서비스 모니터링
- 네트워크 토폴로지 매핑
- 변경사항 감지 및 알림

이러한 기능들은 Slips가 네트워크 환경을 효과적으로 모니터링하고 보안 위협에 대응할 수 있도록 지원합니다.
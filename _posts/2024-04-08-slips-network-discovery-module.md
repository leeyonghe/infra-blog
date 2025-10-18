---
layout: post
title: "Slips ?¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ ?ì„¸ ë¶„ì„"
date: 2024-04-08 16:30:00 +0900
categories: [network-analysis]
tags: [network-discovery, network-security, system-architecture]
---

Slips ?¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ ?ì„¸ ë¶„ì„

?¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ?€ Slips???¤íŠ¸?Œí¬ ?ì‚° ?ì??€ ëª¨ë‹ˆ?°ë§???´ë‹¹?˜ëŠ” ?µì‹¬ ì»´í¬?ŒíŠ¸?…ë‹ˆ?? ??ê¸€?ì„œ???¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ??êµ¬í˜„ê³?ì£¼ìš” ê¸°ëŠ¥???´í´ë³´ê² ?µë‹ˆ??

## 1. ?¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ ê°œìš”

?¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ?€ ?¤íŠ¸?Œí¬ ?´ì˜ ?¸ìŠ¤?? ?œë¹„?? ?¥ì¹˜ë¥??ë™?¼ë¡œ ?ì??˜ê³  ëª¨ë‹ˆ?°ë§?˜ëŠ” ??• ???©ë‹ˆ?? ì£¼ìš” ê¸°ëŠ¥?€ ?¤ìŒê³?ê°™ìŠµ?ˆë‹¤:

- ?¤íŠ¸?Œí¬ ?¸ìŠ¤???ì?
- ?œë¹„??ë°??¬íŠ¸ ?¤ìº”
- ?¤íŠ¸?Œí¬ ? í´ë¡œì? ë§¤í•‘
- ?¤íŠ¸?Œí¬ ë³€ê²??¬í•­ ê°ì?

## 2. ì£¼ìš” ê¸°ëŠ¥

### 2.1 ?¸ìŠ¤???ì?
```python
class NetworkDiscovery(Module):
    def __init__(self):
        super().__init__()
        self.discovered_hosts = {}
        self.active_services = {}
        self.network_topology = {}

    def discover_hosts(self, network):
        """
        ?¤íŠ¸?Œí¬ ?´ì˜ ?¸ìŠ¤?¸ë? ?ì??©ë‹ˆ??
        
        Args:
            network (str): ?ì????¤íŠ¸?Œí¬ ?€??
        """
        try:
            # ARP ?¤ìº”
            arp_hosts = self.arp_scan(network)
            
            # ICMP ?¤ìº”
            icmp_hosts = self.icmp_scan(network)
            
            # ?¸ìŠ¤???•ë³´ ?µí•©
            for host in set(arp_hosts + icmp_hosts):
                host_info = self.get_host_info(host)
                if host_info:
                    self.discovered_hosts[host] = host_info
                    
            self.update_network_topology()
        except Exception as e:
            self.logger.error(f"?¸ìŠ¤???ì? ?¤íŒ¨: {str(e)}")
```

### 2.2 ?œë¹„???ì?
```python
def discover_services(self, host):
    """
    ?¸ìŠ¤?¸ì˜ ?¤í–‰ ì¤‘ì¸ ?œë¹„?¤ë? ?ì??©ë‹ˆ??
    
    Args:
        host (str): ?€???¸ìŠ¤??IP
    
    Returns:
        dict: ?œë¹„???•ë³´
    """
    try:
        services = {
            'tcp': {},
            'udp': {},
            'metadata': {}
        }
        
        # TCP ?¬íŠ¸ ?¤ìº”
        tcp_ports = self.scan_tcp_ports(host)
        for port in tcp_ports:
            service = self.identify_service(host, port, 'tcp')
            if service:
                services['tcp'][port] = service
                
        # UDP ?¬íŠ¸ ?¤ìº”
        udp_ports = self.scan_udp_ports(host)
        for port in udp_ports:
            service = self.identify_service(host, port, 'udp')
            if service:
                services['udp'][port] = service
                
        # ?œë¹„??ë©”í??°ì´???˜ì§‘
        services['metadata'] = self.collect_service_metadata(host, services)
        
        return services
    except Exception as e:
        self.logger.error(f"?œë¹„???ì? ?¤íŒ¨: {str(e)}")
        return None
```

### 2.3 ?¤íŠ¸?Œí¬ ? í´ë¡œì? ë§¤í•‘
```python
def map_network_topology(self):
    """
    ?¤íŠ¸?Œí¬ ? í´ë¡œì?ë¥?ë§¤í•‘?©ë‹ˆ??
    """
    try:
        topology = {
            'hosts': {},
            'connections': [],
            'routers': [],
            'switches': []
        }
        
        # ?¸ìŠ¤???•ë³´ ?˜ì§‘
        for host, info in self.discovered_hosts.items():
            topology['hosts'][host] = {
                'ip': host,
                'mac': info.get('mac'),
                'os': info.get('os'),
                'services': info.get('services', {})
            }
            
        # ?¤íŠ¸?Œí¬ ?°ê²° ë¶„ì„
        for host in topology['hosts']:
            connections = self.analyze_host_connections(host)
            topology['connections'].extend(connections)
            
        # ?¼ìš°??ë°??¤ìœ„ì¹??ë³„
        topology['routers'] = self.identify_routers()
        topology['switches'] = self.identify_switches()
        
        self.network_topology = topology
        return topology
    except Exception as e:
        self.logger.error(f"?¤íŠ¸?Œí¬ ? í´ë¡œì? ë§¤í•‘ ?¤íŒ¨: {str(e)}")
        return None
```

## 3. ë³€ê²??¬í•­ ê°ì?

### 3.1 ?¸ìŠ¤??ë³€ê²?ê°ì?
```python
def detect_host_changes(self):
    """
    ?¤íŠ¸?Œí¬ ?¸ìŠ¤?¸ì˜ ë³€ê²??¬í•­??ê°ì??©ë‹ˆ??
    """
    try:
        changes = {
            'new_hosts': [],
            'removed_hosts': [],
            'modified_hosts': []
        }
        
        # ?„ì¬ ?¸ìŠ¤???¤ìº”
        current_hosts = set(self.discover_hosts(self.network))
        previous_hosts = set(self.discovered_hosts.keys())
        
        # ?ˆë¡œ???¸ìŠ¤??ê°ì?
        changes['new_hosts'] = list(current_hosts - previous_hosts)
        
        # ?œê±°???¸ìŠ¤??ê°ì?
        changes['removed_hosts'] = list(previous_hosts - current_hosts)
        
        # ë³€ê²½ëœ ?¸ìŠ¤??ê°ì?
        for host in current_hosts & previous_hosts:
            if self.has_host_changed(host):
                changes['modified_hosts'].append(host)
                
        return changes
    except Exception as e:
        self.logger.error(f"?¸ìŠ¤??ë³€ê²?ê°ì? ?¤íŒ¨: {str(e)}")
        return None
```

### 3.2 ?œë¹„??ë³€ê²?ê°ì?
```python
def detect_service_changes(self, host):
    """
    ?¸ìŠ¤?¸ì˜ ?œë¹„??ë³€ê²??¬í•­??ê°ì??©ë‹ˆ??
    
    Args:
        host (str): ?€???¸ìŠ¤??IP
    
    Returns:
        dict: ?œë¹„??ë³€ê²??¬í•­
    """
    try:
        changes = {
            'new_services': [],
            'removed_services': [],
            'modified_services': []
        }
        
        # ?„ì¬ ?œë¹„???¤ìº”
        current_services = self.discover_services(host)
        previous_services = self.active_services.get(host, {})
        
        # ?ˆë¡œ???œë¹„??ê°ì?
        for port, service in current_services.get('tcp', {}).items():
            if port not in previous_services.get('tcp', {}):
                changes['new_services'].append(service)
                
        # ?œê±°???œë¹„??ê°ì?
        for port, service in previous_services.get('tcp', {}).items():
            if port not in current_services.get('tcp', {}):
                changes['removed_services'].append(service)
                
        # ë³€ê²½ëœ ?œë¹„??ê°ì?
        for port, service in current_services.get('tcp', {}).items():
            if port in previous_services.get('tcp', {}) and \
               service != previous_services['tcp'][port]:
                changes['modified_services'].append(service)
                
        return changes
    except Exception as e:
        self.logger.error(f"?œë¹„??ë³€ê²?ê°ì? ?¤íŒ¨: {str(e)}")
        return None
```

## 4. ?°ì´??ê´€ë¦?

### 4.1 ?¤íŠ¸?Œí¬ ?•ë³´ ?€??
```python
def store_network_info(self, network_info):
    """
    ?¤íŠ¸?Œí¬ ?•ë³´ë¥??€?¥í•©?ˆë‹¤.
    
    Args:
        network_info (dict): ?€?¥í•  ?¤íŠ¸?Œí¬ ?•ë³´
    """
    try:
        self.db.set('network_info', json.dumps(network_info))
    except Exception as e:
        self.logger.error(f"?¤íŠ¸?Œí¬ ?•ë³´ ?€???¤íŒ¨: {str(e)}")
```

### 4.2 ?¤íŠ¸?Œí¬ ?•ë³´ ê²€??
```python
def search_network_info(self, query):
    """
    ?¤íŠ¸?Œí¬ ?•ë³´ë¥?ê²€?‰í•©?ˆë‹¤.
    
    Args:
        query (dict): ê²€??ì¿¼ë¦¬
    
    Returns:
        list: ê²€??ê²°ê³¼
    """
    try:
        results = []
        network_info = json.loads(self.db.get('network_info'))
        
        for host, info in network_info.get('hosts', {}).items():
            if self._matches_query(info, query):
                results.append(info)
                
        return results
    except Exception as e:
        self.logger.error(f"?¤íŠ¸?Œí¬ ?•ë³´ ê²€???¤íŒ¨: {str(e)}")
        return []
```

## 5. ?±ëŠ¥ ìµœì ??

### 5.1 ìºì‹±
```python
def cache_network_info(self, network_info, ttl=3600):
    """
    ?¤íŠ¸?Œí¬ ?•ë³´ë¥?ìºì‹œ?©ë‹ˆ??
    
    Args:
        network_info (dict): ìºì‹œ???¤íŠ¸?Œí¬ ?•ë³´
        ttl (int): ìºì‹œ ? íš¨ ?œê°„(ì´?
    """
    try:
        self.redis.setex('network_cache', ttl, json.dumps(network_info))
    except Exception as e:
        self.logger.error(f"?¤íŠ¸?Œí¬ ?•ë³´ ìºì‹± ?¤íŒ¨: {str(e)}")
```

## 6. ê²°ë¡ 

?¤íŠ¸?Œí¬ ?ì? ëª¨ë“ˆ?€ Slips???¤íŠ¸?Œí¬ ?ì‚° ê´€ë¦¬ì? ë³´ì•ˆ ëª¨ë‹ˆ?°ë§??ì¤‘ìš”????• ???©ë‹ˆ?? ì£¼ìš” ?¹ì§•?€ ?¤ìŒê³?ê°™ìŠµ?ˆë‹¤:

- ?ë™?”ëœ ?¤íŠ¸?Œí¬ ?ì‚° ?ì?
- ?¤ì‹œê°??œë¹„??ëª¨ë‹ˆ?°ë§
- ?¤íŠ¸?Œí¬ ? í´ë¡œì? ë§¤í•‘
- ë³€ê²??¬í•­ ê°ì? ë°??Œë¦¼

?´ëŸ¬??ê¸°ëŠ¥?¤ì? Slipsê°€ ?¤íŠ¸?Œí¬ ?˜ê²½???¨ê³¼?ìœ¼ë¡?ëª¨ë‹ˆ?°ë§?˜ê³  ë³´ì•ˆ ?„í˜‘???€?‘í•  ???ˆë„ë¡??„ì?ì¤ë‹ˆ?? 
---
layout: post
title: "Slips ?•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ ?ì„¸ ë¶„ì„"
date: 2024-04-08 12:30:00 +0900
categories: [network-analysis]
tags: [data-leakage, security, network-analysis]
---

Slips ?•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ ?ì„¸ ë¶„ì„

?•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ?€ Slips???°ì´??? ì¶œ ?ì??€ ë°©ì?ë¥??´ë‹¹?˜ëŠ” ?µì‹¬ ì»´í¬?ŒíŠ¸?…ë‹ˆ?? ??ê¸€?ì„œ???•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ??êµ¬í˜„ê³?ì£¼ìš” ê¸°ëŠ¥???´í´ë³´ê² ?µë‹ˆ??

## 1. ?•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ ê°œìš”

?•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ?€ ?¤íŠ¸?Œí¬ ?¸ë˜?½ì—??ë¯¼ê°???•ë³´??? ì¶œ???¤ì‹œê°„ìœ¼ë¡??ì??˜ëŠ” ??• ???©ë‹ˆ?? ì£¼ìš” ê¸°ëŠ¥?€ ?¤ìŒê³?ê°™ìŠµ?ˆë‹¤:

- ë¯¼ê° ?•ë³´ ?¨í„´ ?ì?
- ?°ì´??? ì¶œ ?œë„ ê°ì?
- ?¤ì‹œê°??Œë¦¼ ?ì„±
- ? ì¶œ ë°©ì? ?•ì±… ?ìš©

## 2. ì£¼ìš” ê¸°ëŠ¥

### 2.1 ë¯¼ê° ?•ë³´ ?ì?
```python
class LeakDetector(Module):
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = {}
        self.detection_rules = {}
        self.leak_attempts = {}

    def detect_sensitive_data(self, flow):
        """
        ?¤íŠ¸?Œí¬ ?¸ë˜?½ì—??ë¯¼ê°???•ë³´ë¥??ì??©ë‹ˆ??
        
        Args:
            flow (dict): ?¤íŠ¸?Œí¬ ?Œë¡œ???°ì´??
        """
        try:
            # ?¨í‚· ?°ì´??ì¶”ì¶œ
            packet_data = self.extract_packet_data(flow)
            
            # ë¯¼ê° ?•ë³´ ?¨í„´ ê²€??
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = self.search_pattern(packet_data, pattern)
                if matches:
                    self.handle_sensitive_data_detection(flow, pattern_name, matches)
                    
            # ?°ì´??? ì¶œ ?œë„ ê²€??
            if self.is_data_leak_attempt(flow):
                self.handle_leak_attempt(flow)
        except Exception as e:
            self.logger.error(f"ë¯¼ê° ?•ë³´ ?ì? ?¤íŒ¨: {str(e)}")
```

### 2.2 ?°ì´??? ì¶œ ?œë„ ê°ì?
```python
def is_data_leak_attempt(self, flow):
    """
    ?°ì´??? ì¶œ ?œë„ë¥?ê°ì??©ë‹ˆ??
    
    Args:
        flow (dict): ?¤íŠ¸?Œí¬ ?Œë¡œ???°ì´??
    
    Returns:
        bool: ? ì¶œ ?œë„ ?¬ë?
    """
    try:
        # ?€?©ëŸ‰ ?°ì´???„ì†¡ ?•ì¸
        if self.is_large_data_transfer(flow):
            return True
            
        # ë¹„ì •?ì ???„ë¡œ? ì½œ ?¬ìš© ?•ì¸
        if self.is_suspicious_protocol(flow):
            return True
            
        # ?”í˜¸?”ë˜ì§€ ?Šì? ë¯¼ê° ?°ì´???„ì†¡ ?•ì¸
        if self.is_unencrypted_sensitive_data(flow):
            return True
            
        # ë¹„ì •?ì ???œê°„?€ ?µì‹  ?•ì¸
        if self.is_off_hours_communication(flow):
            return True
            
        return False
    except Exception as e:
        self.logger.error(f"? ì¶œ ?œë„ ê°ì? ?¤íŒ¨: {str(e)}")
        return False
```

### 2.3 ? ì¶œ ë°©ì? ?•ì±… ?ìš©
```python
def apply_prevention_policy(self, flow):
    """
    ?°ì´??? ì¶œ ë°©ì? ?•ì±…???ìš©?©ë‹ˆ??
    
    Args:
        flow (dict): ?¤íŠ¸?Œí¬ ?Œë¡œ???°ì´??
    
    Returns:
        dict: ?•ì±… ?ìš© ê²°ê³¼
    """
    try:
        result = {
            'blocked': False,
            'action_taken': None,
            'reason': None
        }
        
        # ? ì¶œ ?œë„ ?•ì¸
        if self.is_data_leak_attempt(flow):
            # ?•ì±…???°ë¥¸ ì¡°ì¹˜
            if self.should_block_flow(flow):
                self.block_flow(flow)
                result['blocked'] = True
                result['action_taken'] = 'block'
                result['reason'] = 'data_leak_attempt'
            else:
                self.log_flow(flow)
                result['action_taken'] = 'log'
                result['reason'] = 'suspicious_activity'
                
        return result
    except Exception as e:
        self.logger.error(f"ë°©ì? ?•ì±… ?ìš© ?¤íŒ¨: {str(e)}")
        return None
```

## 3. ?¨í„´ ê´€ë¦?

### 3.1 ?¨í„´ ?…ë°?´íŠ¸
```python
def update_sensitive_patterns(self, new_patterns):
    """
    ë¯¼ê° ?•ë³´ ?¨í„´???…ë°?´íŠ¸?©ë‹ˆ??
    
    Args:
        new_patterns (dict): ?ˆë¡œ???¨í„´
    """
    try:
        for pattern_name, pattern in new_patterns.items():
            # ?¨í„´ ? íš¨??ê²€??
            if self.validate_pattern(pattern):
                self.sensitive_patterns[pattern_name] = pattern
                
        # ?¨í„´ ?€??
        self.store_patterns()
    except Exception as e:
        self.logger.error(f"?¨í„´ ?…ë°?´íŠ¸ ?¤íŒ¨: {str(e)}")
```

### 3.2 ?¨í„´ ê²€??
```python
def search_pattern(self, data, pattern):
    """
    ?°ì´?°ì—???¨í„´??ê²€?‰í•©?ˆë‹¤.
    
    Args:
        data (str): ê²€?‰í•  ?°ì´??
        pattern (str): ê²€?‰í•  ?¨í„´
    
    Returns:
        list: ê²€??ê²°ê³¼
    """
    try:
        matches = []
        
        # ?•ê·œ???¨í„´ ê²€??
        if isinstance(pattern, str):
            matches = re.finditer(pattern, data)
        # ë¨¸ì‹ ?¬ë‹ ê¸°ë°˜ ?¨í„´ ê²€??
        elif isinstance(pattern, dict):
            matches = self.ml_pattern_search(data, pattern)
            
        return [match.group() for match in matches]
    except Exception as e:
        self.logger.error(f"?¨í„´ ê²€???¤íŒ¨: {str(e)}")
        return []
```

## 4. ?°ì´??ê´€ë¦?

### 4.1 ?ì? ê²°ê³¼ ?€??
```python
def store_detection_results(self, results):
    """
    ?ì? ê²°ê³¼ë¥??€?¥í•©?ˆë‹¤.
    
    Args:
        results (dict): ?€?¥í•  ?ì? ê²°ê³¼
    """
    try:
        self.db.set('leak_detection_results', json.dumps(results))
    except Exception as e:
        self.logger.error(f"?ì? ê²°ê³¼ ?€???¤íŒ¨: {str(e)}")
```

### 4.2 ?ì? ê²°ê³¼ ê²€??
```python
def search_detection_results(self, query):
    """
    ?ì? ê²°ê³¼ë¥?ê²€?‰í•©?ˆë‹¤.
    
    Args:
        query (dict): ê²€??ì¿¼ë¦¬
    
    Returns:
        list: ê²€??ê²°ê³¼
    """
    try:
        results = []
        detection_results = json.loads(self.db.get('leak_detection_results'))
        
        for flow_id, result in detection_results.items():
            if self._matches_query(result, query):
                results.append(result)
                
        return results
    except Exception as e:
        self.logger.error(f"?ì? ê²°ê³¼ ê²€???¤íŒ¨: {str(e)}")
        return []
```

## 5. ?±ëŠ¥ ìµœì ??

### 5.1 ?¨í„´ ì»´íŒŒ??
```python
def compile_patterns(self):
    """
    ?•ê·œ???¨í„´??ì»´íŒŒ?¼í•©?ˆë‹¤.
    """
    try:
        for pattern_name, pattern in self.sensitive_patterns.items():
            if isinstance(pattern, str):
                self.sensitive_patterns[pattern_name] = re.compile(pattern)
    except Exception as e:
        self.logger.error(f"?¨í„´ ì»´íŒŒ???¤íŒ¨: {str(e)}")
```

### 5.2 ìºì‹±
```python
def cache_detection_results(self, results, ttl=3600):
    """
    ?ì? ê²°ê³¼ë¥?ìºì‹œ?©ë‹ˆ??
    
    Args:
        results (dict): ìºì‹œ???ì? ê²°ê³¼
        ttl (int): ìºì‹œ ? íš¨ ?œê°„(ì´?
    """
    try:
        self.redis.setex('leak_detection_cache', ttl, json.dumps(results))
    except Exception as e:
        self.logger.error(f"?ì? ê²°ê³¼ ìºì‹± ?¤íŒ¨: {str(e)}")
```

## 6. ê²°ë¡ 

?•ë³´ ? ì¶œ ?ì? ëª¨ë“ˆ?€ Slips???°ì´??ë³´ì•ˆ??ê°•í™”?˜ëŠ” ì¤‘ìš”??ì»´í¬?ŒíŠ¸?…ë‹ˆ?? ì£¼ìš” ?¹ì§•?€ ?¤ìŒê³?ê°™ìŠµ?ˆë‹¤:

- ?¤ì‹œê°?ë¯¼ê° ?•ë³´ ?ì?
- ?°ì´??? ì¶œ ?œë„ ê°ì?
- ? ì¶œ ë°©ì? ?•ì±… ?ìš©
- ?¨ìœ¨?ì¸ ?¨í„´ ê´€ë¦?

?´ëŸ¬??ê¸°ëŠ¥?¤ì? Slipsê°€ ?°ì´??? ì¶œ???¨ê³¼?ìœ¼ë¡??ì??˜ê³  ë°©ì??????ˆë„ë¡??„ì?ì¤ë‹ˆ?? 
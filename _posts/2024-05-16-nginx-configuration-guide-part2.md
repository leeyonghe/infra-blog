---
layout: post
title: "Nginx 설정 가이드 Part 2 - Virtual Host, 로드 밸런싱, SSL/TLS"
date: 2024-05-16
categories: [Infrastructure, Nginx, Configuration]
tags: [nginx, virtual-host, load-balancing, ssl, tls, upstream]
---

# Nginx 설정 가이드 Part 2 - Virtual Host, 로드 밸런싱, SSL/TLS

이번 포스트에서는 Nginx의 고급 설정 기능들을 다루겠습니다. Virtual Host 설정부터 로드 밸런싱, SSL/TLS 보안 설정까지 실무에서 필요한 핵심 기능들을 학습합니다.

<!--more-->

## 목차
1. [Virtual Host 설정](#virtual-host-설정)
2. [로드 밸런싱](#로드-밸런싱)
3. [업스트림 서버 설정](#업스트림-서버-설정)
4. [SSL/TLS 설정](#ssltls-설정)
5. [HTTP/2 설정](#http2-설정)
6. [실습 예제](#실습-예제)

## Virtual Host 설정

Virtual Host는 하나의 서버에서 여러 웹사이트를 호스팅할 수 있게 해주는 기능입니다.

### 이름 기반 Virtual Host
```nginx
# /etc/nginx/sites-available/site1.com
server {
    listen 80;
    server_name site1.com www.site1.com;
    root /var/www/site1;
    index index.html index.php;
    
    access_log /var/log/nginx/site1.access.log;
    error_log /var/log/nginx/site1.error.log;
    
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
}

# /etc/nginx/sites-available/site2.com
server {
    listen 80;
    server_name site2.com www.site2.com;
    root /var/www/site2;
    index index.html index.php;
    
    access_log /var/log/nginx/site2.access.log;
    error_log /var/log/nginx/site2.error.log;
    
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
}
```

### IP 기반 Virtual Host
```nginx
# 첫 번째 IP
server {
    listen 192.168.1.10:80;
    server_name _;
    root /var/www/site1;
    index index.html;
}

# 두 번째 IP
server {
    listen 192.168.1.11:80;
    server_name _;
    root /var/www/site2;
    index index.html;
}
```

### 포트 기반 Virtual Host
```nginx
# 8080 포트
server {
    listen 8080;
    server_name example.com;
    root /var/www/app1;
    index index.html;
}

# 8081 포트
server {
    listen 8081;
    server_name example.com;
    root /var/www/app2;
    index index.html;
}
```

### 와일드카드 서버 네임
```nginx
server {
    listen 80;
    server_name *.example.com;
    root /var/www/subdomains;
    
    # 서브도메인에 따른 동적 루트 설정
    location / {
        set $subdomain "";
        if ($host ~* "^([^.]+)\.example\.com$") {
            set $subdomain $1;
        }
        root /var/www/subdomains/$subdomain;
        try_files $uri $uri/ /index.html;
    }
}
```

## 로드 밸런싱

Nginx는 여러 백엔드 서버로 트래픽을 분산하는 로드 밸런싱 기능을 제공합니다.

### 기본 로드 밸런싱
```nginx
# 업스트림 정의
upstream backend {
    server 192.168.1.100:8080;
    server 192.168.1.101:8080;
    server 192.168.1.102:8080;
}

server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 로드 밸런싱 알고리즘

#### Round Robin (기본값)
```nginx
upstream backend {
    server server1.example.com;
    server server2.example.com;
    server server3.example.com;
}
```

#### Least Connections
```nginx
upstream backend {
    least_conn;
    server server1.example.com;
    server server2.example.com;
    server server3.example.com;
}
```

#### IP Hash
```nginx
upstream backend {
    ip_hash;
    server server1.example.com;
    server server2.example.com;
    server server3.example.com;
}
```

#### 가중치 기반
```nginx
upstream backend {
    server server1.example.com weight=3;
    server server2.example.com weight=2;
    server server3.example.com weight=1;
}
```

### 서버 상태 제어
```nginx
upstream backend {
    server server1.example.com weight=5;
    server server2.example.com weight=2;
    server server3.example.com backup;        # 백업 서버
    server server4.example.com down;          # 일시적 비활성화
    server server5.example.com max_fails=3 fail_timeout=30s;
}
```

## 업스트림 서버 설정

### 헬스 체크 설정
```nginx
upstream backend {
    server backend1.example.com:8080 max_fails=2 fail_timeout=30s;
    server backend2.example.com:8080 max_fails=2 fail_timeout=30s;
    server backend3.example.com:8080 backup;
    
    # 연결 유지
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://backend;
        
        # 프록시 헤더
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 타임아웃 설정
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # 버퍼링 설정
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # 연결 유지
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 세션 지속성 (Sticky Sessions)
```nginx
# IP 해시 방식
upstream backend {
    ip_hash;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}

# 쿠키 기반 (nginx-plus)
upstream backend {
    sticky cookie srv_id expires=1h domain=.example.com path=/;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}
```

## SSL/TLS 설정

### 기본 SSL 설정
```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name example.com www.example.com;
    
    # SSL 인증서
    ssl_certificate /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;
    
    # SSL 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    
    # SSL 세션 캐시
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
    
    # 보안 헤더
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    root /var/www/html;
    index index.html;
}

# HTTP to HTTPS 리다이렉트
server {
    listen 80;
    listen [::]:80;
    server_name example.com www.example.com;
    return 301 https://$server_name$request_uri;
}
```

### Let's Encrypt 설정
```nginx
# HTTP 챌린지용 설정
server {
    listen 80;
    server_name example.com www.example.com;
    
    # Let's Encrypt 검증
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    # 나머지는 HTTPS로 리다이렉트
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS 서버
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;
    
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    
    # SSL 최적 설정
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
    root /var/www/html;
    index index.html;
}
```

### SSL 보안 강화
```nginx
# /etc/nginx/snippets/ssl.conf
# Modern configuration
ssl_protocols TLSv1.3;
ssl_prefer_server_ciphers off;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;

# SSL 세션 최적화
ssl_session_cache shared:SSL:50m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# DH 파라미터
ssl_dhparam /etc/nginx/dhparam.pem;

# OCSP 설정
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# 보안 헤더
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
```

## HTTP/2 설정

### HTTP/2 활성화
```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name example.com;
    
    # HTTP/2 Push (선택적)
    location = /index.html {
        http2_push /css/main.css;
        http2_push /js/main.js;
        http2_push /images/logo.png;
    }
    
    # 정적 파일 최적화
    location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
    }
}
```

## 실습 예제

### 1. 멀티 도메인 WordPress 호스팅
```bash
# 디렉토리 구조 생성
sudo mkdir -p /var/www/{site1,site2}/public_html
sudo mkdir -p /var/log/nginx/{site1,site2}

# 소유권 설정
sudo chown -R www-data:www-data /var/www/

# Site1 설정
sudo tee /etc/nginx/sites-available/site1.com > /dev/null << 'EOF'
server {
    listen 80;
    server_name site1.com www.site1.com;
    root /var/www/site1/public_html;
    index index.php index.html;
    
    access_log /var/log/nginx/site1/access.log;
    error_log /var/log/nginx/site1/error.log;
    
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
    
    location ~ \.php$ {
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }
    
    location ~ /\.ht {
        deny all;
    }
    
    # WordPress 보안
    location ~ /wp-config.php {
        deny all;
    }
}
EOF

# Site2 설정
sudo tee /etc/nginx/sites-available/site2.com > /dev/null << 'EOF'
server {
    listen 80;
    server_name site2.com www.site2.com;
    root /var/www/site2/public_html;
    index index.php index.html;
    
    access_log /var/log/nginx/site2/access.log;
    error_log /var/log/nginx/site2/error.log;
    
    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }
    
    location ~ \.php$ {
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }
}
EOF

# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/site1.com /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/site2.com /etc/nginx/sites-enabled/
```

### 2. API 게이트웨이 설정
```nginx
# /etc/nginx/sites-available/api-gateway
upstream auth_service {
    server 127.0.0.1:3001;
    server 127.0.0.1:3002 backup;
}

upstream user_service {
    least_conn;
    server 127.0.0.1:3011;
    server 127.0.0.1:3012;
}

upstream product_service {
    server 127.0.0.1:3021 weight=3;
    server 127.0.0.1:3022 weight=2;
}

server {
    listen 80;
    server_name api.example.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # 공통 헤더 설정
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # 인증 서비스
    location /auth/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://auth_service/;
    }
    
    # 사용자 서비스
    location /users/ {
        proxy_pass http://user_service/;
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # 상품 서비스
    location /products/ {
        proxy_pass http://product_service/;
        proxy_cache my_cache;
        proxy_cache_valid 200 5m;
    }
    
    # 헬스 체크
    location /health {
        access_log off;
        return 200 "OK";
        add_header Content-Type text/plain;
    }
}
```

### 3. SSL 자동화 스크립트
```bash
#!/bin/bash
# /usr/local/bin/ssl-setup.sh

DOMAIN=$1
EMAIL=$2

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    echo "Usage: $0 <domain> <email>"
    exit 1
fi

# Let's Encrypt 인증서 발급
certbot certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN \
    -d www.$DOMAIN

# Nginx 설정 업데이트
cat > /etc/nginx/sites-available/$DOMAIN << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;
    
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    
    include /etc/nginx/snippets/ssl.conf;
    
    root /var/www/$DOMAIN;
    index index.html;
    
    location / {
        try_files \$uri \$uri/ =404;
    }
}
EOF

# 사이트 활성화
ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/

# Nginx 테스트 및 리로드
nginx -t && systemctl reload nginx

echo "SSL setup completed for $DOMAIN"
```

## 문제 해결

### 일반적인 문제들

#### 1. 업스트림 서버 연결 실패
```nginx
# 에러 로그 확인
tail -f /var/log/nginx/error.log

# 업스트림 상태 확인
upstream backend {
    server backend1.example.com:8080 max_fails=3 fail_timeout=30s;
    # 백업 서버 추가
    server backup.example.com:8080 backup;
}
```

#### 2. SSL 인증서 문제
```bash
# 인증서 유효성 확인
openssl x509 -in /path/to/cert.pem -text -noout

# SSL 설정 테스트
nginx -t

# SSL Labs 테스트
# https://www.ssllabs.com/ssltest/
```

#### 3. 성능 문제
```nginx
# 워커 프로세스 조정
worker_processes auto;
worker_connections 1024;

# 캐시 설정
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m;
```

## 다음 단계

다음 포스트에서는 다음 내용들을 다루겠습니다:

- 리버스 프록시 고급 설정
- 캐싱 전략
- 압축 최적화
- 정적 파일 서빙 최적화

## 참고 자료

- [Nginx Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
- [SSL Configuration Guide](https://ssl-config.mozilla.org/)
- [Let's Encrypt Nginx](https://certbot.eff.org/instructions?ws=nginx)
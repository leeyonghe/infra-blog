---
layout: post
title: "Nginx 고급 설정 가이드 Part 3 - 리버스 프록시, 캐싱, 압축 최적화"
date: 2024-05-17
categories: [Infrastructure, Nginx, Performance]
tags: [nginx, reverse-proxy, caching, compression, optimization]
---

# Nginx 고급 설정 가이드 Part 3 - 리버스 프록시, 캐싱, 압축 최적화

이번 포스트에서는 Nginx의 성능 최적화를 위한 고급 기능들을 다룹니다. 리버스 프록시 설정부터 캐싱 전략, 압축 최적화까지 웹 서버의 성능을 극대화하는 방법을 학습합니다.

<!--more-->

## 목차
1. [리버스 프록시 고급 설정](#리버스-프록시-고급-설정)
2. [캐싱 전략](#캐싱-전략)
3. [압축 최적화](#압축-최적화)
4. [정적 파일 서빙 최적화](#정적-파일-서빙-최적화)
5. [리다이렉션과 URL 재작성](#리다이렉션과-url-재작성)
6. [성능 튜닝](#성능-튜닝)

## 리버스 프록시 고급 설정

### 기본 프록시 설정
```nginx
server {
    listen 80;
    server_name app.example.com;
    
    location / {
        proxy_pass http://backend_app;
        
        # 필수 헤더 설정
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $server_name;
        
        # 타임아웃 설정
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # 버퍼링 설정
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
        proxy_temp_file_write_size 256k;
        
        # HTTP 버전 및 연결 유지
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### WebSocket 프록시
```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}

upstream websocket {
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}

server {
    listen 80;
    server_name ws.example.com;
    
    location /ws/ {
        proxy_pass http://websocket;
        
        # WebSocket 헤더
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # WebSocket 타임아웃
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        
        # 버퍼링 비활성화
        proxy_buffering off;
    }
}
```

### 마이크로서비스 프록시
```nginx
# API 라우팅
server {
    listen 80;
    server_name api.example.com;
    
    # 인증 서비스
    location /auth/ {
        proxy_pass http://auth-service/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # 인증 응답 캐싱
        proxy_cache auth_cache;
        proxy_cache_valid 200 5m;
        proxy_cache_key "$scheme$request_method$host$request_uri";
    }
    
    # 사용자 서비스
    location /users/ {
        auth_request /auth/verify;
        proxy_pass http://user-service/;
        proxy_set_header Host $host;
        proxy_set_header X-User-ID $upstream_http_user_id;
    }
    
    # 주문 서비스
    location /orders/ {
        auth_request /auth/verify;
        proxy_pass http://order-service/;
        
        # 주문 생성은 캐시하지 않음
        proxy_cache_bypass $request_method ~* ^(POST|PUT|DELETE)$;
    }
    
    # 내부 인증 확인
    location = /auth/verify {
        internal;
        proxy_pass http://auth-service/verify;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }
}
```

## 캐싱 전략

### 프록시 캐시 설정
```nginx
http {
    # 캐시 경로 및 설정
    proxy_cache_path /var/cache/nginx/proxy 
                     levels=1:2 
                     keys_zone=my_cache:10m 
                     max_size=1g 
                     inactive=60m 
                     use_temp_path=off;
    
    # FastCGI 캐시 (PHP용)
    fastcgi_cache_path /var/cache/nginx/fastcgi 
                       levels=1:2 
                       keys_zone=php_cache:10m 
                       max_size=1g 
                       inactive=60m;
}

server {
    listen 80;
    server_name example.com;
    
    # 정적 파일 캐싱
    location ~* \.(jpg|jpeg|png|gif|ico|css|js|pdf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        
        # 조건부 캐싱
        location ~* \.(css|js)$ {
            expires 1M;
            add_header Cache-Control "public";
            
            # 압축 설정
            gzip_static on;
        }
    }
    
    # API 응답 캐싱
    location /api/ {
        proxy_pass http://backend;
        proxy_cache my_cache;
        proxy_cache_revalidate on;
        proxy_cache_min_uses 1;
        proxy_cache_use_stale error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_cache_background_update on;
        proxy_cache_lock on;
        
        # 캐시 키 설정
        proxy_cache_key "$scheme$request_method$host$request_uri$is_args$args";
        
        # 캐시 유효 시간
        proxy_cache_valid 200 302 10m;
        proxy_cache_valid 301 1h;
        proxy_cache_valid any 1m;
        
        # 캐시 우회 조건
        proxy_cache_bypass $cookie_nocache $arg_nocache $arg_comment;
        proxy_cache_bypass $http_pragma $http_authorization;
        
        # 캐시 상태 헤더
        add_header X-Cache-Status $upstream_cache_status;
    }
    
    # PHP 캐싱
    location ~ \.php$ {
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
        
        fastcgi_cache php_cache;
        fastcgi_cache_valid 200 301 302 1h;
        fastcgi_cache_valid 404 10m;
        fastcgi_cache_min_uses 2;
        fastcgi_cache_use_stale error timeout invalid_header http_500;
        
        # 동적 콘텐츠 캐시 우회
        fastcgi_cache_bypass $cookie_PHPSESSID;
        fastcgi_cache_bypass $http_x_requested_with;
        
        add_header X-FastCGI-Cache $upstream_cache_status;
    }
}
```

### 캐시 제거 및 관리
```nginx
# 캐시 퍼지 위치 (특정 모듈 필요)
location ~ /purge(/.*) {
    allow 127.0.0.1;
    allow 192.168.1.0/24;
    deny all;
    
    proxy_cache_purge my_cache "$scheme$request_method$host$1";
    return 200 "Purged\n";
}

# 캐시 상태 확인
location /cache-status {
    allow 127.0.0.1;
    deny all;
    
    access_log off;
    return 200 "Cache Status: $upstream_cache_status\n";
    add_header Content-Type text/plain;
}
```

## 압축 최적화

### Gzip 압축
```nginx
http {
    # Gzip 설정
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        application/atom+xml
        application/geo+json
        application/javascript
        application/x-javascript
        application/json
        application/ld+json
        application/manifest+json
        application/rdf+xml
        application/rss+xml
        application/xhtml+xml
        application/xml
        font/eot
        font/otf
        font/ttf
        image/svg+xml
        text/css
        text/javascript
        text/plain
        text/xml;
    
    # Gzip static
    gzip_static on;
    
    # Brotli 압축 (모듈 필요)
    brotli on;
    brotli_comp_level 6;
    brotli_types
        application/atom+xml
        application/javascript
        application/json
        application/rss+xml
        application/vnd.ms-fontobject
        application/x-font-opentype
        application/x-font-truetype
        application/x-font-ttf
        application/x-javascript
        application/xhtml+xml
        application/xml
        font/eot
        font/opentype
        font/otf
        font/truetype
        image/svg+xml
        image/vnd.microsoft.icon
        image/x-icon
        image/x-win-bitmap
        text/css
        text/javascript
        text/plain
        text/xml;
}
```

### 이미지 최적화
```nginx
server {
    listen 80;
    server_name cdn.example.com;
    root /var/www/images;
    
    # 이미지 변환 (ImageFilter 모듈)
    location ~ ^/resize/(\d+)x(\d+)/(.*) {
        set $width $1;
        set $height $2;
        set $image_path $3;
        
        image_filter resize $width $height;
        image_filter_jpeg_quality 85;
        image_filter_webp_quality 85;
        image_filter_buffer 5M;
        
        try_files /$image_path =404;
        
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # WebP 변환 지원
    location ~* \.(png|jpe?g)$ {
        add_header Vary Accept;
        try_files $uri$webp_suffix $uri =404;
        
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # WebP 확장자 매핑
    map $http_accept $webp_suffix {
        default "";
        "~*webp" ".webp";
    }
}
```

## 정적 파일 서빙 최적화

### 정적 파일 설정
```nginx
server {
    listen 80;
    server_name static.example.com;
    root /var/www/static;
    
    # 보안 설정
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # 파일 타입별 캐시 설정
    location ~* \.(jpg|jpeg|png|gif|webp|avif)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        
        # 조건부 요청 처리
        if_modified_since exact;
        etag on;
    }
    
    location ~* \.(css|js)$ {
        expires 1M;
        add_header Cache-Control "public";
        
        # 소스맵 처리
        location ~* \.map$ {
            expires 1d;
            add_header Cache-Control "public, no-transform";
        }
    }
    
    location ~* \.(woff2?|eot|ttf|otf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Access-Control-Allow-Origin "*";
    }
    
    location ~* \.(pdf|doc|docx|zip|tar\.gz)$ {
        expires 1w;
        add_header Cache-Control "public, no-transform";
        add_header Content-Disposition "attachment";
    }
    
    # 대용량 파일 스트리밍
    location /downloads/ {
        sendfile on;
        tcp_nopush on;
        tcp_nodelay on;
        
        # 대역폭 제한
        limit_rate 1m;
        
        # 범위 요청 지원
        add_header Accept-Ranges bytes;
    }
}
```

### CDN 통합
```nginx
# CDN 오리진 서버 설정
server {
    listen 80;
    server_name origin.example.com;
    
    # CDN 전용 헤더
    add_header X-Origin-Server $hostname;
    add_header Cache-Control "public, max-age=31536000";
    
    # 캐시 검증 헤더
    etag on;
    if_modified_since exact;
    
    location /assets/ {
        root /var/www;
        
        # CDN 캐시 제어
        expires 1y;
        add_header Cache-Control "public, immutable";
        
        # 압축 활성화
        gzip_static on;
        
        # CORS 헤더
        add_header Access-Control-Allow-Origin "*";
        add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS";
    }
}
```

## 리다이렉션과 URL 재작성

### URL 리라이팅
```nginx
server {
    listen 80;
    server_name example.com;
    
    # 기본 리라이팅 규칙
    location / {
        # SEO 친화적 URL
        rewrite ^/product/([0-9]+)/?$ /product.php?id=$1 last;
        rewrite ^/category/([a-zA-Z0-9-]+)/?$ /category.php?slug=$1 last;
        
        # API 버전 라우팅
        rewrite ^/api/v1/(.*)$ /api/v1.php/$1 last;
        rewrite ^/api/v2/(.*)$ /api/v2.php/$1 last;
        
        try_files $uri $uri/ @fallback;
    }
    
    # 조건부 리다이렉트
    if ($host != 'www.example.com') {
        return 301 https://www.example.com$request_uri;
    }
    
    # 레거시 URL 처리
    location /old-blog {
        return 301 https://blog.example.com$request_uri;
    }
    
    # 모바일 리다이렉트
    if ($http_user_agent ~* "(Mobile|Android|iPhone)") {
        return 302 https://m.example.com$request_uri;
    }
    
    # 폴백 처리
    location @fallback {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 정규식 기반 라우팅
```nginx
# 복잡한 URL 패턴
location ~ ^/user/([0-9]+)/posts/([0-9]+)/?$ {
    set $user_id $1;
    set $post_id $2;
    
    proxy_pass http://backend/api/users/$user_id/posts/$post_id;
    proxy_set_header X-User-ID $user_id;
    proxy_set_header X-Post-ID $post_id;
}

# 다국어 지원
location ~ ^/([a-z]{2})/(.*)$ {
    set $lang $1;
    set $path $2;
    
    proxy_pass http://backend/$path;
    proxy_set_header Accept-Language $lang;
    proxy_set_header X-Language $lang;
}
```

## 성능 튜닝

### 워커 프로세스 최적화
```nginx
# nginx.conf
user www-data;
worker_processes auto;  # CPU 코어 수에 맞게 자동 설정
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 8192;
    use epoll;  # Linux에서 최적
    multi_accept on;
    accept_mutex off;
}

http {
    # 연결 최적화
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    
    # 타임아웃 최적화
    keepalive_timeout 65;
    keepalive_requests 1000;
    client_header_timeout 60;
    client_body_timeout 60;
    send_timeout 60;
    
    # 버퍼 최적화
    client_header_buffer_size 4k;
    large_client_header_buffers 4 8k;
    client_body_buffer_size 128k;
    client_max_body_size 100m;
    
    # 해시 테이블 최적화
    server_names_hash_bucket_size 128;
    server_names_hash_max_size 2048;
    types_hash_max_size 2048;
    variables_hash_max_size 2048;
}
```

### 메모리 최적화
```nginx
# 프록시 버퍼 최적화
proxy_buffering on;
proxy_buffer_size 128k;
proxy_buffers 4 256k;
proxy_busy_buffers_size 256k;
proxy_temp_file_write_size 256k;
proxy_max_temp_file_size 1024m;

# FastCGI 버퍼 최적화
fastcgi_buffering on;
fastcgi_buffer_size 128k;
fastcgi_buffers 4 256k;
fastcgi_busy_buffers_size 256k;

# 캐시 최적화
proxy_cache_path /var/cache/nginx/proxy 
                 levels=1:2 
                 keys_zone=cache:100m 
                 max_size=10g 
                 inactive=60m 
                 use_temp_path=off 
                 loader_threshold=300 
                 loader_files=200;
```

### 모니터링 설정
```nginx
# 상태 페이지
location /nginx_status {
    stub_status on;
    allow 127.0.0.1;
    allow 192.168.1.0/24;
    deny all;
    
    access_log off;
}

# 실시간 액세스 로그
location /access_log {
    allow 127.0.0.1;
    deny all;
    
    content_by_lua_block {
        local file = io.open("/var/log/nginx/access.log", "r")
        if file then
            file:seek("end", -1024)
            local content = file:read("*a")
            file:close()
            ngx.say(content)
        end
    }
}
```

## 실습 예제

### 고성능 정적 파일 서버
```nginx
# /etc/nginx/sites-available/static-server
server {
    listen 80;
    server_name static.example.com;
    root /var/www/static;
    
    # 로그 최소화
    access_log /var/log/nginx/static.log main buffer=32k flush=5m;
    error_log /var/log/nginx/static.error.log warn;
    
    # 보안 헤더
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # 메인 정적 파일
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
        
        # 압축 최적화
        gzip_static on;
        brotli_static on;
        
        # 성능 최적화
        sendfile on;
        tcp_nopush on;
        tcp_nodelay on;
        
        # 조건부 요청
        if_modified_since exact;
        etag on;
    }
    
    # API 프록시
    location /api/ {
        proxy_pass http://api_backend/;
        
        # 캐시 설정
        proxy_cache api_cache;
        proxy_cache_valid 200 5m;
        proxy_cache_valid 404 1m;
        
        # 성능 최적화
        proxy_buffering on;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        add_header X-Cache-Status $upstream_cache_status;
    }
}
```

## 다음 단계

다음 포스트에서는 Nginx 운영 관리에 대해 다루겠습니다:

- 프로세스 관리 및 모니터링
- 로그 관리 및 분석
- 백업 및 복구 전략
- 성능 튜닝 가이드

## 참고 자료

- [Nginx Performance Tuning](https://www.nginx.com/blog/tuning-nginx/)
- [Nginx Caching Guide](https://www.nginx.com/blog/nginx-caching-guide/)
- [HTTP/2 Server Push](https://www.nginx.com/blog/nginx-1-13-9-http2-server-push/)
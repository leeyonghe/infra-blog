---
layout: post
title: "Nginx 기본 설정 가이드 Part 1 - 설치부터 기본 서버 블록까지"
date: 2024-05-15
categories: [Infrastructure, Nginx, Configuration]
tags: [nginx, web-server, reverse-proxy, installation, configuration]
---

# Nginx 기본 설정 가이드 Part 1

Nginx는 높은 성능과 안정성으로 널리 사용되는 웹 서버이자 리버스 프록시입니다. 이 시리즈에서는 Nginx의 설치부터 고급 설정까지 단계별로 다루겠습니다.

<!--more-->

## 목차
1. [Nginx 소개](#nginx-소개)
2. [설치](#설치)
3. [디렉토리 구조](#디렉토리-구조)
4. [설정 파일 구조](#설정-파일-구조)
5. [기본 서버 블록 설정](#기본-서버-블록-설정)
6. [설정 테스트 및 서비스 관리](#설정-테스트-및-서비스-관리)

## Nginx 소개

### 특징
- **고성능**: 이벤트 기반 비동기 아키텍처
- **낮은 메모리 사용량**: 효율적인 리소스 관리
- **높은 동시성**: 수천 개의 연결 동시 처리
- **모듈식 구조**: 필요한 기능만 선택적 로드

### 주요 용도
- 정적 파일 서빙
- 리버스 프록시
- 로드 밸런서
- HTTP 캐시
- API 게이트웨이

## 설치

### Ubuntu/Debian
```bash
# 패키지 업데이트
sudo apt update

# Nginx 설치
sudo apt install nginx

# 서비스 시작 및 자동 시작 설정
sudo systemctl start nginx
sudo systemctl enable nginx
```

### CentOS/RHEL
```bash
# EPEL 저장소 설치 (CentOS 7)
sudo yum install epel-release

# Nginx 설치
sudo yum install nginx

# 서비스 시작 및 자동 시작 설정
sudo systemctl start nginx
sudo systemctl enable nginx

# 방화벽 설정
sudo firewall-cmd --permanent --zone=public --add-service=http
sudo firewall-cmd --permanent --zone=public --add-service=https
sudo firewall-cmd --reload
```

### 소스 컴파일 설치
```bash
# 의존성 설치
sudo apt install build-essential libpcre3-dev libssl-dev zlib1g-dev

# 소스 다운로드
wget http://nginx.org/download/nginx-1.24.0.tar.gz
tar -zxvf nginx-1.24.0.tar.gz
cd nginx-1.24.0

# 컴파일 옵션 설정
./configure \
    --prefix=/etc/nginx \
    --sbin-path=/usr/sbin/nginx \
    --conf-path=/etc/nginx/nginx.conf \
    --error-log-path=/var/log/nginx/error.log \
    --http-log-path=/var/log/nginx/access.log \
    --pid-path=/var/run/nginx.pid \
    --lock-path=/var/run/nginx.lock \
    --with-http_ssl_module \
    --with-http_v2_module \
    --with-http_realip_module \
    --with-http_gzip_static_module

# 컴파일 및 설치
make
sudo make install
```

## 디렉토리 구조

### 주요 디렉토리
```
/etc/nginx/                 # 설정 파일 디렉토리
├── nginx.conf             # 메인 설정 파일
├── mime.types             # MIME 타입 정의
├── conf.d/                # 추가 설정 파일들
├── sites-available/       # 사용 가능한 사이트 설정 (Ubuntu/Debian)
├── sites-enabled/         # 활성화된 사이트 설정 (Ubuntu/Debian)
└── modules-enabled/       # 활성화된 모듈들

/var/log/nginx/            # 로그 디렉토리
├── access.log             # 접근 로그
└── error.log              # 에러 로그

/var/www/html/             # 기본 웹 루트 디렉토리
/usr/share/nginx/html/     # CentOS/RHEL 기본 웹 루트

/var/run/nginx.pid         # PID 파일
```

## 설정 파일 구조

### nginx.conf 기본 구조
```nginx
# 전역 컨텍스트
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

# 이벤트 컨텍스트
events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

# HTTP 컨텍스트
http {
    # 기본 설정
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # 로그 형식
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    # 성능 최적화
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # 서버 블록들
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

### 컨텍스트 계층
1. **Main Context**: 전역 설정
2. **Events Context**: 연결 처리 설정
3. **HTTP Context**: HTTP 관련 설정
4. **Server Context**: 가상 호스트 설정
5. **Location Context**: 특정 URI 처리

## 기본 서버 블록 설정

### 기본 웹사이트 설정
```nginx
# /etc/nginx/sites-available/default
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    server_name _;
    root /var/www/html;
    index index.html index.htm index.nginx-debian.html;
    
    # 기본 location
    location / {
        try_files $uri $uri/ =404;
    }
    
    # 에러 페이지
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    
    location = /50x.html {
        root /usr/share/nginx/html;
    }
    
    # 로그 설정
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
}
```

### 커스텀 사이트 설정
```nginx
# /etc/nginx/sites-available/example.com
server {
    listen 80;
    server_name example.com www.example.com;
    root /var/www/example.com;
    index index.html index.php;
    
    # 보안 헤더
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # 정적 파일 캐싱
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # PHP 처리 (필요시)
    location ~ \.php$ {
        include snippets/fastcgi-php.conf;
        fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
    }
    
    # 숨겨진 파일 접근 차단
    location ~ /\. {
        deny all;
    }
}
```

### 서버 블록 주요 지시어

#### listen
```nginx
listen 80;                    # IPv4 80번 포트
listen [::]:80;              # IPv6 80번 포트
listen 80 default_server;    # 기본 서버
listen 443 ssl http2;        # HTTPS with HTTP/2
```

#### server_name
```nginx
server_name example.com;                    # 정확한 도메인
server_name *.example.com;                  # 와일드카드
server_name example.com www.example.com;    # 여러 도메인
server_name ~^(\w+)\.example\.com$;        # 정규식
```

#### root와 index
```nginx
root /var/www/html;                         # 문서 루트
index index.html index.htm index.php;      # 인덱스 파일 우선순위
```

## 설정 테스트 및 서비스 관리

### 설정 파일 테스트
```bash
# 문법 검사
sudo nginx -t

# 설정 덤프
sudo nginx -T

# 특정 설정 파일 테스트
sudo nginx -t -c /etc/nginx/nginx.conf
```

### 서비스 관리
```bash
# 서비스 시작
sudo systemctl start nginx

# 서비스 중지
sudo systemctl stop nginx

# 서비스 재시작
sudo systemctl restart nginx

# 설정 리로드 (무중단)
sudo systemctl reload nginx
sudo nginx -s reload

# 서비스 상태 확인
sudo systemctl status nginx

# 서비스 활성화/비활성화
sudo systemctl enable nginx
sudo systemctl disable nginx
```

### 프로세스 신호
```bash
# 정상 종료
sudo nginx -s quit

# 즉시 종료
sudo nginx -s stop

# 설정 리로드
sudo nginx -s reload

# 로그 파일 재오픈
sudo nginx -s reopen
```

## 기본 보안 설정

### 서버 정보 숨기기
```nginx
# /etc/nginx/nginx.conf
http {
    server_tokens off;
    
    # 커스텀 서버 헤더
    more_set_headers "Server: MyServer/1.0";
}
```

### 기본 보안 헤더
```nginx
server {
    # 보안 헤더 추가
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
}
```

## 실습 예제

### 1. 기본 HTML 사이트 구축
```bash
# 웹 디렉토리 생성
sudo mkdir -p /var/www/mysite

# HTML 파일 작성
sudo tee /var/www/mysite/index.html > /dev/null << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>My Nginx Site</title>
</head>
<body>
    <h1>Welcome to My Nginx Site</h1>
    <p>Server is running on Nginx!</p>
</body>
</html>
EOF

# 사이트 설정 파일 생성
sudo tee /etc/nginx/sites-available/mysite > /dev/null << 'EOF'
server {
    listen 80;
    server_name mysite.local;
    root /var/www/mysite;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
EOF

# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/

# 설정 테스트 및 리로드
sudo nginx -t
sudo systemctl reload nginx
```

### 2. 로그 분석을 위한 커스텀 로그 형식
```nginx
http {
    log_format detailed '$remote_addr - $remote_user [$time_local] '
                       '"$request" $status $body_bytes_sent '
                       '"$http_referer" "$http_user_agent" '
                       '$request_time $upstream_response_time';
    
    server {
        access_log /var/log/nginx/detailed.log detailed;
    }
}
```

## 문제 해결

### 일반적인 오류들

#### 1. 포트 바인딩 실패
```bash
# 포트 사용 확인
sudo netstat -tlnp | grep :80
sudo lsof -i :80

# 다른 웹 서버 중지
sudo systemctl stop apache2
```

#### 2. 권한 문제
```bash
# 파일 권한 설정
sudo chown -R www-data:www-data /var/www/
sudo chmod -R 755 /var/www/

# SELinux 문제 (CentOS/RHEL)
sudo setsebool -P httpd_can_network_connect 1
sudo restorecon -R /var/www/
```

#### 3. 설정 파일 오류
```bash
# 상세한 오류 확인
sudo nginx -t 2>&1

# 로그 확인
sudo tail -f /var/log/nginx/error.log
```

## 다음 단계

이번 포스트에서는 Nginx의 기본 설치와 설정에 대해 알아보았습니다. 다음 포스트에서는:

- Virtual Host 설정
- 로드 밸런싱 구성
- SSL/TLS 설정
- 업스트림 서버 설정

에 대해 자세히 다루겠습니다.

## 참고 자료

- [Nginx 공식 문서](https://nginx.org/en/docs/)
- [Nginx 설정 가이드](https://www.nginx.com/resources/wiki/start/)
- [Nginx 보안 가이드](https://github.com/trimstray/nginx-admins-handbook)
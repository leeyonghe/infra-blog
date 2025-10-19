---
layout: post
title: "리눅스 보안 완전 가이드 3편 - SELinux/AppArmor와 시스템 하드닝 | Linux Security Guide Part 3 - SELinux/AppArmor & System Hardening"
date: 2025-02-19 09:00:00 +0900
categories: [Security, Linux]
tags: [selinux, apparmor, system-hardening, kernel-security, filesystem-security, mandatory-access-control]
---

리눅스 시스템의 핵심 보안 레이어인 SELinux/AppArmor와 시스템 하드닝 기법을 완전히 마스터해보겠습니다. 강제 접근 제어부터 커널 보안 매개변수까지 최고 수준의 시스템 보안을 구축하는 방법을 다룹니다.

## SELinux 완전 마스터 | SELinux Complete Mastery

### 🔐 SELinux 기초부터 고급까지

#### SELinux 개념 및 초기 설정
```bash
# SELinux 상태 및 정보 확인
sestatus                    # 전체 상태 확인
getenforce                  # 현재 모드 확인
seinfo                      # 정책 통계
semanage -l                 # 관리 가능한 객체 목록

# SELinux 모드 설정
# /etc/selinux/config
SELINUX=enforcing           # enforcing, permissive, disabled
SELINUXTYPE=targeted        # targeted, minimum, mls

# 임시 모드 변경 (재부팅 시 원복)
setenforce 1                # enforcing 모드
setenforce 0                # permissive 모드

# SELinux 라벨링 시스템 이해
# 모든 객체는 security context를 가짐: user:role:type:level
ls -Z /var/www/html/        # 파일 컨텍스트 확인
ps auxZ                     # 프로세스 컨텍스트 확인
id -Z                       # 현재 사용자 컨텍스트

# 컨텍스트 구성 요소
# user: SELinux 사용자 (unconfined_u, system_u, user_u 등)
# role: 역할 (unconfined_r, system_r, object_r 등)  
# type: 타입/도메인 (httpd_t, httpd_exec_t, user_home_t 등)
# level: MLS/MCS 레벨 (s0, s0:c0,c1 등)
```

#### 파일 컨텍스트 관리
```bash
# 파일 컨텍스트 확인 및 복원
ls -lZ /var/www/html/
restorecon -Rv /var/www/html/      # 기본 컨텍스트로 복원
restorecon -RvF /var/www/html/     # 강제 복원

# 파일 컨텍스트 수동 설정
chcon -t httpd_exec_t /usr/local/apache2/bin/httpd
chcon -u system_u /var/www/html/index.html
chcon --reference=/var/www/html/index.html /var/www/html/newfile.html

# 영구적인 컨텍스트 설정 (정책에 추가)
semanage fcontext -a -t httpd_exec_t "/usr/local/apache2/bin/httpd"
semanage fcontext -a -t httpd_config_t "/etc/myapp/.*\.conf"
semanage fcontext -a -t user_home_t "/home/[^/]+/mydata(/.*)?"

# 컨텍스트 정책 확인
semanage fcontext -l | grep httpd
matchpathcon /var/www/html/index.html    # 예상 컨텍스트 확인

# 컨텍스트 변경 내역 추적
ausearch -m AVC -ts today              # AVC 거부 로그
sealert -a /var/log/audit/audit.log    # 정책 제안

# 고급 컨텍스트 관리 스크립트
#!/bin/bash
# /usr/local/bin/selinux-context-manager.sh

ACTION=$1
TARGET=$2
CONTEXT=$3

case $ACTION in
    "scan")
        echo "=== SELinux Context Scan ==="
        find "$TARGET" -print0 | xargs -0 ls -lZ | \
        awk '{print $4, $9}' | sort | uniq -c | sort -nr
        ;;
        
    "restore")
        echo "Restoring contexts for $TARGET..."
        restorecon -RvF "$TARGET"
        ;;
        
    "set-bulk")
        echo "Setting bulk context $CONTEXT for $TARGET..."
        find "$TARGET" -type f -exec chcon -t "$CONTEXT" {} \;
        ;;
        
    "analyze")
        echo "=== Context Analysis for $TARGET ==="
        # 비정상적인 컨텍스트 찾기
        find "$TARGET" -type f -exec ls -lZ {} \; | \
        grep -v "system_u:object_r" | \
        awk '{print "Unusual context:", $4, $9}'
        ;;
        
    *)
        echo "Usage: $0 {scan|restore|set-bulk|analyze} <path> [context]"
        exit 1
        ;;
esac
```

#### SELinux 불린 값 관리
```bash
# 불린 값 확인
getsebool -a                          # 모든 불린 값
getsebool -a | grep httpd            # httpd 관련 불린
getsebool httpd_can_network_connect   # 특정 불린

# 불린 값 설정
setsebool httpd_can_network_connect on           # 임시 설정
setsebool -P httpd_can_network_connect on        # 영구 설정

# 주요 웹 서버 불린 값들
setsebool -P httpd_can_network_connect on        # 네트워크 연결 허용
setsebool -P httpd_can_sendmail on               # 메일 발송 허용
setsebool -P httpd_enable_cgi on                 # CGI 실행 허용
setsebool -P httpd_read_user_content on          # 사용자 콘텐츠 읽기
setsebool -P httpd_enable_homedirs on            # 홈 디렉토리 접근
setsebool -P httpd_execmem on                    # 메모리 실행 허용
setsebool -P httpd_use_nfs on                    # NFS 사용 허용

# 데이터베이스 관련 불린
setsebool -P allow_user_mysql_connect on         # MySQL 연결 허용
setsebool -P mysql_connect_any on                # MySQL 임의 연결

# SSH 관련 불린
setsebool -P ssh_chroot_rw_homedirs on           # chroot에서 홈디렉토리 쓰기

# 불린 값 모니터링 스크립트
#!/bin/bash
# /usr/local/bin/selinux-bool-monitor.sh

# 현재 불린 설정 백업
getsebool -a > /var/log/selinux-bools-$(date +%Y%m%d).log

# 변경된 불린 값 감지
if [ -f /var/log/selinux-bools-last.log ]; then
    echo "=== Boolean Changes Detected ==="
    diff /var/log/selinux-bools-last.log /var/log/selinux-bools-$(date +%Y%m%d).log
fi

cp /var/log/selinux-bools-$(date +%Y%m%d).log /var/log/selinux-bools-last.log

# 보안에 민감한 불린 값들 체크
CRITICAL_BOOLS=(
    "httpd_execmem"
    "httpd_enable_cgi"
    "allow_execheap"
    "allow_execstack"
    "selinuxuser_execstack"
)

echo "=== Critical Boolean Status ==="
for bool in "${CRITICAL_BOOLS[@]}"; do
    status=$(getsebool "$bool" 2>/dev/null)
    if echo "$status" | grep -q " on"; then
        echo "WARNING: $bool is enabled"
    else
        echo "OK: $bool is disabled"
    fi
done
```

#### 포트 라벨 관리
```bash
# 포트 라벨 확인
semanage port -l                    # 모든 포트 라벨
semanage port -l | grep http        # HTTP 관련 포트
ss -tlnZ                           # 네트워크 서비스와 컨텍스트

# 새로운 포트에 라벨 할당
semanage port -a -t http_port_t -p tcp 8080      # HTTP 포트 추가
semanage port -a -t ssh_port_t -p tcp 2222       # SSH 포트 추가
semanage port -a -t mysqld_port_t -p tcp 3307    # MySQL 포트 추가

# 포트 라벨 수정
semanage port -m -t http_port_t -p tcp 8080

# 포트 라벨 제거
semanage port -d -t http_port_t -p tcp 8080

# 커스텀 포트 타입 생성 (고급)
# myapp.te 파일 생성
cat > myapp.te << 'EOF'
policy_module(myapp, 1.0)

type myapp_port_t;
corenet_port(myapp_port_t)

allow httpd_t myapp_port_t:tcp_socket name_bind;
EOF

# 정책 컴파일 및 설치
make -f /usr/share/selinux/devel/Makefile myapp.pp
semodule -i myapp.pp
semanage port -a -t myapp_port_t -p tcp 9999
```

### 🛡️ SELinux 커스텀 정책 작성

#### 정책 모듈 개발
```bash
# 정책 개발 환경 준비
yum install selinux-policy-devel    # CentOS/RHEL
apt-get install selinux-policy-dev  # Ubuntu/Debian

# AVC 거부 분석 및 정책 생성
# 1. 서비스 실행하여 AVC 거부 로그 수집
systemctl start myapp
tail -f /var/log/audit/audit.log | grep AVC

# 2. audit2allow로 정책 제안 생성
grep myapp /var/log/audit/audit.log | audit2allow -m myapp_policy
grep myapp /var/log/audit/audit.log | audit2allow -M myapp_policy

# 3. 수동으로 정책 모듈 작성
cat > myapp_custom.te << 'EOF'
policy_module(myapp_custom, 1.0)

########################################
#
# Declarations
#

type myapp_t;
type myapp_exec_t;
init_daemon_domain(myapp_t, myapp_exec_t)

type myapp_config_t;
files_config_file(myapp_config_t)

type myapp_var_lib_t;
files_type(myapp_var_lib_t)

type myapp_log_t;
logging_log_file(myapp_log_t)

type myapp_port_t;
corenet_port(myapp_port_t)

########################################
#
# myapp local policy
#

# 기본 도메인 권한
allow myapp_t self:process { fork signal_perms };
allow myapp_t self:fifo_file rw_fifo_file_perms;
allow myapp_t self:unix_stream_socket create_stream_socket_perms;

# 네트워크 권한
allow myapp_t self:tcp_socket create_stream_socket_perms;
allow myapp_t myapp_port_t:tcp_socket name_bind;
corenet_tcp_sendrecv_generic_if(myapp_t)
corenet_tcp_sendrecv_generic_node(myapp_t)

# 파일 시스템 권한
allow myapp_t myapp_config_t:file read_file_perms;
allow myapp_t myapp_var_lib_t:dir create_dir_perms;
allow myapp_t myapp_var_lib_t:file create_file_perms;
allow myapp_t myapp_log_t:file create_file_perms;

# 시스템 서비스 상호작용
can_exec(myapp_t, myapp_exec_t)
files_read_etc_files(myapp_t)
libs_use_ld_so(myapp_t)
libs_use_shared_libs(myapp_t)

# 로깅
logging_send_syslog_msg(myapp_t)
EOF

# 4. 파일 컨텍스트 정의
cat > myapp_custom.fc << 'EOF'
/usr/local/bin/myapp        --      gen_context(system_u:object_r:myapp_exec_t,s0)
/etc/myapp(/.*)?                    gen_context(system_u:object_r:myapp_config_t,s0)
/var/lib/myapp(/.*)?                gen_context(system_u:object_r:myapp_var_lib_t,s0)
/var/log/myapp(/.*)?                gen_context(system_u:object_r:myapp_log_t,s0)
EOF

# 5. 인터페이스 파일 (다른 모듈에서 사용할 인터페이스)
cat > myapp_custom.if << 'EOF'
## <summary>MyApp custom policy</summary>

########################################
## <summary>
##  Execute myapp in the myapp domain.
## </summary>
## <param name="domain">
##  <summary>
##  Domain allowed to transition.
##  </summary>
## </param>
#
interface(`myapp_domtrans',`
    gen_require(`
        type myapp_t, myapp_exec_t;
    ')

    corecmd_search_bin($1)
    domtrans_pattern($1, myapp_exec_t, myapp_t)
')

########################################
## <summary>
##  Read myapp configuration files.
## </summary>
## <param name="domain">
##  <summary>
##  Domain allowed access.
##  </summary>
## </param>
#
interface(`myapp_read_config',`
    gen_require(`
        type myapp_config_t;
    ')

    files_search_etc($1)
    read_files_pattern($1, myapp_config_t, myapp_config_t)
')
EOF

# 6. 정책 컴파일 및 설치
make -f /usr/share/selinux/devel/Makefile myapp_custom.pp
semodule -i myapp_custom.pp

# 7. 파일 컨텍스트 적용
semanage fcontext -a -f "" -t myapp_exec_t "/usr/local/bin/myapp"
semanage fcontext -a -f "" -t myapp_config_t "/etc/myapp(/.*)?"
restorecon -Rv /usr/local/bin/myapp /etc/myapp

# 8. 포트 라벨 설정
semanage port -a -t myapp_port_t -p tcp 9090
```

#### 정책 디버깅 및 최적화
```bash
#!/bin/bash
# /usr/local/bin/selinux-debug.sh
# SELinux 정책 디버깅 도구

MODE=$1
MODULE=$2

case $MODE in
    "avc-analysis")
        echo "=== AVC Denial Analysis ==="
        ausearch -m AVC -ts today | grep "$MODULE" | \
        while read line; do
            echo "$line" | audit2allow -R
            echo "---"
        done
        ;;
        
    "permissive-test")
        echo "Setting $MODULE to permissive mode for testing..."
        semanage permissive -a ${MODULE}_t
        echo "Test your application, then check denials:"
        echo "ausearch -m AVC -ts now"
        echo "When done, remove permissive mode:"
        echo "semanage permissive -d ${MODULE}_t"
        ;;
        
    "policy-stats")
        echo "=== Policy Statistics ==="
        seinfo -t | grep "$MODULE"
        seinfo -r | grep "$MODULE"
        seinfo -u | grep "$MODULE"
        ;;
        
    "generate-policy")
        echo "Generating policy for $MODULE..."
        ausearch -m AVC -ts today | grep "$MODULE" | \
        audit2allow -M ${MODULE}_additional
        
        echo "Generated policy module: ${MODULE}_additional.pp"
        echo "Install with: semodule -i ${MODULE}_additional.pp"
        ;;
        
    "module-deps")
        echo "=== Module Dependencies for $MODULE ==="
        semodule -l | grep "$MODULE"
        ;;
        
    *)
        echo "Usage: $0 {avc-analysis|permissive-test|policy-stats|generate-policy|module-deps} <module>"
        echo ""
        echo "Examples:"
        echo "  $0 avc-analysis httpd"
        echo "  $0 permissive-test myapp"
        echo "  $0 generate-policy myapp"
        exit 1
        ;;
esac
```

## AppArmor 완전 마스터 | AppArmor Complete Mastery

### 🛡️ AppArmor 프로파일 작성 및 관리

#### AppArmor 기본 관리
```bash
# AppArmor 상태 확인
aa-status                   # 전체 상태
aa-enabled                  # 활성화 여부
aa-unconfined               # 제한되지 않은 프로세스

# 프로파일 모드 관리
aa-enforce /etc/apparmor.d/usr.bin.firefox     # enforce 모드
aa-complain /etc/apparmor.d/usr.bin.firefox    # complain 모드  
aa-disable /etc/apparmor.d/usr.bin.firefox     # 비활성화

# 프로파일 재로드
apparmor_parser -r /etc/apparmor.d/usr.bin.firefox
apparmor_parser -R /etc/apparmor.d/            # 모든 프로파일 재로드

# 프로파일 상태 확인
aa-status | grep firefox
```

#### 고급 AppArmor 프로파일 작성
```bash
# 웹 애플리케이션용 커스텀 프로파일
# /etc/apparmor.d/usr.local.bin.webapp
#include <tunables/global>

profile /usr/local/bin/webapp flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/nameservice>
  #include <abstractions/openssl>
  #include <abstractions/ssl_certs>
  
  # 실행 권한
  /usr/local/bin/webapp mr,
  
  # 라이브러리 접근
  /lib{,32,64}/** mr,
  /usr/lib{,32,64}/** mr,
  /usr/local/lib/** mr,
  
  # 설정 파일 (읽기 전용)
  /etc/webapp/ r,
  /etc/webapp/** r,
  owner /etc/webapp/webapp.conf r,
  
  # 사용자별 설정 및 데이터
  owner @{HOME}/.webapp/ rw,
  owner @{HOME}/.webapp/** rw,
  owner @{HOME}/.webapp/cache/** rwk,
  
  # 애플리케이션 데이터 디렉토리
  /var/lib/webapp/ r,
  /var/lib/webapp/** rw,
  /var/cache/webapp/ r,
  /var/cache/webapp/** rw,
  
  # 로그 파일
  /var/log/webapp/ r,
  /var/log/webapp/*.log w,
  /var/log/webapp/*.log.[0-9] w,
  
  # 네트워크 접근
  network inet stream,
  network inet6 stream,
  network inet dgram,
  network inet6 dgram,
  network netlink raw,
  
  # 프로세스 제어
  capability setuid,
  capability setgid,
  capability dac_override,
  capability net_bind_service,
  
  # 임시 파일
  /tmp/ r,
  /tmp/webapp.** rw,
  owner /tmp/webapp-@{pid}-* rw,
  /var/tmp/ r,
  /var/tmp/webapp.** rw,
  
  # 시스템 정보 접근
  /proc/sys/kernel/random/uuid r,
  /proc/loadavg r,
  /proc/meminfo r,
  /sys/devices/system/cpu/ r,
  /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq r,
  
  # 거부할 접근들 (명시적 거부)
  deny /etc/passwd r,
  deny /etc/shadow r,
  deny /etc/gshadow r,
  deny owner /home/*/.ssh/** rw,
  deny /proc/[0-9]*/maps r,
  deny /proc/[0-9]*/mem r,
  deny /proc/kmem r,
  deny /proc/kcore r,
  deny /boot/** r,
  
  # 하위 프로세스 실행
  /bin/dash ix,
  /bin/bash ix,
  /usr/bin/python3 ix,
  /usr/bin/python3.[0-9] ix,
  /usr/local/bin/webapp-helper Cx -> helper,
  
  # 조건부 실행 규칙
  profile helper {
    #include <abstractions/base>
    
    /usr/local/bin/webapp-helper mr,
    /var/lib/webapp/helper-data/** r,
    /tmp/helper-** rw,
    
    # 부모로부터 상속받은 파일 디스크립터만 사용
    deny network,
    deny capability,
  }
  
  # 신호 처리
  signal (send) set=(term,kill,usr1,usr2) peer=/usr/local/bin/webapp,
  signal (receive) set=(term,kill,usr1,usr2),
  
  # DBus 접근 (필요시)
  dbus (send)
       bus=session
       path=/org/freedesktop/Notifications
       interface=org.freedesktop.Notifications
       member=Notify
       peer=(name=org.freedesktop.Notifications),
}

# 데이터베이스 서버용 프로파일
# /etc/apparmor.d/usr.sbin.mysqld-custom
#include <tunables/global>

profile /usr/sbin/mysqld-custom flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/mysql>
  #include <abstractions/nameservice>
  #include <abstractions/user-tmp>
  
  capability dac_override,
  capability setgid,
  capability setuid,
  capability sys_resource,
  capability net_bind_service,
  
  # MySQL 바이너리
  /usr/sbin/mysqld mr,
  /usr/sbin/mysqld-debug mr,
  
  # 설정 파일
  /etc/mysql/ r,
  /etc/mysql/** r,
  /etc/my.cnf r,
  /etc/my.cnf.d/ r,
  /etc/my.cnf.d/*.cnf r,
  
  # 데이터 디렉토리
  /var/lib/mysql/ r,
  /var/lib/mysql/** rwk,
  /var/lib/mysql-files/ r,
  /var/lib/mysql-files/** rw,
  
  # 로그 파일
  /var/log/mysql/ r,
  /var/log/mysql/*.log rw,
  /var/log/mysql.log rw,
  /var/log/mysql/error.log rw,
  
  # 소켓 파일
  /var/run/mysqld/ rw,
  /var/run/mysqld/mysqld.sock rw,
  /tmp/mysql.sock rw,
  
  # 네트워크
  network tcp,
  
  # 프로세스 간 통신
  /proc/*/status r,
  /proc/sys/vm/overcommit_memory r,
  
  # 임시 파일
  /tmp/ r,
  /tmp/mysql-** rw,
  /var/tmp/ r,
  /var/tmp/mysql-** rw,
  
  # 보안 제한
  deny capability sys_ptrace,
  deny @{PROC}/sys/kernel/core_pattern w,
  deny /etc/passwd r,
  deny /etc/shadow r,
}
```

#### AppArmor 프로파일 자동 생성 및 튜닝
```bash
#!/bin/bash
# /usr/local/bin/apparmor-manager.sh
# AppArmor 프로파일 관리 도구

ACTION=$1
BINARY=$2
PROFILE_NAME=${3:-$(basename $BINARY)}

case $ACTION in
    "generate")
        echo "Generating AppArmor profile for $BINARY..."
        
        # 1. 기본 프로파일 생성
        aa-genprof "$BINARY"
        
        echo "Profile generation completed."
        echo "Test your application and run: $0 tune $BINARY"
        ;;
        
    "tune")
        echo "Tuning AppArmor profile for $BINARY..."
        
        # 로그프로파일링 실행
        aa-logprof
        
        echo "Profile tuning completed."
        ;;
        
    "analyze")
        echo "=== AppArmor Profile Analysis for $PROFILE_NAME ==="
        
        # 프로파일 구문 검사
        apparmor_parser -p /etc/apparmor.d/$PROFILE_NAME 2>&1 | \
        grep -E "(ERROR|WARNING)" || echo "✓ Syntax OK"
        
        # 프로파일 통계
        echo ""
        echo "Profile statistics:"
        grep -c "^[[:space:]]*/" /etc/apparmor.d/$PROFILE_NAME && echo "File rules"
        grep -c "capability" /etc/apparmor.d/$PROFILE_NAME && echo "Capabilities"
        grep -c "network" /etc/apparmor.d/$PROFILE_NAME && echo "Network rules"
        grep -c "deny" /etc/apparmor.d/$PROFILE_NAME && echo "Deny rules"
        
        # 보안 검사
        echo ""
        echo "Security analysis:"
        if grep -q "capability sys_admin" /etc/apparmor.d/$PROFILE_NAME; then
            echo "⚠️  WARNING: sys_admin capability found"
        fi
        
        if grep -q "/etc/shadow" /etc/apparmor.d/$PROFILE_NAME; then
            echo "⚠️  WARNING: Shadow file access found"  
        fi
        
        if grep -q "network raw" /etc/apparmor.d/$PROFILE_NAME; then
            echo "⚠️  WARNING: Raw network access found"
        fi
        ;;
        
    "template")
        TEMPLATE_TYPE=$3
        echo "Creating AppArmor profile template for $BINARY ($TEMPLATE_TYPE)..."
        
        case $TEMPLATE_TYPE in
            "webapp")
                cat > /etc/apparmor.d/$PROFILE_NAME << 'EOF'
#include <tunables/global>

profile BINARY_PATH {
  #include <abstractions/base>
  #include <abstractions/nameservice>
  
  # Binary execution
  BINARY_PATH mr,
  
  # Libraries
  /lib{,32,64}/** mr,
  /usr/lib{,32,64}/** mr,
  
  # Configuration
  /etc/APP_NAME/ r,
  /etc/APP_NAME/** r,
  
  # Data directories
  /var/lib/APP_NAME/** rw,
  /var/log/APP_NAME/** w,
  
  # Network
  network inet stream,
  
  # Capabilities
  capability setuid,
  capability setgid,
  
  # Temporary files
  /tmp/APP_NAME.** rw,
}
EOF
                sed -i "s|BINARY_PATH|$BINARY|g" /etc/apparmor.d/$PROFILE_NAME
                sed -i "s|APP_NAME|$(basename $BINARY)|g" /etc/apparmor.d/$PROFILE_NAME
                ;;
                
            "service")
                cat > /etc/apparmor.d/$PROFILE_NAME << 'EOF'
#include <tunables/global>

profile BINARY_PATH flags=(attach_disconnected) {
  #include <abstractions/base>
  #include <abstractions/nameservice>
  
  # Service binary
  BINARY_PATH mr,
  
  # System libraries
  /lib{,32,64}/** mr,
  /usr/lib{,32,64}/** mr,
  
  # Service configuration
  /etc/APP_NAME/ r,
  /etc/APP_NAME/** r,
  
  # Runtime directories
  /var/run/APP_NAME/ rw,
  /var/run/APP_NAME/** rw,
  
  # Log files
  /var/log/APP_NAME/ r,
  /var/log/APP_NAME/*.log w,
  
  # PID file
  /var/run/APP_NAME.pid w,
  
  # Network access
  network inet stream,
  network inet dgram,
  
  # System capabilities
  capability setuid,
  capability setgid,
  capability net_bind_service,
  
  # Signal handling
  signal (receive) set=(term,kill,usr1),
}
EOF
                sed -i "s|BINARY_PATH|$BINARY|g" /etc/apparmor.d/$PROFILE_NAME
                sed -i "s|APP_NAME|$(basename $BINARY)|g" /etc/apparmor.d/$PROFILE_NAME
                ;;
        esac
        
        echo "Template created: /etc/apparmor.d/$PROFILE_NAME"
        echo "Edit the template and then load it with:"
        echo "apparmor_parser -r /etc/apparmor.d/$PROFILE_NAME"
        ;;
        
    "test")
        echo "Testing AppArmor profile for $BINARY..."
        
        # complain 모드로 전환
        aa-complain /etc/apparmor.d/$PROFILE_NAME
        
        echo "Profile set to complain mode."
        echo "Run your application tests, then check logs:"
        echo "journalctl -f | grep apparmor"
        echo ""
        echo "When testing is complete, switch to enforce mode:"
        echo "aa-enforce /etc/apparmor.d/$PROFILE_NAME"
        ;;
        
    "backup")
        BACKUP_DIR="/etc/apparmor.d/backups/$(date +%Y%m%d)"
        mkdir -p "$BACKUP_DIR"
        cp /etc/apparmor.d/$PROFILE_NAME "$BACKUP_DIR/"
        echo "Profile backed up to: $BACKUP_DIR/$PROFILE_NAME"
        ;;
        
    *)
        echo "Usage: $0 {generate|tune|analyze|template|test|backup} <binary> [profile-name]"
        echo ""
        echo "Template types for 'template' action:"
        echo "  webapp  - Web application template"
        echo "  service - System service template"
        echo ""
        echo "Examples:"
        echo "  $0 generate /usr/local/bin/myapp"
        echo "  $0 template /usr/local/bin/myapp webapp"
        echo "  $0 analyze myapp"
        echo "  $0 test myapp"
        exit 1
        ;;
esac
```

## 시스템 하드닝 | System Hardening

### 🔧 커널 보안 매개변수 최적화

#### 고급 sysctl 보안 설정
```bash
# /etc/sysctl.d/99-security-hardening.conf
# 종합적인 시스템 보안 강화 설정

# ==================== 네트워크 보안 ====================
# IPv4 네트워크 보안
net.ipv4.ip_forward = 0                          # IP 포워딩 비활성화
net.ipv4.conf.all.send_redirects = 0             # ICMP 리디렉트 전송 차단
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0           # ICMP 리디렉트 수신 차단
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0           # 보안 리디렉트도 차단
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.accept_source_route = 0        # 소스 라우팅 차단
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.rp_filter = 1                 # 역방향 경로 필터링
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.log_martians = 1              # 비정상 패킷 로깅
net.ipv4.conf.default.log_martians = 1

# SYN Flood 공격 방지
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# ICMP 보안
net.ipv4.icmp_echo_ignore_broadcasts = 1         # 브로드캐스트 ping 무시
net.ipv4.icmp_ignore_bogus_error_responses = 1   # 잘못된 ICMP 에러 무시
net.ipv4.icmp_echo_ignore_all = 0                # 일반 ping은 허용 (필요시 1로 변경)
net.ipv4.icmp_ratelimit = 100                    # ICMP 속도 제한

# IPv6 보안 (IPv6 사용하지 않는 경우)
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1

# IPv6 사용하는 경우의 보안 설정
# net.ipv6.conf.all.accept_ra = 0                # 라우터 광고 차단
# net.ipv6.conf.default.accept_ra = 0
# net.ipv6.conf.all.accept_redirects = 0         # IPv6 리디렉트 차단
# net.ipv6.conf.default.accept_redirects = 0
# net.ipv6.conf.all.accept_source_route = 0      # IPv6 소스 라우팅 차단
# net.ipv6.conf.default.accept_source_route = 0

# ==================== 메모리 보안 ====================
# ASLR (Address Space Layout Randomization)
kernel.randomize_va_space = 2                    # 전체 주소 공간 무작위화

# 메모리 보호
vm.mmap_min_addr = 65536                        # mmap 최소 주소 (NULL 포인터 역참조 방지)
kernel.exec-shield = 1                          # 실행 쉴드 활성화 (가능한 경우)
kernel.dmesg_restrict = 1                       # dmesg 제한 (일반 사용자 차단)
kernel.kptr_restrict = 2                        # 커널 포인터 정보 제한

# 메모리 할당 보안
vm.overcommit_memory = 2                        # 메모리 오버커밋 제한
vm.overcommit_ratio = 80                        # 오버커밋 비율 80%

# ==================== 프로세스 보안 ====================
# 코어 덤프 보안
fs.suid_dumpable = 0                            # SUID 프로그램 코어 덤프 금지
kernel.core_uses_pid = 1                        # 코어 파일에 PID 포함
kernel.core_pattern = |/bin/false               # 코어 덤프 완전 비활성화

# 프로세스 제한
kernel.pid_max = 65536                          # 최대 프로세스 ID
kernel.threads-max = 65536                      # 최대 스레드 수

# 시스템 호출 보안
kernel.yama.ptrace_scope = 1                    # ptrace 제한 (디버깅 방지)
kernel.unprivileged_bpf_disabled = 1            # 비특권 BPF 비활성화
net.core.bpf_jit_harden = 2                     # BPF JIT 강화

# ==================== 파일 시스템 보안 ====================
# 파일 시스템 보안
fs.protected_hardlinks = 1                      # 하드링크 보호
fs.protected_symlinks = 1                       # 심볼릭링크 보호
fs.protected_fifos = 2                          # FIFO 보호
fs.protected_regular = 2                        # 일반 파일 보호

# 파일 디스크립터 제한
fs.file-max = 2097152                          # 시스템 전체 최대 파일 디스크립터
fs.nr_open = 1048576                           # 프로세스당 최대 파일 디스크립터

# ==================== 시스템 제어 ====================
# 시스템 키 조합 비활성화
kernel.sysrq = 0                               # SysRq 키 비활성화
kernel.ctrl-alt-del = 0                        # Ctrl+Alt+Del 비활성화

# 커널 모듈 보안
kernel.modules_disabled = 1                     # 런타임 커널 모듈 로딩 비활성화 (신중히 사용)
kernel.kexec_load_disabled = 1                 # kexec 비활성화

# ==================== 네트워크 성능 및 보안 ====================
# 네트워크 버퍼 크기
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144  
net.core.wmem_max = 16777216

# TCP 윈도우 스케일링
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# 네트워크 큐 설정
net.core.netdev_max_backlog = 5000

# ==================== 로깅 및 감사 ====================
# 커널 로깅
kernel.printk = 3 4 1 3                        # 로깅 레벨 조정

# 설정 적용 및 확인 스크립트
#!/bin/bash
# /usr/local/bin/apply-sysctl-security.sh

echo "Applying sysctl security settings..."

# 현재 설정 백업
sysctl -a > /etc/sysctl.backup.$(date +%Y%m%d-%H%M%S)

# 새 설정 적용
sysctl -p /etc/sysctl.d/99-security-hardening.conf

# 적용 결과 확인
echo "=== Security Sysctl Settings Applied ==="
echo "Network security:"
sysctl net.ipv4.ip_forward net.ipv4.conf.all.accept_redirects net.ipv4.tcp_syncookies

echo "Memory security:"
sysctl kernel.randomize_va_space vm.mmap_min_addr kernel.dmesg_restrict

echo "Process security:"
sysctl fs.suid_dumpable kernel.yama.ptrace_scope kernel.unprivileged_bpf_disabled

echo "File system security:"
sysctl fs.protected_hardlinks fs.protected_symlinks

echo "System control:"
sysctl kernel.sysrq kernel.ctrl-alt-del

echo ""
echo "Settings applied successfully!"
echo "Reboot recommended to ensure all settings take effect."
```

## 다음 편 예고

다음 포스트에서는 **침입 탐지 시스템과 실시간 모니터링**을 상세히 다룰 예정입니다:
- AIDE 파일 무결성 모니터링
- Fail2Ban 고급 설정 및 커스터마이징
- Wazuh SIEM 구축 및 운영
- 실시간 위협 탐지 및 대응

SELinux/AppArmor와 시스템 하드닝을 완벽하게 마스터하셨나요? 🛡️🔒
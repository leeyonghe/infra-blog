---
layout: post
title: "리눅스 보안 완전 가이드 5편 - Wazuh SIEM과 컨테이너 보안 | Linux Security Guide Part 5 - Wazuh SIEM & Container Security"
date: 2025-05-24 09:00:00 +0900
categories: [Security, Linux]
tags: [wazuh, siem, container-security, docker, kubernetes, compliance, threat-detection, log-analysis]
---

리눅스 보안의 최종 완성 단계인 Wazuh SIEM 구축과 컨테이너 보안을 완전히 마스터해보겠습니다. 엔터프라이즈급 보안 모니터링부터 최신 컨테이너 환경 보안까지, 차세대 보안 인프라를 구축하는 모든 것을 다룹니다.

## Wazuh SIEM 완전 구축 | Complete Wazuh SIEM Implementation

### 🔍 Wazuh 아키텍처 및 설치

#### Wazuh 완전 설치 스크립트
```bash
#!/bin/bash
# /usr/local/bin/wazuh-installer.sh
# Wazuh SIEM 완전 자동 설치 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_color() {
    echo -e "${2}${1}${NC}"
}

# 시스템 정보 확인
OS_ID=$(grep ^ID= /etc/os-release | cut -d= -f2 | tr -d '"')
OS_VERSION=$(grep ^VERSION_ID= /etc/os-release | cut -d= -f2 | tr -d '"')

print_color "=== Wazuh SIEM Installation Started ===" "$BLUE"
print_color "OS: $OS_ID $OS_VERSION" "$BLUE"

# 전제 조건 확인
check_prerequisites() {
    print_color "Checking prerequisites..." "$YELLOW"
    
    # 최소 시스템 요구사항 확인
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$RAM_GB" -lt 2 ]; then
        print_color "WARNING: Minimum 2GB RAM recommended (Current: ${RAM_GB}GB)" "$RED"
    fi
    
    # 디스크 공간 확인
    DISK_GB=$(df / | awk 'NR==2{print int($4/1048576)}')
    if [ "$DISK_GB" -lt 10 ]; then
        print_color "WARNING: Minimum 10GB disk space recommended (Available: ${DISK_GB}GB)" "$RED"
    fi
    
    # 네트워크 연결 확인
    if ! ping -c 1 packages.wazuh.com >/dev/null 2>&1; then
        print_color "ERROR: Cannot reach Wazuh repository" "$RED"
        exit 1
    fi
    
    print_color "Prerequisites check completed" "$GREEN"
}

# Wazuh 저장소 추가
setup_repository() {
    print_color "Setting up Wazuh repository..." "$YELLOW"
    
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            # CentOS/RHEL/Fedora
            rpm --import https://packages.wazuh.com/key/GPG-KEY-WAZUH
            cat > /etc/yum.repos.d/wazuh.repo << 'EOF'
[wazuh]
gpgcheck=1
gpgkey=https://packages.wazuh.com/key/GPG-KEY-WAZUH
enabled=1
name=EL-$releasever - Wazuh
baseurl=https://packages.wazuh.com/4.x/yum/
protect=1
EOF
            ;;
            
        "ubuntu"|"debian")
            # Ubuntu/Debian
            curl -s https://packages.wazuh.com/key/GPG-KEY-WAZUH | gpg --no-default-keyring --keyring gnupg-ring:/usr/share/keyrings/wazuh.gpg --import && chmod 644 /usr/share/keyrings/wazuh.gpg
            echo "deb [signed-by=/usr/share/keyrings/wazuh.gpg] https://packages.wazuh.com/4.x/apt/ stable main" | tee -a /etc/apt/sources.list.d/wazuh.list
            apt-get update
            ;;
            
        *)
            print_color "Unsupported OS: $OS_ID" "$RED"
            exit 1
            ;;
    esac
    
    print_color "Repository setup completed" "$GREEN"
}

# Wazuh Manager 설치
install_wazuh_manager() {
    print_color "Installing Wazuh Manager..." "$YELLOW"
    
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install -y wazuh-manager
            ;;
        "ubuntu"|"debian")
            apt-get install -y wazuh-manager
            ;;
    esac
    
    # 서비스 활성화
    systemctl daemon-reload
    systemctl enable wazuh-manager
    systemctl start wazuh-manager
    
    print_color "Wazuh Manager installed successfully" "$GREEN"
}

# Wazuh API 설치 및 설정
install_wazuh_api() {
    print_color "Installing and configuring Wazuh API..." "$YELLOW"
    
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install -y nodejs npm
            ;;
        "ubuntu"|"debian")
            apt-get install -y nodejs npm
            ;;
    esac
    
    # Wazuh API 설치
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install -y wazuh-api
            ;;
        "ubuntu"|"debian")
            apt-get install -y wazuh-api
            ;;
    esac
    
    # API 설정
    cat > /var/ossec/api/configuration/config.js << 'EOF'
var config = {};

config.port = "55000";
config.host = "0.0.0.0";
config.https = "no";
config.basic_auth = "yes";
config.BehindProxyServer = "no";
config.cors = "yes";

module.exports = config;
EOF
    
    # API 사용자 생성
    cd /var/ossec/api/scripts
    ./configure_api.sh
    
    systemctl enable wazuh-api
    systemctl start wazuh-api
    
    print_color "Wazuh API configured successfully" "$GREEN"
}

# Elastic Stack 설치
install_elastic_stack() {
    print_color "Installing Elastic Stack..." "$YELLOW"
    
    # Elasticsearch 저장소 추가
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            rpm --import https://artifacts.elastic.co/GPG-KEY-elasticsearch
            cat > /etc/yum.repos.d/elasticsearch.repo << 'EOF'
[elasticsearch]
name=Elasticsearch repository for 7.x packages
baseurl=https://artifacts.elastic.co/packages/7.x/yum
gpgcheck=1
gpgkey=https://artifacts.elastic.co/GPG-KEY-elasticsearch
enabled=0
autorefresh=1
type=rpm-md
EOF
            ;;
        "ubuntu"|"debian")
            wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
            echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list
            apt-get update
            ;;
    esac
    
    # Java 설치 (Elasticsearch 요구사항)
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install -y java-11-openjdk
            ;;
        "ubuntu"|"debian")
            apt-get install -y openjdk-11-jdk
            ;;
    esac
    
    # Elasticsearch 설치
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install --enablerepo=elasticsearch -y elasticsearch
            ;;
        "ubuntu"|"debian")
            apt-get install -y elasticsearch
            ;;
    esac
    
    # Elasticsearch 설정
    cat > /etc/elasticsearch/elasticsearch.yml << 'EOF'
cluster.name: wazuh-cluster
node.name: wazuh-node
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch
network.host: localhost
http.port: 9200
cluster.initial_master_nodes: ["wazuh-node"]
EOF
    
    # JVM 힙 크기 설정 (시스템 메모리의 50%)
    HEAP_SIZE=$((RAM_GB / 2))
    if [ $HEAP_SIZE -lt 1 ]; then
        HEAP_SIZE=1
    fi
    
    cat > /etc/elasticsearch/jvm.options.d/wazuh.options << EOF
-Xms${HEAP_SIZE}g
-Xmx${HEAP_SIZE}g
EOF
    
    systemctl daemon-reload
    systemctl enable elasticsearch
    systemctl start elasticsearch
    
    # Elasticsearch 시작 대기
    sleep 30
    
    # Kibana 설치
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install --enablerepo=elasticsearch -y kibana
            ;;
        "ubuntu"|"debian")
            apt-get install -y kibana
            ;;
    esac
    
    # Kibana 설정
    cat > /etc/kibana/kibana.yml << 'EOF'
server.port: 5601
server.host: "0.0.0.0"
elasticsearch.hosts: ["http://localhost:9200"]
logging.dest: /var/log/kibana/kibana.log
EOF
    
    systemctl enable kibana
    systemctl start kibana
    
    print_color "Elastic Stack installed successfully" "$GREEN"
}

# Wazuh Kibana 플러그인 설치
install_wazuh_kibana_plugin() {
    print_color "Installing Wazuh Kibana plugin..." "$YELLOW"
    
    # Kibana 정지
    systemctl stop kibana
    
    # 플러그인 설치
    sudo -u kibana /usr/share/kibana/bin/kibana-plugin install https://packages.wazuh.com/4.x/ui/kibana/wazuh_kibana-4.8.0_7.17.0-1.zip
    
    # Kibana 시작
    systemctl start kibana
    
    print_color "Wazuh Kibana plugin installed successfully" "$GREEN"
}

# Filebeat 설치 및 설정
install_filebeat() {
    print_color "Installing and configuring Filebeat..." "$YELLOW"
    
    case $OS_ID in
        "centos"|"rhel"|"fedora")
            yum install --enablerepo=elasticsearch -y filebeat
            ;;
        "ubuntu"|"debian")
            apt-get install -y filebeat
            ;;
    esac
    
    # Filebeat 설정
    curl -so /etc/filebeat/filebeat.yml https://raw.githubusercontent.com/wazuh/wazuh/4.8/extensions/filebeat/7.x/filebeat.yml
    curl -so /etc/filebeat/wazuh-template.json https://raw.githubusercontent.com/wazuh/wazuh/4.8/extensions/elasticsearch/7.x/wazuh-template.json
    
    # Wazuh 모듈 다운로드
    curl -s https://packages.wazuh.com/4.x/filebeat/wazuh-filebeat-0.4.tar.gz | tar -xvz -C /usr/share/filebeat/module
    
    # Elasticsearch 템플릿 및 파이프라인 설정
    filebeat setup --template
    
    systemctl daemon-reload
    systemctl enable filebeat
    systemctl start filebeat
    
    print_color "Filebeat configured successfully" "$GREEN"
}

# 방화벽 설정
configure_firewall() {
    print_color "Configuring firewall..." "$YELLOW"
    
    # 필요한 포트들
    # 1514/udp: Wazuh agents
    # 1515/tcp: Wazuh agents registration
    # 55000/tcp: Wazuh API
    # 5601/tcp: Kibana
    # 9200/tcp: Elasticsearch
    
    if command -v ufw >/dev/null 2>&1; then
        # Ubuntu UFW
        ufw allow 1514/udp
        ufw allow 1515/tcp
        ufw allow 55000/tcp
        ufw allow 5601/tcp
        ufw allow 9200/tcp
    elif command -v firewall-cmd >/dev/null 2>&1; then
        # CentOS/RHEL firewalld
        firewall-cmd --permanent --add-port=1514/udp
        firewall-cmd --permanent --add-port=1515/tcp
        firewall-cmd --permanent --add-port=55000/tcp
        firewall-cmd --permanent --add-port=5601/tcp
        firewall-cmd --permanent --add-port=9200/tcp
        firewall-cmd --reload
    fi
    
    print_color "Firewall configured successfully" "$GREEN"
}

# 설치 검증
verify_installation() {
    print_color "Verifying installation..." "$YELLOW"
    
    # 서비스 상태 확인
    SERVICES=("wazuh-manager" "wazuh-api" "elasticsearch" "kibana" "filebeat")
    
    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            print_color "✓ $service is running" "$GREEN"
        else
            print_color "✗ $service is not running" "$RED"
        fi
    done
    
    # 포트 확인
    print_color "\nPort status:" "$BLUE"
    ss -tlnp | grep -E ':(1514|1515|5601|9200|55000)'
    
    # Wazuh 에이전트 키 생성 예제
    print_color "\nGenerating example agent key..." "$YELLOW"
    /var/ossec/bin/manage_agents -a -n example-agent -i 001
    
    print_color "\n=== Installation Summary ===" "$BLUE"
    echo "Wazuh Manager: http://$(hostname -I | awk '{print $1}'):55000"
    echo "Kibana Dashboard: http://$(hostname -I | awk '{print $1}'):5601"
    echo "Elasticsearch: http://$(hostname -I | awk '{print $1}'):9200"
    echo ""
    echo "Default credentials:"
    echo "- Wazuh API: admin / admin"
    echo "- Change default passwords immediately!"
}

# 메인 설치 프로세스
main() {
    check_prerequisites
    setup_repository
    install_wazuh_manager
    install_wazuh_api
    install_elastic_stack
    install_wazuh_kibana_plugin
    install_filebeat
    configure_firewall
    verify_installation
    
    print_color "\n🎉 Wazuh SIEM installation completed successfully!" "$GREEN"
    print_color "Access Kibana at: http://$(hostname -I | awk '{print $1}'):5601" "$BLUE"
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

### 🛡️ Wazuh 고급 설정 및 규칙

#### 커스텀 보안 룰 작성
```bash
# /var/ossec/etc/rules/local_rules.xml
# Wazuh 커스텀 보안 룰

<group name="local,syslog,">

  <!-- 커스텀 SSH 공격 탐지 -->
  <rule id="100001" level="10">
    <if_sid>5720</if_sid>
    <match>^Failed|^error: PAM: Authentication failure</match>
    <description>SSH authentication failure.</description>
    <group>authentication_failed,pci_dss_10.2.4,pci_dss_10.2.5,</group>
  </rule>

  <rule id="100002" level="12">
    <if_sid>100001</if_sid>
    <same_source_ip />
    <description>SSH brute force attack detected (multiple authentication failures).</description>
    <mitre>
      <id>T1110</id>
    </mitre>
    <group>authentication_failures,pci_dss_10.2.4,pci_dss_10.2.5,</group>
  </rule>

  <!-- 웹 애플리케이션 공격 탐지 -->
  <rule id="100010" level="6">
    <if_sid>31100</if_sid>
    <url>admin|wp-admin|phpmyadmin|cpanel</url>
    <description>Attempt to access sensitive web directories.</description>
    <group>web,accesslog,attack,</group>
  </rule>

  <rule id="100011" level="8">
    <if_sid>31100</if_sid>
    <url>\.php\?|\.asp\?|\.jsp\?</url>
    <description>Potential web application vulnerability scan.</description>
    <group>web,accesslog,attack,</group>
  </rule>

  <rule id="100012" level="10">
    <if_sid>31100</if_sid>
    <url>union.*select|concat\(|exec\(|system\(</url>
    <description>SQL injection attempt detected.</description>
    <mitre>
      <id>T1190</id>
    </mitre>
    <group>web,accesslog,attack,sql_injection,</group>
  </rule>

  <!-- 파일 시스템 모니터링 -->
  <rule id="100020" level="7">
    <if_sid>550</if_sid>
    <field name="file">/etc/passwd|/etc/shadow|/etc/group</field>
    <description>Critical system file modified.</description>
    <group>syscheck,pci_dss_11.5,</group>
  </rule>

  <rule id="100021" level="12">
    <if_sid>554</if_sid>
    <field name="file">/bin/|/sbin/|/usr/bin/</field>
    <description>System binary file modified or deleted - possible rootkit.</description>
    <mitre>
      <id>T1014</id>
    </mitre>
    <group>syscheck,rootkit,pci_dss_11.5,</group>
  </rule>

  <!-- 네트워크 공격 탐지 -->
  <rule id="100030" level="8">
    <if_sid>4386</if_sid>
    <regex>DPT=(22|23|21|25|53|80|110|143|443|993|995|3389)</regex>
    <description>Port scan detected on critical services.</description>
    <group>recon,pci_dss_11.4,</group>
  </rule>

  <rule id="100031" level="10">
    <if_sid>100030</if_sid>
    <same_source_ip />
    <description>Multiple port scan attempts from same source.</description>
    <mitre>
      <id>T1046</id>
    </mitre>
    <group>recon,pci_dss_11.4,</group>
  </rule>

  <!-- 악성코드 및 의심스러운 프로세스 -->
  <rule id="100040" level="12">
    <if_sid>530</if_sid>
    <match>nc -l|ncat -l|/dev/tcp|/dev/udp</match>
    <description>Potential reverse shell or backdoor detected.</description>
    <mitre>
      <id>T1059</id>
    </mitre>
    <group>attack,malware,</group>
  </rule>

  <rule id="100041" level="10">
    <if_sid>530</if_sid>
    <match>wget.*\.(sh|py|pl)|curl.*\.(sh|py|pl)</match>
    <description>Suspicious script download detected.</description>
    <group>attack,malware,</group>
  </rule>

  <!-- 권한 상승 탐지 -->
  <rule id="100050" level="8">
    <if_sid>5401</if_sid>
    <user>root</user>
    <description>Successful sudo to root.</description>
    <group>privilege_escalation,pci_dss_10.2.2,</group>
  </rule>

  <rule id="100051" level="12">
    <if_sid>5402</if_sid>
    <same_user />
    <description>Multiple failed sudo attempts by same user.</description>
    <mitre>
      <id>T1548</id>
    </mitre>
    <group>privilege_escalation,pci_dss_10.2.2,</group>
  </rule>

  <!-- 데이터 유출 감지 -->
  <rule id="100060" level="10">
    <if_sid>31100</if_sid>
    <status>200</status>
    <size>1048576</size>
    <description>Large file download detected - potential data exfiltration.</description>
    <group>web,data_exfiltration,</group>
  </rule>

  <rule id="100061" level="8">
    <if_sid>530</if_sid>
    <match>scp.*-r|rsync.*-r|tar.*-c</match>
    <description>Bulk data transfer command detected.</description>
    <group>data_exfiltration,</group>
  </rule>

  <!-- 컨테이너 보안 -->
  <rule id="100070" level="10">
    <if_sid>530</if_sid>
    <match>docker.*--privileged|docker.*--cap-add</match>
    <description>Privileged container execution detected.</description>
    <group>docker,container_security,</group>
  </rule>

  <rule id="100071" level="8">
    <if_sid>530</if_sid>
    <match>kubectl.*create|kubectl.*apply</match>
    <description>Kubernetes resource creation detected.</description>
    <group>kubernetes,container_security,</group>
  </rule>

  <!-- 로그 변조 탐지 -->
  <rule id="100080" level="12">
    <if_sid>530</if_sid>
    <match>rm.*log|truncate.*log|>.*log</match>
    <description>Log file manipulation detected.</description>
    <mitre>
      <id>T1070.002</id>
    </mitre>
    <group>log_tampering,attack,</group>
  </rule>

  <!-- 암호화폐 채굴 탐지 -->
  <rule id="100090" level="10">
    <if_sid>530</if_sid>
    <match>xmrig|cpuminer|minerd|ccminer</match>
    <description>Cryptocurrency mining activity detected.</description>
    <group>malware,cryptomining,</group>
  </rule>

  <!-- 컴플라이언스 관련 이벤트 -->
  <rule id="100100" level="5">
    <if_sid>5501</if_sid>
    <user>audit</user>
    <description>Audit log access detected.</description>
    <group>audit,compliance,pci_dss_10.2.3,</group>
  </rule>

</group>
```

#### 고급 디코더 설정
```bash
# /var/ossec/etc/decoders/local_decoder.xml
# Wazuh 커스텀 디코더

<decoder name="custom-ssh">
  <parent>sshd</parent>
  <regex>Failed password for (\S+) from (\d+.\d+.\d+.\d+) port (\d+)</regex>
  <order>user,srcip,srcport</order>
</decoder>

<decoder name="custom-apache-error">
  <parent>apache-errorlog</parent>
  <regex>\[(\w+ \w+ \d+ \d+:\d+:\d+ \d+)\] \[(\w+)\] (\S+): (.+), referer: (\S+)</regex>
  <order>timestamp,level,client,message,referer</order>
</decoder>

<decoder name="custom-mysql">
  <parent>mysql_log</parent>
  <regex>(\d+) Connect\s+Access denied for user '(\S+)'@'(\S+)'</regex>
  <order>id,user,srcip</order>
</decoder>

<decoder name="custom-docker">
  <parent>json</parent>
  <regex>"container_name":"([^"]+)"</regex>
  <order>container_name</order>
</decoder>

<decoder name="custom-kubernetes">
  <parent>json</parent>
  <regex>"verb":"(\S+)","user":{"username":"([^"]+)"}</regex>
  <order>verb,username</order>
</decoder>
```

### 📊 Wazuh 자동화 및 대응

#### 자동 대응 스크립트
```bash
#!/bin/bash
# /var/ossec/active-response/bin/custom-response.sh
# Wazuh 자동 대응 스크립트

# 입력 파라미터 읽기
LOCAL=`echo $0 | cut -d '/' -f4`
PWD=`pwd`
read INPUT_JSON

# JSON 파싱 함수
get_json_value() {
    echo "$INPUT_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('$1', ''))
" 2>/dev/null || echo ""
}

# 로그 함수
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> /var/ossec/logs/active-responses.log
}

# 액션 및 매개변수 추출
ACTION=$(get_json_value "command")
SRCIP=$(get_json_value "parameters.alert.data.srcip")
RULE_ID=$(get_json_value "parameters.alert.rule.id")
USERNAME=$(get_json_value "parameters.alert.data.user")
FILENAME=$(get_json_value "parameters.alert.syscheck.path")

log_message "Active response triggered: Action=$ACTION, Rule=$RULE_ID, SrcIP=$SRCIP"

case $ACTION in
    "ban-ip")
        # IP 차단
        if [ -n "$SRCIP" ]; then
            # iptables 차단
            iptables -I INPUT -s "$SRCIP" -j DROP
            
            # Fail2Ban 차단 (Fail2Ban이 설치된 경우)
            if command -v fail2ban-client >/dev/null 2>&1; then
                fail2ban-client set sshd banip "$SRCIP"
            fi
            
            log_message "IP $SRCIP banned successfully"
            
            # 알림 발송
            /usr/local/bin/send-alert.sh "IP Banned" "IP $SRCIP has been banned due to rule $RULE_ID"
        fi
        ;;
        
    "disable-user")
        # 사용자 계정 비활성화
        if [ -n "$USERNAME" ]; then
            usermod -L "$USERNAME"
            log_message "User $USERNAME disabled due to suspicious activity"
            
            # 활성 세션 종료
            pkill -u "$USERNAME"
            
            # 알림 발송
            /usr/local/bin/send-alert.sh "User Disabled" "User $USERNAME has been disabled due to rule $RULE_ID"
        fi
        ;;
        
    "quarantine-file")
        # 파일 격리
        if [ -n "$FILENAME" ] && [ -f "$FILENAME" ]; then
            QUARANTINE_DIR="/var/ossec/quarantine/$(date +%Y%m%d)"
            mkdir -p "$QUARANTINE_DIR"
            
            # 파일 이동
            mv "$FILENAME" "$QUARANTINE_DIR/"
            log_message "File $FILENAME quarantined to $QUARANTINE_DIR"
            
            # 알림 발송
            /usr/local/bin/send-alert.sh "File Quarantined" "File $FILENAME has been quarantined due to rule $RULE_ID"
        fi
        ;;
        
    "restart-service")
        # 서비스 재시작
        SERVICE=$(get_json_value "parameters.service")
        if [ -n "$SERVICE" ]; then
            systemctl restart "$SERVICE"
            log_message "Service $SERVICE restarted due to rule $RULE_ID"
            
            # 알림 발송
            /usr/local/bin/send-alert.sh "Service Restarted" "Service $SERVICE has been restarted due to rule $RULE_ID"
        fi
        ;;
        
    "collect-evidence")
        # 증거 수집
        EVIDENCE_DIR="/var/ossec/evidence/$(date +%Y%m%d-%H%M%S)-rule$RULE_ID"
        mkdir -p "$EVIDENCE_DIR"
        
        # 시스템 정보 수집
        ps aux > "$EVIDENCE_DIR/processes.txt"
        netstat -tlnp > "$EVIDENCE_DIR/network.txt"
        ss -tlnp > "$EVIDENCE_DIR/sockets.txt"
        last -n 50 > "$EVIDENCE_DIR/logins.txt"
        
        # 로그 파일 복사
        if [ -n "$SRCIP" ]; then
            grep "$SRCIP" /var/log/auth.log > "$EVIDENCE_DIR/auth-logs.txt" 2>/dev/null || true
            grep "$SRCIP" /var/log/apache2/access.log > "$EVIDENCE_DIR/web-logs.txt" 2>/dev/null || true
        fi
        
        # 메모리 덤프 (선택사항)
        if command -v memdump >/dev/null 2>&1; then
            memdump > "$EVIDENCE_DIR/memory.dump" 2>/dev/null || true
        fi
        
        log_message "Evidence collected in $EVIDENCE_DIR"
        ;;
        
    *)
        log_message "Unknown action: $ACTION"
        ;;
esac

# 성공 상태 반환
exit 0
```

#### 알림 시스템 구축
```bash
#!/bin/bash
# /usr/local/bin/send-alert.sh
# 다중 채널 알림 시스템

SUBJECT="$1"
MESSAGE="$2"
SEVERITY="${3:-medium}"

# 설정 파일 로드
if [ -f /etc/wazuh/alert-config.conf ]; then
    source /etc/wazuh/alert-config.conf
fi

# 기본 설정
EMAIL_TO="${EMAIL_TO:-admin@example.com}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-}"
SMS_API_KEY="${SMS_API_KEY:-}"
SMS_TO="${SMS_TO:-}"

# 로그 함수
log_alert() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [$SEVERITY] $SUBJECT: $MESSAGE" >> /var/log/wazuh-alerts.log
}

# 이메일 발송
send_email() {
    if command -v mail >/dev/null 2>&1; then
        echo "$MESSAGE" | mail -s "Wazuh Alert: $SUBJECT" "$EMAIL_TO"
        echo "Email sent to $EMAIL_TO"
    fi
}

# Slack 알림
send_slack() {
    if [ -n "$SLACK_WEBHOOK" ]; then
        # 심각도에 따른 색상
        case $SEVERITY in
            "high"|"critical")
                COLOR="danger"
                EMOJI="🚨"
                ;;
            "medium")
                COLOR="warning"
                EMOJI="⚠️"
                ;;
            *)
                COLOR="good"
                EMOJI="ℹ️"
                ;;
        esac
        
        PAYLOAD=$(cat <<EOF
{
    "text": "$EMOJI Wazuh Security Alert",
    "attachments": [
        {
            "color": "$COLOR",
            "fields": [
                {
                    "title": "Alert",
                    "value": "$SUBJECT",
                    "short": true
                },
                {
                    "title": "Severity",
                    "value": "$SEVERITY",
                    "short": true
                },
                {
                    "title": "Details",
                    "value": "$MESSAGE",
                    "short": false
                },
                {
                    "title": "Timestamp",
                    "value": "$(date)",
                    "short": true
                },
                {
                    "title": "Host",
                    "value": "$(hostname)",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -X POST -H 'Content-type: application/json' \
             --data "$PAYLOAD" \
             "$SLACK_WEBHOOK" >/dev/null 2>&1
        
        echo "Slack notification sent"
    fi
}

# Discord 알림
send_discord() {
    if [ -n "$DISCORD_WEBHOOK" ]; then
        PAYLOAD=$(cat <<EOF
{
    "username": "Wazuh Security Bot",
    "embeds": [
        {
            "title": "🛡️ Security Alert: $SUBJECT",
            "description": "$MESSAGE",
            "color": 15158332,
            "fields": [
                {
                    "name": "Severity",
                    "value": "$SEVERITY",
                    "inline": true
                },
                {
                    "name": "Host",
                    "value": "$(hostname)",
                    "inline": true
                },
                {
                    "name": "Timestamp",
                    "value": "$(date)",
                    "inline": false
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -H "Content-Type: application/json" \
             -X POST \
             -d "$PAYLOAD" \
             "$DISCORD_WEBHOOK" >/dev/null 2>&1
        
        echo "Discord notification sent"
    fi
}

# SMS 발송 (Twilio API 예제)
send_sms() {
    if [ -n "$SMS_API_KEY" ] && [ -n "$SMS_TO" ] && [ "$SEVERITY" = "critical" ]; then
        SMS_MESSAGE="Wazuh CRITICAL Alert: $SUBJECT on $(hostname)"
        
        curl -X POST https://api.twilio.com/2010-04-01/Accounts/$TWILIO_SID/Messages.json \
             --data-urlencode "To=$SMS_TO" \
             --data-urlencode "From=$TWILIO_FROM" \
             --data-urlencode "Body=$SMS_MESSAGE" \
             -u "$TWILIO_SID:$SMS_API_KEY" >/dev/null 2>&1
        
        echo "SMS notification sent"
    fi
}

# 메인 실행
main() {
    log_alert
    
    # 심각도에 따른 알림 채널 선택
    case $SEVERITY in
        "critical")
            send_email
            send_slack
            send_discord
            send_sms
            ;;
        "high")
            send_email
            send_slack
            send_discord
            ;;
        "medium")
            send_slack
            ;;
        "low")
            log_alert  # 로그만 기록
            ;;
    esac
}

main
```

## Docker 컨테이너 보안 | Docker Container Security

### 🐳 Docker 보안 강화

#### 포괄적인 Docker 보안 설정
```bash
#!/bin/bash
# /usr/local/bin/docker-security-hardening.sh
# Docker 보안 강화 스크립트

set -e

print_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Docker 데몬 보안 설정
configure_docker_daemon() {
    print_status "Configuring Docker daemon security..."
    
    # Docker 데몬 설정 파일 생성
    mkdir -p /etc/docker
    
    cat > /etc/docker/daemon.json << 'EOF'
{
    "icc": false,
    "userns-remap": "default",
    "live-restore": true,
    "userland-proxy": false,
    "no-new-privileges": true,
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "default-ulimits": {
        "nofile": {
            "Hard": 64000,
            "Name": "nofile",
            "Soft": 64000
        }
    },
    "disable-legacy-registry": true,
    "experimental": false,
    "metrics-addr": "127.0.0.1:9323",
    "insecure-registries": []
}
EOF
    
    # Docker 서비스 재시작
    systemctl restart docker
    print_status "Docker daemon configured successfully"
}

# Docker 컨테이너 기본 보안 프로파일
create_security_profiles() {
    print_status "Creating security profiles..."
    
    # AppArmor 프로파일 생성
    if command -v aa-status >/dev/null 2>&1; then
        cat > /etc/apparmor.d/docker-default-secure << 'EOF'
#include <tunables/global>

profile docker-default-secure flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  
  # 네트워크 접근 제한
  network inet tcp,
  network inet udp,
  network inet6 tcp,
  network inet6 udp,
  network netlink raw,
  
  # 파일 시스템 접근 제한
  deny /etc/passwd r,
  deny /etc/shadow r,
  deny /etc/group r,
  deny /etc/gshadow r,
  deny /proc/sys/** w,
  deny /sys/** w,
  
  # 특권 상승 방지
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
  deny capability sys_pacct,
  deny capability sys_nice,
  deny capability sys_resource,
  deny capability sys_time,
  deny capability sys_tty_config,
  deny capability mknod,
  deny capability audit_write,
  deny capability audit_control,
  deny capability mac_override,
  deny capability mac_admin,
  deny capability net_admin,
  deny capability syslog,
  deny capability wake_alarm,
  deny capability block_suspend,
  
  # 허용된 capability들
  capability chown,
  capability dac_override,
  capability fowner,
  capability fsetid,
  capability kill,
  capability setgid,
  capability setuid,
  capability setpcap,
  capability linux_immutable,
  capability net_bind_service,
  capability net_broadcast,
  capability net_raw,
  capability ipc_lock,
  capability ipc_owner,
  capability sys_chroot,
  capability sys_ptrace,
  capability lease,
  capability audit_read,
}
EOF
        
        apparmor_parser -r /etc/apparmor.d/docker-default-secure
    fi
    
    # Seccomp 프로파일 생성
    cat > /etc/docker/seccomp-profile.json << 'EOF'
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "archMap": [
        {
            "architecture": "SCMP_ARCH_X86_64",
            "subArchitectures": [
                "SCMP_ARCH_X86",
                "SCMP_ARCH_X32"
            ]
        }
    ],
    "syscalls": [
        {
            "names": [
                "accept",
                "accept4", 
                "access",
                "adjtimex",
                "alarm",
                "bind",
                "brk",
                "chdir",
                "chmod",
                "chown",
                "chroot",
                "clock_getres",
                "clock_gettime",
                "clone",
                "close",
                "connect",
                "dup",
                "dup2",
                "dup3",
                "epoll_create",
                "epoll_create1",
                "epoll_ctl",
                "epoll_wait",
                "eventfd",
                "eventfd2",
                "execve",
                "exit",
                "exit_group",
                "fcntl",
                "fstat",
                "futex",
                "getcwd",
                "getdents",
                "getdents64",
                "getegid",
                "geteuid",
                "getgid",
                "getgroups",
                "getpeername",
                "getpgrp",
                "getpid",
                "getppid",
                "getrlimit",
                "getsid",
                "getsockname",
                "getsockopt",
                "gettid",
                "gettimeofday",
                "getuid",
                "listen",
                "lseek",
                "lstat",
                "madvise",
                "mmap",
                "mprotect",
                "munmap",
                "nanosleep",
                "open",
                "openat",
                "pause",
                "pipe",
                "pipe2",
                "poll",
                "ppoll",
                "prctl",
                "read",
                "recv",
                "recvfrom",
                "recvmsg",
                "rt_sigaction",
                "rt_sigprocmask",
                "rt_sigreturn",
                "sched_getaffinity",
                "sched_yield",
                "select",
                "send",
                "sendmsg",
                "sendto",
                "setgid",
                "setgroups",
                "setrlimit",
                "setsid",
                "setsockopt",
                "setuid",
                "shutdown",
                "sigaltstack",
                "socket",
                "socketpair",
                "stat",
                "statfs",
                "sysinfo",
                "time",
                "uname",
                "unlink",
                "unlinkat",
                "wait4",
                "waitpid",
                "write"
            ],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}
EOF
    
    print_status "Security profiles created successfully"
}

# Docker 이미지 보안 스캐너 설치
install_image_scanner() {
    print_status "Installing Docker image security scanner..."
    
    # Trivy 설치
    if ! command -v trivy >/dev/null 2>&1; then
        case $(uname -m) in
            x86_64)
                ARCH="64bit"
                ;;
            aarch64)
                ARCH="ARM64"
                ;;
            *)
                print_status "Unsupported architecture for Trivy"
                return 1
                ;;
        esac
        
        VERSION=$(curl -s "https://api.github.com/repos/aquasecurity/trivy/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
        wget -q "https://github.com/aquasecurity/trivy/releases/download/${VERSION}/trivy_${VERSION#v}_Linux-${ARCH}.tar.gz"
        tar -xzf "trivy_${VERSION#v}_Linux-${ARCH}.tar.gz"
        mv trivy /usr/local/bin/
        rm -f "trivy_${VERSION#v}_Linux-${ARCH}.tar.gz"
    fi
    
    print_status "Trivy installed successfully"
}

# Docker 컴포즈 보안 템플릿
create_secure_compose_template() {
    print_status "Creating secure Docker Compose template..."
    
    cat > /usr/local/share/docker-compose-secure-template.yml << 'EOF'
version: '3.8'

# 보안 강화된 Docker Compose 템플릿
services:
  app:
    image: your-app:latest
    
    # 보안 설정
    read_only: true                    # 루트 파일시스템 읽기 전용
    cap_drop:                         # 모든 capabilities 제거
      - ALL
    cap_add:                          # 필요한 capabilities만 추가
      - NET_BIND_SERVICE
    
    # 리소스 제한
    mem_limit: 512m
    memswap_limit: 512m
    cpu_count: 1
    pids_limit: 100
    
    # 네트워크 보안
    networks:
      - app-network
    ports:
      - "8080:8080"
    
    # 환경 변수 (secrets 사용 권장)
    environment:
      - NODE_ENV=production
    
    # 볼륨 마운트 (최소한으로)
    volumes:
      - app-data:/app/data
      - /tmp:/tmp:rw,noexec,nosuid,nodev
    
    # 보안 옵션
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default-secure
      - seccomp:/etc/docker/seccomp-profile.json
    
    # 재시작 정책
    restart: unless-stopped
    
    # 헬스체크
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    # 로그 설정
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # 데이터베이스 서비스 예제
  database:
    image: postgres:13-alpine
    
    # 보안 설정
    read_only: true
    cap_drop:
      - ALL
    cap_add:
      - SETUID
      - SETGID
      - DAC_OVERRIDE
    
    # 리소스 제한
    mem_limit: 1g
    memswap_limit: 1g
    
    # 환경 변수
    environment:
      POSTGRES_DB_FILE: /run/secrets/postgres_db
      POSTGRES_USER_FILE: /run/secrets/postgres_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    
    # Secrets 사용
    secrets:
      - postgres_db
      - postgres_user
      - postgres_password
    
    # 볼륨 설정
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - /tmp:/tmp:rw,noexec,nosuid,nodev
    
    # 네트워크
    networks:
      - db-network
    
    # 보안 옵션
    security_opt:
      - no-new-privileges:true
    
    # 헬스체크
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 5

# 네트워크 정의
networks:
  app-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
  db-network:
    driver: bridge
    internal: true                    # 외부 접근 차단

# 볼륨 정의
volumes:
  app-data:
    driver: local
  postgres-data:
    driver: local

# Secrets 정의
secrets:
  postgres_db:
    file: ./secrets/postgres_db.txt
  postgres_user:
    file: ./secrets/postgres_user.txt
  postgres_password:
    file: ./secrets/postgres_password.txt
EOF
    
    print_status "Secure Docker Compose template created"
}

# Docker 보안 스캔 스크립트
create_security_scanner() {
    print_status "Creating Docker security scanner..."
    
    cat > /usr/local/bin/docker-security-scan.sh << 'EOF'
#!/bin/bash
# Docker 보안 스캔 스크립트

IMAGE=$1
REPORT_DIR="/var/log/docker-security-scans"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

if [ -z "$IMAGE" ]; then
    echo "Usage: $0 <image-name>"
    exit 1
fi

mkdir -p "$REPORT_DIR"

echo "Starting security scan for image: $IMAGE"

# Trivy 취약점 스캔
echo "Running Trivy vulnerability scan..."
trivy image --format json --output "$REPORT_DIR/trivy-$TIMESTAMP.json" "$IMAGE"
trivy image --format table "$IMAGE" | tee "$REPORT_DIR/trivy-$TIMESTAMP.txt"

# Docker Bench 보안 스캔 (설치된 경우)
if command -v docker-bench-security >/dev/null 2>&1; then
    echo "Running Docker Bench security check..."
    docker-bench-security > "$REPORT_DIR/docker-bench-$TIMESTAMP.txt"
fi

# 이미지 히스토리 분석
echo "Analyzing image history..."
docker history --no-trunc "$IMAGE" > "$REPORT_DIR/history-$TIMESTAMP.txt"

# 이미지 구성 확인
echo "Checking image configuration..."
docker inspect "$IMAGE" > "$REPORT_DIR/inspect-$TIMESTAMP.json"

# 보안 권고사항 체크
echo "Checking security best practices..."
{
    echo "=== Security Check Results ==="
    echo "Image: $IMAGE"
    echo "Scan Date: $(date)"
    echo ""
    
    # 루트 사용자 체크
    if docker inspect "$IMAGE" | grep -q '"User": ""'; then
        echo "❌ FAIL: Image runs as root user"
    else
        echo "✅ PASS: Image does not run as root"
    fi
    
    # 불필요한 패키지 체크
    echo ""
    echo "=== Package Analysis ==="
    docker run --rm "$IMAGE" sh -c 'which wget curl nc netcat' 2>/dev/null | while read tool; do
        if [ -n "$tool" ]; then
            echo "⚠️  WARNING: Found potentially dangerous tool: $tool"
        fi
    done
    
} > "$REPORT_DIR/security-check-$TIMESTAMP.txt"

echo "Security scan completed. Reports saved in: $REPORT_DIR"
EOF
    
    chmod +x /usr/local/bin/docker-security-scan.sh
    print_status "Docker security scanner created"
}

# 메인 실행
main() {
    print_status "Starting Docker security hardening..."
    
    configure_docker_daemon
    create_security_profiles
    install_image_scanner
    create_secure_compose_template
    create_security_scanner
    
    print_status "Docker security hardening completed successfully!"
    
    echo ""
    echo "Next steps:"
    echo "1. Review Docker daemon configuration in /etc/docker/daemon.json"
    echo "2. Use secure Docker Compose template: /usr/local/share/docker-compose-secure-template.yml"
    echo "3. Scan images with: /usr/local/bin/docker-security-scan.sh <image-name>"
    echo "4. Apply security profiles to containers"
}

main "$@"
```

## 다음 편 예고

다음 포스트에서는 **Kubernetes 보안과 컴플라이언스**를 상세히 다룰 예정입니다:
- Kubernetes 클러스터 보안 강화
- Pod Security Standards 및 정책
- Network Policy와 Service Mesh 보안
- 컴플라이언스 자동화 및 감사

Wazuh SIEM과 Docker 보안으로 엔터프라이즈급 보안 인프라를 완성하셨나요? 🔍🐳🛡️
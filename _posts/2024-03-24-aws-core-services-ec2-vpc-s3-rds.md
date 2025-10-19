---
layout: post
title: "AWS 완전 정복 가이드 2편 - 핵심 서비스 실습과 EC2 마스터 | AWS Complete Guide Part 2 - Core Services & EC2 Mastery"
date: 2024-03-24 14:00:00 +0900
categories: [AWS, Cloud]
tags: [aws, ec2, vpc, s3, rds, hands-on, core-services, compute]
---

AWS의 핵심 서비스들을 실제 실습을 통해 완전히 마스터해보겠습니다. EC2부터 VPC, S3, RDS까지 실무에서 바로 활용할 수 있는 모든 것을 다룹니다.

## EC2 완전 마스터 | Complete EC2 Mastery

### 🖥️ EC2 인스턴스 생성 및 관리

#### EC2 인스턴스 타입 선택 가이드
```bash
# 인스턴스 패밀리별 특징
범용: t3, t4g, m5, m6i
├── t3.micro (1 vCPU, 1GB RAM) - 테스트/개발
├── t3.small (2 vCPU, 2GB RAM) - 소규모 웹사이트
├── m5.large (2 vCPU, 8GB RAM) - 일반적인 워크로드
└── m5.xlarge (4 vCPU, 16GB RAM) - 중간 규모 애플리케이션

컴퓨팅 최적화: c5, c6i
├── c5.large (2 vCPU, 4GB RAM) - CPU 집약적 작업
└── c5.xlarge (4 vCPU, 8GB RAM) - 고성능 웹 서버

메모리 최적화: r5, r6i, x1e
├── r5.large (2 vCPU, 16GB RAM) - 인메모리 데이터베이스
└── r5.xlarge (4 vCPU, 32GB RAM) - 대용량 메모리 필요 작업

스토리지 최적화: i3, d2, h1
├── i3.large (2 vCPU, 15.25GB RAM) - NoSQL 데이터베이스
└── d2.xlarge (4 vCPU, 30.5GB RAM) - 빅데이터 처리

가속화된 컴퓨팅: p3, g4, inf1
├── p3.2xlarge (8 vCPU, 61GB RAM, 1 GPU) - 머신러닝
└── g4dn.xlarge (4 vCPU, 16GB RAM, 1 GPU) - 그래픽 워크로드
```

#### CLI를 통한 EC2 인스턴스 관리
```bash
#!/bin/bash
# EC2 인스턴스 관리 스크립트

# AWS CLI 설정 확인
aws configure list

# 사용 가능한 AMI 조회
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=amzn2-ami-hvm-*" \
          "Name=architecture,Values=x86_64" \
          "Name=virtualization-type,Values=hvm" \
  --query 'Images[*].[ImageId,Name,CreationDate]' \
  --output table

# 키 페어 생성
aws ec2 create-key-pair \
  --key-name MyKeyPair \
  --query 'KeyMaterial' \
  --output text > MyKeyPair.pem

chmod 400 MyKeyPair.pem

# 보안 그룹 생성
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
  --group-name WebServerSecurityGroup \
  --description "Security group for web server" \
  --query 'GroupId' \
  --output text)

# SSH 접근 허용
aws ec2 authorize-security-group-ingress \
  --group-id $SECURITY_GROUP_ID \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# HTTP 접근 허용
aws ec2 authorize-security-group-ingress \
  --group-id $SECURITY_GROUP_ID \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

# HTTPS 접근 허용
aws ec2 authorize-security-group-ingress \
  --group-id $SECURITY_GROUP_ID \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# 사용자 데이터 스크립트 작성
cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y httpd
systemctl start httpd
systemctl enable httpd

# 간단한 웹페이지 생성
cat > /var/www/html/index.html << 'HTML'
<!DOCTYPE html>
<html>
<head>
    <title>AWS EC2 Instance</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #232F3E; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .instance-info { background-color: #f0f0f0; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AWS EC2 웹 서버가 성공적으로 실행 중입니다!</h1>
    </div>
    <div class="content">
        <h2>인스턴스 정보</h2>
        <div class="instance-info">
            <p><strong>인스턴스 ID:</strong> $(curl -s http://169.254.169.254/latest/meta-data/instance-id)</p>
            <p><strong>가용 영역:</strong> $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)</p>
            <p><strong>인스턴스 타입:</strong> $(curl -s http://169.254.169.254/latest/meta-data/instance-type)</p>
            <p><strong>퍼블릭 IP:</strong> $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)</p>
        </div>
        <h2>설치된 소프트웨어</h2>
        <ul>
            <li>Apache HTTP Server</li>
            <li>Amazon Linux 2</li>
        </ul>
    </div>
</body>
</html>
HTML

# 인스턴스 정보를 실시간으로 업데이트
cat > /var/www/html/info.php << 'PHP'
<?php
echo "<h2>실시간 시스템 정보</h2>";
echo "<p>현재 시간: " . date('Y-m-d H:i:s') . "</p>";
echo "<p>서버 가동시간: " . shell_exec('uptime') . "</p>";
echo "<p>메모리 사용량: " . shell_exec('free -h') . "</p>";
echo "<p>디스크 사용량: " . shell_exec('df -h') . "</p>";
?>
PHP

# CloudWatch 에이전트 설치
yum install -y amazon-cloudwatch-agent
EOF

# EC2 인스턴스 시작
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --count 1 \
  --instance-type t3.micro \
  --key-name MyKeyPair \
  --security-group-ids $SECURITY_GROUP_ID \
  --user-data file://user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "인스턴스 $INSTANCE_ID 시작 중..."

# 인스턴스 상태 대기
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# 퍼블릭 IP 조회
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "인스턴스가 성공적으로 시작되었습니다!"
echo "퍼블릭 IP: $PUBLIC_IP"
echo "웹사이트: http://$PUBLIC_IP"
echo "SSH 접속: ssh -i MyKeyPair.pem ec2-user@$PUBLIC_IP"
```

#### EC2 고급 관리 기능
```bash
# EC2 인스턴스 고급 관리 스크립트
#!/bin/bash

# 인스턴스 모니터링 활성화
enable_detailed_monitoring() {
    local instance_id=$1
    aws ec2 monitor-instances --instance-ids $instance_id
    echo "상세 모니터링이 활성화되었습니다: $instance_id"
}

# EBS 볼륨 생성 및 연결
create_and_attach_volume() {
    local instance_id=$1
    local volume_size=$2
    local device_name=$3
    
    # 인스턴스의 가용 영역 확인
    AZ=$(aws ec2 describe-instances \
      --instance-ids $instance_id \
      --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
      --output text)
    
    # EBS 볼륨 생성
    VOLUME_ID=$(aws ec2 create-volume \
      --size $volume_size \
      --volume-type gp3 \
      --availability-zone $AZ \
      --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=DataVolume}]' \
      --query 'VolumeId' \
      --output text)
    
    # 볼륨 생성 대기
    aws ec2 wait volume-available --volume-ids $VOLUME_ID
    
    # 볼륨 연결
    aws ec2 attach-volume \
      --volume-id $VOLUME_ID \
      --instance-id $instance_id \
      --device $device_name
    
    echo "볼륨 $VOLUME_ID가 $instance_id에 연결되었습니다 ($device_name)"
}

# 스냅샷 생성
create_snapshot() {
    local volume_id=$1
    local description=$2
    
    SNAPSHOT_ID=$(aws ec2 create-snapshot \
      --volume-id $volume_id \
      --description "$description" \
      --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=BackupSnapshot}]' \
      --query 'SnapshotId' \
      --output text)
    
    echo "스냅샷 생성 중: $SNAPSHOT_ID"
    return $SNAPSHOT_ID
}

# AMI 생성
create_ami() {
    local instance_id=$1
    local ami_name=$2
    local description=$3
    
    AMI_ID=$(aws ec2 create-image \
      --instance-id $instance_id \
      --name "$ami_name" \
      --description "$description" \
      --no-reboot \
      --query 'ImageId' \
      --output text)
    
    echo "AMI 생성 중: $AMI_ID"
    return $AMI_ID
}

# 인스턴스 크기 조정
resize_instance() {
    local instance_id=$1
    local new_instance_type=$2
    
    echo "인스턴스 중지 중..."
    aws ec2 stop-instances --instance-ids $instance_id
    aws ec2 wait instance-stopped --instance-ids $instance_id
    
    echo "인스턴스 타입 변경 중..."
    aws ec2 modify-instance-attribute \
      --instance-id $instance_id \
      --instance-type Value=$new_instance_type
    
    echo "인스턴스 시작 중..."
    aws ec2 start-instances --instance-ids $instance_id
    aws ec2 wait instance-running --instance-ids $instance_id
    
    echo "인스턴스 크기 조정 완료: $new_instance_type"
}

# 사용 예시
# enable_detailed_monitoring "i-1234567890abcdef0"
# create_and_attach_volume "i-1234567890abcdef0" 20 "/dev/sdf"
# create_snapshot "vol-1234567890abcdef0" "Daily backup"
# create_ami "i-1234567890abcdef0" "WebServer-v1.0" "Production web server AMI"
# resize_instance "i-1234567890abcdef0" "t3.small"
```

## VPC 네트워킹 심화 | Advanced VPC Networking

### 🌐 VPC 설계 및 구축

#### 완전한 VPC 인프라 구축
```bash
#!/bin/bash
# 프로덕션 레벨 VPC 인프라 구축 스크립트

# 변수 설정
VPC_CIDR="10.0.0.0/16"
PUBLIC_SUBNET_1_CIDR="10.0.1.0/24"
PUBLIC_SUBNET_2_CIDR="10.0.2.0/24"
PRIVATE_SUBNET_1_CIDR="10.0.10.0/24"
PRIVATE_SUBNET_2_CIDR="10.0.20.0/24"
DB_SUBNET_1_CIDR="10.0.100.0/24"
DB_SUBNET_2_CIDR="10.0.200.0/24"

# 가용 영역 조회
AZ1=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].ZoneName' --output text)
AZ2=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[1].ZoneName' --output text)

echo "가용 영역: $AZ1, $AZ2"

# 1. VPC 생성
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block $VPC_CIDR \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=Production-VPC}]' \
  --query 'Vpc.VpcId' \
  --output text)

echo "VPC 생성됨: $VPC_ID"

# DNS 호스트명 활성화
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support

# 2. 인터넷 게이트웨이 생성 및 연결
IGW_ID=$(aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=Production-IGW}]' \
  --query 'InternetGateway.InternetGatewayId' \
  --output text)

aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID
echo "인터넷 게이트웨이 생성 및 연결됨: $IGW_ID"

# 3. 퍼블릭 서브넷 생성
PUBLIC_SUBNET_1_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block $PUBLIC_SUBNET_1_CIDR \
  --availability-zone $AZ1 \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-1}]' \
  --query 'Subnet.SubnetId' \
  --output text)

PUBLIC_SUBNET_2_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block $PUBLIC_SUBNET_2_CIDR \
  --availability-zone $AZ2 \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Public-Subnet-2}]' \
  --query 'Subnet.SubnetId' \
  --output text)

# 퍼블릭 IP 자동 할당 활성화
aws ec2 modify-subnet-attribute --subnet-id $PUBLIC_SUBNET_1_ID --map-public-ip-on-launch
aws ec2 modify-subnet-attribute --subnet-id $PUBLIC_SUBNET_2_ID --map-public-ip-on-launch

echo "퍼블릭 서브넷 생성됨: $PUBLIC_SUBNET_1_ID, $PUBLIC_SUBNET_2_ID"

# 4. 프라이빗 서브넷 생성
PRIVATE_SUBNET_1_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block $PRIVATE_SUBNET_1_CIDR \
  --availability-zone $AZ1 \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-1}]' \
  --query 'Subnet.SubnetId' \
  --output text)

PRIVATE_SUBNET_2_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block $PRIVATE_SUBNET_2_CIDR \
  --availability-zone $AZ2 \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=Private-Subnet-2}]' \
  --query 'Subnet.SubnetId' \
  --output text)

echo "프라이빗 서브넷 생성됨: $PRIVATE_SUBNET_1_ID, $PRIVATE_SUBNET_2_ID"

# 5. 데이터베이스 서브넷 생성
DB_SUBNET_1_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block $DB_SUBNET_1_CIDR \
  --availability-zone $AZ1 \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=DB-Subnet-1}]' \
  --query 'Subnet.SubnetId' \
  --output text)

DB_SUBNET_2_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block $DB_SUBNET_2_CIDR \
  --availability-zone $AZ2 \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=DB-Subnet-2}]' \
  --query 'Subnet.SubnetId' \
  --output text)

echo "데이터베이스 서브넷 생성됨: $DB_SUBNET_1_ID, $DB_SUBNET_2_ID"

# 6. NAT 게이트웨이용 Elastic IP 할당
EIP_1_ID=$(aws ec2 allocate-address \
  --domain vpc \
  --tag-specifications 'ResourceType=elastic-ip,Tags=[{Key=Name,Value=NAT-Gateway-1-EIP}]' \
  --query 'AllocationId' \
  --output text)

EIP_2_ID=$(aws ec2 allocate-address \
  --domain vpc \
  --tag-specifications 'ResourceType=elastic-ip,Tags=[{Key=Name,Value=NAT-Gateway-2-EIP}]' \
  --query 'AllocationId' \
  --output text)

echo "Elastic IP 할당됨: $EIP_1_ID, $EIP_2_ID"

# 7. NAT 게이트웨이 생성
NAT_GW_1_ID=$(aws ec2 create-nat-gateway \
  --subnet-id $PUBLIC_SUBNET_1_ID \
  --allocation-id $EIP_1_ID \
  --tag-specifications 'ResourceType=nat-gateway,Tags=[{Key=Name,Value=NAT-Gateway-1}]' \
  --query 'NatGateway.NatGatewayId' \
  --output text)

NAT_GW_2_ID=$(aws ec2 create-nat-gateway \
  --subnet-id $PUBLIC_SUBNET_2_ID \
  --allocation-id $EIP_2_ID \
  --tag-specifications 'ResourceType=nat-gateway,Tags=[{Key=Name,Value=NAT-Gateway-2}]' \
  --query 'NatGateway.NatGatewayId' \
  --output text)

echo "NAT 게이트웨이 생성 중: $NAT_GW_1_ID, $NAT_GW_2_ID"

# NAT 게이트웨이 사용 가능 대기
aws ec2 wait nat-gateway-available --nat-gateway-ids $NAT_GW_1_ID $NAT_GW_2_ID

# 8. 라우팅 테이블 생성 및 설정

# 퍼블릭 라우팅 테이블
PUBLIC_RT_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=Public-Route-Table}]' \
  --query 'RouteTable.RouteTableId' \
  --output text)

# 인터넷 게이트웨이로의 라우트 추가
aws ec2 create-route \
  --route-table-id $PUBLIC_RT_ID \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id $IGW_ID

# 퍼블릭 서브넷과 연결
aws ec2 associate-route-table --subnet-id $PUBLIC_SUBNET_1_ID --route-table-id $PUBLIC_RT_ID
aws ec2 associate-route-table --subnet-id $PUBLIC_SUBNET_2_ID --route-table-id $PUBLIC_RT_ID

echo "퍼블릭 라우팅 테이블 설정 완료: $PUBLIC_RT_ID"

# 프라이빗 라우팅 테이블 1
PRIVATE_RT_1_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=Private-Route-Table-1}]' \
  --query 'RouteTable.RouteTableId' \
  --output text)

aws ec2 create-route \
  --route-table-id $PRIVATE_RT_1_ID \
  --destination-cidr-block 0.0.0.0/0 \
  --nat-gateway-id $NAT_GW_1_ID

aws ec2 associate-route-table --subnet-id $PRIVATE_SUBNET_1_ID --route-table-id $PRIVATE_RT_1_ID

# 프라이빗 라우팅 테이블 2
PRIVATE_RT_2_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=Private-Route-Table-2}]' \
  --query 'RouteTable.RouteTableId' \
  --output text)

aws ec2 create-route \
  --route-table-id $PRIVATE_RT_2_ID \
  --destination-cidr-block 0.0.0.0/0 \
  --nat-gateway-id $NAT_GW_2_ID

aws ec2 associate-route-table --subnet-id $PRIVATE_SUBNET_2_ID --route-table-id $PRIVATE_RT_2_ID

echo "프라이빗 라우팅 테이블 설정 완료: $PRIVATE_RT_1_ID, $PRIVATE_RT_2_ID"

# 데이터베이스 라우팅 테이블
DB_RT_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=DB-Route-Table}]' \
  --query 'RouteTable.RouteTableId' \
  --output text)

aws ec2 associate-route-table --subnet-id $DB_SUBNET_1_ID --route-table-id $DB_RT_ID
aws ec2 associate-route-table --subnet-id $DB_SUBNET_2_ID --route-table-id $DB_RT_ID

echo "데이터베이스 라우팅 테이블 설정 완료: $DB_RT_ID"

# 9. 보안 그룹 생성

# 웹 서버 보안 그룹
WEB_SG_ID=$(aws ec2 create-security-group \
  --group-name WebServer-SG \
  --description "Security group for web servers" \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=WebServer-SG}]' \
  --query 'GroupId' \
  --output text)

aws ec2 authorize-security-group-ingress --group-id $WEB_SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $WEB_SG_ID --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $WEB_SG_ID --protocol tcp --port 22 --source-group $BASTION_SG_ID

# 베스천 호스트 보안 그룹
BASTION_SG_ID=$(aws ec2 create-security-group \
  --group-name Bastion-SG \
  --description "Security group for bastion hosts" \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=Bastion-SG}]' \
  --query 'GroupId' \
  --output text)

aws ec2 authorize-security-group-ingress --group-id $BASTION_SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0

# 데이터베이스 보안 그룹
DB_SG_ID=$(aws ec2 create-security-group \
  --group-name Database-SG \
  --description "Security group for databases" \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=Database-SG}]' \
  --query 'GroupId' \
  --output text)

aws ec2 authorize-security-group-ingress --group-id $DB_SG_ID --protocol tcp --port 3306 --source-group $WEB_SG_ID

echo "보안 그룹 생성 완료: Web=$WEB_SG_ID, Bastion=$BASTION_SG_ID, DB=$DB_SG_ID"

# 10. VPC 플로우 로그 활성화
aws ec2 create-flow-logs \
  --resource-type VPC \
  --resource-ids $VPC_ID \
  --traffic-type ALL \
  --log-destination-type cloud-watch-logs \
  --log-group-name VPCFlowLogs \
  --deliver-logs-permission-arn arn:aws:iam::123456789012:role/flowlogsRole

echo "=== VPC 인프라 구축 완료 ==="
echo "VPC ID: $VPC_ID"
echo "퍼블릭 서브넷: $PUBLIC_SUBNET_1_ID, $PUBLIC_SUBNET_2_ID"
echo "프라이빗 서브넷: $PRIVATE_SUBNET_1_ID, $PRIVATE_SUBNET_2_ID"
echo "데이터베이스 서브넷: $DB_SUBNET_1_ID, $DB_SUBNET_2_ID"
echo "NAT 게이트웨이: $NAT_GW_1_ID, $NAT_GW_2_ID"
```

## S3 스토리지 완전 활용 | Complete S3 Storage Utilization

### 🗄️ S3 고급 활용 및 관리

#### S3 버킷 생성 및 정책 설정
```bash
#!/bin/bash
# S3 종합 관리 스크립트

BUCKET_NAME="my-production-bucket-$(date +%s)"
REGION="us-east-1"

# S3 버킷 생성
create_s3_bucket() {
    echo "S3 버킷 생성 중: $BUCKET_NAME"
    
    if [ "$REGION" != "us-east-1" ]; then
        aws s3api create-bucket \
          --bucket $BUCKET_NAME \
          --region $REGION \
          --create-bucket-configuration LocationConstraint=$REGION
    else
        aws s3api create-bucket --bucket $BUCKET_NAME --region $REGION
    fi
    
    # 퍼블릭 액세스 차단 (기본 보안)
    aws s3api put-public-access-block \
      --bucket $BUCKET_NAME \
      --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
    
    echo "S3 버킷 생성 완료: $BUCKET_NAME"
}

# 버킷 정책 설정
setup_bucket_policy() {
    cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyInsecureConnections",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        },
        {
            "Sid": "AllowSpecificIPAccess",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::$BUCKET_NAME/*",
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": [
                        "203.0.113.0/24",
                        "198.51.100.0/24"
                    ]
                }
            }
        }
    ]
}
EOF

    aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json
    echo "버킷 정책 적용 완료"
}

# 수명 주기 정책 설정
setup_lifecycle_policy() {
    cat > lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "TransitionToIA",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "documents/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 365,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ]
        },
        {
            "ID": "DeleteIncompleteMultipartUploads",
            "Status": "Enabled",
            "Filter": {},
            "AbortIncompleteMultipartUpload": {
                "DaysAfterInitiation": 7
            }
        },
        {
            "ID": "DeleteOldVersions",
            "Status": "Enabled",
            "Filter": {},
            "NoncurrentVersionTransitions": [
                {
                    "NoncurrentDays": 30,
                    "StorageClass": "GLACIER"
                }
            ],
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 365
            }
        }
    ]
}
EOF

    aws s3api put-bucket-lifecycle-configuration \
      --bucket $BUCKET_NAME \
      --lifecycle-configuration file://lifecycle-policy.json
    
    echo "수명 주기 정책 적용 완료"
}

# 버전 관리 활성화
enable_versioning() {
    aws s3api put-bucket-versioning \
      --bucket $BUCKET_NAME \
      --versioning-configuration Status=Enabled
    
    echo "버전 관리 활성화 완료"
}

# 서버 측 암호화 설정
setup_encryption() {
    cat > encryption-config.json << EOF
{
    "Rules": [
        {
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            },
            "BucketKeyEnabled": true
        }
    ]
}
EOF

    aws s3api put-bucket-encryption \
      --bucket $BUCKET_NAME \
      --server-side-encryption-configuration file://encryption-config.json
    
    echo "서버 측 암호화 설정 완료"
}

# 로깅 설정
setup_logging() {
    # 액세스 로그를 저장할 버킷 (기존 버킷이나 별도 로그 버킷 사용)
    LOG_BUCKET="$BUCKET_NAME-logs"
    
    aws s3api create-bucket --bucket $LOG_BUCKET --region $REGION
    
    cat > logging-config.json << EOF
{
    "LoggingEnabled": {
        "TargetBucket": "$LOG_BUCKET",
        "TargetPrefix": "access-logs/"
    }
}
EOF

    aws s3api put-bucket-logging \
      --bucket $BUCKET_NAME \
      --bucket-logging-status file://logging-config.json
    
    echo "액세스 로깅 설정 완료"
}

# CloudFront 배포 생성 (S3와 연동)
create_cloudfront_distribution() {
    cat > cloudfront-config.json << EOF
{
    "CallerReference": "$(date +%s)",
    "Comment": "CloudFront distribution for S3 bucket",
    "DefaultRootObject": "index.html",
    "Origins": {
        "Quantity": 1,
        "Items": [
            {
                "Id": "S3-$BUCKET_NAME",
                "DomainName": "$BUCKET_NAME.s3.amazonaws.com",
                "S3OriginConfig": {
                    "OriginAccessIdentity": ""
                }
            }
        ]
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": "S3-$BUCKET_NAME",
        "ViewerProtocolPolicy": "redirect-to-https",
        "MinTTL": 0,
        "ForwardedValues": {
            "QueryString": false,
            "Cookies": {
                "Forward": "none"
            }
        }
    },
    "Enabled": true,
    "PriceClass": "PriceClass_100"
}
EOF

    DISTRIBUTION_ID=$(aws cloudfront create-distribution \
      --distribution-config file://cloudfront-config.json \
      --query 'Distribution.Id' \
      --output text)
    
    echo "CloudFront 배포 생성 완료: $DISTRIBUTION_ID"
}

# S3 동기화 및 업로드 스크립트
sync_content() {
    local source_dir=$1
    local target_prefix=$2
    
    echo "컨텐츠 동기화 중: $source_dir -> s3://$BUCKET_NAME/$target_prefix"
    
    aws s3 sync "$source_dir" "s3://$BUCKET_NAME/$target_prefix" \
      --delete \
      --exact-timestamps \
      --exclude "*.tmp" \
      --exclude ".DS_Store" \
      --exclude "Thumbs.db"
    
    echo "동기화 완료"
}

# 메인 실행 함수
main() {
    create_s3_bucket
    setup_bucket_policy
    setup_lifecycle_policy
    enable_versioning
    setup_encryption
    setup_logging
    
    echo "=== S3 버킷 설정 완료 ==="
    echo "버킷 이름: $BUCKET_NAME"
    echo "리전: $REGION"
    echo "기능: 버전 관리, 암호화, 수명 주기 정책, 액세스 로깅"
}

# 스크립트 실행
main
```

## RDS 데이터베이스 구축 및 운영 | RDS Database Setup & Operations

### 🗃️ RDS 완전 마스터

#### RDS 인스턴스 생성 및 관리
```bash
#!/bin/bash
# RDS 종합 관리 스크립트

DB_INSTANCE_IDENTIFIER="production-mysql-db"
DB_NAME="productiondb"
MASTER_USERNAME="admin"
MASTER_PASSWORD="MySecurePassword123!"
DB_INSTANCE_CLASS="db.t3.micro"
ENGINE="mysql"
ENGINE_VERSION="8.0.35"
ALLOCATED_STORAGE=20
VPC_SECURITY_GROUP_ID="sg-xxxxxxxxx"  # 이전에 생성한 DB 보안 그룹 ID

# DB 서브넷 그룹 생성
create_db_subnet_group() {
    aws rds create-db-subnet-group \
      --db-subnet-group-name production-db-subnet-group \
      --db-subnet-group-description "Subnet group for production database" \
      --subnet-ids subnet-xxxxxxxxx subnet-yyyyyyyyy \
      --tags Key=Name,Value=Production-DB-Subnet-Group
    
    echo "DB 서브넷 그룹 생성 완료"
}

# RDS 인스턴스 생성
create_rds_instance() {
    echo "RDS 인스턴스 생성 중..."
    
    aws rds create-db-instance \
      --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
      --db-instance-class $DB_INSTANCE_CLASS \
      --engine $ENGINE \
      --engine-version $ENGINE_VERSION \
      --master-username $MASTER_USERNAME \
      --master-user-password $MASTER_PASSWORD \
      --allocated-storage $ALLOCATED_STORAGE \
      --db-name $DB_NAME \
      --vpc-security-group-ids $VPC_SECURITY_GROUP_ID \
      --db-subnet-group-name production-db-subnet-group \
      --backup-retention-period 7 \
      --backup-window "03:00-04:00" \
      --maintenance-window "sun:04:00-sun:05:00" \
      --multi-az \
      --storage-type gp2 \
      --storage-encrypted \
      --monitoring-interval 60 \
      --monitoring-role-arn arn:aws:iam::123456789012:role/rds-monitoring-role \
      --enable-performance-insights \
      --performance-insights-retention-period 7 \
      --deletion-protection \
      --tags Key=Name,Value=Production-MySQL-DB Key=Environment,Value=Production
    
    echo "RDS 인스턴스 생성 중... 완료까지 10-15분 소요됩니다."
    
    # 인스턴스 사용 가능 대기
    aws rds wait db-instance-available --db-instance-identifier $DB_INSTANCE_IDENTIFIER
    
    echo "RDS 인스턴스 생성 완료!"
}

# 읽기 전용 복제본 생성
create_read_replica() {
    local replica_identifier="${DB_INSTANCE_IDENTIFIER}-read-replica"
    
    aws rds create-db-instance-read-replica \
      --db-instance-identifier $replica_identifier \
      --source-db-instance-identifier $DB_INSTANCE_IDENTIFIER \
      --db-instance-class $DB_INSTANCE_CLASS \
      --publicly-accessible \
      --multi-az \
      --storage-encrypted \
      --tags Key=Name,Value=Production-MySQL-DB-Replica Key=Environment,Value=Production
    
    echo "읽기 전용 복제본 생성 중: $replica_identifier"
}

# 데이터베이스 스냅샷 생성
create_snapshot() {
    local snapshot_identifier="${DB_INSTANCE_IDENTIFIER}-snapshot-$(date +%Y%m%d-%H%M%S)"
    
    aws rds create-db-snapshot \
      --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
      --db-snapshot-identifier $snapshot_identifier \
      --tags Key=Name,Value=Manual-Snapshot Key=CreatedBy,Value=Script
    
    echo "스냅샷 생성 중: $snapshot_identifier"
}

# 자동 백업 설정 수정
modify_backup_settings() {
    aws rds modify-db-instance \
      --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
      --backup-retention-period 14 \
      --backup-window "02:00-03:00" \
      --apply-immediately
    
    echo "백업 설정 수정 완료 (보존 기간: 14일)"
}

# 데이터베이스 연결 및 초기 설정
setup_database() {
    # RDS 엔드포인트 가져오기
    DB_ENDPOINT=$(aws rds describe-db-instances \
      --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
      --query 'DBInstances[0].Endpoint.Address' \
      --output text)
    
    echo "데이터베이스 엔드포인트: $DB_ENDPOINT"
    
    # 데이터베이스 초기 설정 SQL 스크립트
    cat > init-database.sql << 'EOF'
-- 데이터베이스 초기 설정
USE productiondb;

-- 사용자 테이블 생성
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 제품 테이블 생성
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    category_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_category (category_id),
    INDEX idx_price (price)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 주문 테이블 생성
CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 성능 최적화 설정
SET GLOBAL innodb_buffer_pool_size = 1073741824; -- 1GB
SET GLOBAL query_cache_size = 67108864; -- 64MB
SET GLOBAL slow_query_log = 1;
SET GLOBAL long_query_time = 2;

-- 샘플 데이터 삽입
INSERT INTO users (username, email, password_hash) VALUES
('admin', 'admin@example.com', SHA2('admin123', 256)),
('john_doe', 'john@example.com', SHA2('password123', 256)),
('jane_smith', 'jane@example.com', SHA2('password456', 256));

INSERT INTO products (name, description, price, stock_quantity, category_id) VALUES
('Laptop Pro', 'High-performance laptop for professionals', 1299.99, 50, 1),
('Wireless Mouse', 'Ergonomic wireless mouse', 29.99, 200, 2),
('USB-C Hub', 'Multi-port USB-C hub with HDMI output', 79.99, 100, 2);

INSERT INTO orders (user_id, total_amount, status) VALUES
(2, 1329.98, 'delivered'),
(3, 109.98, 'processing'),
(2, 29.99, 'shipped');
EOF
    
    # MySQL 클라이언트로 초기 설정 실행 (로컬에 mysql 클라이언트가 설치되어 있어야 함)
    # mysql -h $DB_ENDPOINT -u $MASTER_USERNAME -p$MASTER_PASSWORD < init-database.sql
    
    echo "데이터베이스 초기 설정 스크립트 생성 완료: init-database.sql"
    echo "다음 명령으로 실행하세요: mysql -h $DB_ENDPOINT -u $MASTER_USERNAME -p < init-database.sql"
}

# 데이터베이스 모니터링 설정
setup_monitoring() {
    # CloudWatch 알람 생성
    aws cloudwatch put-metric-alarm \
      --alarm-name "RDS-CPU-Utilization-High" \
      --alarm-description "RDS CPU utilization is too high" \
      --metric-name CPUUtilization \
      --namespace AWS/RDS \
      --statistic Average \
      --period 300 \
      --threshold 80 \
      --comparison-operator GreaterThanThreshold \
      --evaluation-periods 2 \
      --alarm-actions arn:aws:sns:us-east-1:123456789012:rds-alerts \
      --dimensions Name=DBInstanceIdentifier,Value=$DB_INSTANCE_IDENTIFIER
    
    aws cloudwatch put-metric-alarm \
      --alarm-name "RDS-FreeableMemory-Low" \
      --alarm-description "RDS freeable memory is too low" \
      --metric-name FreeableMemory \
      --namespace AWS/RDS \
      --statistic Average \
      --period 300 \
      --threshold 104857600 \
      --comparison-operator LessThanThreshold \
      --evaluation-periods 2 \
      --alarm-actions arn:aws:sns:us-east-1:123456789012:rds-alerts \
      --dimensions Name=DBInstanceIdentifier,Value=$DB_INSTANCE_IDENTIFIER
    
    echo "CloudWatch 알람 설정 완료"
}

# 메인 실행 함수
main() {
    echo "=== RDS 데이터베이스 구축 시작 ==="
    
    create_db_subnet_group
    create_rds_instance
    setup_database
    setup_monitoring
    
    echo "=== RDS 구축 완료 ==="
    echo "DB 인스턴스: $DB_INSTANCE_IDENTIFIER"
    echo "엔진: $ENGINE $ENGINE_VERSION"
    echo "클래스: $DB_INSTANCE_CLASS"
    echo "다중 AZ: 활성화"
    echo "암호화: 활성화"
    echo "백업 보존: 7일"
}

# 스크립트 실행
main
```

## 다음 편 예고

다음 포스트에서는 **AWS 서버리스 아키텍처와 Lambda 완전 마스터**를 상세히 다룰 예정입니다:
- Lambda 함수 개발부터 고급 패턴까지
- API Gateway와 완전 통합
- DynamoDB NoSQL 데이터베이스
- CloudFormation 인프라 자동화

AWS 핵심 서비스들을 완전히 마스터하셨나요? 🚀☁️
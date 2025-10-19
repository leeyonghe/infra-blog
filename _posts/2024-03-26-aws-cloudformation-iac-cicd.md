---
layout: post
title: "AWS 완전 정복 가이드 4편 - CloudFormation IaC와 CI/CD 파이프라인 | AWS Complete Guide Part 4 - CloudFormation IaC & CI/CD Pipeline"
date: 2024-03-26 14:00:00 +0900
categories: [AWS, Cloud]
tags: [aws, cloudformation, iac, cicd, codepipeline, sam, automation, devops]
---

AWS 인프라를 코드로 관리하고 완전 자동화된 CI/CD 파이프라인을 구축해보겠습니다. CloudFormation부터 SAM, CodePipeline까지 DevOps의 모든 것을 마스터합니다.

## Infrastructure as Code (IaC) 개요 | IaC Overview

### 🏗️ IaC의 이해와 장점

#### IaC의 핵심 원칙
- **버전 관리**: 인프라 변경사항 추적
- **재현 가능성**: 동일한 환경 반복 생성
- **문서화**: 코드 자체가 문서
- **자동화**: 수동 작업 최소화

#### AWS IaC 도구 비교
```
CloudFormation: AWS 네이티브, JSON/YAML
├── 장점: AWS 서비스 완전 지원, 무료
└── 단점: AWS 전용, 학습 곡선

SAM: 서버리스 특화 CloudFormation 확장
├── 장점: 서버리스 간편 배포, 로컬 테스트
└── 단점: 서버리스에 제한

CDK: 프로그래밍 언어로 인프라 정의
├── 장점: 타입 안전성, IDE 지원, 재사용성
└── 단점: 복잡성, 학습 비용

Terraform: 멀티 클라우드 지원
├── 장점: 클라우드 중립적, 강력한 상태 관리
└── 단점: 추가 도구, AWS 서비스 지원 지연
```

## CloudFormation 완전 마스터 | Complete CloudFormation Mastery

### 📋 고급 CloudFormation 템플릿

#### 종합적인 웹 애플리케이션 인프라 템플릿
```yaml
# comprehensive-web-app.yaml
# 프로덕션 레벨 웹 애플리케이션 인프라

AWSTemplateFormatVersion: '2010-09-09'
Description: 'Production-ready web application infrastructure with auto-scaling, load balancing, and monitoring'

Parameters:
  EnvironmentName:
    Description: Environment name prefix
    Type: String
    Default: Production
    AllowedValues: [Development, Testing, Production]
    
  VpcCIDR:
    Description: CIDR block for VPC
    Type: String
    Default: 10.0.0.0/16
    AllowedPattern: ^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])(/([0-9]|[1-2][0-9]|3[0-2]))$
    
  InstanceType:
    Description: EC2 instance type
    Type: String
    Default: t3.micro
    AllowedValues: 
      - t3.micro
      - t3.small
      - t3.medium
      - t3.large
      - m5.large
      - m5.xlarge
    ConstraintDescription: Must be a valid EC2 instance type
    
  KeyName:
    Description: EC2 Key Pair for SSH access
    Type: AWS::EC2::KeyPair::KeyName
    ConstraintDescription: Must be the name of an existing EC2 KeyPair
    
  SSLCertificateArn:
    Description: ARN of SSL certificate for HTTPS
    Type: String
    Default: ''
    
  DBPassword:
    Description: Database password
    Type: String
    NoEcho: true
    MinLength: 8
    MaxLength: 41
    AllowedPattern: '[a-zA-Z0-9]*'
    ConstraintDescription: Must contain only alphanumeric characters
    
  NotificationEmail:
    Description: Email for notifications
    Type: String
    AllowedPattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$

Conditions:
  HasSSLCertificate: !Not [!Equals [!Ref SSLCertificateArn, '']]
  IsProduction: !Equals [!Ref EnvironmentName, 'Production']

Mappings:
  RegionMap:
    us-east-1:
      AMI: ami-0abcdef1234567890
    us-west-2:
      AMI: ami-0fedcba0987654321
    eu-west-1:
      AMI: ami-0123456789abcdef0
      
  EnvironmentMap:
    Development:
      InstanceCount: 1
      DBInstanceClass: db.t3.micro
      DBAllocatedStorage: 20
    Testing:
      InstanceCount: 2
      DBInstanceClass: db.t3.small
      DBAllocatedStorage: 20
    Production:
      InstanceCount: 3
      DBInstanceClass: db.t3.medium
      DBAllocatedStorage: 100

Resources:
  # VPC 네트워크 인프라
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCIDR
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-VPC
        - Key: Environment
          Value: !Ref EnvironmentName

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-IGW

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # 퍼블릭 서브넷
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: !Select [0, !Cidr [!Ref VpcCIDR, 6, 8]]
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Public-Subnet-AZ1

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: !Select [1, !Cidr [!Ref VpcCIDR, 6, 8]]
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Public-Subnet-AZ2

  # 프라이빗 서브넷
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: !Select [2, !Cidr [!Ref VpcCIDR, 6, 8]]
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Private-Subnet-AZ1

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: !Select [3, !Cidr [!Ref VpcCIDR, 6, 8]]
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Private-Subnet-AZ2

  # 데이터베이스 서브넷
  DatabaseSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: !Select [4, !Cidr [!Ref VpcCIDR, 6, 8]]
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Database-Subnet-AZ1

  DatabaseSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: !Select [5, !Cidr [!Ref VpcCIDR, 6, 8]]
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Database-Subnet-AZ2

  # NAT 게이트웨이
  NatGateway1EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-NAT-Gateway-1-EIP

  NatGateway2EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-NAT-Gateway-2-EIP

  NatGateway1:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway1EIP.AllocationId
      SubnetId: !Ref PublicSubnet1
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-NAT-Gateway-AZ1

  NatGateway2:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway2EIP.AllocationId
      SubnetId: !Ref PublicSubnet2
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-NAT-Gateway-AZ2

  # 라우팅 테이블
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Public-Routes

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet1

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet2

  PrivateRouteTable1:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Private-Routes-AZ1

  DefaultPrivateRoute1:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway1

  PrivateSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet1

  PrivateRouteTable2:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Private-Routes-AZ2

  DefaultPrivateRoute2:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable2
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway2

  PrivateSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable2
      SubnetId: !Ref PrivateSubnet2

  # 보안 그룹
  LoadBalancerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub ${EnvironmentName}-LoadBalancer-SG
      GroupDescription: Security group for Application Load Balancer
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
          Description: HTTP from anywhere
        - !If
          - HasSSLCertificate
          - IpProtocol: tcp
            FromPort: 443
            ToPort: 443
            CidrIp: 0.0.0.0/0
            Description: HTTPS from anywhere
          - !Ref AWS::NoValue
      SecurityGroupEgress:
        - IpProtocol: -1
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-LoadBalancer-SG

  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub ${EnvironmentName}-WebServer-SG
      GroupDescription: Security group for web servers
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          SourceSecurityGroupId: !Ref LoadBalancerSecurityGroup
          Description: HTTP from Load Balancer
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          SourceSecurityGroupId: !Ref BastionSecurityGroup
          Description: SSH from Bastion
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-WebServer-SG

  BastionSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub ${EnvironmentName}-Bastion-SG
      GroupDescription: Security group for bastion host
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
          Description: SSH from anywhere
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Bastion-SG

  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub ${EnvironmentName}-Database-SG
      GroupDescription: Security group for database
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          SourceSecurityGroupId: !Ref WebServerSecurityGroup
          Description: MySQL from Web Servers
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Database-SG

  # IAM 역할
  EC2Role:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub ${EnvironmentName}-EC2-Role
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                Resource: !Sub '${S3Bucket}/*'
              - Effect: Allow
                Action:
                  - s3:ListBucket
                Resource: !Ref S3Bucket

  EC2InstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      InstanceProfileName: !Sub ${EnvironmentName}-EC2-Profile
      Roles:
        - !Ref EC2Role

  # S3 버킷
  S3Bucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ${EnvironmentName}-web-app-bucket-${AWS::AccountId}
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      LifecycleConfiguration:
        Rules:
          - Id: DeleteIncompleteMultipartUploads
            Status: Enabled
            AbortIncompleteMultipartUpload:
              DaysAfterInitiation: 7
      NotificationConfiguration:
        CloudWatchConfigurations:
          - Event: s3:ObjectCreated:*
            CloudWatchConfiguration:
              LogGroupName: !Ref S3LogGroup

  # 데이터베이스 서브넷 그룹
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupName: !Sub ${EnvironmentName}-database-subnet-group
      DBSubnetGroupDescription: Subnet group for database
      SubnetIds:
        - !Ref DatabaseSubnet1
        - !Ref DatabaseSubnet2
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Database-Subnet-Group

  # RDS 데이터베이스
  Database:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: !If [IsProduction, Snapshot, Delete]
    Properties:
      DBInstanceIdentifier: !Sub ${EnvironmentName}-database
      DBName: webapp
      DBInstanceClass: !FindInMap [EnvironmentMap, !Ref EnvironmentName, DBInstanceClass]
      AllocatedStorage: !FindInMap [EnvironmentMap, !Ref EnvironmentName, DBAllocatedStorage]
      Engine: MySQL
      EngineVersion: '8.0'
      MasterUsername: admin
      MasterUserPassword: !Ref DBPassword
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      BackupRetentionPeriod: !If [IsProduction, 14, 7]
      MultiAZ: !If [IsProduction, true, false]
      StorageEncrypted: true
      DeletionProtection: !If [IsProduction, true, false]
      MonitoringInterval: 60
      MonitoringRoleArn: !GetAtt RDSEnhancedMonitoringRole.Arn
      EnablePerformanceInsights: true
      PerformanceInsightsRetentionPeriod: 7
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Database

  RDSEnhancedMonitoringRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: monitoring.rds.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole

  # Launch Template
  LaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: !Sub ${EnvironmentName}-LaunchTemplate
      LaunchTemplateData:
        ImageId: !FindInMap [RegionMap, !Ref 'AWS::Region', AMI]
        InstanceType: !Ref InstanceType
        KeyName: !Ref KeyName
        IamInstanceProfile:
          Arn: !GetAtt EC2InstanceProfile.Arn
        SecurityGroupIds:
          - !Ref WebServerSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash -xe
            yum update -y
            yum install -y httpd mysql amazon-cloudwatch-agent
            
            # Apache 설정
            systemctl start httpd
            systemctl enable httpd
            
            # 웹사이트 파일 다운로드
            aws s3 cp s3://${S3Bucket}/website/ /var/www/html/ --recursive
            
            # CloudWatch 에이전트 설정
            cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
            {
              "metrics": {
                "namespace": "WebApp/EC2",
                "metrics_collected": {
                  "cpu": {
                    "measurement": ["cpu_usage_idle", "cpu_usage_iowait"],
                    "metrics_collection_interval": 60
                  },
                  "disk": {
                    "measurement": ["used_percent"],
                    "metrics_collection_interval": 60,
                    "resources": ["*"]
                  },
                  "mem": {
                    "measurement": ["mem_used_percent"],
                    "metrics_collection_interval": 60
                  }
                }
              },
              "logs": {
                "logs_collected": {
                  "files": {
                    "collect_list": [
                      {
                        "file_path": "/var/log/httpd/access_log",
                        "log_group_name": "${CloudWatchLogGroup}",
                        "log_stream_name": "{instance_id}/apache/access.log"
                      },
                      {
                        "file_path": "/var/log/httpd/error_log",
                        "log_group_name": "${CloudWatchLogGroup}",
                        "log_stream_name": "{instance_id}/apache/error.log"
                      }
                    ]
                  }
                }
              }
            }
            EOF
            
            # CloudWatch 에이전트 시작
            /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
              -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s
            
            # 인스턴스 정보를 웹페이지에 추가
            cat >> /var/www/html/index.html << EOF
            <p>Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)</p>
            <p>Availability Zone: $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)</p>
            <p>Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)</p>
            EOF
            
            # Signal success to CloudFormation
            /opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource AutoScalingGroup --region ${AWS::Region}
        TagSpecifications:
          - ResourceType: instance
            Tags:
              - Key: Name
                Value: !Sub ${EnvironmentName}-WebServer
              - Key: Environment
                Value: !Ref EnvironmentName

  # Auto Scaling Group
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: !Sub ${EnvironmentName}-ASG
      VPCZoneIdentifier:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: 1
      MaxSize: 10
      DesiredCapacity: !FindInMap [EnvironmentMap, !Ref EnvironmentName, InstanceCount]
      TargetGroupARNs:
        - !Ref ALBTargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-ASG-Instance
          PropagateAtLaunch: true
        - Key: Environment
          Value: !Ref EnvironmentName
          PropagateAtLaunch: true
    CreationPolicy:
      ResourceSignal:
        Count: !FindInMap [EnvironmentMap, !Ref EnvironmentName, InstanceCount]
        Timeout: PT15M
    UpdatePolicy:
      AutoScalingRollingUpdate:
        MinInstancesInService: 1
        MaxBatchSize: 1
        PauseTime: PT15M
        WaitOnResourceSignals: true

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub ${EnvironmentName}-ALB
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-ALB

  ALBTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub ${EnvironmentName}-TG
      Port: 80
      Protocol: HTTP
      VpcId: !Ref VPC
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 5
      TargetGroupAttributes:
        - Key: deregistration_delay.timeout_seconds
          Value: '30'
        - Key: stickiness.enabled
          Value: 'true'
        - Key: stickiness.type
          Value: lb_cookie
        - Key: stickiness.lb_cookie.duration_seconds
          Value: '86400'

  ALBListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref ALBTargetGroup
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Port: 80
      Protocol: HTTP

  # HTTPS 리스너 (SSL 인증서가 있는 경우)
  ALBListenerHTTPS:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Condition: HasSSLCertificate
    Properties:
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref ALBTargetGroup
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Port: 443
      Protocol: HTTPS
      Certificates:
        - CertificateArn: !Ref SSLCertificateArn

  # Auto Scaling 정책
  ScaleUpPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AdjustmentType: ChangeInCapacity
      AutoScalingGroupName: !Ref AutoScalingGroup
      Cooldown: 300
      ScalingAdjustment: 1
      PolicyType: SimpleScaling

  ScaleDownPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AdjustmentType: ChangeInCapacity
      AutoScalingGroupName: !Ref AutoScalingGroup
      Cooldown: 300
      ScalingAdjustment: -1
      PolicyType: SimpleScaling

  # CloudWatch 알람
  CPUAlarmHigh:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub ${EnvironmentName}-CPU-High
      AlarmDescription: Scale up on high CPU
      MetricName: CPUUtilization
      Namespace: AWS/EC2
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 70
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: AutoScalingGroupName
          Value: !Ref AutoScalingGroup
      AlarmActions:
        - !Ref ScaleUpPolicy
        - !Ref SNSTopic

  CPUAlarmLow:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub ${EnvironmentName}-CPU-Low
      AlarmDescription: Scale down on low CPU
      MetricName: CPUUtilization
      Namespace: AWS/EC2
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 20
      ComparisonOperator: LessThanThreshold
      Dimensions:
        - Name: AutoScalingGroupName
          Value: !Ref AutoScalingGroup
      AlarmActions:
        - !Ref ScaleDownPolicy

  # CloudWatch 로그 그룹
  CloudWatchLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /aws/ec2/${EnvironmentName}
      RetentionInDays: !If [IsProduction, 365, 14]

  S3LogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /aws/s3/${EnvironmentName}
      RetentionInDays: 30

  # SNS 토픽
  SNSTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub ${EnvironmentName}-Alerts
      DisplayName: !Sub ${EnvironmentName} Application Alerts

  SNSSubscription:
    Type: AWS::SNS::Subscription
    Properties:
      Protocol: email
      TopicArn: !Ref SNSTopic
      Endpoint: !Ref NotificationEmail

  # CloudFront 배포 (선택사항)
  CloudFrontDistribution:
    Type: AWS::CloudFront::Distribution
    Condition: IsProduction
    Properties:
      DistributionConfig:
        Enabled: true
        Comment: !Sub ${EnvironmentName} CloudFront Distribution
        DefaultRootObject: index.html
        Origins:
          - Id: ALBOrigin
            DomainName: !GetAtt ApplicationLoadBalancer.DNSName
            CustomOriginConfig:
              HTTPPort: 80
              HTTPSPort: 443
              OriginProtocolPolicy: http-only
        DefaultCacheBehavior:
          TargetOriginId: ALBOrigin
          ViewerProtocolPolicy: redirect-to-https
          Compress: true
          AllowedMethods: [GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE]
          CachedMethods: [GET, HEAD, OPTIONS]
          ForwardedValues:
            QueryString: true
            Headers: ['*']
        PriceClass: PriceClass_100
        ViewerCertificate: !If
          - HasSSLCertificate
          - AcmCertificateArn: !Ref SSLCertificateArn
            SslSupportMethod: sni-only
            MinimumProtocolVersion: TLSv1.2_2021
          - CloudFrontDefaultCertificate: true

Outputs:
  VPC:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub ${EnvironmentName}-VPC-ID

  PublicSubnets:
    Description: Public subnet IDs
    Value: !Join [',', [!Ref PublicSubnet1, !Ref PublicSubnet2]]
    Export:
      Name: !Sub ${EnvironmentName}-Public-Subnets

  PrivateSubnets:
    Description: Private subnet IDs
    Value: !Join [',', [!Ref PrivateSubnet1, !Ref PrivateSubnet2]]
    Export:
      Name: !Sub ${EnvironmentName}-Private-Subnets

  LoadBalancerURL:
    Description: Application Load Balancer URL
    Value: !Sub 
      - http${Protocol}://${DNSName}
      - Protocol: !If [HasSSLCertificate, 's', '']
        DNSName: !GetAtt ApplicationLoadBalancer.DNSName
    Export:
      Name: !Sub ${EnvironmentName}-ALB-URL

  CloudFrontURL:
    Description: CloudFront Distribution URL
    Condition: IsProduction
    Value: !Sub https://${CloudFrontDistribution.DomainName}
    Export:
      Name: !Sub ${EnvironmentName}-CloudFront-URL

  DatabaseEndpoint:
    Description: RDS database endpoint
    Value: !GetAtt Database.Endpoint.Address
    Export:
      Name: !Sub ${EnvironmentName}-DB-Endpoint

  S3BucketName:
    Description: S3 bucket name
    Value: !Ref S3Bucket
    Export:
      Name: !Sub ${EnvironmentName}-S3-Bucket
```

#### CloudFormation 배포 스크립트
```bash
#!/bin/bash
# CloudFormation 스택 배포 스크립트

STACK_NAME="web-app-infrastructure"
TEMPLATE_FILE="comprehensive-web-app.yaml"
PARAMETERS_FILE="parameters.json"
REGION="us-east-1"

# 파라미터 파일 생성
cat > $PARAMETERS_FILE << 'EOF'
[
  {
    "ParameterKey": "EnvironmentName",
    "ParameterValue": "Production"
  },
  {
    "ParameterKey": "VpcCIDR",
    "ParameterValue": "10.0.0.0/16"
  },
  {
    "ParameterKey": "InstanceType",
    "ParameterValue": "t3.micro"
  },
  {
    "ParameterKey": "KeyName",
    "ParameterValue": "my-ec2-keypair"
  },
  {
    "ParameterKey": "DBPassword",
    "ParameterValue": "MySecurePassword123"
  },
  {
    "ParameterKey": "NotificationEmail",
    "ParameterValue": "admin@example.com"
  }
]
EOF

echo "CloudFormation 스택 배포 시작..."

# 스택 존재 여부 확인
if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION >/dev/null 2>&1; then
    echo "기존 스택 업데이트 중..."
    
    # 변경 세트 생성
    CHANGE_SET_NAME="update-$(date +%Y%m%d-%H%M%S)"
    
    aws cloudformation create-change-set \
      --stack-name $STACK_NAME \
      --change-set-name $CHANGE_SET_NAME \
      --template-body file://$TEMPLATE_FILE \
      --parameters file://$PARAMETERS_FILE \
      --capabilities CAPABILITY_NAMED_IAM \
      --region $REGION
    
    echo "변경 세트 생성 중..."
    aws cloudformation wait change-set-create-complete \
      --change-set-name $CHANGE_SET_NAME \
      --stack-name $STACK_NAME \
      --region $REGION
    
    # 변경 사항 확인
    echo "변경 사항:"
    aws cloudformation describe-change-set \
      --change-set-name $CHANGE_SET_NAME \
      --stack-name $STACK_NAME \
      --region $REGION \
      --query 'Changes[*].[Action,ResourceChange.ResourceType,ResourceChange.LogicalResourceId]' \
      --output table
    
    # 사용자 확인
    read -p "변경 사항을 적용하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        aws cloudformation execute-change-set \
          --change-set-name $CHANGE_SET_NAME \
          --stack-name $STACK_NAME \
          --region $REGION
        
        echo "스택 업데이트 중..."
        aws cloudformation wait stack-update-complete \
          --stack-name $STACK_NAME \
          --region $REGION
    else
        aws cloudformation delete-change-set \
          --change-set-name $CHANGE_SET_NAME \
          --stack-name $STACK_NAME \
          --region $REGION
        echo "변경 사항이 취소되었습니다."
        exit 0
    fi
else
    echo "새 스택 생성 중..."
    aws cloudformation create-stack \
      --stack-name $STACK_NAME \
      --template-body file://$TEMPLATE_FILE \
      --parameters file://$PARAMETERS_FILE \
      --capabilities CAPABILITY_NAMED_IAM \
      --enable-termination-protection \
      --tags Key=Project,Value=WebApp Key=Environment,Value=Production \
      --region $REGION
    
    echo "스택 생성 중... (10-15분 소요)"
    aws cloudformation wait stack-create-complete \
      --stack-name $STACK_NAME \
      --region $REGION
fi

# 스택 출력 정보 표시
echo "=== 스택 배포 완료 ==="
aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --region $REGION \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue,Description]' \
  --output table

# 스택 이벤트 확인 (오류가 있는 경우)
if [ $? -ne 0 ]; then
    echo "스택 배포 중 오류 발생. 이벤트 확인:"
    aws cloudformation describe-stack-events \
      --stack-name $STACK_NAME \
      --region $REGION \
      --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`||ResourceStatus==`UPDATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
      --output table
fi

# 정리
rm -f $PARAMETERS_FILE
```

## 다음 편 예고

다음 포스트에서는 **AWS 보안과 모니터링 완전 마스터**를 상세히 다룰 예정입니다:
- IAM 고급 정책 및 권한 관리
- AWS Security Hub와 GuardDuty
- CloudWatch와 X-Ray 심화 모니터링
- AWS Config 규정 준수 관리

CloudFormation IaC와 CI/CD 파이프라인을 완전히 마스터하셨나요? 🏗️⚙️
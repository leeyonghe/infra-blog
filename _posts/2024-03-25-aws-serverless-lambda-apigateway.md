---
layout: post
title: "AWS 완전 정복 가이드 3편 - 서버리스 아키텍처와 Lambda 마스터 | AWS Complete Guide Part 3 - Serverless Architecture & Lambda Mastery"
date: 2024-03-25 14:00:00 +0900
categories: [AWS, Cloud]
tags: [aws, lambda, serverless, api-gateway, dynamodb, cloudformation, event-driven]
---

AWS 서버리스 생태계를 완전히 마스터해보겠습니다. Lambda부터 API Gateway, DynamoDB, CloudFormation까지 현대적인 서버리스 아키텍처 구축의 모든 것을 다룹니다.

## 서버리스 아키텍처 개요 | Serverless Architecture Overview

### 🚀 서버리스의 이해

#### 서버리스 컴퓨팅의 장점
- **자동 스케일링**: 트래픽에 따른 자동 확장/축소
- **비용 효율성**: 실행 시간에 대해서만 과금
- **운영 부담 감소**: 서버 관리 불필요
- **빠른 개발**: 인프라보다 비즈니스 로직에 집중

#### AWS 서버리스 서비스 스택
```
프론트엔드: S3 + CloudFront
API 계층: API Gateway
컴퓨팅: Lambda Functions
데이터: DynamoDB, S3
인증: Cognito
모니터링: CloudWatch, X-Ray
배포: CloudFormation, SAM
```

## Lambda 완전 마스터 | Complete Lambda Mastery

### ⚡ Lambda 함수 개발 및 관리

#### Lambda 함수 생성 및 배포 (Python)
```python
# lambda_function.py
# 종합적인 Lambda 함수 예제

import json
import boto3
import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import uuid

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS 서비스 클라이언트 초기화
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
ses = boto3.client('ses')
sns = boto3.client('sns')

# 환경 변수
TABLE_NAME = os.environ['DYNAMODB_TABLE']
BUCKET_NAME = os.environ['S3_BUCKET']
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']

class DecimalEncoder(json.JSONEncoder):
    """DynamoDB Decimal 타입을 JSON으로 변환"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def lambda_handler(event, context):
    """메인 Lambda 핸들러"""
    
    try:
        # 요청 로깅
        logger.info(f"Event: {json.dumps(event)}")
        
        # HTTP 메서드별 처리
        http_method = event.get('httpMethod', '')
        path = event.get('path', '')
        
        if http_method == 'GET':
            return handle_get_request(event, context)
        elif http_method == 'POST':
            return handle_post_request(event, context)
        elif http_method == 'PUT':
            return handle_put_request(event, context)
        elif http_method == 'DELETE':
            return handle_delete_request(event, context)
        else:
            return create_response(405, {'error': 'Method not allowed'})
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return create_response(500, {'error': 'Internal server error'})

def handle_get_request(event, context):
    """GET 요청 처리"""
    
    path_parameters = event.get('pathParameters') or {}
    query_parameters = event.get('queryStringParameters') or {}
    
    # 단일 아이템 조회
    if 'id' in path_parameters:
        return get_item_by_id(path_parameters['id'])
    
    # 목록 조회
    return get_items_list(query_parameters)

def handle_post_request(event, context):
    """POST 요청 처리 - 새 아이템 생성"""
    
    try:
        body = json.loads(event.get('body', '{}'))
        
        # 필수 필드 검증
        required_fields = ['name', 'email']
        for field in required_fields:
            if field not in body:
                return create_response(400, {'error': f'Missing required field: {field}'})
        
        # 새 아이템 생성
        item_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        item = {
            'id': item_id,
            'name': body['name'],
            'email': body['email'],
            'description': body.get('description', ''),
            'created_at': timestamp,
            'updated_at': timestamp,
            'status': 'active'
        }
        
        # DynamoDB에 저장
        table = dynamodb.Table(TABLE_NAME)
        table.put_item(Item=item)
        
        # 알림 발송
        send_notification(f"New item created: {item['name']}")
        
        # S3에 백업 (선택사항)
        backup_to_s3(item)
        
        logger.info(f"Created item with ID: {item_id}")
        
        return create_response(201, item)
        
    except json.JSONDecodeError:
        return create_response(400, {'error': 'Invalid JSON in request body'})
    except Exception as e:
        logger.error(f"Error creating item: {str(e)}")
        return create_response(500, {'error': 'Failed to create item'})

def handle_put_request(event, context):
    """PUT 요청 처리 - 아이템 업데이트"""
    
    path_parameters = event.get('pathParameters') or {}
    
    if 'id' not in path_parameters:
        return create_response(400, {'error': 'Missing item ID'})
    
    try:
        body = json.loads(event.get('body', '{}'))
        item_id = path_parameters['id']
        
        table = dynamodb.Table(TABLE_NAME)
        
        # 아이템 존재 확인
        response = table.get_item(Key={'id': item_id})
        if 'Item' not in response:
            return create_response(404, {'error': 'Item not found'})
        
        # 업데이트 표현식 구성
        update_expression = "SET updated_at = :timestamp"
        expression_values = {':timestamp': datetime.utcnow().isoformat()}
        
        for key, value in body.items():
            if key not in ['id', 'created_at']:  # 변경 불가 필드 제외
                update_expression += f", {key} = :{key}"
                expression_values[f":{key}"] = value
        
        # 아이템 업데이트
        response = table.update_item(
            Key={'id': item_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ReturnValues='ALL_NEW'
        )
        
        updated_item = response['Attributes']
        
        logger.info(f"Updated item: {item_id}")
        
        return create_response(200, updated_item)
        
    except json.JSONDecodeError:
        return create_response(400, {'error': 'Invalid JSON in request body'})
    except Exception as e:
        logger.error(f"Error updating item: {str(e)}")
        return create_response(500, {'error': 'Failed to update item'})

def handle_delete_request(event, context):
    """DELETE 요청 처리 - 아이템 삭제"""
    
    path_parameters = event.get('pathParameters') or {}
    
    if 'id' not in path_parameters:
        return create_response(400, {'error': 'Missing item ID'})
    
    try:
        item_id = path_parameters['id']
        table = dynamodb.Table(TABLE_NAME)
        
        # 아이템 존재 확인
        response = table.get_item(Key={'id': item_id})
        if 'Item' not in response:
            return create_response(404, {'error': 'Item not found'})
        
        # 아이템 삭제
        table.delete_item(Key={'id': item_id})
        
        logger.info(f"Deleted item: {item_id}")
        
        return create_response(200, {'message': 'Item deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting item: {str(e)}")
        return create_response(500, {'error': 'Failed to delete item'})

def get_item_by_id(item_id):
    """ID로 단일 아이템 조회"""
    
    try:
        table = dynamodb.Table(TABLE_NAME)
        response = table.get_item(Key={'id': item_id})
        
        if 'Item' not in response:
            return create_response(404, {'error': 'Item not found'})
        
        return create_response(200, response['Item'])
        
    except Exception as e:
        logger.error(f"Error getting item: {str(e)}")
        return create_response(500, {'error': 'Failed to get item'})

def get_items_list(query_parameters):
    """아이템 목록 조회 (페이지네이션 지원)"""
    
    try:
        table = dynamodb.Table(TABLE_NAME)
        
        # 페이지네이션 파라미터
        limit = int(query_parameters.get('limit', 20))
        last_key = query_parameters.get('last_key')
        
        scan_kwargs = {'Limit': limit}
        
        if last_key:
            scan_kwargs['ExclusiveStartKey'] = {'id': last_key}
        
        response = table.scan(**scan_kwargs)
        
        result = {
            'items': response['Items'],
            'count': len(response['Items'])
        }
        
        if 'LastEvaluatedKey' in response:
            result['last_key'] = response['LastEvaluatedKey']['id']
        
        return create_response(200, result)
        
    except Exception as e:
        logger.error(f"Error getting items list: {str(e)}")
        return create_response(500, {'error': 'Failed to get items list'})

def send_notification(message):
    """SNS를 통한 알림 발송"""
    
    try:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=message,
            Subject='Lambda Function Notification'
        )
        logger.info(f"Notification sent: {message}")
        
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")

def backup_to_s3(item):
    """S3에 아이템 백업"""
    
    try:
        backup_key = f"backups/{datetime.utcnow().strftime('%Y/%m/%d')}/{item['id']}.json"
        
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=backup_key,
            Body=json.dumps(item, cls=DecimalEncoder),
            ContentType='application/json'
        )
        
        logger.info(f"Item backed up to S3: {backup_key}")
        
    except Exception as e:
        logger.error(f"Failed to backup to S3: {str(e)}")

def create_response(status_code, body):
    """HTTP 응답 생성"""
    
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS'
        },
        'body': json.dumps(body, cls=DecimalEncoder)
    }

# 추가 유틸리티 함수들

def validate_email(email):
    """이메일 주소 유효성 검증"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def generate_presigned_url(bucket, key, expiration=3600):
    """S3 객체용 사전 서명된 URL 생성"""
    try:
        response = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return response
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        return None

def send_email(to_email, subject, body):
    """SES를 통한 이메일 발송"""
    try:
        response = ses.send_email(
            Source='noreply@example.com',
            Destination={'ToAddresses': [to_email]},
            Message={
                'Subject': {'Data': subject},
                'Body': {'Text': {'Data': body}}
            }
        )
        logger.info(f"Email sent to {to_email}")
        return response
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return None
```

#### Lambda 배포 스크립트
```bash
#!/bin/bash
# Lambda 함수 배포 스크립트

FUNCTION_NAME="serverless-api-function"
ROLE_ARN="arn:aws:iam::123456789012:role/lambda-execution-role"
REGION="us-east-1"

# 의존성 설치 및 패키징
echo "Lambda 함수 패키징 중..."

# 임시 디렉토리 생성
mkdir -p lambda-package
cd lambda-package

# 함수 코드 복사
cp ../lambda_function.py .

# requirements.txt가 있다면 의존성 설치
if [ -f "../requirements.txt" ]; then
    pip install -r ../requirements.txt -t .
fi

# 패키지 생성
zip -r ../lambda-function.zip .

cd ..
rm -rf lambda-package

echo "패키징 완료: lambda-function.zip"

# Lambda 함수 존재 확인
if aws lambda get-function --function-name $FUNCTION_NAME >/dev/null 2>&1; then
    echo "기존 함수 업데이트 중..."
    
    # 함수 코드 업데이트
    aws lambda update-function-code \
      --function-name $FUNCTION_NAME \
      --zip-file fileb://lambda-function.zip
    
    # 함수 설정 업데이트
    aws lambda update-function-configuration \
      --function-name $FUNCTION_NAME \
      --timeout 30 \
      --memory-size 256 \
      --environment Variables='{
        "DYNAMODB_TABLE":"serverless-api-table",
        "S3_BUCKET":"serverless-api-bucket",
        "SNS_TOPIC_ARN":"arn:aws:sns:us-east-1:123456789012:serverless-notifications"
      }'
else
    echo "새 Lambda 함수 생성 중..."
    
    # 새 함수 생성
    aws lambda create-function \
      --function-name $FUNCTION_NAME \
      --runtime python3.9 \
      --role $ROLE_ARN \
      --handler lambda_function.lambda_handler \
      --zip-file fileb://lambda-function.zip \
      --timeout 30 \
      --memory-size 256 \
      --environment Variables='{
        "DYNAMODB_TABLE":"serverless-api-table",
        "S3_BUCKET":"serverless-api-bucket", 
        "SNS_TOPIC_ARN":"arn:aws:sns:us-east-1:123456789012:serverless-notifications"
      }' \
      --tags Environment=Production,Project=ServerlessAPI
fi

# 함수 별칭 생성/업데이트
aws lambda publish-version --function-name $FUNCTION_NAME

# PROD 별칭 생성 또는 업데이트
aws lambda update-alias \
  --function-name $FUNCTION_NAME \
  --name PROD \
  --function-version '$LATEST' 2>/dev/null || \
aws lambda create-alias \
  --function-name $FUNCTION_NAME \
  --name PROD \
  --function-version '$LATEST'

echo "Lambda 함수 배포 완료!"

# 함수 정보 출력
aws lambda get-function --function-name $FUNCTION_NAME \
  --query 'Configuration.[FunctionName,Runtime,Timeout,MemorySize,LastModified]' \
  --output table

# 정리
rm -f lambda-function.zip
```

## API Gateway 완전 통합 | Complete API Gateway Integration

### 🌐 REST API 구축

#### API Gateway 생성 및 설정
```bash
#!/bin/bash
# API Gateway 생성 스크립트

API_NAME="serverless-api"
LAMBDA_FUNCTION_ARN="arn:aws:lambda:us-east-1:123456789012:function:serverless-api-function"
REGION="us-east-1"
ACCOUNT_ID="123456789012"

# REST API 생성
echo "REST API 생성 중..."

API_ID=$(aws apigateway create-rest-api \
  --name $API_NAME \
  --description "Serverless API with Lambda integration" \
  --endpoint-configuration types=REGIONAL \
  --query 'id' \
  --output text)

echo "API 생성됨: $API_ID"

# 루트 리소스 ID 가져오기
ROOT_RESOURCE_ID=$(aws apigateway get-resources \
  --rest-api-id $API_ID \
  --query 'items[?path==`/`].id' \
  --output text)

# /items 리소스 생성
ITEMS_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_RESOURCE_ID \
  --path-part items \
  --query 'id' \
  --output text)

# /items/{id} 리소스 생성
ITEM_RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ITEMS_RESOURCE_ID \
  --path-part '{id}' \
  --query 'id' \
  --output text)

echo "리소스 생성 완료: /items, /items/{id}"

# Lambda 통합을 위한 공통 함수
create_lambda_integration() {
    local resource_id=$1
    local http_method=$2
    
    # 메서드 생성
    aws apigateway put-method \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method $http_method \
      --authorization-type NONE \
      --api-key-required
    
    # Lambda 통합 설정
    aws apigateway put-integration \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method $http_method \
      --type AWS_PROXY \
      --integration-http-method POST \
      --uri arn:aws:apigateway:$REGION:lambda:path/2015-03-31/functions/$LAMBDA_FUNCTION_ARN/invocations
    
    # Lambda 실행 권한 부여
    aws lambda add-permission \
      --function-name serverless-api-function \
      --statement-id "${API_ID}-${resource_id}-${http_method}" \
      --action lambda:InvokeFunction \
      --principal apigateway.amazonaws.com \
      --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*/$http_method/*" 2>/dev/null || true
}

# CORS 옵션 메서드 추가
add_cors_options() {
    local resource_id=$1
    
    # OPTIONS 메서드 생성
    aws apigateway put-method \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --authorization-type NONE
    
    # Mock 통합 설정
    aws apigateway put-integration \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --type MOCK \
      --request-templates '{"application/json": "{\"statusCode\": 200}"}'
    
    # 메서드 응답 설정
    aws apigateway put-method-response \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --status-code 200 \
      --response-parameters method.response.header.Access-Control-Allow-Headers=false,method.response.header.Access-Control-Allow-Methods=false,method.response.header.Access-Control-Allow-Origin=false
    
    # 통합 응답 설정
    aws apigateway put-integration-response \
      --rest-api-id $API_ID \
      --resource-id $resource_id \
      --http-method OPTIONS \
      --status-code 200 \
      --response-parameters '{"method.response.header.Access-Control-Allow-Headers":"'"'"'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"'"'","method.response.header.Access-Control-Allow-Methods":"'"'"'GET,POST,PUT,DELETE,OPTIONS'"'"'","method.response.header.Access-Control-Allow-Origin":"'"'"'*'"'"'}'
}

# /items 리소스에 메서드 추가
echo "API 메서드 생성 중..."

# GET /items (목록 조회)
create_lambda_integration $ITEMS_RESOURCE_ID GET

# POST /items (새 아이템 생성)
create_lambda_integration $ITEMS_RESOURCE_ID POST

# CORS 옵션 추가
add_cors_options $ITEMS_RESOURCE_ID

# /items/{id} 리소스에 메서드 추가

# GET /items/{id} (단일 아이템 조회)
create_lambda_integration $ITEM_RESOURCE_ID GET

# PUT /items/{id} (아이템 업데이트)
create_lambda_integration $ITEM_RESOURCE_ID PUT

# DELETE /items/{id} (아이템 삭제)
create_lambda_integration $ITEM_RESOURCE_ID DELETE

# CORS 옵션 추가
add_cors_options $ITEM_RESOURCE_ID

# API 키 생성
API_KEY_ID=$(aws apigateway create-api-key \
  --name "${API_NAME}-key" \
  --description "API key for serverless API" \
  --enabled \
  --query 'id' \
  --output text)

API_KEY_VALUE=$(aws apigateway get-api-key \
  --api-key $API_KEY_ID \
  --include-value \
  --query 'value' \
  --output text)

echo "API 키 생성됨: $API_KEY_VALUE"

# 사용량 계획 생성
USAGE_PLAN_ID=$(aws apigateway create-usage-plan \
  --name "${API_NAME}-usage-plan" \
  --description "Usage plan for serverless API" \
  --throttle burstLimit=200,rateLimit=100 \
  --quota limit=10000,period=MONTH \
  --query 'id' \
  --output text)

# API 키를 사용량 계획에 연결
aws apigateway create-usage-plan-key \
  --usage-plan-id $USAGE_PLAN_ID \
  --key-id $API_KEY_ID \
  --key-type API_KEY

# 배포 스테이지 생성
DEPLOYMENT_ID=$(aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod \
  --stage-description "Production stage" \
  --description "Production deployment" \
  --query 'id' \
  --output text)

# 사용량 계획에 API 스테이지 추가
aws apigateway update-usage-plan \
  --usage-plan-id $USAGE_PLAN_ID \
  --patch-ops op=add,path="/apiStages",value="${API_ID}:prod"

# CloudWatch 로깅 활성화
aws apigateway update-stage \
  --rest-api-id $API_ID \
  --stage-name prod \
  --patch-ops op=replace,path="/*/logging/loglevel",value=INFO \
             op=replace,path="/*/logging/dataTrace",value=true \
             op=replace,path="/*/metrics/enabled",value=true

echo "=== API Gateway 설정 완료 ==="
echo "API ID: $API_ID"
echo "API URL: https://$API_ID.execute-api.$REGION.amazonaws.com/prod"
echo "API Key: $API_KEY_VALUE"
echo ""
echo "사용 예시:"
echo "curl -H 'X-API-Key: $API_KEY_VALUE' https://$API_ID.execute-api.$REGION.amazonaws.com/prod/items"
```

## DynamoDB NoSQL 데이터베이스 | DynamoDB NoSQL Database

### 📊 DynamoDB 설계 및 최적화

#### DynamoDB 테이블 생성 및 설정
```bash
#!/bin/bash
# DynamoDB 테이블 생성 및 설정 스크립트

TABLE_NAME="serverless-api-table"
REGION="us-east-1"

# DynamoDB 테이블 생성
echo "DynamoDB 테이블 생성 중..."

aws dynamodb create-table \
  --table-name $TABLE_NAME \
  --attribute-definitions \
    AttributeName=id,AttributeType=S \
    AttributeName=created_at,AttributeType=S \
    AttributeName=status,AttributeType=S \
    AttributeName=email,AttributeType=S \
  --key-schema \
    AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --global-secondary-indexes \
    IndexName=CreatedAtIndex,KeySchema=[{AttributeName=status,KeyType=HASH},{AttributeName=created_at,KeyType=RANGE}],Projection='{ProjectionType=ALL}',ProvisionedThroughput='{ReadCapacityUnits=5,WriteCapacityUnits=5}' \
    IndexName=EmailIndex,KeySchema=[{AttributeName=email,KeyType=HASH}],Projection='{ProjectionType=ALL}',ProvisionedThroughput='{ReadCapacityUnits=5,WriteCapacityUnits=5}' \
  --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES \
  --tags Key=Environment,Value=Production Key=Project,Value=ServerlessAPI

echo "테이블 생성 중... 완료까지 잠시 기다려주세요."

# 테이블 생성 완료 대기
aws dynamodb wait table-exists --table-name $TABLE_NAME

echo "DynamoDB 테이블 생성 완료: $TABLE_NAME"

# Point-in-Time Recovery 활성화
aws dynamodb update-continuous-backups \
  --table-name $TABLE_NAME \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true

echo "Point-in-Time Recovery 활성화 완료"

# 테이블 정보 출력
aws dynamodb describe-table --table-name $TABLE_NAME \
  --query 'Table.[TableName,TableStatus,ItemCount,TableSizeBytes]' \
  --output table

# DynamoDB 스트림을 위한 Lambda 함수 생성
cat > stream_processor.py << 'EOF'
import json
import boto3
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """DynamoDB 스트림 이벤트 처리"""
    
    for record in event['Records']:
        event_name = record['eventName']
        
        if event_name == 'INSERT':
            handle_insert(record)
        elif event_name == 'MODIFY':
            handle_modify(record)
        elif event_name == 'REMOVE':
            handle_remove(record)
    
    return {'statusCode': 200}

def handle_insert(record):
    """새 레코드 삽입 처리"""
    new_image = record['dynamodb']['NewImage']
    
    logger.info(f"New item created: {new_image}")
    
    # 여기에 추가 처리 로직 구현
    # 예: 알림 발송, 검색 인덱스 업데이트 등

def handle_modify(record):
    """레코드 수정 처리"""
    old_image = record['dynamodb'].get('OldImage', {})
    new_image = record['dynamodb']['NewImage']
    
    logger.info(f"Item modified: {old_image} -> {new_image}")

def handle_remove(record):
    """레코드 삭제 처리"""
    old_image = record['dynamodb']['OldImage']
    
    logger.info(f"Item removed: {old_image}")
EOF

# 스트림 처리 Lambda 함수 배포
zip stream_processor.zip stream_processor.py

aws lambda create-function \
  --function-name dynamodb-stream-processor \
  --runtime python3.9 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler stream_processor.lambda_handler \
  --zip-file fileb://stream_processor.zip \
  --timeout 60 \
  --memory-size 128

# DynamoDB 스트림 ARN 가져오기
STREAM_ARN=$(aws dynamodb describe-table \
  --table-name $TABLE_NAME \
  --query 'Table.LatestStreamArn' \
  --output text)

# 이벤트 소스 매핑 생성
aws lambda create-event-source-mapping \
  --event-source-arn $STREAM_ARN \
  --function-name dynamodb-stream-processor \
  --starting-position LATEST \
  --batch-size 10

echo "DynamoDB 스트림 처리 설정 완료"

# 정리
rm -f stream_processor.py stream_processor.zip
```

#### DynamoDB 고급 쿼리 예제
```python
# advanced_dynamodb_operations.py
# DynamoDB 고급 작업 예제

import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal
import json

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('serverless-api-table')

class DynamoDBOperations:
    
    def __init__(self, table_name):
        self.table = dynamodb.Table(table_name)
    
    def batch_write_items(self, items):
        """배치로 아이템 작성"""
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)
    
    def query_by_status_and_date(self, status, start_date, end_date=None):
        """상태와 생성일자로 쿼리"""
        key_condition = Key('status').eq(status) & Key('created_at').gte(start_date)
        
        if end_date:
            key_condition = key_condition & Key('created_at').lte(end_date)
        
        response = self.table.query(
            IndexName='CreatedAtIndex',
            KeyConditionExpression=key_condition,
            ScanIndexForward=False  # 최신순 정렬
        )
        
        return response['Items']
    
    def scan_with_filter(self, filter_expression, limit=None):
        """필터 조건으로 스캔"""
        scan_kwargs = {
            'FilterExpression': filter_expression
        }
        
        if limit:
            scan_kwargs['Limit'] = limit
        
        response = self.table.scan(**scan_kwargs)
        return response['Items']
    
    def update_item_conditionally(self, item_id, updates, condition):
        """조건부 업데이트"""
        try:
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                update_expression += f"{key} = :{key}, "
                expression_values[f":{key}"] = value
            
            update_expression = update_expression.rstrip(", ")
            
            response = self.table.update_item(
                Key={'id': item_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ConditionExpression=condition,
                ReturnValues='ALL_NEW'
            )
            
            return response['Attributes']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                print("조건부 업데이트 실패: 조건을 만족하지 않음")
            raise
    
    def paginated_scan(self, page_size=20):
        """페이지네이션을 통한 스캔"""
        scan_kwargs = {
            'Limit': page_size
        }
        
        while True:
            response = self.table.scan(**scan_kwargs)
            
            for item in response['Items']:
                yield item
            
            if 'LastEvaluatedKey' not in response:
                break
            
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
    
    def get_item_with_projection(self, item_id, attributes):
        """특정 속성만 조회"""
        response = self.table.get_item(
            Key={'id': item_id},
            ProjectionExpression=','.join(attributes)
        )
        
        return response.get('Item')
    
    def transaction_write(self, operations):
        """트랜잭션 쓰기"""
        transact_items = []
        
        for op in operations:
            if op['action'] == 'put':
                transact_items.append({
                    'Put': {
                        'TableName': self.table.table_name,
                        'Item': op['item']
                    }
                })
            elif op['action'] == 'update':
                transact_items.append({
                    'Update': {
                        'TableName': self.table.table_name,
                        'Key': op['key'],
                        'UpdateExpression': op['update_expression'],
                        'ExpressionAttributeValues': op['expression_values']
                    }
                })
            elif op['action'] == 'delete':
                transact_items.append({
                    'Delete': {
                        'TableName': self.table.table_name,
                        'Key': op['key']
                    }
                })
        
        dynamodb.meta.client.transact_write_items(
            TransactItems=transact_items
        )

# 사용 예제
if __name__ == "__main__":
    db_ops = DynamoDBOperations('serverless-api-table')
    
    # 배치 작성 예제
    sample_items = [
        {
            'id': 'item1',
            'name': 'Sample Item 1',
            'status': 'active',
            'created_at': '2024-03-25T10:00:00Z'
        },
        {
            'id': 'item2', 
            'name': 'Sample Item 2',
            'status': 'inactive',
            'created_at': '2024-03-25T11:00:00Z'
        }
    ]
    
    db_ops.batch_write_items(sample_items)
    
    # 쿼리 예제
    active_items = db_ops.query_by_status_and_date('active', '2024-03-25T00:00:00Z')
    print(f"Active items: {len(active_items)}")
    
    # 필터링 스캔 예제
    filtered_items = db_ops.scan_with_filter(
        Attr('name').contains('Sample'),
        limit=10
    )
    print(f"Filtered items: {len(filtered_items)}")
```

## 다음 편 예고

다음 포스트에서는 **CloudFormation Infrastructure as Code와 CI/CD 파이프라인**을 상세히 다룰 예정입니다:
- CloudFormation 템플릿 완전 마스터
- SAM(Serverless Application Model) 활용
- CodePipeline으로 자동 배포 구축
- 모니터링 및 로깅 완전 설정

AWS 서버리스 아키텍처를 완전히 마스터하셨나요? ⚡🚀
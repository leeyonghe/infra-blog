---
layout: post
title: "Open-Sora 설정 시스템 상세 분석 - 하이퍼파라미터 튜닝 및 구성 관리"
date: 2024-12-14 21:00:00 +0900
categories: [AI, Video Generation, Configuration, Hyperparameter Tuning]
tags: [opensora, configuration, hyperparameters, settings, tuning]
description: "Open-Sora의 설정 시스템 상세 분석 - 모듈식 설정 구조, 하이퍼파라미터 튜닝 전략, 다단계 훈련 설정 및 추론 옵션 관리"
---

## 개요

Open-Sora는 복잡한 AI 비디오 생성 모델의 다양한 설정을 체계적으로 관리하기 위해 정교한 설정 시스템을 갖추고 있습니다. 이번 포스트에서는 Open-Sora의 설정 파일 구조, 모듈식 상속 시스템, 하이퍼파라미터 튜닝 전략, 그리고 훈련과 추론을 위한 다양한 구성 옵션들을 상세히 분석하겠습니다.

## 1. 설정 시스템 구조 개요

### 1.1 전체 구조

```
configs/
├── diffusion/              # Diffusion 모델 설정
│   ├── train/              # 훈련 설정
│   │   ├── image.py        # 기본 이미지 설정
│   │   ├── stage1.py       # 1단계 훈련
│   │   ├── stage2.py       # 2단계 훈련
│   │   ├── stage1_i2v.py   # I2V 1단계
│   │   └── stage2_i2v.py   # I2V 2단계
│   └── inference/          # 추론 설정
│       ├── 256px.py        # 256p 추론
│       ├── 768px.py        # 768p 추론
│       ├── t2i2v_256px.py  # T2I2V 256p
│       └── plugins/        # 플러그인 설정
└── vae/                    # VAE 모델 설정
    ├── train/              # VAE 훈련 설정
    └── inference/          # VAE 추론 설정
```

### 1.2 설정 시스템 특징

1. **모듈식 상속**: `_base_` 키워드를 통한 설정 상속
2. **계층적 구조**: 기본 설정 위에 특화 설정 오버라이드
3. **Python 기반**: 동적 설정 생성 및 조건부 로직 지원
4. **타입별 분리**: Diffusion, VAE 등 모델별 설정 분리

## 2. 기본 설정 구조 분석

### 2.1 이미지 기본 설정

```python
# configs/diffusion/train/image.py
# Dataset settings
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=24,  # 훈련 목표 FPS
    vmaf=True,   # VMAF 점수를 텍스트에 로드
)

# Gradient Checkpoint 설정
grad_ckpt_settings = (8, 100)  # (층 간격, 최대 단계)

# Bucket 설정 - 해상도별 배치 구성
bucket_config = {
    "256px": {1: (1.0, 50)},    # 1프레임: (확률, 배치 크기)
    "768px": {1: (0.5, 11)},    # 0.5 확률로 배치 크기 11
    "1024px": {1: (0.5, 7)},    # 0.5 확률로 배치 크기 7
}

# 모델 컴포넌트 정의
model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    
    # 모델 아키텍처
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
)
```

**핵심 설정 요소:**
- **Dataset**: 데이터 형식 및 전처리 설정
- **Bucket Config**: 해상도/프레임별 배치 크기 최적화
- **Model Architecture**: Transformer 구조 파라미터
- **Gradient Checkpointing**: 메모리 효율성 설정

### 2.2 텍스트 임베딩 설정

```python
# 텍스트 드롭아웃 확률
dropout_ratio = {
    "t5": 0.31622777,     # T5 드롭아웃 확률
    "clip": 0.31622777,   # CLIP 드롭아웃 확률
}

# T5 텍스트 인코더
t5 = dict(
    type="text_embedder",
    from_pretrained="google/t5-v1_1-xxl",
    cache_dir="/mnt/ddn/sora/tmp_load/huggingface/hub/",
    max_length=512,
    shardformer=True,  # 분산 최적화 활성화
)

# CLIP 텍스트 인코더
clip = dict(
    type="text_embedder",
    from_pretrained="openai/clip-vit-large-patch14",
    cache_dir="/mnt/ddn/sora/tmp_load/huggingface/hub/",
    max_length=77,
)
```

### 2.3 VAE 설정

```python
# VAE (Video Auto-Encoder)
ae = dict(
    type="hunyuan_vae",
    from_pretrained="./ckpts/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,   # 공간 타일링으로 메모리 절약
    use_temporal_tiling=False, # 시간 타일링 비활성화
)
is_causal_vae = True  # Causal VAE 사용
```

## 3. 최적화 설정 분석

### 3.1 옵티마이저 설정

```python
# 학습률 및 옵티마이저
lr = 1e-5
eps = 1e-15

optim = dict(
    cls="HybridAdam",        # ColossalAI의 하이브리드 Adam
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,         # AdamW 모드 활성화
)

# 학습률 스케줄링
warmup_steps = 0             # 웜업 단계 수
update_warmup_steps = True   # 체크포인트 로드시 웜업 업데이트

# 그래디언트 클리핑
grad_clip = 1.0
accumulation_steps = 1       # 그래디언트 누적 단계
ema_decay = None             # EMA 비활성화 (None)
```

**최적화 전략:**
- **HybridAdam**: 분산 환경에 최적화된 Adam 변형
- **Gradient Clipping**: 안정적인 훈련을 위한 그래디언트 제한
- **Warmup**: 점진적 학습률 증가

### 3.2 가속화 설정

```python
# 데이터 로딩 최적화
prefetch_factor = 2          # 프리페치 배수
num_workers = 12             # 데이터 로더 워커 수
num_bucket_build_workers = 64 # 버킷 구성 워커 수

# 정밀도 및 플러그인
dtype = "bf16"               # BFloat16 사용
plugin = "zero2"             # ZeRO Stage 2
grad_checkpoint = True       # Gradient Checkpointing 활성화

# 플러그인 상세 설정
plugin_config = dict(
    reduce_bucket_size_in_m=128,  # Reduce 버킷 크기 (MB)
    overlap_allgather=False,      # AllGather 중첩 비활성화
)

# 메모리 캐시 사전 할당
pin_memory_cache_pre_alloc_numels = [
    (260 + 20) * 1024 * 1024
] * 24 + [
    (34 + 20) * 1024 * 1024
] * 4

async_io = False  # 비동기 I/O 비활성화
```

**성능 최적화 요소:**
- **Mixed Precision**: BF16으로 메모리 절약
- **ZeRO Optimization**: 모델 파라미터 분산
- **Memory Prefetch**: 효율적인 데이터 로딩
- **Gradient Checkpointing**: 메모리 vs 계산 트레이드오프

## 4. 단계별 훈련 설정

### 4.1 Stage 1 훈련 설정

```python
# configs/diffusion/train/stage1.py
_base_ = ["image.py"]  # 기본 설정 상속

# 메모리 효율성 설정
dataset = dict(memory_efficient=False)

# 새로운 설정
grad_ckpt_settings = (8, 100)

# 확장된 버킷 설정
bucket_config = {
    "_delete_": True,  # 기존 설정 삭제
    "256px": {
        1: (1.0, 45),    # 1프레임
        5: (1.0, 12),    # 5프레임
        9: (1.0, 12),    # 9프레임
        13: (1.0, 12),   # 13프레임
        # ... 더 많은 프레임 설정 ...
        129: (1.0, 3),   # 129프레임
    },
    "768px": {1: (0.5, 13)},
    "1024px": {1: (0.5, 7)},
}

# 모델에 그래디언트 체크포인트 적용
model = dict(grad_ckpt_settings=grad_ckpt_settings)

# 업데이트된 하이퍼파라미터
lr = 5e-5
optim = dict(lr=lr)
ckpt_every = 2000      # 체크포인트 저장 간격
keep_n_latest = 20     # 최근 체크포인트 보관 수
```

**Stage 1 특징:**
- **다양한 프레임 수**: 1~129 프레임 지원
- **적응적 배치 크기**: 프레임 수에 따른 배치 크기 조정
- **높은 학습률**: 초기 빠른 학습을 위한 5e-5
- **빈번한 체크포인트**: 2000 스텝마다 저장

### 4.2 I2V (Image-to-Video) 설정

```python
# configs/diffusion/train/stage1_i2v.py
_base_ = ["stage1.py"]  # Stage 1 설정 상속

# I2V 특화 데이터셋 설정
dataset = dict(
    condition_config=dict(
        i2v_head=0.5,      # 첫 프레임 조건 확률 50%
        i2v_tail=0.1,      # 마지막 프레임 조건 확률 10%
        i2v_loop=0.1,      # 루프 조건 확률 10%
        t2v=0.3,           # 무조건 생성 확률 30%
    ),
)

# I2V에 최적화된 버킷 설정
bucket_config = {
    "_delete_": True,
    "256px": {
        # 더 긴 비디오 시퀀스 지원
        33: (1.0, 8),
        65: (1.0, 4),
        97: (1.0, 2),
        129: (1.0, 1),
    },
}
```

**I2V 훈련 특징:**
- **조건부 확률**: 다양한 조건 유형의 확률적 적용
- **긴 시퀀스**: 더 긴 비디오 생성 지원
- **단계적 배치 크기**: 시퀀스 길이에 반비례

## 5. 추론 설정 분석

### 5.1 기본 추론 설정

```python
# configs/diffusion/inference/256px.py
save_dir = "samples"  # 저장 디렉토리
seed = 42             # 랜덤 시드
batch_size = 1        # 배치 크기
dtype = "bf16"        # 데이터 타입

# 조건부 추론 옵션
cond_type = "t2v"     # 기본: text-to-video

# 조건부 추론 옵션들:
# t2v: text-to-video
# i2v_head: image-to-video (첫 프레임)
# i2v_tail: image-to-video (마지막 프레임)
# i2v_loop: 이미지 연결
# v2v_head_half: 비디오 확장 (첫 절반)
# v2v_tail_half: 비디오 확장 (두 번째 절반)

# 데이터셋 설정
dataset = dict(type="text")

# 샘플링 옵션
sampling_option = dict(
    resolution="256px",        # 해상도
    aspect_ratio="16:9",       # 종횡비
    num_frames=129,            # 프레임 수
    num_steps=50,              # 샘플링 단계
    shift=True,                # 시간 이동 활성화
    temporal_reduction=4,      # 시간 압축 비율
    is_causal_vae=True,        # Causal VAE 사용
    guidance=7.5,              # 텍스트 가이던스
    guidance_img=3.0,          # 이미지 가이던스
    text_osci=True,            # 텍스트 가이던스 진동
    image_osci=True,           # 이미지 가이던스 진동
    scale_temporal_osci=True,  # 시간 진동 스케일링
    method="i2v",              # 샘플링 방법
    seed=None,                 # z를 위한 랜덤 시드
)

motion_score = "4"    # 모션 점수
fps_save = 24         # 저장 FPS
```

**추론 옵션 분석:**
- **다양한 조건 타입**: T2V, I2V, V2V 지원
- **가이던스 제어**: 텍스트/이미지 가이던스 강도 조절
- **진동 기법**: 더 자연스러운 생성을 위한 가이던스 진동
- **시간 제어**: 프레임 수 및 시간 압축 설정

### 5.2 고해상도 추론 설정

```python
# configs/diffusion/inference/768px.py
_base_ = [
    "256px.py",        # 기본 256px 설정 상속
    "plugins/sp.py",   # Sequence Parallel 플러그인 사용
]

# 해상도만 오버라이드
sampling_option = dict(
    resolution="768px",
)
```

**고해상도 특징:**
- **플러그인 시스템**: Sequence Parallel로 메모리 효율성
- **설정 상속**: 기본 설정에서 해상도만 변경
- **스케일링**: 자동 배치 크기 및 메모리 조정

### 5.3 T2I2V (Text-to-Image-to-Video) 설정

```python
# configs/diffusion/inference/t2i2v_256px.py
_base_ = ["256px.py"]

# T2I2V 특화 설정
use_t2i2v = True
img_resolution = "768px"  # 중간 이미지 해상도

# 이미지 생성을 위한 별도 모델
img_flux = dict(
    type="flux_img",
    from_pretrained="./ckpts/flux_img.safetensors",
    # ... 이미지 모델 설정 ...
)

img_flux_ae = dict(
    type="flux_img_ae",
    from_pretrained="./ckpts/flux_img_ae.safetensors",
    # ... 이미지 VAE 설정 ...
)
```

## 6. VAE 훈련 설정 분석

### 6.1 기본 VAE 설정

```python
# configs/vae/train/video_dc_ae.py
# 모델 설정
model = dict(
    type="dc_ae",
    model_name="dc-ae-f32t4c128",
    from_scratch=True,
    from_pretrained=None,
)

# 데이터 설정
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    data_path="datasets/pexels_45k_necessary.csv",
    fps_max=24,
)

# VAE 특화 버킷 설정
bucket_config = {
    "256px_ar1:1": {32: (1.0, 1)},  # 1:1 종횡비, 32프레임
}

# 옵티마이저 설정
optim = dict(
    cls="HybridAdam",
    lr=5e-5,
    eps=1e-8,
    weight_decay=0.0,
    adamw_mode=True,
    betas=(0.9, 0.98),  # VAE에 최적화된 베타 값
)

# 혼합 전략
mixed_strategy = "mixed_video_image"
mixed_image_ratio = 0.2  # 이미지:비디오 = 1:4

# EMA 설정
ema_decay = 0.99  # VAE는 EMA 사용

# 손실 설정
vae_loss_config = dict(
    perceptual_loss_weight=0.5,  # 지각적 손실 가중치
    kl_loss_weight=0,            # KL 손실 비활성화
)
```

**VAE 훈련 특징:**
- **Mixed Training**: 비디오와 이미지 혼합 훈련
- **Perceptual Loss**: 시각적 품질 향상
- **EMA 사용**: 안정적인 VAE 학습
- **특화 베타 값**: VAE에 최적화된 Adam 파라미터

## 7. 플러그인 시스템

### 7.1 Sequence Parallel 플러그인

```python
# configs/diffusion/inference/plugins/sp.py
plugin = "hybrid"
plugin_config = dict(
    sp_size=2,           # Sequence Parallel 크기
    tp_size=1,           # Tensor Parallel 크기
    zero_stage=0,        # ZeRO 비활성화 (추론)
    enable_all_optimization=False,
    enable_flash_attention=False,
    enable_jit_fused=True,  # JIT 융합 활성화
    enable_sequence_parallelism=True,
)
```

### 7.2 커스텀 정책

```python
# MMDiT를 위한 커스텀 정책
custom_policy = "MMDiTPolicy"  # 특화된 최적화 정책
```

## 8. 동적 설정 생성

### 8.1 조건부 설정

```python
# 동적 설정 예제
def create_dynamic_config(resolution, num_frames):
    """해상도와 프레임 수에 따른 동적 설정 생성"""
    
    # 해상도별 배치 크기 계산
    if resolution == "256px":
        base_batch_size = 12
    elif resolution == "768px":
        base_batch_size = 4
    else:  # 1024px
        base_batch_size = 2
    
    # 프레임 수에 따른 배치 크기 조정
    if num_frames > 100:
        batch_size = max(1, base_batch_size // 4)
    elif num_frames > 50:
        batch_size = max(1, base_batch_size // 2)
    else:
        batch_size = base_batch_size
    
    return {
        "bucket_config": {
            resolution: {num_frames: (1.0, batch_size)}
        }
    }

# 사용 예제
config_256p_long = create_dynamic_config("256px", 129)
config_768p_short = create_dynamic_config("768px", 33)
```

### 8.2 환경별 설정

```python
# 환경별 설정 자동 조정
import torch

def get_environment_config():
    """현재 환경에 맞는 설정 반환"""
    gpu_count = torch.cuda.device_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    if gpu_memory > 40:  # A100 80GB
        return {
            "batch_size": 8,
            "plugin": "zero2",
            "dtype": "bf16",
            "grad_checkpoint": False,
        }
    elif gpu_memory > 20:  # RTX 4090, A100 40GB
        return {
            "batch_size": 4,
            "plugin": "zero2",
            "dtype": "bf16",
            "grad_checkpoint": True,
        }
    else:  # 일반 GPU
        return {
            "batch_size": 2,
            "plugin": "zero1",
            "dtype": "fp16",
            "grad_checkpoint": True,
        }
```

## 9. 하이퍼파라미터 튜닝 전략

### 9.1 학습률 스케줄링

```python
# 적응적 학습률 설정
learning_rate_configs = {
    "stage1": {
        "base_lr": 5e-5,
        "warmup_steps": 1000,
        "scheduler": "cosine",
        "min_lr": 1e-7,
    },
    "stage2": {
        "base_lr": 1e-5,
        "warmup_steps": 500,
        "scheduler": "linear",
        "min_lr": 1e-8,
    },
    "finetune": {
        "base_lr": 5e-6,
        "warmup_steps": 100,
        "scheduler": "constant",
    }
}
```

### 9.2 배치 크기 최적화

```python
# 해상도별 최적 배치 크기
optimal_batch_sizes = {
    "256px": {
        "short": (1, 16),     # 1-32 프레임
        "medium": (33, 8),    # 33-64 프레임  
        "long": (65, 4),      # 65-128 프레임
    },
    "768px": {
        "short": (1, 4),
        "medium": (33, 2),
        "long": (65, 1),
    },
    "1024px": {
        "short": (1, 2),
        "medium": (17, 1),
        "long": (33, 1),
    }
}
```

### 9.3 메모리 최적화 설정

```python
# 메모리 사용량별 설정
memory_optimization_configs = {
    "low_memory": {
        "grad_checkpoint": True,
        "activation_checkpointing": True,
        "offload_optimizer": True,
        "pin_memory": False,
    },
    "balanced": {
        "grad_checkpoint": True,
        "activation_checkpointing": False,
        "offload_optimizer": False,
        "pin_memory": True,
    },
    "high_memory": {
        "grad_checkpoint": False,
        "activation_checkpointing": False,
        "offload_optimizer": False,
        "pin_memory": True,
        "prefetch_factor": 4,
    }
}
```

## 10. 실제 사용 예제

### 10.1 커스텀 훈련 설정

```python
# custom_training_config.py
_base_ = ["configs/diffusion/train/stage1.py"]

# 커스텀 데이터셋
dataset = dict(
    data_path="my_custom_dataset.csv",
    fps_max=30,  # 30 FPS
    transform_name="center_crop",
)

# 커스텀 버킷 설정
bucket_config = {
    "_delete_": True,
    "512px": {
        1: (0.3, 8),
        17: (0.4, 4),
        33: (0.3, 2),
    }
}

# 더 보수적인 학습률
lr = 1e-5
optim = dict(lr=lr, weight_decay=0.01)

# 자주 체크포인트 저장
ckpt_every = 500
keep_n_latest = 10

# 커스텀 EMA 설정
ema_decay = 0.9999
```

### 10.2 고품질 추론 설정

```python
# high_quality_inference.py
_base_ = ["configs/diffusion/inference/768px.py"]

# 고품질 샘플링 설정
sampling_option = dict(
    resolution="768px",
    num_frames=65,
    num_steps=100,      # 더 많은 샘플링 단계
    guidance=10.0,      # 더 강한 가이던스
    guidance_img=5.0,
    shift=True,
    temporal_reduction=2,  # 더 세밀한 시간 해상도
)

# 더 높은 저장 FPS
fps_save = 60

# 시드 고정으로 재현 가능한 결과
seed = 42
sampling_option["seed"] = 42
```

### 10.3 빠른 프로토타이핑 설정

```python
# fast_prototype.py
_base_ = ["configs/diffusion/inference/256px.py"]

# 빠른 생성을 위한 설정
sampling_option = dict(
    resolution="256px",
    num_frames=17,      # 짧은 비디오
    num_steps=20,       # 적은 샘플링 단계
    guidance=5.0,       # 중간 가이던스
    temporal_reduction=8,  # 큰 시간 압축
)

# 낮은 저장 FPS
fps_save = 12

# 배치 처리
batch_size = 4
```

## 11. 설정 검증 및 디버깅

### 11.1 설정 유효성 검사

```python
# config_validator.py
def validate_config(config):
    """설정 유효성 검사"""
    errors = []
    
    # 필수 키 검사
    required_keys = ["model", "dataset", "optim"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    
    # 배치 크기 검사
    if "bucket_config" in config:
        for resolution, frames_config in config["bucket_config"].items():
            for frames, (prob, batch_size) in frames_config.items():
                if batch_size <= 0:
                    errors.append(f"Invalid batch size for {resolution}:{frames}")
                if not 0 <= prob <= 1:
                    errors.append(f"Invalid probability for {resolution}:{frames}")
    
    # 학습률 검사
    if "optim" in config and "lr" in config["optim"]:
        lr = config["optim"]["lr"]
        if lr <= 0 or lr > 1:
            errors.append(f"Invalid learning rate: {lr}")
    
    return errors

# 사용 예제
config = parse_configs()
validation_errors = validate_config(config)
if validation_errors:
    for error in validation_errors:
        print(f"Config Error: {error}")
```

### 11.2 설정 비교 도구

```python
# config_diff.py
def compare_configs(config1, config2):
    """두 설정 간 차이점 분석"""
    differences = {}
    
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in all_keys:
        if key not in config1:
            differences[key] = {"status": "added", "value": config2[key]}
        elif key not in config2:
            differences[key] = {"status": "removed", "value": config1[key]}
        elif config1[key] != config2[key]:
            differences[key] = {
                "status": "modified", 
                "old": config1[key], 
                "new": config2[key]
            }
    
    return differences

# 사용 예제
stage1_config = parse_configs("configs/diffusion/train/stage1.py")
stage2_config = parse_configs("configs/diffusion/train/stage2.py")
diff = compare_configs(stage1_config, stage2_config)
```

## 12. 한계점 및 개선 방향

### 12.1 현재 한계점

1. **복잡성**: 다양한 설정 옵션으로 인한 높은 학습 곡선
2. **의존성**: 설정 간 복잡한 의존 관계
3. **검증**: 런타임에서만 발견되는 설정 오류
4. **문서화**: 일부 고급 옵션의 부족한 문서화

### 12.2 개선 방향

```python
# 미래 개선 방향 (예시)
class NextGenConfigSystem:
    """차세대 설정 시스템"""
    
    def __init__(self):
        self.schema_validator = ConfigSchemaValidator()
        self.auto_tuner = AutoConfigTuner()
        self.dependency_manager = ConfigDependencyManager()
        
    def intelligent_config_generation(self, task_description):
        """태스크 설명으로부터 자동 설정 생성"""
        # LLM 기반 설정 생성
        # 하드웨어 자동 감지
        # 최적 하이퍼파라미터 추천
        pass
        
    def runtime_config_adaptation(self):
        """런타임 중 설정 적응"""
        # 메모리 사용량 모니터링
        # 자동 배치 크기 조정
        # 동적 최적화 설정 변경
        pass
        
    def config_explanation(self, config):
        """설정 옵션 자동 설명"""
        # 각 설정의 의미와 영향 설명
        # 성능 트레이드오프 분석
        # 대안 설정 제안
        pass
```

## 결론

Open-Sora의 설정 시스템은 복잡한 AI 비디오 생성 모델의 다양한 요구사항을 체계적으로 관리하는 정교한 시스템입니다.

**핵심 성과:**
- **모듈식 설계**: 상속과 오버라이드를 통한 유연한 설정 관리
- **단계별 최적화**: 다단계 훈련을 위한 체계적인 설정 구조  
- **하드웨어 적응**: 다양한 GPU 환경에 맞는 최적화 설정
- **타입별 특화**: Diffusion, VAE 등 모델별 최적화된 설정

이러한 설정 시스템은 Open-Sora가 연구용 프로토타입부터 프로덕션 환경까지 다양한 용도로 활용될 수 있게 하는 핵심 인프라입니다. 앞으로 더욱 지능적이고 자동화된 설정 관리 시스템으로 발전하여 사용자 편의성과 모델 성능을 동시에 향상시킬 것으로 기대됩니다.
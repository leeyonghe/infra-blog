---
layout: post
title: "Open-Sora 모델 아키텍처 심층 분석: STDiT3, VAE, T5의 완벽 이해"
date: 2024-06-03 11:00:00 +0900
categories: [AI, Deep Learning, Computer Vision]
tags: [opensora, transformer, vae, t5, stdit, diffusion-models, video-generation]
author: Lee Yonghe
description: "Open-Sora의 핵심 모델 구조를 상세히 분석합니다. STDiT3 트랜스포머, VAE 인코더/디코더, T5 텍스트 임베더의 구조와 동작 원리를 살펴봅니다."
image: /assets/images/opensora-architecture.png
---

## 개요

Open-Sora는 세 가지 핵심 컴포넌트로 구성된 복잡한 아키텍처를 가지고 있습니다. 이번 포스트에서는 각 컴포넌트의 구조와 역할, 그리고 서로 어떻게 상호작용하는지 상세히 분석해보겠습니다.

## 전체 아키텍처 개요

Open-Sora는 다음 세 가지 주요 모델로 구성됩니다:

```
Text Input → T5 Text Encoder → Text Embeddings
                                      ↓
Video/Image → VAE Encoder → Latent Space → STDiT3 Diffusion → Denoised Latent
                                                                      ↓
                                              VAE Decoder ← Generated Video/Image
```

### 핵심 구성 요소

1. **T5 Text Encoder**: 텍스트 프롬프트를 임베딩으로 변환
2. **VAE (Variational AutoEncoder)**: 비디오/이미지를 잠재 공간으로 압축/복원
3. **STDiT3 (Space-Time Diffusion Transformer)**: 잠재 공간에서 확산 과정 수행

## 1. T5 Text Encoder 분석

### 구조 및 특징

```python
@MODELS.register_module("text_embedder")
class HFEmbedder(nn.Module):
    def __init__(self, from_pretrained: str, max_length: int, shardformer: bool = False, **hf_kwargs):
        super().__init__()
        self.is_clip = "openai" in from_pretrained
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(from_pretrained, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(from_pretrained, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                from_pretrained, max_length=max_length, legacy=True
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(from_pretrained, **hf_kwargs)
```

### T5 vs CLIP 선택

**T5 선택 이유:**
- **더 긴 시퀀스 처리**: 복잡한 비디오 설명 가능
- **다국어 지원**: 한국어 포함 다양한 언어
- **강력한 텍스트 이해**: 문맥적 관계 파악

### Shardformer 최적화

```python
def shardformer_t5(t5: T5EncoderModel) -> T5EncoderModel:
    """T5 모델을 샤딩하여 최적화"""
    dtype = t5.shared.weight.dtype
    shard_config = ShardConfig(
        enable_tensor_parallelism=False,  # 텐서 병렬화 비활성화
        enable_jit_fused=True,           # JIT 융합 최적화 활성화
    )
    shard_former = ShardFormer(shard_config=shard_config)
    optim_model, _ = shard_former.optimize(t5, policy=T5EncoderPolicy())
    return optim_model.to(dtype).eval().requires_grad_(False)
```

**최적화 효과:**
- **JIT 융합**: 연산 그래프 최적화로 속도 향상
- **메모리 효율성**: 그래디언트 계산 비활성화
- **정밀도 보존**: 원본 데이터 타입 유지

### 텍스트 처리 파이프라인

```python
def forward(self, text: list[str], added_tokens: int = 0, seq_align: int = 1) -> Tensor:
    # 1. 토크나이징
    batch_encoding = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_length,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    
    # 2. 시퀀스 정렬 (배치 처리 최적화)
    seq_len = batch_encoding["input_ids"].shape[1]
    if (added_tokens + seq_len) % seq_align != 0:
        num_pad_tokens = seq_align - (added_tokens + seq_len) % seq_align
        batch_encoding["input_ids"] = nn.functional.pad(
            batch_encoding["input_ids"], (0, num_pad_tokens), value=self.tokenizer.pad_token_id
        )

    # 3. 임베딩 생성
    outputs = self.hf_module(
        input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
        attention_mask=None,
        output_hidden_states=False,
    )
    return outputs[self.output_key]
```

## 2. VAE (Variational AutoEncoder) 분석

### AutoEncoder 구성

```python
@dataclass
class AutoEncoderConfig:
    from_pretrained: str | None
    cache_dir: str | None
    resolution: int          # 해상도
    in_channels: int         # 입력 채널 수
    ch: int                  # 기본 채널 수
    out_ch: int             # 출력 채널 수
    ch_mult: list[int]      # 채널 배수 (다운샘플링용)
    num_res_blocks: int     # 잔차 블록 수
    z_channels: int         # 잠재 공간 채널 수
    scale_factor: float     # 스케일 팩터
    shift_factor: float     # 시프트 팩터
    sample: bool = True     # 샘플링 여부
```

### Attention Block 구조

```python
class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Group Normalization
        self.norm = nn.GroupNorm(
            num_groups=32, 
            num_channels=in_channels, 
            eps=1e-6, 
            affine=True
        )
        
        # Query, Key, Value 프로젝션
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        
        # 출력 프로젝션
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
```

**핵심 특징:**
- **Self-Attention**: 공간적 관계 학습
- **Group Normalization**: 배치 크기에 무관한 정규화
- **1x1 Convolution**: 효율적인 특징 변환

### 가우시안 분포 처리

```python
class DiagonalGaussianDistribution:
    def __init__(self, parameters: Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> Tensor:
        if self.deterministic:
            return self.mean
        else:
            return self.mean + self.std * torch.randn_like(self.mean)
```

**역할:**
- **잠재 공간 샘플링**: 확률적 인코딩
- **정규화**: 잠재 벡터의 분포 제어
- **안정성**: logvar 클램핑으로 수치 안정성

## 3. STDiT3 (Space-Time Diffusion Transformer)

### 아키텍처 개요

STDiT3는 Open-Sora의 핵심 생성 모델로, 다음과 같은 특징을 가집니다:

```python
# Gradio 앱에서의 STDiT3 로딩
from opensora.models.stdit.stdit3 import STDiT3

model_kwargs = {k: v for k, v in config.model.items() 
                if k not in ("type", "from_pretrained", "force_huggingface")}

print("Load STDIT3 from ", weight_path)
stdit = STDiT3.from_pretrained(weight_path, **model_kwargs).cuda()
```

### 핵심 혁신 사항

#### 1. Space-Time Attention
- **공간적 어텐션**: 프레임 내 픽셀 간 관계
- **시간적 어텐션**: 프레임 간 시간적 연속성
- **통합 처리**: 3D 어텐션으로 효율적 처리

#### 2. 다중 해상도 지원
```python
# 해상도별 모델 가중치
HF_STDIT_MAP = {
    "t2v": {
        "360p": "hpcaitech/OpenSora-STDiT-v4-360p",
        "720p": "hpcaitech/OpenSora-STDiT-v4",
    },
    "i2v": "hpcaitech/OpenSora-STDiT-v4-i2v",
}
```

#### 3. 조건부 생성
- **텍스트 조건**: T5 임베딩 활용
- **이미지 조건**: 첫 프레임 고정
- **모션 점수**: 동작 강도 제어

### 확산 과정 (Diffusion Process)

```python
# 샘플링 과정
samples = scheduler.sample(
    stdit,                    # STDiT3 모델
    text_encoder,            # T5 텍스트 인코더
    z=z,                     # 초기 노이즈
    z_cond=ref,              # 조건부 잠재 벡터
    z_cond_mask=x_cond_mask, # 조건부 마스크
    prompts=batch_prompts_loop,  # 배치 프롬프트
    device=device,
    additional_args=model_args,
    progress=True,
    mask=masks,
    mask_index=mask_index,
    image_cfg_scale=image_cfg_scale,
    use_sdedit=use_sdedit,
    use_oscillation_guidance_for_text=use_oscillation_guidance_for_text,
    use_oscillation_guidance_for_image=use_oscillation_guidance_for_image,
)
```

## 모델 간 상호작용

### 1. 텍스트 → 비디오 생성 파이프라인

```python
def build_models(mode, resolution, enable_optimization=False):
    """모델 구축 및 초기화"""
    
    # 1. VAE 구축
    vae = build_module(config.vae, MODELS).cuda()
    
    # 2. 텍스트 인코더 구축
    text_encoder = build_module(config.text_encoder, MODELS)
    text_encoder.t5.model = text_encoder.t5.model.cuda()
    
    # 3. STDiT3 구축
    stdit = STDiT3.from_pretrained(weight_path, **model_kwargs).cuda()
    
    # 4. 스케줄러 구축
    scheduler = build_module(config.scheduler, SCHEDULERS)
    
    # 5. CFG를 위한 임베더 연결
    text_encoder.y_embedder = stdit.y_embedder
    
    # 6. 최적화 및 평가 모드
    vae = vae.to(torch.bfloat16).eval()
    text_encoder.t5.model = text_encoder.t5.model.eval()
    stdit = stdit.to(torch.bfloat16).eval()
    
    return vae, text_encoder, stdit, scheduler, config
```

### 2. 이미지 → 비디오 생성 파이프라인

```python
# 참조 이미지 처리
refs = collect_references_batch(refs, vae, image_size)

# 조건부 참조 준비
if mode == "i2v":
    image_cfg_scale = config.get("image_cfg_scale", 7.5)
    ref, mask_index = prep_ref_and_mask(
        cond_type, condition_frame_length, refs, target_shape, 
        num_loop, device, dtype
    )
```

## 성능 최적화 기법

### 1. 메모리 최적화

```python
# 모델을 bfloat16으로 변환
vae = vae.to(torch.bfloat16).eval()
stdit = stdit.to(torch.bfloat16).eval()

# T5는 fp32 유지 (정확도 보장)
text_encoder.t5.model = text_encoder.t5.model.eval()
```

### 2. 분산 처리 지원

```python
# 멀티 GPU 추론
torchrun --nproc_per_node 8 --standalone scripts/diffusion/inference.py \
    configs/diffusion/inference/t2i2v_768px.py \
    --save-dir samples --prompt "raining, sea"
```

### 3. 동적 메모리 관리

```python
# CUDA 메모리 정리
torch.cuda.empty_cache()

# 그래디언트 비활성화
with torch.inference_mode():
    # 추론 코드
    pass
```

## 모델 크기 및 성능

### 파라미터 수
- **STDiT3**: 11B 파라미터 (메인 생성 모델)
- **T5**: 약 3B 파라미터 (텍스트 인코더)
- **VAE**: 약 83M 파라미터 (인코더/디코더)

### 메모리 요구사항
- **256x256 해상도**: 최소 32GB GPU 메모리
- **768x768 해상도**: 최소 80GB GPU 메모리 (A100 권장)

### 추론 시간
- **256x256, 5초 비디오**: 약 60초 (H100 1GPU)
- **768x768, 5초 비디오**: 약 276초 (H100 8GPU)

## 훈련 전략

### 1. 단계별 훈련
1. **VAE 사전 훈련**: 이미지/비디오 압축 학습
2. **텍스트 인코더 고정**: T5 가중치 동결
3. **STDiT3 훈련**: 확산 과정 학습

### 2. 다중 해상도 훈련
```python
# 해상도별 점진적 학습
resolutions = ["144p", "360p", "720p"]
for resolution in resolutions:
    train_model(resolution)
```

### 3. 데이터 효율성
- **캡션 품질**: 고품질 텍스트-비디오 쌍
- **다양성**: 다양한 장르와 스타일
- **길이 변화**: 2초~15초 비디오

## 실제 활용 시나리오

### 1. 콘텐츠 제작
```python
# 소셜 미디어 숏폼 생성
prompt = "A cat playing with a ball in a sunny garden, cute and playful"
video = generate_video(prompt, aspect_ratio="9:16", length="5s")
```

### 2. 영상 편집
```python
# 기존 이미지에서 비디오 생성
reference_image = load_image("portrait.jpg")
prompt = "The person smiles and nods gently"
video = generate_video(prompt, reference_image=reference_image)
```

### 3. 프로토타이핑
```python
# 광고 컨셉 시각화
prompt = "Modern minimalist product showcase, elegant lighting"
video = generate_video(prompt, resolution="720p", motion_score=3)
```

## 결론

Open-Sora의 아키텍처는 **모듈화**와 **최적화**의 완벽한 조합입니다.

**핵심 장점:**
- **분업화**: 각 컴포넌트의 명확한 역할 분담
- **확장성**: 해상도와 길이에 따른 유연한 스케일링
- **효율성**: bfloat16, Flash Attention 등 최신 최적화 기법
- **범용성**: 텍스트-투-비디오, 이미지-투-비디오 모두 지원

이러한 설계 철학은 다른 멀티모달 생성 모델 개발에도 중요한 참고 자료가 될 것입니다. 다음 포스트에서는 Gradio를 활용한 웹 인터페이스 구현을 살펴보겠습니다.

---

*이 글이 도움이 되셨다면 공유해주세요! 궁금한 점이 있으시면 댓글로 남겨주시기 바랍니다.*
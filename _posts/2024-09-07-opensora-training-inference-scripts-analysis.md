---
layout: post
title: "Open-Sora 훈련/추론 스크립트 상세 분석 - 실제 워크플로우와 파이프라인"
date: 2024-09-07 20:00:00 +0900
categories: [AI, Video Generation, Training, Inference, Pipeline]
tags: [opensora, training, inference, scripts, workflow, pipeline]
description: "Open-Sora의 훈련 및 추론 스크립트 상세 분석 - 실제 워크플로우, 파이프라인 구성, 분산 학습, 추론 엔진 및 데이터 처리 과정"
---

## 개요

Open-Sora의 scripts 디렉토리는 실제 모델 훈련과 추론을 수행하는 핵심 스크립트들을 포함하고 있습니다. 이번 포스트에서는 Diffusion 모델의 훈련 및 추론 스크립트, VAE 훈련 스크립트, 그리고 데이터 전처리 도구들을 상세히 분석하여 Open-Sora의 실제 동작 원리와 워크플로우를 파악하겠습니다.

## 1. 스크립트 구조 개요

### 1.1 전체 구조

```
scripts/
├── diffusion/              # Diffusion 모델 관련
│   ├── train.py            # Diffusion 모델 훈련
│   └── inference.py        # Diffusion 모델 추론
├── vae/                    # VAE 모델 관련
│   ├── train.py            # VAE 모델 훈련
│   ├── inference.py        # VAE 추론
│   └── stats.py            # VAE 통계 분석
└── cnv/                    # 데이터 변환 및 전처리
    ├── meta.py             # 메타데이터 추출
    └── shard.py            # 데이터 샤딩
```

### 1.2 핵심 기능 영역

1. **Diffusion 훈련**: 대규모 분산 훈련 파이프라인
2. **Diffusion 추론**: 조건부 비디오 생성 엔진
3. **VAE 훈련**: 비디오 인코더/디코더 학습
4. **데이터 처리**: 메타데이터 추출 및 전처리

## 2. Diffusion 추론 스크립트 분석

### 2.1 추론 파이프라인 구조

```python
# scripts/diffusion/inference.py
@torch.inference_mode()
def main():
    # ======================================================
    # 1. 설정 및 런타임 변수 초기화
    # ======================================================
    torch.set_grad_enabled(False)  # 추론 모드 설정

    # 설정 파싱
    cfg = parse_configs()
    cfg = parse_alias(cfg)

    # 디바이스 및 데이터 타입 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    seed = cfg.get("seed", 1024)
    if seed is not None:
        set_seed(seed)

    # 분산 환경 초기화
    init_inference_environment()
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    
    is_saving_process = get_is_saving_process(cfg)
    booster = get_booster(cfg)
    booster_ae = get_booster(cfg, ae=True)
```

**핵심 특징:**
- **Gradient 비활성화**: 추론 전용 최적화
- **Mixed Precision**: BF16 지원으로 메모리 효율성
- **분산 추론**: 멀티 GPU 추론 지원
- **설정 기반**: YAML 설정 파일로 유연한 구성

### 2.2 데이터셋 및 데이터로더 구성

```python
    # ======================================================
    # 2. 데이터셋 및 데이터로더 구성
    # ======================================================
    logger.info("Building dataset...")

    # 저장 디렉토리 설정
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 데이터셋 구성
    if cfg.get("prompt"):
        # 프롬프트가 주어진 경우 임시 CSV 생성
        cfg.dataset.data_path = create_tmp_csv(
            save_dir, cfg.prompt, cfg.get("ref", None), 
            create=is_main_process()
        )
    
    dist.barrier()  # 모든 프로세스 동기화
    dataset = build_module(cfg.dataset, DATASETS)

    # 범위 선택
    start_index = cfg.get("start_index", 0)
    end_index = cfg.get("end_index", None)
    if end_index is None:
        end_index = start_index + cfg.get("num_samples", len(dataset.data) + 1)
    dataset.data = dataset.data[start_index:end_index]
    logger.info("Dataset contains %s samples.", len(dataset))

    # 데이터로더 구성
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, _ = prepare_dataloader(**dataloader_args)
```

**데이터 처리 특징:**
- **동적 CSV 생성**: 프롬프트 기반 임시 데이터셋
- **범위 선택**: 부분 데이터셋 처리 지원
- **분산 데이터로더**: 멀티 GPU 환경 최적화
- **메모리 최적화**: Pin memory 및 prefetch 지원

### 2.3 모델 로딩 및 API 준비

```python
    # ======================================================
    # 3. 모델 구성
    # ======================================================
    logger.info("Building models...")

    # Flux 모델 구성
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=cfg.get("offload_model", False)
    )
    log_cuda_max_memory("build model")

    # Booster 적용
    if booster:
        model, _, _, _, _ = booster.boost(model=model)
        model = model.unwrap()
    if booster_ae:
        model_ae, _, _, _, _ = booster_ae.boost(model=model_ae)
        model_ae = model_ae.unwrap()

    # API 함수 준비
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    # T2I2V를 위한 이미지 Flux 모델 준비
    if use_t2i2v:
        api_fn_img = prepare_api(
            optional_models["img_flux"], 
            optional_models["img_flux_ae"], 
            model_t5, model_clip, optional_models
        )
```

**모델 관리 특징:**
- **다중 모델**: Diffusion, VAE, T5, CLIP 통합
- **선택적 오프로딩**: 메모리 절약을 위한 모델 이동
- **Booster 최적화**: ColossalAI 최적화 적용
- **API 추상화**: 통합된 추론 인터페이스

### 2.4 추론 루프

```python
    # ======================================================
    # 4. 추론 실행
    # ======================================================
    for epoch in range(num_sample):  # 다양한 시드로 다중 샘플 생성
        dataloader_iter = iter(dataloader)
        with tqdm(
            enumerate(dataloader_iter, start=0),
            desc="Inference progress",
            disable=not is_main_process(),
            initial=0,
            total=len(dataloader),
        ) as pbar:
            for _, batch in pbar:
                original_text = batch.pop("text")
                
                if use_t2i2v:
                    # T2I2V: 텍스트 → 이미지 → 비디오
                    batch["text"] = original_text if not prompt_refine else refine_prompts(original_text, type="t2i")
                    
                    # 이미지 생성을 위한 샘플링 옵션 수정
                    sampling_option_t2i = modify_option_to_t2i(
                        sampling_option,
                        distilled=True,
                        img_resolution=cfg.get("img_resolution", "768px"),
                    )
                    
                    # 모델 오프로딩 (메모리 절약)
                    if cfg.get("offload_model", False):
                        model_move_start = time.time()
                        model = model.to("cpu", dtype)
                        model_ae = model_ae.to("cpu", dtype)
                        optional_models["img_flux"].to(device, dtype)
                        optional_models["img_flux_ae"].to(device, dtype)
                        logger.info(
                            "offload video diffusion model to cpu, load image flux model to gpu: %s s",
                            time.time() - model_move_start,
                        )

                    # 이미지 조건 생성
                    logger.info("Generating image condition by flux...")
                    x_cond = api_fn_img(
                        sampling_option_t2i,
                        "t2v",
                        seed=sampling_option.seed + epoch if sampling_option.seed else None,
                        channel=cfg["img_flux"]["in_channels"],
                        **batch,
                    ).cpu()

                    # 이미지 저장
                    batch["name"] = process_and_save(
                        x_cond, batch, cfg, img_sub_dir,
                        sampling_option_t2i, epoch, start_index,
                        saving=is_saving_process,
                    )
                    dist.barrier()
```

**추론 프로세스 특징:**
- **다중 시드**: 다양한 결과 생성
- **T2I2V 지원**: 텍스트→이미지→비디오 파이프라인
- **동적 모델 관리**: 메모리 효율적인 모델 교체
- **진행률 표시**: tqdm을 통한 시각적 피드백

## 3. Diffusion 훈련 스크립트 분석

### 3.1 훈련 환경 설정

```python
# scripts/diffusion/train.py
def main():
    # ======================================================
    # 1. 설정 및 런타임 변수
    # ======================================================
    cfg = parse_configs()

    # 데이터 타입 및 디바이스 설정
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    
    # Gradient Checkpointing 버퍼 설정
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    
    # 메모리 캐시 설정
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels
```

**훈련 환경 특징:**
- **Mixed Precision**: BF16으로 메모리 효율성
- **Activation Checkpointing**: 메모리 절약
- **분산 환경**: 멀티 GPU 훈련
- **메모리 캐시**: 효율적인 데이터 로딩

### 3.2 모델 및 옵티마이저 설정

```python
    # ======================================================
    # 2. 모델 구성
    # ======================================================
    logger.info("Building model...")

    # VAE 모델 구성
    model_ae = build_module(cfg.ae, MODELS)
    model_ae = model_ae.to(device, dtype).eval()
    model_ae.requires_grad_(False)

    # 텍스트 인코더 구성
    if cfg.get("text_encoder", None) is not None:
        model_t5 = build_module(cfg.text_encoder, MODELS)
        model_t5 = model_t5.to(device, dtype).eval()
        model_t5.requires_grad_(False)
    else:
        model_t5 = lambda x: x

    model_clip = build_module(cfg.clip, MODELS)
    model_clip = model_clip.to(device, dtype).eval()
    model_clip.requires_grad_(False)

    # 메인 Diffusion 모델
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    model.train()

    # Gradient checkpointing 설정
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(
            model,
            use_fp32_attention=cfg.get("fp32_attention", False),
            gc_step=cfg.get("gc_step", 1),
        )

    # EMA 모델 설정
    if cfg.get("ema_decay", None) is not None:
        ema = deepcopy(model).eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
    else:
        ema = None
        ema_shape_dict = None

    # 옵티마이저 생성
    optimizer = create_optimizer(model, cfg.optim)
    lr_scheduler = create_lr_scheduler(optimizer, num_steps_per_epoch, **cfg.get("scheduler", {}))
```

**모델 구성 특징:**
- **모듈식 구조**: 각 컴포넌트 독립 구성
- **EMA 모델**: 안정적인 학습을 위한 지수이동평균
- **Gradient Checkpointing**: 메모리 효율적 역전파
- **고정 모델**: VAE와 텍스트 인코더는 고정

### 3.3 분산 훈련 준비

```python
    # =======================================================
    # 4. ColossalAI를 이용한 분산 훈련 준비
    # =======================================================
    logger.info("Preparing for distributed training...")
    
    # Boosting
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")
    log_cuda_memory("boost")
```

**분산 훈련 특징:**
- **ColossalAI Booster**: 자동 분산 최적화
- **ZeRO 최적화**: 메모리 효율적 파라미터 분산
- **데이터 병렬화**: 배치 분산 처리
- **Mixed Precision**: 자동 정밀도 관리

### 3.4 훈련 루프

```python
    # =======================================================
    # 5. 훈련 루프
    # =======================================================
    for epoch in range(start_epoch, cfg_epochs):
        for step, batch in enumerate(dataloader, start=start_step):
            # 시각적 조건 준비
            if cfg.get("causal", True):
                x_0, cond = prepare_visual_condition_causal(batch["video"], condition_config, model_ae)
            else:
                x_0, cond = prepare_visual_condition_uncausal(batch["video"], condition_config, model_ae)

            # 텍스트 조건 준비
            txt = model_t5(batch["text"], added_tokens=z_0.shape[1], seq_align=seq_align)
            vec = model_clip(batch["text"])

            # Dropout 적용
            txt = dropout_condition(cfg.get("text_dropout", 0.0), txt, null_txt)
            vec = dropout_condition(cfg.get("text_dropout", 0.0), vec, null_vec)

            # 노이즈 및 시간스텝 샘플링
            u = torch.randn_like(z_0)
            t = torch.rand(z_0.shape[0], device=device, dtype=torch.float32)
            
            # 시간 이동 (Time shift)
            t = time_shift(t, cfg.get("time_shift", 1.0), 1.0)

            # Rectified flow 준비
            zt = (1 - t) * z_0 + t * u
            vt = u - z_0

            # 모델 예측
            model_pred = model(
                x=pack([zt] + ([] if cond is None else [cond]), C),
                t=t.view(-1, 1, 1, 1),
                txt=txt,
                vec=vec,
                **pack_args,
            )

            # 손실 계산
            loss = get_batch_loss(model_pred, vt, cond[:, :1])  # 마스크 사용

            # 역전파 및 최적화
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # EMA 업데이트
            if ema is not None:
                update_ema(ema, model.module, optimizer, decay=cfg.ema_decay)
```

**훈련 프로세스 특징:**
- **조건부 학습**: 시각적/텍스트 조건 통합
- **Rectified Flow**: 효율적인 확산 과정
- **마스크 기반 손실**: 조건부 영역만 학습
- **EMA 업데이트**: 안정적인 모델 유지

## 4. VAE 훈련 스크립트 분석

### 4.1 VAE 특화 훈련 설정

```python
# scripts/vae/train.py
def main():
    # ======================================================
    # 1. 설정 및 런타임 변수
    # ======================================================
    cfg = parse_configs()

    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))

    # ColossalAI 플러그인 설정
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
    )

    booster = Booster(plugin=plugin)
```

**VAE 훈련 특징:**
- **ZeRO 최적화**: 대용량 VAE 모델 지원
- **혼합 정밀도**: 메모리 효율성
- **분산 훈련**: 멀티 GPU 지원

### 4.2 VAE 모델 및 손실 함수

```python
    # ======================================================
    # 2. VAE 모델 구성
    # ======================================================
    # 생성자 (Encoder-Decoder)
    generator = build_module(cfg.vae, MODELS)
    generator = generator.to(device, dtype)

    # 판별자 (Discriminator)
    discriminator = build_module(cfg.discriminator, MODELS) if cfg.get("discriminator", None) else None
    if discriminator is not None:
        discriminator = discriminator.to(device, dtype)

    # 손실 함수
    generator_loss_fn = GeneratorLoss(
        pixel_loss=cfg.generator.pixel_loss,
        perceptual_loss=cfg.generator.perceptual_loss,
        adversarial_loss=cfg.generator.adversarial_loss,
    )

    discriminator_loss_fn = DiscriminatorLoss(
        loss=cfg.discriminator.loss,
    ) if discriminator is not None else None

    vae_loss_fn = VAELoss(
        kl_loss=cfg.vae.kl_loss,
        pixel_loss=cfg.vae.pixel_loss,
    )
```

**VAE 구성 요소:**
- **생성자**: 인코더-디코더 구조
- **판별자**: 적대적 훈련용 (선택적)
- **다중 손실**: Pixel, Perceptual, KL, Adversarial 손실

### 4.3 VAE 훈련 루프

```python
    # ======================================================
    # 3. VAE 훈련 루프
    # ======================================================
    for epoch in range(start_epoch, cfg.max_epochs):
        for step, batch in enumerate(dataloader, start=start_step):
            # ===== Generator 훈련 =====
            # VAE forward
            posterior = generator.encode(batch["video"])
            z = posterior.sample()
            reconstructed = generator.decode(z)

            # VAE 손실 (KL + Reconstruction)
            vae_loss = vae_loss_fn(
                inputs=batch["video"],
                reconstructions=reconstructed,
                posteriors=posterior,
            )

            # Generator 손실 (Pixel + Perceptual + Adversarial)
            if discriminator is not None:
                # 판별자로부터 피드백
                logits_fake = discriminator(reconstructed)
            else:
                logits_fake = None

            generator_loss = generator_loss_fn(
                inputs=batch["video"],
                reconstructions=reconstructed,
                logits_fake=logits_fake,
            )

            total_g_loss = vae_loss + generator_loss

            # Generator 업데이트
            optimizer_g.zero_grad()
            booster.backward(total_g_loss, optimizer_g)
            optimizer_g.step()

            # ===== Discriminator 훈련 =====
            if discriminator is not None and step % cfg.get("d_reg_every", 1) == 0:
                # Real/Fake 분류
                logits_real = discriminator(batch["video"])
                logits_fake = discriminator(reconstructed.detach())

                # Discriminator 손실
                discriminator_loss = discriminator_loss_fn(
                    logits_real=logits_real,
                    logits_fake=logits_fake,
                )

                # Discriminator 업데이트
                optimizer_d.zero_grad()
                booster.backward(discriminator_loss, optimizer_d)
                optimizer_d.step()
```

**VAE 훈련 특징:**
- **이중 최적화**: Generator와 Discriminator 교대 훈련
- **다중 손실**: KL, Pixel, Perceptual, Adversarial 손실 조합
- **적대적 훈련**: GAN 스타일 훈련 (선택적)
- **정규화**: 주기적 판별자 업데이트

## 5. 데이터 전처리 도구 분석

### 5.1 메타데이터 추출 도구

```python
# scripts/cnv/meta.py
def get_video_info(path: str) -> pd.Series:
    """비디오 파일에서 메타데이터 추출"""
    vframes, _, vinfo = read_video(path, pts_unit="sec", output_format="TCHW")
    num_frames, C, height, width = vframes.shape
    fps = round(vinfo["video_fps"], 3)
    aspect_ratio = height / width if width > 0 else np.nan
    resolution = height * width

    ret = pd.Series(
        [height, width, fps, num_frames, aspect_ratio, resolution],
        index=[
            "height", "width", "fps", "num_frames", 
            "aspect_ratio", "resolution",
        ],
        dtype=object,
    )
    return ret

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    
    # 병렬 처리 설정
    apply = set_parallel(args.num_workers)
    
    # 메타데이터 추출
    result = apply(df["path"], get_video_info)
    for col in result.columns:
        df[col] = result[col]
    
    df.to_csv(args.output, index=False)
```

**메타데이터 추출 특징:**
- **병렬 처리**: 다중 프로세스로 빠른 처리
- **포괄적 정보**: 해상도, FPS, 프레임 수, 종횡비
- **배치 처리**: CSV 파일 단위 처리
- **진행률 표시**: tqdm 통합

### 5.2 데이터 병렬 처리

```python
def set_parallel(num_workers: int = None) -> callable:
    """병렬 처리 함수 설정"""
    if num_workers == 0:
        # 단일 프로세스 모드
        return lambda x, *args, **kwargs: x.progress_apply(*args, **kwargs)
    else:
        # 다중 프로세스 모드
        if num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
        return lambda x, *args, **kwargs: x.parallel_apply(*args, **kwargs)
```

## 6. 실제 사용 예제

### 6.1 추론 실행 예제

```bash
# 기본 텍스트-투-비디오 추론
python scripts/diffusion/inference.py \
    --config configs/opensora-v1-2/inference/sample.yaml \
    --prompt "A beautiful sunset over the ocean with waves" \
    --save_dir ./outputs \
    --num_samples 4 \
    --seed 42

# T2I2V (텍스트→이미지→비디오) 추론
python scripts/diffusion/inference.py \
    --config configs/opensora-v1-2/inference/sample.yaml \
    --prompt "A cat playing in the garden" \
    --use_t2i2v true \
    --img_resolution 768px \
    --save_dir ./outputs

# 조건부 비디오 생성 (I2V)
python scripts/diffusion/inference.py \
    --config configs/opensora-v1-2/inference/sample.yaml \
    --prompt "Continue this scene" \
    --cond_type i2v_head \
    --ref /path/to/reference_image.jpg \
    --save_dir ./outputs
```

### 6.2 훈련 실행 예제

```bash
# Diffusion 모델 훈련
python scripts/diffusion/train.py \
    --config configs/opensora-v1-2/train/stage1.yaml \
    --data_path /path/to/video_dataset.csv \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4

# VAE 모델 훈련
python scripts/vae/train.py \
    --config configs/vae/train.yaml \
    --data_path /path/to/video_dataset.csv \
    --max_epochs 50 \
    --batch_size 8 \
    --use_discriminator true

# 체크포인트에서 재개
python scripts/diffusion/train.py \
    --config configs/opensora-v1-2/train/stage2.yaml \
    --load /path/to/checkpoint \
    --start_epoch 10 \
    --update_warmup_steps true
```

### 6.3 데이터 전처리 예제

```bash
# 비디오 메타데이터 추출
python scripts/cnv/meta.py \
    --input video_list.csv \
    --output video_with_metadata.csv \
    --num_workers 8

# 입력 CSV 형식
# path
# /path/to/video1.mp4
# /path/to/video2.mp4

# 출력 CSV 형식 (메타데이터 추가됨)
# path,height,width,fps,num_frames,aspect_ratio,resolution
# /path/to/video1.mp4,720,1280,30.0,150,0.5625,921600
```

## 7. 성능 최적화 및 모니터링

### 7.1 메모리 최적화 기법

```python
# 메모리 효율적 추론 설정
@torch.inference_mode()
def memory_efficient_inference():
    # Gradient 비활성화
    torch.set_grad_enabled(False)
    
    # 모델 오프로딩
    if cfg.get("offload_model", False):
        # 사용하지 않는 모델을 CPU로 이동
        model.to("cpu")
        model_ae.to("cpu")
        
        # 필요할 때만 GPU로 로드
        model.to(device)
        output = model(input_data)
        model.to("cpu")
    
    # Mixed precision 사용
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_data)
```

### 7.2 분산 훈련 모니터링

```python
# 훈련 진행률 및 메트릭 모니터링
def training_monitoring():
    # 메모리 사용량 로깅
    log_cuda_memory("after forward")
    log_cuda_memory("after backward")
    log_cuda_max_memory("training step")
    
    # 손실 로깅
    if step % log_every == 0:
        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}")
        
        # Tensorboard 로깅
        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
    
    # 체크포인트 저장
    if step % ckpt_every == 0:
        checkpoint_io.save(
            booster, save_dir, model=model, ema=ema,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            epoch=epoch, step=step, global_step=global_step
        )
```

### 7.3 성능 프로파일링

```python
# PyTorch Profiler를 이용한 성능 분석
from torch.profiler import profile, ProfilerActivity

def profile_training():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, batch in enumerate(dataloader):
            # 훈련 코드
            loss = training_step(batch)
            loss.backward()
            optimizer.step()
            
            prof.step()  # 프로파일러 스텝
            
            if step >= (1 + 1 + 3) * 1:
                break
```

## 8. 고급 워크플로우 패턴

### 8.1 다단계 훈련 파이프라인

```python
# 다단계 훈련 워크플로우
class MultiStageTraining:
    def __init__(self, config):
        self.config = config
        self.current_stage = 0
        
    def run_stage(self, stage_config):
        """단일 스테이지 실행"""
        logger.info(f"Starting training stage {self.current_stage}")
        
        # 모델 및 데이터 준비
        model = self.prepare_model(stage_config)
        dataloader = self.prepare_dataloader(stage_config)
        optimizer = self.create_optimizer(model, stage_config)
        
        # 이전 스테이지에서 체크포인트 로드
        if self.current_stage > 0:
            self.load_checkpoint(model, optimizer)
        
        # 훈련 실행
        for epoch in range(stage_config.epochs):
            self.train_epoch(model, dataloader, optimizer)
            
        # 다음 스테이지를 위한 체크포인트 저장
        self.save_checkpoint(model, optimizer)
        self.current_stage += 1
        
    def run_full_pipeline(self):
        """전체 다단계 파이프라인 실행"""
        for stage_config in self.config.stages:
            self.run_stage(stage_config)
```

### 8.2 적응적 배치 크기 조정

```python
# 메모리에 따른 적응적 배치 크기
class AdaptiveBatchSize:
    def __init__(self, initial_batch_size=4, max_batch_size=16):
        self.batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.oom_count = 0
        
    def adjust_batch_size(self, dataloader):
        """OOM 발생시 배치 크기 조정"""
        try:
            # 정상 훈련
            for batch in dataloader:
                yield batch
                
        except torch.cuda.OutOfMemoryError:
            self.oom_count += 1
            
            if self.batch_size > 1:
                # 배치 크기 절반으로 감소
                self.batch_size = max(1, self.batch_size // 2)
                logger.warning(f"OOM detected. Reducing batch size to {self.batch_size}")
                
                # 새로운 데이터로더 생성
                new_dataloader = self.create_dataloader(self.batch_size)
                yield from self.adjust_batch_size(new_dataloader)
            else:
                raise  # 배치 크기가 1인데도 OOM이면 에러
```

## 9. 한계점 및 개선 방향

### 9.1 현재 한계점

1. **메모리 사용량**: 대용량 비디오 처리시 높은 메모리 요구
2. **훈련 시간**: 긴 훈련 시간과 수렴 속도
3. **설정 복잡성**: 다양한 하이퍼파라미터 튜닝 필요
4. **디버깅**: 분산 환경에서의 디버깅 어려움

### 9.2 개선 방향

```python
# 미래 개선 방향 (예시)
class NextGenTrainingPipeline:
    """차세대 훈련 파이프라인"""
    
    def __init__(self):
        self.auto_tuner = AutoHyperparameterTuner()
        self.memory_optimizer = IntelligentMemoryManager()
        self.convergence_accelerator = ConvergenceAccelerator()
        
    def intelligent_training(self):
        """지능형 훈련 시스템"""
        # 자동 하이퍼파라미터 튜닝
        optimal_params = self.auto_tuner.find_optimal_params()
        
        # 적응적 메모리 관리
        self.memory_optimizer.optimize_memory_usage()
        
        # 수렴 가속화
        self.convergence_accelerator.accelerate_training()
        
    def real_time_monitoring(self):
        """실시간 모니터링 및 알림"""
        # 성능 이상 감지
        # 자동 체크포인트 저장
        # 슬랙/이메일 알림
        pass
```

## 결론

Open-Sora의 훈련 및 추론 스크립트는 대규모 AI 비디오 생성 모델을 위한 완전한 파이프라인을 제공합니다.

**핵심 성과:**
- **완전한 워크플로우**: 데이터 전처리부터 모델 배포까지
- **분산 처리**: 멀티 GPU 훈련 및 추론 지원
- **메모리 효율성**: 다양한 최적화 기법 적용
- **유연한 설정**: 설정 기반의 모듈식 구조

이러한 스크립트들은 Open-Sora가 연구용 프로토타입을 넘어 실제 프로덕션 환경에서 사용 가능한 수준의 완성도를 보여줍니다. 앞으로 더욱 지능적이고 자동화된 훈련/추론 시스템으로 발전하여 사용자 편의성과 성능을 동시에 향상시킬 것으로 기대됩니다.
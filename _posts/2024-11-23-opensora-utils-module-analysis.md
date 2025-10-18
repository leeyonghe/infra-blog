---
layout: post
title: "Open-Sora 유틸리티 모듈 상세 분석 - 추론, 학습, 체크포인트 및 메모리 관리"
date: 2024-11-23 19:00:00 +0900
categories: [AI, Video Generation, Utilities, Memory Management]
tags: [opensora, utils, inference, training, checkpoint, optimization]
description: "Open-Sora의 유틸리티 모듈 상세 분석 - 추론 엔진, 학습 파이프라인, 체크포인트 관리, 메모리 최적화 및 핵심 헬퍼 함수들"
---

## 개요

Open-Sora의 유틸리티 모듈은 AI 비디오 생성 모델의 핵심 기능을 지원하는 다양한 헬퍼 함수와 클래스들을 포함하고 있습니다. 이번 포스트에서는 추론 엔진, 학습 파이프라인, 체크포인트 관리, 메모리 최적화 등 Open-Sora 시스템의 실제 동작을 뒷받침하는 핵심 유틸리티들을 상세히 분석하겠습니다.

## 1. 유틸리티 모듈 구조 개요

### 1.1 전체 구조

```
opensora/utils/
├── inference.py         # 추론 엔진 및 조건부 생성
├── train.py            # 학습 파이프라인 및 최적화
├── ckpt.py             # 체크포인트 관리 시스템
├── misc.py             # 메모리 모니터링 및 기타 유틸리티
├── optimizer.py        # 옵티마이저 및 스케줄러
├── sampling.py         # 샘플링 및 텍스트 처리
├── config.py           # 설정 관리
├── logger.py           # 로깅 시스템
├── prompt_refine.py    # 프롬프트 개선
└── cai.py             # ColossalAI 통합
```

### 1.2 핵심 기능 영역

1. **추론 시스템**: 조건부 생성 및 샘플 처리
2. **학습 파이프라인**: 분산 학습 및 최적화
3. **체크포인트 관리**: 모델 저장/로드 시스템
4. **메모리 관리**: 메모리 모니터링 및 최적화
5. **헬퍼 함수**: 다양한 보조 기능들

## 2. 추론 시스템 상세 분석

### 2.1 조건부 비디오 생성

```python
# opensora/utils/inference.py
def prepare_inference_condition(
    z: torch.Tensor,
    mask_cond: str,
    ref_list: list[list[torch.Tensor]] = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    추론을 위한 시각적 조건 준비
    
    Args:
        z: 잠재 노이즈 텐서 [B, C, T, H, W]
        mask_cond: 조건 타입 ("i2v_head", "v2v_head", "t2v" 등)
        ref_list: 참조 미디어 리스트
        causal: Causal VAE 사용 여부
        
    Returns:
        masks, masked_z: 마스크와 조건부 잠재 벡터
    """
    B, C, T, H, W = z.shape
    
    # 마스크 및 조건부 텐서 초기화
    masks = torch.zeros(B, 1, T, H, W)
    masked_z = torch.zeros(B, C, T, H, W)
    
    if ref_list is None:
        assert mask_cond == "t2v", f"reference is required for {mask_cond}"

    for i in range(B):
        ref = ref_list[i]
        
        if ref is not None and T > 1:  # 비디오 생성
            if mask_cond == "i2v_head":  # 첫 프레임 조건
                masks[i, :, 0, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
                
            elif mask_cond == "i2v_tail":  # 마지막 프레임 조건
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
                
            elif mask_cond == "v2v_head":  # 비디오 시작 부분 조건
                k = 8 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
                
            elif mask_cond == "v2v_tail":  # 비디오 끝 부분 조건
                k = 8 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
                
            elif mask_cond == "i2v_loop":  # 루프 비디오 (시작+끝 조건)
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
                
            else:
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"

    masks = masks.to(z.device, z.dtype)
    masked_z = masked_z.to(z.device, z.dtype)
    return masks, masked_z
```

**조건부 생성 타입:**
- **i2v_head**: 이미지 → 비디오 (첫 프레임 고정)
- **i2v_tail**: 이미지 → 비디오 (마지막 프레임 고정)
- **i2v_loop**: 루프 비디오 (시작과 끝 프레임 고정)
- **v2v_head/tail**: 비디오 → 비디오 (일부 프레임 조건)
- **t2v**: 텍스트 → 비디오 (무조건부)

### 2.2 참조 미디어 수집

```python
def collect_references_batch(
    reference_paths: list[str],
    cond_type: str,
    model_ae: nn.Module,
    image_size: tuple[int, int],
    is_causal=False,
):
    """
    배치 단위로 참조 미디어 수집 및 인코딩
    """
    refs_x = []
    device = next(model_ae.parameters()).device
    dtype = next(model_ae.parameters()).dtype
    
    for reference_path in reference_paths:
        if reference_path == "":
            refs_x.append(None)
            continue
            
        ref_path = reference_path.split(";")
        ref = []

        if "v2v" in cond_type:
            # 비디오-투-비디오: 연속 프레임 처리
            r = read_from_path(ref_path[0], image_size, transform_name="resize_crop")
            actual_t = r.size(1)
            target_t = 64 if (actual_t >= 64 and "easy" in cond_type) else 32
            
            if is_causal:
                target_t += 1
                
            assert actual_t >= target_t, f"need at least {target_t} reference frames"
            
            if "head" in cond_type:
                r = r[:, :target_t]
            elif "tail" in cond_type:
                r = r[:, -target_t:]
                
            r_x = model_ae.encode(r.unsqueeze(0).to(device, dtype))
            ref.append(r_x.squeeze(0))
            
        elif cond_type == "i2v_head":
            # 이미지-투-비디오: 첫 프레임
            r = read_from_path(ref_path[0], image_size, transform_name="resize_crop")
            r = r[:, :1]  # 첫 프레임만
            r_x = model_ae.encode(r.unsqueeze(0).to(device, dtype))
            ref.append(r_x.squeeze(0))
            
        elif cond_type == "i2v_loop":
            # 루프 비디오: 첫 프레임 + 마지막 프레임
            r_head = read_from_path(ref_path[0], image_size, transform_name="resize_crop")
            r_head = r_head[:, :1]
            r_x_head = model_ae.encode(r_head.unsqueeze(0).to(device, dtype))
            ref.append(r_x_head.squeeze(0))
            
            r_tail = read_from_path(ref_path[-1], image_size, transform_name="resize_crop")
            r_tail = r_tail[:, -1:]
            r_x_tail = model_ae.encode(r_tail.unsqueeze(0).to(device, dtype))
            ref.append(r_x_tail.squeeze(0))

        refs_x.append(ref)
    
    return refs_x
```

### 2.3 샘플 처리 및 저장

```python
def process_and_save(
    x: torch.Tensor,
    batch: dict,
    cfg: dict,
    sub_dir: str,
    generate_sampling_option,
    epoch: int,
    start_index: int,
    saving: bool = True,
):
    """
    생성된 샘플 처리 및 디스크 저장
    """
    fallback_name = cfg.dataset.data_path.split("/")[-1].split(".")[0]
    prompt_as_path = cfg.get("prompt_as_path", False)
    fps_save = cfg.get("fps_save", 16)
    save_dir = cfg.save_dir

    names = batch["name"] if "name" in batch else [None] * len(x)
    indices = batch["index"] if "index" in batch else [None] * len(x)
    prompts = batch["text"]

    ret_names = []
    is_image = generate_sampling_option.num_frames == 1
    
    for img, name, index, prompt in zip(x, names, indices, prompts):
        # 저장 경로 생성
        save_path = get_save_path_name(
            save_dir, sub_dir, 
            save_prefix=cfg.get("save_prefix", ""),
            name=name, fallback_name=fallback_name,
            index=index, num_sample_pos=epoch,
            prompt_as_path=prompt_as_path, prompt=prompt,
        )
        
        ret_name = get_names_from_path(save_path)
        ret_names.append(ret_name)

        if saving:
            # 프롬프트 텍스트 저장
            with open(save_path + ".txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            # 샘플 저장 (비디오/이미지)
            save_sample(img, save_path=save_path, fps=fps_save)

            # T2I2V를 위한 이미지 리사이징
            if (cfg.get("use_t2i2v", False) and is_image and 
                generate_sampling_option.resolution != generate_sampling_option.resized_resolution):
                height, width = get_image_size(
                    generate_sampling_option.resized_resolution, 
                    generate_sampling_option.aspect_ratio
                )
                rescale_image_by_path(save_path + ".png", width, height)

    return ret_names
```

## 3. 학습 파이프라인 분석

### 3.1 분산 학습 환경 설정

```python
# opensora/utils/train.py
def setup_device() -> tuple[torch.device, DistCoordinator]:
    """
    디바이스 및 분산 코디네이터 설정
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # 매우 긴 타임아웃 설정 (24시간)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    
    coordinator = DistCoordinator()
    device = get_current_device()
    
    return device, coordinator

def create_colossalai_plugin(
    plugin: str,
    dtype: str,
    grad_clip: float,
    **kwargs,
) -> LowLevelZeroPlugin | HybridParallelPlugin:
    """
    ColossalAI 플러그인 생성
    """
    plugin_kwargs = dict(
        precision=dtype,
        initial_scale=2**16,
        max_norm=grad_clip,
        overlap_allgather=True,
        cast_inputs=False,
        reduce_bucket_size_in_m=20,
    )
    plugin_kwargs.update(kwargs)
    sp_size = plugin_kwargs.get("sp_size", 1)
    
    if plugin == "zero1" or plugin == "zero2":
        assert sp_size == 1, "Zero plugin does not support sequence parallelism"
        stage = 1 if plugin == "zero1" else 2
        plugin = LowLevelZeroPlugin(stage=stage, **plugin_kwargs)
        set_data_parallel_group(dist.group.WORLD)
        
    elif plugin == "hybrid":
        plugin_kwargs["find_unused_parameters"] = True
        plugin_kwargs["enable_metadata_cache"] = False
        
        custom_policy = plugin_kwargs.pop("custom_policy", None)
        if custom_policy is not None:
            custom_policy = custom_policy()
            
        plugin = HybridParallelPlugin(custom_policy=custom_policy, **plugin_kwargs)
        set_tensor_parallel_group(plugin.tp_group)
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
        
    else:
        raise ValueError(f"Unknown plugin {plugin}")
        
    return plugin
```

### 3.2 시각적 조건 준비 (학습용)

```python
def prepare_visual_condition_causal(
    x: torch.Tensor, 
    condition_config: dict, 
    model_ae: torch.nn.Module
) -> torch.Tensor:
    """
    Causal VAE를 위한 시각적 조건 준비
    """
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    masks = torch.zeros(B, 1, T, H, W).to(x.device, x.dtype)
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    
    if T > 1:  # 비디오
        # 짧은 비디오에 적용되지 않는 조건들 제거
        if T <= (32 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head", None)
            condition_config.pop("v2v_tail", None)
            condition_config.pop("v2v_head_easy", None)
            condition_config.pop("v2v_tail_easy", None)

        mask_cond_options = list(condition_config.keys())
        mask_cond_weights = list(condition_config.values())

        for i in range(B):
            # 확률에 따른 마스크 조건 랜덤 선택
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            
            if mask_cond == "i2v_head":
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                
            elif mask_cond == "i2v_loop":
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
                
            elif "v2v_head" in mask_cond:
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = model_ae.encode(x[i, :, :ref_t, :, :].unsqueeze(0))
                
            elif "v2v_tail" in mask_cond:
                ref_t = 33 if not "easy" in mask_cond else 65
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = model_ae.encode(x[i, :, -ref_t:, :, :].unsqueeze(0))
                
            else:
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
    else:  # 이미지
        x_0 = model_ae.encode(x)

    latent = masks * latent
    cond = torch.cat((masks, latent), dim=1)
    return x_0, cond
```

### 3.3 EMA 업데이트 시스템

```python
@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, 
    model: torch.nn.Module, 
    optimizer=None, 
    decay: float = 0.9999, 
    sharded: bool = True
):
    """
    EMA 모델을 현재 모델 방향으로 업데이트
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
            
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
```

### 3.4 배치 손실 계산

```python
def get_batch_loss(model_pred, v_t, masks=None):
    """
    I2V를 위한 배치 손실 계산 (생성된 프레임만 포함)
    """
    if masks is not None:
        num_frames, height, width = masks.shape[-3:]
        masks = masks[:, :, 0, 0]  # [B, T]만 보기
        
        # 텐서 재배열
        model_pred = rearrange(
            model_pred,
            "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
            h=height // 2, w=width // 2, t=num_frames, ph=2, pw=2,
        )
        v_t = rearrange(
            v_t,
            "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
            h=height // 2, w=width // 2, t=num_frames, ph=2, pw=2,
        )

        batch_loss = 0
        for i in range(model_pred.size(0)):
            pred_val = model_pred[i]
            target_val = v_t[i]
            
            # 앞/뒤 패딩이 있는 경우 제외
            if masks[i][0] == 1 and (not 1 in masks[i][1:-1]):
                pred_val = pred_val[:, 1:]
                target_val = target_val[:, 1:]
            if masks[i][-1] == 1 and (not 1 in masks[i][1:-1]):
                pred_val = pred_val[:, :-1]
                target_val = target_val[:, :-1]
                
            batch_loss += F.mse_loss(pred_val.float(), target_val.float(), reduction="mean")
            
        loss = batch_loss / model_pred.size(0)
    else:
        loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")
    
    return loss
```

## 4. 체크포인트 관리 시스템

### 4.1 다양한 체크포인트 로드

```python
# opensora/utils/ckpt.py
def load_checkpoint(
    model: nn.Module,
    path: str,
    cache_dir: str = None,
    device_map: torch.device | str = "cpu",
    cai_model_name: str = "model",
    strict: bool = False,
    rename_keys: dict = None,
) -> nn.Module:
    """
    다양한 형태의 체크포인트 로드 지원:
    1. Hugging Face safetensors
    2. 로컬 .pt/.pth 파일
    3. ColossalAI 샤드 체크포인트
    """
    if not os.path.exists(path):
        log_message(f"Checkpoint not found at {path}, trying to download from Hugging Face Hub")
        path = load_from_hf_hub(path, cache_dir)
    
    assert os.path.exists(path), f"Could not find checkpoint at {path}"
    log_message(f"Loading checkpoint from {path}")
    
    if path.endswith(".safetensors"):
        ckpt = load_file(path, device='cpu')
        
        # 키 이름 변경 (fine-tuning 지원)
        if rename_keys is not None:
            renamed_ckpt = {}
            for old_key, v in ckpt.items():
                new_key = old_key
                for old_key_prefix, new_key_prefix in rename_keys.items():
                    if old_key_prefix in old_key:
                        new_key = old_key.replace(old_key_prefix, new_key_prefix)
                        print(f"Renamed {old_key} to {new_key} in the loaded state_dict")
                        break
                renamed_ckpt[new_key] = v
            ckpt = renamed_ckpt

        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print_load_warning(missing, unexpected)
        
    elif path.endswith(".pt") or path.endswith(".pth"):
        ckpt = torch.load(path, map_location=device_map)
        missing, unexpected = model.load_state_dict(ckpt, strict=strict)
        print_load_warning(missing, unexpected)
        
    else:
        assert os.path.isdir(path), f"Invalid checkpoint path: {path}"
        load_from_sharded_state_dict(model, path, model_name=cai_model_name, strict=strict)
    
    return model
```

### 4.2 고성능 체크포인트 I/O

```python
class CheckpointIO:
    """
    비동기 I/O를 지원하는 고성능 체크포인트 관리자
    """
    def __init__(self, n_write_entries: int = 32):
        self.n_write_entries = n_write_entries
        self.writer: Optional[AsyncFileWriter] = None
        self.pinned_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.master_pinned_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.master_writer: Optional[AsyncFileWriter] = None

    def save(
        self,
        booster: Booster,
        save_dir: str,
        model: nn.Module = None,
        ema: nn.Module = None,
        optimizer: Optimizer = None,
        lr_scheduler: _LRScheduler = None,
        sampler=None,
        epoch: int = None,
        step: int = None,
        global_step: int = None,
        batch_size: int = None,
        lora: bool = False,
        actual_update_step: int = None,
        ema_shape_dict: dict = None,
        async_io: bool = True,
        include_master_weights: bool = False,
    ) -> str:
        """
        포괄적인 체크포인트 저장
        """
        self._sync_io()
        save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{actual_update_step}")
        os.environ["TENSORNVME_DEBUG_LOG"] = os.path.join(save_dir, "async_file_io.log")
        
        # 모델 저장
        if model is not None:
            if not lora:
                os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
                booster.save_model(
                    model, os.path.join(save_dir, "model"),
                    shard=True, use_safetensors=True, size_per_shard=4096,
                    use_async=async_io,
                )
            else:
                os.makedirs(os.path.join(save_dir, "lora"), exist_ok=True)
                booster.save_lora_as_pretrained(model, os.path.join(save_dir, "lora"))
        
        # 옵티마이저 저장
        if optimizer is not None:
            booster.save_optimizer(
                optimizer, os.path.join(save_dir, "optimizer"),
                shard=True, size_per_shard=4096, use_async=async_io
            )
            if include_master_weights:
                self._prepare_master_pinned_state_dict(model, optimizer)
                master_weights_gathering(model, optimizer, self.master_pinned_state_dict)
        
        # EMA 모델 저장
        if ema is not None:
            self._prepare_pinned_state_dict(ema, ema_shape_dict)
            model_gathering(ema, ema_shape_dict, self.pinned_state_dict)
        
        # 메타데이터 저장 (rank 0만)
        if dist.get_rank() == 0:
            running_states = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "batch_size": batch_size,
                "actual_update_step": actual_update_step,
            }
            save_json(running_states, os.path.join(save_dir, "running_states.json"))

            if ema is not None:
                if async_io:
                    self.writer = async_save(os.path.join(save_dir, "ema.safetensors"), self.pinned_state_dict)
                else:
                    torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))

            if optimizer is not None and include_master_weights:
                self.master_writer = async_save(
                    os.path.join(save_dir, "master.safetensors"), self.master_pinned_state_dict
                )

        dist.barrier()
        return save_dir
```

### 4.3 분산 모델 수집

```python
def model_gathering(model: torch.nn.Module, model_shape_dict: dict, pinned_state_dict: dict) -> None:
    """
    여러 GPU에서 모델 파라미터 수집
    """
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    params = set()
    
    for name, param in model.named_parameters():
        params.add(name)
        # 모든 rank에서 파라미터 수집
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            gathered_param = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name])
            pinned_state_dict[name].copy_(gathered_param)
    
    # 버퍼 처리 (rank 0만)
    if int(global_rank) == 0:
        for k, v in model.state_dict(keep_vars=True).items():
            if k not in params:
                pinned_state_dict[k].copy_(v)

    dist.barrier()
```

## 5. 메모리 관리 및 모니터링

### 5.1 CUDA 메모리 모니터링

```python
# opensora/utils/misc.py
GIGABYTE = 1024**3

def log_cuda_memory(stage: str = None):
    """
    현재 CUDA 메모리 사용량 로깅
    """
    text = "CUDA memory usage"
    if stage is not None:
        text += f" at {stage}"
    log_message(text + ": %.1f GB", torch.cuda.memory_allocated() / GIGABYTE)

def log_cuda_max_memory(stage: str = None):
    """
    최대 CUDA 메모리 사용량 로깅
    """
    torch.cuda.synchronize()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()
    
    log_message("CUDA max memory allocated at " + stage + ": %.1f GB", max_memory_allocated / GIGABYTE)
    log_message("CUDA max memory reserved at " + stage + ": %.1f GB", max_memory_reserved / GIGABYTE)

def get_model_numel(model: torch.nn.Module) -> tuple[int, int]:
    """
    모델 파라미터 수 계산
    """
    num_params = 0
    num_params_trainable = 0
    
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
            
    return num_params, num_params_trainable

def log_model_params(model: nn.Module):
    """
    모델 파라미터 수 로깅
    """
    num_params, num_params_trainable = get_model_numel(model)
    log_message(f"Model parameters: {num_params:,} total, {num_params_trainable:,} trainable")
```

### 5.2 Tensorboard 및 로깅

```python
def create_tensorboard_writer(exp_dir: str) -> SummaryWriter:
    """
    Tensorboard writer 생성
    """
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer
```

## 6. 옵티마이저 및 스케줄러

### 6.1 옵티마이저 생성

```python
# opensora/utils/optimizer.py
def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    옵티마이저 생성
    """
    optimizer_name = optimizer_config.pop("cls", "HybridAdam")
    
    if optimizer_name == "HybridAdam":
        optimizer_cls = HybridAdam
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_config,
    )
    return optimizer
```

### 6.2 학습률 스케줄러

```python
class LinearWarmupLR(_LRScheduler):
    """
    선형 웜업 학습률 스케줄러
    """
    def __init__(self, optimizer, initial_lr=0, warmup_steps: int = 0, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 웜업 단계: 선형 증가
            return [
                self.initial_lr + (self.last_epoch + 1) / (self.warmup_steps + 1) * (lr - self.initial_lr)
                for lr in self.base_lrs
            ]
        else:
            # 웜업 완료: 기본 학습률 사용
            return self.base_lrs

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_steps_per_epoch: int,
    epochs: int = 1000,
    warmup_steps: int | None = None,
    use_cosine_scheduler: bool = False,
    initial_lr: float = 1e-6,
) -> _LRScheduler | None:
    """
    학습률 스케줄러 생성
    """
    if warmup_steps is None and not use_cosine_scheduler:
        lr_scheduler = None
    elif use_cosine_scheduler:
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer,
            total_steps=num_steps_per_epoch * epochs,
            warmup_steps=warmup_steps,
        )
    else:
        lr_scheduler = LinearWarmupLR(optimizer, initial_lr=initial_lr, warmup_steps=warmup_steps)

    return lr_scheduler
```

## 7. 실제 사용 예제

### 7.1 추론 파이프라인 예제

```python
# 추론을 위한 조건 준비 예제
def inference_example():
    # 설정
    device = torch.device("cuda")
    batch_size = 2
    channels = 16
    frames = 64
    height = 32
    width = 32
    
    # 잠재 노이즈 텐서
    z = torch.randn(batch_size, channels, frames, height, width).to(device)
    
    # 참조 이미지/비디오 (예시)
    ref_list = [
        [torch.randn(channels, 1, height, width).to(device)],  # 첫 번째 배치: 이미지
        [torch.randn(channels, 8, height, width).to(device)]   # 두 번째 배치: 비디오
    ]
    
    # I2V 조건 준비
    masks, masked_z = prepare_inference_condition(
        z=z,
        mask_cond="i2v_head",
        ref_list=ref_list,
        causal=True
    )
    
    print(f"Masks shape: {masks.shape}")
    print(f"Masked z shape: {masked_z.shape}")
    print(f"Number of conditioned frames: {masks.sum()}")

# 텍스트 처리 예제
def text_processing_example():
    prompts = [
        "A beautiful sunset over the ocean",
        "A cat playing in the garden"
    ]
    
    # FPS 정보 추가
    modified_prompts = add_fps_info_to_text(prompts, fps=24)
    print("Modified prompts:", modified_prompts)
    
    # 모션 스코어 추가
    motion_prompts = add_motion_score_to_text(prompts, motion_score=5)
    print("Motion prompts:", motion_prompts)
```

### 7.2 학습 파이프라인 예제

```python
# 분산 학습 설정 예제
def training_setup_example():
    # 디바이스 설정
    device, coordinator = setup_device()
    
    # ColossalAI 플러그인 생성
    plugin = create_colossalai_plugin(
        plugin="hybrid",
        dtype="bf16",
        grad_clip=1.0,
        sp_size=2,
        tp_size=2,
        zero_stage=2,
    )
    
    # 가상 모델 생성
    model = torch.nn.Linear(1024, 1024).to(device)
    
    # 옵티마이저 설정
    optimizer_config = {
        "cls": "HybridAdam",
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
    }
    optimizer = create_optimizer(model, optimizer_config)
    
    # 스케줄러 설정
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=1000,
        epochs=100,
        warmup_steps=1000,
        use_cosine_scheduler=True,
    )
    
    return model, optimizer, lr_scheduler

# EMA 업데이트 예제
def ema_update_example():
    # 메인 모델과 EMA 모델
    main_model = torch.nn.Linear(1024, 512)
    ema_model = torch.nn.Linear(1024, 512)
    
    # EMA 모델 초기화 (메인 모델 가중치 복사)
    ema_model.load_state_dict(main_model.state_dict())
    
    # 학습 루프에서 EMA 업데이트
    for step in range(100):
        # ... 실제 학습 코드 ...
        
        # EMA 업데이트 (매 스텝마다)
        update_ema(
            ema_model=ema_model,
            model=main_model,
            decay=0.9999,
            sharded=False
        )
        
        if step % 10 == 0:
            print(f"Step {step}: EMA updated")
```

### 7.3 체크포인트 관리 예제

```python
# 체크포인트 저장/로드 예제
def checkpoint_example():
    # 모델 생성
    model = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(512, 8),
        num_layers=6
    )
    
    # 체크포인트 로드 (다양한 형식 지원)
    model = load_checkpoint(
        model=model,
        path="path/to/checkpoint.safetensors",
        strict=False,
        rename_keys={
            "old_prefix": "new_prefix"  # Fine-tuning 지원
        }
    )
    
    # CheckpointIO를 사용한 고성능 저장
    checkpoint_io = CheckpointIO()
    
    # EMA 모델과 함께 저장
    ema_model = copy.deepcopy(model)
    ema_shape_dict = record_model_param_shape(ema_model)
    
    save_path = checkpoint_io.save(
        booster=booster,  # ColossalAI Booster
        save_dir="./checkpoints",
        model=model,
        ema=ema_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=10,
        step=1000,
        global_step=10000,
        batch_size=32,
        ema_shape_dict=ema_shape_dict,
        async_io=True,
        include_master_weights=True,
    )
    
    print(f"Checkpoint saved to: {save_path}")
```

## 8. 성능 최적화 및 모니터링

### 8.1 메모리 사용량 추적

```python
# 메모리 모니터링 예제
def memory_monitoring_example():
    # 초기 메모리 상태
    log_cuda_memory("initialization")
    
    # 모델 로드
    model = torch.nn.Linear(10000, 10000).cuda()
    log_cuda_memory("after model load")
    
    # 데이터 로드
    data = torch.randn(1000, 10000).cuda()
    log_cuda_memory("after data load")
    
    # Forward pass
    output = model(data)
    log_cuda_memory("after forward")
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    log_cuda_memory("after backward")
    
    # 최대 메모리 사용량 로그
    log_cuda_max_memory("training step")
    
    # 모델 파라미터 수 로그
    log_model_params(model)
```

### 8.2 성능 프로파일링

```python
# 성능 측정 예제
def performance_profiling():
    import time
    
    model = torch.nn.Linear(1024, 1024).cuda()
    data = torch.randn(32, 1024).cuda()
    
    # 웜업
    for _ in range(10):
        _ = model(data)
    
    torch.cuda.synchronize()
    
    # 실제 측정
    start_time = time.time()
    torch.cuda.synchronize()
    
    for _ in range(100):
        output = model(data)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    throughput = 32 / avg_time  # 배치 크기 / 시간
    
    print(f"Average inference time: {avg_time:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
```

## 9. 한계점 및 개선 방향

### 9.1 현재 한계점

1. **메모리 오버헤드**: 다양한 조건부 생성으로 인한 메모리 사용량 증가
2. **I/O 병목**: 대용량 체크포인트 저장/로드 시간
3. **복잡성**: 다양한 조건 타입으로 인한 코드 복잡도
4. **디버깅**: 분산 환경에서의 디버깅 어려움

### 9.2 개선 방향

```python
# 미래 개선 방향 (예시)
class NextGenUtilities:
    """차세대 유틸리티 시스템"""
    
    def __init__(self):
        self.smart_memory_manager = SmartMemoryManager()
        self.adaptive_checkpoint_io = AdaptiveCheckpointIO()
        self.unified_condition_system = UnifiedConditionSystem()
        
    def smart_memory_optimization(self):
        """지능형 메모리 최적화"""
        # 동적 메모리 할당 및 해제
        # 예측 기반 메모리 관리
        pass
        
    def compressed_checkpoint_io(self):
        """압축된 체크포인트 I/O"""
        # 실시간 압축/압축해제
        # 점진적 체크포인트 저장
        pass
        
    def unified_condition_handling(self):
        """통합된 조건 처리 시스템"""
        # 단일 인터페이스로 모든 조건 타입 지원
        # 자동 조건 최적화
        pass
```

## 결론

Open-Sora의 유틸리티 모듈은 AI 비디오 생성 시스템의 핵심 기능을 지원하는 포괄적인 도구 모음입니다.

**핵심 성과:**
- **유연한 추론**: 다양한 조건부 생성 모드 지원
- **효율적 학습**: 분산 학습 및 EMA 업데이트 시스템
- **강력한 체크포인트**: 비동기 I/O 및 다양한 형식 지원
- **메모리 관리**: 실시간 모니터링 및 최적화 도구

이러한 유틸리티들은 Open-Sora가 대규모 비디오 생성 태스크를 안정적이고 효율적으로 수행할 수 있게 하는 핵심 인프라를 제공합니다. 앞으로 더욱 지능적이고 자동화된 시스템으로 발전하여 사용자 편의성과 성능을 동시에 향상시킬 것으로 기대됩니다.
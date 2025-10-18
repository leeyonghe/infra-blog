---
layout: post
title: "Adversarial Networks 심화 분석 - AudioCraft Custom 프로젝트"
date: 2024-12-20
categories: [AI, Deep Learning, Adversarial Networks]
tags: [AudioCraft, GAN, Adversarial Training, MPD, MSD, MS-STFT-D, Audio Quality]
author: "AI Blog"
---

# Adversarial Networks 심화 분석

AudioCraft Custom 프로젝트의 핵심 품질 향상 메커니즘인 적대적 네트워크 시스템을 심층 분석해보겠습니다. Multi-Period Discriminator (MPD), Multi-Scale Discriminator (MSD), Multi-Scale STFT Discriminator (MS-STFT-D) 등 세 가지 판별자가 어떻게 협력하여 고품질 오디오를 생성하는지 살펴보겠습니다.

## 📋 목차
1. [적대적 학습 기본 개념](#적대적-학습-기본-개념)
2. [Multi-Period Discriminator (MPD)](#multi-period-discriminator-mpd)
3. [Multi-Scale Discriminator (MSD)](#multi-scale-discriminator-msd)  
4. [Multi-Scale STFT Discriminator (MS-STFT-D)](#multi-scale-stft-discriminator-ms-stft-d)
5. [다중 판별자 협력 메커니즘](#다중-판별자-협력-메커니즘)
6. [실제 성능 향상 분석](#실제-성능-향상-분석)

## 적대적 학습 기본 개념

### 기본 아키텍처

```python
class MultiDiscriminator(ABC, nn.Module):
    """Base implementation for discriminators composed of sub-discriminators acting at different scales.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        ...

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        """Number of discriminators."""
        ...
```

#### 🔧 핵심 타입 정의
```python
FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
MultiDiscriminatorOutputType = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]
```

### 적대적 학습의 목표
- **생성자**: 판별자를 속이는 고품질 오디오 생성
- **판별자**: 실제와 생성된 오디오를 정확히 구분
- **균형**: 두 네트워크의 경쟁을 통한 품질 향상

## Multi-Period Discriminator (MPD)

### 기본 개념 및 설계

```python
class MultiPeriodDiscriminator(MultiDiscriminator):
    """Multi-Period (MPD) Discriminator.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        periods (Sequence[int]): Periods between samples of audio for the sub-discriminators.
        **kwargs: Additional args for `PeriodDiscriminator`
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 periods: tp.Sequence[int] = [2, 3, 5, 7, 11], **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, in_channels, out_channels, **kwargs) for p in periods
        ])
```

#### 🎵 주기별 분석의 핵심
- **다양한 주기**: [2, 3, 5, 7, 11] - 소수 주기로 다양한 패턴 포착
- **주기적 패턴**: 음악의 리듬, 비트, 주기적 구조 분석
- **1D→2D 변환**: 주기별로 오디오를 2차원으로 재구성

### Period Sub-Discriminator 구현

```python
class PeriodDiscriminator(nn.Module):
    """Period sub-discriminator.
    
    Args:
        period (int): Period between samples of audio.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_layers (int): Number of convolutional layers.
        kernel_sizes (list of int): Kernel sizes for convolutions.
        stride (int): Stride for convolutions.
        filters (int): Initial number of filters in convolutions.
        filters_scale (int): Multiplier of number of filters as we increase depth.
        max_filters (int): Maximum number of filters.
    """
```

#### 🔄 1D→2D 변환 메커니즘

```python
def forward(self, x: torch.Tensor):
    fmap = []
    # 1d to 2d 변환
    b, c, t = x.shape
    if t % self.period != 0:  # 패딩 먼저
        n_pad = self.period - (t % self.period)
        x = F.pad(x, (0, n_pad), 'reflect')
        t = t + n_pad
    x = x.view(b, c, t // self.period, self.period)
    
    for conv in self.convs:
        x = conv(x)
        x = self.activation(x)
        fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    
    return x, fmap
```

#### 🧮 변환 과정 세부 분석
1. **길이 조정**: 주기로 나누어떨어지도록 반사 패딩 추가
2. **차원 재구성**: `[B, C, T]` → `[B, C, T//period, period]`
3. **2D 컨볼루션**: 주기 내 패턴과 주기 간 패턴 동시 분석
4. **특징 맵 수집**: 각 레이어별 중간 특징 맵 저장

### 컨볼루션 레이어 구성

```python
def __init__(self, period: int, in_channels: int = 1, out_channels: int = 1,
             n_layers: int = 5, kernel_sizes: tp.List[int] = [5, 3], stride: int = 3,
             filters: int = 8, filters_scale: int = 4, max_filters: int = 1024,
             norm: str = 'weight_norm', activation: str = 'LeakyReLU',
             activation_params: dict = {'negative_slope': 0.2}):
    
    # 메인 컨볼루션 레이어들
    for i in range(self.n_layers):
        out_chs = min(filters * (filters_scale ** (i + 1)), max_filters)
        eff_stride = 1 if i == self.n_layers - 1 else stride
        self.convs.append(NormConv2d(in_chs, out_chs, 
                                    kernel_size=(kernel_sizes[0], 1), 
                                    stride=(eff_stride, 1),
                                    padding=((kernel_sizes[0] - 1) // 2, 0), 
                                    norm=norm))
        in_chs = out_chs
    
    # 최종 출력 레이어
    self.conv_post = NormConv2d(in_chs, out_channels, 
                               kernel_size=(kernel_sizes[1], 1), stride=1,
                               padding=((kernel_sizes[1] - 1) // 2, 0), norm=norm)
```

#### 📈 필터 증가 패턴
- **초기 필터**: 8개에서 시작
- **증가 비율**: 각 레이어마다 4배씩 증가
- **최대 제한**: 1024개까지 제한
- **적응적 스트라이드**: 마지막 레이어에서만 stride=1

## Multi-Scale Discriminator (MSD)

### 다중 스케일 설계

```python
class MultiScaleDiscriminator(MultiDiscriminator):
    """Multi-Scale (MSD) Discriminator,
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample_factor (int): Downsampling factor between the different scales.
        scale_norms (Sequence[str]): Normalization for each sub-discriminator.
        **kwargs: Additional args for ScaleDiscriminator.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, downsample_factor: int = 2,
                 scale_norms: tp.Sequence[str] = ['weight_norm', 'weight_norm', 'weight_norm'], **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(in_channels, out_channels, norm=norm, **kwargs) for norm in scale_norms
        ])
        self.downsample = nn.AvgPool1d(downsample_factor * 2, downsample_factor, padding=downsample_factor)
```

#### 🔍 스케일별 분석
1. **원본 스케일**: 고해상도 세부 사항 분석
2. **다운샘플 스케일**: 중간 해상도 구조 분석  
3. **더 작은 스케일**: 저해상도 전역 패턴 분석

### Scale Sub-Discriminator 구현

```python
def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
    logits = []
    fmaps = []
    for i, disc in enumerate(self.discriminators):
        if i != 0:
            x = self.downsample(x)  # 스케일 감소
        logit, fmap = disc(x)
        logits.append(logit)
        fmaps.append(fmap)
    return logits, fmaps
```

#### ⚡ 다운샘플링 전략
- **평균 풀링**: `AvgPool1d`로 부드러운 다운샘플링
- **점진적 감소**: 첫 번째 이후 스케일마다 적용
- **정보 보존**: 평균 풀링으로 급격한 정보 손실 방지

### 스케일별 컨볼루션 설계

```python
class ScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes: tp.Sequence[int] = [5, 3],
                 filters: int = 16, max_filters: int = 1024, 
                 downsample_scales: tp.Sequence[int] = [4, 4, 4, 4],
                 inner_kernel_sizes: tp.Optional[tp.Sequence[int]] = None, 
                 groups: tp.Optional[tp.Sequence[int]] = None,
                 norm: str = 'weight_norm', activation: str = 'LeakyReLU'):
```

#### 🏗️ 적응적 커널 설계
```python
for i, downsample_scale in enumerate(downsample_scales):
    out_chs = min(in_chs * downsample_scale, max_filters)
    default_kernel_size = downsample_scale * 10 + 1  # 동적 커널 크기
    default_stride = downsample_scale
    default_padding = (default_kernel_size - 1) // 2
    default_groups = in_chs // 4  # 그룹 컨볼루션
```

#### 💡 설계 철학
- **동적 커널**: 다운샘플 스케일에 비례한 커널 크기
- **그룹 컨볼루션**: 계산 효율성과 특징 다양성 균형
- **최대 필터 제한**: 메모리 사용량 제어

## Multi-Scale STFT Discriminator (MS-STFT-D)

### STFT 기반 분석

```python
class MultiScaleSTFTDiscriminator(MultiDiscriminator):
    """Multi-Scale STFT (MS-STFT) discriminator.
    
    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        sep_channels (bool): Separate channels to distinct samples for stereo support.
        n_ffts (Sequence[int]): Size of FFT for each scale.
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale.
        win_lengths (Sequence[int]): Window size for each scale.
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1, sep_channels: bool = False,
                 n_ffts: tp.List[int] = [1024, 2048, 512], 
                 hop_lengths: tp.List[int] = [256, 512, 128],
                 win_lengths: tp.List[int] = [1024, 2048, 512], **kwargs):
```

#### 🎛️ 다중 스케일 STFT 설정
- **고해상도**: n_fft=2048, 상세한 주파수 분석
- **중해상도**: n_fft=1024, 균형잡힌 시간-주파수 해상도
- **저해상도**: n_fft=512, 빠른 시간 변화 포착

### STFT Sub-Discriminator 구현

```python
class DiscriminatorSTFT(nn.Module):
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, 
                 max_filters: int = 1024, filters_scale: int = 1, 
                 kernel_size: tp.Tuple[int, int] = (3, 9), 
                 dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True):
```

#### 🔄 STFT 변환 과정

```python
def forward(self, x: torch.Tensor):
    fmap = []
    z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
    z = torch.cat([z.real, z.imag], dim=1)  # 실수/허수 결합
    z = rearrange(z, 'b c w t -> b c t w')  # 차원 재배열
    
    for i, layer in enumerate(self.convs):
        z = layer(z)
        z = self.activation(z)
        fmap.append(z)
    z = self.conv_post(z)
    return z, fmap
```

#### 🌊 스펙트로그램 처리
1. **STFT 변환**: 시간 도메인 → 주파수 도메인
2. **복소수 처리**: 실수부와 허수부를 별도 채널로 처리
3. **2D 컨볼루션**: 시간-주파수 2차원에서 패턴 분석
4. **팽창 컨볼루션**: 다양한 팽창률로 넓은 수용 영역

### 팽창 컨볼루션 활용

```python
for i, dilation in enumerate(dilations):
    out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
    self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                 dilation=(dilation, 1), 
                                 padding=get_2d_padding(kernel_size, (dilation, 1)),
                                 norm=norm))
    in_chs = out_chs
```

#### 📊 팽창률 전략
- **dilation=[1, 2, 4]**: 점진적으로 수용 영역 확장
- **시간축만 팽창**: `(dilation, 1)` - 주파수축은 국소적 유지
- **다중 해상도**: 다양한 시간 스케일의 패턴 동시 포착

## 다중 판별자 협력 메커니즘

### 전체 아키텍처 통합

```python
# 세 가지 판별자의 협력
discriminators = {
    'mpd': MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11]),
    'msd': MultiScaleDiscriminator(downsample_factor=2),
    'msstftd': MultiScaleSTFTDiscriminator(n_ffts=[1024, 2048, 512])
}
```

#### 🎯 각 판별자의 특화 영역
- **MPD**: 주기적 패턴과 리듬 구조
- **MSD**: 다중 스케일 파형 특성
- **MS-STFT-D**: 주파수 도메인 세부 사항

### 손실 함수 결합

```python
def adversarial_loss(discriminators, real_audio, fake_audio):
    total_loss = 0
    
    for disc_name, discriminator in discriminators.items():
        # 실제 오디오 판별
        real_logits, real_fmaps = discriminator(real_audio)
        # 생성된 오디오 판별  
        fake_logits, fake_fmaps = discriminator(fake_audio)
        
        # 적대적 손실 계산
        disc_loss = compute_discriminator_loss(real_logits, fake_logits)
        gen_loss = compute_generator_loss(fake_logits)
        
        # 특징 맵 매칭 손실
        fm_loss = compute_feature_matching_loss(real_fmaps, fake_fmaps)
        
        total_loss += disc_loss + gen_loss + fm_loss
    
    return total_loss
```

### 특징 맵 매칭

각 판별자는 logits뿐만 아니라 중간 특징 맵도 반환하여 생성자가 더 세밀한 특징을 학습할 수 있도록 도움:

```python
# 각 레이어별 특징 맵 수집
for conv in self.convs:
    x = conv(x)
    x = self.activation(x)
    fmap.append(x)  # 중간 특징 저장
```

## 실제 성능 향상 분석

### 1. 다차원 품질 평가
- **MPD**: 리듬감과 주기적 일관성 향상
- **MSD**: 전체적인 파형 품질과 자연스러움
- **MS-STFT-D**: 주파수 성분의 정확성과 선명도

### 2. 상호 보완적 작용
```python
# 각 판별자가 포착하는 다른 측면들
mpd_focuses_on = ["rhythmic_patterns", "periodic_structures", "beat_consistency"]
msd_focuses_on = ["waveform_quality", "multi_scale_features", "global_structure"]  
msstftd_focuses_on = ["frequency_accuracy", "spectral_clarity", "harmonic_content"]
```

### 3. 적응적 학습
- **동적 균형**: 각 판별자의 성능에 따른 가중치 조절
- **단계적 학습**: 서로 다른 속도로 수렴하는 판별자들의 조화
- **안정성**: 다중 판별자를 통한 학습 안정성 향상

## 🔍 핵심 인사이트

### 1. 다면적 접근
- **시간 도메인**: MPD와 MSD의 파형 분석
- **주파수 도메인**: MS-STFT-D의 스펙트럼 분석
- **다중 스케일**: 각기 다른 해상도에서의 품질 평가

### 2. 특화된 설계
- **MPD**: 주기별 2D 재구성으로 리듬 패턴 포착
- **MSD**: 스케일별 다운샘플링으로 계층적 분석
- **MS-STFT-D**: 팽창 컨볼루션으로 다중 시간 스케일 분석

### 3. 효율적 구현
- **모듈화**: 공통 인터페이스를 통한 일관된 설계
- **메모리 효율성**: 최대 필터 수 제한과 적응적 크기 조절
- **계산 최적화**: 그룹 컨볼루션과 효율적인 다운샘플링

### 4. 견고한 학습
- **다양성**: 서로 다른 특징에 집중하는 다중 판별자
- **안정성**: 특징 맵 매칭을 통한 세밀한 피드백
- **적응성**: 동적 파라미터 조절로 다양한 오디오 타입 대응

## 🎯 결론

AudioCraft의 적대적 네트워크 시스템은 단일 판별자의 한계를 극복하고 다면적 품질 평가를 통해 고품질 오디오 생성을 실현합니다. MPD, MSD, MS-STFT-D의 협력을 통해 시간과 주파수 도메인에서 동시에 최적화되는 강력한 시스템을 구축했습니다.

다음 포스트에서는 이러한 AI 모델들을 통합하는 FastAPI 서버의 구현을 분석하며, REST API 설계와 모델 통합 전략을 살펴보겠습니다.

---

*이 분석은 AudioCraft Custom 프로젝트의 실제 소스 코드를 기반으로 작성되었습니다. 더 자세한 구현 내용은 [AudioCraft 공식 저장소](https://github.com/facebookresearch/audiocraft)에서 확인할 수 있습니다.*
---
layout: post
title: "Audiocraft Docker ë¹Œë“œ ?¤ì¹˜"
date: 2024-04-03 12:30:00 +0900
categories: [audiocraft]
tags: [docker, python, pip, audiocraft, module-not-found]
---

Audiocraftë¥?Docker ?˜ê²½?ì„œ ?¤í–‰?˜ê¸° ?„í•œ ë¹Œë“œ ?˜ê²½ ?¤ì • ë°©ë²•???ˆë‚´?´ë“œë¦¬ê² ?µë‹ˆ??

## ?¬ì „ ?”êµ¬?¬í•­

- Dockerê°€ ?¤ì¹˜?˜ì–´ ?ˆì–´???©ë‹ˆ??
- Git???¤ì¹˜?˜ì–´ ?ˆì–´???©ë‹ˆ??
- ìµœì†Œ 8GB ?´ìƒ??RAM???„ìš”?©ë‹ˆ??
- CUDA ì§€??GPUê°€ ê¶Œì¥?©ë‹ˆ??

## Docker ?´ë?ì§€ ë¹Œë“œ

1. Audiocraft ?€?¥ì†Œ ?´ë¡ :
```bash
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
```

2. Dockerfile ?ì„±:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# ?œìŠ¤???¨í‚¤ì§€ ?¤ì¹˜
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python ?¨í‚¤ì§€ ?¤ì¹˜
COPY requirements.txt .
RUN pip install -r requirements.txt

# Audiocraft ?¤ì¹˜
COPY . .

# ?‘ì—… ?”ë ‰? ë¦¬ ?¤ì •
WORKDIR /app

# ê¸°ë³¸ ?¬íŠ¸ ?¤ì •
EXPOSE 8000

# ?¤í–‰ ëª…ë ¹
CMD ["python", "app.py"]
```

3. Docker ?´ë?ì§€ ë¹Œë“œ:
```bash
docker build -t audiocraft:latest .
```

## Docker ì»¨í…Œ?´ë„ˆ ?¤í–‰

```bash
docker run -it --gpus all -p 8000:8000 audiocraft:latest
```

## ì£¼ì˜?¬í•­

1. GPU ?¬ìš©???„í•´?œëŠ” NVIDIA Container Toolkit???¤ì¹˜?˜ì–´ ?ˆì–´???©ë‹ˆ??
2. ë©”ëª¨ë¦??¬ìš©?‰ì´ ë§ìœ¼ë¯€ë¡?ì¶©ë¶„???œìŠ¤??ë¦¬ì†Œ?¤ê? ?„ìš”?©ë‹ˆ??
3. ì²??¤í–‰ ??ëª¨ë¸ ?¤ìš´ë¡œë“œë¡??¸í•´ ?œê°„??ê±¸ë¦´ ???ˆìŠµ?ˆë‹¤.

## ë¬¸ì œ ?´ê²°

### ?¼ë°˜?ì¸ ?¤ë¥˜

1. CUDA ê´€???¤ë¥˜:
   - NVIDIA ?œë¼?´ë²„ê°€ ìµœì‹  ë²„ì „?¸ì? ?•ì¸
   - Docker??GPU ì§€?ì´ ?œì„±?”ë˜???ˆëŠ”ì§€ ?•ì¸

2. ë©”ëª¨ë¦?ë¶€ì¡??¤ë¥˜:
   - Docker ì»¨í…Œ?´ë„ˆ??ë©”ëª¨ë¦??œí•œ???˜ë¦¬ê¸?
   - `--memory` ?µì…˜ ?¬ìš©

3. ëª¨ë“ˆ??ì°¾ì„ ???†ëŠ” ?¤ë¥˜:
   - requirements.txtê°€ ?¬ë°”ë¥´ê²Œ ?¤ì¹˜?˜ì—ˆ?”ì? ?•ì¸
   - Python ê²½ë¡œê°€ ?¬ë°”ë¥´ê²Œ ?¤ì •?˜ì–´ ?ˆëŠ”ì§€ ?•ì¸

## ì¶”ê? ?¤ì •

### ?˜ê²½ ë³€???¤ì •

```bash
docker run -it --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MODEL_PATH=/app/models \
  -p 8000:8000 \
  audiocraft:latest
```

### ë³¼ë¥¨ ë§ˆìš´??

```bash
docker run -it --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  -p 8000:8000 \
  audiocraft:latest
```

?´ë ‡ê²??¤ì •?˜ë©´ Audiocraftë¥?Docker ?˜ê²½?ì„œ ?ˆì •?ìœ¼ë¡??¤í–‰?????ˆìŠµ?ˆë‹¤. ?„ìš”??ê²½ìš° ì¶”ê??ì¸ ?¤ì •?´ë‚˜ ë¬¸ì œ ?´ê²° ë°©ë²•??ë¬¸ì˜??ì£¼ì„¸??


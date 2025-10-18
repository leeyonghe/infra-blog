---
layout: post
title: "Sora Docker Project: Running Open-Sora in a Container | Sora Docker ?„ë¡œ?íŠ¸: ì»¨í…Œ?´ë„ˆ?ì„œ Open-Sora ?¤í–‰?˜ê¸°"
date: 2024-03-23 12:00:00 +0900
categories: [Blog]
tags: [docker, open-sora, video-generation, ai, machine-learning]
---

# Sora Docker Project: Running Open-Sora in a Container | Sora Docker ?„ë¡œ?íŠ¸: ì»¨í…Œ?´ë„ˆ?ì„œ Open-Sora ?¤í–‰?˜ê¸°

![OpenAI Logo](https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg){: width="300" height="300"}

Open-Sora is an open-source initiative dedicated to efficiently producing high-quality video content. This project provides a Docker-based setup that makes it easy to run Open-Sora in a containerized environment.

Open-Sora??ê³ í’ˆì§?ë¹„ë””??ì½˜í…ì¸ ë? ?¨ìœ¨?ìœ¼ë¡??œì‘?˜ê¸° ?„í•œ ?¤í”ˆ?ŒìŠ¤ ?„ë¡œ?íŠ¸?…ë‹ˆ?? ???„ë¡œ?íŠ¸??Docker ê¸°ë°˜ ?¤ì •???œê³µ?˜ì—¬ Open-Soraë¥?ì»¨í…Œ?´ë„ˆ?”ëœ ?˜ê²½?ì„œ ?½ê²Œ ?¤í–‰?????ˆê²Œ ?©ë‹ˆ??

## What is Open-Sora? | Open-Sora?€?

Open-Sora is a powerful video generation model that can create high-quality videos from text descriptions. It supports various features including:

Open-Sora???ìŠ¤???¤ëª…?¼ë¡œë¶€??ê³ í’ˆì§?ë¹„ë””?¤ë? ?ì„±?????ˆëŠ” ê°•ë ¥??ë¹„ë””???ì„± ëª¨ë¸?…ë‹ˆ?? ?¤ìŒê³?ê°™ì? ?¤ì–‘??ê¸°ëŠ¥??ì§€?í•©?ˆë‹¤:

- Text-to-video generation | ?ìŠ¤????ë¹„ë””???ì„±
- Image-to-video generation | ?´ë?ì§€-??ë¹„ë””???ì„±
- Multiple resolution support (from 144p to 720p) | ?¤ì–‘???´ìƒ??ì§€??(144pë¶€??720pê¹Œì?)
- Variable video lengths (2s to 15s) | ?¤ì–‘??ë¹„ë””??ê¸¸ì´ (2ì´ˆë???15ì´ˆê¹Œì§€)
- Support for different aspect ratios | ?¤ì–‘???”ë©´ ë¹„ìœ¨ ì§€??

## Project Structure | ?„ë¡œ?íŠ¸ êµ¬ì¡°

The Sora Docker project includes several key components:

Sora Docker ?„ë¡œ?íŠ¸???¤ìŒê³?ê°™ì? ì£¼ìš” êµ¬ì„± ?”ì†Œë¥??¬í•¨?©ë‹ˆ??

- `Dockerfile`: Defines the container environment | ì»¨í…Œ?´ë„ˆ ?˜ê²½???•ì˜
- `docker-compose.yml`: Orchestrates the services | ?œë¹„??ì¡°ì •
- `requirements.txt`: Lists Python dependencies | Python ?˜ì¡´??ëª©ë¡
- `setup.py`: Package configuration | ?¨í‚¤ì§€ ?¤ì •
- Various configuration files and directories for the Open-Sora implementation | Open-Sora êµ¬í˜„???„í•œ ?¤ì–‘???¤ì • ?Œì¼ê³??”ë ‰? ë¦¬

## Key Features | ì£¼ìš” ê¸°ëŠ¥

1. **Containerized Environment**: The project is packaged in Docker containers, making it easy to deploy and run consistently across different environments.

   **ì»¨í…Œ?´ë„ˆ?”ëœ ?˜ê²½**: ?„ë¡œ?íŠ¸??Docker ì»¨í…Œ?´ë„ˆë¡??¨í‚¤ì§•ë˜???ˆì–´ ?¤ì–‘???˜ê²½?ì„œ ?¼ê??˜ê²Œ ë°°í¬?˜ê³  ?¤í–‰?˜ê¸° ?½ìŠµ?ˆë‹¤.

2. **Multiple Resolution Support**: The model can generate videos in various resolutions:
   
   **?¤ì–‘???´ìƒ??ì§€??*: ëª¨ë¸?€ ?¤ìŒê³?ê°™ì? ?¤ì–‘???´ìƒ?„ë¡œ ë¹„ë””?¤ë? ?ì„±?????ˆìŠµ?ˆë‹¤:
   - 256x256
   - 768x768
   - Custom aspect ratios (16:9, 9:16, 1:1, 2.39:1) | ?¬ìš©???•ì˜ ?”ë©´ ë¹„ìœ¨ (16:9, 9:16, 1:1, 2.39:1)

3. **Flexible Generation Options**:
   
   **? ì—°???ì„± ?µì…˜**:
   - Text-to-video generation | ?ìŠ¤????ë¹„ë””???ì„±
   - Image-to-video generation | ?´ë?ì§€-??ë¹„ë””???ì„±
   - Support for different video lengths | ?¤ì–‘??ë¹„ë””??ê¸¸ì´ ì§€??
   - Motion score control | ëª¨ì…˜ ?ìˆ˜ ?œì–´

## Getting Started | ?œì‘?˜ê¸°

To use the Sora Docker project:

Sora Docker ?„ë¡œ?íŠ¸ë¥??¬ìš©?˜ë ¤ë©?

1. Clone the repository | ?€?¥ì†Œ ?´ë¡ 
2. Build the Docker container | Docker ì»¨í…Œ?´ë„ˆ ë¹Œë“œ
3. Run the container with appropriate parameters | ?ì ˆ??ë§¤ê°œë³€?˜ë¡œ ì»¨í…Œ?´ë„ˆ ?¤í–‰
4. Generate videos using text prompts or reference images | ?ìŠ¤???„ë¡¬?„íŠ¸??ì°¸ì¡° ?´ë?ì§€ë¥??¬ìš©?˜ì—¬ ë¹„ë””???ì„±

## Example Usage | ?¬ìš© ?ˆì‹œ

Here's a basic example of generating a video:

ë¹„ë””???ì„±??ê¸°ë³¸ ?ˆì‹œ?…ë‹ˆ??

```bash
# Text-to-video generation | ?ìŠ¤????ë¹„ë””???ì„±
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_256px.py --save-dir samples --prompt "raining, sea"

# Image-to-video generation | ?´ë?ì§€-??ë¹„ë””???ì„±
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --cond_type i2v_head --prompt "Your prompt here" --ref path/to/image.png
```

## Advanced Features | ê³ ê¸‰ ê¸°ëŠ¥

The project includes several advanced features:

?„ë¡œ?íŠ¸???¤ìŒê³?ê°™ì? ê³ ê¸‰ ê¸°ëŠ¥???¬í•¨?©ë‹ˆ??

1. **Motion Score Control**: Adjust the motion intensity of generated videos | **ëª¨ì…˜ ?ìˆ˜ ?œì–´**: ?ì„±??ë¹„ë””?¤ì˜ ëª¨ì…˜ ê°•ë„ ì¡°ì •
2. **Multi-GPU Support**: Scale up generation with multiple GPUs | **?¤ì¤‘ GPU ì§€??*: ?¬ëŸ¬ GPUë¡??ì„± ?•ì¥
3. **Memory Optimization**: Options for memory-efficient generation | **ë©”ëª¨ë¦?ìµœì ??*: ë©”ëª¨ë¦??¨ìœ¨?ì¸ ?ì„±???„í•œ ?µì…˜
4. **Dynamic Motion Scoring**: Evaluate and adjust motion scores automatically | **?™ì  ëª¨ì…˜ ?ìˆ˜**: ëª¨ì…˜ ?ìˆ˜ë¥??ë™?¼ë¡œ ?‰ê??˜ê³  ì¡°ì •

## Conclusion | ê²°ë¡ 

The Sora Docker project makes it easy to run Open-Sora in a containerized environment, providing a powerful tool for video generation. Whether you're interested in text-to-video or image-to-video generation, this project offers a flexible and efficient solution.

Sora Docker ?„ë¡œ?íŠ¸??Open-Soraë¥?ì»¨í…Œ?´ë„ˆ?”ëœ ?˜ê²½?ì„œ ?½ê²Œ ?¤í–‰?????ˆê²Œ ?˜ì—¬, ê°•ë ¥??ë¹„ë””???ì„± ?„êµ¬ë¥??œê³µ?©ë‹ˆ?? ?ìŠ¤????ë¹„ë””?¤ë‚˜ ?´ë?ì§€-??ë¹„ë””???ì„±??ê´€?¬ì´ ?ˆë“ , ???„ë¡œ?íŠ¸??? ì—°?˜ê³  ?¨ìœ¨?ì¸ ?”ë£¨?˜ì„ ?œê³µ?©ë‹ˆ??

For more information and updates, visit the [Open-Sora GitHub repository](https://github.com/hpcaitech/Open-Sora).

??ë§ì? ?•ë³´?€ ?…ë°?´íŠ¸??[Open-Sora GitHub ?€?¥ì†Œ](https://github.com/hpcaitech/Open-Sora)ë¥?ë°©ë¬¸?˜ì„¸?? 
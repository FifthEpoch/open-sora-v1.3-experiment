# LoRA Experiment

## Overview

This directory contains the implementation of LoRA (Low-Rank Adaptation) for efficient test-time adaptation of Open-Sora v1.3 video generation models.

## Motivation

From our naive fine-tuning experiments, we found:
- Full fine-tuning **does improve** video continuation quality
- Best config: 100 steps @ 5e-5 (PSNR: 12.38, +3.48 vs baseline)
- **Problem**: Full fine-tuning is too slow (~16 min/video) for practical use

LoRA offers a solution:
- Train only ~0.1% of parameters (vs 100% in full fine-tuning)
- Much faster training and lower memory usage
- Prevents catastrophic forgetting
- Can be applied/removed at inference time

## Directory Structure

```
lora_experiment/
├── configs/           # LoRA configuration files
├── scripts/           # Training and inference scripts
├── results/           # Experiment outputs
└── README.md          # This file
```

## Expected Benefits

| Metric | Full Fine-tuning | LoRA (Expected) |
|--------|------------------|-----------------|
| Trainable params | ~1.1B (100%) | ~1-10M (<1%) |
| Training time | ~16 min/video | ~2-3 min/video |
| Memory usage | ~80GB | ~20-30GB |
| Quality | PSNR +3.48 | TBD |

## Implementation Plan

1. **Add LoRA layers** to STDiT3 attention modules
2. **Create LoRA training script** with per-video adaptation
3. **Benchmark** against full fine-tuning on same videos
4. **Optimize** rank and alpha hyperparameters


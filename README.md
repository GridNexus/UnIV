# Video-Image Unified Encoding Distillation

A unified image-video representation encoder based on heterogeneous knowledge distillation for visual understanding tasks.

## Project Overview

This project implements a unified encoder architecture that processes both images and video sequences.

## Project Structure

```
.
├── lib/                           # Core library
│   ├── teacher_modeling.py        # Teacher network (DINOv3 + SigLIP2)
│   ├── student_conv_only.py       # Student network (PureConvNext3D)
│   ├── loss.py                    # Distillation loss functions
│   ├── dataset_v3.py              # Dataset loading
│   ├── utils.py                   # Utility functions
│   │
│   ├── heads/                     # Multi-task detection heads
│   │   ├── projection.py          # Spatiotemporal compression
│   │   ├── detection_head.py      # Object detection head
│   │   ├── action_localization_head.py  # Action localization head
│   │   ├── anomaly_detection_head.py    # Anomaly detection head
│   │   └── __init__.py
│   │
│   └── multi_task_trainer.py      # Multi-task trainer
│
├── train_*.py                     # Distillation training scripts
├── train_multitask.py             # Multi-task training script
├── eval.py                        # Evaluation script
│
├── run/                           # Shell scripts
│   ├── eval.sh
│   └── ablation_*.sh
│
└── scripts_old/                   # Ablation experiment scripts
```

## Core Modules

### Stage 1: Heterogeneous Distillation

| Module | File | Description |
|--------|------|-------------|
| Teacher | `lib/teacher_modeling.py` | DINOv3 + SigLIP2 dual teachers, LayerNorm fusion |
| Student | `lib/student_conv_only.py` | PureConvNext3D full-convolution architecture |
| Distillation Loss | `lib/loss.py` | MSE + Cosine combined loss |

### Stage 2: Multi-Task Detection

| Module | File | Description |
|--------|------|-------------|
| Spatiotemporal | `lib/heads/projection.py` | ViT attention compression, outputs Z_seq |
| Object Detection | `lib/heads/detection_head.py` | RT-DETR style, end-to-end detection |
| Action Localization | `lib/heads/action_localization_head.py` | CSCAN mechanism, temporal boundary regression |
| Anomaly Detection | `lib/heads/anomaly_detection_head.py` | MIL weakly supervised, segment-level classification |
| Trainer | `lib/multi_task_trainer.py` | Two-stage optimization, uncertainty weighting |

## Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.9+
- Transformers
- SwanLab (experiment tracking)

## Training Pipeline

### Stage 1: Heterogeneous Distillation

```bash
# Single GPU training
python train_conv_combined.py

# Distributed training
torchrun --nproc_per_node=8 train_conv_combined.py
```

**Environment variables:**
```bash
export DINOV3_PATH="/path/to/dinov3-vitb16-pretrain-lvd1689m"
export SIGLIP2_PATH="/path/to/siglip2-base-patch16-512"
export TRAIN_DATA_PATH="/path/to/train_sample.txt"
export VAL_DATA_PATH="/path/to/val_sample.txt"
export SAVE_PATH="./checkpoints/distillation"
```

### Stage 2: Multi-Task Joint Training

```bash
# Multi-task training
python train_multitask.py

# Resume training
python train_multitask.py --resume ./checkpoints/multitask/checkpoint_best.pth
```

**Environment variables:**
```bash
export PRETRAINED_ENCODER="./checkpoints/distillation/checkpoint_best.pth"
export TRAIN_DATA_PATH="/path/to/train_data.json"
export VAL_DATA_PATH="/path/to/val_data.json"
export SWANLAB_PROJECT="Video-Multitask-Detection"
```

## Detection Task Configuration

| Task | Detection Head | Classes | Output |
|------|----------------|---------|--------|
| Object Detection | DetectionHead | 4 (person/pole/rod/grounding_rod) | boxes, scores |
| Action Localization | ActionLocalizationHead | 5 (climb/ground_test/attach/remove/descend) | action_scores, boundaries |
| Anomaly Detection | AnomalyDetectionHead | 2 (normal/anomaly) | segment_scores, video_scores |

## Multi-Task Loss Function

```
L_total = ω1*L_det + ω2*L_tad + ω3*L_ano

Where ωi is automatically learned via homoscedastic uncertainty:
L_i_weighted = exp(-log_var_i) * L_i + log_var_i
```

## Two-Stage Optimization Strategy

1. **First 5 Epochs (Frozen Stage)**
   - Encoder parameters frozen
   - Only train projection layer + detection heads
   - Learning rate: 5e-4

2. **Subsequent Epochs (Fine-tuning Stage)**
   - Unfreeze encoder
   - Full parameter fine-tuning
   - Learning rate: encoder 1e-4, heads 5e-4
   - Cosine annealing scheduler

## Training Scripts

### `train_multitask.py`

Multi-task joint training main script, includes:
- Pretrained encoder loading
- Multi-task detector initialization
- Two-stage training loop
- Distributed training support
- SwanLab logging

### `lib/multi_task_trainer.py`

`MultitaskDetector` class integrates:
- PureConvNext3D encoder
- SpatiotemporalCompressor module
- Three detection heads
- Uncertainty-weighted log_vars

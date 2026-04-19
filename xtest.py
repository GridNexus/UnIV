import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import swanlab
import os
from tqdm import tqdm
import warnings
import math  # 新增，用于余弦退火

from lib.teacher_modeling import TeacherEnsemble
from lib.student_modeling import VisualFlowEncoder, SpatiotemporalFeatureEncoder
from lib.student_modeling_v2 import SpatiotemporalFeatureEncoderV2
from lib.dataset_v3 import VideoDynamicDataset
from lib.loss import DecoupledDistillationLoss
from lib.evaluation import validate_one_epoch
from lib.utils import setup_ddp, cleanup_ddp, model_summary_to_string, save_model_summary_to_txt
from lib.diagnostic import FeatureDiagnostic

warnings.filterwarnings("ignore")


# ================= 配置区域 =================
DINOV3_PATH = os.getenv("DINOV3_PATH", "/path/to/dinov3-vitb16-pretrain-lvd1689m")
SIGLIP2_PATH = os.getenv("SIGLIP2_PATH", "/path/to/siglip2-base-patch16-512")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/path/to/train_sample.txt")
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "/path/to/val_sample.txt")
DATASET_SAMPLE_RATE = 0.1

RESUME_PATH = None
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "Video-Distillation")
SWANLAB_RUN_NAME = f"dinov3_siglip2_stvit2_debug"
SAVE_PATH = os.getenv("SAVE_PATH", f"./checkpoints/{SWANLAB_PROJECT}/{SWANLAB_RUN_NAME}")
os.makedirs(SAVE_PATH, exist_ok=True)

# ================= 关键改动：梯度累积配置 =================
BATCH_SIZE = 2            # 单卡batch size
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 有效batch size = 8

CLIPS_PER_VIDEO = 2      
EPOCHS = 300
NUM_FRAMES = 8           
# ================= 关键改动：学习率按有效batch size缩放 =================
BASE_LR = 3e-4  # 基准学习率（对应batch size 8）
# 线性缩放：lr ∝ batch_size，如果batch_size=8时lr=3e-4，那么有效batch=8时保持3e-4
# 如果之前batch=2时lr=3e-4，现在有效batch=8，应该增大到 1.2e-3
# 但蒸馏任务通常需要较小lr，建议保守设置
LEARNING_RATE = 6e-4  # 保持或略微增大，见下方建议
NUM_WORKERS = 8
LOSS_TYPE = ["mse", "cosine"]
MODEL_TYPE = "SpatiotemporalFeatureEncoderV2"
RANDOM_SEED = 0

# ================= 关键改动：学习率调度配置 =================
USE_COSINE_SCHEDULER = True  # 视觉编码器蒸馏建议使用余弦退火
WARMUP_EPOCHS = 5  # 预热epoch数
MIN_LR_RATIO = 0.1  # 最小学习率 = MAX_LR * 0.1

# save settings
all_settings = {
    "dinov3_path": DINOV3_PATH, "siglip2_path": SIGLIP2_PATH, "train_data_path": TRAIN_DATA_PATH,
    "val_data_path": VAL_DATA_PATH, "dataset_sample_rate": DATASET_SAMPLE_RATE,
    "resume_path": RESUME_PATH, "swanlab_project": SWANLAB_PROJECT, 
    "swanlab_run_name": SWANLAB_RUN_NAME, "save_path": SAVE_PATH,
    "batch_size": BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "clips_per_video": CLIPS_PER_VIDEO, "epochs": EPOCHS,
    "num_frames": NUM_FRAMES, "learning_rate": LEARNING_RATE, "num_workers": NUM_WORKERS,
    "loss_type": LOSS_TYPE, "random_seed": RANDOM_SEED,
    "use_cosine_scheduler": USE_COSINE_SCHEDULER, "warmup_epochs": WARMUP_EPOCHS
}
with open(os.path.join(SAVE_PATH, "config.txt"), 'w') as f:
    for k, v in all_settings.items():
        f.write(f"{k}: {v}\n")


import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    setup_ddp()
    set_seed(RANDOM_SEED)
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    
    # 计算总步数（用于余弦退火）
    # 注意：需要在DataLoader创建后计算，这里先估算
    # 实际在训练循环中使用 len(train_dataloader)

    # 1. 准备 Teacher
    if rank == 0: 
        print(f"Loading Teachers (DINOv3 + SigLIP2)...")
        print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
        print(f"Effective Batch Size per GPU: {EFFECTIVE_BATCH_SIZE}")
        print(f"World Size: {dist.get_world_size()}, Total Effective Batch: {EFFECTIVE_BATCH_SIZE * dist.get_world_size()}")
    
    teachers = TeacherEnsemble(DINOV3_PATH, SIGLIP2_PATH, align_type="layernorm").to(device).to(torch.bfloat16)
    teachers.eval()
    analyzer = FeatureDiagnostic(save_dir="./feature_analysis/layernorm")


    # 2. 数据集
    train_dataset = VideoDynamicDataset(
        TRAIN_DATA_PATH, 
        dinov3_path=DINOV3_PATH,
        siglip2_path=SIGLIP2_PATH,
        num_frames=NUM_FRAMES, 
        clips_per_video=CLIPS_PER_VIDEO,
        dataset_sample_rate=DATASET_SAMPLE_RATE,
        seed=RANDOM_SEED,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # 3. 学生网络
    if MODEL_TYPE == "VisualFlowEncoder":
        student = VisualFlowEncoder(hidden_dim=2048, output_dim=1536, num_frames=NUM_FRAMES)
    elif MODEL_TYPE == "SpatialTemporalEncoder":
        student = SpatiotemporalFeatureEncoder(latent_channels=2048, output_channels=1536, num_frames=NUM_FRAMES)
    elif MODEL_TYPE == "SpatiotemporalFeatureEncoderV2":
        student = SpatiotemporalFeatureEncoderV2(latent_channels=2048, output_channels=1536, num_frames=NUM_FRAMES)
    student = student.to(device)
    
    # 模型摘要（使用小batch避免OOM）
    if rank == 0:
        with torch.no_grad():
            model_summary = model_summary_to_string(
                student, 
                torch.randn(2, 3, NUM_FRAMES, 512, 512).to(device)
            )
            print("Student Model Summary:")
            print(model_summary)
            save_model_summary_to_txt(model_summary, os.path.join(SAVE_PATH, "model_summary.txt"))
    # quit()
    student = DDP(student, device_ids=[local_rank])

    # ... 之前的初始化代码 ...

    # 取一个 batch 进行分析
    batch_data = next(iter(train_dataloader))
    
    dino_in = batch_data['dino'].to(device, non_blocking=True).to(torch.bfloat16)
    siglip_in = batch_data['siglip'].to(device, non_blocking=True).to(torch.bfloat16)
    B, K, T, C, H, W = dino_in.shape
    dino_in = dino_in.view(B*K, T, C, H, W)
    siglip_in = siglip_in.view(B*K, T, C, H, W)

    # 模拟 TeacherEnsemble 内部逻辑
    with torch.no_grad():
        # 注意：这里要分别提取 LayerNorm 前后的特征进行对比
        target = teachers(dino_in, siglip_in)  # (B*K, T, 32, 32, 1536)
        
        B, T, C, H, W = target.shape
        out_dino = target.view(B*T, C, H, W)[:, :768, :, :] # (B, C ,H, W)
        out_siglip = target.view(B*T, C, H, W)[:, 768:, :, :]
        
        out_dino = out_dino.permute(0, 2, 3, 1).view(B*T, H*W, 768) # (B*T, H*W, C_dino)
        out_siglip = out_siglip.permute(0, 2, 3, 1).view(B*T, H*W, 768) # (B*T, H*W, C_siglip)
        analyzer.collect(out_dino, out_siglip)
        # analyzer.visualize_spatial_heatmaps(out_dino, out_siglip)
        analyzer.run_analysis()
        # 收集统计信息


    # 生成报告
    # analyzer.run_analysis()

if __name__ == "__main__":
    main()
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
from lib.student_conv_only import PureConvNext3DModel
from lib.dataset_v3 import VideoDynamicDataset
from lib.loss import DecoupledDistillationLoss
from lib.evaluation import validate_one_epoch
from lib.utils import setup_ddp, cleanup_ddp, model_summary_to_string, save_model_summary_to_txt


warnings.filterwarnings("ignore")


# ================= 配置区域 =================
DINOV3_PATH = os.getenv("DINOV3_PATH", "/path/to/dinov3-vitb16-pretrain-lvd1689m")
SIGLIP2_PATH = os.getenv("SIGLIP2_PATH", "/path/to/siglip2-base-patch16-512")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/path/to/train_sample.txt")
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "/path/to/val_sample.txt")
DATASET_SAMPLE_RATE = 0.1

RESUME_PATH = os.getenv("RESUME_PATH", None)
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "Video-Distillation")
SWANLAB_RUN_NAME = f"conv_loss_localstats"
SAVE_PATH = os.getenv("SAVE_PATH", f"./checkpoints/{SWANLAB_PROJECT}/{SWANLAB_RUN_NAME}")
os.makedirs(SAVE_PATH, exist_ok=True)

# ================= 关键改动：梯度累积配置 =================
BATCH_SIZE = 2            # 单卡batch size
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 有效batch size = 8

CLIPS_PER_VIDEO = 2      
EPOCHS = 100
NUM_FRAMES = 8           
# ================= 关键改动：学习率按有效batch size缩放 =================
BASE_LR = 3e-4  # 基准学习率（对应batch size 8）
# 线性缩放：lr ∝ batch_size，如果batch_size=8时lr=3e-4，那么有效batch=8时保持3e-4
# 如果之前batch=2时lr=3e-4，现在有效batch=8，应该增大到 1.2e-3
# 但蒸馏任务通常需要较小lr，建议保守设置
LEARNING_RATE = 6e-4  # 保持或略微增大，见下方建议
NUM_WORKERS = 8
LOSS_TYPE = ["mse", "cosine", "local_stats"]
MODEL_TYPE = "PureConvNext3DModel"
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
    
    teachers = TeacherEnsemble(DINOV3_PATH, SIGLIP2_PATH, "layernorm").to(device).to(torch.bfloat16)
    teachers.eval()

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

    if rank == 0: 
        print("Loading Validation Dataset...")
    
    val_dataset = VideoDynamicDataset(
        VAL_DATA_PATH,
        dinov3_path=DINOV3_PATH,
        siglip2_path=SIGLIP2_PATH,
        num_frames=NUM_FRAMES,
        clips_per_video=CLIPS_PER_VIDEO,
        dataset_sample_rate=DATASET_SAMPLE_RATE,
        seed=RANDOM_SEED
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # 验证可以用更大batch（不需要梯度）
        sampler=val_sampler,
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
    elif MODEL_TYPE == "PureConvNext3DModel":
        student = PureConvNext3DModel(latent_channels=2048, output_channels=1536, num_frames=NUM_FRAMES)
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
    
    student = DDP(student, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
    loss_fn = DecoupledDistillationLoss(
        loss_types=LOSS_TYPE, 
        split_dim=768, 
        channel_dim=2, 
        dino_weight=1.0, 
        siglip_weight=1.0
    ).to(device)

    # ================= 关键改动：学习率调度器 =================
    num_training_steps = len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS * EPOCHS
    warmup_steps = len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS * WARMUP_EPOCHS
    
    if USE_COSINE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=num_training_steps - warmup_steps,  # 第一个周期长度（不含warmup）
            T_mult=1,
            eta_min=LEARNING_RATE * MIN_LR_RATIO
        )
        # 或者使用更简单的CosineAnnealingLR
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=num_training_steps,
        #     eta_min=LEARNING_RATE * MIN_LR_RATIO
        # )
    else:
        scheduler = None

    # SwanLab 初始化
    if rank == 0:
        swanlab.init(
            project=SWANLAB_PROJECT,
            name=SWANLAB_RUN_NAME,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE * dist.get_world_size(),
                "lr": LEARNING_RATE,
                "model": MODEL_TYPE,
                "resume": RESUME_PATH,
                "scheduler": "cosine" if USE_COSINE_SCHEDULER else "constant",
                "warmup_epochs": WARMUP_EPOCHS if USE_COSINE_SCHEDULER else 0
            }
        )

    # 断点续训
    start_epoch = 0
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        student.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")
    else:
        if rank == 0 and RESUME_PATH:
            print(f"Warning: Resume path {RESUME_PATH} not found, starting from scratch.")

    if rank == 0:
        print(f"Start Training: {EPOCHS} epochs, now at epoch {start_epoch}.")
        print(f"Total optimization steps: {num_training_steps}, Warmup steps: {warmup_steps}")

    best_decision_metric = float('inf')
    global_step = 0  # 优化器步骤计数（考虑梯度累积）
    
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)
        student.train()
        
        iterator = tqdm(train_dataloader, desc=f"Ep {epoch+1}", disable=(rank != 0))
        step_loss_sum = 0.0
        accumulated_loss = 0.0  # 累积损失
        steps = 0
        optimizer.zero_grad()  # 每个epoch开始时清零
        
        for batch_idx, batch_data in enumerate(iterator):
            dino_in = batch_data['dino'].to(device, non_blocking=True).to(torch.bfloat16)
            siglip_in = batch_data['siglip'].to(device, non_blocking=True).to(torch.bfloat16)
            student_in = batch_data['student'].to(device, non_blocking=True).to(torch.bfloat16)
            
            # Flatten B and K
            B, K, T, C, H, W = dino_in.shape
            dino_in = dino_in.view(B*K, T, C, H, W)
            siglip_in = siglip_in.view(B*K, T, C, H, W)
            student_in = student_in.view(B*K, T, C, H, W)
            student_in = student_in.permute(0, 2, 1, 3, 4)  # → (BK, 3, T, 512, 512)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Teacher Inference
                with torch.no_grad():
                    teacher_target = teachers(dino_in, siglip_in)  # (B*K, T, 32, 32, 1536)

                # Student Forward
                features = student(student_in) # B*K, T, 1536, 32, 32
                
                # ================= 关键改动：损失除以累积步数 =================
                loss = loss_fn(features, teacher_target) / GRADIENT_ACCUMULATION_STEPS

            # ================= 关键改动：梯度累积 =================
            loss.backward()
            
            accumulated_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # 还原真实损失用于记录
            
            # 每N步更新一次参数
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_dataloader):
                
                # 梯度裁剪（可选但推荐）
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 学习率调度（按优化器步骤）
                if scheduler is not None:
                    # Warmup处理
                    if global_step < warmup_steps:
                        lr_scale = min(1.0, float(global_step + 1) / float(warmup_steps))
                        for pg in optimizer.param_groups:
                            pg['lr'] = LEARNING_RATE * lr_scale
                    else:
                        scheduler.step()
                
                # 记录和日志
                reduced_loss = torch.tensor(accumulated_loss / GRADIENT_ACCUMULATION_STEPS, device=device)
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
                curr_loss = reduced_loss.item()
                step_loss_sum += curr_loss
                steps += 1
                global_step += 1
                
                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    swanlab.log({
                        "train/step_loss": curr_loss, 
                        "train/lr": current_lr, 
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step
                    })
                    iterator.set_postfix({
                        "loss": f"{curr_loss:.6f}", 
                        "lr": f"{current_lr:.2e}"
                    })
                
                accumulated_loss = 0.0  # 重置累积损失

        # Epoch结束处理（如果最后几个batch不够一个accumulation step）
        # 上面的逻辑已经处理了边界情况

        torch.cuda.empty_cache()
        
        # 验证
        val_metrics, decision_metric = validate_one_epoch(
            student, teachers, val_dataloader, device, split_dim=768
        )

        if rank == 0:
            avg_train_loss = step_loss_sum / max(steps, 1)
            swanlab.log({"train/avg_train_loss": avg_train_loss})
            
            print(f"Validation Ep {epoch+1}: val_metrics: {val_metrics}, decision_metric: {decision_metric:.6f}")
            
            swanlab_log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
            swanlab_log_dict["val/decision_metric"] = decision_metric
            swanlab.log(swanlab_log_dict)
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': student.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'avg_train_loss': avg_train_loss,
                'val_metrics': val_metrics,
                'decision_metric': decision_metric,
                'global_step': global_step
            }
            
            if decision_metric < best_decision_metric:
                best_decision_metric = decision_metric
                torch.save(save_dict, os.path.join(SAVE_PATH, "checkpoint_best.pth"))
                print(f"New best model saved! decision_metric: {decision_metric:.6f}")
            
            torch.save(save_dict, os.path.join(SAVE_PATH, "checkpoint_latest.pth"))
            print(f"Epoch {epoch+1} Done. Avg Loss: {avg_train_loss:.6f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
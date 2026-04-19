import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import swanlab # 新增
import os
from tqdm import tqdm
import warnings

from lib.teacher_modeling import TeacherEnsemble
from lib.student_modeling import VisualFlowEncoder, SpatiotemporalFeatureEncoder
from lib.dataset_v3 import VideoDynamicDataset
from lib.loss import DecoupledDistillationLoss
from lib.evaluation import validate_one_epoch
from lib.utils import setup_ddp, cleanup_ddp, model_summary_to_string, save_model_summary_to_txt


# 忽略无关警告
warnings.filterwarnings("ignore")


# ================= 配置区域 =================
DINOV3_PATH = os.getenv("DINOV3_PATH", "/path/to/dinov3-vitb16-pretrain-lvd1689m")
SIGLIP2_PATH = os.getenv("SIGLIP2_PATH", "/path/to/siglip2-base-patch16-512")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/path/to/train_sample.txt")
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "/path/to/val_sample.txt")
DATASET_SAMPLE_RATE = 0.1

RESUME_PATH = os.getenv("RESUME_PATH", None)
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "Video-Distillation")
SWANLAB_RUN_NAME = f"dinov3_siglip2_stvit"
SAVE_PATH = os.getenv("SAVE_PATH", f"./checkpoints/{SWANLAB_PROJECT}/{SWANLAB_RUN_NAME}")
os.makedirs(SAVE_PATH, exist_ok=True)

# 训练超参
BATCH_SIZE = 2           
CLIPS_PER_VIDEO = 3      
EPOCHS = 100
NUM_FRAMES = 8           
LEARNING_RATE = 3e-4
NUM_WORKERS = 8
LOSS_TYPE = ["mse", "cosine"] # 目前仅支持 MSE Loss
MODEL_TYPE = "SpatialTemporalEncoder" # 可选 "VisualFlowEncoder" 或 "SpatialTemporalEncoder"
# MODEL_TYPE = "VisualFlowEncoder"
RANDOM_SEED = 0

# save settings
all_settings = {
    # path
    "dinov3_path": DINOV3_PATH, "siglip2_path": SIGLIP2_PATH, "train_data_path": TRAIN_DATA_PATH,
    "val_data_path": VAL_DATA_PATH, "dataset_sample_rate": DATASET_SAMPLE_RATE,
    "resume_path": RESUME_PATH, "swanlab_project": SWANLAB_PROJECT, "swanlab_run_name": SWANLAB_RUN_NAME, "save_path": SAVE_PATH,
    # hyperparameters
    "batch_size": BATCH_SIZE,   "clips_per_video": CLIPS_PER_VIDEO, "epochs": EPOCHS,
    "num_frames": NUM_FRAMES,   "learning_rate": LEARNING_RATE,     "num_workers": NUM_WORKERS,
    "loss_type": LOSS_TYPE,     "random_seed": RANDOM_SEED
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
    # 必须尽早设置，确保后续 Dataloader 行为一致
    set_seed(RANDOM_SEED) 
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")

    # 1. 准备 Teacher
    if rank == 0: print("Loading Teachers (DINOv3 + SigLIP2)...")
    teachers = TeacherEnsemble(DINOV3_PATH, SIGLIP2_PATH).to(device).to(torch.bfloat16)
    teachers.eval()

    # 2. 数据集 (传入模型路径以初始化 Processors)
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

    if rank == 0: print("Loading Validation Dataset...")
    val_dataset = VideoDynamicDataset(
        VAL_DATA_PATH, # 传入验证列表
        dinov3_path=DINOV3_PATH,
        siglip2_path=SIGLIP2_PATH,
        num_frames=NUM_FRAMES,
        clips_per_video=CLIPS_PER_VIDEO, # 验证时通常也可以切片，或者设为1
        dataset_sample_rate=DATASET_SAMPLE_RATE,
        seed=RANDOM_SEED
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, # 验证 batch size 可以稍微大一点，如果显存允许
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 3. 学生网络
    if MODEL_TYPE == "VisualFlowEncoder":
        student = VisualFlowEncoder(hidden_dim=2048, output_dim=1536, num_frames=NUM_FRAMES)
    elif MODEL_TYPE == "SpatialTemporalEncoder":
        student = SpatiotemporalFeatureEncoder(latent_channels=2048, output_channels=1536, num_frames=NUM_FRAMES)
    
    student = student.to(device)
    model_summary = model_summary_to_string(student, torch.randn(16, 3, NUM_FRAMES, 512, 512).to(device))
    if rank == 0:
        print("Student Model Summary:")
        print(model_summary)
        save_model_summary_to_txt(model_summary, os.path.join(SAVE_PATH, "model_summary.txt"))
    student = DDP(student, device_ids=[local_rank])

    

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
    loss_fn = DecoupledDistillationLoss(loss_types=LOSS_TYPE, split_dim=768, channel_dim=2, dino_weight=1.0, siglip_weight=1.0).to(device)

    # ================= 关键改动 1: SwanLab 初始化 (仅 Rank 0) =================
    if rank == 0:
        swanlab.init(
            project=SWANLAB_PROJECT,
            name=SWANLAB_RUN_NAME,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "model": MODEL_TYPE,
                "resume": RESUME_PATH
            }
        )

    # ================= 关键改动 2: 断点续训逻辑 =================
    start_epoch = 0
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        student.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")
    else:
        if rank == 0 and RESUME_PATH:
            print(f"Warning: Resume path {RESUME_PATH} not found, starting from scratch.")

    if rank == 0:
        print(f"Start Training: {EPOCHS} epochs, now at epoch {start_epoch}.")

    best_decision_metric = float('inf') # 假设越小越好，具体取决于你的决策指标
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)
        student.train()
        
        iterator = tqdm(train_dataloader, desc=f"Ep {epoch+1}", disable=(rank != 0))
        step_loss_sum = 0.0
        steps = 0
        
        for batch_data in iterator:
            # batch_data 是一个 dict: {'dino': ..., 'siglip': ..., 'student': ...}
            # 形状: (B, K, T, 3, 512, 512)
            
            dino_in = batch_data['dino'].to(device, non_blocking=True).to(torch.bfloat16)
            siglip_in = batch_data['siglip'].to(device, non_blocking=True).to(torch.bfloat16)
            student_in = batch_data['student'].to(device, non_blocking=True).to(torch.bfloat16)
            
            # Flatten B and K -> (B*K, T, 3, 512, 512)
            B, K, T, C, H, W = dino_in.shape
            
            dino_in = dino_in.view(B*K, T, C, H, W)
            siglip_in = siglip_in.view(B*K, T, C, H, W)
            student_in = student_in.view(B*K, T, C, H, W)
            student_in = student_in.permute(0, 2, 1, 3, 4)  # → (BK, 3, T, 512, 512)
            
            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 1. Teacher Inference (直接使用处理好的 Tensor)
                with torch.no_grad():
                    teacher_target = teachers(dino_in, siglip_in) # (B*K, T, 1536, 32, 32)
                    target_permuted = teacher_target.permute(0, 1, 4, 2, 3)

                # 2. Student Forward
                features = student(student_in)  # 只用 features 做蒸馏
                loss = loss_fn(features, target_permuted)

            loss.backward()
            optimizer.step()

            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
            curr_loss = reduced_loss.item()
            step_loss_sum += curr_loss
            steps += 1
            
            # Logging
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                swanlab.log({"train/step_loss": curr_loss, "train/lr": current_lr, "train/epoch": epoch + 1})
                iterator.set_postfix({"loss": f"{curr_loss:.6f}"})

        # 释放显存
        torch.cuda.empty_cache()
        
        # 运行验证
        val_metrics, decision_metric = validate_one_epoch(student, teachers, val_dataloader, device, split_dim=768)

        if rank == 0:
            # 记录 Epoch 级别的 Loss
            avg_train_loss = step_loss_sum / len(train_dataloader)
            swanlab.log({"train/avg_train_loss": avg_train_loss})

            print(f"Validation Ep {epoch+1}: val_metrics: {val_metrics}, decision_metric: {decision_metric:.6f}")
            # 记录到 SwanLab
            swanlab_log_dict = {}
            for k, v in val_metrics.items():
                swanlab_log_dict[f"val/{k}"] = v
            swanlab_log_dict["val/decision_metric"] = decision_metric
            swanlab.log(swanlab_log_dict)
            
            # 保存最佳模型和最新模型
            save_dict = {
                'epoch': epoch,
                'model_state_dict': student.module.state_dict(), # 取 module 存，方便非 DDP 加载
                'optimizer_state_dict': optimizer.state_dict(), 
                'avg_train_loss': avg_train_loss,
                'val_metrics': val_metrics, # 记录验证指标
                'decision_metric': decision_metric
            }
 
            
            # 保存 latest 和 best
            if decision_metric < best_decision_metric:
                best_decision_metric = decision_metric
                torch.save(save_dict, os.path.join(SAVE_PATH, f"checkpoint_best.pth"))
            torch.save(save_dict, os.path.join(SAVE_PATH, f"checkpoint_latest.pth"))
            
            print(f"Epoch {epoch+1} Done. Avg Loss: {decision_metric:.6f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
from tqdm import tqdm
import argparse
import warnings

# 复用你的库文件
from lib.teacher_modeling import TeacherEnsemble
from lib.student_modeling import VisualFlowEncoder, SpatiotemporalFeatureEncoder
from lib.student_conv_only import PureConvNext3DModel
from lib.dataset_v3 import VideoDynamicDataset
from lib.utils import setup_ddp, cleanup_ddp

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 默认配置，可以通过命令行参数覆盖
DEFAULT_CHECKPOINT = os.getenv("CHECKPOINT_PATH", "/path/to/checkpoint_best.pth")

# 路径配置 (需与训练保持一致)
DINOV3_PATH = os.getenv("DINOV3_PATH", "/path/to/dinov3-vitb16-pretrain-lvd1689m")
SIGLIP2_PATH = os.getenv("SIGLIP2_PATH", "/path/to/siglip2-base-patch16-512")
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "/path/to/val_sample.txt")

# 模型参数
MODEL_TYPE = "PureConvNext3DModel"
NUM_FRAMES = 8
CLIPS_PER_VIDEO = 2
BATCH_SIZE = 8  # 测试时Batch Size可以比训练大，因为不需要存梯度
NUM_WORKERS = 8
SPLIT_DIM = 768 # 特征分割维度 (前768是DINO, 后768是SigLIP)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Video Distillation")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of data workers')
    return parser.parse_args()

def calculate_metrics(student_out, teacher_out, split_dim):
    """
    计算 MSE 和 Cosine Similarity，支持整体和分段计算
    """
    # 1. 整体指标
    mse_total = F.mse_loss(student_out, teacher_out).item()
    
    # Cosine 需要把特征展平: (B, T, C, H, W) -> (N, C)
    B, T, C, H, W = student_out.shape
    s_flat = student_out.permute(0, 1, 3, 4, 2).reshape(-1, C).float()
    t_flat = teacher_out.permute(0, 1, 3, 4, 2).reshape(-1, C).float()
    
    cos_sim_total = F.cosine_similarity(s_flat, t_flat, dim=1).mean().item()

    # 2. 分段指标 (Part 1: DINO, Part 2: SigLIP)
    s_p1, s_p2 = torch.split(student_out, split_dim, dim=2)
    t_p1, t_p2 = torch.split(teacher_out, split_dim, dim=2)
    
    mse_p1 = F.mse_loss(s_p1, t_p1).item()
    mse_p2 = F.mse_loss(s_p2, t_p2).item()
    
    # Part 1 Cosine
    s_p1_flat = s_p1.permute(0, 1, 3, 4, 2).reshape(-1, split_dim).float()
    t_p1_flat = t_p1.permute(0, 1, 3, 4, 2).reshape(-1, split_dim).float()
    cos_p1 = F.cosine_similarity(s_p1_flat, t_p1_flat, dim=1).mean().item()
    
    # Part 2 Cosine
    s_p2_flat = s_p2.permute(0, 1, 3, 4, 2).reshape(-1, split_dim).float()
    t_p2_flat = t_p2.permute(0, 1, 3, 4, 2).reshape(-1, split_dim).float()
    cos_p2 = F.cosine_similarity(s_p2_flat, t_p2_flat, dim=1).mean().item()

    return {
        "mse_total": mse_total,
        "cos_total": cos_sim_total,
        "mse_dino": mse_p1,
        "cos_dino": cos_p1,
        "mse_siglip": mse_p2,
        "cos_siglip": cos_p2
    }

def main():
    args = parse_args()
    
    # DDP 初始化
    setup_ddp()
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"==========================================")
        print(f"Testing Model: {MODEL_TYPE}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Batch Size: {args.batch_size} (Per GPU)")
        print(f"==========================================")

    # 1. 准备 Teacher (生成 Ground Truth)
    if rank == 0: print("Loading Teachers...")
    teachers = TeacherEnsemble(DINOV3_PATH, SIGLIP2_PATH, "layernorm").to(device).to(torch.bfloat16)
    teachers.eval()

    # 2. 准备 Student 模型
    if MODEL_TYPE == "VisualFlowEncoder":
        student = VisualFlowEncoder(hidden_dim=2048, output_dim=1536, num_frames=NUM_FRAMES)
    elif MODEL_TYPE == "SpatialTemporalEncoder":
        student = SpatiotemporalFeatureEncoder(latent_channels=2048, output_channels=1536, num_frames=NUM_FRAMES)
    elif MODEL_TYPE == "PureConvNext3DModel":
        student = PureConvNext3DModel(latent_channels=2048, output_channels=1536, num_frames=NUM_FRAMES)
    
    student = student.to(device)

    # 加载权重
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # 处理 DDP 保存时多出来的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        student.load_state_dict(new_state_dict)
        if rank == 0:
            print(f"Loaded weights from epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    # 封装 DDP (即使是验证，为了保持一致性通常也用DDP，或者仅使用单卡)
    student = DDP(student, device_ids=[local_rank])
    student.eval()

    # 3. 数据集
    val_dataset = VideoDynamicDataset(
        VAL_DATA_PATH,
        dinov3_path=DINOV3_PATH,
        siglip2_path=SIGLIP2_PATH,
        num_frames=NUM_FRAMES,
        clips_per_video=CLIPS_PER_VIDEO,
        dataset_sample_rate=0.1, # 测试时通常用全量数据
        seed=0
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 4. 评估循环
    metrics_sum = {
        "mse_total": 0.0, "cos_total": 0.0,
        "mse_dino": 0.0, "cos_dino": 0.0,
        "mse_siglip": 0.0, "cos_siglip": 0.0
    }
    total_steps = 0

    if rank == 0:
        pbar = tqdm(total=len(val_dataloader), desc="Evaluating")

    with torch.no_grad():
        for batch_data in val_dataloader:
            dino_in = batch_data['dino'].to(device, non_blocking=True).to(torch.bfloat16)
            siglip_in = batch_data['siglip'].to(device, non_blocking=True).to(torch.bfloat16)
            student_in = batch_data['student'].to(device, non_blocking=True).to(torch.bfloat16)

            # Flatten B and K
            B, K, T, C, H, W = dino_in.shape
            dino_in = dino_in.view(B*K, T, C, H, W)
            siglip_in = siglip_in.view(B*K, T, C, H, W)
            student_in = student_in.view(B*K, T, C, H, W)
            student_in = student_in.permute(0, 2, 1, 3, 4) # (BK, C, T, H, W)

            # Teacher Inference
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_target = teachers(dino_in, siglip_in) # 16, 8, 1536, 32, 32
                student_features = student(student_in) # 16, 8, 1536, 32, 32

                import pdb
                pdb.set_trace()

                # 计算当前Batch指标
                batch_metrics = calculate_metrics(student_features, teacher_target, SPLIT_DIM)

            # 累加
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            
            total_steps += 1
            if rank == 0:
                pbar.update(1)

    if rank == 0:
        pbar.close()

    # 5. 多卡汇总
    final_metrics = {}
    total_steps_tensor = torch.tensor(total_steps, device=device)
    dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)
    global_steps = total_steps_tensor.item()

    for k, v in metrics_sum.items():
        val_tensor = torch.tensor(v, device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM) # 汇总所有卡的sum
        final_metrics[k] = val_tensor.item() / global_steps

    # 6. 输出结果
    if rank == 0:
        print("\n" + "="*30)
        print(" Evaluation Results ")
        print("="*30)
        print(f"{'Metric':<15} | {'Value':<10}")
        print("-" * 28)
        print(f"{'MSE Total':<15} | {final_metrics['mse_total']:.6f}")
        print(f"{'Cos Total':<15} | {final_metrics['cos_total']:.6f}")
        print("-" * 28)
        print(f"{'MSE DINO':<15} | {final_metrics['mse_dino']:.6f}")
        print(f"{'Cos DINO':<15} | {final_metrics['cos_dino']:.6f}")
        print("-" * 28)
        print(f"{'MSE SigLIP':<15} | {final_metrics['mse_siglip']:.6f}")
        print(f"{'Cos SigLIP':<15} | {final_metrics['cos_siglip']:.6f}")
        print("="*30)

        # 保存结果到txt
        save_file = os.path.join(os.path.dirname(args.checkpoint), "eval_results.txt")
        with open(save_file, "w") as f:
            for k, v in final_metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"Results saved to {save_file}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
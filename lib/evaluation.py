import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import torch.nn as nn

import torch
import torch.nn.functional as F

def cosine_similarity_loss(pred, target, dim=2):
    """
    计算余弦相似度。
    Input: (B, T, C, H, W)
    Return: Scalar (Mean Cosine Similarity)
    注意：通常我们希望相似度越高越好，这里返回的是相似度值（1.0为完全相同，-1.0为完全相反）。
    如果要作为 Loss，通常使用 1 - cosine_similarity。
    """
    # F.cosine_similarity 会计算指定维度的相似度，返回 (B, T, H, W)
    sim = F.cosine_similarity(pred, target, dim=dim, eps=1e-8)
    return 1 - sim.mean().item()

def linear_cka(pred, target):
    """
    计算 Linear CKA (Centered Kernel Alignment).
    CKA 用于衡量两个特征矩阵之间的结构相似性，具有各向同性缩放和正交变换不变性。
    
    Input: (B, T, C, H, W) -> 需要 Flatten 为 (N, Features)
    我们视 (B*T) 为样本数 N，视 (C*H*W) 为特征维度 D。
    或者视 (B*T*H*W) 为样本数，C 为特征维度。
    
    通常在 Vision Transformer 中，我们比较的是 Sample 间的相关性矩阵。
    这里采用: N = Batch * Time, D = Channel * H * W
    """
    # Flatten: (B, T, C, H, W) -> (B*T, C*H*W)
    B, T, C, H, W = pred.shape
    X = pred.reshape(B * T, -1)
    Y = target.reshape(B * T, -1)
    
    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Calculate Gram matrices
    gram_x = torch.mm(X, X.t())
    gram_y = torch.mm(Y, Y.t())

    # CKA calculation
    # HSIC(K, L) = tr(K H L H) / (n-1)^2, with centered data K=XX^T
    scaled_hsic_xy = torch.trace(torch.mm(gram_x, gram_y))
    scaled_hsic_xx = torch.trace(torch.mm(gram_x, gram_x))
    scaled_hsic_yy = torch.trace(torch.mm(gram_y, gram_y))

    cka = scaled_hsic_xy / (torch.sqrt(scaled_hsic_xx) * torch.sqrt(scaled_hsic_yy))
    return cka.item()


def feature_kl_loss(pred, target, temperature=1.0, eps=1e-8):
    """
    KL on channel dimension: 每个空间位置的特征分布
    Input: (B, T, C, H, W)
    """
    B, T, C, H, W = pred.shape
    
    # Reshape: (B*T*H*W, C) - 每个空间位置有 C 维特征
    pred_flat = pred.permute(0, 1, 3, 4, 2).reshape(-1, C)    # (N, C), N=B*T*H*W
    target_flat = target.permute(0, 1, 3, 4, 2).reshape(-1, C)
    
    # LayerNorm 稳定数值（可选）
    pred_flat = F.layer_norm(pred_flat, (C,))
    target_flat = F.layer_norm(target_flat, (C,))
    
    # 在通道维度 C 上做 softmax（关键！）
    pred_log = F.log_softmax(pred_flat / temperature, dim=-1)   # dim=-1 是 C
    target_prob = F.softmax(target_flat / temperature, dim=-1)
    
    # 裁剪防止极端值
    pred_log = torch.clamp(pred_log, min=-20, max=20)
    target_prob = torch.clamp(target_prob, min=eps, max=1.0)
    
    kl = F.kl_div(pred_log, target_prob, reduction='batchmean')
    
    return kl


def procrustes_distance(pred, target):
    """
    计算 Procrustes Distance。
    寻找最佳的正交变换矩阵 Q (旋转/反射)，使得 ||Target - Pred * Q||_F 最小。
    返回的是最小化后的 Frobenius Norm 的平方。
    
    Input: Flattened to (N, D) similar to CKA.
    """
    B, T, C, H, W = pred.shape
    # Reshape: (N, D) -> (B*T, C*H*W)
    # 注意：Procrustes通常要求 N >= D 或者 D 较大。
    # 这里为了计算效率和物理意义，我们将 Channel 视为特征维度，空间和Batch视为样本
    # Reshape -> (B*T*H*W, C)
    X = pred.permute(0, 1, 3, 4, 2).reshape(-1, C).float()
    Y = target.permute(0, 1, 3, 4, 2).reshape(-1, C).float()
    
    # 1. Compute X^T Y
    M = torch.mm(Y.t(), X)
    
    # 2. SVD
    # torch.linalg.svd 可能会比较慢，且在 float16/bfloat16 下容易不稳定，建议转 float32
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except RuntimeError:
        # SVD 不收敛时的 fallback，返回 -1 或极大值
        return -1.0

    # 3. Calculate Trace (Nuclear Norm of M) = Sum of singular values
    # Procrustes Distance d^2 = ||Y||^2 + ||X||^2 - 2 * tr(Sigma)
    # 其中 tr(Sigma) 是 M 的奇异值之和
    
    norm_sq_X = torch.sum(X ** 2)
    norm_sq_Y = torch.sum(Y ** 2)
    nuclear_norm = torch.sum(S)
    
    # d^2
    dist_sq = norm_sq_X + norm_sq_Y - 2 * nuclear_norm
    
    # 为了指标的可读性，通常可以归一化，或者直接返回距离
    return torch.abs(dist_sq).sqrt().item() # 开根号返回距离

@torch.no_grad()
def validate_one_epoch(student, teachers, dataloader, device, split_dim=768):
    student.eval()
    teachers.eval()
    
    # 初始化累加器
    metrics = {
        "loss_mse_all": 0.0, "loss_mse_dino": 0.0, "loss_mse_siglip": 0.0,
        "loss_cos_all": 0.0, "loss_cos_dino": 0.0, "loss_cos_siglip": 0.0,
        # "loss_kl_all": 0.0,  "loss_kl_dino": 0.0,  "loss_kl_siglip": 0.0, # 越低越好
        "cka_all": 0.0, "cka_dino": 0.0, "cka_siglip": 0.0, # 越高越好
    }
    
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)
    total_batches = 0
    rank = dist.get_rank()
    iterator = tqdm(dataloader, desc="Validating", disable=(rank != 0))

    for batch_data in iterator:
        dino_in = batch_data['dino'].to(device, non_blocking=True).to(torch.bfloat16)
        siglip_in = batch_data['siglip'].to(device, non_blocking=True).to(torch.bfloat16)
        student_in = batch_data['student'].to(device, non_blocking=True).to(torch.bfloat16)
        
        B, K, T, C, H, W = dino_in.shape
        dino_in = dino_in.view(B*K, T, C, H, W)
        siglip_in = siglip_in.view(B*K, T, C, H, W)
        student_in = student_in.view(B*K, T, C, H, W)
        student_in = student_in.permute(0, 2, 1, 3, 4)  # → (BK, 3, T, 512, 512)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            target = teachers(dino_in, siglip_in) 
            # target = teacher_out.permute(0, 1, 4, 2, 3) # (N, T, C, H, W)
            pred = student(student_in) # (N, T, C, H, W)

            target_dino = target[:, :, :split_dim, :, :]
            target_siglip = target[:, :, split_dim:, :, :]
            pred_dino = pred[:, :, :split_dim, :, :]
            pred_siglip = pred[:, :, split_dim:, :, :]

            # --- Existing Metrics ---
            metrics["loss_mse_all"] += F.mse_loss(pred, target).item()
            metrics["loss_mse_dino"] += F.mse_loss(pred_dino, target_dino).item()
            metrics["loss_mse_siglip"] += F.mse_loss(pred_siglip, target_siglip).item()
            
            # Cosine Similarity Loss (1 - cos, 越低越好)
            metrics["loss_cos_all"] += cosine_similarity_loss(pred, target)
            metrics["loss_cos_dino"] += cosine_similarity_loss(pred_dino, target_dino)
            metrics["loss_cos_siglip"] += cosine_similarity_loss(pred_siglip, target_siglip)

            # 辅助指标（不作为 Loss，但用于分析）
            # metrics["loss_kl_all"] += feature_kl_loss(pred, target, temperature=2.0).item()
            # metrics["loss_kl_dino"] += feature_kl_loss(pred_dino, target_dino, temperature=2.0).item()
            # metrics["loss_kl_siglip"] += feature_kl_loss(pred_siglip, target_siglip, temperature=2.0).item()
            
            # 2. CKA (越高越好, max 1.0)
            # 注意：CKA 计算量大且需要 float32 以保证精度，建议只计算 All 或者在 Validation 时 sub-sample
            metrics["cka_all"] += linear_cka(pred.float(), target.float())
            metrics["cka_dino"] += linear_cka(pred_dino.float(), target_dino.float())
            metrics["cka_siglip"] += linear_cka(pred_siglip.float(), target_siglip.float())

            total_batches += 1

    # DDP Aggregation
    metrics_tensor = torch.tensor(list(metrics.values()), device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    
    total_batches_tensor = torch.tensor([total_batches], device=device)
    dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)
    
    avg_metrics = metrics_tensor / total_batches_tensor
    result = {k: avg_metrics[i].item() for i, k in enumerate(metrics.keys())}

    decision_metric = (result["loss_mse_all"] + result["loss_cos_all"] + 1 - result["cka_all"]) / 3
    
    return result, decision_metric
import torch.nn as nn
import torch.nn.functional as F
import torch

LOSSES = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss()
}

class CosineLoss(nn.Module):
    """
    SOTA 常用: 针对 DINO/SigLIP 等 Representation Learning 模型。
    最大化特征向量之间的余弦相似度 (即最小化 1 - cos)。
    """
    def __init__(self, dim=2, eps=1e-8):
        super().__init__()
        self.dim = dim # 这里的 dim 需要对应 Channel 维度
        self.eps = eps

    def forward(self, pred, target):
        # pred, target shape: (B*K, T, C, H, W) 或者是 permuted 后的形状
        # 我们需要确保在 Channel 维度上计算
        # 假设输入已经被 permute 成 (Batch, Time, Channel, H, W) -> Channel 是 dim 2
        
        # 将非 Channel 维度展平以便统一计算，或者直接指定 dim
        # 这里直接使用 PyTorch 的 cosine_similarity
        cosine_sim = F.cosine_similarity(pred, target, dim=self.dim, eps=self.eps)
        # loss = 1 - mean(similarity)
        return 1.0 - cosine_sim.mean()

class ChannelWiseDivergenceLoss(nn.Module):
    """
    源自 Channel-wise Distillation (CW)。
    将 Channel 维度通过 Softmax 归一化，计算 KL 散度。
    这有助于 Student 学习特征的“分布”而不是具体的数值。
    """
    def __init__(self, temperature=1.0, channel_dim=2):
        super().__init__()
        self.T = temperature
        self.dim = channel_dim

    def forward(self, pred, target):
        # 1. 沿 Channel 维度做 LogSoftmax (Student) 和 Softmax (Teacher)
        # 假设 Input Shape: (..., C, ...)
        
        pred_log_prob = F.log_softmax(pred / self.T, dim=self.dim)
        target_prob = F.softmax(target / self.T, dim=self.dim)
        
        # 2. 计算 KL Divergence
        # reduction='batchmean' 也就是对 batch 求平均，对分布求和
        loss = F.kl_div(pred_log_prob, target_prob, reduction='batchmean')
        
        return loss * (self.T ** 2)

class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer (AT):
    将高维特征在 Channel 维度求平均（或平方和），得到 Spatial Attention Map。
    强迫 Student 的注意力热力图与 Teacher 一致。
    """
    def __init__(self, channel_dim=2):
        super().__init__()
        self.dim = channel_dim
        self.mse = nn.MSELoss()

    def get_attention_map(self, x):
        # 基于 Activation 的 Attention: sum(|x|, dim=C) 或 mean(x^2, dim=C)
        # 这里使用 mean(x^2) 是一种经典做法
        return torch.mean(x.pow(2), dim=self.dim)

    def forward(self, pred, target):
        at_pred = self.get_attention_map(pred)
        at_target = self.get_attention_map(target)
        
        # 对 Attention Map 进行 L2 归一化使得数值在同一量级 (可选，但在异构蒸馏中很重要)
        at_pred_norm = F.normalize(at_pred.view(at_pred.size(0), -1), p=2, dim=1)
        at_target_norm = F.normalize(at_target.view(at_target.size(0), -1), p=2, dim=1)
        
        return self.mse(at_pred_norm, at_target_norm)

# --- 更新你的 LOSSES 字典 ---

# 注意：根据你下文的 permute (0, 1, 4, 2, 3)，
# 原始 shape (B, T, 32, 32, 1536) -> (B, T, 1536, 32, 32)
# Channel 维度变成了 index 2。
CHANNEL_DIM = 2 

LOSSES.update({
    "cosine": CosineLoss(dim=CHANNEL_DIM),
    "cw_kl": ChannelWiseDivergenceLoss(temperature=4.0, channel_dim=CHANNEL_DIM),
    "at": AttentionTransferLoss(channel_dim=CHANNEL_DIM)
})

from lib.diagnostic import FeatureDiagnostic
analyzer = FeatureDiagnostic(save_dir="./feature_analysis")


# 我同时使用了 mse + cos，权重都是0.5
class DecoupledDistillationLoss(nn.Module):
    def __init__(self, loss_types, split_dim=768, channel_dim=2, dino_weight=1.0, siglip_weight=1.0):
        """
        Args:
            loss_types: list, 例如 ["mse", "cosine", "at"]
            split_dim: int, 切分点，这里是 768
            channel_dim: int, channel 所在的维度索引
            dino_weight: float, DINO 部分 loss 的权重
            siglip_weight: float, SigLIP 部分 loss 的权重
        """
        super().__init__()
        self.loss_types = loss_types
        self.split_dim = split_dim
        self.channel_dim = channel_dim
        self.dino_weight = dino_weight
        self.siglip_weight = siglip_weight
        
        # 初始化基础 Loss 函数
        self.loss_fns = {}
        for lt in loss_types:
            if lt in LOSSES:
                self.loss_fns[lt] = LOSSES[lt]
            else:
                print(f"Warning: {lt} not found in LOSSES dict.")

    def forward(self, pred, target):
        """
        pred, target: (B, T, 1536, H, W)
        """
        # 1. 解耦特征
        # torch.split 会返回两个 tensor
        # print(target.shape)
        # 收集统计信息
        # B, T, C, H, W = target.shape
        # out_dino = target.view(B*T, C, H, W)[:, :self.split_dim, :, :] # (B, C ,H, W)
        # out_siglip = target.view(B*T, C, H, W)[:, self.split_dim:, :, :]
        
        # out_dino = out_dino.permute(0, 2, 3, 1).view(B*T, H*W, self.split_dim) # (B*T, H*W, C_dino)
        # out_siglip = out_siglip.permute(0, 2, 3, 1).view(B*T, H*W, self.split_dim) # (B*T, H*W, C_siglip)
        # analyzer.collect(out_dino, out_siglip)
        # # analyzer.visualize_spatial_heatmaps(out_dino, out_siglip)
        # analyzer.run_analysis()
        # quit()
        pred_dino, pred_siglip = torch.split(pred, self.split_dim, dim=self.channel_dim)
        target_dino, target_siglip = torch.split(target, self.split_dim, dim=self.channel_dim)
        
        total_loss = 0.0
        log_metrics = {}

        # 2. 遍历定义的 Loss 类型 (MSE, Cosine, etc.)
        for name, fn in self.loss_fns.items():
            # 计算 DINO Loss
            l_dino = fn(pred_dino, target_dino)
            
            # 计算 SigLIP Loss
            l_siglip = fn(pred_siglip, target_siglip)
            
            # 加权求和
            term_loss = (l_dino * self.dino_weight) + (l_siglip * self.siglip_weight)
            total_loss += term_loss
            
            # 记录分项用于调试 (可选)
            # log_metrics[f"{name}_dino"] = l_dino.item()
            # log_metrics[f"{name}_siglip"] = l_siglip.item()

        return total_loss


class CombinedDistillationLoss(nn.Module):
    def __init__(self, loss_types, split_dim=768, channel_dim=2, dino_weight=1.0, siglip_weight=1.0):
        """
        Args:
            loss_types: list, 例如 ["mse", "cosine", "at"]
            split_dim: int, 切分点，这里是 768
            channel_dim: int, channel 所在的维度索引
            dino_weight: float, DINO 部分 loss 的权重
            siglip_weight: float, SigLIP 部分 loss 的权重
        """
        super().__init__()
        self.loss_types = loss_types
        self.split_dim = split_dim
        self.channel_dim = channel_dim
        self.dino_weight = dino_weight
        self.siglip_weight = siglip_weight
        
        # 初始化基础 Loss 函数
        self.loss_fns = {}
        for lt in loss_types:
            if lt in LOSSES:
                self.loss_fns[lt] = LOSSES[lt]
            else:
                print(f"Warning: {lt} not found in LOSSES dict.")

    def forward(self, pred, target):
        """
        pred, target: (B, T, 1536, H, W)
        """
        total_loss = 0.0

        for name, fn in self.loss_fns.items():
            l = fn(pred, target)            
            total_loss += l

        return total_loss


        
class HybridDistillationLoss(nn.Module):
    def __init__(self, loss_types, split_dim=768, channel_dim=2, dino_weight=1.0, siglip_weight=1.0):
        """
        Args:
            loss_types: list, 例如 ["mse", "cosine", "at"]
            split_dim: int, 切分点，这里是 768
            channel_dim: int, channel 所在的维度索引
            dino_weight: float, DINO 部分 loss 的权重
            siglip_weight: float, SigLIP 部分 loss 的权重
        """
        super().__init__()
        self.loss_types = loss_types
        self.split_dim = split_dim
        self.channel_dim = channel_dim
        self.dino_weight = dino_weight
        self.siglip_weight = siglip_weight
        
        # 初始化基础 Loss 函数
        self.loss_fns = {}
        for lt in loss_types:
            if lt in LOSSES:
                self.loss_fns[lt] = LOSSES[lt]
            else:
                print(f"Warning: {lt} not found in LOSSES dict.")

    def forward(self, pred, target):
        """
        pred, target: (B, T, 1536, H, W)
        """
        pred_dino, pred_siglip = torch.split(pred, self.split_dim, dim=self.channel_dim)
        target_dino, target_siglip = torch.split(target, self.split_dim, dim=self.channel_dim)
        
        total_loss = 0.0
        log_metrics = {}

        # 2. 遍历定义的 Loss 类型 (MSE, Cosine, etc.)
        for name, fn in self.loss_fns.items():
            # 计算 DINO Loss
            l_dino = fn(pred_dino, target_dino)
            
            # 计算 SigLIP Loss
            l_siglip = fn(pred_siglip, target_siglip)

            # overall
            l_overall = fn(pred, target)
            
            # 加权求和
            term_loss = ((l_dino * self.dino_weight) + (l_siglip * self.siglip_weight)) * 0.5 + l_overall * 0.5
            total_loss += term_loss

        return total_loss





class StatsMatchingLoss(nn.Module):
    """
    匹配特征的均值和标准差，消除整体分布的漂移。
    帮助解决 CKA 上升但 MSE 上升的问题。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        # x, y shape: (B, T, C, H, W)
        # 在 (B, T, H, W) 维度上计算均值和方差，保留 C
        # 也就是让 Student 学会对每个 Channel 的统计特性
        
        # 展平除 Channel 外的维度
        b, t, c, h, w = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(-1, c) # (N, C)
        y_flat = y.permute(0, 1, 3, 4, 2).reshape(-1, c) # (N, C)

        x_mean = x_flat.mean(dim=0)
        x_std = x_flat.std(dim=0)
        y_mean = y_flat.mean(dim=0)
        y_std = y_flat.std(dim=0)

        loss_mean = F.mse_loss(x_mean, y_mean)
        loss_std = F.mse_loss(x_std, y_std)
        
        return loss_mean + loss_std


class LocalStatsMatchingLoss(nn.Module):
    """
    在空间局部区域匹配统计，保留空间结构
    """
    def __init__(self, window_size=8, eps=1e-5):  # 建议将 eps 稍微调大为 1e-5 更稳
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.pool = nn.AvgPool2d(window_size, stride=window_size)
        
    def forward(self, x, y):
        # 【修改 1】强制转换为 float32，防止 float16 下 x**2 溢出产生 Inf 和 NaN
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        # x, y: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # 合并 B,T 维度处理
        x = x.reshape(B*T, C, H, W)
        y = y.reshape(B*T, C, H, W)
        
        # 局部均值（保留空间结构）
        x_mean_local = self.pool(x)  # (B*T, C, H//w, W//w)
        y_mean_local = self.pool(y)
        
        # 局部方差
        # 【修改 2】增加 torch.clamp(..., min=0.0)
        # 防止浮点精度误差导致 E[X^2] - (E[X])^2 变成微小的负数，进而导致 sqrt(负数) 报错
        x_var_local = torch.clamp(self.pool(x**2) - x_mean_local**2, min=0.0)
        y_var_local = torch.clamp(self.pool(y**2) - y_mean_local**2, min=0.0)
        
        loss_mean = F.mse_loss(x_mean_local, y_mean_local)
        
        # 加 eps 并求 sqrt（此时输入必定 >= eps，梯度和数值都非常安全）
        loss_var = F.mse_loss(torch.sqrt(x_var_local + self.eps), 
                              torch.sqrt(y_var_local + self.eps))
        
        return loss_mean + loss_var
    
LOSSES.update({
    "stats": StatsMatchingLoss(),
    "local_stats": LocalStatsMatchingLoss()
})


class AdaptiveDecoupledDistillationLoss(nn.Module):
    def __init__(self, loss_types=["smooth_l1", "cosine"], split_dim=768, channel_dim=2):
        super().__init__()
        self.loss_types = loss_types
        self.split_dim = split_dim
        self.channel_dim = channel_dim
        
        # === 核心改进 1: 自动权重学习 (Learnable Weights) ===
        # 参数 eta 用于平衡 DINO 和 SigLIP 的 Loss 规模
        # Loss = Loss_1 / exp(eta_1) + eta_1 + Loss_2 / exp(eta_2) + eta_2
        self.eta_dino = nn.Parameter(torch.zeros(1))
        self.eta_siglip = nn.Parameter(torch.zeros(1))

        # === 核心改进 2: 损失函数升级 ===
        # 推荐使用 SmoothL1 替代 MSE，因为它在误差较大时梯度更稳定，且对离群点不敏感
        self.reg_loss = nn.SmoothL1Loss(beta=1.0)
        
        # 统计匹配 Loss
        self.stats_loss = StatsMatchingLoss()

    def cosine_loss(self, pred, target):
        # pred, target: (B, T, C, H, W)
        # Cosine Similarity 
        # dim=2 是 Channel 维度
        cosine_sim = F.cosine_similarity(pred, target, dim=self.channel_dim, eps=1e-8)
        return 1.0 - cosine_sim.mean()

    def forward(self, pred, target):
        """
        pred, target: (B, T, 1536, 32, 32)
        """
        # 1. 解耦特征
        pred_dino, pred_siglip = torch.split(pred, self.split_dim, dim=self.channel_dim)
        target_dino, target_siglip = torch.split(target, self.split_dim, dim=self.channel_dim)

        # -----------------------------------------------------------
        # 计算 DINO 部分 Loss
        # -----------------------------------------------------------
        l_dino_reg = self.reg_loss(pred_dino, target_dino) # 回归 Loss
        l_dino_cos = self.cosine_loss(pred_dino, target_dino) # 角度 Loss
        # 添加统计约束，防止漂移
        l_dino_stats = self.stats_loss(pred_dino, target_dino) * 0.1 
        
        total_loss_dino = l_dino_reg + l_dino_cos + l_dino_stats

        # -----------------------------------------------------------
        # 计算 SigLIP 部分 Loss
        # -----------------------------------------------------------
        l_siglip_reg = self.reg_loss(pred_siglip, target_siglip)
        l_siglip_cos = self.cosine_loss(pred_siglip, target_siglip)
        l_siglip_stats = self.stats_loss(pred_siglip, target_siglip) * 0.1
        
        total_loss_siglip = l_siglip_reg + l_siglip_cos + l_siglip_stats

        # -----------------------------------------------------------
        # === 核心改进 3: 应用不确定性加权 ===
        # 这种方式允许模型动态降低"难学"任务（Loss 较大的任务）的权重，
        # 防止 SigLIP 的高 Loss 破坏共享特征
        # -----------------------------------------------------------
        loss = (0.5 * torch.exp(-self.eta_dino) * total_loss_dino + 0.5 * self.eta_dino) + \
               (0.5 * torch.exp(-self.eta_siglip) * total_loss_siglip + 0.5 * self.eta_siglip)

        return loss

    def get_learnable_params(self):
        """将这些参数加入优化器"""
        return [self.eta_dino, self.eta_siglip]


import math
class AdaptiveWeightedLossV2(nn.Module):
    def __init__(self, loss_types, split_dim=768, channel_dim=2):
        super().__init__()
        self.stats_loss = StatsMatchingLoss()
        self.mse_loss = nn.MSELoss()
        self.cos_loss = CosineLoss(dim=channel_dim)
        
    def forward(self, pred, target, epoch=None, total_epochs=None):
        # 基础损失
        l_mse = self.mse_loss(pred, target)
        l_cos = self.cos_loss(pred, target)
        l_stats = self.stats_loss(pred, target)
        
        # 策略：早期 Stats 权重高，后期 MSE/Cos 权重高
        if epoch is not None:
            # 余弦退火权重
            progress = epoch / total_epochs
            w_stats = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
            w_fine = 1  
        else:
            w_stats = 0.3
            w_fine = 0.7
        
        # 自适应权重（不确定性加权）
        w_mse = 1.0
        w_cos = 1.0
        w_stats_adaptive = 0.5
        
        # 组合
        loss = w_fine * (w_mse * l_mse + w_cos * l_cos) + \
               w_stats * w_stats_adaptive * l_stats
        
        return loss #, {
        #     'mse': l_mse.item(), 
        #     'cos': l_cos.item(), 
        #     'stats': l_stats.item(),
        #     'w_stats': w_stats
        # }

LOSSES.update({
    "adaptive_decoupled": AdaptiveDecoupledDistillationLoss(),
    "adaptive_v2": AdaptiveWeightedLossV2(loss_types=["mse", "cosine", "stats"], split_dim=768, channel_dim=CHANNEL_DIM)
})
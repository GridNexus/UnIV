"""
Temporal action localization head (CSCAN mechanism).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StateSpaceFeaturePyramid(nn.Module):
    def __init__(self, embed_dim: int = 1536, state_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.input_proj = nn.Linear(embed_dim, state_dim)
        self.ssm_layers = nn.ModuleList([StateSpaceBlock(state_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(state_dim, embed_dim)
        
    def forward(self, x):
        h = self.input_proj(x)
        for layer in self.ssm_layers:
            h = layer(h)
        return self.output_proj(h)


class StateSpaceBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.x_proj = nn.Linear(dim, dim * 2)
        self.dt_proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
    def forward(self, x):
        B, T, D = x.shape
        x_gate = self.x_proj(x)
        x1, x2 = x_gate.chunk(2, dim=-1)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = torch.matmul(h, self.A.T) + torch.matmul(x1[:, t], self.B.T)
            y = torch.matmul(h, self.C.T) + x2[:, t]
            outputs.append(y)
        return self.norm(x + torch.stack(outputs, dim=1))


class TopDownPath(nn.Module):
    def __init__(self, embed_dim: int = 1536, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3),
        ])
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        
    def forward(self, features):
        processed = []
        for i, feat in enumerate(features):
            x = feat.permute(0, 2, 1)
            y = self.scale_convs[i](x).permute(0, 2, 1)
            processed.append(y)
        return self.fusion(torch.cat(processed, dim=-1))


class CrossScaleSelectiveFusion(nn.Module):
    def __init__(self, embed_dim: int = 1536):
        super().__init__()
        self.select_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim // 4), nn.ReLU(), nn.Linear(embed_dim // 4, embed_dim), nn.Sigmoid())
        self.global_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        
    def forward(self, local_feat, global_feat):
        gate = self.select_gate(local_feat)
        fused = gate * local_feat + (1 - gate) * global_feat
        return fused + self.global_proj(fused)


class ActionLocalizationHead(nn.Module):
    def __init__(self, embed_dim: int = 1536, state_dim: int = 256, num_classes: int = 5, num_scales: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ssm_pyramid = StateSpaceFeaturePyramid(embed_dim=embed_dim, state_dim=state_dim, num_layers=4)
        self.top_down = TopDownPath(embed_dim=embed_dim, num_scales=num_scales)
        self.cross_scale = CrossScaleSelectiveFusion(embed_dim=embed_dim)
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=4, dilation=4),
        ])
        self.class_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_classes))
        self.boundary_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 2))
        self.gaussian_sampling = GaussianSampling()
        
    def forward(self, z_seq, return_intermediate=False):
        multi_scale_features = []
        x = z_seq.permute(0, 2, 1)
        for conv in self.scale_convs:
            multi_scale_features.append(conv(x).permute(0, 2, 1))
        ssm_out = self.ssm_pyramid(z_seq)
        top_down_out = self.top_down(multi_scale_features)
        fused = self.cross_scale(top_down_out, ssm_out)
        out = fused + z_seq
        action_scores = self.class_embed(out)
        boundary_offsets = self.boundary_embed(out)
        result = {'action_scores': action_scores, 'boundary_offsets': boundary_offsets, 'features': out}
        if return_intermediate:
            result['multi_scale_features'] = multi_scale_features
            result['ssm_features'] = ssm_out
        return result


class GaussianSampling(nn.Module):
    def __init__(self, sigma_min=0.5, sigma_max=2.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def forward(self, features, boundary_offsets):
        B, T, C = features.shape
        duration = boundary_offsets[:, :, 1] - boundary_offsets[:, :, 0]
        sigma = torch.rand(B, device=features.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        t = torch.arange(T, device=features.device).float()
        gaussian_weights = []
        for b in range(B):
            center = (boundary_offsets[b, :, 0] + boundary_offsets[b, :, 1]) / 2
            gauss = torch.exp(-((t - center) ** 2) / (2 * sigma[b] ** 2))
            gaussian_weights.append(gauss)
        gaussian_weights = torch.stack(gaussian_weights)
        gaussian_weights = gaussian_weights / gaussian_weights.sum(dim=1, keepdim=True)
        return features * gaussian_weights.unsqueeze(-1)


class ActionLocalizationLoss(nn.Module):
    def __init__(self, num_classes: int = 5, lambda_cls=1.0, lambda_reg=1.0, lambda_scale=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_scale = lambda_scale
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.l1_loss = nn.L1Loss(reduction='none')
        
    def forward(self, predictions, targets):
        action_scores = predictions['action_scores']
        boundary_offsets = predictions['boundary_offsets']
        action_labels = targets['action_labels']
        boundary_labels = targets['boundary_labels']
        B, T, C = action_scores.shape
        action_scores_flat = action_scores.reshape(B * T, C)
        action_labels_flat = action_labels.reshape(B * T)
        cls_loss = self.focal_loss(action_scores_flat, action_labels_flat)
        reg_loss = self.l1_loss(boundary_offsets, boundary_labels).mean()
        scale_loss = 0.0
        if 'multi_scale_features' in predictions:
            ms_feat = predictions['multi_scale_features']
            if len(ms_feat) >= 2:
                scale_loss = F.mse_loss(ms_feat[0], ms_feat[1])
        total_loss = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss + self.lambda_scale * scale_loss
        return total_loss, {'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item(), 'scale_loss': scale_loss}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

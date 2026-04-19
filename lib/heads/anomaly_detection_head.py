"""
Segment-level anomaly detection head (MIL-based).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv(x))))


class MILAnomalyHead(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, num_classes: int = 1, num_conv_layers: int = 4):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = input_dim
        dilations = [1, 2, 4, 8]
        kernel_sizes = [3, 3, 3, 3]
        for i in range(num_conv_layers):
            out_ch = hidden_dim if i < num_conv_layers - 1 else hidden_dim // 2
            self.conv_layers.append(TemporalConvBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sizes[i], dilation=dilations[i]))
            in_ch = out_ch
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_classes),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z_seq):
        x = z_seq.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.permute(0, 2, 1)
        return self.sigmoid(self.mlp(x).squeeze(-1))


class AnomalyLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 0.1, alpha_update_method: str = "batch_freq"):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.alpha_update_method = alpha_update_method
        self.register_buffer('alpha', torch.tensor(1.0))
        
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        pos_weight = self.alpha
        loss_bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        weight = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))
        loss_bce = (loss_bce * weight).mean()
        diff = predictions[:, 1:] - predictions[:, :-1]
        loss_smooth = (diff ** 2).mean()
        return loss_bce + self.lambda_smooth * loss_smooth, {'bce_loss': loss_bce.item(), 'smooth_loss': loss_smooth.item(), 'alpha': self.alpha.item()}
    
    def update_alpha(self, positive_ratio):
        if self.alpha_update_method == "batch_freq":
            new_alpha = (1.0 - positive_ratio) / (positive_ratio + 1e-8)
            self.alpha = self.alpha * 0.9 + new_alpha * 0.1


class MILClassifier(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        self.aggregator = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, 1))
        
    def forward(self, bag_features):
        bag_repr = bag_features.mean(dim=1)
        return self.classifier(self.aggregator(bag_repr))


class VideoLevelAnomalyClassifier(nn.Module):
    def __init__(self, input_dim: int = 1536, num_classes: int = 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(input_dim, input_dim // 4), nn.Tanh(), nn.Linear(input_dim // 4, 1))
        self.classifier = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(input_dim // 2, num_classes))
        
    def forward(self, z_seq):
        feat = self.feature_extractor(z_seq)
        attn_weights = F.softmax(self.attention(feat), dim=1)
        video_repr = (feat * attn_weights).sum(dim=1)
        return self.classifier(video_repr), attn_weights.squeeze(-1)


class AnomalyDetectionHead(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.segment_detector = MILAnomalyHead(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=1)
        self.video_classifier = VideoLevelAnomalyClassifier(input_dim=input_dim, num_classes=num_classes)
        self.instance_aggregator = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        
    def forward(self, z_seq):
        segment_scores = self.segment_detector(z_seq)
        instance_features = self.instance_aggregator(z_seq)
        video_scores, attention_weights = self.video_classifier(instance_features)
        return {'segment_scores': segment_scores, 'video_scores': video_scores, 'attention_weights': attention_weights, 'instance_features': instance_features}


class SmoothAnomalyLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 0.2, lambda_temporal: float = 0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_temporal = lambda_temporal
        self.register_buffer('pos_weight', torch.tensor(5.0))
        
    def forward(self, predictions, targets):
        if isinstance(predictions, dict):
            segment_scores = predictions['segment_scores']
            video_scores = predictions['video_scores']
            if 'video_labels' in targets:
                video_labels = targets['video_labels']
                video_loss = F.cross_entropy(video_scores, video_labels)
                seg_loss = F.binary_cross_entropy(segment_scores, targets.get('segment_labels', torch.zeros_like(segment_scores)), weight=self.pos_weight)
                return video_loss + 0.3 * seg_loss, {'video_loss': video_loss.item(), 'segment_loss': seg_loss.item()}
        segment_scores = predictions if not isinstance(predictions, dict) else predictions['segment_scores']
        seg_loss = F.binary_cross_entropy(segment_scores, targets, weight=self.pos_weight)
        diff = segment_scores[:, 1:] - segment_scores[:, :-1]
        smooth_loss = (diff ** 2).mean()
        temporal_loss = F.mse_loss(segment_scores[:, :-1], segment_scores[:, 1:])
        total_loss = seg_loss + self.lambda_smooth * smooth_loss + self.lambda_temporal * temporal_loss
        return total_loss, {'seg_loss': seg_loss.item(), 'smooth_loss': smooth_loss, 'temporal_loss': temporal_loss}

"""
Spatiotemporal feature projection and compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatiotemporalProjection(nn.Module):
    def __init__(self, in_channels: int = 2048, out_channels: int = 1536, hidden_channels: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.projection(x)
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        return out


class TemporalExpansion(nn.Module):
    def __init__(self, in_channels: int = 1536, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_embed = nn.Parameter(torch.randn(1, in_channels, num_frames, 1, 1) * 0.02)
        
    def forward(self, x):
        x = x.expand(-1, -1, self.num_frames, -1, -1)
        x = x + self.temporal_embed
        return x


class SpatialAttentionCompression(nn.Module):
    def __init__(self, embed_dim: int = 1536, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = 256
        
        self.spatial_flatten = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        assert H == 16 and W == 16
        
        outputs = []
        for t in range(T):
            x_t = x[:, :, t, :, :].permute(0, 2, 3, 4, 1).reshape(B, self.seq_len, C)
            x_t = self.spatial_flatten(x_t)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_t = torch.cat([cls_tokens, x_t], dim=1)
            x_t = x_t + self.pos_embed
            attn_out, _ = self.attention(x_t, x_t, x_t)
            x_t = self.norm1(x_t + attn_out)
            x_t = self.norm2(x_t + self.mlp(x_t))
            cls_output = x_t[:, 0]
            outputs.append(cls_output)
        
        return torch.stack(outputs, dim=1)


class TemporalSequenceBuilder(nn.Module):
    def __init__(self, embed_dim: int = 1536, hidden_dim: int = 4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim),
        )
        
    def forward(self, clip_features):
        return clip_features


class SpatiotemporalCompressor(nn.Module):
    def __init__(self, encoder_channels: int = 2048, embed_dim: int = 1536, num_frames: int = 8, num_heads: int = 12):
        super().__init__()
        self.projection = SpatiotemporalProjection(in_channels=encoder_channels, out_channels=embed_dim)
        self.temporal_expansion = TemporalExpansion(in_channels=embed_dim, num_frames=num_frames)
        self.attention_compression = SpatialAttentionCompression(embed_dim=embed_dim, num_heads=num_heads)
        self.sequence_builder = TemporalSequenceBuilder(embed_dim=embed_dim)
        
    def forward(self, encoder_output):
        x = self.projection(encoder_output)
        x = self.temporal_expansion(x)
        x = self.attention_compression(x)
        x = self.sequence_builder(x)
        return x

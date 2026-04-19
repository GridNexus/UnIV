import torch
import torch.nn as nn


class VisualFlowEncoder(nn.Module):
    def __init__(self, hidden_dim=2048, output_dim=1536, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 128, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3), bias=False),
            nn.GroupNorm(2, 128), nn.GELU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(2, 256), nn.GELU(),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(2, 512), nn.GELU(),
            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(2, 1024), nn.GELU(),
            nn.Conv3d(1024, hidden_dim, kernel_size=(8, 1, 1), stride=(1, 1, 1), bias=False),
            nn.GroupNorm(2, hidden_dim), nn.GELU() 
        )
        self.temporal_embed = nn.Parameter(torch.zeros(1, hidden_dim, num_frames, 1, 1))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        self.decoder = nn.Sequential(
            # kernel_size=(3, 3, 3), padding=(1, 1, 1) -> 允许跨帧信息交互
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False), 
            nn.GroupNorm(2, hidden_dim), nn.GELU(),
            
            # Upsample 保持不变
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(2, hidden_dim), nn.GELU(),
            
            # 输出层
            nn.Conv3d(hidden_dim, output_dim, kernel_size=1, bias=True)
        )

    def forward(self, x):
        if x.shape[2] == 3: x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        hidden = self.encoder(x)
        z = hidden.expand(-1, -1, self.num_frames, -1, -1)
        z = z + self.temporal_embed
        out = self.decoder(z)
        out = out.permute(0, 2, 1, 3, 4)
        return out


"""
Modern Spatiotemporal Feature Encoder combining ConvNext3D and ViT3D
Inspired by DINOv3 and SigLIP architectures

Architecture Flow:
1. Input: 8 frames of 512x512x3 RGB images (raw preprocessed images)
2. ConvNext3D Encoder: Local spatiotemporal feature extraction
3. ViT3D Encoder: Global spatiotemporal attention modeling  
4. Compressed Representation: 1x16x16x2048 spatiotemporal tokens
5. Decoder: Expand to 8x32x32x1536 per-frame features for distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LayerNorm3d(nn.Module):
    """3D Layer Normalization (for ConvNext3D)"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ConvNext3DBlock(nn.Module):
    """
    ConvNext Block adapted for 3D spatiotemporal data
    
    Following ConvNeXt design:
    - Depthwise conv (7x7x7) -> LayerNorm -> 1x1x1 expansion -> GELU -> 1x1x1 projection
    - Layer scale + stochastic depth
    """
    def __init__(
        self, 
        dim: int, 
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim
        )  # Depthwise conv
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)  # Expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)  # Projection
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath3D(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma[:, None, None, None] * x
        
        x = residual + self.drop_path(x)
        return x


class DropPath3D(nn.Module):
    """Drop paths (Stochastic Depth) for 3D tensors"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        random_tensor = torch.floor(random_tensor)  # 不 inplace
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for ViT3D
    Converts (B, C, T, H, W) to (B, num_patches, embed_dim)
    """
    def __init__(
        self,
        temporal_patch_size: int = 2,
        spatial_patch_size: int = 1,
        in_channels: int = 2048,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
        )
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', C)
        return x, (T, H, W)


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding
    Inspired by DINOv3's RoPE implementation
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        # Create frequency bands - 注意这里 dim 应该是 head_dim
        # 由于我们会对 (real, imag) pairs 应用旋转，所以只需要 dim//2 个频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim // 2, 1).float() / (dim // 2)))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, t: int, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """
        Generate 3D rotary embeddings
        
        Args:
            t, h, w: Dimensions of the 3D grid
        Returns:
            cos, sin: Position embeddings of shape (t*h*w, dim)
        """
        # Create 3D coordinate grid
        t_coords = torch.arange(t, device=device, dtype=dtype)
        h_coords = torch.arange(h, device=device, dtype=dtype)
        w_coords = torch.arange(w, device=device, dtype=dtype)
        
        # Normalize coordinates to [-1, 1]
        t_coords = 2 * t_coords / max(t - 1, 1) - 1
        h_coords = 2 * h_coords / max(h - 1, 1) - 1
        w_coords = 2 * w_coords / max(w - 1, 1) - 1
        
        # Create meshgrid
        coords = torch.stack(torch.meshgrid(t_coords, h_coords, w_coords, indexing='ij'), dim=-1)
        coords = coords.flatten(0, 2)  # (t*h*w, 3)
        
        # 为三个维度分配频率（均匀分配）
        dim_t = len(self.inv_freq) // 3
        dim_h = len(self.inv_freq) // 3
        dim_w = len(self.inv_freq) - dim_t - dim_h
        
        # Compute angles for each dimension
        angles_t = coords[:, 0:1] @ self.inv_freq[:dim_t].unsqueeze(0)  # (N, dim_t)
        angles_h = coords[:, 1:2] @ self.inv_freq[dim_t:dim_t+dim_h].unsqueeze(0)  # (N, dim_h)
        angles_w = coords[:, 2:3] @ self.inv_freq[dim_t+dim_h:].unsqueeze(0)  # (N, dim_w)
        
        # Concatenate angles
        angles = torch.cat([angles_t, angles_h, angles_w], dim=-1)  # (N, dim//2)
        
        # 重复以匹配 head_dim (因为每个 angle 对应一对 cos/sin)
        angles = torch.cat([angles, angles], dim=-1)  # (N, dim)
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        return cos, sin


def apply_rotary_emb_3d(x, cos, sin):
    """
    Apply 3D rotary embeddings
    
    Args:
        x: (B, num_heads, N, head_dim)
        cos, sin: (N, head_dim)
    """
    # 将 x 重塑为复数对的形式
    # x: (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim//2, 2)
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    
    # 分离实部和虚部
    x_real = x_reshape[..., 0]  # (B, num_heads, N, head_dim//2)
    x_imag = x_reshape[..., 1]  # (B, num_heads, N, head_dim//2)
    
    # 扩展 cos 和 sin
    cos = cos[None, None, :, :]  # (1, 1, N, head_dim)
    sin = sin[None, None, :, :]  # (1, 1, N, head_dim)
    
    # 同样重塑 cos 和 sin
    cos_reshape = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_reshape = sin.reshape(*sin.shape[:-1], -1, 2)
    
    cos_real = cos_reshape[..., 0]  # (1, 1, N, head_dim//2)
    sin_real = sin_reshape[..., 0]  # (1, 1, N, head_dim//2)
    
    # 应用旋转
    x_rotated_real = x_real * cos_real - x_imag * sin_real
    x_rotated_imag = x_real * sin_real + x_imag * cos_real
    
    # 重新组合
    x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
    x_rotated = x_rotated.flatten(-2)  # (B, num_heads, N, head_dim)
    
    return x_rotated


class Attention3D(nn.Module):
    """
    Multi-head attention for 3D spatiotemporal data
    Inspired by DINOv3 attention
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if provided (skip CLS token)
        if rope_cos is not None and rope_sin is not None:
            # 分离 CLS token (第一个位置) 和其他 tokens
            q_cls, q_spatial = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_spatial = k[:, :, :1, :], k[:, :, 1:, :]
            
            # 只对 spatial tokens 应用 RoPE
            q_spatial = apply_rotary_emb_3d(q_spatial, rope_cos, rope_sin)
            k_spatial = apply_rotary_emb_3d(k_spatial, rope_cos, rope_sin)
            
            # 重新组合
            q = torch.cat([q_cls, q_spatial], dim=2)
            k = torch.cat([k_cls, k_spatial], dim=2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class ViT3DBlock(nn.Module):
    """
    Vision Transformer block for 3D data
    Inspired by DINOv3 layer design
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.drop_path = DropPath3D(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        
        # Layer scale (like DINOv3)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
    
    def forward(self, x, rope_cos=None, rope_sin=None):
        # Attention block with residual
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), rope_cos, rope_sin))
        
        # MLP block with residual
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        
        return x


class VideoToTokenEncoder(nn.Module):
    """
    Encoder: Raw video frames -> Compressed spatiotemporal tokens
    
    Input: (B, 3, 8, 512, 512) - 8 frames of 512x512 RGB images
    Output: (B, 2048, 1, 16, 16) - Compressed spatiotemporal token representation
    """
    def __init__(
        self,
        in_channels: int = 3,  # RGB images
        convnext_dims: list = [96, 192, 384, 768],  # Adjusted for 512x512 input
        convnext_depths: list = [2, 2, 3, 2],
        vit_embed_dim: int = 896,
        vit_depth: int = 12,
        vit_num_heads: int = 14,
        drop_path_rate: float = 0.1,
        output_channels: int = 2048,
    ):
        super().__init__()
        
        # === ConvNext3D Stem: 512x512 -> 128x128 ===
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, convnext_dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            LayerNorm3d(convnext_dims[0]),
        )
        
        # === ConvNext3D Stages ===
        # Stage 0: 128x128, dim=96
        # Stage 1: 64x64, dim=192, temporal downsample 8->4
        # Stage 2: 32x32, dim=384
        # Stage 3: 16x16, dim=768
        
        self.convnext_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(convnext_depths))]
        cur = 0
        
        for i in range(4):
            # Downsampling layer
            if i > 0:
                # Temporal downsample only at stage 1 (8 frames -> 4 frames)
                temporal_stride = 2 if i == 1 else 1
                downsample = nn.Sequential(
                    LayerNorm3d(convnext_dims[i-1]),
                    nn.Conv3d(
                        convnext_dims[i-1],
                        convnext_dims[i],
                        kernel_size=(temporal_stride, 2, 2),
                        stride=(temporal_stride, 2, 2),
                    ),
                )
            else:
                downsample = nn.Identity()
            
            # Blocks for this stage
            blocks = []
            for j in range(convnext_depths[i]):
                blocks.append(
                    ConvNext3DBlock(
                        dim=convnext_dims[i],
                        drop_path=dp_rates[cur + j],
                    )
                )
            cur += convnext_depths[i]
            
            stage = nn.Sequential(downsample, *blocks)
            self.convnext_stages.append(stage)
        
        # After ConvNext stages: (B, 768, 4, 16, 16)
        
        # === Bridge to ViT dimension ===
        self.bridge = nn.Sequential(
            nn.Conv3d(convnext_dims[-1], output_channels, kernel_size=1),
            LayerNorm3d(output_channels),
        )
        # After bridge: (B, 2048, 4, 16, 16)
        
        # === Transition to ViT ===
        self.patch_embed = PatchEmbed3D(
            temporal_patch_size=2,  # 4 -> 2
            spatial_patch_size=1,   # Keep 16x16
            in_channels=output_channels,
            embed_dim=vit_embed_dim,
        )
        
        # CLS token (like DINOv3)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        
        # === ViT3D Blocks ===
        self.rope = RoPE3D(dim=vit_embed_dim // vit_num_heads)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, vit_depth)]
        self.vit_blocks = nn.ModuleList([
            ViT3DBlock(
                dim=vit_embed_dim,
                num_heads=vit_num_heads,
                drop_path=dpr[i],
            )
            for i in range(vit_depth)
        ])
        
        self.norm = nn.LayerNorm(vit_embed_dim)
        
        # === Compression head ===
        self.compress_head = nn.Sequential(
            nn.Linear(vit_embed_dim, output_channels),
            nn.LayerNorm(output_channels),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 8, 512, 512) - Raw video frames
        Returns:
            (B, 2048, 1, 16, 16) - Compressed spatiotemporal tokens
        """
        B = x.shape[0]
        
        # === ConvNext3D: Local feature extraction ===
        x = self.stem(x)  # (B, 96, 8, 128, 128)
        
        for stage in self.convnext_stages:
            x = stage(x)
        # After stages: (B, 768, 4, 16, 16)
        
        x = self.bridge(x)  # (B, 2048, 4, 16, 16)
        
        # === ViT3D: Global attention ===
        x, (T, H, W) = self.patch_embed(x)  # (B, 2*16*16, 1024) = (B, 512, 1024)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 513, 1024)
        
        # Generate RoPE embeddings (for non-CLS tokens)
        rope_cos, rope_sin = self.rope(T, H, W, x.device, x.dtype)
        
        # Apply transformer blocks
        for blk in self.vit_blocks:
            x = blk(x, rope_cos, rope_sin)
        
        x = self.norm(x)
        
        # Remove CLS token and compress
        x = x[:, 1:, :]  # (B, 512, 1024)
        x = self.compress_head(x)  # (B, 512, 2048)
        
        # Reshape to spatial format
        x = x.reshape(B, T, H, W, 2048)
        x = x.permute(0, 4, 1, 2, 3)  # (B, 2048, 2, 16, 16)
        
        # Global average pooling over time to get 1x16x16
        x = x.mean(dim=2, keepdim=True)  # (B, 2048, 1, 16, 16)
        
        return x


class TokenToFeatureDecoder(nn.Module):
    """
    Decoder: Compressed tokens -> Per-frame distillation features
    
    Input: (B, 2048, 1, 16, 16) - Compressed spatiotemporal tokens
    Output: (B, 1536, 8, 32, 32) - Per-frame features for distillation
    """
    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 1536,
        num_frames: int = 8,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_frames = num_frames
        
        # Temporal expansion with learnable embeddings
        self.temporal_embed = nn.Parameter(torch.randn(1, in_channels, num_frames, 1, 1))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # Decoder pathway
        self.decoder = nn.Sequential(
            # Initial projection
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
            LayerNorm3d(hidden_dim),
            nn.GELU(),
            
            # Spatiotemporal processing with cross-frame interaction
            ConvNext3DBlock(hidden_dim, kernel_size=3),
            ConvNext3DBlock(hidden_dim, kernel_size=3),
            
            # Spatial upsample 16x16 -> 32x32
            nn.ConvTranspose3d(
                hidden_dim, hidden_dim,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
            LayerNorm3d(hidden_dim),
            nn.GELU(),
            
            # More refinement
            ConvNext3DBlock(hidden_dim, kernel_size=3),
            
            # Final projection to output channels
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 2048, 1, 16, 16) - Compressed tokens
        Returns:
            (B, 1536, 8, 32, 32) - Per-frame distillation features
        """
        B = x.shape[0]
        
        # Expand temporally by broadcasting and adding learned embeddings
        x = x.expand(-1, -1, self.num_frames, -1, -1)  # (B, 2048, 8, 16, 16)
        x = x + self.temporal_embed
        
        # Decode through network
        x = self.decoder(x)  # (B, 1536, 8, 32, 32)
        
        return x


class SpatiotemporalFeatureEncoder(nn.Module):
    """
    Complete spatiotemporal feature encoder for distillation
    
    Mimics the architecture style of DINOv3 and SigLIP with:
    - Modular design
    - Modern components (LayerNorm, GELU, Layer Scale, DropPath)
    - RoPE positional encoding
    - Hybrid Conv + Transformer architecture
    
    Pipeline:
    Raw Images (8x512x512x3) -> Compressed Tokens (1x16x16x2048) 
                               -> Distillation Features (8x32x32x1536)
    """
    def __init__(
        self,
        # Input/Output config
        in_channels: int = 3,  # RGB images
        latent_channels: int = 2048,
        output_channels: int = 1536,  # For distillation
        num_frames: int = 8,
        # Encoder params
        convnext_dims: list = [96, 192, 384, 768],
        convnext_depths: list = [2, 2, 3, 2],
        vit_embed_dim: int = 896,
        vit_depth: int = 10,
        vit_num_heads: int = 14,
        # Decoder params
        decoder_hidden_dim: int = 1024,
        # Regularization
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = VideoToTokenEncoder(
            in_channels=in_channels,
            convnext_dims=convnext_dims,
            convnext_depths=convnext_depths,
            vit_embed_dim=vit_embed_dim,
            vit_depth=vit_depth,
            vit_num_heads=vit_num_heads,
            drop_path_rate=drop_path_rate,
            output_channels=latent_channels,
        )
        
        self.decoder = TokenToFeatureDecoder(
            in_channels=latent_channels,
            out_channels=output_channels,
            num_frames=num_frames,
            hidden_dim=decoder_hidden_dim,
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 8, 512, 512) - Raw video frames
        Returns:
            features: (B, 1536, 8, 32, 32) - Per-frame features for distillation
            tokens: (B, 2048, 1, 16, 16) - Compressed spatiotemporal tokens
        """
        tokens = self.encoder(x)
        features = self.decoder(tokens)
        
        # 确认形状
        B, C, T, H, W = features.shape
        assert T == 8 and C == 1536 and H == 32 and W == 32, f"Expected features shape (B, 1536, 8, 32, 32), got {features.shape}"
        features = features.permute(0, 2, 1, 3, 4)  # (B, 8, 1536, 32, 32)
        # print(f"Output features shape: {features.shape} (B, T, C, H, W)")
        # quit()

        return features
    
    def encode(self, x):
        """Encode video frames to compressed tokens"""
        return self.encoder(x)
    
    def decode(self, tokens):
        """Decode tokens to per-frame features"""
        return self.decoder(tokens)


# === Example Usage ===
if __name__ == "__main__":
    # Create model
    model = SpatiotemporalFeatureEncoder(
        in_channels=3,  # RGB input
        latent_channels=2048,
        output_channels=1536,  # For distillation
        num_frames=8,
        convnext_dims=[96, 192, 384, 768],
        convnext_depths=[2, 2, 3, 2],
        vit_embed_dim=896,
        vit_depth=10,
        vit_num_heads=14,
        decoder_hidden_dim=1024,
        drop_path_rate=0.1,
    )
    
    # Test forward pass with RAW IMAGES
    x = torch.randn(2, 3, 8, 512, 512)  # 2 batches, RGB, 8 frames, 512x512
    
    print(f"Input shape (raw images): {x.shape}")
    print(f"  - Batch size: {x.shape[0]}")
    print(f"  - Channels (RGB): {x.shape[1]}")
    print(f"  - Frames: {x.shape[2]}")
    print(f"  - Resolution: {x.shape[3]}x{x.shape[4]}")
    
    # Full pipeline
    features, tokens = model(x)
    print(f"\n=== Forward Pass ===")
    print(f"Compressed tokens shape: {tokens.shape}")
    print(f"  - Expected: (B, 2048, 1, 16, 16)")
    print(f"Distillation features shape: {features.shape}")
    print(f"  - Expected: (B, 1536, 8, 32, 32)")
    
    # Encoder only
    print(f"\n=== Encoder Only ===")
    tokens = model.encode(x)
    print(f"Tokens shape: {tokens.shape}")
    
    # Decoder only
    print(f"\n=== Decoder Only ===")
    features = model.decode(tokens)
    print(f"Features shape: {features.shape}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  - Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    print(f"\n=== Architecture Summary ===")
    print("Pipeline: Raw Images -> Compressed Tokens -> Distillation Features")
    print("  1. Input: 8 frames of 512x512 RGB images")
    print("  2. ConvNext3D: Local spatiotemporal feature extraction")
    print("  3. ViT3D: Global attention with RoPE")
    print("  4. Tokens: 1x16x16x2048 compressed representation")
    print("  5. Decoder: Expand to 8x32x32x1536 for distillation")
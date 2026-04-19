"""
Pure ViT3D Spatiotemporal Feature Encoder (Corrected)

Architecture Flow:
1. Input: 8 frames of 512x512x3 RGB images
2. Full Transformer pathway: Global spatiotemporal attention
3. Compressed Representation: 512 tokens with 2048 dimensions
4. Transformer Decoder: Expand to 8x32x32x1536 per-frame features
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class RoPE3D(nn.Module):
    """3D Rotary Position Embedding"""
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim // 2, 1).float() / (dim // 2)))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, t: int, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """Generate 3D rotary embeddings"""
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
        
        # Allocate frequencies for three dimensions
        dim_t = len(self.inv_freq) // 3
        dim_h = len(self.inv_freq) // 3
        dim_w = len(self.inv_freq) - dim_t - dim_h
        
        # Compute angles for each dimension
        angles_t = coords[:, 0:1] @ self.inv_freq[:dim_t].unsqueeze(0)
        angles_h = coords[:, 1:2] @ self.inv_freq[dim_t:dim_t+dim_h].unsqueeze(0)
        angles_w = coords[:, 2:3] @ self.inv_freq[dim_t+dim_h:].unsqueeze(0)
        
        angles = torch.cat([angles_t, angles_h, angles_w], dim=-1)
        angles = torch.cat([angles, angles], dim=-1)  # Duplicate for cos/sin pairs
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        return cos, sin


def apply_rotary_emb_3d(x, cos, sin):
    """Apply 3D rotary embeddings"""
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_reshape[..., 0]
    x_imag = x_reshape[..., 1]
    
    # cos, sin shape: (seq_len, dim) -> (1, 1, seq_len, dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    
    cos_reshape = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_reshape = sin.reshape(*sin.shape[:-1], -1, 2)
    
    cos_real = cos_reshape[..., 0]
    sin_real = sin_reshape[..., 0]
    
    x_rotated_real = x_real * cos_real - x_imag * sin_real
    x_rotated_imag = x_real * sin_real + x_imag * cos_real
    
    x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
    x_rotated = x_rotated.flatten(-2)
    
    return x_rotated


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        random_tensor = torch.floor(random_tensor)
        output = x.div(keep_prob) * random_tensor
        return output


class Attention3D(nn.Module):
    """Multi-head attention with 3D RoPE"""
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
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            # Check dimensions to distinguish between Encoder (with CLS) and Decoder (Full Sequence)
            seq_len = q.shape[2]
            rope_len = rope_cos.shape[0]
            
            if rope_len == seq_len:
                # Decoder case: RoPE is prepared for the full sequence
                q = apply_rotary_emb_3d(q, rope_cos, rope_sin)
                k = apply_rotary_emb_3d(k, rope_cos, rope_sin)
            elif rope_len == seq_len - 1:
                # Encoder case: First token is CLS, RoPE applies to the rest
                q_cls, q_spatial = q[:, :, :1, :], q[:, :, 1:, :]
                k_cls, k_spatial = k[:, :, :1, :], k[:, :, 1:, :]
                
                q_spatial = apply_rotary_emb_3d(q_spatial, rope_cos, rope_sin)
                k_spatial = apply_rotary_emb_3d(k_spatial, rope_cos, rope_sin)
                
                q = torch.cat([q_cls, q_spatial], dim=2)
                k = torch.cat([k_cls, k_spatial], dim=2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
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
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
    
    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), rope_cos, rope_sin))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding - converts video frames to tokens"""
    def __init__(
        self,
        temporal_patch_size: int = 1,
        spatial_patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
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
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', C)
        x = self.norm(x)
        return x, (T, H, W)


class ViT3DEncoder(nn.Module):
    """
    Pure ViT3D Encoder: Raw video frames -> Compressed spatiotemporal tokens
    """
    def __init__(
        self,
        in_channels: int = 3,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        output_dim: int = 2048,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.rope = RoPE3D(dim=embed_dim // num_heads)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.compress = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        x, (T, H, W) = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        rope_cos, rope_sin = self.rope(T, H, W, x.device, x.dtype)
        
        for blk in self.blocks:
            x = blk(x, rope_cos, rope_sin)
        
        x = self.norm(x)
        x = x[:, 1:, :]  # Remove CLS token
        x = self.compress(x)
        
        return x, (T, H, W)


class ViT3DDecoder(nn.Module):
    """
    Pure ViT3D Decoder: Compressed tokens -> Per-frame features
    """
    def __init__(
        self,
        input_dim: int = 2048,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        output_dim: int = 1536,
        num_frames: int = 8,
        output_resolution: int = 32,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.output_resolution = output_resolution
        self.num_output_tokens = num_frames * output_resolution * output_resolution
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self.output_queries = nn.Parameter(
            torch.randn(1, self.num_output_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.output_queries, std=0.02)
        
        self.rope = RoPE3D(dim=embed_dim // num_heads)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, spatial_shape):
        B = x.shape[0]
        # x: (B, N, input_dim)
        
        # Project input tokens
        x = self.input_proj(x)
        
        # Expand output queries
        queries = self.output_queries.expand(B, -1, -1)
        
        # Concatenate: [Input Tokens | Output Queries]
        x = torch.cat([x, queries], dim=1)
        
        # Generate RoPE for output resolution
        rope_cos, rope_sin = self.rope(
            self.num_frames, 
            self.output_resolution, 
            self.output_resolution,
            x.device, 
            x.dtype
        )
        
        # Pad RoPE to match concatenated length
        # For input tokens (context), we use Identity rotation (cos=1, sin=0)
        N_input = x.shape[1] - self.num_output_tokens
        
        # FIX: Pad with 1s for Cos (Identity) instead of 0s
        rope_cos_pad = torch.ones(N_input, rope_cos.shape[1], device=rope_cos.device, dtype=rope_cos.dtype)
        # Pad with 0s for Sin
        rope_sin_pad = torch.zeros(N_input, rope_sin.shape[1], device=rope_sin.device, dtype=rope_sin.dtype)
        
        rope_cos = torch.cat([rope_cos_pad, rope_cos], dim=0)
        rope_sin = torch.cat([rope_sin_pad, rope_sin], dim=0)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x, rope_cos, rope_sin)
        
        x = self.norm(x)
        
        # Extract output tokens
        x = x[:, -self.num_output_tokens:, :]
        x = self.output_proj(x)
        
        x = x.reshape(B, self.num_frames, self.output_resolution, self.output_resolution, -1)
        x = x.permute(0, 4, 1, 2, 3)
        
        return x


class PureViT3DModel(nn.Module):
    """
    Complete Pure ViT3D Model for spatiotemporal feature encoding
    """
    def __init__(
        self,
        in_channels: int = 3,
        spatial_patch_size: int = 32,
        temporal_patch_size: int = 1,
        encoder_embed_dim: int = 1024,
        encoder_depth: int = 10,
        encoder_num_heads: int = 16,
        decoder_embed_dim: int = 768,
        decoder_depth: int = 3,
        decoder_num_heads: int = 12,
        latent_channels: int = None,
        latent_dim: int = 2048,
        output_channels: int = None,
        output_dim: int = 1536,
        num_frames: int = 8,
        output_resolution: int = 32,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if latent_channels is not None:
            latent_dim = latent_channels
        if output_channels is not None:
            output_dim = output_channels

        self.encoder = ViT3DEncoder(
            in_channels=in_channels,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            output_dim=latent_dim,
        )
        
        self.decoder = ViT3DDecoder(
            input_dim=latent_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            output_dim=output_dim,
            num_frames=num_frames,
            output_resolution=output_resolution,
        )
    
    def forward(self, x):
        tokens, spatial_shape = self.encoder(x)
        features = self.decoder(tokens, spatial_shape)
        # Permute to (B, T, C, H, W)
        features = features.permute(0, 2, 1, 3, 4)
        return features
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, tokens, spatial_shape):
        return self.decoder(tokens, spatial_shape)


if __name__ == "__main__":
    model = PureViT3DModel(
        in_channels=3,
        spatial_patch_size=32,
        temporal_patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=10,
        encoder_num_heads=16,
        decoder_embed_dim=768,
        decoder_depth=3,
        decoder_num_heads=12,
        latent_dim=2048,
        output_dim=1536,
        num_frames=8,
        output_resolution=32,
        drop_path_rate=0.1,
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 8, 512, 512)
    
    print(f"Input shape: {x.shape}")
    print(f"  - (B, C, T, H, W)")
    
    # Full pipeline
    features = model(x)
    print(f"\nOutput features shape: {features.shape}")
    print(f"  - Expected: (B, T, C, H, W) = (2, 8, 1536, 32, 32)")
    
    # Encoder only
    tokens, spatial_shape = model.encode(x)
    print(f"\nCompressed tokens shape: {tokens.shape}")
    print(f"  - Spatial shape from encoder: T={spatial_shape[0]}, H={spatial_shape[1]}, W={spatial_shape[2]}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\n=== Pure ViT3D Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  - Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    print(f"\n=== Architecture Summary ===")
    print("Pure ViT3D architecture - all transformers, no convolutions")
    print("Advantages:")
    print("  - Better long-range dependencies")
    print("  - More flexible attention patterns")
    print("  - Scalable to larger models")
    print("  - Better transfer learning potential")

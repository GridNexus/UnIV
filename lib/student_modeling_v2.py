import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


# ============================================================
# 基础组件
# ============================================================

class LayerNorm3d(nn.Module):
    """3D Layer Normalization (channel-first)"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class DropPath3D(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        return x.div(keep_prob) * mask


class ConvNext3DBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-6, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) \
            if layer_scale_init_value > 0 else None
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
        return residual + self.drop_path(x)


# ============================================================
# RoPE 3D
# ============================================================

class RoPE3D(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        self.register_buffer('inv_freq', inv_freq)
        self.dim = dim

    def forward(self, t, h, w, device, dtype):
        t_c = 2 * torch.arange(t, device=device, dtype=dtype) / max(t - 1, 1) - 1
        h_c = 2 * torch.arange(h, device=device, dtype=dtype) / max(h - 1, 1) - 1
        w_c = 2 * torch.arange(w, device=device, dtype=dtype) / max(w - 1, 1) - 1
        coords = torch.stack(torch.meshgrid(t_c, h_c, w_c, indexing='ij'), dim=-1).flatten(0, 2)

        nf = len(self.inv_freq)
        dt, dh = nf // 3, nf // 3
        dw = nf - dt - dh

        angles = torch.cat([
            coords[:, 0:1] @ self.inv_freq[:dt].unsqueeze(0),
            coords[:, 1:2] @ self.inv_freq[dt:dt+dh].unsqueeze(0),
            coords[:, 2:3] @ self.inv_freq[dt+dh:].unsqueeze(0),
        ], dim=-1)
        angles = torch.cat([angles, angles], dim=-1)
        return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb_3d(x, cos, sin):
    """x: (B, H, N, D)"""
    x_r = x.reshape(*x.shape[:-1], -1, 2)
    xr, xi = x_r[..., 0], x_r[..., 1]
    # reshape cos/sin from (N, D) -> (1, 1, N, D//2)
    half = cos.shape[-1] // 2
    cos_ = cos[:, :half][None, None]
    sin_ = sin[:, :half][None, None]
    out = torch.stack([xr * cos_ - xi * sin_, xr * sin_ + xi * cos_], dim=-1)
    return out.flatten(-2)


# ============================================================
# Self-Attention & Cross-Attention
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if rope_cos is not None:
            q_cls, q_s = q[:, :, :1], q[:, :, 1:]
            k_cls, k_s = k[:, :, :1], k[:, :, 1:]
            q_s = apply_rotary_emb_3d(q_s, rope_cos, rope_sin)
            k_s = apply_rotary_emb_3d(k_s, rope_cos, rope_sin)
            q = torch.cat([q_cls, q_s], dim=2)
            k = torch.cat([k_cls, k_s], dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_a = nn.LayerNorm(dim, eps=1e-6)
        self.norm_b = nn.LayerNorm(dim, eps=1e-6)

        # A -> B
        self.q_a  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_b = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_a = nn.Linear(dim, dim)

        # B -> A
        self.q_b  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_a = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_b = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Layer scale
        self.gamma_a = nn.Parameter(torch.ones(dim) * 1e-4)
        self.gamma_b = nn.Parameter(torch.ones(dim) * 1e-4)

    def _cross(self, q, kv_src):
        B, N_q, C = q.shape
        kv = kv_src.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q_ = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q_ @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        return (attn @ v).transpose(1, 2).reshape(B, N_q, C)

    def forward(self, x_a, x_b):
        na, nb = self.norm_a(x_a), self.norm_b(x_b)
        # A attends B
        out_a = self.proj_drop(self.proj_a(self._cross(self.q_a(na), self.kv_b(nb))))
        # B attends A
        out_b = self.proj_drop(self.proj_b(self._cross(self.q_b(nb), self.kv_a(na))))
        return x_a + self.gamma_a * out_a, x_b + self.gamma_b * out_b


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath3D(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_h, dim), nn.Dropout(drop),
        )
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), rope_cos, rope_sin))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbed3D(nn.Module):
    def __init__(self, temporal_patch_size=2, spatial_patch_size=1,
                 in_channels=1024, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (T, H, W)


# ============================================================
# TemporalTokenUnfolding（已修复）
# ============================================================

class UnfoldingDecoderLayer(nn.Module):
    """
    单独定义的 Block，避免在 ModuleList 中混入 Parameter 导致的报错
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # Parameter 属于 Module 的属性，会自动注册
        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-4)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(self, q, k, v):
        # q: queries (B, T, C)
        # k, v: keys/values (B, HW, C)
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(k), self.norm_kv(v))
        q = q + self.gamma1 * attn_out
        q = q + self.gamma2 * self.mlp(self.norm_ff(q))
        return q


class TemporalTokenUnfolding(nn.Module):
    """
    将压缩时空 token (B, C_in, 1, H, W) 解折叠为 T 帧特征。
    """
    def __init__(self, in_channels=2048, out_channels=1024, num_frames=8,
                 num_heads=16, mlp_ratio=4.0, num_layers=3):
        super().__init__()
        self.num_frames = num_frames
        self.out_channels = out_channels

        # 空间 token 投影为 key/value
        self.kv_proj = nn.Linear(in_channels, out_channels * 2)

        # 可学习帧 query（时序语义）
        self.frame_queries = nn.Parameter(torch.zeros(1, num_frames, out_channels))
        nn.init.trunc_normal_(self.frame_queries, std=0.02)

        # 固定 sinusoidal 帧位置编码
        pe = self._build_pe(num_frames, out_channels)
        self.register_buffer('frame_pe', pe)

        # 修复：使用标准的 Block 类，而不是混合 list
        self.layers = nn.ModuleList([
            UnfoldingDecoderLayer(out_channels, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # 空间残差投影（保留 H×W 信息）
        self.spatial_proj = nn.Linear(in_channels, out_channels)

        self._init_weights()

    def _build_pe(self, length, dim):
        pe = torch.zeros(1, length, dim)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:pe.shape[-1] // 2])
        return pe

    def _init_weights(self):
        nn.init.trunc_normal_(self.kv_proj.weight, std=0.02)
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.trunc_normal_(self.spatial_proj.weight, std=0.02)
        nn.init.zeros_(self.spatial_proj.bias)
        # UnfoldingDecoderLayer 内的权重会自动初始化，也可以在这里补充初始化逻辑

    def forward(self, tokens):
        """
        tokens: (B, C_in, 1, H, W)
        Returns: (B, C_out, T, H, W)
        """
        B, C_in, _, H, W = tokens.shape

        # 展平空间维度: (B, H*W, C_in)
        spatial = tokens.squeeze(2).flatten(2).permute(0, 2, 1)

        # 投影 key/value
        kv = self.kv_proj(spatial)   # (B, HW, 2*C_out)
        k, v = kv.chunk(2, dim=-1)   # (B, HW, C_out)

        # 初始化帧 query
        q = self.frame_queries.expand(B, -1, -1) + self.frame_pe  # (B, T, C_out)

        # Cross-attention 解码：每帧 query 关注所有空间 token
        for layer in self.layers:
            q = layer(q, k, v)

        # q: (B, T, C_out) — 每帧的全局时序特征
        # spatial_feat: (B, HW, C_out) — 空间细节
        spatial_feat = self.spatial_proj(spatial)  # (B, HW, C_out)

        # 外积展开: 时序特征广播到空间维度
        q_expand = q.unsqueeze(2)                    # (B, T, 1, C_out)
        sf_expand = spatial_feat.unsqueeze(1)         # (B, 1, HW, C_out)
        out = q_expand + sf_expand                    # (B, T, HW, C_out)  broadcast

        # 变形为 (B, C_out, T, H, W)
        out = out.reshape(B, self.num_frames, H, W, self.out_channels)
        out = out.permute(0, 4, 1, 2, 3)
        return out


# ============================================================
# 共享低层 ConvNext Stem（1024 维）
# ============================================================

class SharedConvNextStem(nn.Module):
    def __init__(self, in_channels=3, dims=(96, 192, 384, 768),
                 depths=(2, 2, 3, 2), out_channels=1024, drop_path_rate=0.05):
        super().__init__()
        self.stem_conv = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            LayerNorm3d(dims[0]),
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            if i > 0:
                t_s = 2 if i == 1 else 1
                down = nn.Sequential(
                    LayerNorm3d(dims[i - 1]),
                    nn.Conv3d(dims[i-1], dims[i], kernel_size=(t_s, 2, 2), stride=(t_s, 2, 2)),
                )
            else:
                down = nn.Identity()
            blocks = [ConvNext3DBlock(dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            cur += depths[i]
            self.stages.append(nn.Sequential(down, *blocks))

        self.bridge = nn.Sequential(
            nn.Conv3d(dims[-1], out_channels, kernel_size=1),
            LayerNorm3d(out_channels),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_conv(x)
        for stage in self.stages:
            x = stage(x)
        return self.bridge(x)


# ============================================================
# 双塔 ViT 编码器
# ============================================================

class DualTowerViTEncoder(nn.Module):
    def __init__(self, shared_channels=1024, tower_dim=768, num_heads=12,
                 total_depth=12, cross_attn_interval=3, use_cross_attention=True,
                 output_channels=2048, drop_path_rate=0.1):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.cross_attn_interval = cross_attn_interval
        self.output_channels = output_channels

        # 独立 patch embedding（两塔）
        self.patch_embed_a = PatchEmbed3D(2, 1, shared_channels, tower_dim)
        self.patch_embed_b = PatchEmbed3D(2, 1, shared_channels, tower_dim)

        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, tower_dim))
        self.cls_token_b = nn.Parameter(torch.zeros(1, 1, tower_dim))

        self.rope = RoPE3D(dim=tower_dim // num_heads)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.blocks_a = nn.ModuleList([ViTBlock(tower_dim, num_heads, drop_path=dpr[i]) for i in range(total_depth)])
        self.blocks_b = nn.ModuleList([ViTBlock(tower_dim, num_heads, drop_path=dpr[i]) for i in range(total_depth)])

        # 双向交叉注意力层（按 interval 数量预建）
        n_cross = total_depth // max(cross_attn_interval, 1) if cross_attn_interval > 0 else 0
        self.cross_layers = nn.ModuleList([
            BidirectionalCrossAttention(tower_dim, num_heads) for _ in range(n_cross)
        ])

        self.norm_a = nn.LayerNorm(tower_dim, eps=1e-6)
        self.norm_b = nn.LayerNorm(tower_dim, eps=1e-6)

        self.compress_a = nn.Sequential(nn.Linear(tower_dim, output_channels//2), nn.LayerNorm(output_channels//2, eps=1e-6))
        self.compress_b = nn.Sequential(nn.Linear(tower_dim, output_channels//2), nn.LayerNorm(output_channels//2, eps=1e-6))
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token_a, std=0.02)
        nn.init.trunc_normal_(self.cls_token_b, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, shared_feat):
        B = shared_feat.shape[0]

        x_a, (T, H, W) = self.patch_embed_a(shared_feat)
        x_b, _         = self.patch_embed_b(shared_feat)

        x_a = torch.cat([self.cls_token_a.expand(B, -1, -1), x_a], dim=1)
        x_b = torch.cat([self.cls_token_b.expand(B, -1, -1), x_b], dim=1)

        rope_cos, rope_sin = self.rope(T, H, W, shared_feat.device, shared_feat.dtype)

        cross_idx = 0
        for i, (blk_a, blk_b) in enumerate(zip(self.blocks_a, self.blocks_b)):
            x_a = blk_a(x_a, rope_cos, rope_sin)
            x_b = blk_b(x_b, rope_cos, rope_sin)

            if (self.use_cross_attention and self.cross_attn_interval > 0 and
                    (i + 1) % self.cross_attn_interval == 0 and
                    cross_idx < len(self.cross_layers)):
                x_a, x_b = self.cross_layers[cross_idx](x_a, x_b)
                cross_idx += 1

        x_a = self.norm_a(x_a)[:, 1:]   # remove CLS
        x_b = self.norm_b(x_b)[:, 1:]

        x_a = self.compress_a(x_a).reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).mean(2, keepdim=True)
        x_b = self.compress_b(x_b).reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).mean(2, keepdim=True)

        return torch.cat([x_a, x_b], dim=1)   # (B, 2048, 1, 16, 16)


# ============================================================
# 双路解码器
# ============================================================

class TokenToFeatureDecoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=1536, num_frames=8,
                 hidden_dim=1024, unfolding_layers=3):
        super().__init__()
        # Branch A: For DINOv3 Reconstruction
        self.unfold_a = TemporalTokenUnfolding(in_channels//2, hidden_dim, num_frames, 
                                               num_layers=unfolding_layers)
        self.refine_a = nn.Sequential(
            ConvNext3DBlock(hidden_dim, kernel_size=3),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            LayerNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, out_channels//2, kernel_size=1) # Output 768
        )

        # Branch B: For SigLIP Reconstruction
        self.unfold_b = TemporalTokenUnfolding(in_channels//2, hidden_dim, num_frames, 
                                               num_layers=unfolding_layers)
        self.refine_b = nn.Sequential(
            ConvNext3DBlock(hidden_dim, kernel_size=3),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            LayerNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, out_channels//2, kernel_size=1) # Output 768
        )

        # self.output_norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tokens):
        token_a, token_b = tokens.chunk(2, dim=1) # 各 (B, 1024, 1, 16, 16)
        
        # 1. 解码 DINO
        feat_a = self.unfold_a(token_a) # 需要调整 Unfolding 的 in_channels 为 1024
        out_dino = self.refine_a(feat_a)

        # 2. 解码 SigLIP
        feat_b = self.unfold_b(token_b)
        out_siglip = self.refine_b(feat_b)

        out_feat = torch.cat([out_dino, out_siglip], dim=1) # (B, 1536, 8, 32, 32)

        # out_feat = out_feat.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        # out_feat = self.output_norm(out_feat)       # LayerNorm on C
        # out_feat = out_feat.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        return out_feat # 返回两个张量


# ============================================================
# 完整模型
# ============================================================

class SpatiotemporalFeatureEncoderV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 2048,
        output_channels: int = 1536,
        num_frames: int = 8,
        # Shared stem
        convnext_dims: List[int] = [96, 192, 384, 768],
        convnext_depths: List[int] = [2, 2, 3, 2],
        shared_out_channels: int = 1024,
        # Dual-tower ViT
        tower_dim: int = 512,
        tower_num_heads: int = 8,
        tower_depth: int = 10,
        cross_attn_interval: int = 3,
        use_cross_attention: bool = True,   # ← 开关
        # Decoder
        decoder_hidden_dim: int = 768,
        decoder_unfolding_layers: int = 3,
        # Reg
        drop_path_rate: float = 0.1,
        # output norm
        use_output_norm: bool = False,
    ):
        super().__init__()

        self.shared_stem = SharedConvNextStem(
            in_channels=in_channels,
            dims=convnext_dims,
            depths=convnext_depths,
            out_channels=shared_out_channels,
            drop_path_rate=drop_path_rate * 0.5,
        )

        self.dual_tower = DualTowerViTEncoder(
            shared_channels=shared_out_channels,
            tower_dim=tower_dim,
            num_heads=tower_num_heads,
            total_depth=tower_depth,
            cross_attn_interval=cross_attn_interval,
            use_cross_attention=use_cross_attention,
            output_channels=latent_channels,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = TokenToFeatureDecoder(
            in_channels=latent_channels,
            out_channels=output_channels,
            num_frames=num_frames,
            hidden_dim=decoder_hidden_dim,
            unfolding_layers=decoder_unfolding_layers,
        )

        if use_output_norm:
            self.output_norm_dino = nn.LayerNorm(output_channels//2, eps=1e-6)
            self.output_norm_siglip = nn.LayerNorm(output_channels//2, eps=1e-6)
        else:
            self.output_norm_dino = None
            self.output_norm_siglip = None

    def forward(self, x):
        """x: (B, 3, 8, 512, 512) -> (B, 8, 1536, 32, 32)"""
        tokens = self.encode(x)          # (B, 2048, 1, 16, 16)
        features = self.decode(tokens)   # (B, 1536, 8, 32, 32)
        return features

    def encode(self, x):
        return self.dual_tower(self.shared_stem(x))

    def decode(self, tokens):
        out = self.decoder(tokens)
        B, C, T, H, W = out.shape
        if self.output_norm_dino is not None and self.output_norm_siglip is not None:
            out_dino, out_siglip = out.chunk(2, dim=1) # 各 (B, 768, T, H, W)
            out_dino = out_dino.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
            out_dino = self.output_norm_dino(out_dino)        # LayerNorm on C
            out_dino = out_dino.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            
            out_siglip = out_siglip.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
            out_siglip = self.output_norm_siglip(out_siglip)        # LayerNorm on C
            out_siglip = out_siglip.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            
            out = torch.cat([out_dino,out_siglip], dim=1)
        B, C, T, H, W = out.shape
        assert T == 8 and C == 1536 and H == 32 and W == 32, f"Shape error: {out.shape}"
        return out.permute(0, 2, 1, 3, 4)        # (B, 8, 1536, 32, 32)
    

# 向后兼容别名，训练脚本可以直接替换
SpatiotemporalFeatureEncoder = SpatiotemporalFeatureEncoderV2


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SpatiotemporalFeatureEncoderV2 — Architecture Test")
    print("=" * 60)

    configs = [
        ("With Cross-Attention  (interval=3)", True),
        ("Without Cross-Attention            ", False),
    ]

    for label, use_ca in configs:
        model = SpatiotemporalFeatureEncoderV2(
            in_channels=3,
            latent_channels=2048,
            output_channels=1536,
            num_frames=8,
            convnext_dims=[96, 192, 384, 768],
            convnext_depths=[2, 2, 3, 2],
            shared_out_channels=1024,
            tower_dim=512,
            tower_num_heads=8,
            tower_depth=10,
            cross_attn_interval=3,
            use_cross_attention=use_ca,
            decoder_hidden_dim=768,
            decoder_unfolding_layers=3,
            drop_path_rate=0.1,
        ).eval()

        x = torch.randn(1, 3, 8, 512, 512)
        with torch.no_grad():
            out = model(x)

        p_total = sum(p.numel() for p in model.parameters())
        
        print(f"\n[{label}]")
        print(f"  Input  : {list(x.shape)}")
        print(f"  Output : {list(out.shape)}  ✓" if list(out.shape) == [1, 8, 1536, 32, 32] else f"  Output : {list(out.shape)}  ✗")
        print(f"  Params : {p_total:,}  ({p_total*4/1024/1024:.1f} MB fp32)")

    print("\n" + "=" * 60)
    print("  Test passed!")
    print("=" * 60)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# ============================================================
# 1. 基础组件 (Basic Components)
# ============================================================

class LayerNormChFirst(nn.Module):
    """ 支持 Channel First 的 LayerNorm (B, C, T, H, W). """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

class DropPath(nn.Module):
    """ Stochastic Depth """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor

# ============================================================
# 2. 核心构建块 (Core Building Blocks)
# ============================================================

class ConvNext3DBlock(nn.Module):
    """
    标准的 ConvNeXt Block (3D版).
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNormChFirst(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1, 1)
            
        return input + self.drop_path(x)

class SpatiotemporalAttentionBlock(nn.Module):
    """
    Spatiotemporal Attention.
    用于混合阶段，只在分辨率 <= 32x32 时使用。
    """
    def __init__(self, dim, num_heads=16, drop_path=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = LayerNormChFirst(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.norm2 = LayerNormChFirst(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        B, C, T, H, W = x.shape
        residual = x
        
        # 1. Attention
        x = self.norm1(x)
        x_flat = x.flatten(2).transpose(1, 2) # (B, N, C)
        
        qkv = self.qkv(x_flat).reshape(B, T*H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, T*H*W, C)
        x_attn = self.proj(x_attn)
        x_attn = x_attn.transpose(1, 2).reshape(B, C, T, H, W)
        
        x = residual + self.drop_path(x_attn)
        
        # 2. MLP
        residual = x
        x = self.norm2(x)
        x_flat = x.flatten(2).transpose(1, 2)
        x_ffn = self.mlp(x_flat)
        x_ffn = x_ffn.transpose(1, 2).reshape(B, C, T, H, W)
        
        return residual + self.drop_path(x_ffn)

# ============================================================
# 3. 架构阶段 (Architecture Stages)
# ============================================================

class ChronosStem(nn.Module):
    """
    Stem Stage:
    Raw Input -> Patch Embed (Stride 4) -> ConvNext Blocks.
    Constraint: Only uses ConvNextBLOCK.
    """
    def __init__(self, in_channels, out_channels, depth, drop_rates):
        super().__init__()
        # 1. Patch Embedding (4x downsample)
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            LayerNormChFirst(out_channels)
        )
        
        # 2. Refine with ConvNext Blocks
        blocks = []
        for i in range(depth):
            blocks.append(ConvNext3DBlock(out_channels, drop_path=drop_rates[i]))
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x

class ChronosEncoder(nn.Module):
    """
    Encoder Stage:
    Downsample -> [ConvNext Blocks] + [Optional Attention Blocks].
    Constraint: Attention appears ONLY when resolution <= 32.
    """
    def __init__(self, in_dim, out_dim, depth, drop_rates, stride, current_res):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNormChFirst(in_dim),
            nn.Conv3d(in_dim, out_dim, kernel_size=stride, stride=stride)
        )
        
        self.blocks = nn.ModuleList()
        
        # 混合策略：
        # 如果分辨率 > 32: 全卷积
        # 如果分辨率 <= 32: 混合模式 (大部分是 Conv, 穿插 Attention)
        use_attention = (current_res <= 32)
        
        for i in range(depth):
            # 简单的混合逻辑：在最后几个 Block 或者每隔 N 个 Block 插入 Attention
            # 这里采用：如果启用 Attention，则每 3 个 Block 的最后一个替换为 Attention
            # 且保证 Attention Block 在 Conv Block 后面
            is_attn_layer = use_attention and ((i + 1) % 3 == 0)
            
            if is_attn_layer:
                self.blocks.append(SpatiotemporalAttentionBlock(out_dim, drop_path=drop_rates[i]))
            else:
                self.blocks.append(ConvNext3DBlock(out_dim, drop_path=drop_rates[i]))

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class ChronosDecoder(nn.Module):
    """
    Decoder Stage:
    Temporal Expand -> [ConvNext Blocks] -> Upsample -> [ConvNext Blocks].
    Constraint: Only uses ConvNextBLOCK.
    """
    def __init__(self, latent_dim, out_dims, depths, drop_rates, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        
        # 1. Temporal Identity (Bias)
        self.temporal_embed = nn.Parameter(torch.zeros(1, latent_dim, num_frames, 1, 1))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # 2. Initial Projection (2048 -> 768)
        self.init_proj = nn.Sequential(
            nn.Conv3d(latent_dim, out_dims[0], kernel_size=1),
            LayerNormChFirst(out_dims[0])
        )
        
        # 3. Decoder Stage 1 (Low Res: 16x16) - Pure ConvNext
        self.stage1_blocks = nn.Sequential(*[
            ConvNext3DBlock(out_dims[0], drop_path=drop_rates[i]) 
            for i in range(depths[0])
        ])
        
        # 4. Upsample (16 -> 32)
        self.upsample = nn.Sequential(
            LayerNormChFirst(out_dims[0]),
            nn.ConvTranspose3d(out_dims[0], out_dims[1], kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 5. Decoder Stage 2 (High Res: 32x32) - Pure ConvNext
        # Adjust drop rates index
        start_idx = depths[0]
        self.stage2_blocks = nn.Sequential(*[
            ConvNext3DBlock(out_dims[1], drop_path=drop_rates[start_idx + i]) 
            for i in range(depths[1])
        ])
        
    def forward(self, x):
        # x: (B, 2048, 1, 16, 16)
        
        # Expand Time
        x = x.expand(-1, -1, self.num_frames, -1, -1)
        x = x + self.temporal_embed
        
        # Stage 1
        x = self.init_proj(x)
        x = self.stage1_blocks(x)
        
        # Upsample
        x = self.upsample(x)
        
        # Stage 2
        x = self.stage2_blocks(x)
        
        return x

# ============================================================
# 4. 主模型: ChronosNexusV1
# ============================================================

class ChronosNexusV1(nn.Module):
    """
    ChronosNexusV1
    
    Structure:
    1. Stem (ConvNext Blocks)
    2. Encoder (Hybrid: ConvNext + Attention when res <= 32)
    3. Latent Bottleneck
    4. Decoder (ConvNext Blocks Only)
    
    Total Blocks > 24.
    """
    def __init__(
        self, 
        in_channels=3, 
        latent_channels=2048, 
        output_channels=1536, 
        num_frames=8,
        # Dimensions for Stem, Enc1, Enc2, Enc3
        dims=[96, 192, 384, 768, 864, 1152], 
        # Depths: Stem=2, Enc1=3, Enc2=6, Enc3=9 -> Total Encoder path = 20 blocks
        # Decoder Depths: Dec1=3, Dec2=3 -> Total Decoder path = 6 blocks
        # Grand Total = 26 Blocks (>24)
        depths_stem=[2],
        depths_encoder=[3, 6, 3, 3, 3],
        depths_decoder=[3, 3, 2],
        drop_path_rate=0.2
    ):
        super().__init__()
        
        # Calculate total depth for drop path rule
        total_depth = sum(depths_stem) + sum(depths_encoder) + sum(depths_decoder)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0
        
        # --- 1. STEM ---
        # Input: 512x512 -> 128x128. Uses ConvNext Blocks.
        self.stem = ChronosStem(
            in_channels, dims[0], 
            depth=depths_stem[0], 
            drop_rates=dpr[cur:cur+depths_stem[0]]
        )
        cur += depths_stem[0]
        
        # --- 2. ENCODER ---
        self.encoder_stages = nn.ModuleList()
        
        # Enc Stage 1: 128->64. Res=64 (>32), so Pure Conv.
        self.encoder_stages.append(ChronosEncoder(
            dims[0], dims[1], 
            depth=depths_encoder[0], 
            drop_rates=dpr[cur:cur+depths_encoder[0]],
            stride=(2,2,2), current_res=64
        ))
        cur += depths_encoder[0]
        
        # Enc Stage 2: 64->32. Res=32 (<=32), so Hybrid (Conv+Attn).
        self.encoder_stages.append(ChronosEncoder(
            dims[1], dims[2], 
            depth=depths_encoder[1], 
            drop_rates=dpr[cur:cur+depths_encoder[1]],
            stride=(2,2,2), current_res=32
        ))
        cur += depths_encoder[1]
        
        # Enc Stage 3: 32->16. Res=16 (<=32), so Hybrid (Conv+Attn).
        self.encoder_stages.append(ChronosEncoder(
            dims[2], dims[3], 
            depth=depths_encoder[2], 
            drop_rates=dpr[cur:cur+depths_encoder[2]],
            stride=(2,2,2), current_res=16
        ))
        cur += depths_encoder[2]

        # Enc Stage 4: 16->16. Res=16 (<=32), so Hybrid (Conv+Attn).
        self.encoder_stages.append(ChronosEncoder(
            dims[3], dims[4], 
            depth=depths_encoder[3], 
            drop_rates=dpr[cur:cur+depths_encoder[3]],
            stride=(1,1,1), current_res=16
        ))
        cur += depths_encoder[3]

        # Enc Stage 5: 16->16. Res=16 (<=32), so Hybrid (Conv+Attn).
        self.encoder_stages.append(ChronosEncoder(
            dims[4], dims[5], 
            depth=depths_encoder[4], 
            drop_rates=dpr[cur:cur+depths_encoder[4]],
            stride=(1,1,1), current_res=16
        ))
        cur += depths_encoder[4]
        
        # Bottleneck Projection
        self.to_latent = nn.Sequential(
            LayerNormChFirst(dims[5]),
            nn.Conv3d(dims[5], latent_channels, kernel_size=1)
        )
        
        # --- 3. DECODER ---
        # Dec Stage 1 (16x16) & Dec Stage 2 (32x32)
        # Input Latent -> Output Features
        dec_dims = [dims[-1], dims[-2], dims[-3]] # [1536, 1024, 768]
        self.decoder = ChronosDecoder(
            latent_channels, dec_dims, 
            depths=depths_decoder, 
            drop_rates=dpr[cur:cur+sum(depths_decoder)],
            num_frames=num_frames
        )
        
        # Final Projection to 1536
        self.final_proj = nn.Sequential(
            LayerNormChFirst(dec_dims[1]),
            nn.Conv3d(dec_dims[1], output_channels, kernel_size=1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear, nn.ConvTranspose3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        # x: (B, 3, 8, 512, 512)
        x = self.stem(x) # -> (B, 96, 8, 128, 128)
        
        for stage in self.encoder_stages:
            x = stage(x)
        
        # x: (B, 768, 1, 16, 16)
        x = self.to_latent(x)
        # x: (B, 2048, 1, 16, 16)
        return x

    def decode(self, tokens):
        # tokens: (B, 2048, 1, 16, 16)
        x = self.decoder(tokens) # -> (B, 384, 8, 32, 32)
        x = self.final_proj(x)   # -> (B, 1536, 8, 32, 32)
        return x

    def forward(self, x):
        tokens = self.encode(x)
        features = self.decode(tokens)
        return features.permute(0, 2, 1, 3, 4) # (B, 8, 1536, 32, 32)

# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 配置验证
    print(f"Initializing ChronosNexusV1...")
    model = ChronosNexusV1(
        in_channels=3,
        latent_channels=2048,
        output_channels=1536,
        num_frames=8
    ).eval()
    
    # 1. 参数统计
    p_total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model Statistics]")
    print(f"  Name         : ChronosNexusV1")
    print(f"  Params       : {p_total:,} ({p_total*4/1024/1024:.1f} MB fp32)")
    
    # 2. 结构验证 (计算 Block 数量)
    total_blocks = 0
    # Stem
    total_blocks += len(model.stem.blocks)
    # Encoder
    for stage in model.encoder_stages:
        total_blocks += len(stage.blocks)
    # Decoder
    total_blocks += len(model.decoder.stage1_blocks)
    total_blocks += len(model.decoder.stage2_blocks)
    
    print(f"  Total Layers : {total_blocks} Blocks")
    print(f"  Constraint   : > 24 Blocks? {'YES' if total_blocks > 24 else 'NO'}")

    # 3. 形状流验证
    x = torch.randn(1, 3, 8, 512, 512)
    print("\n[Forward Pass Check]")
    print(f"  Input        : {list(x.shape)}")
    
    with torch.no_grad():
        # Encode
        latent = model.encode(x)
        print(f"  Latent       : {list(latent.shape)} (Target: [B, 2048, 1, 16, 16])")
        
        # Decode
        out = model.decode(latent)
        # Final Permute for output
        out_final = out.permute(0, 2, 1, 3, 4)
        print(f"  Output       : {list(out_final.shape)} (Target: [B, 8, 1536, 32, 32])")
    
    # 4. 混合策略验证 (检查 Attention 是否只出现在深层)
    print("\n[Hybrid Strategy Verification]")
    print("  Checking Encoder Stages for Attention Blocks...")
    for i, stage in enumerate(model.encoder_stages):
        has_attn = any(isinstance(b, SpatiotemporalAttentionBlock) for b in stage.blocks)
        res_map = {0: "64x64", 1: "32x32", 2: "16x16", 3: "16x16", 4: "16x16"}
        print(f"  Stage {i+1} ({res_map[i]}): Contains Attention? {has_attn}")
        if i == 0 and has_attn: print("  Error: Attention found in high res stage!")
        if i > 0 and not has_attn: print("  Warning: Expected Attention in low res stage!")

    print("\nChronosNexusV1 Ready.")
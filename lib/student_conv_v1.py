"""
Pure ConvNext3D Spatiotemporal Feature Encoder

Architecture Flow:
1. Input: 8 frames of 512x512x3 RGB images
2. Full ConvNext3D pathway: Local spatiotemporal feature extraction
3. Compressed Representation: 1x16x16x2048 spatiotemporal tokens
4. ConvNext3D Decoder: Expand to 8x32x32x1536 per-frame features
"""

import torch
import torch.nn as nn
from typing import Optional

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)

ACTS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'swish': Swish,
}

class LayerNorm3d(nn.Module):
    """3D Layer Normalization"""
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
        )
        self.norm = LayerNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        
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
        random_tensor = torch.floor(random_tensor)
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNext3DEncoder(nn.Module):
    """
    Pure ConvNext3D Encoder: Raw video frames -> Compressed spatiotemporal tokens
    
    Input: (B, 3, 8, 512, 512)
    Output: (B, 2048, 1, 16, 16)
    """
    def __init__(
        self,
        in_channels: int = 3,
        convnext_dims: list = [96, 192, 384, 768, 1024, 1536],
        convnext_depths: list = [3, 3, 9, 3, 3, 3],
        drop_path_rate: float = 0.1,
        output_channels: int = 2048,
    ):
        super().__init__()
        
        # Stem: 512x512 -> 128x128
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, convnext_dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            LayerNorm3d(convnext_dims[0]),
        )
        
        # ConvNext3D Stages with progressive downsampling
        # Stage 0: 128x128, dim=96
        # Stage 1: 64x64, dim=192, temporal downsample 8->4
        # Stage 2: 32x32, dim=384
        # Stage 3: 16x16, dim=768
        # Stage 4: 16x16, dim=1024 (spatial maintained, more depth)
        # Stage 5: 16x16, dim=1536 (final refinement)
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(convnext_depths))]
        cur = 0
        
        for i in range(6):
            # Downsampling layer
            if i > 0:
                # Temporal downsample at stage 1, spatial downsample at stages 1-3
                if i == 1:
                    downsample = nn.Sequential(
                        LayerNorm3d(convnext_dims[i-1]),
                        nn.Conv3d(
                            convnext_dims[i-1],
                            convnext_dims[i],
                            kernel_size=(2, 2, 2),
                            stride=(2, 2, 2),
                        ),
                    )
                elif i == 2 or i == 3:
                    downsample = nn.Sequential(
                        LayerNorm3d(convnext_dims[i-1]),
                        nn.Conv3d(
                            convnext_dims[i-1],
                            convnext_dims[i],
                            kernel_size=(1, 2, 2),
                            stride=(1, 2, 2),
                        ),
                    )
                else:  # i >= 4, no spatial downsample, just channel projection
                    downsample = nn.Sequential(
                        LayerNorm3d(convnext_dims[i-1]),
                        nn.Conv3d(convnext_dims[i-1], convnext_dims[i], kernel_size=1),
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
                        kernel_size=7 if i < 4 else 5,  # Smaller kernel for later stages
                    )
                )
            cur += convnext_depths[i]
            
            stage = nn.Sequential(downsample, *blocks)
            self.stages.append(stage)
        
        # After stages: (B, 1536, 4, 16, 16)
        
        # Global temporal pooling + projection to output channels
        self.head = nn.Sequential(
            LayerNorm3d(convnext_dims[-1]),
            nn.Conv3d(convnext_dims[-1], output_channels, kernel_size=1),
            LayerNorm3d(output_channels),
        )
        
        # Temporal pooling to compress 4 frames -> 1
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 8, 512, 512)
        Returns:
            (B, 2048, 1, 16, 16)
        """
        x = self.stem(x)  # (B, 96, 8, 128, 128)
        
        for stage in self.stages:
            x = stage(x)
        # After stages: (B, 1536, 4, 16, 16)
        
        x = self.head(x)  # (B, 2048, 4, 16, 16)
        x = self.temporal_pool(x)  # (B, 2048, 1, 16, 16)
        
        return x


class ConvNext3DDecoder(nn.Module):
    """
    Pure ConvNext3D Decoder: Compressed tokens -> Per-frame features
    
    Input: (B, 2048, 1, 16, 16)
    Output: (B, 1536, 8, 32, 32)
    """
    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 1536,
        num_frames: int = 8,
        decoder_dims: list = [1536, 1024, 768],
        decoder_depths: list = [3, 3, 2],
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        
        # Temporal expansion with learnable embeddings
        self.temporal_embed = nn.Parameter(torch.randn(1, in_channels, num_frames, 1, 1))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, decoder_dims[0], kernel_size=1),
            LayerNorm3d(decoder_dims[0]),
            nn.GELU(),
        )
        
        # Decoder stages
        # Stage 0: 8x16x16, dim=1536
        # Stage 1: 8x16x16, dim=1024 (refinement)
        # Stage 2: 8x32x32, dim=768 (spatial upsample)
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]
        cur = 0
        
        for i in range(len(decoder_dims)):
            # Blocks for this stage
            blocks = []
            dim = decoder_dims[i]
            
            for j in range(decoder_depths[i]):
                blocks.append(
                    ConvNext3DBlock(
                        dim=dim,
                        drop_path=dp_rates[cur + j],
                        kernel_size=5,
                    )
                )
            cur += decoder_depths[i]
            
            # Upsample or channel projection
            if i < len(decoder_dims) - 1:
                # Channel projection for stage 0->1
                if i == 0:
                    post_process = nn.Sequential(
                        LayerNorm3d(dim),
                        nn.Conv3d(dim, decoder_dims[i+1], kernel_size=1),
                    )
                # Spatial upsample for stage 1->2
                else:
                    post_process = nn.Sequential(
                        LayerNorm3d(dim),
                        nn.ConvTranspose3d(
                            dim, decoder_dims[i+1],
                            kernel_size=(1, 4, 4),
                            stride=(1, 2, 2),
                            padding=(0, 1, 1),
                        ),
                    )
            else:
                post_process = nn.Identity()
            
            stage = nn.Sequential(*blocks, post_process)
            self.stages.append(stage)
        
        # Final output projection
        self.output_proj = nn.Sequential(
            LayerNorm3d(decoder_dims[-1]),
            nn.Conv3d(decoder_dims[-1], out_channels, kernel_size=1),
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
            x: (B, 2048, 1, 16, 16)
        Returns:
            (B, 1536, 8, 32, 32)
        """
        # Temporal expansion
        x = x.expand(-1, -1, self.num_frames, -1, -1)  # (B, 2048, 8, 16, 16)
        x = x + self.temporal_embed
        
        # Initial projection
        x = self.proj(x)  # (B, 1536, 8, 16, 16)
        
        # Decoder stages
        for stage in self.stages:
            x = stage(x)
        # After stages: (B, 768, 8, 32, 32)
        
        # Final projection
        x = self.output_proj(x)  # (B, 1536, 8, 32, 32)
        
        return x


class PureConvNext3DModel(nn.Module):
    """
    Complete Pure ConvNext3D Model for spatiotemporal feature encoding
    
    Pipeline:
    Raw Images (8x512x512x3) -> Compressed Tokens (1x16x16x2048) 
                               -> Distillation Features (8x32x32x1536)
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 2048,
        output_channels: int = 1536,
        num_frames: int = 8,
        encoder_dims: list = [96, 192, 384, 768, 1024, 1536],
        encoder_depths: list = [3, 3, 6, 3, 2, 2],
        decoder_dims: list = [1536, 1024, 768],
        decoder_depths: list = [2, 2, 2],
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = ConvNext3DEncoder(
            in_channels=in_channels,
            convnext_dims=encoder_dims,
            convnext_depths=encoder_depths,
            drop_path_rate=drop_path_rate,
            output_channels=latent_channels,
        )
        
        self.decoder = ConvNext3DDecoder(
            in_channels=latent_channels,
            out_channels=output_channels,
            num_frames=num_frames,
            decoder_dims=decoder_dims,
            decoder_depths=decoder_depths,
            drop_path_rate=drop_path_rate,
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 8, 512, 512)
        Returns:
            features: (B, 8, 1536, 32, 32)
        """
        tokens = self.encoder(x)  # (B, 2048, 1, 16, 16)
        features = self.decoder(tokens)  # (B, 1536, 8, 32, 32)
        
        # Permute to (B, T, C, H, W)
        features = features.permute(0, 2, 1, 3, 4)  # (B, 8, 1536, 32, 32)
        
        return features
    
    def encode(self, x):
        """Encode video frames to compressed tokens"""
        return self.encoder(x)
    
    def decode(self, tokens):
        """Decode tokens to per-frame features"""
        return self.decoder(tokens)


# === Example Usage ===
if __name__ == "__main__":
    model = PureConvNext3DModel(
        in_channels=3,
        latent_channels=2048,
        output_channels=1536,
        num_frames=8,
        encoder_dims=[96, 192, 384, 768, 1024, 1536],
        encoder_depths=[3, 3, 6, 3, 2, 2],
        decoder_dims=[1536, 1024, 768],
        decoder_depths=[2, 2, 2],
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
    tokens = model.encode(x)
    print(f"\nCompressed tokens shape: {tokens.shape}")
    print(f"  - Expected: (B, C, T, H, W) = (2, 2048, 1, 16, 16)")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\n=== Pure ConvNext3D Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  - Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    print(f"\n=== Architecture Summary ===")
    print("Pure ConvNext3D architecture - all convolution, no transformers")
    print("Advantages:")
    print("  - More efficient for dense spatial features")
    print("  - Better inductive bias for local patterns")
    print("  - Lower memory footprint")
    print("  - Easier to optimize")

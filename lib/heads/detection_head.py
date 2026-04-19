"""
Object detection head (RT-DETR style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class CrossScaleFeatureFusion(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.upsample_top = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.downsample_bottom = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == 3
        f32, f16, f8 = features
        f32_up = self.upsample_top(f32)
        f16_fused = f16 + f32_up
        f16_up = self.upsample_top(f16_fused)
        f8_fused = f8 + f16_up
        f8_down = self.downsample_bottom(f8_fused)
        f16_enhanced = f16_fused + f8_down
        f16_down = self.downsample_bottom(f16_enhanced)
        f32_enhanced = f32 + f16_down
        f8_final = self.fusion(torch.cat([f8_fused, f8_down], dim=1))
        f16_final = self.fusion(torch.cat([f16_enhanced, f32_up], dim=1))
        f32_final = self.fusion(torch.cat([f32_enhanced, f16_down], dim=1))
        return [f8_final, f16_final, f32_final]


class DeformableAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.num_points = num_points
        self.im2col_step = 64
        
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
        B, N, C = query.shape
        num_heads = self.num_heads
        num_points = self.num_points
        sampling_offset = self.sampling_offsets(query).view(B, N, num_heads, num_points, 2)
        attention_weights = self.attention_weights(query).view(B, N, num_heads, num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        value = self.value_proj(input_flatten)
        output = torch.zeros_like(query)
        attn_output, _ = nn.functional.multi_head_attention_forward(
            query, value, self.embed_dim, num_heads, torch.empty([]), torch.empty([]),
            False, 0.0, self.output_proj.weight, self.output_proj.bias, training=self.training,
        )
        return attn_output


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.object_queries = nn.Parameter(torch.zeros(1, 300, d_model))
        nn.init.trunc_normal_(self.object_queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, memory, target_feature=None):
        B = memory.shape[0]
        _, C, H, W = memory.shape
        memory_flat = memory.flatten(2).permute(0, 2, 1)
        queries = self.object_queries.expand(B, -1, -1)
        if target_feature is not None:
            queries = queries + target_feature
        output = self.transformer_decoder(queries, memory_flat)
        return self.output_proj(output)


class DetectionHead(nn.Module):
    def __init__(self, num_classes: int = 4, embed_dim: int = 256, num_decoder_layers: int = 6, num_heads: int = 8, num_queries: int = 300):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.feature_fusion = CrossScaleFeatureFusion(channels=embed_dim)
        self.feat_proj = nn.ModuleDict({
            '8': nn.Sequential(nn.Conv2d(1536, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU()),
            '16': nn.Sequential(nn.Conv2d(1536, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU()),
            '32': nn.Sequential(nn.Conv2d(1536, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU()),
        })
        self.deformable_attn = nn.ModuleList([DeformableAttention(embed_dim, num_heads, num_points=4) for _ in range(3)])
        self.decoder = TransformerDecoder(d_model=embed_dim, nhead=num_heads, num_decoder_layers=num_decoder_layers)
        self.bbox_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 4))
        self.class_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_classes + 1))
        self.reference_points = nn.Linear(embed_dim, 2)
        
    def forward(self, multi_scale_features, temporal_feature=None):
        B = multi_scale_features[0].shape[0]
        fused_features = self.feature_fusion(multi_scale_features)
        main_feat = fused_features[0]
        main_feat_flat = main_feat.flatten(2).permute(0, 2, 1)
        reference = self.reference_points(self.decoder.object_queries)
        reference = reference.expand(B, -1, -1).sigmoid()
        deform_feat = main_feat_flat
        for da in self.deformable_attn:
            deform_feat = deform_feat + da(deform_feat, reference[:, :, :2], main_feat_flat, [(main_feat.shape[2], main_feat.shape[3])])
        decoder_output = self.decoder(main_feat, temporal_feature)
        outputs_coord = self.bbox_embed(decoder_output).sigmoid()
        outputs_class = self.class_embed(decoder_output)
        return {'boxes': outputs_coord, 'scores': outputs_class, 'features': decoder_output}

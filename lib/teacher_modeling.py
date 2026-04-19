import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

def count_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters for {model_name}: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("\n")


class TeacherEnsemble(nn.Module):
    def __init__(self, dinov3_path, siglip2_path, align_type="rmsnorm"):
        super().__init__()
        # 注意：Processor 现在移到了 Dataset 中，这里只加载模型
        self.align_type = align_type # layernorm, rmsnorm, or None
        
        # 1. 加载模型
        self.dinov3 = AutoModel.from_pretrained(dinov3_path)
        self.siglip2 = AutoModel.from_pretrained(siglip2_path).vision_model

        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False

        count_parameters(self.dinov3, "DINOv3")
        count_parameters(self.siglip2, "SigLIP2")

    @torch.no_grad()
    def forward(self, dino_pixel_values, siglip_pixel_values):
        """
        Args:
            dino_pixel_values:  (B, T, 3, 512, 512) DINO 预处理后的 Tensor
            siglip_pixel_values: (B, T, 3, 512, 512) SigLIP 预处理后的 Tensor
        """
        B, T, C, H, W = dino_pixel_values.shape
        # Flatten time: (B*T, 3, 512, 512)
        flat_dino_imgs = dino_pixel_values.view(B * T, C, H, W)
        flat_siglip_imgs = siglip_pixel_values.view(B * T, C, H, W)
        
        # 1. DINOv3 Forward
        out_dino = self.dinov3(pixel_values=flat_dino_imgs).last_hidden_state
        
        seq_len = (512 // 16) ** 2
        feat_dino = out_dino[:, -seq_len:, :] # (B*T, 1024, 768)

        # 2. SigLIP2 Forward
        out_siglip = self.siglip2(pixel_values=flat_siglip_imgs).last_hidden_state
        
        if out_siglip.shape[1] > seq_len:
             feat_siglip = out_siglip[:, -seq_len:, :]
        else:
             feat_siglip = out_siglip

        # 3. Concatenate
        if self.align_type == "layernorm":
            # print('using layernorm for feature alignment')
            feat_dino = F.layer_norm(feat_dino, normalized_shape=[feat_dino.shape[-1]])
            feat_siglip = F.layer_norm(feat_siglip, normalized_shape=[feat_siglip.shape[-1]])
        elif self.align_type == "rmsnorm":
            # print('using rmsnorm for feature alignment')
            feat_dino = F.rms_norm(feat_dino, normalized_shape=[feat_dino.shape[-1]])
            feat_siglip = F.rms_norm(feat_siglip, normalized_shape=[feat_siglip.shape[-1]])
        else:
            pass  # 不进行任何对齐处理
        combined = torch.cat([feat_dino, feat_siglip], dim=-1) # (B*T, 1024, 1536)
        
        # 4. Reshape
        combined = combined.view(B, T, 32, 32, 1536)
        combined = combined.permute(0, 1, 4, 2, 3)  # BK, T, 1536, 32, 32

        return combined
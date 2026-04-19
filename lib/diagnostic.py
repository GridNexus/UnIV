import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

class FeatureDiagnostic:
    def __init__(self, save_dir="analysis_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.stats = {"dino": [], "siglip": []}

    @torch.no_grad()
    def collect(self, feat_dino, feat_siglip):
        """
        输入形状: (B*T, N, C)
        """
        # 转为 float32 避免计算精度问题
        dino = feat_dino.detach().float()
        siglip = feat_siglip.detach().float()

        # 1. 计算 L2 Norm (每个 token 的模长)
        norm_dino = torch.norm(dino, p=2, dim=-1).cpu().numpy().flatten()
        norm_siglip = torch.norm(siglip, p=2, dim=-1).cpu().numpy().flatten()

        dino = dino.cpu().numpy().reshape(-1, dino.shape[-1]) # (B*T*N, C)
        siglip = siglip.cpu().numpy().reshape(-1, siglip.shape[-1]) # (B*T*N, C)

        # 2. 计算激活值的绝对值分布 (Sparsity/Saturation)
        act_dino = dino.abs().cpu().numpy().flatten()
        act_siglip = siglip.abs().cpu().numpy().flatten()

        self.stats["dino_norm"] = self.stats.get("dino_norm", []) + [norm_dino]
        self.stats["siglip_norm"] = self.stats.get("siglip_norm", []) + [norm_siglip]
        self.stats["dino_act"] = self.stats.get("dino_act", []) + [act_dino]
        self.stats["siglip_act"] = self.stats.get("siglip_act", []) + [act_siglip]

    def run_analysis(self):
        print("Starting deep statistical analysis...")
        
        # 合并数据
        dino_norms = np.concatenate(self.stats["dino_norm"])
        siglip_norms = np.concatenate(self.stats["siglip_norm"])
        dino_acts = np.concatenate(self.stats["dino_act"])
        siglip_acts = np.concatenate(self.stats["siglip_act"])

        plt.figure(figsize=(15, 10))

        # --- 图 1: L2 Norm 分布对比 ---
        plt.subplot(2, 2, 1)
        sns.histplot(dino_norms, color="skyblue", label="DINOv3", kde=True, stat="probability")
        sns.histplot(siglip_norms, color="salmon", label="SigLIP2", kde=True, stat="probability")
        plt.title("L2 Norm Distribution (Token-wise)")
        plt.legend()
        print('finish 1')

        # --- 图 2: 值 (Raw Values) 分布 ---
        plt.subplot(2, 2, 2)
        sns.histplot(dino_acts, color="skyblue", label="DINOv3", kde=True, stat="probability")
        sns.histplot(siglip_acts, color="salmon", label="SigLIP2", kde=True, stat="probability")
        plt.title("Value Distribution (Token-wise)")
        plt.legend()
        print('finish 2')

        # --- 图 3: Norm 的方差随样本的变化 ---
        plt.subplot(2, 2, 3)
        plt.boxplot([dino_norms, siglip_norms], labels=['DINOv3', 'SigLIP2'])
        plt.title("Norm Stability (Outliers Check)")
        print('finish 3')

        # --- 图 4: 特征余弦相似度热图 (取一个样本展示空间结构差异) ---
        # 这一部分建议在 collect 里单独采样一张图做可视化
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "feature_stats.png"))
        print(f"Stats plot saved to {self.save_dir}")

        # 打印数值报告
        print("\n--- Numerical Report ---")
        for name, data in [("DINOv3", dino_norms), ("SigLIP2", siglip_norms)]:
            print(f"{name}: Mean={np.mean(data):.4f}, Std={np.std(data):.4f}, Max={np.max(data):.4f}")

    @torch.no_grad()
    def visualize_spatial_heatmaps(self, feat_dino: torch.Tensor, feat_siglip: torch.Tensor, B_idx=0):
        """
        可视化空间注意力/特征强度: (B*T, 1024, C) -> (32, 32)
        """
        feat_dino = feat_dino.to(torch.float32)
        feat_siglip = feat_siglip.to(torch.float32)
        # 取第 B_idx 个样本，取其特征的 L2 Norm 作为热力图
        # 形状为 (1024,) -> (32, 32)
        d_map = torch.norm(feat_dino[B_idx], dim=-1).view(32, 32).cpu().numpy()
        s_map = torch.norm(feat_siglip[B_idx], dim=-1).view(32, 32).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im1 = axes[0].imshow(d_map, cmap='viridis')
        axes[0].set_title("DINOv3 Feature Intensity")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(s_map, cmap='magma')
        axes[1].set_title("SigLIP2 Feature Intensity")
        plt.colorbar(im2, ax=axes[1])

        plt.savefig(os.path.join(self.save_dir, "spatial_intensity.png"))
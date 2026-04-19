import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from transformers import AutoModel, AutoProcessor
import os

# ================= 配置区域 =================
# 请确保路径正确，并且显存足够（双Teacher + Student需要较大显存）
DINOV3_PATH = os.getenv("DINOV3_PATH", "/path/to/dinov3-vitb16-pretrain-lvd1689m")
SIGLIP2_PATH = os.getenv("SIGLIP2_PATH", "/path/to/siglip2-base-patch16-512")
VIDEO_PATH = os.getenv("VIDEO_PATH", "example.mp4") # 替换为实际视频路径

BATCH_SIZE = 2             # 显存不足可调小
LEARNING_RATE = 1e-4
EPOCHS = 100
NUM_FRAMES = 16
FRAME_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 优化的数据集 =================
class VideoDataset(Dataset):
    def __init__(self, video_path, num_frames=16, frame_size=512, sample_interval=5):
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_interval = sample_interval
        
        # 预先检查视频帧数，避免运行时崩溃
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            print("Warning: Video path not found, using dummy data mode.")
            self.total_frames = 1000

    def __len__(self):
        # 假设每个样本之间有重叠，或者随机采样
        return max(1, (self.total_frames - self.num_frames * self.sample_interval) // 10)

    def __getitem__(self, idx):
        # 实际场景中建议使用 Decord 或 PyAV，OpenCV seek 较慢
        # 为了保持依赖简单，这里优化了 OpenCV 的读取逻辑
        if not os.path.exists(str(self.video_path)):
             # 返回假数据用于测试流程
            return torch.randn(self.num_frames, 3, self.frame_size, self.frame_size)

        cap = cv2.VideoCapture(self.video_path)
        # 随机起始位置，增强鲁棒性
        start_frame = np.random.randint(0, max(1, self.total_frames - self.num_frames * self.sample_interval))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 简单的丢帧策略模拟间隔采样
            for _ in range(self.sample_interval - 1):
                cap.read()
                
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            # 标准的 ImageNet 归一化 (DINO/SigLIP 通常需要)
            frames.append(frame)
        
        cap.release()
        
        # Padding if video ends early
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros(3, self.frame_size, self.frame_size))
            
        return torch.stack(frames) # (T, C, H, W)

# ================= 2. 教师模型包装器 =================
class TeacherEnsemble(nn.Module):
    def __init__(self, dinov3_path, siglip2_path):
        super().__init__()
        print(f"Loading Teachers...\nDINO: {dinov3_path}\nSigLIP: {siglip2_path}")
        self.dinov3 = AutoModel.from_pretrained(dinov3_path).eval()
        self.siglip2 = AutoModel.from_pretrained(siglip2_path).eval()
        
        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        # x: (B, T, C, H, W) -> 需要展平为 (B*T, C, H, W) 输入 transformer
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        with torch.no_grad():
            # 获取 DINOv3 特征 (假设输出 batch x seq_len x dim)
            # 注意：需去除 CLS token (索引 0)，保留 patch tokens
            # Patch size 16, img 512 -> 32*32 = 1024 patches
            out_dino = self.dinov3(pixel_values=x_flat).last_hidden_state
            feat_dino = out_dino[:, 1:, :] # (B*T, 1024, 768)
            
            # 获取 SigLIP2 特征
            out_siglip = self.siglip2(pixel_values=x_flat).last_hidden_state
            feat_siglip = out_siglip[:, 1:, :] # (B*T, 1024, 768)
            
            # 拼接特征
            combined = torch.cat([feat_dino, feat_siglip], dim=-1) # (B*T, 1024, 1536)
            
        # 恢复时序维度
        # 输出: (B, 16, 1024, 1536)
        return combined.view(B, T, 1024, -1)

# ================= 3. 改进的学生网络 =================
class StudentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器: 利用 Conv3D 更好地提取时空特征
        # 输入: (B, 3, 16, 512, 512)
        # 目标: 压缩为 (B, 2048, 1024) -> 其中 1024 是空间 token 数
        
        self.encoder_cnn = nn.Sequential(
            # 下采样空间，压缩时间
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3)), # -> (16, 128, 128)
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)), # -> (16, 64, 64)
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # -> (8, 32, 32)
            nn.BatchNorm3d(256), nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)), # -> (4, 32, 32)
            nn.BatchNorm3d(512), nn.ReLU(),
            # 最终压缩时间到 1，通道增加
            nn.Conv3d(512, 2048, kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=0),   # -> (1, 32, 32)
        )
        
        # 此时形状: (B, 2048, 1, 32, 32) -> Flatten -> (B, 2048, 1024)
        
        # 解码器: 需要将 (1, 1024, 2048) 扩展回 (16, 1024, 1536)
        # 使用 Linear 投影维度，并使用可学习的 Query 来恢复时间轴
        self.feature_proj = nn.Linear(2048, 1536)
        
        # 简单的时序扩展器：将 T=1 广播到 T=16 并加上时序位置编码
        self.temporal_embed = nn.Parameter(torch.zeros(1, 16, 1, 1536))
        
        # 一个轻量级的 MLP Mixer 或者 Conv1D 来处理时序平滑
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(1536, 1536, kernel_size=3, padding=1, groups=16),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, 16, 3, 512, 512) -> Conv3d 需要 (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4) 
        
        # Encoder
        z = self.encoder_cnn(x) # (B, 2048, 1, 32, 32)
        B, C, T, H, W = z.shape
        
        # Reshape to (B, 1, H*W, C) = (B, 1, 1024, 2048)
        z = z.view(B, C, T, -1).permute(0, 2, 3, 1) 
        
        # Bottleneck Z 特征 (B, 1, 1024, 2048)
        latent_z = z
        
        # Decoder 流程
        # 1. 投影特征维度: 2048 -> 1536
        f_base = self.feature_proj(latent_z) # (B, 1, 1024, 1536)
        
        # 2. 扩展时间维度 T=1 -> T=16
        f2 = f_base.repeat(1, 16, 1, 1) # (B, 16, 1024, 1536)
        
        # 3. 加上时序位置编码 (广播)
        f2 = f2 + self.temporal_embed
        
        # 4. 时序混合 (Reshape for Conv1d: B*1024, C, T)
        # 这样处理是为了让网络学习每帧之间的变化，而不是简单复制
        B, T, N, D = f2.shape
        f2_reshaped = f2.permute(0, 2, 3, 1).reshape(B * N, D, T)
        f2_refined = self.temporal_mixer(f2_reshaped)
        
        # 还原形状
        f2_out = f2_refined.reshape(B, N, D, T).permute(0, 3, 1, 2) # (B, 16, 1024, 1536)
        
        return f2_out

# ================= 4. 高级蒸馏损失函数 =================
class AdvancedDistillationLoss(nn.Module):
    def __init__(self, alpha_mse=1.0, beta_cos=1.0):
        super().__init__()
        self.alpha = alpha_mse
        self.beta = beta_cos
        self.mse = nn.MSELoss()
    
    def forward(self, student_feat, teacher_feat):
        """
        student_feat: (B, 16, 1024, 1536)
        teacher_feat: (B, 16, 1024, 1536)
        """
        # 1. 基础重建损失 (MSE) - 确保数值幅度一致
        loss_mse = self.mse(student_feat, teacher_feat)
        
        # 2. 语义方向损失 (Cosine Similarity) - 确保特征含义一致
        # 将特征展平为 (N, D) 计算
        s_flat = student_feat.reshape(-1, 1536)
        t_flat = teacher_feat.reshape(-1, 1536)
        
        # CosineEmbeddingLoss target 为 1 表示希望相似度为 1
        # 但手动计算更灵活: 1 - mean(cosine_sim)
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
        loss_cos = 1.0 - cos_sim.mean()
        
        total_loss = self.alpha * loss_mse + self.beta * loss_cos
        return total_loss, loss_mse.item(), loss_cos.item()

# ================= 5. 训练循环 =================
def train():
    # 初始化
    teachers = TeacherEnsemble(DINOV3_PATH, SIGLIP2_PATH).to(DEVICE)
    student = StudentNetwork().to(DEVICE)
    
    # 冻结教师显存优化
    teachers.eval()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = AdvancedDistillationLoss(alpha_mse=20.0, beta_cos=1.0) # MSE权重通常需要调大因为数值较小
    
    dataset = VideoDataset(VIDEO_PATH, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # 混合精度训练 (Modern Standard)
    scaler = torch.cuda.amp.GradScaler()
    
    print("Start Training...")
    
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(DEVICE) # (B, T, C, H, W)
            
            # 使用混合精度
            with torch.cuda.amp.autocast():
                # 1. 提取教师特征 (无需计算梯度)
                with torch.no_grad():
                    teacher_feats = teachers(frames) # (B, 16, 1024, 1536)
                
                # 2. 学生网络前向
                student_feats = student(frames) # (B, 16, 1024, 1536)
                
                # 3. 计算损失
                loss, mse_val, cos_val = loss_fn(student_feats, teacher_feats)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (MSE: {mse_val:.4f}, Cos: {cos_val:.4f})")
        
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch} Done. Avg Loss: {avg_loss:.4f} ===")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save(student.state_dict(), f"student_ckpt_ep{epoch}.pth")

if __name__ == "__main__":
    train()
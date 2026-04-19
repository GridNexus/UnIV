import torch
from torch.utils.data import Dataset
import random
from decord import VideoReader, cpu
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

class VideoDynamicDataset(Dataset):
    def __init__(self, 
                 txt_path, 
                 dinov3_path=None, # 兼容旧参数，但现在 dataset 内不直接使用 AutoImageProcessor 了
                 siglip2_path=None, # 兼容旧参数，但现在 dataset 内不直接使用 AutoImageProcessor 了
                 num_frames=8, 
                 clips_per_video=4, 
                 dataset_sample_rate=1.0):
        with open(txt_path, 'r') as f:
            self.video_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        if dataset_sample_rate < 1.0:
            sample_size = max(1, int(len(self.video_paths) * dataset_sample_rate))
            self.video_paths = random.sample(self.video_paths, sample_size)
            print(f"Sampled {len(self.video_paths)} videos for training.")

        self.num_frames = num_frames
        self.clips_per_video = clips_per_video
        self.valid_strides = [4, 8, 16, 24]
        
        # --- 预定义 Transform (移除 AutoImageProcessor) ---
        # 两个模型都要求 512x512，我们合并 Resize 操作
        self.target_size = (512, 512)
        
        # DINOv3 (ImageNet Mean/Std)
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # SigLIP (0.5 Mean/Std)
        self.siglip_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.siglip_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

        # 构造 Resize 变换 pipeline (支持 Batch)
        # InterpolationMode.BILINEAR 是默认值 (对应 config 中的 resample: 2)
        self.common_transform = v2.Compose([
            v2.Resize(self.target_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True), # 归一化到 [0, 1] 也就是 rescale_factor
        ])

    def __len__(self):
        return len(self.video_paths)

    def _get_clip_indices(self, total_frames):
        all_indices = []
        for _ in range(self.clips_per_video):
            possible_strides = [s for s in self.valid_strides if (self.num_frames - 1) * s + 1 <= total_frames]
            stride = 1 if not possible_strides else random.choice(possible_strides)
            max_start = total_frames - (self.num_frames - 1) * stride - 1
            start_idx = random.randint(0, max(0, max_start))
            clip_indices = [start_idx + i * stride for i in range(self.num_frames)]
            # 边界保护
            clip_indices = [min(idx, total_frames - 1) for idx in clip_indices]
            all_indices.extend(clip_indices)
        return all_indices

    def __getitem__(self, idx):
        path = self.video_paths[idx]

        try:
            # --- 优化点 1: 只初始化一次 VideoReader ---
            vr = VideoReader(path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames <= 0: 
                return self.dummy_data()

            # 生成索引
            target_indices = self._get_clip_indices(total_frames)
            
            # --- 优化点 2: 批量读取 ---
            # decord 返回 (N, H, W, C) 的 array (uint8)
            buffer = vr.get_batch(target_indices).asnumpy()
            
            # 释放 decord 资源（虽有 GC，但显式删除有时对多进程更友好）
            del vr 

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.dummy_data()

        # --- 优化点 3: 快速预处理流水线 ---
        
        # Step A: Numpy (N, H, W, C) -> Tensor (N, C, H, W)
        # 此时还是 uint8，速度快
        video_tensor = torch.from_numpy(buffer).permute(0, 3, 1, 2).contiguous()

        # Step B: 统一 Resize 和 Rescale (/255)
        # 这一步最耗时，现在只做一次。batch 操作比处理 list 快得多
        # common_transform 输出 float32, 范围 [0, 1]
        video_tensor = self.common_transform(video_tensor)

        # Step C: 分支 Normalize (原地操作或广播减法，速度极快)
        # DINOv3 分支
        dino_pixels = (video_tensor - self.dino_mean) / self.dino_std
        
        # SigLIP 分支
        siglip_pixels = (video_tensor - self.siglip_mean) / self.siglip_std

        # --- Reshape ---
        K, T = self.clips_per_video, self.num_frames
        
        # 检查帧数是否对齐 (极少数情况 decord 可能返回少于请求的帧)
        if dino_pixels.shape[0] != K * T:
            return self.dummy_data()

        dino_pixels = dino_pixels.view(K, T, 3, 512, 512)
        siglip_pixels = siglip_pixels.view(K, T, 3, 512, 512)
        
        # --- 优化点 4: 避免 Clone ---
        # 如果 student input 就是 siglip 的输入，直接引用即可 (除非后续在 dataset 外面会修改它)
        # 如果必须复制，clone() 是对的，但如果只是用来作为网络输入，通常不用 clone
        student_input = siglip_pixels 

        return {
            "dino": dino_pixels, 
            "siglip": siglip_pixels, 
            "student": student_input
        }

    def dummy_data(self):
        K, T = self.clips_per_video, self.num_frames
        dummy = torch.zeros(K, T, 3, 512, 512)
        return {"dino": dummy, "siglip": dummy, "student": dummy}
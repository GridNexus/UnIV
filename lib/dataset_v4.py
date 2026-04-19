import torch
from torch.utils.data import Dataset
import random
from decord import VideoReader, cpu
import numpy as np
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
import os

class BalancedVideoDataset(Dataset):
    def __init__(self, 
                 txt_path, 
                 num_frames=8, 
                 clips_per_video=4, 
                 dataset_sample_rate=1.0,
                 seed=0):
        
        self.num_frames = num_frames
        self.clips_per_video = clips_per_video # K=4
        self.valid_strides = [4, 8, 16, 24]
        self.target_size = (512, 512)
        
        # 定义扩展名
        self.IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.webm', '.mov')

        # --- 1. 读取并拆分数据 ---
        with open(txt_path, 'r') as f:
            raw_paths = [line.strip() for line in f.readlines() if line.strip()]
        raw_paths.sort() 

        # 采样 (可选)
        if dataset_sample_rate < 1.0:
            rng = random.Random(seed) 
            sample_size = max(1, int(len(raw_paths) * dataset_sample_rate))
            raw_paths = rng.sample(raw_paths, sample_size)
            print(f"Sampled {len(raw_paths)} raw items.")

        # 分离图片和视频
        self.video_paths = [] 
        self.image_paths = [] 

        for path in raw_paths:
            if path.lower().endswith(self.IMAGE_EXTS):
                self.image_paths.append(path)
            else:
                self.video_paths.append(path)

        # --- 2. 核心逻辑：定义数据集长度 ---
        # 计算有多少组图片 (每组 clips_per_video 张)
        # 例如 120,000 张图 / 4 = 30,000 个 Image Samples
        self.num_image_samples = len(self.image_paths) // self.clips_per_video
        
        # 强制让视频样本数 = 图片样本数 (1:1 平衡)
        # 我们会在 __getitem__ 里用取模的方式循环使用真实的视频文件
        self.num_video_samples = self.num_image_samples 
        
        # 真实的视频文件数量
        self.num_real_videos = len(self.video_paths)

        print(f"Dataset Summary (Balanced Strategy):")
        print(f"  - Real Video Files: {self.num_real_videos}")
        print(f"  - Real Total Images: {len(self.image_paths)}")
        print(f"  - Image Samples (Groups): {self.num_image_samples}")
        print(f"  - Video Samples (Virtual): {self.num_video_samples} (Upsampled from {self.num_real_videos} files)")
        print(f"  - Total Dataset Length: {self.num_image_samples + self.num_video_samples}")

        # --- 3. 定义 Transform ---
        
        # A. 基础处理：Resize + 转浮点
        self.resize_transform = v2.Compose([
            v2.Resize(self.target_size, antialias=True),
            # 此时还是 uint8 [0, 255]
        ])

        # B. 数据增强 (仅针对 uint8 图片)
        # 对图片进行较强的增强，对视频进行较弱的增强(或一致性增强)
        self.aug_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

        # C. 转 Dtype (放在增强之后，Normalize 之前)
        self.to_dtype = v2.ToDtype(torch.bfloat16, scale=True) # [0,1]

        # Normalization consts
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.siglip_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.siglip_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    def __len__(self):
        # 总长度 = 图片组数 + 虚拟视频组数 (1:1)
        return self.num_image_samples + self.num_video_samples

    def _get_clip_indices(self, total_frames):
        """完全随机采样，确保每次调用都能拿到视频的不同片段"""
        all_indices = []
        for _ in range(self.clips_per_video):
            # 随机选择步长
            possible_strides = [s for s in self.valid_strides if (self.num_frames - 1) * s + 1 <= total_frames]
            stride = 1 if not possible_strides else random.choice(possible_strides)
            
            # 随机选择起始点 (这是实现无限采样的关键)
            max_start = total_frames - (self.num_frames - 1) * stride - 1
            start_idx = random.randint(0, max(0, max_start))
            
            clip_indices = [start_idx + i * stride for i in range(self.num_frames)]
            clip_indices = [min(idx, total_frames - 1) for idx in clip_indices]
            all_indices.extend(clip_indices)
        return all_indices

    def _process_static_images(self, path_list):
        """
        处理图片组：
        1. 读取
        2. 数据增强 (Aug) -> 增加图片多样性
        3. Resize
        4. 复制成视频序列 (Expand)
        """
        # 读取 -> List of (3, H_raw, W_raw)
        tensors = []
        for p in path_list:
            try:
                img = read_image(p, mode=ImageReadMode.RGB)
            except:
                img = torch.zeros(3, 512, 512, dtype=torch.uint8)
            tensors.append(self.resize_transform(img))
        
        # 组成 Batch 进行处理方便 (K, 3, H, W)
        # 注意：这里 K=4，每张图是独立的，所以 Transform 会独立随机应用（例如有的翻转有的不翻转）
        batch_img = torch.stack(tensors) 
        
        # Augment + Resize
        batch_img = self.aug_transform(batch_img)
        batch_img = self.to_dtype(batch_img) # -> float [0,1]

        # Expand Time: (K, C, H, W) -> (K, T, C, H, W)
        # 这里的 T 维度是完全复制的，因为这是静态图
        K, C, H, W = batch_img.shape
        T = self.num_frames
        
        batch_video = batch_img.unsqueeze(1).expand(-1, T, -1, -1, -1)
        
        # Reshape to (K*T, C, H, W) for normalization compatibility
        return batch_video.reshape(K * T, C, H, W)

    def _process_video(self, path):
        """
        处理视频/帧目录：
        1. 采样 (Temporal Sampling)
        2. 读取
        3. Resize
        4. 弱增强 (Optional Flip)
        """
        tensor_data = None
        
        # --- Loading Logic ---
        if os.path.isdir(path):
            # 帧目录模式
            try:
                frames = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(self.IMAGE_EXTS)])
                total_frames = len(frames)
                if total_frames > 0:
                    indices = self._get_clip_indices(total_frames)
                    # 逐帧读取
                    loaded_frames = []
                    for i in indices:
                        try:
                            loaded_frames.append(read_image(frames[i], mode=ImageReadMode.RGB))
                        except:
                            loaded_frames.append(torch.zeros(3, 512, 512, dtype=torch.uint8))
                    tensor_data = torch.stack(loaded_frames) # (K*T, 3, H, W)
            except:
                pass
        else:
            # 视频文件模式
            try:
                vr = VideoReader(path, ctx=cpu(0))
                total_frames = len(vr)
                if total_frames > 0:
                    indices = self._get_clip_indices(total_frames)
                    tensor_data = vr.get_batch(indices).asnumpy()
                    tensor_data = torch.from_numpy(tensor_data).permute(0, 3, 1, 2) # (K*T, 3, H, W)
            except:
                pass

        if tensor_data is None:
            # Fallback black frames
            tensor_data = torch.zeros(self.clips_per_video * self.num_frames, 3, 512, 512, dtype=torch.uint8)

        # --- Transform Logic ---
        # 1. Resize
        tensor_data = self.resize_transform(tensor_data)
        
        # 2. Augment (针对视频的特殊处理)
        # 我们希望同一段视频的所有帧做一致的翻转，否则会破坏时序动作
        # 但这里 tensor_data 是 (K*T, ...)。
        # 简单起见，可以对整个 batch 做统一翻转，或者不做几何变换只做颜色变换
        # 这里演示：应用和图片一样的增强，但通常视频训练会对 RandomFlip 做同步处理
        # 鉴于你的需求是平衡，这里直接应用 resize 后的 augment 也是可以的，
        # 但为了严谨，这里我们暂时只做 ToDtype，或者你可以把 self.aug_transform 加在这里
        
        # 视频通常只做水平翻转，不做颜色抖动(容易闪烁)，这里演示只做Flip
        if random.random() < 0.5:
             tensor_data = v2.functional.hflip(tensor_data)

        # 3. To Float
        tensor_data = self.to_dtype(tensor_data)

        return tensor_data

    def __getitem__(self, idx):
        # === 核心平衡逻辑 ===
        # 如果 idx 在前半段 -> 取图片
        # 如果 idx 在后半段 -> 取视频 (使用取模映射回真实的视频列表)
        
        video_tensor = None
        
        if idx < self.num_image_samples:
            # === Case A: 图片样本 ===
            group_idx = idx
            start = group_idx * self.clips_per_video
            end = start + self.clips_per_video
            # 获取这组的4张图片路径
            paths = self.image_paths[start:end]
            
            # 如果不够4张(最后一点数据)，补全
            if len(paths) < self.clips_per_video:
                paths = paths + [paths[-1]] * (self.clips_per_video - len(paths))
                
            video_tensor = self._process_static_images(paths)
            
        else:
            # === Case B: 视频样本 (上采样) ===
            # 将大索引映射回真实的视频索引
            virtual_idx = idx - self.num_image_samples
            real_video_idx = virtual_idx % self.num_real_videos
            
            path = self.video_paths[real_video_idx]
            video_tensor = self._process_video(path)

        # --- Normalize ---
        # video_tensor shape: (K*T, 3, 512, 512) float [0,1]
        
        dino_pixels = (video_tensor - self.dino_mean) / self.dino_std
        siglip_pixels = (video_tensor - self.siglip_mean) / self.siglip_std

        # --- Reshape Output (K, T, C, H, W) ---
        K, T = self.clips_per_video, self.num_frames
        
        dino_pixels = dino_pixels.view(K, T, 3, 512, 512)
        siglip_pixels = siglip_pixels.view(K, T, 3, 512, 512)
        
        student_input = siglip_pixels 

        return {
            "dino": dino_pixels, 
            "siglip": siglip_pixels, 
            "student": student_input
        }
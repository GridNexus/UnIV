import torch
from torch.utils.data import Dataset
import random
from decord import VideoReader, cpu
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image, ImageReadMode
import os

class VideoDynamicDataset(Dataset):
    def __init__(self, 
                 txt_path, 
                 dinov3_path=None, 
                 siglip2_path=None, 
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

        # 1. 读取原始路径
        with open(txt_path, 'r') as f:
            raw_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        # 显式排序，保证不同机器/进程读取顺序一致
        raw_paths.sort() 
        
        # 采样数据集 (确定性采样)
        if dataset_sample_rate < 1.0:
            # 使用局部 Random 对象，不依赖全局状态，且必须先排序
            rng = random.Random(seed) 
            sample_size = max(1, int(len(raw_paths) * dataset_sample_rate))
            raw_paths = rng.sample(raw_paths, sample_size)
            print(f"Sampled {len(raw_paths)} raw items for training (Seed={seed}).")


        # 2. 分类整理数据
        self.video_samples = [] # 存单个视频/帧目录的路径
        self.all_image_paths = [] # 存所有图片的路径（拍平的列表）

        for path in raw_paths:
            if path.lower().endswith(self.IMAGE_EXTS):
                self.all_image_paths.append(path)
            else:
                # 视频文件 或 帧目录 视为同一种 "Video Sample"
                self.video_samples.append(path)

        # 3. 计算图片样本数量
        # 逻辑：每 clips_per_video (4) 张图片组成一个样本
        # 例如：6000张图 -> 1500个样本。多余的图（不足4张）会被丢弃
        self.num_image_samples = len(self.all_image_paths) // self.clips_per_video
        
        print(f"Dataset Summary:")
        print(f"  - Video Samples: {len(self.video_samples)}")
        print(f"  - Total Images: {len(self.all_image_paths)}")
        print(f"  - Image Groups (Samples): {self.num_image_samples} (Group size: {self.clips_per_video})")
        print(f"  - Total Dataset Length: {len(self.video_samples) + self.num_image_samples}")

        # --- 预定义 Transform ---
        # DINOv3 (ImageNet Mean/Std)
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # SigLIP (0.5 Mean/Std)
        self.siglip_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.siglip_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

        # 构造 Resize 变换 pipeline
        self.common_transform = v2.Compose([
            v2.Resize(self.target_size, antialias=True),
            v2.ToDtype(torch.bfloat16, scale=True), # [0, 255] uint8 -> [0, 1] float32
        ])

    def __len__(self):
        # 数据集总长度 = 视频数 + 图片组数
        return len(self.video_samples) + self.num_image_samples

    def _get_clip_indices(self, total_frames):
        """视频/帧目录通用的采样逻辑"""
        all_indices = []
        for _ in range(self.clips_per_video):
            possible_strides = [s for s in self.valid_strides if (self.num_frames - 1) * s + 1 <= total_frames]
            stride = 1 if not possible_strides else random.choice(possible_strides)
            max_start = total_frames - (self.num_frames - 1) * stride - 1
            start_idx = random.randint(0, max(0, max_start))
            clip_indices = [start_idx + i * stride for i in range(self.num_frames)]
            clip_indices = [min(idx, total_frames - 1) for idx in clip_indices]
            all_indices.extend(clip_indices)
        return all_indices

    def _read_single_image_as_tensor(self, path):
        """读取单张图片并转为 Tensor (C, H, W)"""
        try:
            img_tensor = read_image(path, mode=ImageReadMode.RGB)
            return img_tensor
        except Exception as e:
            print(f"Error reading image {path}: {e}")
            return torch.zeros(3, 512, 512, dtype=torch.uint8)

    def _load_video_decord(self, path):
        """加载视频文件"""
        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0: return None
        
        target_indices = self._get_clip_indices(total_frames)
        buffer = vr.get_batch(target_indices).asnumpy()
        del vr
        return torch.from_numpy(buffer).permute(0, 3, 1, 2).contiguous()

    def _load_frame_directory(self, dir_path):
        """加载帧目录"""
        try:
            frames = sorted([
                os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                if f.lower().endswith(self.IMAGE_EXTS)
            ])
        except Exception as e:
            print(f"Error accessing dir {dir_path}: {e}")
            return None

        total_frames = len(frames)
        if total_frames == 0: return None

        target_indices = self._get_clip_indices(total_frames)
        
        tensor_list = []
        for idx in target_indices:
            path = frames[idx]
            img = self._read_single_image_as_tensor(path)
            tensor_list.append(img)
            
        return torch.stack(tensor_list)

    def _load_specific_image_group(self, path_list):
        """
        加载指定的 K=4 张图片路径。
        path_list: 长度为 clips_per_video (4) 的列表
        """
        tensor_list = []
        for path in path_list:
            # 读取 (C, H, W)，此时可能是任意尺寸，如 480x640
            img = self._read_single_image_as_tensor(path)
            
            # --- 修复点：在 stack 之前必须先 Resize ---
            # 这里调用 common_transform 将其变为 (3, 512, 512) 且转为 Float
            img = self.common_transform(img)
            
            tensor_list.append(img)
        
        # Stack 起来: (K, C, H, W) -> (4, 3, 512, 512)
        # 因为都在上面 resize 过了，所以这里不会再报错
        batch_images = torch.stack(tensor_list)
        return batch_images 

    def __getitem__(self, idx):
        video_tensor = None
        is_static_image_group = False
        path_info = "" # 用于报错信息

        # --- 1. 判断 Index 类型并获取数据 ---
        num_videos = len(self.video_samples)

        if idx < num_videos:
            # === Case A: 视频数据 ===
            path = self.video_samples[idx]
            path_info = path
            if os.path.isdir(path):
                video_tensor = self._load_frame_directory(path)
            else:
                video_tensor = self._load_video_decord(path)

            # (N, C, H, W) 来自 视频 (N = K*T = 32)
            video_tensor = self.common_transform(video_tensor)
        
        else:
            # === Case B: 图片组数据 ===
            # 计算在图片列表中的偏移量
            # group_idx: 第几组图片
            group_idx = idx - num_videos
            
            # 计算这组图片的起始和结束索引
            start_img_idx = group_idx * self.clips_per_video
            end_img_idx = start_img_idx + self.clips_per_video
            
            # 获取这4个路径
            target_paths = self.all_image_paths[start_img_idx : end_img_idx]
            path_info = f"Image Group {group_idx}: {target_paths[0]}..."
            
            # 确保取到了足够的路径（理论上 init 里算好了，这里做个防御）
            if len(target_paths) < self.clips_per_video:
                print(f"Warning: Not enough images for group {group_idx}")
                raise ValueError("VideoDynamicDataset: Not enough images for a complete group.")

            video_tensor = self._load_specific_image_group(target_paths)
            is_static_image_group = True

        if video_tensor is None:
            raise ValueError("VideoDynamicDataset: Failed to load data.")

        # --- 2. 针对图片组的特殊处理 (Expand Time) ---
        if is_static_image_group:
            # 当前 shape: (K, C, H, W) -> (4, 3, 512, 512)
            # 目标 shape: (K * T, C, H, W) -> (32, 3, 512, 512)
            K, C, H, W = video_tensor.shape
            T = self.num_frames
            
            # 逻辑: 每张图作为一个Clip，自身复制T次
            video_tensor = video_tensor.unsqueeze(1).expand(-1, T, -1, -1, -1)
            video_tensor = video_tensor.reshape(K * T, C, H, W)

        # --- 3. Normalize ---
        dino_pixels = (video_tensor - self.dino_mean) / self.dino_std
        siglip_pixels = (video_tensor - self.siglip_mean) / self.siglip_std

        # --- 4. Reshape Output ---
        K, T = self.clips_per_video, self.num_frames
        
        if dino_pixels.shape[0] != K * T:
            print(f"Shape mismatch: {dino_pixels.shape[0]} != {K*T}")
            raise ValueError("Unexpected number of frames after processing.")

        dino_pixels = dino_pixels.view(K, T, 3, 512, 512)
        siglip_pixels = siglip_pixels.view(K, T, 3, 512, 512)
        
        student_input = siglip_pixels 

        return {
            "dino": dino_pixels, 
            "siglip": siglip_pixels, 
            "student": student_input
        }


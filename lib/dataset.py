import torch
from torch.utils.data import Dataset
import random
from transformers import AutoImageProcessor
from decord import VideoReader, cpu
import random

class VideoDynamicDataset(Dataset):
    def __init__(self, txt_path, dinov3_path, siglip2_path, num_frames=8, clips_per_video=4, dataset_sample_rate=1.0):
        with open(txt_path, 'r') as f:
            self.video_paths = [line.strip() for line in f.readlines() if line.strip()]
        if dataset_sample_rate < 1.0:
            sample_size = max(1, int(len(self.video_paths) * dataset_sample_rate))
            self.video_paths = random.sample(self.video_paths, sample_size)
            print(f"Sampled {len(self.video_paths)} videos for training (sample rate: {dataset_sample_rate})")
        
        self.num_frames = num_frames
        self.clips_per_video = clips_per_video
        self.valid_strides = [4, 8, 16, 24]

        print(f"Loading Processors from: {dinov3_path} and {siglip2_path}")
        self.dino_processor = AutoImageProcessor.from_pretrained(dinov3_path)
        self.siglip_processor = AutoImageProcessor.from_pretrained(siglip2_path)

    def __len__(self):
        return len(self.video_paths)

    def _get_clip_indices(self, total_frames):
        """
        步骤 1: 采样出 self.clips_per_video 个帧索引序列
        返回: List[int], 包含所有 clips 展平后的索引列表 (长度 = K * T)
        """
        all_indices = []
        
        for _ in range(self.clips_per_video):
            # 动态步长与起始点逻辑
            possible_strides = [s for s in self.valid_strides if (self.num_frames - 1) * s + 1 <= total_frames]
            stride = 1 if not possible_strides else random.choice(possible_strides)
            
            # 计算最大可能的起始位置
            max_start = total_frames - (self.num_frames - 1) * stride - 1
            start_idx = random.randint(0, max(0, max_start))
            
            # 生成当前 clip 的 8 帧索引
            clip_indices = [start_idx + i * stride for i in range(self.num_frames)]
            
            # 边界保护：防止越界（虽然上面的逻辑应该保证了，但加一层保险）
            clip_indices = [min(idx, total_frames - 1) for idx in clip_indices]
            
            all_indices.extend(clip_indices)
            
        return all_indices


    def _read_frames_decord(self, path, indices, vr):
        """
        步骤 3 实现 B: Decord get_batch 方式读取 (通常更快)
        """

        # Decord context 设为 CPU (如果需要在 GPU 预处理可改为 gpu(0))
        vr = VideoReader(path, ctx=cpu(0))
        
        # get_batch 直接接受索引列表，返回 (N, H, W, C) 的 array
        # decord 默认读取就是 RGB
        frames = vr.get_batch(indices).asnumpy()
        
        # 转为 List[numpy array] 以适配 Processor 接口
        final_frames = [frames[i] for i in range(frames.shape[0])]
        return final_frames

    def __getitem__(self, idx):
        path = self.video_paths[idx]

        # --- 1. 获取视频总帧数并生成索引 ---
        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0: return self.dummy_data()

        # 生成所有 clip 需要的索引
        target_indices = self._get_clip_indices(total_frames)
        all_frames_list = []
        frames_decord = self._read_frames_decord(path, target_indices, vr)
        all_frames_list = frames_decord
        del vr # 释放资源

        # --- 2. Processor 预处理 (保持不变) ---
        dino_enc = self.dino_processor(images=all_frames_list, return_tensors="pt")
        dino_pixels = dino_enc['pixel_values']

        siglip_enc = self.siglip_processor(images=all_frames_list, return_tensors="pt")
        siglip_pixels = siglip_enc['pixel_values']

        # --- 4. Reshape & Return ---
        K, T = self.clips_per_video, self.num_frames
        
        # 确保形状正确 (处理读取失败导致的帧数不足情况，虽然上面逻辑已尽力避免)
        if len(all_frames_list) != K * T:
             return self.dummy_data()

        dino_pixels = dino_pixels.view(K, T, 3, 512, 512)
        siglip_pixels = siglip_pixels.view(K, T, 3, 512, 512)
        student_input = siglip_pixels.clone()

        return {
            "dino": dino_pixels, 
            "siglip": siglip_pixels, 
            "student": student_input
        }

    def dummy_data(self):
        K, T = self.clips_per_video, self.num_frames
        dummy = torch.zeros(K, T, 3, 512, 512)
        return {"dino": dummy, "siglip": dummy, "student": dummy}
import cv2
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

VIDEO_LIST_PATH = os.getenv("VIDEO_LIST_PATH", "/path/to/video_paths.txt")
NUM_WORKERS = 16  # 根据CPU核心数调整

def count_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames
    except:
        return 0

def estimate_storage_distill(total_clips, num_frames=8, feat_dim=1536, h=32, w=32):
    # 估算 Teacher Target (BF16) 的大小
    # Shape: (8, 32, 32, 1536) * 2 bytes (BF16)
    total_clips = total_clips // 10  # 假设我们只存每10个clip中的1个，实际策略可能更复杂
    size_per_clip_bytes = num_frames * h * w * feat_dim * 2
    total_size_gb = (total_clips * size_per_clip_bytes) / (1024**3)
    
    # 还要存 Input Pixels (BF16, 512x512)
    # Shape: (8, 3, 512, 512) * 2 bytes
    input_size_gb = (total_clips * num_frames * 3 * 512 * 512 * 2) / (1024**3)
    
    return total_size_gb, input_size_gb


def estimate_storage_train(total_clips, num_frames=8, feat_dim=2048, h=32, w=32):
    # 估算 Teacher Target (BF16) 的大小
    # Shape: (8, 32, 32, 1536) * 2 bytes (BF16)
    total_clips += 0
    size_per_clip_bytes = 1 * h * w * feat_dim * 2
    total_size_gb = (total_clips * size_per_clip_bytes) / (1024**3)
    
    # 还要存 Input Pixels (BF16, 512x512)
    # Shape: (8, 3, 512, 512) * 2 bytes
    # input_size_gb = (total_clips * num_frames * 3 * 512 * 512 * 2) / (1024**3)
    
    return total_size_gb


if __name__ == "__main__":
    with open(VIDEO_LIST_PATH, 'r') as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Total videos: {len(paths)}")
    print(f"Counting frames with {NUM_WORKERS} workers...")

    # total_frames = 0
    # with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     results = list(tqdm(executor.map(count_frames, paths), total=len(paths)))
    
    # total_frames = sum(results)
    # print(f"Total Frames in dataset: {total_frames}")
    total_frames = 7345558
    
    # 假设策略：每隔 8 帧取一个片段 (非重叠切片)
    clip_len = 8
    frame_stride = 12 # 假设每4帧取一帧
    feat_hw = 16
    # 简单的估算：如果是不重叠切分 (Non-overlapping)
    # 实际上我们通常按照 clip_duration 切分
    # 假设我们只存 "连续的8帧(带间隔)"，且不重叠
    estimated_clips = total_frames // (clip_len * frame_stride)
    estimated_clips_train = total_frames // (clip_len * 1) # 训练时可能更密集一些，假设每4帧取一个clip

    print(f"Estimated Clips (Len={clip_len}, Stride={frame_stride}, Non-overlapping): {estimated_clips}")
    
    feat_gb, input_gb = estimate_storage_distill(estimated_clips, h=32, w=32)
    print(f"Estimated Storage for DISTILL Features: {feat_gb:.2f} GB")
    print(f"Estimated Storage for DISTILL Inputs: {input_gb:.2f} GB")
    print(f"Total DISTILL Cache Size: {feat_gb + input_gb:.2f} GB")
    
    feat_gb = estimate_storage_train(estimated_clips_train, h=16, w=16)

    print(f"Estimated Storage for TRAIN Features: {feat_gb:.2f} GB")

"""
Multi-task joint training script.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
import warnings
from tqdm import tqdm

from lib.student_conv_only import PureConvNext3DModel
from lib.multi_task_trainer import MultitaskDetector, MultitaskLoss, MultitaskTrainer
from lib.utils import setup_ddp, cleanup_ddp


warnings.filterwarnings("ignore")


# Config
DINOV3_PATH = os.getenv("DINOV3_PATH", "/path/to/dinov3-vitb16-pretrain-lvd1689m")
SIGLIP2_PATH = os.getenv("SIGLIP2_PATH", "/path/to/siglip2-base-patch16-512")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "/path/to/train_data.json")
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "/path/to/val_data.json")
PRETRAINED_ENCODER = os.getenv("PRETRAINED_ENCODER", "/path/to/pretrained_encoder.pth")

SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "Video-Multitask-Detection")
SWANLAB_RUN_NAME = os.getenv("SWANLAB_RUN_NAME", "multitask_v1")
SAVE_PATH = os.getenv("SAVE_PATH", f"./checkpoints/{SWANLAB_PROJECT}/{SWANLAB_RUN_NAME}")
os.makedirs(SAVE_PATH, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 4
NUM_FRAMES = 8
EPOCHS = 30
LEARNING_RATE = 5e-4
NUM_WORKERS = 8
RANDOM_SEED = 0

NUM_DET_CLASSES = 4
NUM_ACTION_CLASSES = 5
NUM_ANOMALY_CLASSES = 2

FREEZE_ENCODER_EPOCHS = 5


class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, num_frames: int = 8, image_size: int = 512):
        super().__init__()
        self.data_path = data_path
        self.num_frames = num_frames
        self.image_size = image_size
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        return {
            'video': torch.randn(3, self.num_frames, self.image_size, self.image_size),
            'gt_boxes': torch.zeros(0, 4),
            'gt_labels': torch.zeros(0, dtype=torch.long),
            'action_labels': torch.zeros(self.num_frames, dtype=torch.long),
            'boundary_offsets': torch.zeros(self.num_frames, 2),
            'anomaly_labels': torch.zeros(self.num_frames, dtype=torch.float),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task joint training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    setup_ddp()
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    
    set_seed(RANDOM_SEED)
    
    if rank == 0:
        print("=" * 60)
        print("Multi-Task Joint Training")
        print("=" * 60)
        print(f"Rank: {rank}, World Size: {dist.get_world_size()}")
    
    # Load encoder
    if rank == 0:
        print("\n[1/5] Loading pretrained encoder...")
    
    encoder = PureConvNext3DModel(
        in_channels=3,
        latent_channels=2048,
        output_channels=1536,
        num_frames=NUM_FRAMES,
    )
    
    if PRETRAINED_ENCODER and os.path.exists(PRETRAINED_ENCODER):
        checkpoint = torch.load(PRETRAINED_ENCODER, map_location="cpu")
        encoder.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        if rank == 0:
            print(f"  Loaded: {PRETRAINED_ENCODER}")
    else:
        if rank == 0:
            print("  Warning: No pretrained encoder found")
    
    # Create detector
    if rank == 0:
        print("\n[2/5] Creating multi-task detector...")
    
    model = MultitaskDetector(
        pretrained_encoder=encoder,
        num_det_classes=NUM_DET_CLASSES,
        num_action_classes=NUM_ACTION_CLASSES,
        num_anomaly_classes=NUM_ANOMALY_CLASSES,
        embed_dim=1536,
        encoder_channels=2048,
        num_frames=NUM_FRAMES,
        freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS,
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params:,}")
    
    # Prepare data
    if rank == 0:
        print("\n[3/5] Loading dataset...")
    
    train_dataset = MultitaskDataset(TRAIN_DATA_PATH, num_frames=NUM_FRAMES)
    val_dataset = MultitaskDataset(VAL_DATA_PATH, num_frames=NUM_FRAMES)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    if rank == 0:
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Setup trainer
    if rank == 0:
        print("\n[4/5] Configuring trainer...")
    
    loss_fn = MultitaskLoss(lambda_det=1.0, lambda_action=1.0, lambda_anomaly=1.0)
    
    trainer = MultitaskTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_config={
            'encoder_lr': 1e-4,
            'head_lr': LEARNING_RATE,
            'weight_decay': 1e-4,
            'warmup_epochs': 2,
        }
    )
    trainer.setup_optimizer()
    
    if rank == 0:
        import swanlab
        swanlab.init(
            project=SWANLAB_PROJECT,
            name=SWANLAB_RUN_NAME,
            config={
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                "num_frames": NUM_FRAMES, "freeze_encoder_epochs": FREEZE_ENCODER_EPOCHS,
            }
        )
    
    # Training loop
    if rank == 0:
        print("\n[5/5] Starting training...")
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank == 0:
            print(f"  Resumed from epoch {start_epoch}")
    
    if args.eval_only:
        if rank == 0:
            val_loss = trainer.validate(val_dataloader)
            print(f"Val loss: {val_loss:.4f}")
        cleanup_ddp()
        return
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)
        model.module.set_epoch(epoch)
        
        if rank == 0:
            is_frozen = epoch < FREEZE_ENCODER_EPOCHS
            print(f"\nEpoch {epoch+1}/{EPOCHS} - Encoder {'frozen' if is_frozen else 'unfrozen'}")
        
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(train_dataloader, desc=f"Train", disable=(rank != 0))
        
        for batch_idx, batch_data in enumerate(iterator):
            video_input = batch_data['video'].to(device)
            
            targets = {
                'det_targets': {
                    'boxes': batch_data.get('gt_boxes', torch.zeros(1, 0, 4)).to(device),
                    'labels': batch_data.get('gt_labels', torch.zeros(1, 0, dtype=torch.long)).to(device),
                },
                'action_targets': {
                    'action_labels': batch_data.get('action_labels', torch.zeros(1, NUM_FRAMES, dtype=torch.long)).to(device),
                    'boundary_labels': batch_data.get('boundary_offsets', torch.zeros(1, NUM_FRAMES, 2)).to(device),
                },
                'anomaly_targets': batch_data.get('anomaly_labels', torch.zeros(1, NUM_FRAMES)).to(device),
            }
            
            outputs = model(video_input)
            loss, metrics = loss_fn(outputs, targets, model.module.log_vars)
            
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if rank == 0:
                current_lr = trainer.optimizer.param_groups[0]['lr']
                iterator.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
                
                swanlab.log({
                    "train/loss": loss.item(), "train/lr": current_lr,
                    "train/det_loss": metrics.get('det_loss_raw', 0),
                    "train/action_loss": metrics.get('action_loss_raw', 0),
                    "train/anomaly_loss": metrics.get('anomaly_loss_raw', 0),
                })
        
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        avg_train_loss = train_loss / num_batches
        val_loss = trainer.validate(val_dataloader)
        
        if rank == 0:
            print(f"  Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            swanlab.log({
                "val/loss": val_loss, "val/avg_train_loss": avg_train_loss,
            })
            
            save_dict = {
                'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'train_loss': avg_train_loss, 'val_loss': val_loss,
            }
            
            torch.save(save_dict, os.path.join(SAVE_PATH, "checkpoint_latest.pth"))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(save_dict, os.path.join(SAVE_PATH, "checkpoint_best.pth"))
                print(f"  Saved best model (val_loss={val_loss:.4f})")
    
    cleanup_ddp()
    
    if rank == 0:
        print("\nTraining complete! Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()

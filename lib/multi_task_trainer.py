"""
Multi-task joint optimization and gradient decoupling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from lib.heads.projection import SpatiotemporalCompressor
from lib.heads.detection_head import DetectionHead
from lib.heads.action_localization_head import ActionLocalizationHead, ActionLocalizationLoss
from lib.heads.anomaly_detection_head import AnomalyDetectionHead, SmoothAnomalyLoss


class MultitaskDetector(nn.Module):
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        num_det_classes: int = 4,
        num_action_classes: int = 5,
        num_anomaly_classes: int = 2,
        embed_dim: int = 1536,
        encoder_channels: int = 2048,
        num_frames: int = 8,
        freeze_encoder_epochs: int = 5,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.current_epoch = 0
        
        self.encoder = pretrained_encoder
        self._freeze_encoder()
        
        self.compressor = SpatiotemporalCompressor(
            encoder_channels=encoder_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
        )
        
        self.detection_head = DetectionHead(num_classes=num_det_classes, embed_dim=256)
        self.action_head = ActionLocalizationHead(embed_dim=embed_dim, state_dim=256, num_classes=num_action_classes)
        self.anomaly_head = AnomalyDetectionHead(input_dim=embed_dim, hidden_dim=512, num_classes=num_anomaly_classes)
        
        self.log_vars = nn.Parameter(torch.zeros(3))
        
    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def _unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch >= self.freeze_encoder_epochs:
            self._unfreeze_encoder()
        else:
            self._freeze_encoder()
    
    def forward(self, video_input):
        encoder_output = self.encoder.encode(video_input)
        temporal_features = self.compressor(encoder_output)
        
        outputs = {}
        multi_scale_features = [temporal_features] * 3
        detection_output = self.detection_head(multi_scale_features, temporal_features)
        outputs['detection'] = detection_output
        
        action_output = self.action_head(temporal_features)
        outputs['action'] = action_output
        
        anomaly_output = self.anomaly_head(temporal_features)
        outputs['anomaly'] = anomaly_output
        
        return outputs
    
    def get_trainable_params(self):
        trainable_params = []
        
        if self.current_epoch >= self.freeze_encoder_epochs:
            trainable_params.append({'params': self.encoder.parameters(), 'lr': 1e-4, 'name': 'encoder'})
        
        trainable_params.append({'params': self.compressor.parameters(), 'lr': 5e-4, 'name': 'compressor'})
        trainable_params.append({'params': self.detection_head.parameters(), 'lr': 5e-4, 'name': 'detection'})
        trainable_params.append({'params': self.action_head.parameters(), 'lr': 5e-4, 'name': 'action'})
        trainable_params.append({'params': self.anomaly_head.parameters(), 'lr': 5e-4, 'name': 'anomaly'})
        trainable_params.append({'params': [self.log_vars], 'lr': 1e-3, 'name': 'log_vars'})
        
        return trainable_params


class MultitaskLoss(nn.Module):
    def __init__(self, lambda_det: float = 1.0, lambda_action: float = 1.0, lambda_anomaly: float = 1.0):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_action = lambda_action
        self.lambda_anomaly = lambda_anomaly
        
        self.detection_loss_fn = DetectionLoss()
        self.action_loss_fn = ActionLocalizationLoss()
        self.anomaly_loss_fn = SmoothAnomalyLoss()
        
    def forward(self, outputs, targets, log_vars):
        det_targets = targets.get('det_targets', {})
        action_targets = targets.get('action_targets', {})
        anomaly_targets = targets.get('anomaly_targets', {})
        
        det_loss, det_metrics = self.detection_loss_fn(outputs['detection'], det_targets)
        action_loss, action_metrics = self.action_loss_fn(outputs['action'], action_targets)
        anomaly_loss, anomaly_metrics = self.anomaly_loss_fn(outputs['anomaly'], anomaly_targets)
        
        precision_0 = torch.exp(-log_vars[0])
        precision_1 = torch.exp(-log_vars[1])
        precision_2 = torch.exp(-log_vars[2])
        
        weighted_det = precision_0 * det_loss + log_vars[0]
        weighted_action = precision_1 * action_loss + log_vars[1]
        weighted_anomaly = precision_2 * anomaly_loss + log_vars[2]
        
        total_loss = (
            weighted_det * self.lambda_det +
            weighted_action * self.lambda_action +
            weighted_anomaly * self.lambda_anomaly
        )
        
        metrics = {
            'total_loss': total_loss.item(),
            'det_loss_raw': det_loss.item(),
            'action_loss_raw': action_loss.item(),
            'anomaly_loss_raw': anomaly_loss.item(),
            'det_loss_weighted': weighted_det.item(),
            'action_loss_weighted': weighted_action.item(),
            'anomaly_loss_weighted': weighted_anomaly.item(),
            'log_var_0': log_vars[0].item(),
            'log_var_1': log_vars[1].item(),
            'log_var_2': log_vars[2].item(),
            **det_metrics,
            **action_metrics,
            **anomaly_metrics,
        }
        
        return total_loss, metrics


class DetectionLoss(nn.Module):
    def __init__(self, lambda_cls: float = 2.0, lambda_L1: float = 5.0, lambda_giou: float = 2.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_L1 = lambda_L1
        self.lambda_giou = lambda_giou
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def forward(self, outputs, targets):
        pred_boxes = outputs['boxes']
        pred_scores = outputs['scores']
        
        gt_boxes = targets.get('boxes', torch.zeros_like(pred_boxes))
        gt_labels = targets.get('labels', torch.zeros(pred_scores.shape[:2], dtype=torch.long, device=pred_scores.device))
        
        B, N, C = pred_scores.shape
        pred_scores_flat = pred_scores.reshape(B * N, C)
        gt_labels_flat = gt_labels.reshape(B * N)
        
        cls_loss = self.focal_loss(pred_scores_flat, gt_labels_flat)
        box_loss = F.l1_loss(pred_boxes, gt_boxes, reduction='mean')
        giou_loss = 0.0
        
        total_loss = self.lambda_cls * cls_loss + self.lambda_L1 * box_loss + self.lambda_giou * giou_loss
        
        return total_loss, {'cls_loss': cls_loss.item(), 'box_loss': box_loss, 'giou_loss': giou_loss}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultitaskTrainer:
    def __init__(self, model: MultitaskDetector, loss_fn: MultitaskLoss, optimizer_config: Optional[Dict] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config or {
            'encoder_lr': 1e-4,
            'head_lr': 5e-4,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
        }
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self):
        params = []
        encoder_params = list(self.model.encoder.parameters())
        params.append({'params': encoder_params, 'lr': self.optimizer_config['encoder_lr']})
        
        other_params = list(self.model.compressor.parameters()) + \
                      list(self.model.detection_head.parameters()) + \
                      list(self.model.action_head.parameters()) + \
                      list(self.model.anomaly_head.parameters()) + \
                      [self.model.log_vars]
        params.append({'params': other_params, 'lr': self.optimizer_config['head_lr']})
        
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.optimizer_config['weight_decay'])
        
        total_epochs = 30
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_epochs - self.optimizer_config['warmup_epochs'],
            T_mult=1,
            eta_min=self.optimizer_config['head_lr'] * 0.1,
        )
        
    def train_epoch(self, dataloader, epoch):
        self.model.set_epoch(epoch)
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            video_input = batch_data['video'].to(next(self.model.parameters()).device)
            
            targets = {
                'det_targets': {'boxes': batch_data.get('gt_boxes'), 'labels': batch_data.get('gt_labels')},
                'action_targets': {'action_labels': batch_data.get('action_labels'), 'boundary_labels': batch_data.get('boundary_labels')},
                'anomaly_targets': batch_data.get('anomaly_labels'),
            }
            
            outputs = self.model(video_input)
            loss, metrics = self.loss_fn(outputs, targets, self.model.log_vars)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                video_input = batch_data['video'].to(next(self.model.parameters()).device)
                
                targets = {
                    'det_targets': {'boxes': batch_data.get('gt_boxes'), 'labels': batch_data.get('gt_labels')},
                    'action_targets': {'action_labels': batch_data.get('action_labels'), 'boundary_labels': batch_data.get('boundary_labels')},
                    'anomaly_targets': batch_data.get('anomaly_labels'),
                }
                
                outputs = self.model(video_input)
                loss, _ = self.loss_fn(outputs, targets, self.model.log_vars)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches


def create_multitask_detector(pretrained_encoder_path: Optional[str] = None, device: str = "cuda") -> MultitaskDetector:
    if pretrained_encoder_path:
        from lib.student_conv_only import PureConvNext3DModel
        encoder = PureConvNext3DModel(in_channels=3, latent_channels=2048, output_channels=1536, num_frames=8)
        checkpoint = torch.load(pretrained_encoder_path, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained encoder from {pretrained_encoder_path}")
    else:
        from lib.student_conv_only import PureConvNext3DModel
        encoder = PureConvNext3DModel(in_channels=3, latent_channels=2048, output_channels=1536, num_frames=8)
        print("Warning: Using untrained encoder.")
    
    model = MultitaskDetector(
        pretrained_encoder=encoder,
        num_det_classes=4,
        num_action_classes=5,
        num_anomaly_classes=2,
        embed_dim=1536,
        encoder_channels=2048,
        num_frames=8,
        freeze_encoder_epochs=5,
    )
    
    return model.to(device)

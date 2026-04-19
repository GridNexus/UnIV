"""
Multi-task detection heads.
"""

from .projection import SpatiotemporalCompressor
from .detection_head import DetectionHead
from .action_localization_head import ActionLocalizationHead
from .anomaly_detection_head import AnomalyDetectionHead

__all__ = [
    'SpatiotemporalCompressor',
    'DetectionHead',
    'ActionLocalizationHead',
    'AnomalyDetectionHead',
]

"""
Loss functions module for ViEmoText.
Includes various loss functions for emotion classification.
"""

from .focal_loss import FocalLoss
from .weighted_cross_entropy import WeightedCrossEntropyLoss

__all__ = [
    'FocalLoss',
    'WeightedCrossEntropyLoss'
]

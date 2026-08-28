"""
Weighted Cross Entropy Loss for handling class imbalance.
"""

import torch
import torch.nn as nn
from typing import Optional


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with class weights to handle imbalanced datasets.
    
    Args:
        weights: Weight for each class [num_classes]
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Weighted Cross Entropy Loss.
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Weighted cross entropy loss value
        """
        return nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.weights,
            reduction=self.reduction
        )


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    Uses inverse frequency weighting: weight = total_samples / (num_classes * class_count)
    
    Args:
        labels: Array of labels [num_samples]
        num_classes: Number of classes
    
    Returns:
        Class weights [num_classes]
    """
    # Count samples per class
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    
    # Compute weights (inverse frequency)
    total_samples = len(labels)
    weights = total_samples / (num_classes * class_counts)
    
    # Handle zero counts
    weights[class_counts == 0] = 0.0
    
    return weights

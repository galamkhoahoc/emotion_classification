"""
Focal Loss implementation for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in emotion classification.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               or a list of weights for each class
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Focal loss value
        """
        # Get softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Calculate focal loss
        loss = focal_term * ce_loss
        
        # Apply alpha if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Alpha is a list/tensor of weights per class
                alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

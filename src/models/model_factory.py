"""
Model factory for creating emotion classifiers and tokenizers.

This module implements the Factory pattern for centralized model instantiation
based on configuration. It supports creating both PhoBERT and BamiBERT models
with their corresponding tokenizers.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Tuple, Optional
import logging

from src.models.base_classifier import BaseEmotionClassifier
from src.models.phobert_emotion import PhoBERTEmotionClassifier
from src.models.bamibert_emotion import BamiBERTEmotionClassifier
from src.losses.focal_loss import FocalLoss
from src.losses.weighted_cross_entropy import WeightedCrossEntropyLoss, compute_class_weights

logger = logging.getLogger(__name__)


def create_loss_function(config, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration object with loss_type, focal_loss_alpha, focal_loss_gamma
        class_weights: Optional class weights tensor for weighted cross entropy
    
    Returns:
        Loss function module
    
    Raises:
        ValueError: If loss_type is invalid
    """
    loss_type = config.loss_type.lower()
    
    logger.info(f"Creating loss function: {loss_type}")
    
    # Check problem type for multilabel
    if getattr(config, 'problem_type', "multiclass_classification") == "multilabel_classification":
        if loss_type != "cross_entropy":
            logger.warning(
                f"Loss type '{loss_type}' requested for multilabel classification, "
                f"but BCEWithLogitsLoss is required. Overriding loss_type."
            )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        logger.info("Created BCEWithLogitsLoss for multilabel classification")
        return loss_fn
    
    if loss_type == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal_loss":
        alpha = config.focal_loss_alpha
        gamma = config.focal_loss_gamma
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info(f"FocalLoss with alpha={alpha}, gamma={gamma}")
    elif loss_type == "weighted_ce":
        if class_weights is None:
            logger.warning("weighted_ce selected but no class_weights provided, using uniform weights")
        loss_fn = WeightedCrossEntropyLoss(weights=class_weights)
        logger.info(f"WeightedCrossEntropyLoss with weights: {class_weights}")
    else:
        raise ValueError(
            f"Invalid loss_type: '{loss_type}'. "
            f"Must be 'cross_entropy', 'focal_loss', or 'weighted_ce'"
        )
    
    return loss_fn


def create_model(config, loss_fn: Optional[nn.Module] = None) -> Tuple[BaseEmotionClassifier, AutoTokenizer]:
    """
    Factory function to create emotion classifier and tokenizer.
    
    Creates the appropriate model and tokenizer based on the model_type
    specified in the configuration. Supports 'phobert' and 'bamibert'.
    
    Args:
        config: Configuration object with model_type, model_name, num_labels,
            and optional dropout_prob and hidden_size attributes.
        loss_fn: Optional custom loss function. If None, uses CrossEntropyLoss.
    
    Returns:
        Tuple of (model, tokenizer) where model is a BaseEmotionClassifier
        subclass and tokenizer is an AutoTokenizer.
    
    Raises:
        ValueError: If model_type is invalid.
        RuntimeError: If model creation fails (e.g., download error).
    """
    model_type = config.model_type.lower()
    
    logger.info(f"Creating model: {model_type} ({config.model_name})")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        logger.info(f"Loaded tokenizer from {config.model_name}")
        
        # Create model based on type
        if model_type == "phobert":
            model = PhoBERTEmotionClassifier(
                model_name=config.model_name,
                num_labels=config.num_labels,
                dropout_prob=getattr(config, 'dropout_prob', 0.1),
                hidden_size=getattr(config, 'hidden_size', 768),
                loss_fn=loss_fn
            )
        elif model_type == "bamibert":
            model = BamiBERTEmotionClassifier(
                model_name=config.model_name,
                num_labels=config.num_labels,
                dropout_prob=getattr(config, 'dropout_prob', 0.1),
                hidden_size=getattr(config, 'hidden_size', 768),
                loss_fn=loss_fn
            )
        else:
            raise ValueError(
                f"Invalid model_type: '{model_type}'. "
                f"Must be 'phobert' or 'bamibert'"
            )
        
        logger.info(f"Successfully created {model_type} model")
        return model, tokenizer
    
    except ValueError:
        # Re-raise ValueError without wrapping in RuntimeError
        raise
    except OSError as e:
        # Network errors, file not found, corrupted cache
        logger.error(f"Failed to download or load model {config.model_name}: {str(e)}")
        raise RuntimeError(
            f"Model download/load failed for {config.model_name}. "
            f"Check your internet connection and model name. Error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Failed to create model {config.model_name}: {str(e)}")
        raise RuntimeError(
            f"Model creation failed for {config.model_name}: {str(e)}"
        ) from e


def create_model_from_checkpoint(
    checkpoint_path: str,
    config,
    loss_fn: Optional[nn.Module] = None
) -> Tuple[BaseEmotionClassifier, AutoTokenizer]:
    """
    Create model and load weights from checkpoint.
    
    Creates a fresh model using the factory, then loads trained weights
    from a checkpoint file. Supports both checkpoint formats:
    - Full checkpoint with 'model_state_dict' key (from training)
    - Direct state dict (from final model save)
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        config: Configuration object with model settings
        loss_fn: Optional custom loss function
    
    Returns:
        Tuple of (model with loaded weights, tokenizer)
    
    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist
        RuntimeError: If model creation or weight loading fails
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create fresh model and tokenizer
    model, tokenizer = create_model(config, loss_fn=loss_fn)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Successfully loaded model weights")
    return model, tokenizer

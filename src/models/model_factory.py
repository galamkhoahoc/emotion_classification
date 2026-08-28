"""
Model factory for creating emotion classifiers and tokenizers.

This module implements the Factory pattern for centralized model instantiation
based on configuration. It supports creating both PhoBERT and BamiBERT models
with their corresponding tokenizers.
"""

import torch
from transformers import AutoTokenizer
from typing import Tuple
import logging

from src.models.base_classifier import BaseEmotionClassifier
from src.models.phobert_emotion import PhoBERTEmotionClassifier
from src.models.bamibert_emotion import BamiBERTEmotionClassifier

logger = logging.getLogger(__name__)


def create_model(config) -> Tuple[BaseEmotionClassifier, AutoTokenizer]:
    """
    Factory function to create emotion classifier and tokenizer.
    
    Creates the appropriate model and tokenizer based on the model_type
    specified in the configuration. Supports 'phobert' and 'bamibert'.
    
    Args:
        config: Configuration object with model_type, model_name, num_labels,
            and optional dropout_prob and hidden_size attributes.
    
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
                hidden_size=getattr(config, 'hidden_size', 768)
            )
        elif model_type == "bamibert":
            model = BamiBERTEmotionClassifier(
                model_name=config.model_name,
                num_labels=config.num_labels,
                dropout_prob=getattr(config, 'dropout_prob', 0.1),
                hidden_size=getattr(config, 'hidden_size', 768)
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
    except Exception as e:
        logger.error(f"Failed to create model {config.model_name}: {str(e)}")
        raise RuntimeError(
            f"Model creation failed for {config.model_name}: {str(e)}"
        ) from e


def create_model_from_checkpoint(
    checkpoint_path: str,
    config
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
    
    Returns:
        Tuple of (model with loaded weights, tokenizer)
    
    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist
        RuntimeError: If model creation or weight loading fails
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create fresh model and tokenizer
    model, tokenizer = create_model(config)
    
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

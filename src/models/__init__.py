"""
Models module for ViEmoText.
Contains model architectures, factory, and emoji embedding utilities.
"""

from .base_classifier import BaseEmotionClassifier
from .phobert_emotion import PhoBERTEmotionClassifier
from .bamibert_emotion import BamiBERTEmotionClassifier
from .model_factory import create_model, create_model_from_checkpoint
from .emoji_embeddings import apply_emoji_embeddings

__all__ = [
    'BaseEmotionClassifier',
    'PhoBERTEmotionClassifier',
    'BamiBERTEmotionClassifier',
    'create_model',
    'create_model_from_checkpoint',
    'apply_emoji_embeddings'
]

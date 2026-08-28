"""
Models module for ViEmoText.
Contains model architectures and emoji embedding utilities.
"""

from .phobert_emotion import PhoBERTEmotionClassifier
from .emoji_embeddings import apply_emoji_embeddings

__all__ = [
    'PhoBERTEmotionClassifier',
    'apply_emoji_embeddings'
]

"""
Abstract base class for emotion classification models.

This module defines the common interface that all emotion classifiers must implement,
ensuring consistency across different model architectures (PhoBERT, BamiBERT, etc.).
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Optional


class BaseEmotionClassifier(nn.Module, ABC):
    """
    Abstract base class for emotion classification models.
    
    All emotion classifier implementations must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface across different
    model architectures, enabling seamless switching between models through the factory pattern.
    
    Args:
        num_labels: Number of emotion classes to predict
    
    Attributes:
        num_labels: Number of emotion classes
    """
    
    def __init__(self, num_labels: int):
        """
        Initialize the base emotion classifier.
        
        Args:
            num_labels: Number of emotion classes to predict
        """
        super().__init__()
        self.num_labels = num_labels
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        This method must be implemented by all subclasses to perform the forward
        computation from input tokens to emotion classification logits.
        
        Args:
            input_ids: Input token IDs with shape [batch_size, seq_length].
                These are the tokenized input texts.
            attention_mask: Attention mask with shape [batch_size, seq_length].
                Indicates which tokens should be attended to (1) and which should
                be ignored (0). Optional - if not provided, all tokens are attended to.
            labels: Ground truth emotion labels with shape [batch_size].
                Optional - if provided, the method should compute and return loss.
        
        Returns:
            Dictionary with the following keys:
                - 'loss': Loss tensor (scalar) if labels are provided, None otherwise.
                    The loss should be computed using an appropriate loss function
                    (e.g., CrossEntropyLoss for multi-class classification).
                - 'logits': Classification logits with shape [batch_size, num_labels].
                    These are the raw, unnormalized scores for each emotion class.
        
        Example:
            >>> model = SomeEmotionClassifier(num_labels=7)
            >>> input_ids = torch.randint(0, 1000, (2, 10))
            >>> attention_mask = torch.ones(2, 10)
            >>> labels = torch.tensor([0, 3])
            >>> outputs = model(input_ids, attention_mask, labels)
            >>> outputs['loss']  # scalar tensor
            >>> outputs['logits'].shape  # torch.Size([2, 7])
        """
        pass
    
    @abstractmethod
    def get_embeddings(self) -> nn.Embedding:
        """
        Get the embedding layer of the model.
        
        This method must be implemented by all subclasses to provide access to
        the token embedding layer. This is used for operations like emoji embedding
        injection, where we need to add new tokens and copy embeddings from existing
        vocabulary words.
        
        Returns:
            The embedding layer (nn.Embedding) of the underlying transformer model.
            This layer maps token IDs to dense vector representations.
        
        Example:
            >>> model = SomeEmotionClassifier(num_labels=7)
            >>> embeddings = model.get_embeddings()
            >>> embeddings.weight.shape  # torch.Size([vocab_size, hidden_size])
        """
        pass
    
    @abstractmethod
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to accommodate new tokens.
        
        This method must be implemented by all subclasses to support vocabulary
        expansion. This is particularly useful when adding special tokens like
        emojis to the model's vocabulary after initialization.
        
        The method should:
        1. Resize the embedding matrix to the new vocabulary size
        2. Preserve existing embeddings for original tokens
        3. Initialize new embeddings (typically with small random values)
        
        Args:
            new_num_tokens: The new vocabulary size. Must be greater than or equal
                to the current vocabulary size. Typically this is
                original_vocab_size + number_of_new_tokens.
        
        Returns:
            None. The method modifies the model's embedding layer in-place.
        
        Example:
            >>> model = SomeEmotionClassifier(num_labels=7)
            >>> original_vocab = len(tokenizer)  # e.g., 64000
            >>> tokenizer.add_tokens(['😊', '😢', '😠'])  # Add emoji tokens
            >>> new_vocab = len(tokenizer)  # e.g., 64003
            >>> model.resize_token_embeddings(new_vocab)
        """
        pass

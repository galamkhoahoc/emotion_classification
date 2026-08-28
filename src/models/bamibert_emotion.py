"""
BamiBERT-based model for Vietnamese emotion classification.

BamiBERT is Qualcomm's Vietnamese BERT model with:
- Extended context length (2048 tokens vs PhoBERT's 256)
- No word segmentation required (works with raw text)
- State-of-the-art Vietnamese language understanding
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict
from .base_classifier import BaseEmotionClassifier


class BamiBERTEmotionClassifier(BaseEmotionClassifier):
    """
    BamiBERT model for emotion classification.
    
    BamiBERT is Qualcomm's Vietnamese BERT model that operates on raw text
    without requiring word segmentation, and supports up to 2048 tokens
    of context.
    
    Args:
        model_name: Name of the pretrained BamiBERT model
        num_labels: Number of emotion classes
        dropout_prob: Dropout probability for the classifier
        hidden_size: Hidden size of the BamiBERT model (default: 768)
    """
    
    def __init__(
        self,
        model_name: str = "Qualcomm-AI-Research/BamiBERT",
        num_labels: int = 7,
        dropout_prob: float = 0.1,
        hidden_size: int = 768,
        loss_fn: Optional[nn.Module] = None
    ):
        super().__init__(num_labels=num_labels)
        
        # Load BamiBERT configuration
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Load pretrained BamiBERT model
        self.bamibert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Loss function (default to CrossEntropyLoss if not provided)
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Ground truth labels [batch_size] (optional)
        
        Returns:
            Dictionary containing loss (if labels provided) and logits
        """
        # Get BamiBERT outputs
        outputs = self.bamibert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if len(labels.shape) == 1:
                # Multiclass [batch_size]
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                # Multilabel [batch_size, num_labels]
                loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def get_embeddings(self) -> nn.Embedding:
        """
        Get the embedding layer of the model.
        
        Returns:
            Embedding layer
        """
        return self.bamibert.embeddings.word_embeddings
    
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to accommodate new tokens (e.g., emojis).
        
        Args:
            new_num_tokens: New vocabulary size
        """
        self.bamibert.resize_token_embeddings(new_num_tokens)

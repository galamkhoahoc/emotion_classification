"""
PhoBERT-based model for Vietnamese emotion classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict
from .base_classifier import BaseEmotionClassifier


class PhoBERTEmotionClassifier(BaseEmotionClassifier):
    """
    PhoBERT model for emotion classification with optional emoji support.
    
    Args:
        model_name: Name of the pretrained PhoBERT model
        num_labels: Number of emotion classes
        dropout_prob: Dropout probability for the classifier
        hidden_size: Hidden size of the PhoBERT model (default: 768)
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = 7,
        dropout_prob: float = 0.1,
        hidden_size: int = 768,
        loss_fn: Optional[nn.Module] = None
    ):
        super().__init__(num_labels=num_labels)
        
        # Load PhoBERT configuration
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Load pretrained PhoBERT model
        self.phobert = AutoModel.from_pretrained(model_name, config=self.config)
        
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
        # Get PhoBERT outputs
        outputs = self.phobert(
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
        return self.phobert.embeddings.word_embeddings
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resize token embeddings to accommodate new tokens (e.g., emojis).
        
        Args:
            new_num_tokens: New vocabulary size
        """
        self.phobert.resize_token_embeddings(new_num_tokens)

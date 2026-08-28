"""
Dataset classes and data loading utilities for emotion classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class EmotionDataset(Dataset):
    """
    Custom Dataset for Vietnamese emotion classification.
    
    Args:
        texts: List of input texts
        labels: List of emotion labels (integers)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_uit_vsmec_dataset(
    dataset_name: str = "uit-nlp/vietnamese_students_feedback",
    cache_dir: Optional[str] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Load UIT-VSMEC dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        cache_dir: Directory to cache the dataset
    
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Extract splits
    train_data = {
        'texts': dataset['train']['sentence'],
        'labels': dataset['train']['label']
    }
    
    val_data = {
        'texts': dataset['validation']['sentence'],
        'labels': dataset['validation']['label']
    }
    
    test_data = {
        'texts': dataset['test']['sentence'],
        'labels': dataset['test']['label']
    }
    
    print(f"Train size: {len(train_data['texts'])}")
    print(f"Validation size: {len(val_data['texts'])}")
    print(f"Test size: {len(test_data['texts'])}")
    
    return train_data, val_data, test_data


def create_dataloaders(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test sets.
    
    Args:
        train_data: Dictionary with 'texts' and 'labels' keys
        val_data: Dictionary with 'texts' and 'labels' keys
        test_data: Dictionary with 'texts' and 'labels' keys
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = EmotionDataset(
        train_data['texts'],
        train_data['labels'],
        tokenizer,
        max_length
    )
    
    val_dataset = EmotionDataset(
        val_data['texts'],
        val_data['labels'],
        tokenizer,
        max_length
    )
    
    test_dataset = EmotionDataset(
        test_data['texts'],
        test_data['labels'],
        tokenizer,
        max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def preprocess_text(text: str, use_segmentation: bool = False) -> str:
    """
    Preprocess Vietnamese text.
    
    Args:
        text: Input text
        use_segmentation: Whether to use word segmentation (VnCoreNLP)
    
    Returns:
        Preprocessed text
    """
    # Basic preprocessing
    text = text.strip()
    
    # TODO: Add VnCoreNLP word segmentation if needed
    if use_segmentation:
        # This requires VnCoreNLP to be set up
        pass
    
    return text

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
    
    Raises:
        ValueError: If dataset format is invalid or missing required splits/keys
        RuntimeError: If dataset download fails
    """
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}'. "
            f"Check your internet connection and dataset name. Error: {str(e)}"
        ) from e
    
    # Validate required splits exist
    required_splits = ['train', 'validation', 'test']
    missing_splits = [split for split in required_splits if split not in dataset]
    if missing_splits:
        raise ValueError(
            f"Dataset is missing required splits: {missing_splits}. "
            f"Available splits: {list(dataset.keys())}"
        )
    
    # Validate required keys exist in each split
    required_keys = ['sentence', 'label']
    for split_name in required_splits:
        split_data = dataset[split_name]
        missing_keys = [key for key in required_keys if key not in split_data.column_names]
        if missing_keys:
            raise ValueError(
                f"Split '{split_name}' is missing required keys: {missing_keys}. "
                f"Available keys: {split_data.column_names}"
            )
    
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
    
    # Validate data integrity
    for split_name, split_data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        # Check for empty texts
        empty_count = sum(1 for text in split_data['texts'] if not text or not text.strip())
        if empty_count > 0:
            print(f"Warning: {split_name} split has {empty_count} empty texts")
        
        # Check label range
        labels = split_data['labels']
        if labels:
            min_label = min(labels)
            max_label = max(labels)
            if min_label < 0:
                raise ValueError(f"Invalid labels in {split_name} split: found negative label {min_label}")
            print(f"{split_name.capitalize()} label range: [{min_label}, {max_label}]")
    
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
    
    Note:
        Word segmentation with VnCoreNLP is recommended for PhoBERT
        but requires VnCoreNLP to be installed and configured.
        For BamiBERT, word segmentation is not required.
    """
    # Basic preprocessing
    text = text.strip()
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Word segmentation using VnCoreNLP (if enabled and available)
    if use_segmentation:
        try:
            # VnCoreNLP integration would go here
            # This requires vncorenlp to be installed:
            # pip install vncorenlp
            # And VnCoreNLP-1.1.1.jar to be downloaded
            # 
            # Example usage:
            # from vncorenlp import VnCoreNLP
            # annotator = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg")
            # sentences = annotator.tokenize(text)
            # text = ' '.join([' '.join(sent) for sent in sentences])
            
            # For now, just log a warning
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Word segmentation requested but VnCoreNLP is not configured. "
                "Returning text without segmentation. "
                "To use word segmentation, install vncorenlp and configure vncorenlp_path in config."
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in word segmentation: {e}. Returning unsegmented text.")
    
    return text

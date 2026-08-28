"""
Dataset classes and data loading utilities for emotion classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from src.utils.labels import convert_label_to_tensor
from configs.config import Config


class EmotionDataset(Dataset):
    """
    Custom Dataset for Vietnamese emotion classification.
    
    Args:
        texts: List of input texts
        labels: List of emotion labels (integers or list of integers)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_labels: Total number of labels available
        problem_type: Type of classification
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Union[List[int], List[List[int]]],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        num_labels: int = 7,
        problem_type: str = "multiclass_classification"
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        self.problem_type = problem_type
        
        self._validate_labels()
        
    def _validate_labels(self):
        """Validate that labels match the specified problem_type and num_labels."""
        if not self.labels:
            return
            
        for i, label in enumerate(self.labels[:10]):  # Sample check
            try:
                convert_label_to_tensor(label, self.num_labels, self.problem_type)
            except Exception as e:
                raise ValueError(f"Invalid label at index {i}: {label}. Error: {e}")
    
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
        
        label_tensor = convert_label_to_tensor(label, self.num_labels, self.problem_type)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
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
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
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
    
    # Determine column names (handle both lowercase and capitalized versions)
    sample_split = dataset['train']
    column_names = sample_split.column_names
    
    # Find text column (sentence, Sentence, text, Text, etc.)
    text_col = None
    for col in ['sentence', 'Sentence', 'text', 'Text']:
        if col in column_names:
            text_col = col
            break
    
    # Find label column (label, Label, emotion, Emotion, etc.)
    label_col = None
    for col in ['label', 'Label', 'emotion', 'Emotion']:
        if col in column_names:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find required columns. Available columns: {column_names}. "
            f"Expected text column (sentence/text) and label column (label/emotion)."
        )
    
    print(f"Using column mapping: text='{text_col}', label='{label_col}'")
    
    # Define label mapping (emotion names to integers)
    label_to_int = {
        "Other": 0,
        "Disgust": 1,
        "Enjoyment": 2,
        "Sadness": 3,
        "Fear": 4,
        "Surprise": 5,
        "Anger": 6
    }
    
    # Extract splits and convert labels
    train_data = {
        'texts': dataset['train'][text_col],
        'labels': [label_to_int.get(label, 0) for label in dataset['train'][label_col]]
    }
    
    val_data = {
        'texts': dataset['validation'][text_col],
        'labels': [label_to_int.get(label, 0) for label in dataset['validation'][label_col]]
    }
    
    test_data = {
        'texts': dataset['test'][text_col],
        'labels': [label_to_int.get(label, 0) for label in dataset['test'][label_col]]
    }
    
    # Validate data integrity
    for split_name, split_data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        # Check for empty texts
        empty_count = sum(1 for text in split_data['texts'] if not text or not text.strip())
        if empty_count > 0:
            print(f"Warning: {split_name} split has {empty_count} empty texts")
        
        # Check label range - only if labels are numeric
        labels = split_data['labels']
        if labels:
            # Check if labels are numeric
            if isinstance(labels[0], (int, float)):
                min_label = min(labels)
                max_label = max(labels)
                if min_label < 0:
                    raise ValueError(f"Invalid labels in {split_name} split: found negative label {min_label}")
                print(f"{split_name.capitalize()} label range: [{min_label}, {max_label}]")
            else:
                # String labels - just show unique values
                unique_labels = list(set(labels))
                print(f"{split_name.capitalize()} unique labels: {len(unique_labels)}")
    
    print(f"Train size: {len(train_data['texts'])}")
    print(f"Validation size: {len(val_data['texts'])}")
    print(f"Test size: {len(test_data['texts'])}")
    
    return train_data, val_data, test_data


def load_vigoemotions_dataset(
    dataset_name: str = "uitnlp/vigoemotions",
    cache_dir: Optional[str] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Load ViGoEmotions dataset from HuggingFace.
    """
    print(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}'. "
            f"Check your internet connection and dataset name. Error: {str(e)}"
        ) from e
        
    required_splits = ['train', 'validation', 'test']
    missing_splits = [split for split in required_splits if split not in dataset]
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}.")
        
    required_keys = ['text', 'labels']
    for split_name in required_splits:
        split_data = dataset[split_name]
        missing_keys = [key for key in required_keys if key not in split_data.column_names]
        if missing_keys:
            raise ValueError(f"Split '{split_name}' is missing keys: {missing_keys}.")
            
    train_data = {'texts': dataset['train']['text'], 'labels': dataset['train']['labels']}
    val_data = {'texts': dataset['validation']['text'], 'labels': dataset['validation']['labels']}
    test_data = {'texts': dataset['test']['text'], 'labels': dataset['test']['labels']}
    
    for split_name, split_data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        empty_count = sum(1 for text in split_data['texts'] if not text or not text.strip())
        if empty_count > 0:
            print(f"Warning: {split_name} split has {empty_count} empty texts")
            
        all_zeros = sum(1 for label in split_data['labels'] if not label)
        if all_zeros > 0:
            print(f"Warning: {split_name} split has {all_zeros} samples with empty labels")
            
    print(f"Train size: {len(train_data['texts'])}")
    print(f"Validation size: {len(val_data['texts'])}")
    print(f"Test size: {len(test_data['texts'])}")
    
    return train_data, val_data, test_data


def load_dataset_by_config(
    config: Config,
    cache_dir: Optional[str] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Load dataset based on configuration.
    """
    if config.dataset_name == "uitnlp/vigoemotions":
        return load_vigoemotions_dataset(config.dataset_name, cache_dir)
    else:
        return load_uit_vsmec_dataset(config.dataset_name, cache_dir)


def create_dataloaders(
    config: Config,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test sets.
    """
    # Create datasets
    train_dataset = EmotionDataset(
        texts=train_data['texts'],
        labels=train_data['labels'],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_labels=config.num_labels,
        problem_type=config.problem_type
    )
    
    val_dataset = EmotionDataset(
        texts=val_data['texts'],
        labels=val_data['labels'],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_labels=config.num_labels,
        problem_type=config.problem_type
    )
    
    test_dataset = EmotionDataset(
        texts=test_data['texts'],
        labels=test_data['labels'],
        tokenizer=tokenizer,
        max_length=config.max_length,
        num_labels=config.num_labels,
        problem_type=config.problem_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
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

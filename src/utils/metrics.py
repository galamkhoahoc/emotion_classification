"""
Metrics computation utilities for model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels [num_samples]
        labels: Ground truth labels [num_samples]
        label_names: Names of the emotion labels
    
    Returns:
        Dictionary of metrics
    """
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',
        zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        normalize: Whether to normalize the confusion matrix
    
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str = 'Confusion Matrix',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        label_names: Names of the emotion labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if cm.dtype == float else 'd',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=True
    )
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
):
    """
    Print detailed classification report.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        label_names: Names of the emotion labels
    """
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        digits=4
    )
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")


def get_predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Get predicted class labels from logits.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
    
    Returns:
        Predicted class labels [batch_size]
    """
    return torch.argmax(logits, dim=1)

"""
Metrics computation utilities for model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    hamming_loss
)
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_multiclass_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute classification metrics for multiclass problems."""
    accuracy = accuracy_score(labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }


def compute_multilabel_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute classification metrics for multilabel problems."""
    subset_accuracy = accuracy_score(labels, predictions)
    h_loss = hamming_loss(labels, predictions)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )
    
    # Calculate sample-based F1 (often useful for multilabel)
    sample_f1s = []
    for pred, label in zip(predictions, labels):
        intersection = np.sum(np.logical_and(pred, label))
        union = np.sum(pred) + np.sum(label)
        if union == 0:
            sample_f1s.append(1.0 if np.sum(label) == 0 and np.sum(pred) == 0 else 0.0)
        else:
            sample_f1s.append(2 * intersection / union)
    sample_f1 = np.mean(sample_f1s)
    
    return {
        'subset_accuracy': subset_accuracy,
        'hamming_loss': h_loss,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_sample': sample_f1
    }


def compute_per_label_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each individual label in multilabel classification."""
    if len(label_names) != labels.shape[1]:
        raise ValueError(f"label_names length ({len(label_names)}) does not match number of labels ({labels.shape[1]})")
        
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    metrics = {}
    for i, name in enumerate(label_names):
        metrics[name] = {
            'precision': float(precisions[i]),
            'recall': float(recalls[i]),
            'f1': float(f1s[i]),
            'support': int(supports[i])
        }
    return metrics


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    problem_type: str = "multiclass_classification",
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Unified metrics router."""
    if problem_type == "multiclass_classification":
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            raise ValueError("Expected 1D predictions for multiclass classification.")
        return compute_multiclass_metrics(predictions, labels, label_names)
    elif problem_type == "multilabel_classification":
        if len(predictions.shape) != 2 or len(labels.shape) != 2:
            raise ValueError("Expected 2D predictions and labels for multilabel classification.")
        return compute_multilabel_metrics(predictions, labels, label_names)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def get_probabilities_from_logits(
    logits: torch.Tensor,
    problem_type: str = "multiclass_classification"
) -> torch.Tensor:
    """Convert logits to probabilities."""
    if problem_type == "multiclass_classification":
        return torch.softmax(logits, dim=1)
    elif problem_type == "multilabel_classification":
        return torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def get_predictions_from_logits(
    logits: torch.Tensor,
    problem_type: str = "multiclass_classification",
    threshold: float = 0.5
) -> torch.Tensor:
    """Convert logits to predicted classes."""
    probs = get_probabilities_from_logits(logits, problem_type)
    if problem_type == "multiclass_classification":
        return torch.argmax(probs, dim=1)
    elif problem_type == "multilabel_classification":
        return (probs >= threshold).to(torch.float32)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def print_metrics_report(metrics: Dict[str, float], problem_type: str = "multiclass_classification"):
    """Print formatted metrics report."""
    print("\n" + "="*60)
    print("METRICS REPORT")
    print("="*60)
    if problem_type == "multiclass_classification":
        print(f"Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"F1 Macro:           {metrics.get('f1_macro', 0):.4f}")
        print(f"F1 Weighted:        {metrics.get('f1_weighted', 0):.4f}")
    else:
        print(f"Subset Accuracy:    {metrics.get('subset_accuracy', 0):.4f}")
        print(f"Hamming Loss:       {metrics.get('hamming_loss', 0):.4f}")
        print(f"F1 Macro:           {metrics.get('f1_macro', 0):.4f}")
        print(f"F1 Micro:           {metrics.get('f1_micro', 0):.4f}")
        print(f"F1 Sample:          {metrics.get('f1_sample', 0):.4f}")
    print("="*60 + "\n")


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """Compute confusion matrix for multiclass."""
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
    """Plot confusion matrix as a heatmap."""
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
    
    plt.close()


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
):
    """Print detailed classification report."""
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

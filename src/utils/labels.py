"""
Label conversion utilities for multiclass and multilabel classification.
"""

import torch
from typing import List, Union


def convert_label_to_tensor(
    label: Union[int, List[int]],
    num_labels: int,
    problem_type: str = "multiclass_classification"
) -> torch.Tensor:
    """
    Convert a label (int or list of ints) to a tensor.
    
    Args:
        label: Int for multiclass, List[int] for multilabel.
        num_labels: Total number of labels available.
        problem_type: Type of classification ("multiclass_classification" or "multilabel_classification").
        
    Returns:
        torch.Tensor (long for multiclass, float32 binary vector for multilabel).
    """
    if problem_type == "multiclass_classification":
        if not isinstance(label, int):
            try:
                label = int(label)
            except (ValueError, TypeError):
                raise TypeError(f"Label must be an integer for multiclass, got {type(label)}")
                
        if label < 0 or label >= num_labels:
            raise ValueError(f"Label index {label} out of range for num_labels={num_labels}")
            
        return torch.tensor(label, dtype=torch.long)
        
    elif problem_type == "multilabel_classification":
        if not isinstance(label, (list, tuple)):
            raise TypeError(f"Label must be a list/tuple for multilabel, got {type(label)}")
            
        if len(label) == 0:
            raise ValueError("Multilabel samples must have at least one label.")
            
        tensor = torch.zeros(num_labels, dtype=torch.float32)
        for idx in label:
            if idx < 0 or idx >= num_labels:
                raise ValueError(f"Label index {idx} out of range for num_labels={num_labels}")
            tensor[idx] = 1.0
            
        return tensor
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def convert_binary_vector_to_labels(
    binary_vector: Union[torch.Tensor, List[float]],
    emotion_labels: List[str]
) -> List[str]:
    """
    Convert a binary vector to a list of emotion names.
    
    Args:
        binary_vector: Binary vector of predictions (1.0 for present, 0.0 for absent).
        emotion_labels: List of emotion names corresponding to indices.
        
    Returns:
        List of predicted emotion names.
    """
    if isinstance(binary_vector, torch.Tensor):
        binary_vector = binary_vector.tolist()
        
    if len(binary_vector) != len(emotion_labels):
        raise ValueError(f"Binary vector length ({len(binary_vector)}) does not match emotion labels ({len(emotion_labels)})")
        
    result = []
    for i, val in enumerate(binary_vector):
        if val > 0.5:  # Consider present if > 0.5
            result.append(emotion_labels[i])
            
    return result

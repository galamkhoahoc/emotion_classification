"""
Prediction utilities for inference and threshold optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import matplotlib.pyplot as plt

from src.models.base_classifier import BaseEmotionClassifier
from configs.config import Config
from src.utils.metrics import get_predictions_from_logits, get_probabilities_from_logits, compute_multilabel_metrics
from src.utils.labels import convert_binary_vector_to_labels


class Predictor:
    """
    Handles model inference for both single texts and batches.
    """
    def __init__(self, model: BaseEmotionClassifier, tokenizer, config: Config, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model.eval()
        
    def predict_text(self, text: str, threshold: Optional[float] = None) -> Dict:
        """Predict emotion for a single text."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            logits = outputs['logits']
            
        return self._process_logits(logits, threshold)
        
    def predict_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, threshold: Optional[float] = None) -> Dict:
        """Predict emotion for a batch of tokenized texts."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
        return self._process_logits(logits, threshold)
        
    def _process_logits(self, logits: torch.Tensor, threshold: Optional[float] = None) -> Dict:
        """Process model logits into probabilities and predictions."""
        thresh = threshold if threshold is not None else self.config.sigmoid_threshold
        
        probs = get_probabilities_from_logits(logits, self.config.problem_type)
        preds = get_predictions_from_logits(logits, self.config.problem_type, thresh)
        
        results = {
            "probabilities": probs.cpu().numpy(),
            "predictions": preds.cpu().numpy()
        }
        
        # Add human-readable labels if it's a single prediction
        if logits.size(0) == 1:
            if self.config.is_multilabel():
                labels = convert_binary_vector_to_labels(preds[0], self.config.emotion_labels)
            else:
                idx = int(preds[0].item())
                labels = [self.config.emotion_labels[idx]]
            results["emotion_labels"] = labels
            
        return results


class ThresholdOptimizer:
    """
    Optimizes the decision threshold for multilabel classification.
    """
    def __init__(self, config: Config):
        if not config.is_multilabel():
            raise ValueError("ThresholdOptimizer is only for multilabel classification.")
        self.config = config
        self.thresholds = np.arange(0.1, 0.95, 0.05)
        
    def sweep_thresholds(self, all_logits: torch.Tensor, all_labels: torch.Tensor) -> Dict:
        """
        Sweep across multiple thresholds to find the best one for f1_macro.
        """
        best_threshold = 0.5
        best_f1 = 0.0
        results_history = []
        
        for t in self.thresholds:
            preds = get_predictions_from_logits(
                all_logits, 
                problem_type="multilabel_classification", 
                threshold=t
            ).cpu().numpy()
            
            labels = all_labels.cpu().numpy()
            metrics = compute_multilabel_metrics(preds, labels)
            
            results_history.append({
                "threshold": float(t),
                "f1_macro": metrics["f1_macro"],
                "f1_micro": metrics["f1_micro"],
                "f1_sample": metrics["f1_sample"]
            })
            
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_threshold = float(t)
                
        return {
            "best_threshold": best_threshold,
            "best_f1_macro": best_f1,
            "history": results_history
        }
        
    def plot_threshold_curve(self, sweep_results: Dict, save_path: str):
        """Plot metrics vs. threshold."""
        history = sweep_results["history"]
        thresholds = [h["threshold"] for h in history]
        f1_macros = [h["f1_macro"] for h in history]
        f1_micros = [h["f1_micro"] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_macros, marker='o', label='F1 Macro')
        plt.plot(thresholds, f1_micros, marker='s', label='F1 Micro')
        
        best_t = sweep_results["best_threshold"]
        plt.axvline(x=best_t, color='r', linestyle='--', label=f'Best Threshold ({best_t:.2f})')
        
        plt.title('Threshold Optimization Curve')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Threshold curve saved to {save_path}")
        plt.close()

"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json


def setup_logger(
    name: str = "ViEmoText",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: Dict[str, float],
    epoch: Optional[int] = None,
    prefix: str = "",
    logger: Optional[logging.Logger] = None
):
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        epoch: Epoch number (optional)
        prefix: Prefix for the log message
        logger: Logger instance (creates new one if None)
    """
    if logger is None:
        logger = logging.getLogger("ViEmoText")
    
    # Create log message
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    prefix_str = f"{prefix} - " if prefix else ""
    
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    logger.info(f"{epoch_str}{prefix_str}{metrics_str}")


def save_metrics_to_file(
    metrics: Dict,
    filepath: str,
    mode: str = 'w'
):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
        mode: File mode ('w' for write, 'a' for append)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, mode, encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write('\n')
    
    print(f"Metrics saved to {filepath}")


class MetricsTracker:
    """
    Track and store metrics during training.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def update(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None
    ):
        """Update metrics history."""
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if train_metrics is not None:
            self.history['train_metrics'].append(train_metrics)
        if val_metrics is not None:
            self.history['val_metrics'].append(val_metrics)
    
    def get_best_epoch(self, metric: str = 'f1_macro') -> int:
        """Get the epoch with the best validation metric."""
        if not self.history['val_metrics']:
            return -1
        
        values = [m.get(metric, 0) for m in self.history['val_metrics']]
        return int(np.argmax(values))
    
    def save(self, filepath: str):
        """Save metrics history to file."""
        save_metrics_to_file(self.history, filepath)
    
    def load(self, filepath: str):
        """Load metrics history from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)

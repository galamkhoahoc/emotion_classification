"""
Configuration file for ViEmoText model training and evaluation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Config:
    """Main configuration class for the project."""
    
    # Project metadata
    project_name: str = "ViEmoText"
    version: str = "2.0"
    
    # Model configuration
    model_name: str = "vinai/phobert-base"
    num_labels: int = 7
    max_length: int = 256
    
    # Emotion labels
    emotion_labels: List[str] = field(default_factory=lambda: [
        "Other",
        "Disgust", 
        "Enjoyment",
        "Sadness",
        "Fear",
        "Surprise",
        "Anger"
    ])
    
    # Dataset configuration
    dataset_name: str = "uit-nlp/vietnamese_students_feedback"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Optimizer configuration
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Loss configuration
    loss_type: str = "cross_entropy"  # Options: "cross_entropy", "focal_loss", "weighted_ce"
    focal_loss_alpha: Optional[float] = None
    focal_loss_gamma: float = 2.0
    
    # Emoji configuration
    enable_emoji_embedding: bool = True
    emoji_mapping_file: Optional[str] = None
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    cache_dir: str = ".cache"
    
    # Training configuration
    seed: int = 42
    device: str = "cuda"  # Options: "cuda", "cpu", "mps"
    num_workers: int = 4
    fp16: bool = True
    
    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # VnCoreNLP configuration (for preprocessing)
    vncorenlp_path: Optional[str] = None
    use_word_segmentation: bool = True
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)

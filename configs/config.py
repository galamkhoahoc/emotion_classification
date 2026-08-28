"""
Configuration file for ViEmoText model training and evaluation.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Config:
    """Main configuration class for the project."""
    
    # Project metadata
    project_name: str = "ViEmoText"
    version: str = "2.0"
    
    # Model configuration
    model_type: str = "phobert"  # Options: "phobert", "bamibert"
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
        """Validate configuration and set model-specific defaults."""
        logger = logging.getLogger(__name__)
        
        # Validate model_type
        valid_types = ["phobert", "bamibert"]
        if self.model_type.lower() not in valid_types:
            raise ValueError(
                f"Invalid model_type: '{self.model_type}'. "
                f"Must be one of {valid_types}"
            )
        
        # Normalize model_type to lowercase
        self.model_type = self.model_type.lower()
        
        # Set model-specific defaults
        if self.model_type == "phobert":
            self.model_name = "vinai/phobert-base"
            # PhoBERT max context: 256 tokens
            if self.max_length > 256:
                logger.warning(
                    f"max_length ({self.max_length}) exceeds PhoBERT's "
                    f"recommended maximum (256). Setting to 256."
                )
                self.max_length = 256
        
        elif self.model_type == "bamibert":
            self.model_name = "Qualcomm-AI-Research/BamiBERT"
            # BamiBERT default max_length is 2048 (unless user set it lower)
            if self.max_length == 256:
                # User likely didn't override, set to BamiBERT's default
                self.max_length = 2048
            # BamiBERT max context: 2048 tokens
            if self.max_length > 2048:
                logger.warning(
                    f"max_length ({self.max_length}) exceeds BamiBERT's "
                    f"maximum (2048). Setting to 2048."
                )
                self.max_length = 2048
            
            # BamiBERT doesn't need word segmentation
            if self.use_word_segmentation:
                logger.warning(
                    "BamiBERT works with raw text. "
                    "Disabling word segmentation."
                )
                self.use_word_segmentation = False
        
        # Validate num_labels
        if self.num_labels < 2:
            raise ValueError(
                f"num_labels must be at least 2, got {self.num_labels}"
            )
        
        # Create necessary directories
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

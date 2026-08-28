"""
Configuration file for ViEmoText model training and evaluation.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# Dataset presets for dual-mode classification support
DATASET_PRESETS = {
    "tridm/UIT-VSMEC": {
        "problem_type": "multiclass_classification",
        "num_labels": 7,
        "emotion_labels": [
            "Other",
            "Disgust",
            "Enjoyment",
            "Sadness",
            "Fear",
            "Surprise",
            "Anger"
        ]
    },
    "uit-nlp/vietnamese_students_feedback": {
        "problem_type": "multiclass_classification",
        "num_labels": 7,
        "emotion_labels": [
            "Other",
            "Disgust",
            "Enjoyment",
            "Sadness",
            "Fear",
            "Surprise",
            "Anger"
        ]
    },
    "uitnlp/vigoemotions": {
        "problem_type": "multilabel_classification",
        "num_labels": 28,
        "emotion_labels": [
            "amusement",
            "excitement",
            "joy",
            "love",
            "desire",
            "optimism",
            "caring",
            "pride",
            "admiration",
            "gratitude",
            "relief",
            "approval",
            "realization",
            "surprise",
            "curiosity",
            "confusion",
            "fear",
            "nervousness",
            "remorse",
            "embarrassment",
            "disappointment",
            "sadness",
            "grief",
            "disgust",
            "anger",
            "annoyance",
            "disapproval",
            "neutral"
        ]
    }
}


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
    
    # Multilabel classification configuration
    problem_type: str = "multiclass_classification"  # Options: "multiclass_classification", "multilabel_classification"
    sigmoid_threshold: float = 0.5  # For multilabel prediction (threshold for converting probabilities to labels)
    
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
        
        # Auto-configure based on dataset preset if available
        if self.dataset_name in DATASET_PRESETS:
            preset = DATASET_PRESETS[self.dataset_name]
            
            default_problem_type = "multiclass_classification"
            default_num_labels = 7
            default_emotion_labels = ["Other", "Disgust", "Enjoyment", "Sadness", "Fear", "Surprise", "Anger"]
            
            # Only override if the user did not explicitly set a different value
            if self.problem_type == default_problem_type:
                self.problem_type = preset.get("problem_type", self.problem_type)
            if self.num_labels == default_num_labels:
                self.num_labels = preset.get("num_labels", self.num_labels)
            if self.emotion_labels == default_emotion_labels:
                self.emotion_labels = preset.get("emotion_labels", self.emotion_labels)
            
        # Validate based on problem type
        valid_problem_types = ["multiclass_classification", "multilabel_classification"]
        if self.problem_type not in valid_problem_types:
            raise ValueError(
                f"Invalid problem_type: '{self.problem_type}'. "
                f"Must be one of {valid_problem_types}"
            )
            
        # Validate sigmoid_threshold
        if not (0.0 <= self.sigmoid_threshold <= 1.0):
            raise ValueError(f"sigmoid_threshold must be between 0 and 1, got {self.sigmoid_threshold}")
            
        if self.is_multilabel():
            self._validate_multilabel_config()
        else:
            self._validate_multiclass_config()
            
        # Validate num_labels
        if self.num_labels < 2:
            raise ValueError(
                f"num_labels must be at least 2, got {self.num_labels}"
            )
            
        if len(self.emotion_labels) != self.num_labels:
            logger.warning(
                f"Length of emotion_labels ({len(self.emotion_labels)}) does not match "
                f"num_labels ({self.num_labels})."
            )
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _validate_multilabel_config(self):
        """Validate configuration specific to multilabel classification."""
        pass
            
    def _validate_multiclass_config(self):
        """Validate configuration specific to multiclass classification."""
        pass
    
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
    
    def is_multilabel(self) -> bool:
        """Check if configuration is for multilabel classification."""
        return self.problem_type == "multilabel_classification"

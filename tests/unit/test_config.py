"""
Unit tests for Config class.

Tests Requirements:
- 1.1: Config accepts model_type parameter
- 1.4: Config raises error for invalid model_type
- 1.5: Config sets model_name for phobert
- 1.6: Config sets model_name for bamibert
- 3.1, 3.2: Context length handling
- 4.5: Word segmentation control
- 11.1-11.5: Configuration validation
"""

import pytest
from configs.config import Config


class TestConfigModelType:
    """Test suite for model_type parameter in Config."""

    def test_default_model_type_is_phobert(self):
        """Test that default model_type is 'phobert' for backward compatibility."""
        config = Config()
        assert config.model_type == "phobert"

    def test_accepts_phobert_lowercase(self):
        """Test that Config accepts 'phobert' (lowercase)."""
        config = Config(model_type="phobert")
        assert config.model_type == "phobert"

    def test_accepts_phobert_uppercase(self):
        """Test that Config accepts 'PhoBERT' (mixed case) and normalizes it."""
        config = Config(model_type="PhoBERT")
        assert config.model_type == "phobert"

    def test_accepts_phobert_allcaps(self):
        """Test that Config accepts 'PHOBERT' (uppercase) and normalizes it."""
        config = Config(model_type="PHOBERT")
        assert config.model_type == "phobert"

    def test_accepts_bamibert_lowercase(self):
        """Test that Config accepts 'bamibert' (lowercase)."""
        config = Config(model_type="bamibert")
        assert config.model_type == "bamibert"

    def test_accepts_bamibert_mixedcase(self):
        """Test that Config accepts 'BamiBERT' (mixed case) and normalizes it."""
        config = Config(model_type="BamiBERT")
        assert config.model_type == "bamibert"

    def test_accepts_bamibert_allcaps(self):
        """Test that Config accepts 'BAMIBERT' (uppercase) and normalizes it."""
        config = Config(model_type="BAMIBERT")
        assert config.model_type == "bamibert"

    def test_rejects_invalid_model_type(self):
        """Test that Config raises ValueError for invalid model_type."""
        with pytest.raises(ValueError) as exc_info:
            Config(model_type="invalid_model")
        
        error_message = str(exc_info.value)
        assert "Invalid model_type" in error_message
        assert "invalid_model" in error_message
        assert "phobert" in error_message
        assert "bamibert" in error_message

    def test_rejects_empty_string(self):
        """Test that Config raises ValueError for empty string model_type."""
        with pytest.raises(ValueError) as exc_info:
            Config(model_type="")
        
        error_message = str(exc_info.value)
        assert "Invalid model_type" in error_message

    def test_rejects_bert_model(self):
        """Test that Config rejects 'bert' which is not a valid model_type."""
        with pytest.raises(ValueError):
            Config(model_type="bert")

    def test_rejects_roberta_model(self):
        """Test that Config rejects 'roberta' which is not a valid model_type."""
        with pytest.raises(ValueError):
            Config(model_type="roberta")

    def test_preserves_other_config_fields_with_phobert(self):
        """Test that setting model_type='phobert' preserves other config fields."""
        config = Config(
            model_type="phobert",
            batch_size=32,
            learning_rate=1e-5,
            num_epochs=20
        )
        
        assert config.model_type == "phobert"
        assert config.batch_size == 32
        assert config.learning_rate == 1e-5
        assert config.num_epochs == 20

    def test_preserves_other_config_fields_with_bamibert(self):
        """Test that setting model_type='bamibert' preserves other config fields."""
        config = Config(
            model_type="bamibert",
            batch_size=8,
            learning_rate=2e-5,
            num_epochs=15
        )
        
        assert config.model_type == "bamibert"
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 15

    def test_config_to_dict_includes_model_type(self):
        """Test that to_dict() includes the model_type field."""
        config = Config(model_type="bamibert")
        config_dict = config.to_dict()
        
        assert "model_type" in config_dict
        assert config_dict["model_type"] == "bamibert"

    def test_config_from_dict_with_model_type(self):
        """Test that from_dict() correctly sets model_type."""
        config_dict = {
            "model_type": "bamibert",
            "batch_size": 16
        }
        config = Config.from_dict(config_dict)
        
        assert config.model_type == "bamibert"
        assert config.batch_size == 16


class TestConfigModelName:
    """Test suite for model_name auto-setting based on model_type.
    
    Validates Requirements: 1.5, 1.6
    """

    def test_phobert_sets_correct_model_name(self):
        """Test model_name is 'vinai/phobert-base' when model_type='phobert'."""
        config = Config(model_type="phobert")
        assert config.model_name == "vinai/phobert-base"

    def test_bamibert_sets_correct_model_name(self):
        """Test model_name is 'Qualcomm-AI-Research/BamiBERT' when model_type='bamibert'."""
        config = Config(model_type="bamibert")
        assert config.model_name == "Qualcomm-AI-Research/BamiBERT"

    def test_model_name_overridden_by_model_type(self):
        """Test that model_name is always set based on model_type, ignoring explicit model_name."""
        config = Config(model_type="bamibert", model_name="some-other-model")
        assert config.model_name == "Qualcomm-AI-Research/BamiBERT"

    def test_default_model_name_is_phobert(self):
        """Test default model_name is vinai/phobert-base."""
        config = Config()
        assert config.model_name == "vinai/phobert-base"


class TestConfigMaxLength:
    """Test suite for max_length auto-setting and validation.
    
    Validates Requirements: 3.1, 3.2, 11.2, 11.3
    """

    def test_phobert_default_max_length(self):
        """Test default max_length for PhoBERT is 256."""
        config = Config(model_type="phobert")
        assert config.max_length == 256

    def test_bamibert_default_max_length(self):
        """Test default max_length for BamiBERT is 2048."""
        config = Config(model_type="bamibert")
        assert config.max_length == 2048

    def test_phobert_caps_max_length_at_256(self):
        """Test PhoBERT max_length capped at 256 if exceeding."""
        config = Config(model_type="phobert", max_length=512)
        assert config.max_length == 256

    def test_bamibert_caps_max_length_at_2048(self):
        """Test BamiBERT max_length capped at 2048 if exceeding."""
        config = Config(model_type="bamibert", max_length=4096)
        assert config.max_length == 2048

    def test_phobert_preserves_smaller_max_length(self):
        """Test PhoBERT preserves max_length if smaller than 256."""
        config = Config(model_type="phobert", max_length=128)
        assert config.max_length == 128

    def test_bamibert_preserves_custom_max_length(self):
        """Test BamiBERT preserves custom max_length if less than 2048."""
        config = Config(model_type="bamibert", max_length=512)
        assert config.max_length == 512


class TestConfigWordSegmentation:
    """Test suite for word segmentation control.
    
    Validates Requirements: 4.5, 11.4
    """

    def test_phobert_keeps_word_segmentation_enabled(self):
        """Test PhoBERT keeps use_word_segmentation=True by default."""
        config = Config(model_type="phobert")
        assert config.use_word_segmentation is True

    def test_bamibert_disables_word_segmentation(self):
        """Test BamiBERT auto-disables word segmentation."""
        config = Config(model_type="bamibert")
        assert config.use_word_segmentation is False

    def test_bamibert_overrides_explicit_word_segmentation(self):
        """Test BamiBERT disables segmentation even if explicitly set to True."""
        config = Config(model_type="bamibert", use_word_segmentation=True)
        assert config.use_word_segmentation is False

    def test_phobert_allows_disabling_word_segmentation(self):
        """Test PhoBERT allows explicit disabling of word segmentation."""
        config = Config(model_type="phobert", use_word_segmentation=False)
        assert config.use_word_segmentation is False


class TestConfigNumLabelsValidation:
    """Test suite for num_labels validation.
    
    Validates Requirements: 11.5
    """

    def test_valid_num_labels(self):
        """Test that num_labels >= 2 is accepted."""
        config = Config(num_labels=7)
        assert config.num_labels == 7

    def test_minimum_num_labels(self):
        """Test that num_labels=2 is accepted."""
        config = Config(num_labels=2)
        assert config.num_labels == 2

    def test_rejects_num_labels_less_than_2(self):
        """Test that num_labels < 2 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config(num_labels=1)
        assert "num_labels must be at least 2" in str(exc_info.value)

    def test_rejects_zero_num_labels(self):
        """Test that num_labels=0 raises ValueError."""
        with pytest.raises(ValueError):
            Config(num_labels=0)

    def test_rejects_negative_num_labels(self):
        """Test that negative num_labels raises ValueError."""
        with pytest.raises(ValueError):
            Config(num_labels=-1)


class TestConfigDirectories:
    """Test suite for directory creation in Config."""

    def test_creates_directories_on_initialization(self):
        """Test that Config creates necessary directories on initialization."""
        import os
        
        config = Config()
        
        assert os.path.exists(config.output_dir)
        assert os.path.exists(config.checkpoint_dir)
        assert os.path.exists(config.log_dir)
        assert os.path.exists(config.cache_dir)



class TestConfigMultilabelFields:
    """Test suite for multilabel classification configuration fields.
    
    Validates Requirements: 3.3, 3.4, 3.5
    """

    def test_default_problem_type_is_multiclass(self):
        """Test that default problem_type is 'multiclass_classification'."""
        config = Config()
        assert config.problem_type == "multiclass_classification"

    def test_accepts_multilabel_problem_type(self):
        """Test that Config accepts 'multilabel_classification' problem_type."""
        config = Config(problem_type="multilabel_classification")
        assert config.problem_type == "multilabel_classification"

    def test_default_sigmoid_threshold(self):
        """Test that default sigmoid_threshold is 0.5."""
        config = Config()
        assert config.sigmoid_threshold == 0.5

    def test_custom_sigmoid_threshold(self):
        """Test that Config accepts custom sigmoid_threshold."""
        config = Config(sigmoid_threshold=0.7)
        assert config.sigmoid_threshold == 0.7

    def test_is_multilabel_returns_false_for_multiclass(self):
        """Test is_multilabel() returns False for multiclass configuration."""
        config = Config(problem_type="multiclass_classification")
        assert config.is_multilabel() is False

    def test_is_multilabel_returns_true_for_multilabel(self):
        """Test is_multilabel() returns True for multilabel configuration."""
        config = Config(problem_type="multilabel_classification")
        assert config.is_multilabel() is True

    def test_is_multilabel_with_default_config(self):
        """Test is_multilabel() returns False with default configuration."""
        config = Config()
        assert config.is_multilabel() is False

    def test_problem_type_in_to_dict(self):
        """Test that to_dict() includes problem_type field."""
        config = Config(problem_type="multilabel_classification")
        config_dict = config.to_dict()
        
        assert "problem_type" in config_dict
        assert config_dict["problem_type"] == "multilabel_classification"

    def test_sigmoid_threshold_in_to_dict(self):
        """Test that to_dict() includes sigmoid_threshold field."""
        config = Config(sigmoid_threshold=0.6)
        config_dict = config.to_dict()
        
        assert "sigmoid_threshold" in config_dict
        assert config_dict["sigmoid_threshold"] == 0.6

    def test_from_dict_with_multilabel_fields(self):
        """Test that from_dict() correctly sets multilabel fields."""
        config_dict = {
            "problem_type": "multilabel_classification",
            "sigmoid_threshold": 0.8,
            "num_labels": 28
        }
        config = Config.from_dict(config_dict)
        
        assert config.problem_type == "multilabel_classification"
        assert config.sigmoid_threshold == 0.8
        assert config.num_labels == 28
        assert config.is_multilabel() is True

    def test_multilabel_config_preserves_other_fields(self):
        """Test that multilabel configuration preserves other config fields."""
        config = Config(
            problem_type="multilabel_classification",
            sigmoid_threshold=0.65,
            num_labels=28,
            batch_size=32,
            learning_rate=1e-5
        )
        
        assert config.problem_type == "multilabel_classification"
        assert config.sigmoid_threshold == 0.65
        assert config.num_labels == 28
        assert config.batch_size == 32
        assert config.learning_rate == 1e-5


class TestConfigAutoDetectionAndValidation:
    """Test suite for auto-detection and validation logic.
    
    Validates Requirements: 3.6, 3.7
    """
    def test_vigoemotions_preset_auto_configuration(self):
        config = Config(dataset_name="uitnlp/vigoemotions")
        assert config.problem_type == "multilabel_classification"
        assert config.num_labels == 28
        assert len(config.emotion_labels) == 28
        assert "joy" in config.emotion_labels
        
    def test_invalid_problem_type_rejection(self):
        with pytest.raises(ValueError, match="Invalid problem_type"):
            Config(problem_type="unknown_type")
            
    def test_sigmoid_threshold_validation(self):
        with pytest.raises(ValueError, match="sigmoid_threshold must be between 0 and 1"):
            Config(problem_type="multilabel_classification", sigmoid_threshold=-0.1)
        
        with pytest.raises(ValueError, match="sigmoid_threshold must be between 0 and 1"):
            Config(problem_type="multilabel_classification", sigmoid_threshold=1.5)
            
    def test_num_labels_emotion_labels_mismatch(self, caplog):
        # We expect a warning to be logged
        Config(num_labels=5, emotion_labels=["A", "B", "C"])
        assert "Length of emotion_labels (3) does not match num_labels (5)" in caplog.text

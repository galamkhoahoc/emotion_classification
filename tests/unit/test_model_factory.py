"""
Unit tests for model factory.

Tests Requirements:
- 1.2: Factory instantiates PhoBERT for model_type="phobert"
- 1.3: Factory instantiates BamiBERT for model_type="bamibert"
- 8.1-8.6: Model loading and initialization
- 10.1: Checkpoint loading
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from configs.config import Config
from src.models.base_classifier import BaseEmotionClassifier
from src.models.phobert_emotion import PhoBERTEmotionClassifier


class TestCreateModelPhoBERT:
    """Test suite for factory creating PhoBERT models."""

    @pytest.fixture
    def phobert_config(self):
        """Create a PhoBERT config."""
        return Config(model_type="phobert")

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.PhoBERTEmotionClassifier')
    def test_creates_phobert_model(self, mock_phobert_cls, mock_tokenizer, phobert_config):
        """Test factory creates PhoBERTEmotionClassifier for model_type='phobert'."""
        from src.models.model_factory import create_model
        
        mock_model = MagicMock(spec=PhoBERTEmotionClassifier)
        mock_phobert_cls.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        model, tokenizer = create_model(phobert_config)
        
        mock_phobert_cls.assert_called_once()
        assert model == mock_model

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.PhoBERTEmotionClassifier')
    def test_loads_correct_tokenizer_for_phobert(self, mock_phobert_cls, mock_tokenizer, phobert_config):
        """Test factory loads tokenizer from vinai/phobert-base."""
        from src.models.model_factory import create_model
        
        mock_phobert_cls.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        model, tokenizer = create_model(phobert_config)
        
        mock_tokenizer.from_pretrained.assert_called_once_with("vinai/phobert-base")

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.PhoBERTEmotionClassifier')
    def test_returns_tuple(self, mock_phobert_cls, mock_tokenizer, phobert_config):
        """Test factory returns a tuple of (model, tokenizer)."""
        from src.models.model_factory import create_model
        
        mock_phobert_cls.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        result = create_model(phobert_config)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.PhoBERTEmotionClassifier')
    def test_passes_num_labels_to_model(self, mock_phobert_cls, mock_tokenizer, phobert_config):
        """Test factory passes num_labels from config to model."""
        from src.models.model_factory import create_model
        
        mock_phobert_cls.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        create_model(phobert_config)
        
        call_kwargs = mock_phobert_cls.call_args
        assert call_kwargs.kwargs.get('num_labels') == 7 or call_kwargs[1].get('num_labels') == 7


class TestCreateModelBamiBERT:
    """Test suite for factory creating BamiBERT models."""

    @pytest.fixture
    def bamibert_config(self):
        """Create a BamiBERT config."""
        return Config(model_type="bamibert")

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.BamiBERTEmotionClassifier')
    def test_creates_bamibert_model(self, mock_bamibert_cls, mock_tokenizer, bamibert_config):
        """Test factory creates BamiBERTEmotionClassifier for model_type='bamibert'."""
        from src.models.model_factory import create_model
        
        mock_model = MagicMock()
        mock_bamibert_cls.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        model, tokenizer = create_model(bamibert_config)
        
        mock_bamibert_cls.assert_called_once()
        assert model == mock_model

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.BamiBERTEmotionClassifier')
    def test_loads_correct_tokenizer_for_bamibert(self, mock_bamibert_cls, mock_tokenizer, bamibert_config):
        """Test factory loads tokenizer from Qualcomm-AI-Research/BamiBERT."""
        from src.models.model_factory import create_model
        
        mock_bamibert_cls.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        model, tokenizer = create_model(bamibert_config)
        
        mock_tokenizer.from_pretrained.assert_called_once_with("Qualcomm-AI-Research/BamiBERT")


class TestCreateModelInvalid:
    """Test suite for factory error handling."""

    def test_raises_value_error_for_invalid_model_type(self):
        """Test factory raises ValueError for invalid model_type."""
        from src.models.model_factory import create_model
        
        # Config already validates, so we need to mock config
        config = MagicMock()
        config.model_type = "invalid"
        config.model_name = "some-model"
        
        with pytest.raises((ValueError, RuntimeError)):
            create_model(config)


class TestCreateModelFromCheckpoint:
    """Test suite for checkpoint loading function."""

    @patch('src.models.model_factory.torch')
    @patch('src.models.model_factory.create_model')
    def test_loads_checkpoint_with_model_state_dict(self, mock_create_model, mock_torch):
        """Test loading a checkpoint with 'model_state_dict' key."""
        from src.models.model_factory import create_model_from_checkpoint
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_create_model.return_value = (mock_model, mock_tokenizer)
        
        checkpoint = {
            'epoch': 5,
            'model_state_dict': {'some_key': 'some_value'},
            'best_f1': 0.85
        }
        mock_torch.load.return_value = checkpoint
        
        config = MagicMock()
        model, tokenizer = create_model_from_checkpoint('path/to/checkpoint.pt', config)
        
        mock_model.load_state_dict.assert_called_once_with(checkpoint['model_state_dict'])

    @patch('src.models.model_factory.torch')
    @patch('src.models.model_factory.create_model')
    def test_loads_checkpoint_without_model_state_dict(self, mock_create_model, mock_torch):
        """Test loading a direct state dict checkpoint."""
        from src.models.model_factory import create_model_from_checkpoint
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_create_model.return_value = (mock_model, mock_tokenizer)
        
        state_dict = {'layer.weight': torch.zeros(3)}
        mock_torch.load.return_value = state_dict
        
        config = MagicMock()
        model, tokenizer = create_model_from_checkpoint('path/to/checkpoint.pt', config)
        
        mock_model.load_state_dict.assert_called_once_with(state_dict)

    @patch('src.models.model_factory.torch')
    @patch('src.models.model_factory.create_model')
    def test_returns_model_and_tokenizer(self, mock_create_model, mock_torch):
        """Test checkpoint loading returns (model, tokenizer) tuple."""
        from src.models.model_factory import create_model_from_checkpoint
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_create_model.return_value = (mock_model, mock_tokenizer)
        mock_torch.load.return_value = {'model_state_dict': {}}
        
        config = MagicMock()
        result = create_model_from_checkpoint('path/to/checkpoint.pt', config)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == mock_model
        assert result[1] == mock_tokenizer

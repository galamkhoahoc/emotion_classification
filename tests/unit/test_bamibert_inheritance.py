"""
Unit tests for BamiBERT inheritance from BaseEmotionClassifier.

Tests Requirements:
- 2.1: BamiBERT provides forward method with correct signature
- 2.2: BamiBERT forward returns dict with loss and logits
- 2.3: BamiBERT provides get_embeddings method
- 2.4: BamiBERT provides resize_token_embeddings method
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from src.models.base_classifier import BaseEmotionClassifier
from src.models.bamibert_emotion import BamiBERTEmotionClassifier


class TestBamiBERTInheritance:
    """Test suite for BamiBERT inheritance from BaseEmotionClassifier."""

    @pytest.fixture
    def bamibert_model(self):
        """Create a mocked BamiBERT model to avoid loading from HuggingFace."""
        with patch('src.models.bamibert_emotion.AutoModel') as mock_auto_model, \
             patch('src.models.bamibert_emotion.AutoConfig') as mock_auto_config:
            
            # Mock the config
            mock_config = MagicMock()
            mock_auto_config.from_pretrained.return_value = mock_config
            
            # Mock the BERT model
            mock_bert = MagicMock()
            mock_bert.embeddings.word_embeddings = nn.Embedding(100, 768)
            mock_bert.resize_token_embeddings = MagicMock()
            
            # Create a side effect that returns output with correct batch size
            def bert_forward(input_ids, attention_mask=None):
                batch_size = input_ids.shape[0]
                seq_length = input_ids.shape[1]
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(batch_size, seq_length, 768)
                return mock_output
            
            mock_bert.side_effect = bert_forward
            
            mock_auto_model.from_pretrained.return_value = mock_bert
            
            model = BamiBERTEmotionClassifier(
                model_name="Qualcomm-AI-Research/BamiBERT",
                num_labels=7,
                dropout_prob=0.1,
                hidden_size=768
            )
            
            yield model

    def test_bamibert_is_instance_of_base_classifier(self, bamibert_model):
        """Test that BamiBERT is an instance of BaseEmotionClassifier."""
        assert isinstance(bamibert_model, BaseEmotionClassifier)

    def test_bamibert_is_instance_of_nn_module(self, bamibert_model):
        """Test that BamiBERT is also an nn.Module (through BaseEmotionClassifier)."""
        assert isinstance(bamibert_model, nn.Module)

    def test_bamibert_has_num_labels_attribute(self, bamibert_model):
        """Test that BamiBERT inherits num_labels from base class."""
        assert hasattr(bamibert_model, 'num_labels')
        assert bamibert_model.num_labels == 7

    def test_forward_returns_dict(self, bamibert_model):
        """Test that forward() returns a dictionary."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = bamibert_model.forward(input_ids, attention_mask)
        
        assert isinstance(result, dict)

    def test_forward_returns_correct_keys(self, bamibert_model):
        """Test that forward() returns dict with 'loss' and 'logits' keys."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = bamibert_model.forward(input_ids, attention_mask)
        
        assert 'loss' in result
        assert 'logits' in result

    def test_forward_returns_logits_without_labels(self, bamibert_model):
        """Test that forward() returns None for loss when labels not provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = bamibert_model.forward(input_ids, attention_mask)
        
        assert result['loss'] is None
        assert result['logits'] is not None

    def test_forward_returns_loss_with_labels(self, bamibert_model):
        """Test that forward() returns loss when labels are provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        labels = torch.randint(0, 7, (2,))
        
        result = bamibert_model.forward(input_ids, attention_mask, labels=labels)
        
        assert result['loss'] is not None
        assert isinstance(result['loss'], torch.Tensor)
        assert result['logits'] is not None

    def test_forward_logits_shape(self, bamibert_model):
        """Test that forward() returns logits with correct shape [batch_size, num_labels]."""
        batch_size = 4
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        result = bamibert_model.forward(input_ids, attention_mask)
        
        assert result['logits'].shape == (batch_size, bamibert_model.num_labels)

    def test_forward_loss_is_scalar(self, bamibert_model):
        """Test that forward() returns scalar loss when labels provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        labels = torch.randint(0, 7, (2,))
        
        result = bamibert_model.forward(input_ids, attention_mask, labels=labels)
        
        # Loss should be a scalar (0-dimensional tensor)
        assert result['loss'].dim() == 0

    def test_get_embeddings_returns_nn_embedding(self, bamibert_model):
        """Test that get_embeddings() returns nn.Embedding."""
        embeddings = bamibert_model.get_embeddings()
        
        assert isinstance(embeddings, nn.Embedding)

    def test_get_embeddings_has_weight_parameter(self, bamibert_model):
        """Test that returned embeddings have weight parameter."""
        embeddings = bamibert_model.get_embeddings()
        
        assert hasattr(embeddings, 'weight')
        assert isinstance(embeddings.weight, torch.Tensor)

    def test_resize_token_embeddings_method_exists(self, bamibert_model):
        """Test that resize_token_embeddings() method exists."""
        assert hasattr(bamibert_model, 'resize_token_embeddings')
        assert callable(bamibert_model.resize_token_embeddings)

    def test_resize_token_embeddings_accepts_int(self, bamibert_model):
        """Test that resize_token_embeddings() accepts integer parameter."""
        new_vocab_size = 150
        
        # Should not raise an error
        try:
            bamibert_model.resize_token_embeddings(new_vocab_size)
        except Exception as e:
            pytest.fail(f"resize_token_embeddings raised an exception: {e}")

    def test_resize_token_embeddings_calls_underlying_model(self, bamibert_model):
        """Test that resize_token_embeddings() calls the underlying BERT model's method."""
        new_vocab_size = 150
        
        bamibert_model.resize_token_embeddings(new_vocab_size)
        
        # Verify the underlying bamibert model's resize method was called
        bamibert_model.bamibert.resize_token_embeddings.assert_called_once_with(new_vocab_size)

    def test_forward_works_without_attention_mask(self, bamibert_model):
        """Test that forward() works when attention_mask is not provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        result = bamibert_model.forward(input_ids)
        
        assert 'loss' in result
        assert 'logits' in result
        assert result['logits'].shape == (2, 7)

    def test_forward_with_multilabel_labels(self, bamibert_model):
        """Test that forward() handles multilabel labels (2D tensor)."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        # Multilabel: [batch_size, num_labels]
        labels = torch.rand(2, 7)
        
        result = bamibert_model.forward(input_ids, attention_mask, labels=labels)
        
        assert result['loss'] is not None
        assert isinstance(result['loss'], torch.Tensor)

    def test_bamibert_initializes_with_custom_parameters(self):
        """Test that BamiBERT can be initialized with custom parameters."""
        with patch('src.models.bamibert_emotion.AutoModel') as mock_auto_model, \
             patch('src.models.bamibert_emotion.AutoConfig') as mock_auto_config:
            
            mock_config = MagicMock()
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_bert = MagicMock()
            mock_auto_model.from_pretrained.return_value = mock_bert
            
            model = BamiBERTEmotionClassifier(
                model_name="custom/model",
                num_labels=10,
                dropout_prob=0.2,
                hidden_size=512
            )
            
            assert model.num_labels == 10
            assert model.dropout.p == 0.2


class TestBamiBERTInterfaceSignature:
    """Test suite to verify BamiBERT method signatures match base class."""

    @pytest.fixture
    def bamibert_model(self):
        """Create a mocked BamiBERT model."""
        with patch('src.models.bamibert_emotion.AutoModel') as mock_auto_model, \
             patch('src.models.bamibert_emotion.AutoConfig') as mock_auto_config:
            
            mock_config = MagicMock()
            mock_auto_config.from_pretrained.return_value = mock_config
            
            mock_bert = MagicMock()
            mock_bert.embeddings.word_embeddings = nn.Embedding(100, 768)
            
            # Create a side effect that returns output with correct batch size
            def bert_forward(input_ids, attention_mask=None):
                batch_size = input_ids.shape[0]
                seq_length = input_ids.shape[1]
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(batch_size, seq_length, 768)
                return mock_output
            
            mock_bert.side_effect = bert_forward
            
            mock_auto_model.from_pretrained.return_value = mock_bert
            
            model = BamiBERTEmotionClassifier()
            
            yield model

    def test_forward_signature_matches_base(self, bamibert_model):
        """Test that forward() signature matches BaseEmotionClassifier."""
        import inspect
        
        base_forward = BaseEmotionClassifier.forward
        bamibert_forward = BamiBERTEmotionClassifier.forward
        
        base_sig = inspect.signature(base_forward)
        bamibert_sig = inspect.signature(bamibert_forward)
        
        # Check parameter names match (excluding 'self')
        base_params = list(base_sig.parameters.keys())[1:]  # Skip 'self'
        bamibert_params = list(bamibert_sig.parameters.keys())[1:]  # Skip 'self'
        
        assert base_params == bamibert_params

    def test_get_embeddings_signature_matches_base(self, bamibert_model):
        """Test that get_embeddings() signature matches BaseEmotionClassifier."""
        import inspect
        
        base_method = BaseEmotionClassifier.get_embeddings
        bamibert_method = BamiBERTEmotionClassifier.get_embeddings
        
        base_sig = inspect.signature(base_method)
        bamibert_sig = inspect.signature(bamibert_method)
        
        # Check parameter names match
        base_params = list(base_sig.parameters.keys())
        bamibert_params = list(bamibert_sig.parameters.keys())
        
        assert base_params == bamibert_params

    def test_resize_token_embeddings_signature_matches_base(self, bamibert_model):
        """Test that resize_token_embeddings() signature matches BaseEmotionClassifier."""
        import inspect
        
        base_method = BaseEmotionClassifier.resize_token_embeddings
        bamibert_method = BamiBERTEmotionClassifier.resize_token_embeddings
        
        base_sig = inspect.signature(base_method)
        bamibert_sig = inspect.signature(bamibert_method)
        
        # Check parameter names match
        base_params = list(base_sig.parameters.keys())
        bamibert_params = list(bamibert_sig.parameters.keys())
        
        assert base_params == bamibert_params

"""
Unit tests for PhoBERT inheritance from BaseEmotionClassifier.

Tests Requirements:
- 2.1: PhoBERT provides forward method with correct signature
- 2.2: PhoBERT forward returns dict with loss and logits
- 2.3: PhoBERT provides get_embeddings method
- 2.4: PhoBERT provides resize_token_embeddings method
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from src.models.base_classifier import BaseEmotionClassifier
from src.models.phobert_emotion import PhoBERTEmotionClassifier


class TestPhoBERTInheritance:
    """Test suite for PhoBERT inheritance from BaseEmotionClassifier."""

    @pytest.fixture
    def phobert_model(self):
        """Create a mocked PhoBERT model to avoid loading from HuggingFace."""
        with patch('src.models.phobert_emotion.AutoModel') as mock_auto_model, \
             patch('src.models.phobert_emotion.AutoConfig') as mock_auto_config:
            
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
            
            model = PhoBERTEmotionClassifier(
                model_name="vinai/phobert-base",
                num_labels=7,
                dropout_prob=0.1,
                hidden_size=768
            )
            
            yield model

    def test_phobert_is_instance_of_base_classifier(self, phobert_model):
        """Test that PhoBERT is an instance of BaseEmotionClassifier."""
        assert isinstance(phobert_model, BaseEmotionClassifier)

    def test_phobert_is_instance_of_nn_module(self, phobert_model):
        """Test that PhoBERT is also an nn.Module (through BaseEmotionClassifier)."""
        assert isinstance(phobert_model, nn.Module)

    def test_phobert_has_num_labels_attribute(self, phobert_model):
        """Test that PhoBERT inherits num_labels from base class."""
        assert hasattr(phobert_model, 'num_labels')
        assert phobert_model.num_labels == 7

    def test_forward_returns_dict(self, phobert_model):
        """Test that forward() returns a dictionary."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = phobert_model.forward(input_ids, attention_mask)
        
        assert isinstance(result, dict)

    def test_forward_returns_correct_keys(self, phobert_model):
        """Test that forward() returns dict with 'loss' and 'logits' keys."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = phobert_model.forward(input_ids, attention_mask)
        
        assert 'loss' in result
        assert 'logits' in result

    def test_forward_returns_logits_without_labels(self, phobert_model):
        """Test that forward() returns None for loss when labels not provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = phobert_model.forward(input_ids, attention_mask)
        
        assert result['loss'] is None
        assert result['logits'] is not None

    def test_forward_returns_loss_with_labels(self, phobert_model):
        """Test that forward() returns loss when labels are provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        labels = torch.randint(0, 7, (2,))
        
        result = phobert_model.forward(input_ids, attention_mask, labels=labels)
        
        assert result['loss'] is not None
        assert isinstance(result['loss'], torch.Tensor)
        assert result['logits'] is not None

    def test_forward_logits_shape(self, phobert_model):
        """Test that forward() returns logits with correct shape [batch_size, num_labels]."""
        batch_size = 4
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        result = phobert_model.forward(input_ids, attention_mask)
        
        assert result['logits'].shape == (batch_size, phobert_model.num_labels)

    def test_forward_loss_is_scalar(self, phobert_model):
        """Test that forward() returns scalar loss when labels provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        labels = torch.randint(0, 7, (2,))
        
        result = phobert_model.forward(input_ids, attention_mask, labels=labels)
        
        # Loss should be a scalar (0-dimensional tensor)
        assert result['loss'].dim() == 0

    def test_get_embeddings_returns_nn_embedding(self, phobert_model):
        """Test that get_embeddings() returns nn.Embedding."""
        embeddings = phobert_model.get_embeddings()
        
        assert isinstance(embeddings, nn.Embedding)

    def test_get_embeddings_has_weight_parameter(self, phobert_model):
        """Test that returned embeddings have weight parameter."""
        embeddings = phobert_model.get_embeddings()
        
        assert hasattr(embeddings, 'weight')
        assert isinstance(embeddings.weight, torch.Tensor)

    def test_resize_token_embeddings_method_exists(self, phobert_model):
        """Test that resize_token_embeddings() method exists."""
        assert hasattr(phobert_model, 'resize_token_embeddings')
        assert callable(phobert_model.resize_token_embeddings)

    def test_resize_token_embeddings_accepts_int(self, phobert_model):
        """Test that resize_token_embeddings() accepts integer parameter."""
        new_vocab_size = 150
        
        # Should not raise an error
        try:
            phobert_model.resize_token_embeddings(new_vocab_size)
        except Exception as e:
            pytest.fail(f"resize_token_embeddings raised an exception: {e}")

    def test_resize_token_embeddings_calls_underlying_model(self, phobert_model):
        """Test that resize_token_embeddings() calls the underlying BERT model's method."""
        new_vocab_size = 150
        
        phobert_model.resize_token_embeddings(new_vocab_size)
        
        # Verify the underlying phobert model's resize method was called
        phobert_model.phobert.resize_token_embeddings.assert_called_once_with(new_vocab_size)

    def test_forward_works_without_attention_mask(self, phobert_model):
        """Test that forward() works when attention_mask is not provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        result = phobert_model.forward(input_ids)
        
        assert 'loss' in result
        assert 'logits' in result
        assert result['logits'].shape == (2, 7)

    def test_forward_with_multilabel_labels(self, phobert_model):
        """Test that forward() handles multilabel labels (2D tensor)."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        # Multilabel: [batch_size, num_labels]
        labels = torch.rand(2, 7)
        
        result = phobert_model.forward(input_ids, attention_mask, labels=labels)
        
        assert result['loss'] is not None
        assert isinstance(result['loss'], torch.Tensor)

    def test_phobert_initializes_with_custom_parameters(self):
        """Test that PhoBERT can be initialized with custom parameters."""
        with patch('src.models.phobert_emotion.AutoModel') as mock_auto_model, \
             patch('src.models.phobert_emotion.AutoConfig') as mock_auto_config:
            
            mock_config = MagicMock()
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_bert = MagicMock()
            mock_auto_model.from_pretrained.return_value = mock_bert
            
            model = PhoBERTEmotionClassifier(
                model_name="custom/model",
                num_labels=10,
                dropout_prob=0.2,
                hidden_size=512
            )
            
            assert model.num_labels == 10
            assert model.dropout.p == 0.2


class TestPhoBERTInterfaceSignature:
    """Test suite to verify PhoBERT method signatures match base class."""

    @pytest.fixture
    def phobert_model(self):
        """Create a mocked PhoBERT model."""
        with patch('src.models.phobert_emotion.AutoModel') as mock_auto_model, \
             patch('src.models.phobert_emotion.AutoConfig') as mock_auto_config:
            
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
            
            model = PhoBERTEmotionClassifier()
            
            yield model

    def test_forward_signature_matches_base(self, phobert_model):
        """Test that forward() signature matches BaseEmotionClassifier."""
        import inspect
        
        base_forward = BaseEmotionClassifier.forward
        phobert_forward = PhoBERTEmotionClassifier.forward
        
        base_sig = inspect.signature(base_forward)
        phobert_sig = inspect.signature(phobert_forward)
        
        # Check parameter names match (excluding 'self')
        base_params = list(base_sig.parameters.keys())[1:]  # Skip 'self'
        phobert_params = list(phobert_sig.parameters.keys())[1:]  # Skip 'self'
        
        assert base_params == phobert_params

    def test_get_embeddings_signature_matches_base(self, phobert_model):
        """Test that get_embeddings() signature matches BaseEmotionClassifier."""
        import inspect
        
        base_method = BaseEmotionClassifier.get_embeddings
        phobert_method = PhoBERTEmotionClassifier.get_embeddings
        
        base_sig = inspect.signature(base_method)
        phobert_sig = inspect.signature(phobert_method)
        
        # Check parameter names match
        base_params = list(base_sig.parameters.keys())
        phobert_params = list(phobert_sig.parameters.keys())
        
        assert base_params == phobert_params

    def test_resize_token_embeddings_signature_matches_base(self, phobert_model):
        """Test that resize_token_embeddings() signature matches BaseEmotionClassifier."""
        import inspect
        
        base_method = BaseEmotionClassifier.resize_token_embeddings
        phobert_method = PhoBERTEmotionClassifier.resize_token_embeddings
        
        base_sig = inspect.signature(base_method)
        phobert_sig = inspect.signature(phobert_method)
        
        # Check parameter names match
        base_params = list(base_sig.parameters.keys())
        phobert_params = list(phobert_sig.parameters.keys())
        
        assert base_params == phobert_params

"""
Unit tests for emoji embeddings with BamiBERT.

Tests Requirements:
- 7.2: WHEN enable_emoji_embedding is true AND model_type is "bamiBERT", 
       THE ViEmoText_System SHALL apply emoji embeddings to the BamiBERT model
- 7.3: THE emoji embedding component SHALL add emoji tokens to the tokenizer vocabulary
- 7.4: THE emoji embedding component SHALL resize model embeddings to accommodate new emoji tokens
- 7.5: THE emoji embedding component SHALL copy embeddings from Vietnamese words to corresponding emojis
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from configs.config import Config
from src.models.emoji_embeddings import apply_emoji_embeddings


class TestEmojiEmbeddingsWithBamiBERT:
    """Test suite for verifying emoji embeddings work with BamiBERT."""

    @pytest.fixture
    def bamibert_config(self):
        """Create a BamiBERT config with emoji embeddings enabled."""
        return Config(
            model_type="bamibert",
            enable_emoji_embedding=True
        )

    @pytest.fixture
    def simple_emoji_mapping(self):
        """Simple emoji mapping for testing."""
        return {
            "😊": "vui",
            "😢": "buồn",
            "😡": "giận"
        }

    @patch('src.models.emoji_embeddings.logger')
    def test_emoji_tokens_added_to_vocabulary(self, mock_logger, simple_emoji_mapping):
        """Test that emoji tokens are added to tokenizer vocabulary (Requirement 7.3)."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer mock
        mock_tokenizer.get_vocab.return_value = {"vui": 1, "buồn": 2, "giận": 3}
        mock_tokenizer.__len__.return_value = 1000
        mock_tokenizer.add_tokens.return_value = 3  # 3 new tokens added
        mock_tokenizer.tokenize.side_effect = lambda x: [x]  # Simple tokenization
        
        # convert_tokens_to_ids can receive single token or list of tokens
        # Emoji IDs should be 1000, 1001, 1002 (valid for tensor of size 1003)
        def convert_fn(tokens):
            mapping = {"vui": 1, "buồn": 2, "giận": 3, "😊": 1000, "😢": 1001, "😡": 1002}
            if isinstance(tokens, list):
                return [mapping.get(t, 0) for t in tokens]
            return mapping.get(tokens, 0)
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        # Setup model mock
        mock_embeddings = MagicMock()
        mock_embeddings.weight = torch.randn(1003, 768)  # After adding emojis
        mock_model.get_embeddings.return_value = mock_embeddings
        
        # Apply emoji embeddings
        result = apply_emoji_embeddings(mock_model, mock_tokenizer, simple_emoji_mapping)
        
        # Verify tokens were added
        mock_tokenizer.add_tokens.assert_called_once()
        added_tokens = mock_tokenizer.add_tokens.call_args[0][0]
        assert "😊" in added_tokens
        assert "😢" in added_tokens
        assert "😡" in added_tokens

    @patch('src.models.emoji_embeddings.logger')
    def test_model_embeddings_resized(self, mock_logger, simple_emoji_mapping):
        """Test that model embeddings are resized to accommodate new tokens (Requirement 7.4)."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer mock
        original_vocab_size = 1000
        new_vocab_size = 1003
        mock_tokenizer.get_vocab.return_value = {}  # All emojis need to be added
        mock_tokenizer.__len__.side_effect = [original_vocab_size, new_vocab_size]  # Before and after
        mock_tokenizer.add_tokens.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: [x]
        
        def convert_fn(tokens):
            mapping = {"vui": 1, "buồn": 2, "giận": 3, "😊": 1000, "😢": 1001, "😡": 1002}
            if isinstance(tokens, list):
                return [mapping.get(t, 0) for t in tokens]
            return mapping.get(tokens, 0)
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        # Setup model mock
        mock_embeddings = MagicMock()
        mock_embeddings.weight = torch.randn(new_vocab_size, 768)
        mock_model.get_embeddings.return_value = mock_embeddings
        
        # Apply emoji embeddings
        result = apply_emoji_embeddings(mock_model, mock_tokenizer, simple_emoji_mapping)
        
        # Verify resize was called with correct size
        mock_model.resize_token_embeddings.assert_called_once_with(new_vocab_size)

    @patch('src.models.emoji_embeddings.logger')
    def test_embeddings_copied_from_vietnamese_words(self, mock_logger, simple_emoji_mapping):
        """Test that embeddings are copied from Vietnamese words to emojis (Requirement 7.5)."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer
        mock_tokenizer.get_vocab.return_value = {}
        mock_tokenizer.__len__.return_value = 1003
        mock_tokenizer.add_tokens.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: [x]
        
        def convert_fn(tokens):
            mapping = {"vui": 1, "buồn": 2, "giận": 3, "😊": 1001, "😢": 1002, "😡": 1003}
            if isinstance(tokens, list):
                return [mapping.get(t, 0) for t in tokens]
            return mapping.get(tokens, 0)
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        # Create real embeddings to test copying
        embedding_dim = 768
        embeddings_tensor = torch.randn(1004, embedding_dim)
        
        # Set specific patterns for Vietnamese words (for verification)
        embeddings_tensor[1] = torch.ones(embedding_dim) * 1.0  # "vui"
        embeddings_tensor[2] = torch.ones(embedding_dim) * 2.0  # "buồn"
        embeddings_tensor[3] = torch.ones(embedding_dim) * 3.0  # "giận"
        
        mock_embeddings = MagicMock()
        mock_embeddings.weight = embeddings_tensor
        mock_model.get_embeddings.return_value = mock_embeddings
        
        # Apply emoji embeddings
        result = apply_emoji_embeddings(mock_model, mock_tokenizer, simple_emoji_mapping)
        
        # Verify embeddings were copied (check that emoji embeddings match Vietnamese word embeddings)
        assert torch.allclose(embeddings_tensor[1001], embeddings_tensor[1], atol=1e-6)  # 😊 -> vui
        assert torch.allclose(embeddings_tensor[1002], embeddings_tensor[2], atol=1e-6)  # 😢 -> buồn
        assert torch.allclose(embeddings_tensor[1003], embeddings_tensor[3], atol=1e-6)  # 😡 -> giận

    @patch('src.models.model_factory.AutoTokenizer')
    @patch('src.models.model_factory.BamiBERTEmotionClassifier')
    def test_emoji_embeddings_work_with_bamibert_model(self, mock_bamibert_cls, mock_tokenizer_cls, bamibert_config, simple_emoji_mapping):
        """
        Test complete flow: BamiBERT model creation + emoji embedding application (Requirement 7.2).
        
        This tests the full workflow:
        1. Create BamiBERT model via factory
        2. Apply emoji embeddings
        3. Verify all steps work correctly
        """
        from src.models.model_factory import create_model
        
        # Setup mock BamiBERT model
        mock_model = MagicMock()
        mock_bamibert_cls.return_value = mock_model
        
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.get_vocab.return_value = {"vui": 1, "buồn": 2, "giận": 3}
        mock_tokenizer.__len__.side_effect = [1000, 1003]
        mock_tokenizer.add_tokens.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: [x]
        
        def convert_fn(tokens):
            mapping = {"vui": 1, "buồn": 2, "giận": 3, "😊": 1001, "😢": 1002, "😡": 1003}
            if isinstance(tokens, list):
                return [mapping.get(t, 0) for t in tokens]
            return mapping.get(tokens, 0)
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        # Setup embeddings
        embedding_dim = 768
        embeddings_tensor = torch.randn(1004, embedding_dim)
        mock_embeddings = MagicMock()
        mock_embeddings.weight = embeddings_tensor
        mock_model.get_embeddings.return_value = mock_embeddings
        
        # Create model
        model, tokenizer = create_model(bamibert_config)
        
        # Verify BamiBERT was created
        mock_bamibert_cls.assert_called_once()
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, simple_emoji_mapping)
        
        # Verify the complete flow worked
        assert mock_tokenizer.add_tokens.called
        assert mock_model.resize_token_embeddings.called
        assert model is not None

    @patch('src.models.emoji_embeddings.logger')
    def test_no_tokens_added_if_already_in_vocab(self, mock_logger, simple_emoji_mapping):
        """Test that no tokens are added if emojis already exist in vocabulary."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer with emojis already in vocabulary
        mock_tokenizer.get_vocab.return_value = {
            "😊": 1001,
            "😢": 1002,
            "😡": 1003,
            "vui": 1,
            "buồn": 2,
            "giận": 3
        }
        mock_tokenizer.__len__.return_value = 1004
        mock_tokenizer.tokenize.side_effect = lambda x: [x]
        
        def convert_fn(tokens):
            mapping = {"😊": 1001, "😢": 1002, "😡": 1003, "vui": 1, "buồn": 2, "giận": 3}
            if isinstance(tokens, list):
                return [mapping.get(t, 0) for t in tokens]
            return mapping.get(tokens, 0)
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        # Setup embeddings
        embeddings_tensor = torch.randn(1004, 768)
        mock_embeddings = MagicMock()
        mock_embeddings.weight = embeddings_tensor
        mock_model.get_embeddings.return_value = mock_embeddings
        
        # Apply emoji embeddings
        result = apply_emoji_embeddings(mock_model, mock_tokenizer, simple_emoji_mapping)
        
        # Verify no tokens were added (but embeddings were still copied)
        mock_tokenizer.add_tokens.assert_not_called()
        mock_model.resize_token_embeddings.assert_not_called()

    @patch('src.models.emoji_embeddings.logger')
    def test_handles_multi_token_vietnamese_words(self, mock_logger):
        """Test that multi-token Vietnamese words are averaged correctly."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Emoji mapping with compound word
        emoji_mapping = {"🤢": "ghê_tởm"}
        
        # Setup tokenizer
        mock_tokenizer.get_vocab.return_value = {}
        mock_tokenizer.__len__.return_value = 1001
        mock_tokenizer.add_tokens.return_value = 1
        
        # "ghê_tởm" tokenizes to multiple tokens
        def custom_tokenize(text):
            if text == "ghê_tởm":
                return ["ghê", "tởm"]
            return [text]
        
        mock_tokenizer.tokenize = custom_tokenize
        
        def convert_fn(tokens):
            mapping = {"ghê": 1, "tởm": 2, "🤢": 1001}
            if isinstance(tokens, list):
                return [mapping.get(t, 0) for t in tokens]
            return mapping.get(tokens, 0)
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        # Create embeddings
        embedding_dim = 768
        embeddings_tensor = torch.randn(1002, embedding_dim)
        embeddings_tensor[1] = torch.ones(embedding_dim) * 1.0  # "ghê"
        embeddings_tensor[2] = torch.ones(embedding_dim) * 2.0  # "tởm"
        
        mock_embeddings = MagicMock()
        mock_embeddings.weight = embeddings_tensor
        mock_model.get_embeddings.return_value = mock_embeddings
        
        # Apply emoji embeddings
        result = apply_emoji_embeddings(mock_model, mock_tokenizer, emoji_mapping)
        
        # Verify emoji embedding is average of the two Vietnamese tokens
        expected_embedding = (embeddings_tensor[1] + embeddings_tensor[2]) / 2
        assert torch.allclose(embeddings_tensor[1001], expected_embedding, atol=1e-6)


class TestEmojiEmbeddingsInterface:
    """Test that emoji embeddings use the correct interface methods."""

    @patch('src.models.emoji_embeddings.logger')
    def test_uses_get_embeddings_method(self, mock_logger):
        """Verify that apply_emoji_embeddings calls model.get_embeddings()."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Minimal setup
        mock_tokenizer.get_vocab.return_value = {}
        mock_tokenizer.__len__.return_value = 1001
        mock_tokenizer.add_tokens.return_value = 1
        mock_tokenizer.tokenize.return_value = ["test"]
        
        def convert_fn(tokens):
            if isinstance(tokens, list):
                return [1] * len(tokens)
            return 1
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        embeddings_tensor = torch.randn(1001, 768)
        mock_embeddings = MagicMock()
        mock_embeddings.weight = embeddings_tensor
        mock_model.get_embeddings.return_value = mock_embeddings
        
        emoji_mapping = {"😊": "test"}
        apply_emoji_embeddings(mock_model, mock_tokenizer, emoji_mapping)
        
        # Verify get_embeddings was called
        mock_model.get_embeddings.assert_called()

    @patch('src.models.emoji_embeddings.logger')
    def test_uses_resize_token_embeddings_method(self, mock_logger):
        """Verify that apply_emoji_embeddings calls model.resize_token_embeddings()."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup to add new tokens
        mock_tokenizer.get_vocab.return_value = {}
        mock_tokenizer.__len__.side_effect = [1000, 1001]
        mock_tokenizer.add_tokens.return_value = 1
        mock_tokenizer.tokenize.return_value = ["test"]
        
        def convert_fn(tokens):
            if isinstance(tokens, list):
                return [1000] * len(tokens)
            return 1000
        
        mock_tokenizer.convert_tokens_to_ids = convert_fn
        
        embeddings_tensor = torch.randn(1001, 768)
        mock_embeddings = MagicMock()
        mock_embeddings.weight = embeddings_tensor
        mock_model.get_embeddings.return_value = mock_embeddings
        
        emoji_mapping = {"😊": "test"}
        apply_emoji_embeddings(mock_model, mock_tokenizer, emoji_mapping)
        
        # Verify resize_token_embeddings was called with new vocab size
        mock_model.resize_token_embeddings.assert_called_once_with(1001)

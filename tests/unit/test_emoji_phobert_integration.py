"""
Integration tests for emoji embeddings with PhoBERT model.

Task 13.1: Verify emoji embeddings work with PhoBERT
- Run existing emoji embedding code with PhoBERT
- Verify embeddings added to vocabulary
- Verify weights copied correctly

Tests Requirements:
- 7.1: WHEN enable_emoji_embedding is true AND model_type is "phobert", 
       THE ViEmoText_System SHALL apply emoji embeddings to the PhoBERT model
- 7.3: THE emoji embedding component SHALL add emoji tokens to the tokenizer vocabulary
- 7.4: THE emoji embedding component SHALL resize model embeddings to accommodate new emoji tokens
- 7.5: THE emoji embedding component SHALL copy embeddings from Vietnamese words to corresponding emojis
"""

import pytest
import torch
from transformers import AutoTokenizer

from src.models.phobert_emotion import PhoBERTEmotionClassifier
from src.models.emoji_embeddings import apply_emoji_embeddings
from configs.config import Config


class TestEmojiEmbeddingsPhoBERTIntegration:
    """Integration tests for emoji embedding functionality with real PhoBERT model."""
    
    @pytest.fixture
    def phobert_config(self):
        """Create test configuration for PhoBERT."""
        config = Config()
        config.model_type = "phobert"
        config.model_name = "vinai/phobert-base"
        config.num_labels = 7
        config.enable_emoji_embedding = True
        return config
    
    @pytest.fixture
    def phobert_model_and_tokenizer(self, phobert_config):
        """Create real PhoBERT model and tokenizer for testing."""
        # Create model
        model = PhoBERTEmotionClassifier(
            model_name=phobert_config.model_name,
            num_labels=phobert_config.num_labels
        )
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(phobert_config.model_name)
        
        return model, tokenizer
    
    @pytest.fixture
    def test_emoji_mapping(self):
        """Small emoji mapping for testing."""
        return {
            "😊": "vui",
            "😢": "buồn",
            "😡": "giận"
        }
    
    def test_emoji_tokens_added_to_vocabulary_phobert(
        self, phobert_model_and_tokenizer, test_emoji_mapping
    ):
        """
        Test that emoji tokens are added to PhoBERT tokenizer vocabulary.
        Validates: Requirement 7.3
        """
        model, tokenizer = phobert_model_and_tokenizer
        
        # Get initial vocabulary size
        original_vocab_size = len(tokenizer)
        
        # Get initial vocabulary
        original_vocab = set(tokenizer.get_vocab().keys())
        
        # Check emojis not in vocabulary initially
        for emoji in test_emoji_mapping.keys():
            assert emoji not in original_vocab, \
                f"Emoji {emoji} should not be in initial PhoBERT vocabulary"
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, test_emoji_mapping)
        
        # Check emojis are now in vocabulary
        updated_vocab = tokenizer.get_vocab()
        for emoji in test_emoji_mapping.keys():
            assert emoji in updated_vocab, \
                f"Emoji {emoji} should be in PhoBERT vocabulary after apply_emoji_embeddings"
        
        # Verify vocabulary size increased
        new_vocab_size = len(tokenizer)
        expected_new_size = original_vocab_size + len(test_emoji_mapping)
        assert new_vocab_size == expected_new_size, \
            f"Expected vocabulary size {expected_new_size}, got {new_vocab_size}"
        
        print(f"✓ PhoBERT vocabulary expanded from {original_vocab_size} to {new_vocab_size} tokens")
    
    def test_model_embeddings_resized_phobert(
        self, phobert_model_and_tokenizer, test_emoji_mapping
    ):
        """
        Test that PhoBERT model embeddings are resized to accommodate new emoji tokens.
        Validates: Requirement 7.4
        """
        model, tokenizer = phobert_model_and_tokenizer
        
        # Get initial embedding size
        original_embedding_size = model.get_embeddings().weight.shape[0]
        original_vocab_size = len(tokenizer)
        
        assert original_embedding_size == original_vocab_size, \
            "Initial embedding size should match vocabulary size"
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, test_emoji_mapping)
        
        # Get new embedding size
        new_embedding_size = model.get_embeddings().weight.shape[0]
        new_vocab_size = len(tokenizer)
        
        # Verify embedding size matches new vocabulary size
        assert new_embedding_size == new_vocab_size, \
            f"Embedding size ({new_embedding_size}) should match vocabulary size ({new_vocab_size})"
        
        # Verify size increased by number of emojis
        expected_new_size = original_embedding_size + len(test_emoji_mapping)
        assert new_embedding_size == expected_new_size, \
            f"Expected embedding size {expected_new_size}, got {new_embedding_size}"
        
        print(f"✓ PhoBERT embeddings resized from {original_embedding_size} to {new_embedding_size}")
    
    def test_weights_copied_from_vietnamese_words_phobert(
        self, phobert_model_and_tokenizer, test_emoji_mapping
    ):
        """
        Test that embedding weights are copied from Vietnamese words to emojis in PhoBERT.
        Validates: Requirement 7.5
        """
        model, tokenizer = phobert_model_and_tokenizer
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, test_emoji_mapping)
        
        # Get embedding layer
        embeddings = model.get_embeddings()
        
        # Verify each emoji has embeddings similar to its Vietnamese word
        for emoji, vietnamese_word in test_emoji_mapping.items():
            # Get emoji embedding
            emoji_id = tokenizer.convert_tokens_to_ids(emoji)
            emoji_embedding = embeddings.weight[emoji_id]
            
            # Get Vietnamese word embedding(s)
            word_tokens = tokenizer.tokenize(vietnamese_word)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            
            # Ensure we got valid embeddings
            assert emoji_embedding is not None, f"Emoji {emoji} has no embedding"
            assert len(word_ids) > 0, f"Vietnamese word {vietnamese_word} has no tokens"
            
            if len(word_ids) == 1:
                # Single token: should be identical
                word_embedding = embeddings.weight[word_ids[0]]
                assert torch.allclose(emoji_embedding, word_embedding, atol=1e-6), \
                    f"Emoji '{emoji}' embedding should match Vietnamese word '{vietnamese_word}'"
                print(f"✓ Emoji '{emoji}' embedding copied from '{vietnamese_word}' (single token)")
            else:
                # Multiple tokens: should be average
                word_embeddings = torch.stack([embeddings.weight[wid] for wid in word_ids])
                expected_embedding = word_embeddings.mean(dim=0)
                assert torch.allclose(emoji_embedding, expected_embedding, atol=1e-6), \
                    f"Emoji '{emoji}' embedding should be average of Vietnamese word '{vietnamese_word}' tokens"
                print(f"✓ Emoji '{emoji}' embedding averaged from '{vietnamese_word}' ({len(word_ids)} tokens)")
    
    def test_full_emoji_integration_with_phobert(
        self, phobert_model_and_tokenizer, test_emoji_mapping
    ):
        """
        Full integration test: verify complete emoji embedding workflow with PhoBERT.
        Validates: Requirements 7.1, 7.3, 7.4, 7.5
        """
        model, tokenizer = phobert_model_and_tokenizer
        
        # Initial state
        original_vocab_size = len(tokenizer)
        original_embedding_size = model.get_embeddings().weight.shape[0]
        
        print(f"\nInitial state:")
        print(f"  Vocabulary size: {original_vocab_size}")
        print(f"  Embedding size: {original_embedding_size}")
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, test_emoji_mapping)
        
        # Verify all changes
        new_vocab_size = len(tokenizer)
        new_embedding_size = model.get_embeddings().weight.shape[0]
        
        print(f"\nAfter applying emoji embeddings:")
        print(f"  Vocabulary size: {new_vocab_size}")
        print(f"  Embedding size: {new_embedding_size}")
        
        # Check vocabulary expanded
        assert new_vocab_size == original_vocab_size + len(test_emoji_mapping), \
            "Vocabulary size should increase by number of emojis"
        
        # Check embeddings resized
        assert new_embedding_size == new_vocab_size, \
            "Embedding size should match vocabulary size"
        
        # Check all emojis in vocabulary
        vocab = tokenizer.get_vocab()
        for emoji in test_emoji_mapping.keys():
            assert emoji in vocab, f"Emoji {emoji} should be in vocabulary"
        
        # Check model can process text with emojis
        test_text = "Tôi rất vui 😊 nhưng cũng buồn 😢"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        print(f"\nProcessing test text: '{test_text}'")
        print(f"  Tokenized length: {inputs['input_ids'].shape[1]}")
        
        # Forward pass should work without errors
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        # Check output structure
        assert 'logits' in outputs, "Outputs should contain 'logits'"
        assert outputs['logits'].shape[0] == 1, "Batch size should be 1"
        assert outputs['logits'].shape[1] == model.num_labels, \
            f"Should have {model.num_labels} emotion classes"
        
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print(f"\n✓ Full integration test passed - PhoBERT can process emoji-enriched text")
    
    def test_preserves_existing_embeddings_phobert(
        self, phobert_model_and_tokenizer, test_emoji_mapping
    ):
        """
        Test that applying emoji embeddings doesn't corrupt existing PhoBERT embeddings.
        Validates: Requirement 7.5 (implicit - preserve existing weights)
        """
        model, tokenizer = phobert_model_and_tokenizer
        
        # Sample some existing tokens
        sample_tokens = ["<s>", "</s>", "và", "của", "có"]
        sample_ids = []
        original_embeddings = []
        
        embeddings = model.get_embeddings()
        
        for token in sample_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id >= 0:
                sample_ids.append(token_id)
                original_embeddings.append(embeddings.weight[token_id].clone())
        
        assert len(sample_ids) > 0, "Should have some sample tokens to test"
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, test_emoji_mapping)
        
        # Check original embeddings unchanged
        updated_embeddings = model.get_embeddings()
        
        preserved_count = 0
        for i, token_id in enumerate(sample_ids):
            if torch.allclose(original_embeddings[i], updated_embeddings.weight[token_id], atol=1e-6):
                preserved_count += 1
            else:
                print(f"Warning: Embedding at index {token_id} changed!")
        
        assert preserved_count == len(sample_ids), \
            f"All {len(sample_ids)} sampled embeddings should be preserved, but only {preserved_count} were"
        
        print(f"✓ All {preserved_count} sampled PhoBERT embeddings preserved")
    
    def test_config_integration_phobert(self, phobert_config):
        """
        Test emoji embeddings work when enable_emoji_embedding is True in PhoBERT config.
        Validates: Requirement 7.1
        """
        # Verify config has emoji embedding enabled for PhoBERT
        assert phobert_config.enable_emoji_embedding is True, \
            "Config should have emoji embeddings enabled"
        assert phobert_config.model_type == "phobert", \
            "Config should specify PhoBERT model"
        
        # Create model and tokenizer
        model = PhoBERTEmotionClassifier(
            model_name=phobert_config.model_name,
            num_labels=phobert_config.num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(phobert_config.model_name)
        
        # Test emoji mapping
        test_mapping = {"😊": "vui", "😢": "buồn", "😡": "giận"}
        
        # Apply emoji embeddings
        model = apply_emoji_embeddings(model, tokenizer, test_mapping)
        
        # Verify emojis added
        vocab = tokenizer.get_vocab()
        for emoji in test_mapping.keys():
            assert emoji in vocab, f"Emoji {emoji} should be in PhoBERT vocabulary"
        
        print(f"✓ Config integration test passed - PhoBERT with emoji embeddings enabled")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

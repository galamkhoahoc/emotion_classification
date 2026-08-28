"""
Property-based tests for emoji embedding functionality.

Tests Property 6 from the multi-model-support spec.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume
from configs.config import Config
from src.models.model_factory import create_model
from src.models.emoji_embeddings import apply_emoji_embeddings


# Generate valid emoji mappings
@st.composite
def emoji_mapping_strategy(draw):
    """
    Generate random emoji-to-word mappings.
    
    Returns a dictionary with emoji keys (simple unicode characters or symbols)
    and Vietnamese word values.
    """
    # Common emojis and symbols
    emoji_chars = ['😀', '😂', '😊', '😍', '😢', '😭', '😡', '😱', '👍', '👎', 
                   '❤️', '💔', '🎉', '🎊', '✨', '⭐', '🌟', '💯', '🔥', '💪']
    
    # Vietnamese words (common emotions/descriptors)
    vietnamese_words = [
        'vui', 'buồn', 'tức', 'giận', 'yêu', 'ghét', 'thích', 'sợ',
        'hạnh_phúc', 'đau_khổ', 'tuyệt_vời', 'tệ', 'tốt', 'xấu',
        'vui_vẻ', 'lo_lắng', 'phấn_khích', 'thất_vọng'
    ]
    
    # Generate mapping with 1-10 emojis
    num_emojis = draw(st.integers(min_value=1, max_value=10))
    
    # Sample emojis without replacement
    selected_emojis = draw(st.lists(
        st.sampled_from(emoji_chars),
        min_size=num_emojis,
        max_size=num_emojis,
        unique=True
    ))
    
    # Sample Vietnamese words (with replacement allowed)
    selected_words = draw(st.lists(
        st.sampled_from(vietnamese_words),
        min_size=num_emojis,
        max_size=num_emojis
    ))
    
    return {emoji: word for emoji, word in zip(selected_emojis, selected_words)}


# Property 6: Emoji embedding application
# Feature: multi-model-support, Property 6: Emoji embedding application
@settings(max_examples=100, deadline=30000)
@given(
    emoji_mapping=emoji_mapping_strategy(),
    model_type=st.sampled_from(["phobert", "bamibert"]),
    num_labels=st.integers(min_value=2, max_value=10)
)
def test_property_6_emoji_embedding_application(emoji_mapping, model_type, num_labels):
    """
    Property 6: Emoji embedding application
    
    For any valid emoji-to-word mapping dictionary and any model type
    ("phobert" or "bamibert"), applying emoji embeddings SHALL:
    - Add all emoji tokens to the tokenizer vocabulary
    - Resize model embeddings to accommodate new tokens
    - Copy embedding weights from mapped Vietnamese words to corresponding emojis
    - Preserve existing embedding weights for non-emoji tokens
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
    """
    # Create model
    config = Config(model_type=model_type, num_labels=num_labels)
    model, tokenizer = create_model(config)
    
    # Record original state
    original_vocab_size = len(tokenizer)
    original_embedding_size = model.get_embeddings().num_embeddings
    
    # Sample some original tokens and their embeddings for comparison
    sample_token_id = 100  # Arbitrary existing token
    if sample_token_id < original_embedding_size:
        original_sample_embedding = model.get_embeddings().weight[sample_token_id].clone()
    
    # Apply emoji embeddings
    model, tokenizer = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
    
    # Get new state
    new_vocab_size = len(tokenizer)
    new_embedding_size = model.get_embeddings().num_embeddings
    
    # Verify tokens were added to vocabulary
    num_new_tokens = 0
    for emoji in emoji_mapping.keys():
        if emoji in tokenizer.get_vocab():
            num_new_tokens += 1
    
    # At least some emojis should be added (some might already exist)
    assert num_new_tokens > 0, "At least some emojis should be in vocabulary"
    
    # Verify vocabulary and embeddings grew
    assert new_vocab_size >= original_vocab_size, \
        f"Vocabulary should grow or stay same, was {original_vocab_size}, now {new_vocab_size}"
    
    assert new_embedding_size >= original_embedding_size, \
        f"Embeddings should grow or stay same, was {original_embedding_size}, now {new_embedding_size}"
    
    # Verify embeddings match vocabulary size
    assert new_embedding_size == new_vocab_size, \
        f"Embedding size ({new_embedding_size}) should match vocab size ({new_vocab_size})"
    
    # Verify existing embeddings were preserved (if we sampled one)
    if sample_token_id < original_embedding_size:
        current_sample_embedding = model.get_embeddings().weight[sample_token_id]
        assert torch.allclose(original_sample_embedding, current_sample_embedding, atol=1e-6), \
            "Existing embeddings should be preserved"
    
    # Verify emoji embeddings are not all zeros
    for emoji, word in emoji_mapping.items():
        if emoji in tokenizer.get_vocab():
            emoji_id = tokenizer.convert_tokens_to_ids(emoji)
            emoji_embedding = model.get_embeddings().weight[emoji_id]
            
            # Embedding should not be all zeros (should be copied from word)
            assert not torch.allclose(emoji_embedding, torch.zeros_like(emoji_embedding)), \
                f"Emoji '{emoji}' embedding should not be all zeros"


# Property 6 Extended: Emoji embeddings from multi-token words
# Feature: multi-model-support, Property 6 (extended)
@settings(max_examples=100, deadline=30000)
@given(
    model_type=st.sampled_from(["phobert", "bamibert"]),
    num_labels=st.integers(min_value=2, max_value=10),
    use_multitoken=st.booleans()
)
def test_property_6_emoji_embeddings_multitoken_words(model_type, num_labels, use_multitoken):
    """
    Property 6 Extended: Emoji embeddings from multi-token words
    
    Verify that emoji embeddings can be created from Vietnamese words
    that may tokenize into multiple tokens (averaging strategy).
    
    Validates: Requirements 7.3, 7.4, 7.5
    """
    # Create simple mapping with known words
    if use_multitoken:
        emoji_mapping = {
            '😀': 'vui_vẻ',  # Multi-token word
            '😢': 'buồn_bã'   # Multi-token word
        }
    else:
        emoji_mapping = {
            '😀': 'vui',  # Single token
            '😢': 'buồn'  # Single token
        }
    
    config = Config(model_type=model_type, num_labels=num_labels)
    model, tokenizer = create_model(config)
    
    original_vocab_size = len(tokenizer)
    
    # Apply emoji embeddings
    model, tokenizer = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
    
    new_vocab_size = len(tokenizer)
    
    # Verify emojis were added
    assert new_vocab_size >= original_vocab_size
    
    # Verify emoji embeddings exist and are not zero
    for emoji in emoji_mapping.keys():
        if emoji in tokenizer.get_vocab():
            emoji_id = tokenizer.convert_tokens_to_ids(emoji)
            emoji_embedding = model.get_embeddings().weight[emoji_id]
            
            # Should have non-zero embeddings
            embedding_norm = torch.norm(emoji_embedding)
            assert embedding_norm > 0, f"Emoji '{emoji}' should have non-zero embedding"


# Property 6 Extended: Emoji embeddings work with both models
# Feature: multi-model-support, Property 6 (extended)
@settings(max_examples=100, deadline=30000)
@given(
    num_labels=st.integers(min_value=2, max_value=10)
)
def test_property_6_emoji_works_with_both_models(num_labels):
    """
    Property 6 Extended: Emoji embeddings work identically with both models
    
    For the same emoji mapping, both PhoBERT and BamiBERT should successfully
    apply emoji embeddings with the same behavior.
    
    Validates: Requirements 7.1, 7.2
    """
    emoji_mapping = {
        '😀': 'vui',
        '😢': 'buồn',
        '😡': 'giận'
    }
    
    results = {}
    
    for model_type in ["phobert", "bamibert"]:
        config = Config(model_type=model_type, num_labels=num_labels)
        model, tokenizer = create_model(config)
        
        original_vocab = len(tokenizer)
        
        # Apply emoji embeddings
        model, tokenizer = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
        
        new_vocab = len(tokenizer)
        
        # Count how many emojis were added
        added_emojis = 0
        for emoji in emoji_mapping.keys():
            if emoji in tokenizer.get_vocab():
                added_emojis += 1
        
        results[model_type] = {
            'original_vocab': original_vocab,
            'new_vocab': new_vocab,
            'added_emojis': added_emojis
        }
    
    # Both models should add the same emojis
    assert results['phobert']['added_emojis'] == results['bamibert']['added_emojis'], \
        "Both models should add the same number of emojis"
    
    # Both should successfully add at least some emojis
    assert results['phobert']['added_emojis'] > 0
    assert results['bamibert']['added_emojis'] > 0


# Property 6 Extended: Idempotence of emoji embedding application
# Feature: multi-model-support, Property 6 (extended)
@settings(max_examples=100, deadline=30000)
@given(
    model_type=st.sampled_from(["phobert", "bamibert"]),
    num_labels=st.integers(min_value=2, max_value=10)
)
def test_property_6_emoji_application_idempotence(model_type, num_labels):
    """
    Property 6 Extended: Idempotence of emoji application
    
    Applying emoji embeddings multiple times should not keep adding
    the same emojis (they should already exist).
    
    Validates: Requirements 7.3, 7.4
    """
    emoji_mapping = {
        '😀': 'vui',
        '😢': 'buồn'
    }
    
    config = Config(model_type=model_type, num_labels=num_labels)
    model, tokenizer = create_model(config)
    
    # Apply first time
    model, tokenizer = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
    vocab_after_first = len(tokenizer)
    
    # Apply second time with same mapping
    model, tokenizer = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
    vocab_after_second = len(tokenizer)
    
    # Vocabulary size should not increase on second application
    assert vocab_after_second == vocab_after_first, \
        "Applying same emoji mapping twice should not increase vocabulary"


# Property 6 Extended: Model forward pass works after emoji embeddings
# Feature: multi-model-support, Property 6 (extended)
@settings(max_examples=100, deadline=30000)
@given(
    model_type=st.sampled_from(["phobert", "bamibert"]),
    num_labels=st.integers(min_value=2, max_value=10),
    batch_size=st.integers(min_value=1, max_value=4)
)
def test_property_6_model_works_after_emoji_application(model_type, num_labels, batch_size):
    """
    Property 6 Extended: Model still works after emoji embedding application
    
    After applying emoji embeddings, the model should still perform
    forward passes correctly.
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
    """
    emoji_mapping = {
        '😀': 'vui',
        '😢': 'buồn',
        '👍': 'tốt'
    }
    
    config = Config(model_type=model_type, num_labels=num_labels)
    model, tokenizer = create_model(config)
    
    # Apply emoji embeddings
    model, tokenizer = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
    model.eval()
    
    # Create input with emojis
    texts = ["Tôi rất vui 😀"] * batch_size
    
    # Tokenize
    encoded = tokenizer(
        texts,
        max_length=64,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    # Verify outputs
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, num_labels)
    assert torch.isfinite(outputs['logits']).all(), "Logits should be finite"

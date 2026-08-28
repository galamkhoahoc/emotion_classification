"""
Property-based tests for tokenization and word segmentation.

Tests Properties 4 and 5 from the multi-model-support spec.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume
from transformers import AutoTokenizer
from configs.config import Config


# Generate Vietnamese-like text strings
def vietnamese_text_strategy():
    """Generate strings that could represent Vietnamese text."""
    # Generate text with Vietnamese characteristics
    return st.text(
        alphabet=st.characters(
            whitelist_categories=('Ll', 'Lu', 'Lt', 'Nd'),
            whitelist_characters=' áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ'
        ),
        min_size=1,
        max_size=500
    ).filter(lambda x: len(x.strip()) > 0)


# Fixtures for reusing tokenizers
@pytest.fixture(scope="module")
def phobert_tokenizer():
    """Cached PhoBERT tokenizer for property tests."""
    return AutoTokenizer.from_pretrained("vinai/phobert-base")


@pytest.fixture(scope="module")
def bamibert_tokenizer():
    """Cached BamiBERT tokenizer for property tests."""
    return AutoTokenizer.from_pretrained("Qualcomm-AI-Research/BamiBERT")


# Property 4: Tokenization length enforcement
# Feature: multi-model-support, Property 4: Tokenization length enforcement
@settings(max_examples=100, deadline=None)
@given(
    text=vietnamese_text_strategy(),
    max_length=st.integers(min_value=10, max_value=512),
    model_type=st.sampled_from(["phobert", "bamibert"])
)
def test_property_4_tokenization_length_enforcement(text, max_length, model_type, phobert_tokenizer, bamibert_tokenizer):
    """
    Property 4: Tokenization length enforcement
    
    For any input text of arbitrary length, tokenization with max_length parameter
    SHALL produce token sequences with length <= max_length, truncating longer
    sequences appropriately.
    
    Validates: Requirements 3.3, 3.4, 3.5
    """
    # Adjust max_length based on model constraints
    if model_type == "phobert" and max_length > 256:
        max_length = 256
    elif model_type == "bamibert" and max_length > 2048:
        max_length = 2048
    
    # Use cached tokenizer
    tokenizer = phobert_tokenizer if model_type == "phobert" else bamibert_tokenizer
    
    # Tokenize text
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Verify length constraint
    assert encoded['input_ids'].shape[1] <= max_length, \
        f"Tokenized length {encoded['input_ids'].shape[1]} exceeds max_length {max_length}"
    
    # With padding='max_length', length should equal max_length exactly
    assert encoded['input_ids'].shape[1] == max_length, \
        f"Expected exact length {max_length}, got {encoded['input_ids'].shape[1]}"
    
    # Verify attention mask has same length
    assert encoded['attention_mask'].shape[1] == max_length


# Property 4 Extended: Tokenization without padding
# Feature: multi-model-support, Property 4 (extended)
@settings(max_examples=100, deadline=None)
@given(
    text=vietnamese_text_strategy(),
    max_length=st.integers(min_value=10, max_value=256),
    model_type=st.sampled_from(["phobert", "bamibert"])
)
def test_property_4_tokenization_truncation_only(text, max_length, model_type, phobert_tokenizer, bamibert_tokenizer):
    """
    Property 4 Extended: Tokenization truncation without padding
    
    For any text, tokenization with truncation=True (no padding) should
    produce sequences with length <= max_length.
    
    Validates: Requirements 3.3, 3.4, 3.5
    """
    # Adjust max_length based on model constraints
    if model_type == "phobert" and max_length > 256:
        max_length = 256
    elif model_type == "bamibert" and max_length > 2048:
        max_length = 2048
    
    # Use cached tokenizer
    tokenizer = phobert_tokenizer if model_type == "phobert" else bamibert_tokenizer
    
    # Tokenize without padding
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    # Verify length constraint (should be <= max_length)
    actual_length = encoded['input_ids'].shape[1]
    assert actual_length <= max_length, \
        f"Tokenized length {actual_length} exceeds max_length {max_length}"


# Property 5: Word segmentation control
# Feature: multi-model-support, Property 5: Word segmentation control
@settings(max_examples=100, deadline=None)
@given(
    text=vietnamese_text_strategy(),
    model_type=st.sampled_from(["phobert", "bamibert"]),
    use_word_segmentation_input=st.booleans()
)
def test_property_5_word_segmentation_control(text, model_type, use_word_segmentation_input):
    """
    Property 5: Word segmentation control
    
    For any Vietnamese text input:
    - WHEN model_type is "phobert" AND use_word_segmentation is true,
      the text SHOULD be segmented before tokenization
    - WHEN model_type is "bamibert", the raw text SHOULD be tokenized
      without segmentation regardless of the use_word_segmentation flag
    
    Validates: Requirements 4.1, 4.2, 4.3, 4.4
    
    Note: This test verifies the configuration behavior. Actual word segmentation
    is applied in the dataset/preprocessing layer, not in the model factory.
    """
    config = Config(
        model_type=model_type,
        use_word_segmentation=use_word_segmentation_input
    )
    
    if model_type == "phobert":
        # PhoBERT should keep the use_word_segmentation setting
        # (unless explicitly set to False)
        if use_word_segmentation_input:
            assert config.use_word_segmentation is True
        else:
            assert config.use_word_segmentation is False
    
    elif model_type == "bamibert":
        # BamiBERT should always have word segmentation disabled
        assert config.use_word_segmentation is False, \
            "BamiBERT should always have use_word_segmentation=False"


# Property 5 Extended: Config enforces word segmentation for BamiBERT
# Feature: multi-model-support, Property 5 (extended)
@settings(max_examples=100)
@given(
    use_word_segmentation=st.booleans(),
    other_param=st.integers(min_value=1, max_value=100)
)
def test_property_5_bamibert_always_disables_segmentation(use_word_segmentation, other_param):
    """
    Property 5 Extended: BamiBERT always disables word segmentation
    
    For any configuration with model_type="bamibert", the use_word_segmentation
    flag SHALL be set to False regardless of input value.
    
    Validates: Requirements 4.2, 4.4, 4.5
    """
    config = Config(
        model_type="bamibert",
        use_word_segmentation=use_word_segmentation,
        batch_size=other_param  # Add another param to verify field preservation
    )
    
    assert config.use_word_segmentation is False
    assert config.batch_size == other_param


# Property 4 & 5 Integration: Max length and tokenization
# Feature: multi-model-support, Properties 4 & 5 integration
@settings(max_examples=100, deadline=None)
@given(
    text=vietnamese_text_strategy(),
    model_type=st.sampled_from(["phobert", "bamibert"])
)
def test_property_4_5_integration_model_specific_tokenization(text, model_type, phobert_tokenizer, bamibert_tokenizer):
    """
    Properties 4 & 5 Integration: Model-specific tokenization
    
    Verify that tokenization respects model-specific constraints:
    - PhoBERT: max_length defaults to 256
    - BamiBERT: max_length defaults to 2048, no word segmentation
    
    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 4.2, 4.4
    """
    config = Config(model_type=model_type)
    tokenizer = phobert_tokenizer if model_type == "phobert" else bamibert_tokenizer
    
    # Verify model-specific max_length defaults
    if model_type == "phobert":
        assert config.max_length == 256, f"PhoBERT should default to max_length=256"
        # PhoBERT has word segmentation enabled by default
        assert config.use_word_segmentation is True
    elif model_type == "bamibert":
        assert config.max_length == 2048, f"BamiBERT should default to max_length=2048"
        # BamiBERT should have word segmentation disabled
        assert config.use_word_segmentation is False
    
    # Tokenize with model-specific settings
    encoded = tokenizer(
        text,
        max_length=config.max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Verify tokenization respects max_length
    assert encoded['input_ids'].shape[1] == config.max_length


# Property: Tokenization produces valid model inputs
# Feature: multi-model-support, Property 4 (extended)
@settings(max_examples=50, deadline=None)  # Reduced examples since this creates models
@given(
    texts=st.lists(vietnamese_text_strategy(), min_size=1, max_size=3),  # Reduced batch size
    model_type=st.sampled_from(["phobert"]),  # Only PhoBERT for speed
    max_length=st.integers(min_value=16, max_value=64)  # Smaller max_length
)
def test_property_4_tokenization_produces_valid_inputs(texts, model_type, max_length):
    """
    Property 4 Extended: Tokenization produces valid model inputs
    
    For any batch of texts, tokenization should produce tensors that
    can be fed into the model without errors.
    
    Validates: Requirements 3.3, 3.4, 3.5, 2.1
    
    Note: This test creates models, so it's slower. Reduced to 50 examples.
    """
    from src.models.model_factory import create_model
    
    # Adjust max_length based on model constraints
    if model_type == "phobert" and max_length > 256:
        max_length = 256
    elif model_type == "bamibert" and max_length > 2048:
        max_length = 2048
    
    config = Config(model_type=model_type, max_length=max_length, num_labels=7)
    model, tokenizer = create_model(config)
    model.eval()
    
    # Tokenize batch
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Verify batch dimensions
    batch_size = len(texts)
    assert encoded['input_ids'].shape[0] == batch_size
    assert encoded['input_ids'].shape[1] == max_length
    assert encoded['attention_mask'].shape == encoded['input_ids'].shape
    
    # Verify model can process these inputs
    with torch.no_grad():
        outputs = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    # Verify valid outputs
    assert 'logits' in outputs
    assert outputs['logits'].shape[0] == batch_size

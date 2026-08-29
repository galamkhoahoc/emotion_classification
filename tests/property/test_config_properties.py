"""
Property-based tests for configuration validation.

Tests Properties 1, 2, and 9 from the multi-model-support spec.
"""

import pytest
from hypothesis import given, strategies as st, settings
from configs.config import Config


# Property 1: Invalid model type rejection
# Feature: multi-model-support, Property 1: Invalid model type rejection
@settings(max_examples=100)
@given(model_type=st.text().filter(lambda x: x.lower() not in ["phobert", "bamibert"] and len(x) > 0))
def test_property_1_invalid_model_type_rejected(model_type):
    """
    Property 1: Invalid model type rejection
    
    For any string that is not "phobert" or "bamibert" (case-insensitive),
    the Config initialization SHALL raise a ValueError with a message
    indicating valid options.
    
    **Validates: Requirements 1.4**
    """
    with pytest.raises(ValueError) as exc_info:
        Config(model_type=model_type)
    
    error_message = str(exc_info.value)
    # Verify the error message is descriptive and indicates the invalid value
    assert "Invalid model_type" in error_message or "model_type" in error_message
    # Verify the error message mentions valid options
    assert "phobert" in error_message.lower() or "bamibert" in error_message.lower()


# Property 1: Extended edge cases for invalid model type rejection
@pytest.mark.parametrize("invalid_model_type", [
    "",  # Empty string
    "   ",  # Whitespace only
    "PHOBERT_V2",  # Similar but invalid
    "bamibert-large",  # Similar but invalid
    "bert",  # Generic BERT
    "roberta",  # Different model entirely
    "123",  # Numeric
    "pho bert",  # With space
    "bami_bert",  # With underscore
    "xlmroberta",  # Different model
    "vibert",  # Similar sounding but different
])
def test_property_1_invalid_model_type_edge_cases(invalid_model_type):
    """
    Property 1: Edge cases for invalid model type rejection
    
    Test specific edge cases to ensure comprehensive validation
    of invalid model types, including empty strings, whitespace,
    similar-looking names, and completely different model types.
    
    **Validates: Requirements 1.4**
    """
    with pytest.raises(ValueError) as exc_info:
        Config(model_type=invalid_model_type)
    
    error_message = str(exc_info.value)
    # Verify the error message mentions the issue with model_type
    assert "Invalid model_type" in error_message or "model_type" in error_message


# Property 1: Verify error message quality
def test_property_1_error_message_quality():
    """
    Property 1: Error message quality verification
    
    Verify that the error message for invalid model types includes:
    1. Clear indication of what's wrong (Invalid model_type)
    2. The actual invalid value that was provided
    3. The valid options (phobert and bamibert)
    
    **Validates: Requirements 1.4**
    """
    invalid_value = "invalid_model"
    
    with pytest.raises(ValueError) as exc_info:
        Config(model_type=invalid_value)
    
    error_message = str(exc_info.value)
    
    # Check that error message is descriptive
    assert "Invalid model_type" in error_message, "Error message should indicate invalid model_type"
    
    # Check that the invalid value is mentioned
    assert invalid_value in error_message, "Error message should show the invalid value provided"
    
    # Check that valid options are mentioned
    error_lower = error_message.lower()
    assert "phobert" in error_lower, "Error message should mention 'phobert' as a valid option"
    assert "bamibert" in error_lower, "Error message should mention 'bamibert' as a valid option"


# Property 2: Configuration field preservation
# Feature: multi-model-support, Property 2: Configuration field preservation
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=128),
    learning_rate=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False),
    num_epochs=st.integers(min_value=1, max_value=100),
    num_labels=st.integers(min_value=2, max_value=50),
    model_type=st.sampled_from(["phobert", "bamibert", "PhoBERT", "BamiBERT"]),
    weight_decay=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    warmup_steps=st.integers(min_value=0, max_value=1000),
    gradient_accumulation_steps=st.integers(min_value=1, max_value=8),
    adam_beta1=st.floats(min_value=0.8, max_value=0.95, allow_nan=False, allow_infinity=False),
    adam_beta2=st.floats(min_value=0.95, max_value=0.999, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000),
    logging_steps=st.integers(min_value=10, max_value=500),
    early_stopping_patience=st.integers(min_value=1, max_value=10)
)
def test_property_2_config_field_preservation(
    batch_size, learning_rate, num_epochs, num_labels, model_type, weight_decay,
    warmup_steps, gradient_accumulation_steps, adam_beta1, adam_beta2, seed,
    logging_steps, early_stopping_patience
):
    """
    Property 2: Configuration field preservation
    
    For any valid configuration with arbitrary values in non-model fields
    (batch_size, learning_rate, num_epochs, etc.), setting or changing
    the model_type SHALL preserve all other field values unchanged.
    
    Validates: Requirements 1.7, 6.1, 6.6
    """
    config = Config(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_labels=num_labels,
        model_type=model_type,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        seed=seed,
        logging_steps=logging_steps,
        early_stopping_patience=early_stopping_patience
    )
    
    # Verify all non-model fields are preserved
    assert config.batch_size == batch_size
    assert config.learning_rate == learning_rate
    assert config.num_epochs == num_epochs
    assert config.num_labels == num_labels
    assert config.weight_decay == weight_decay
    assert config.warmup_steps == warmup_steps
    assert config.gradient_accumulation_steps == gradient_accumulation_steps
    assert config.adam_beta1 == adam_beta1
    assert config.adam_beta2 == adam_beta2
    assert config.seed == seed
    assert config.logging_steps == logging_steps
    assert config.early_stopping_patience == early_stopping_patience
    
    # model_type should be normalized to lowercase
    assert config.model_type == model_type.lower()


# Property 2 Extended: Configuration field preservation when changing model type
# Feature: multi-model-support, Property 2: Configuration field preservation
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=128),
    learning_rate=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False),
    num_epochs=st.integers(min_value=1, max_value=100),
    num_labels=st.integers(min_value=2, max_value=50),
    weight_decay=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    warmup_steps=st.integers(min_value=0, max_value=1000),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_2_config_preservation_across_model_change(
    batch_size, learning_rate, num_epochs, num_labels, weight_decay, warmup_steps, seed
):
    """
    Property 2 Extended: Configuration field preservation when changing model type
    
    For any valid configuration, changing from one model_type to another
    SHALL preserve all non-model-specific field values unchanged.
    
    This test explicitly verifies preservation when switching models.
    
    Validates: Requirements 1.7, 6.1, 6.6
    """
    # Create config with PhoBERT first
    config_phobert = Config(
        model_type="phobert",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_labels=num_labels,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seed=seed
    )
    
    # Create config with BamiBERT using same parameters
    config_bamibert = Config(
        model_type="bamibert",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_labels=num_labels,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seed=seed
    )
    
    # Verify all non-model fields are preserved across model types
    assert config_phobert.batch_size == config_bamibert.batch_size == batch_size
    assert config_phobert.learning_rate == config_bamibert.learning_rate == learning_rate
    assert config_phobert.num_epochs == config_bamibert.num_epochs == num_epochs
    assert config_phobert.num_labels == config_bamibert.num_labels == num_labels
    assert config_phobert.weight_decay == config_bamibert.weight_decay == weight_decay
    assert config_phobert.warmup_steps == config_bamibert.warmup_steps == warmup_steps
    assert config_phobert.seed == config_bamibert.seed == seed
    
    # Model-specific fields should differ
    assert config_phobert.model_type == "phobert"
    assert config_bamibert.model_type == "bamibert"
    assert config_phobert.model_name == "vinai/phobert-base"
    assert config_bamibert.model_name == "Qualcomm-AI-Research/BamiBERT"


# Property 9: Configuration validation enforcement
# Feature: multi-model-support, Property 9: Configuration validation enforcement
@settings(max_examples=100)
@given(
    scenario=st.sampled_from([
        "phobert_large_max_length",
        "bamibert_large_max_length", 
        "invalid_num_labels_zero",
        "invalid_num_labels_negative",
        "invalid_num_labels_one"
    ])
)
def test_property_9_config_validation_enforcement(scenario):
    """
    Property 9: Configuration validation enforcement
    
    For any configuration where:
    - max_length > 256 for model_type="phobert", OR
    - max_length > 2048 for model_type="bamibert", OR
    - num_labels < 2
    
    The Config initialization SHALL either raise an error (for num_labels < 2)
    or auto-correct invalid values where applicable.
    
    Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
    """
    if scenario == "phobert_large_max_length":
        # PhoBERT should cap max_length at 256
        config = Config(model_type="phobert", max_length=512)
        assert config.max_length == 256
        
    elif scenario == "bamibert_large_max_length":
        # BamiBERT should cap max_length at 2048
        config = Config(model_type="bamibert", max_length=4096)
        assert config.max_length == 2048
        
    elif scenario in ["invalid_num_labels_zero", "invalid_num_labels_negative", "invalid_num_labels_one"]:
        # Invalid num_labels should raise ValueError
        num_labels_map = {
            "invalid_num_labels_zero": 0,
            "invalid_num_labels_negative": -5,
            "invalid_num_labels_one": 1
        }
        with pytest.raises(ValueError) as exc_info:
            Config(num_labels=num_labels_map[scenario])
        assert "num_labels" in str(exc_info.value)


# Additional validation property for word segmentation
# Feature: multi-model-support, Property 9 (extended)
@settings(max_examples=100)
@given(
    use_word_segmentation=st.booleans(),
    model_type=st.sampled_from(["bamibert", "BamiBERT"])
)
def test_property_9_bamibert_word_segmentation_disabled(use_word_segmentation, model_type):
    """
    Property 9 Extended: BamiBERT auto-disables word segmentation
    
    For any model_type="bamibert", the Config SHALL set
    use_word_segmentation to False regardless of the input value.
    
    Validates: Requirements 4.5, 11.4
    """
    config = Config(model_type=model_type, use_word_segmentation=use_word_segmentation)
    assert config.use_word_segmentation is False
    assert config.model_type == "bamibert"


# Property for preserving valid max_length values
# Feature: multi-model-support, Property 9 (extended)
@settings(max_examples=100)
@given(
    max_length=st.integers(min_value=10, max_value=256),
    model_type=st.sampled_from(["phobert", "PhoBERT"])
)
def test_property_9_phobert_preserves_valid_max_length(max_length, model_type):
    """
    Property 9 Extended: PhoBERT preserves valid max_length
    
    For any max_length <= 256 with model_type="phobert",
    the Config SHALL preserve the specified max_length.
    
    Validates: Requirements 3.1, 11.2
    """
    config = Config(model_type=model_type, max_length=max_length)
    assert config.max_length == max_length


@settings(max_examples=100)
@given(
    max_length=st.integers(min_value=10, max_value=2048),
    model_type=st.sampled_from(["bamibert", "BamiBERT"])
)
def test_property_9_bamibert_preserves_valid_max_length(max_length, model_type):
    """
    Property 9 Extended: BamiBERT preserves valid max_length
    
    For any max_length <= 2048 with model_type="bamibert",
    the Config SHALL preserve the specified max_length.
    
    Validates: Requirements 3.2, 11.3
    """
    config = Config(model_type=model_type, max_length=max_length)
    assert config.max_length == max_length

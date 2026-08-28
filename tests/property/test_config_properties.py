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
@given(model_type=st.text().filter(lambda x: x.lower() not in ["phobert", "bamibert"]))
def test_property_1_invalid_model_type_rejected(model_type):
    """
    Property 1: Invalid model type rejection
    
    For any string that is not "phobert" or "bamibert" (case-insensitive),
    the Config initialization SHALL raise a ValueError with a message
    indicating valid options.
    
    Validates: Requirements 1.4
    """
    with pytest.raises(ValueError) as exc_info:
        Config(model_type=model_type)
    
    error_message = str(exc_info.value)
    assert "Invalid model_type" in error_message or "model_type" in error_message


# Property 2: Configuration field preservation
# Feature: multi-model-support, Property 2: Configuration field preservation
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=128),
    learning_rate=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False),
    num_epochs=st.integers(min_value=1, max_value=100),
    num_labels=st.integers(min_value=2, max_value=50),
    model_type=st.sampled_from(["phobert", "bamibert", "PhoBERT", "BamiBERT"]),
    weight_decay=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
)
def test_property_2_config_field_preservation(
    batch_size, learning_rate, num_epochs, num_labels, model_type, weight_decay
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
        weight_decay=weight_decay
    )
    
    # Verify all non-model fields are preserved
    assert config.batch_size == batch_size
    assert config.learning_rate == learning_rate
    assert config.num_epochs == num_epochs
    assert config.num_labels == num_labels
    assert config.weight_decay == weight_decay
    
    # model_type should be normalized to lowercase
    assert config.model_type == model_type.lower()


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

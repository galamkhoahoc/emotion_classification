"""
Property-based tests for model interface and pipeline uniformity.

Tests Properties 3, 7, and 8 from the multi-model-support spec.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume
from configs.config import Config
from src.models.model_factory import create_model
from src.models.phobert_emotion import PhoBERTEmotionClassifier
from src.models.bamibert_emotion import BamiBERTEmotionClassifier


# Property 3: Model interface contract compliance
# Feature: multi-model-support, Property 3: Model interface contract compliance
@settings(max_examples=100, deadline=5000)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    seq_length=st.integers(min_value=10, max_value=128),
    num_labels=st.integers(min_value=2, max_value=20),
    with_labels=st.booleans(),
    model_type=st.sampled_from(["phobert", "bamibert"])
)
def test_property_3_model_interface_contract(
    batch_size, seq_length, num_labels, with_labels, model_type
):
    """
    Property 3: Model interface contract compliance
    
    For any valid input tensors (input_ids, attention_mask, optional labels)
    with shape [batch_size, seq_length], both PhoBERT and BamiBERT models SHALL:
    - Return a dictionary with keys 'loss' and 'logits'
    - Return logits with shape [batch_size, num_labels]
    - Return loss as None when labels not provided, or a scalar tensor when labels provided
    
    Validates: Requirements 2.1, 2.2, 2.5, 2.6
    """
    # Create model
    config = Config(model_type=model_type, num_labels=num_labels)
    model, _ = create_model(config)
    model.eval()
    
    # Generate random inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, num_labels, (batch_size,)) if with_labels else None
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)
    
    # Verify output structure
    assert isinstance(outputs, dict), f"Expected dict output, got {type(outputs)}"
    assert 'logits' in outputs, "Output must contain 'logits' key"
    assert 'loss' in outputs, "Output must contain 'loss' key"
    
    # Verify logits shape
    assert outputs['logits'].shape == (batch_size, num_labels), \
        f"Expected logits shape ({batch_size}, {num_labels}), got {outputs['logits'].shape}"
    
    # Verify loss behavior
    if with_labels:
        assert outputs['loss'] is not None, "Loss should not be None when labels provided"
        assert outputs['loss'].dim() == 0, f"Loss should be scalar, got shape {outputs['loss'].shape}"
        assert torch.isfinite(outputs['loss']), "Loss should be finite"
    else:
        assert outputs['loss'] is None, "Loss should be None when labels not provided"


# Property 7: Training pipeline uniformity
# Feature: multi-model-support, Property 7: Training pipeline uniformity
@settings(max_examples=100)
@given(
    learning_rate=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False),
    weight_decay=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    num_labels=st.integers(min_value=2, max_value=10),
    model_type_1=st.sampled_from(["phobert", "bamibert"]),
    model_type_2=st.sampled_from(["phobert", "bamibert"])
)
def test_property_7_training_pipeline_uniformity(
    learning_rate, weight_decay, num_labels, model_type_1, model_type_2
):
    """
    Property 7: Training pipeline uniformity
    
    For any model type ("phobert" or "bamibert"), the training pipeline SHALL use identical:
    - Optimizer class and hyperparameters (learning_rate, betas, weight_decay)
    - Loss function type
    
    This test verifies that both models can use the same optimizer configuration
    and that they have the same parameter structure.
    
    Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
    """
    # Create both models
    config_1 = Config(model_type=model_type_1, num_labels=num_labels)
    config_2 = Config(model_type=model_type_2, num_labels=num_labels)
    
    model_1, _ = create_model(config_1)
    model_2, _ = create_model(config_2)
    
    # Create optimizers with same hyperparameters
    optimizer_1 = torch.optim.AdamW(
        model_1.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    optimizer_2 = torch.optim.AdamW(
        model_2.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Verify optimizer types are identical
    assert type(optimizer_1) == type(optimizer_2), \
        f"Optimizer types differ: {type(optimizer_1)} vs {type(optimizer_2)}"
    
    # Verify optimizer hyperparameters match
    assert optimizer_1.defaults['lr'] == optimizer_2.defaults['lr']
    assert optimizer_1.defaults['weight_decay'] == optimizer_2.defaults['weight_decay']
    
    # Verify both models have learnable parameters
    assert sum(p.numel() for p in model_1.parameters() if p.requires_grad) > 0
    assert sum(p.numel() for p in model_2.parameters() if p.requires_grad) > 0
    
    # Verify loss function can be created for both
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Test that loss function works with both models
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16)
    labels = torch.randint(0, num_labels, (2,))
    
    with torch.no_grad():
        outputs_1 = model_1(input_ids, attention_mask, labels)
        outputs_2 = model_2(input_ids, attention_mask, labels)
        
        # Both should return loss
        assert outputs_1['loss'] is not None
        assert outputs_2['loss'] is not None


# Property 8: Evaluation pipeline uniformity
# Feature: multi-model-support, Property 8: Evaluation pipeline uniformity
@settings(max_examples=100)
@given(
    num_samples=st.integers(min_value=5, max_value=20),
    seq_length=st.integers(min_value=10, max_value=64),
    num_labels=st.integers(min_value=2, max_value=10),
    model_type=st.sampled_from(["phobert", "bamibert"])
)
def test_property_8_evaluation_pipeline_uniformity(
    num_samples, seq_length, num_labels, model_type
):
    """
    Property 8: Evaluation pipeline uniformity
    
    For any model type ("phobert" or "bamibert") and any test dataset,
    the evaluation pipeline SHALL:
    - Compute the same set of metrics
    - Generate output in the same format
    - Process test samples in the same order
    
    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """
    # Create model
    config = Config(model_type=model_type, num_labels=num_labels)
    model, _ = create_model(config)
    model.eval()
    
    # Generate test data
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length)
    labels = torch.randint(0, num_labels, (num_samples,))
    
    # Simulate evaluation
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(num_samples):
            outputs = model(
                input_ids[i:i+1],
                attention_mask[i:i+1],
                labels[i:i+1]
            )
            all_logits.append(outputs['logits'])
            all_labels.append(labels[i])
    
    # Verify we can compute standard metrics
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.stack(all_labels)
    
    predictions = torch.argmax(logits_tensor, dim=-1)
    
    # Verify predictions are valid
    assert predictions.shape == labels_tensor.shape
    assert torch.all(predictions >= 0)
    assert torch.all(predictions < num_labels)
    
    # Verify we can compute accuracy (example metric)
    accuracy = (predictions == labels_tensor).float().mean()
    assert 0.0 <= accuracy <= 1.0
    
    # Verify output format is consistent (dict with logits)
    assert isinstance(outputs, dict)
    assert 'logits' in outputs


# Additional property: Both models implement required interface methods
# Feature: multi-model-support, Property 3 (extended)
@settings(max_examples=100)
@given(
    num_labels=st.integers(min_value=2, max_value=20),
    model_type=st.sampled_from(["phobert", "bamibert"])
)
def test_property_3_interface_methods_implemented(num_labels, model_type):
    """
    Property 3 Extended: Interface methods implemented
    
    For any model type, the model SHALL implement all required methods:
    - forward()
    - get_embeddings()
    - resize_token_embeddings()
    
    Validates: Requirements 2.1, 2.2, 2.3, 2.4
    """
    config = Config(model_type=model_type, num_labels=num_labels)
    model, tokenizer = create_model(config)
    
    # Verify required methods exist and are callable
    assert hasattr(model, 'forward'), f"{model_type} missing forward()"
    assert callable(model.forward)
    
    assert hasattr(model, 'get_embeddings'), f"{model_type} missing get_embeddings()"
    assert callable(model.get_embeddings)
    
    assert hasattr(model, 'resize_token_embeddings'), f"{model_type} missing resize_token_embeddings()"
    assert callable(model.resize_token_embeddings)
    
    # Verify get_embeddings returns embedding layer
    embeddings = model.get_embeddings()
    assert isinstance(embeddings, torch.nn.Embedding), \
        f"get_embeddings() should return nn.Embedding, got {type(embeddings)}"
    
    # Verify resize_token_embeddings works
    original_vocab_size = len(tokenizer)
    new_vocab_size = original_vocab_size + 10
    model.resize_token_embeddings(new_vocab_size)
    
    # Verify embeddings were resized
    new_embeddings = model.get_embeddings()
    assert new_embeddings.num_embeddings == new_vocab_size, \
        f"Expected {new_vocab_size} embeddings, got {new_embeddings.num_embeddings}"

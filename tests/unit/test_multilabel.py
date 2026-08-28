"""
Tests for multilabel pipelines, covering dataset, models, metrics, and predictor.
"""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer
from configs.config import Config
from src.data.dataset import EmotionDataset
from src.models.model_factory import create_model, create_loss_function
from src.utils.metrics import (
    compute_multilabel_metrics,
    get_predictions_from_logits,
    get_probabilities_from_logits
)
from src.utils.predictor import Predictor, ThresholdOptimizer
from hypothesis import given, strategies as st


@pytest.fixture
def mock_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base")

@pytest.fixture
def multilabel_config():
    return Config(
        problem_type="multilabel_classification",
        num_labels=5,
        dataset_name="custom",
        model_name="vinai/phobert-base",
        model_type="phobert"
    )

def test_dataset_multilabel(mock_tokenizer, multilabel_config):
    texts = ["Test text"]
    labels = [[1, 0, 1, 0, 0]]
    dataset = EmotionDataset(
        texts=texts,
        labels=labels,
        tokenizer=mock_tokenizer,
        max_length=16,
        num_labels=5,
        problem_type=multilabel_config.problem_type
    )
    
    item = dataset[0]
    assert 'input_ids' in item
    assert 'labels' in item
    assert item['labels'].shape == (5,)
    assert item['labels'].dtype == torch.float32

def test_loss_function_factory_multilabel(multilabel_config):
    loss_fn = create_loss_function(multilabel_config)
    assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)

def test_phobert_forward_multilabel(multilabel_config):
    model, _ = create_model(multilabel_config)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones((2, 16))
    labels = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]], dtype=torch.float32)
    
    outputs = model(input_ids, attention_mask, labels=labels)
    assert 'logits' in outputs
    assert 'loss' in outputs
    assert outputs['logits'].shape == (2, 5)
    assert outputs['loss'] is not None

def test_metrics_multilabel():
    preds = np.array([[1, 0, 1], [0, 1, 0]])
    labels = np.array([[1, 0, 1], [1, 1, 0]])
    
    metrics = compute_multilabel_metrics(preds, labels)
    assert 'f1_macro' in metrics
    assert 'hamming_loss' in metrics
    assert metrics['hamming_loss'] == 1.0 / 6.0  # 1 mistake out of 6 elements

def test_prediction_conversions():
    logits = torch.tensor([[-10.0, 10.0, 0.0], [10.0, -10.0, 10.0]])
    probs = get_probabilities_from_logits(logits, "multilabel_classification")
    preds = get_predictions_from_logits(logits, "multilabel_classification", threshold=0.5)
    
    assert torch.allclose(probs[0, 1], torch.tensor(1.0), atol=1e-3)
    assert preds[0, 1] == 1.0
    assert preds[0, 0] == 0.0
    assert preds[0, 2] == 1.0

from hypothesis import settings, HealthCheck
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    logits=st.lists(st.lists(st.floats(min_value=-10, max_value=10), min_size=3, max_size=3), min_size=2, max_size=2)
)
def test_predictor_threshold_property(logits):
    t_logits = torch.tensor(logits)
    probs = get_probabilities_from_logits(t_logits, "multilabel_classification")
    preds = get_predictions_from_logits(t_logits, "multilabel_classification", threshold=0.5)
    assert preds.shape == t_logits.shape
    assert torch.all((preds == 0.0) | (preds == 1.0))

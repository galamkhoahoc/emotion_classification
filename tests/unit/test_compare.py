"""
Tests for compare_experiments.py and Hugging Face Hub model loading.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
import pandas as pd
from scripts.compare_experiments import load_experiment_metrics, create_comparison_table, print_best_experiment
from transformers import AutoModel, AutoTokenizer

def test_compare_experiments_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "exp1"
        exp_dir.mkdir()
        
        # Write mock metrics
        metrics = {
            "train_loss": [0.5, 0.4],
            "val_loss": [0.6, 0.5],
            "val_metrics": [{"f1_macro": 0.5}, {"f1_macro": 0.7}]
        }
        with open(exp_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f)
            
        # Write mock config
        config = {"model_type": "phobert", "learning_rate": 2e-5}
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f)
            
        data = load_experiment_metrics(str(exp_dir))
        assert data is not None
        assert data["name"] == "exp1"
        assert data["metrics"]["val_loss"][1] == 0.5
        assert data["config"]["model_type"] == "phobert"


def test_compare_experiments_table():
    experiments = [
        {
            "name": "exp1",
            "metrics": {"val_metrics": [{"f1_macro": 0.8, "accuracy": 0.9}]},
            "config": {"batch_size": 16}
        },
        {
            "name": "exp2",
            "metrics": {"val_metrics": [{"f1_macro": 0.85, "accuracy": 0.92}]},
            "config": {"batch_size": 32}
        }
    ]
    df = create_comparison_table(experiments)
    assert len(df) == 2
    assert "F1 Macro" in df.columns
    assert df.iloc[0]["F1 Macro"] == 0.8
    assert df.iloc[1]["F1 Macro"] == 0.85


def test_huggingface_hub_loading():
    """Test loading PhoBERT from HF hub."""
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    assert tokenizer is not None
    assert tokenizer.vocab_size > 0
    # Model loading might be too heavy for a quick test, but tokenizers are fast and verify connectivity.

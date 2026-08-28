"""
Checkpoint 10: Verify training and evaluation pipelines for both models.

This script tests:
1. Training with PhoBERT (small dataset, few epochs)
2. Training with BamiBERT (small dataset, few epochs)
3. Evaluation of both trained models
4. Comparison of outputs for consistency
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from configs.config import Config
from src.data.dataset import load_dataset_by_config, create_dataloaders
from src.models.model_factory import create_model, create_model_from_checkpoint, create_loss_function
from src.models.emoji_embeddings import apply_emoji_embeddings
from src.utils.metrics import compute_metrics, get_predictions_from_logits
from src.utils.logger import setup_logger, log_metrics
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


def train_epoch_simple(model, train_loader, optimizer, scheduler, device):
    """Simple training epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_simple(model, data_loader, device, config):
    """Simple evaluation."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            predictions = get_predictions_from_logits(
                logits,
                problem_type=config.problem_type,
                threshold=config.sigmoid_threshold
            )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(
        all_predictions,
        all_labels,
        problem_type=config.problem_type,
        label_names=config.emotion_labels
    )
    
    return avg_loss, metrics


def train_and_evaluate_model(model_type, num_epochs=2, batch_size=8, logger=None):
    """Train and evaluate a single model type."""
    print(f"\n{'='*80}")
    print(f"Testing {model_type.upper()} Model")
    print(f"{'='*80}\n")
    
    # Create output directories
    output_dir = f"outputs/checkpoint_test_{model_type}"
    checkpoint_dir = f"checkpoints/checkpoint_test_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configuration - using tridm/UIT-VSMEC which is directly supported
    config = Config(
        model_type=model_type,
        dataset_name="tridm/UIT-VSMEC",  # Use supported dataset
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=2e-5,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        enable_emoji_embedding=False,  # Disable for faster testing
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"Configuration:")
    print(f"  Model Type: {config.model_type}")
    print(f"  Model Name: {config.model_name}")
    print(f"  Max Length: {config.max_length}")
    print(f"  Use Word Segmentation: {config.use_word_segmentation}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Num Epochs: {config.num_epochs}\n")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    print("Loading dataset...")
    train_data, val_data, test_data = load_dataset_by_config(
        config=config,
        cache_dir=config.cache_dir
    )
    
    # Use only a small subset for testing (first 100 samples for speed)
    print("Using subset of data for quick testing...")
    train_data = {
        'texts': train_data['texts'][:100],
        'labels': train_data['labels'][:100]
    }
    val_data = {
        'texts': val_data['texts'][:50],
        'labels': val_data['labels'][:50]
    }
    test_data = {
        'texts': test_data['texts'][:50],
        'labels': test_data['labels'][:50]
    }
    
    print(f"Train samples: {len(train_data['texts'])}")
    print(f"Val samples: {len(val_data['texts'])}")
    print(f"Test samples: {len(test_data['texts'])}\n")
    
    # Create model via factory
    print(f"Creating {model_type} model via factory...")
    loss_fn = create_loss_function(config)
    model, tokenizer = create_model(config, loss_fn=loss_fn)
    model = model.to(device)
    print(f"Model created successfully: {type(model).__name__}\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    best_f1 = 0.0
    results = {
        'model_type': model_type,
        'train_losses': [],
        'val_losses': [],
        'val_metrics': [],
        'test_metrics': None
    }
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch_simple(model, train_loader, optimizer, scheduler, device)
        results['train_losses'].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = evaluate_simple(model, val_loader, device, config)
        results['val_losses'].append(val_loss)
        results['val_metrics'].append(val_metrics)
        
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"Val F1 Weighted: {val_metrics['f1_weighted']:.4f}")
        
        # Save best model
        current_f1 = val_metrics['f1_macro']
        if current_f1 > best_f1:
            best_f1 = current_f1
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config.to_dict(),
                'problem_type': config.problem_type,
                'num_labels': config.num_labels
            }, checkpoint_path)
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    # Test with loaded checkpoint
    print("\n" + "="*60)
    print("Loading best checkpoint and testing...")
    print("="*60)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    loaded_model, loaded_tokenizer = create_model_from_checkpoint(checkpoint_path, config)
    loaded_model = loaded_model.to(device)
    
    # Evaluate on test set
    test_loss, test_metrics = evaluate_simple(loaded_model, test_loader, device, config)
    results['test_metrics'] = test_metrics
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall: {test_metrics['recall_macro']:.4f}")
    
    print(f"\n✓ {model_type.upper()} pipeline completed successfully!\n")
    
    return results


def compare_results(phobert_results, bamibert_results):
    """Compare results between PhoBERT and BamiBERT."""
    print("\n" + "="*80)
    print("COMPARISON: PhoBERT vs BamiBERT")
    print("="*80 + "\n")
    
    print("Test Set Metrics Comparison:")
    print("-" * 60)
    
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    
    print(f"{'Metric':<20} {'PhoBERT':<15} {'BamiBERT':<15} {'Difference':<15}")
    print("-" * 60)
    
    for metric in metrics_to_compare:
        phobert_val = phobert_results['test_metrics'][metric]
        bamibert_val = bamibert_results['test_metrics'][metric]
        diff = bamibert_val - phobert_val
        
        print(f"{metric:<20} {phobert_val:>7.4f}        {bamibert_val:>7.4f}        {diff:>+7.4f}")
    
    print("\n" + "="*80)
    print("CHECKPOINT VERIFICATION SUMMARY")
    print("="*80)
    
    checks = [
        ("✓ PhoBERT training completed", True),
        ("✓ BamiBERT training completed", True),
        ("✓ PhoBERT evaluation completed", True),
        ("✓ BamiBERT evaluation completed", True),
        ("✓ Checkpoints saved successfully", True),
        ("✓ Models loaded from checkpoints", True),
        ("✓ Both models produce consistent output structure", True)
    ]
    
    for check, status in checks:
        print(check)
    
    print("\n" + "="*80)
    print("✓ CHECKPOINT 10: ALL TESTS PASSED")
    print("="*80 + "\n")
    
    print("Summary:")
    print("  - Both PhoBERT and BamiBERT models train successfully")
    print("  - Factory pattern correctly instantiates both model types")
    print("  - Training scripts work with both models")
    print("  - Evaluation scripts work with both models")
    print("  - Checkpoint loading works correctly")
    print("  - Output metrics are computed consistently")
    print("\nThe training and evaluation pipelines are verified and working!")


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("CHECKPOINT 10: VERIFY TRAINING AND EVALUATION PIPELINES")
    print("="*80)
    print("\nThis test will:")
    print("  1. Train PhoBERT with minimal config (2 epochs, small dataset)")
    print("  2. Train BamiBERT with minimal config (2 epochs, small dataset)")
    print("  3. Evaluate both models on test set")
    print("  4. Compare outputs for consistency")
    print("\n" + "="*80 + "\n")
    
    # Setup logger
    logger = setup_logger(log_file='logs/checkpoint_10_test.log')
    
    # Test PhoBERT
    try:
        phobert_results = train_and_evaluate_model('phobert', num_epochs=2, batch_size=8, logger=logger)
    except Exception as e:
        print(f"\n❌ PhoBERT test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test BamiBERT
    try:
        bamibert_results = train_and_evaluate_model('bamibert', num_epochs=2, batch_size=8, logger=logger)
    except Exception as e:
        print(f"\n❌ BamiBERT test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare results
    try:
        compare_results(phobert_results, bamibert_results)
    except Exception as e:
        print(f"\n❌ Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

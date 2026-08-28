"""
Training script for ViEmoText emotion classification model.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from src.data.dataset import load_uit_vsmec_dataset, create_dataloaders
from src.models.model_factory import create_model
from src.models.emoji_embeddings import apply_emoji_embeddings, load_emoji_mapping_from_file
from src.losses.focal_loss import FocalLoss
from src.losses.weighted_cross_entropy import WeightedCrossEntropyLoss, compute_class_weights
from src.utils.metrics import compute_metrics, get_predictions_from_logits
from src.utils.logger import setup_logger, log_metrics, MetricsTracker

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    logger
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Logger instance
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    logger
) -> tuple:
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        logger: Logger instance
    
    Returns:
        Tuple of (average_loss, metrics_dict, predictions, labels)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = get_predictions_from_logits(logits)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics, all_predictions, all_labels


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train ViEmoText model")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model_type', type=str, default=None, choices=['phobert', 'bamibert'],
                        help='Model type to use (phobert or bamibert)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no_emoji', action='store_true', help='Disable emoji embeddings')
    args = parser.parse_args()
    
    # Load configuration
    config_kwargs = {}
    if args.model_type:
        config_kwargs['model_type'] = args.model_type
    if args.batch_size:
        config_kwargs['batch_size'] = args.batch_size
    if args.num_epochs:
        config_kwargs['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config_kwargs['learning_rate'] = args.learning_rate
    if args.output_dir:
        config_kwargs['output_dir'] = args.output_dir
    if args.no_emoji:
        config_kwargs['enable_emoji_embedding'] = False
    
    config = Config(**config_kwargs)
    
    # Setup logger
    log_file = os.path.join(config.log_dir, 'training.log')
    logger = setup_logger(log_file=log_file)
    
    logger.info("="*60)
    logger.info("ViEmoText Training")
    logger.info("="*60)
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model and tokenizer via factory
    logger.info(f"Creating model via factory: {config.model_type} ({config.model_name})")
    model, tokenizer = create_model(config)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_data, val_data, test_data = load_uit_vsmec_dataset(
        dataset_name=config.dataset_name,
        cache_dir=config.cache_dir
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers
    )
    
    # Apply emoji embeddings
    if config.enable_emoji_embedding:
        logger.info("Applying emoji embeddings...")
        emoji_mapping = None
        if config.emoji_mapping_file:
            emoji_mapping = load_emoji_mapping_from_file(config.emoji_mapping_file)
        model = apply_emoji_embeddings(model, tokenizer, emoji_mapping)
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    logger.info(f"Total steps: {total_steps}")
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger
        )
        
        # Evaluate on validation set
        val_loss, val_metrics, val_preds, val_labels = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            logger=logger
        )
        
        # Log metrics
        log_metrics({'loss': train_loss}, epoch=epoch+1, prefix="Train", logger=logger)
        log_metrics(val_metrics, epoch=epoch+1, prefix="Validation", logger=logger)
        log_metrics({'loss': val_loss}, epoch=epoch+1, prefix="Validation", logger=logger)
        
        # Update metrics tracker
        metrics_tracker.update(
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics
        )
        
        # Save checkpoint if best model
        current_f1 = val_metrics['f1_macro']
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': config.to_dict()
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path} (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save metrics history
    metrics_path = os.path.join(config.output_dir, 'training_metrics.json')
    metrics_tracker.save(metrics_path)
    
    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_metrics, test_preds, test_labels = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        logger=logger
    )
    
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)
    log_metrics(test_metrics, prefix="Test", logger=logger)
    log_metrics({'loss': test_loss}, prefix="Test", logger=logger)
    logger.info("="*60)
    
    logger.info("\nTraining completed!")


if __name__ == '__main__':
    main()

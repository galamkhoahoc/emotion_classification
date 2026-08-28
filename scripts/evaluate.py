"""
Evaluation script for ViEmoText emotion classification model.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from src.data.dataset import load_uit_vsmec_dataset, create_dataloaders
from src.models.model_factory import create_model_from_checkpoint, create_loss_function
from src.losses.weighted_cross_entropy import compute_class_weights
from src.utils.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    print_classification_report,
    get_predictions_from_logits
)
from src.utils.logger import setup_logger, log_metrics

from tqdm import tqdm


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    logger
) -> tuple:
    """
    Evaluate the model and return predictions.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        logger: Logger instance
    
    Returns:
        Tuple of (predictions, labels, all_logits)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
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
                attention_mask=attention_mask
            )
            
            logits = outputs['logits']
            predictions = get_predictions_from_logits(logits)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_logits)
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate ViEmoText model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--plot_cm', action='store_true',
                        help='Plot and save confusion matrix')
    parser.add_argument('--model_type', type=str, default=None, choices=['phobert', 'bamibert'],
                        help='Model type to use (phobert or bamibert)')
    args = parser.parse_args()
    
    # Load configuration
    config_kwargs = {}
    if args.model_type:
        config_kwargs['model_type'] = args.model_type
    config = Config(**config_kwargs)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(args.output_dir, 'evaluation.log')
    logger = setup_logger(log_file=log_file)
    
    logger.info("="*60)
    logger.info("ViEmoText Evaluation")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model from checkpoint via factory
    logger.info(f"Creating model via factory: {config.model_type} ({config.model_name})")
    model, tokenizer = create_model_from_checkpoint(args.checkpoint, config)
    
    model = model.to(device)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_data, val_data, test_data = load_uit_vsmec_dataset(
        dataset_name=config.dataset_name,
        cache_dir=config.cache_dir
    )
    
    # Select split
    if args.split == 'train':
        eval_data = train_data
    elif args.split == 'val':
        eval_data = val_data
    else:
        eval_data = test_data
    
    # Create dataloader
    logger.info("Creating dataloader...")
    from src.data.dataset import EmotionDataset
    from torch.utils.data import DataLoader
    
    eval_dataset = EmotionDataset(
        texts=eval_data['texts'],
        labels=eval_data['labels'],
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    logger.info(f"\nEvaluating on {args.split} set...")
    predictions, labels, logits = evaluate_model(
        model=model,
        data_loader=eval_loader,
        device=device,
        logger=logger
    )
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics = compute_metrics(predictions, labels, config.emotion_labels)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    log_metrics(metrics, prefix=args.split.capitalize(), logger=logger)
    logger.info("="*60)
    
    # Print detailed classification report
    print_classification_report(predictions, labels, config.emotion_labels)
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(predictions, labels, normalize=False)
    cm_normalized = compute_confusion_matrix(predictions, labels, normalize=True)
    
    logger.info("\nConfusion Matrix (counts):")
    logger.info("\n" + str(cm))
    
    logger.info("\nConfusion Matrix (normalized):")
    logger.info("\n" + str(cm_normalized))
    
    # Plot confusion matrix if requested
    if args.plot_cm:
        logger.info("\nPlotting confusion matrix...")
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png')
        plot_confusion_matrix(
            cm_normalized,
            config.emotion_labels,
            title=f'Confusion Matrix - {args.split.capitalize()} Set',
            save_path=cm_path
        )
    
    # Save predictions if requested
    if args.save_predictions:
        logger.info("\nSaving predictions...")
        predictions_path = os.path.join(args.output_dir, f'predictions_{args.split}.npz')
        np.savez(
            predictions_path,
            predictions=predictions,
            labels=labels,
            logits=logits,
            texts=eval_data['texts']
        )
        logger.info(f"Predictions saved to {predictions_path}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, f'metrics_{args.split}.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("="*60 + "\n")
    
    logger.info(f"\nMetrics saved to {metrics_path}")
    logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()

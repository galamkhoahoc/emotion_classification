"""
Evaluation script for ViEmoText emotion classification model.
"""

import os
import sys
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from src.data.dataset import load_dataset_by_config
from src.models.model_factory import create_model_from_checkpoint
from src.utils.metrics import (
    compute_metrics,
    compute_per_label_metrics,
    print_metrics_report,
    compute_confusion_matrix,
    plot_confusion_matrix,
    get_predictions_from_logits
)
from src.utils.predictor import ThresholdOptimizer
from src.utils.logger import setup_logger, log_metrics
from tqdm import tqdm


def plot_per_label_f1(metrics_df: pd.DataFrame, save_path: str):
    """Plot F1 scores for each label as a bar chart."""
    df_sorted = metrics_df.sort_values(by='f1', ascending=True)
    
    plt.figure(figsize=(10, 12))
    colors = ['#d9534f' if f1 < 0.5 else '#5bc0de' if f1 < 0.7 else '#5cb85c' for f1 in df_sorted['f1']]
    plt.barh(df_sorted.index, df_sorted['f1'], color=colors)
    plt.title('Per-Label F1 Scores')
    plt.xlabel('F1 Score')
    plt.ylabel('Emotion Label')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
    threshold: float,
    logger
) -> tuple:
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            predictions = get_predictions_from_logits(
                logits,
                problem_type=config.problem_type,
                threshold=threshold
            )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_logits)
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViEmoText model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--plot_cm', action='store_true')
    parser.add_argument('--model_type', type=str, default=None, choices=['phobert', 'bamibert'])
    parser.add_argument('--threshold', type=float, default=None, help='Custom threshold for multilabel')
    parser.add_argument('--sweep_threshold', action='store_true', help='Run threshold sweep on validation set')
    args = parser.parse_args()
    
    config_kwargs = {}
    if args.model_type:
        config_kwargs['model_type'] = args.model_type
    config = Config(**config_kwargs)
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'evaluation.log')
    logger = setup_logger(log_file=log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model, tokenizer = create_model_from_checkpoint(args.checkpoint, config)
    model = model.to(device)
    
    train_data, val_data, test_data = load_dataset_by_config(config=config, cache_dir=config.cache_dir)
    
    # Optional Threshold Sweeping on Validation set
    if args.sweep_threshold and config.is_multilabel():
        logger.info("\nRunning threshold sweep on validation set...")
        from src.data.dataset import EmotionDataset
        from torch.utils.data import DataLoader
        
        val_dataset = EmotionDataset(
            texts=val_data['texts'], labels=val_data['labels'], tokenizer=tokenizer, 
            max_length=config.max_length, num_labels=config.num_labels, problem_type=config.problem_type
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        _, val_labels_arr, val_logits_arr = evaluate_model(model, val_loader, device, config, 0.5, logger)
        
        optimizer = ThresholdOptimizer(config)
        sweep_results = optimizer.sweep_thresholds(torch.tensor(val_logits_arr), torch.tensor(val_labels_arr))
        
        best_t = sweep_results["best_threshold"]
        logger.info(f"Best threshold found: {best_t:.2f} (F1 Macro: {sweep_results['best_f1_macro']:.4f})")
        
        curve_path = os.path.join(args.output_dir, 'threshold_curve.png')
        optimizer.plot_threshold_curve(sweep_results, curve_path)
        
        # Override threshold with the best one for evaluation
        eval_threshold = best_t
    else:
        eval_threshold = args.threshold if args.threshold is not None else config.sigmoid_threshold

    if args.split == 'train':
        eval_data = train_data
    elif args.split == 'val':
        eval_data = val_data
    else:
        eval_data = test_data
        
    from src.data.dataset import EmotionDataset
    from torch.utils.data import DataLoader
    eval_dataset = EmotionDataset(
        texts=eval_data['texts'], labels=eval_data['labels'], tokenizer=tokenizer,
        max_length=config.max_length, num_labels=config.num_labels, problem_type=config.problem_type
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"\nEvaluating on {args.split} set with threshold {eval_threshold:.2f}...")
    predictions, labels, logits = evaluate_model(model, eval_loader, device, config, eval_threshold, logger)
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels, config.problem_type, config.emotion_labels)
    print_metrics_report(metrics, config.problem_type)
    
    # Compute per-label metrics for multilabel mode
    if config.is_multilabel():
        per_label_metrics = compute_per_label_metrics(predictions, labels, config.emotion_labels)
        df_per_label = pd.DataFrame.from_dict(per_label_metrics, orient='index')
        df_per_label.index.name = 'Emotion'
        csv_path = os.path.join(args.output_dir, f'per_label_metrics_{args.split}.csv')
        df_per_label.to_csv(csv_path)
        logger.info(f"Saved per-label metrics to {csv_path}")
        
        # Plot F1 bar chart
        chart_path = os.path.join(args.output_dir, f'per_label_f1_{args.split}.png')
        plot_per_label_f1(df_per_label, chart_path)
    
    if not config.is_multilabel():
        print_classification_report(predictions, labels, config.emotion_labels)
        cm_normalized = compute_confusion_matrix(predictions, labels, normalize=True)
        if args.plot_cm:
            cm_path = os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png')
            plot_confusion_matrix(cm_normalized, config.emotion_labels, title=f'Confusion Matrix - {args.split.capitalize()} Set', save_path=cm_path)
    else:
        if args.plot_cm:
            mcm = multilabel_confusion_matrix(labels, predictions)
            # Not plotting all 28, just indicating support. A more comprehensive approach can be added later.
            logger.info("Multilabel confusion matrices computed but individual plotting skipped.")
            
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, f'predictions_{args.split}.npz')
        np.savez(
            predictions_path,
            predictions=predictions,
            labels=labels,
            logits=logits,
            texts=eval_data['texts']
        )
        logger.info(f"Predictions saved to {predictions_path}")
    
    logger.info("\nEvaluation completed!")

if __name__ == '__main__':
    main()

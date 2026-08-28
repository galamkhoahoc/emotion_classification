"""
Compare results from multiple experiments.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


def load_experiment_metrics(experiment_dir: str) -> Dict:
    """Load metrics from an experiment directory."""
    exp_path = Path(experiment_dir)
    
    # Try to load training_metrics.json
    metrics_file = exp_path / 'training_metrics.json'
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Try to load config
    config_file = exp_path / 'config.json'
    config = None
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return {
        'name': exp_path.name,
        'metrics': metrics,
        'config': config
    }


def plot_training_curves(experiments: List[Dict], save_path: str = None):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    for exp in experiments:
        if exp is None:
            continue
        metrics = exp['metrics']
        name = exp['name']
        
        if 'train_loss' in metrics:
            axes[0].plot(metrics['train_loss'], label=f"{name} (train)", alpha=0.7)
        if 'val_loss' in metrics:
            axes[0].plot(metrics['val_loss'], label=f"{name} (val)", alpha=0.7, linestyle='--')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score curves
    for exp in experiments:
        if exp is None:
            continue
        metrics = exp['metrics']
        name = exp['name']
        
        if 'val_metrics' in metrics:
            f1_scores = [m.get('f1_macro', 0) for m in metrics['val_metrics']]
            axes[1].plot(f1_scores, label=name, marker='o', alpha=0.7)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Macro Score')
    axes[1].set_title('Validation F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    """Create a comparison table of final metrics."""
    data = []
    
    for exp in experiments:
        if exp is None:
            continue
        
        name = exp['name']
        metrics = exp['metrics']
        config = exp.get('config', {})
        
        # Get final metrics
        if 'val_metrics' in metrics and len(metrics['val_metrics']) > 0:
            final_metrics = metrics['val_metrics'][-1]
        else:
            final_metrics = {}
        
        row = {
            'Experiment': name,
            'Accuracy': final_metrics.get('accuracy', 'N/A'),
            'F1 Macro': final_metrics.get('f1_macro', 'N/A'),
            'F1 Weighted': final_metrics.get('f1_weighted', 'N/A'),
            'Precision': final_metrics.get('precision_macro', 'N/A'),
            'Recall': final_metrics.get('recall_macro', 'N/A'),
            'Batch Size': config.get('batch_size', 'N/A'),
            'Learning Rate': config.get('learning_rate', 'N/A'),
            'Epochs': len(metrics.get('val_loss', [])),
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def print_best_experiment(df: pd.DataFrame, metric: str = 'F1 Macro'):
    """Print the best experiment based on a metric."""
    if metric not in df.columns:
        print(f"Metric {metric} not found")
        return
    
    # Filter out N/A values
    valid_df = df[df[metric] != 'N/A'].copy()
    if len(valid_df) == 0:
        print("No valid experiments found")
        return
    
    best_idx = valid_df[metric].astype(float).idxmax()
    best_exp = valid_df.loc[best_idx]
    
    print(f"\n{'='*60}")
    print(f"Best Experiment (by {metric})")
    print(f"{'='*60}")
    print(f"Name: {best_exp['Experiment']}")
    print(f"{metric}: {best_exp[metric]:.4f}")
    print(f"Accuracy: {best_exp['Accuracy']:.4f}")
    print(f"F1 Weighted: {best_exp['F1 Weighted']:.4f}")
    print(f"Batch Size: {best_exp['Batch Size']}")
    print(f"Learning Rate: {best_exp['Learning Rate']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiments")
    parser.add_argument('experiments', nargs='+', 
                        help='Paths to experiment directories')
    parser.add_argument('--output', type=str, default='comparison',
                        help='Output directory for comparison results')
    parser.add_argument('--metric', type=str, default='F1 Macro',
                        help='Metric to use for finding best experiment')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load all experiments
    print("Loading experiments...")
    experiments = []
    for exp_path in args.experiments:
        print(f"  Loading {exp_path}...")
        exp_data = load_experiment_metrics(exp_path)
        experiments.append(exp_data)
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(experiments)
    
    # Print table
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save table
    table_path = output_dir / 'comparison_table.csv'
    df.to_csv(table_path, index=False)
    print(f"\nTable saved to {table_path}")
    
    # Print best experiment
    print_best_experiment(df, metric=args.metric)
    
    # Plot curves
    print("\nGenerating plots...")
    plot_path = output_dir / 'training_curves.png'
    plot_training_curves(experiments, save_path=plot_path)
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()

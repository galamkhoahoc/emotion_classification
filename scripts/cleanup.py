"""
Cleanup script for managing logs, checkpoints, and outputs.
"""

import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta


def get_file_age_days(filepath: str) -> int:
    """Get file age in days."""
    file_time = os.path.getmtime(filepath)
    file_date = datetime.fromtimestamp(file_time)
    age = datetime.now() - file_date
    return age.days


def cleanup_old_files(directory: str, pattern: str, days: int, dry_run: bool = True):
    """
    Remove files older than specified days.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.log", "*.pt")
        days: Files older than this many days will be removed
        dry_run: If True, only show what would be deleted
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return
    
    files_to_delete = []
    total_size = 0
    
    for filepath in directory.rglob(pattern):
        if filepath.is_file():
            age_days = get_file_age_days(str(filepath))
            if age_days > days:
                file_size = filepath.stat().st_size
                files_to_delete.append((filepath, age_days, file_size))
                total_size += file_size
    
    if not files_to_delete:
        print(f"No files older than {days} days found in {directory}")
        return
    
    print(f"\nFound {len(files_to_delete)} files older than {days} days:")
    print(f"Total size: {total_size / (1024**3):.2f} GB\n")
    
    for filepath, age, size in files_to_delete:
        print(f"  {filepath} (age: {age} days, size: {size / (1024**2):.2f} MB)")
    
    if dry_run:
        print("\n[DRY RUN] No files were deleted. Use --execute to actually delete.")
    else:
        confirm = input("\nAre you sure you want to delete these files? (yes/no): ")
        if confirm.lower() == 'yes':
            for filepath, _, _ in files_to_delete:
                filepath.unlink()
                print(f"Deleted: {filepath}")
            print(f"\nDeleted {len(files_to_delete)} files, freed {total_size / (1024**3):.2f} GB")
        else:
            print("Cancelled.")


def keep_best_checkpoints_only(checkpoint_dir: str, dry_run: bool = True):
    """
    Remove all checkpoints except best_model.pt in each experiment folder.
    
    Args:
        checkpoint_dir: Root checkpoint directory
        dry_run: If True, only show what would be deleted
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    files_to_delete = []
    total_size = 0
    
    for checkpoint_file in checkpoint_dir.rglob("*.pt"):
        if checkpoint_file.name not in ['best_model.pt', 'final_model.pt']:
            file_size = checkpoint_file.stat().st_size
            files_to_delete.append((checkpoint_file, file_size))
            total_size += file_size
    
    if not files_to_delete:
        print(f"No checkpoint files to clean in {checkpoint_dir}")
        return
    
    print(f"\nFound {len(files_to_delete)} checkpoint files to remove:")
    print(f"Total size: {total_size / (1024**3):.2f} GB\n")
    
    for filepath, size in files_to_delete:
        print(f"  {filepath} (size: {size / (1024**2):.2f} MB)")
    
    if dry_run:
        print("\n[DRY RUN] No files were deleted. Use --execute to actually delete.")
    else:
        confirm = input("\nKeep only best_model.pt and final_model.pt? (yes/no): ")
        if confirm.lower() == 'yes':
            for filepath, _ in files_to_delete:
                filepath.unlink()
                print(f"Deleted: {filepath}")
            print(f"\nDeleted {len(files_to_delete)} files, freed {total_size / (1024**3):.2f} GB")
        else:
            print("Cancelled.")


def list_disk_usage(base_dir: str = "."):
    """
    Show disk usage for each major directory.
    
    Args:
        base_dir: Base directory to analyze
    """
    base_path = Path(base_dir)
    
    directories = [
        'outputs',
        'checkpoints',
        'logs',
        'data',
        'evaluation_results'
    ]
    
    print("\nDisk Usage Summary:")
    print("=" * 60)
    
    total_size = 0
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            continue
        
        dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        total_size += dir_size
        
        file_count = sum(1 for _ in dir_path.rglob('*') if _.is_file())
        
        print(f"{dir_name:20s}: {dir_size / (1024**3):8.2f} GB ({file_count:5d} files)")
    
    print("=" * 60)
    print(f"{'TOTAL':20s}: {total_size / (1024**3):8.2f} GB")
    print()


def archive_experiment(experiment_path: str, archive_dir: str = "archives"):
    """
    Archive an experiment (move to archives directory).
    
    Args:
        experiment_path: Path to experiment directory
        archive_dir: Archive destination directory
    """
    experiment_path = Path(experiment_path)
    archive_dir = Path(archive_dir)
    
    if not experiment_path.exists():
        print(f"Experiment {experiment_path} does not exist")
        return
    
    archive_dir.mkdir(exist_ok=True)
    
    # Create archive with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{experiment_path.name}_{timestamp}"
    archive_path = archive_dir / archive_name
    
    print(f"Archiving {experiment_path} to {archive_path}...")
    shutil.move(str(experiment_path), str(archive_path))
    print(f"Archived successfully!")


def main():
    parser = argparse.ArgumentParser(description="Cleanup logs, checkpoints, and outputs")
    parser.add_argument('--action', type=str, required=True,
                        choices=['cleanup-logs', 'cleanup-checkpoints', 'keep-best-only', 
                                'disk-usage', 'archive'],
                        help='Action to perform')
    parser.add_argument('--days', type=int, default=30,
                        help='Remove files older than this many days')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory to operate on')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the deletion (default is dry run)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment path to archive')
    
    args = parser.parse_args()
    
    if args.action == 'cleanup-logs':
        cleanup_old_files(
            directory=args.dir or 'logs',
            pattern='*.log',
            days=args.days,
            dry_run=not args.execute
        )
    
    elif args.action == 'cleanup-checkpoints':
        cleanup_old_files(
            directory=args.dir or 'checkpoints',
            pattern='*.pt',
            days=args.days,
            dry_run=not args.execute
        )
    
    elif args.action == 'keep-best-only':
        keep_best_checkpoints_only(
            checkpoint_dir=args.dir or 'checkpoints',
            dry_run=not args.execute
        )
    
    elif args.action == 'disk-usage':
        list_disk_usage(base_dir=args.dir or '.')
    
    elif args.action == 'archive':
        if not args.experiment:
            print("Error: --experiment is required for archive action")
            return
        archive_experiment(args.experiment)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

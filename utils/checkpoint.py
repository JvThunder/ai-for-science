"""
Checkpoint utilities for saving and resuming training.

Provides functions for:
- Saving model checkpoints with training state
- Loading checkpoints to resume training
- Saving final trained models
- Managing checkpoint directories

Checkpoint Format:
    {
        'epoch': int,
        'model_state_dict': OrderedDict,
        'optimizer_state_dict': dict,
        'scheduler_state_dict': dict (optional),
        'best_val_loss': float,
        'config': dict,
        'history': dict (optional)
    }
"""

import torch
import os
from typing import Dict, Optional
from pathlib import Path
import json


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    metrics: Dict,
    config: Dict,
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None,
    save_best: bool = False,
    history: Optional[Dict] = None
) -> str:
    """
    Save model checkpoint with full training state.

    Args:
        epoch: Current epoch number
        model: The model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        metrics: Dictionary with current metrics (train_loss, val_loss, etc.)
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoints
        filename: Custom filename (default: checkpoint_epoch_{epoch}.pt)
        save_best: If True, also save as 'best_model.pt'
        history: Training history dict (optional)

    Returns:
        checkpoint_path: Path to saved checkpoint file
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Default filename
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pt"

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': metrics.get('best_val_loss', float('inf')),
        'config': config,
        'metrics': metrics
    }

    # Add scheduler if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add history if provided
    if history is not None:
        checkpoint['history'] = history

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")

    # Save best model if requested
    if save_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = None
) -> Dict:
    """
    Load model checkpoint and optionally restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        device: Device to load tensors to (default: CPU)

    Returns:
        checkpoint_info: Dictionary with epoch, metrics, config, history
    """
    if device is None:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Extract info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'config': checkpoint.get('config', {}),
        'metrics': checkpoint.get('metrics', {}),
        'history': checkpoint.get('history', None)
    }

    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Best val loss: {checkpoint_info['best_val_loss']:.6f}")

    return checkpoint_info


def save_final_model(
    model: torch.nn.Module,
    config: Dict,
    metrics: Dict,
    save_dir: str = "models",
    filename: str = "final_model.pt"
) -> str:
    """
    Save final trained model (model weights + config only).

    This saves a lighter checkpoint for inference, without optimizer/scheduler state.

    Args:
        model: Trained model
        config: Configuration dict
        metrics: Final metrics dict
        save_dir: Directory to save model
        filename: Model filename

    Returns:
        model_path: Path to saved model file
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, filename)

    # Prepare lightweight checkpoint
    model_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': metrics
    }

    torch.save(model_checkpoint, model_path)

    print(f"Final model saved: {model_path}")

    return model_path


def load_model_for_inference(
    model_path: str,
    model: torch.nn.Module,
    device: torch.device = None
) -> Dict:
    """
    Load model for inference (no optimizer/scheduler).

    Args:
        model_path: Path to model file
        model: Model instance to load weights into
        device: Device to load to

    Returns:
        model_info: Dictionary with config and metrics
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(model_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract info
    model_info = {
        'config': checkpoint.get('config', {}),
        'metrics': checkpoint.get('final_metrics', {})
    }

    print(f"Model loaded for inference from: {model_path}")

    return model_info


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True
) -> None:
    """
    Remove old checkpoints to save disk space.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
        keep_best: Always keep best_model.pt
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Get all checkpoint files
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
    ]

    # Sort by epoch number
    checkpoint_files.sort()

    # Determine which to delete
    if len(checkpoint_files) > keep_last_n:
        files_to_delete = checkpoint_files[:-keep_last_n]

        for filename in files_to_delete:
            filepath = os.path.join(checkpoint_dir, filename)
            os.remove(filepath)
            print(f"Removed old checkpoint: {filename}")

        print(f"Kept {keep_last_n} most recent checkpoints")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        checkpoint_path: Path to latest checkpoint, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Get all checkpoint files
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
    ]

    if not checkpoint_files:
        return None

    # Sort by epoch number and get latest
    checkpoint_files.sort()
    latest_file = checkpoint_files[-1]

    return os.path.join(checkpoint_dir, latest_file)

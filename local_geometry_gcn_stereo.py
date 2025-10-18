#!/usr/bin/env python3
"""
Stereochemistry-Aware 3D Reconstruction GCN - Main Training Script

This script implements a Graph Convolutional Network for predicting local
molecular geometry in cylindrical coordinates (r, θ, φ), enabling
stereochemically-aware 3D structure reconstruction from learned representations.

Key Features:
- Multi-file streaming: Train on TB-scale datasets without RAM caching
- Cylindrical coordinates: Rotationally-invariant (r, θ, φ) representation
- Stereochemistry-aware: Ordered neighbors with azimuthal angles
- Percentage-based loss: Better scaling across distance ranges
- File-based epochs: Each CSV file = 1 epoch, cycle through all files

Usage:
    # Train with default config
    python local_geometry_gcn_stereo.py --config config.yaml

    # Resume from checkpoint
    python local_geometry_gcn_stereo.py --config config.yaml --resume checkpoints/best_model.pt

    # Override config parameters
    python local_geometry_gcn_stereo.py --config config.yaml --batch-size 64 --lr 0.001

Author: Claude Code
Date: 2025-10-13
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import utilities
from utils.config import load_config, validate_config, print_config, merge_configs
from utils.model import LocalGeometryGCN, count_parameters, initialize_weights
from utils.dataset import MultiFileStreamingDataset, analyze_element_frequencies
from utils.training import train_multi_file
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.metrics import MetricsLogger


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def setup_device(config):
    """
    Setup training device (CPU/CUDA).

    Args:
        config: Configuration dict

    Returns:
        device: torch.device
    """
    device_config = config.get('hardware', {}).get('device', 'auto')

    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def create_datasets(config):
    """
    Create training and validation datasets.

    Args:
        config: Configuration dict

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        element_to_idx: Element vocabulary mapping
    """
    print("\n" + "="*60)
    print("Creating Datasets")
    print("="*60)

    # Analyze element frequencies across all training files
    print("\nAnalyzing element frequencies...")
    element_to_idx = analyze_element_frequencies(
        config['data']['train_files'],
        min_count=config['data'].get('min_element_count', 10)
    )

    vocab_size = len(element_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Update config with vocab size
    config['model']['vocab_size'] = vocab_size

    # Get validation split ratio (default to 0.1 if not specified)
    val_split_ratio = config['data'].get('val_split_ratio', 0.1)

    # Create training dataset (90% of each file)
    print("\nCreating training dataset...")
    train_dataset = MultiFileStreamingDataset(
        file_paths=config['data']['train_files'],
        element_to_idx=element_to_idx,
        max_bonds=config['data']['max_bonds'],
        chunk_size=config['data'].get('chunk_size', 10000),
        cache_ram_mb=config['data'].get('cache_ram_mb', 4096),
        training=True,  # Enable rotation augmentation
        val_split_ratio=val_split_ratio,
        split='train'
    )

    print(f"Training dataset created:")
    print(f"  Files: {len(train_dataset.file_paths)}")
    print(f"  Current file: {train_dataset.file_paths[train_dataset.current_file_idx]}")
    print(f"  Bonds in current file: {len(train_dataset)}")
    print(f"  Split: {int((1-val_split_ratio)*100)}% train / {int(val_split_ratio*100)}% val per file")

    # Create validation dataset (10% of each file)
    print("\nCreating validation dataset...")
    val_dataset = MultiFileStreamingDataset(
        file_paths=config['data']['train_files'],
        element_to_idx=element_to_idx,
        max_bonds=config['data']['max_bonds'],
        chunk_size=config['data'].get('chunk_size', 10000),
        cache_ram_mb=config['data'].get('cache_ram_mb', 4096),
        training=False,  # No augmentation for validation
        val_split_ratio=val_split_ratio,
        split='val'
    )
    print(f"Validation dataset created:")
    print(f"  Files: {len(val_dataset.file_paths)}")
    print(f"  Current file: {val_dataset.file_paths[val_dataset.current_file_idx]}")
    print(f"  Bonds in current file: {len(val_dataset)}")

    print("="*60 + "\n")

    return train_dataset, val_dataset, element_to_idx


def create_model(config, device):
    """
    Create and initialize model.

    Args:
        config: Configuration dict
        device: torch.device

    Returns:
        model: LocalGeometryGCN model
    """
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)

    model = LocalGeometryGCN(
        num_node_features=config['model']['vocab_size'],
        hidden_dim=config['model']['hidden_dim'],
        max_bonds=config['model']['max_bonds'],
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        depth=config['model'].get('depth', 1)
    ).to(device)

    # Initialize weights
    initialize_weights(model)

    # Count parameters
    num_params = count_parameters(model)
    print(f"\nModel architecture:")
    print(f"  Hidden dim: {config['model']['hidden_dim']}")
    print(f"  Embedding dim: {config['model']['embedding_dim']}")
    print(f"  Vocabulary size: {config['model']['vocab_size']}")
    print(f"  Max bonds: {config['model']['max_bonds']}")
    print(f"  Depth (GCN layers): {config['model'].get('depth', 1)}")
    print(f"  Total parameters: {num_params:,}")

    print("="*60 + "\n")

    return model


def create_optimizer_and_scheduler(model, config):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize
        config: Configuration dict

    Returns:
        optimizer: Optimizer
        scheduler: Learning rate scheduler (or None)
    """
    print("Creating optimizer and scheduler...")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.001)
    )

    # Scheduler
    scheduler_type = config['training'].get('lr_scheduler', None)
    scheduler = None

    if scheduler_type == 'plateau':
        params = config['training'].get('lr_scheduler_params', {})
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.7),
            patience=params.get('patience', 2),
            verbose=True
        )
        print(f"  Scheduler: ReduceLROnPlateau (factor={params.get('factor', 0.7)}, patience={params.get('patience', 2)})")

    elif scheduler_type == 'step':
        params = config['training'].get('lr_scheduler_params', {})
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 10),
            gamma=params.get('gamma', 0.7)
        )
        print(f"  Scheduler: StepLR (step_size={params.get('step_size', 10)}, gamma={params.get('gamma', 0.7)})")

    elif scheduler_type == 'cosine':
        params = config['training'].get('lr_scheduler_params', {})
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', config['training']['num_epochs']),
            eta_min=params.get('eta_min', 1e-6)
        )
        print(f"  Scheduler: CosineAnnealingLR")

    else:
        print("  No scheduler")

    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training'].get('weight_decay', 0.001)}")

    return optimizer, scheduler


def init_wandb(config, args):
    """
    Initialize Weights & Biases logging.

    Args:
        config: Configuration dict
        args: Command-line arguments

    Returns:
        bool: True if wandb initialized successfully
    """
    use_wandb = config['logging'].get('use_wandb', False)

    if not use_wandb:
        return False

    if not WANDB_AVAILABLE:
        print("Warning: wandb requested in config but not installed.")
        print("Install with: pip install wandb")
        print("Continuing without wandb logging...")
        return False

    # Generate run name if not specified
    run_name = config['logging'].get('wandb_run_name')
    if not run_name:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_name = f"run-{timestamp}"

    # Initialize wandb
    wandb.init(
        project=config['logging'].get('wandb_project', 'stereochem-gcn'),
        entity=config['logging'].get('wandb_entity'),
        name=run_name,
        tags=config['logging'].get('wandb_tags', []),
        notes=config['logging'].get('wandb_notes'),
        config={
            'batch_size': config['data']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_epochs'],
            'hidden_dim': config['model']['hidden_dim'],
            'embedding_dim': config['model']['embedding_dim'],
            'max_bonds': config['model']['max_bonds'],
            'depth': config['model'].get('depth', 1),
            'val_split_ratio': config['data'].get('val_split_ratio', 0.0),
            'weight_decay': config['training'].get('weight_decay', 0.001),
            'grad_clip': config['training'].get('grad_clip', 1.0),
            'loss_weights': {
                'distance': config['loss']['distance_weight'],
                'geometry': config['loss']['geometry_weight'],
                'r': config['loss']['r_weight'],
                'theta': config['loss']['theta_weight'],
                'phi': config['loss']['phi_weight']
            }
        }
    )

    print(f"\nWandB initialized: {wandb.run.name}")
    print(f"View at: {wandb.run.url}\n")

    return True


def main(args):
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("Stereochemistry-Aware 3D Reconstruction GCN")
    print("="*60)
    print(f"Config file: {args.config}")
    print("="*60 + "\n")

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.epochs:
        config['training']['num_epochs'] = args.epochs

    # Validate configuration
    validate_config(config)

    # Print configuration
    print("\nConfiguration:")
    print_config(config)
    print()

    # Set random seeds
    set_random_seeds(config.get('random_seed', 42))

    # Setup device
    device = setup_device(config)

    # Create datasets
    train_dataset, val_dataset, element_to_idx = create_datasets(config)

    # Create model
    model = create_model(config, device)

    # Initialize wandb if enabled
    use_wandb = init_wandb(config, args)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Create metrics logger
    metrics_logger = MetricsLogger(
        log_dir=config['logging']['log_dir'],
        use_wandb=use_wandb
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint_info['epoch']
        print(f"Resuming from epoch {start_epoch}")

    # Checkpoint callback
    def checkpoint_callback(epoch, model, optimizer, scheduler, metrics):
        """Save checkpoint after each epoch."""
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            config=config,
            checkpoint_dir=config['checkpointing']['checkpoint_dir'],
            save_best=(metrics.get('val_loss') is not None and
                      metrics['val_loss'] <= metrics.get('best_val_loss', float('inf')))
        )

    # Training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    history = train_multi_file(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_callback=checkpoint_callback,
        metrics_logger=metrics_logger
    )

    # Save metrics and plots
    print("\nSaving training metrics...")
    metrics_logger.save()
    metrics_logger.plot(save=True, show=False)
    metrics_logger.summary()

    # Save final model
    from utils.checkpoint import save_final_model
    save_final_model(
        model=model,
        config=config,
        metrics={'history': history},
        save_dir="models",
        filename="final_model.pt"
    )

    # Finish wandb logging
    metrics_logger.finish()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train stereochemistry-aware 3D reconstruction GCN",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (default: None)'
    )

    # Override config parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate from config'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()

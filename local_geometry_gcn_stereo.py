#!/usr/bin/env python3
"""
Stereochemistry-Aware 3D Reconstruction GCN - Main Training Script

This script implements a Graph Convolutional Network for predicting local
molecular geometry in Spherical coordinates (r, θ, φ), enabling
stereochemically-aware 3D structure reconstruction from learned representations.

Key Features:
- Multi-file streaming: Train on TB-scale datasets without RAM caching
- Spherical coordinates: Rotationally-invariant (r, θ, φ) representation
- Stereochemistry-aware: Ordered neighbors with azimuthal angles
- Percentage-based loss: Better scaling across distance ranges
- Training scheme: 1 epoch = all files, 1 iteration = 1 file

Usage:
    # Train with config.yaml (all settings configured in the file)
    python local_geometry_gcn_stereo.py

    # To resume from a specific checkpoint, set 'resume_checkpoint' in config.yaml:
    # training:
    #   resume_checkpoint: "checkpoints/best_model.pt"

    # To automatically resume from latest checkpoint, set 'resume_from_latest' in config.yaml:
    # training:
    #   resume_from_latest: true

Author: Claude Code
Date: 2025-10-13
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
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
from utils.dataset import MultiFileStreamingDataset, get_element_vocab
from utils.training import train_multi_file
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.metrics import MetricsLogger
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="UnsupportedFieldAttributeWarning")

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

    # Get all CSV files from train_folder
    train_folder = Path(config['data']['train_folder'])
    train_files = sorted(train_folder.glob('*.csv'))

    if not train_files:
        raise ValueError(f"No CSV files found in {train_folder}")

    print(f"\nFound {len(train_files)} CSV files in {train_folder}")
    for f in train_files:
        print(f"  - {f.name}")

    # Load or create element vocabulary from txt file
    print("\nLoading element vocabulary...")
    elements_txt_path = config['data'].get('elements_txt', 'elements.txt')
    element_to_idx = get_element_vocab(
        txt_path=elements_txt_path,
        file_paths=[str(f) for f in train_files],
        max_molecules=config['data'].get('max_molecules_for_vocab', 100000)
    )

    vocab_size = len(element_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Update config with vocab size
    config['model']['vocab_size'] = vocab_size

    # Get validation split ratio (default to 0.1 if not specified)
    val_split_ratio = config['data'].get('val_split_ratio', 0.1)
    use_fixed_validation = config['data'].get('use_fixed_validation', False)

    # Create training dataset (90% of each file)
    print("\nCreating training dataset...")
    train_dataset = MultiFileStreamingDataset(
        file_paths=[str(f) for f in train_files],
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

    # Create validation dataset
    print("\nCreating validation dataset...")
    if use_fixed_validation:
        # Use only the first file for validation (stays consistent across epochs)
        val_files = [str(train_files[0])]
        print(f"  Mode: FIXED validation (same molecules every epoch)")
        print(f"  Using file: {train_files[0].name}")
    else:
        # Use all files for validation (cycles through files like training)
        val_files = [str(f) for f in train_files]
        print(f"  Mode: CYCLING validation (different molecules each epoch)")

    val_dataset = MultiFileStreamingDataset(
        file_paths=val_files,
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
        depth=config['model'].get('depth', 1),
        use_bond_types=config['model'].get('use_bond_types', True),
        num_bond_types=config['model'].get('num_bond_types', 4),
        bond_embedding_dim=config['model'].get('bond_embedding_dim', 8)
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
    print(f"  Depth (GAT layers): {config['model'].get('depth', 1)}")
    print(f"  Use bond types: {config['model'].get('use_bond_types', True)}")
    if config['model'].get('use_bond_types', True):
        print(f"  Bond types: {config['model'].get('num_bond_types', 4)} (single, aromatic, double, triple)")
        print(f"  Bond embedding dim: {config['model'].get('bond_embedding_dim', 8)}")
    print(f"  Total parameters: {num_params:,}")

    print("="*60 + "\n")

    return model


def export_validation_ids_to_file(config, train_dataset, val_dataset):
    """
    Export validation molecule IDs to JSON file.

    Args:
        config: Configuration dict
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    print("\n" + "="*60)
    print("Exporting Validation IDs")
    print("="*60)

    output_path = Path(config['logging'].get('validation_ids_file', 'logs/validation_ids.json'))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get validation split info for all files
    all_validation_info = []

    for file_idx, file_path in enumerate(train_dataset.file_paths):
        # Get validation indices from the dataset's split logic
        # We need to replicate the split logic to get the indices
        import pandas as pd
        import random as rand

        # Read file to count valid molecules
        df = pd.read_csv(file_path)
        valid_indices = []

        from utils.dataset import parse_xyz_string, parse_bonds_from_dataset

        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                if 'geometry_xyz' not in row or pd.isna(row['geometry_xyz']):
                    continue
                if 'bonds_geom_valence' not in row or pd.isna(row['bonds_geom_valence']):
                    continue

                coordinates, elements = parse_xyz_string(row['geometry_xyz'])
                bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])

                if coordinates and len(coordinates) >= 2 and bonds_data:
                    valid_indices.append(idx)
            except Exception:
                continue

        # Apply same split logic as dataset
        val_split_ratio = config['data'].get('val_split_ratio', 0.1)
        split_seed = config.get('random_seed', 42)

        rng = rand.Random(split_seed + hash(file_path))
        shuffled_indices = list(valid_indices)
        rng.shuffle(shuffled_indices)

        n_total = len(shuffled_indices)
        n_val = int(n_total * val_split_ratio)
        n_train = n_total - n_val

        val_indices = sorted(shuffled_indices[n_train:])
        train_indices = sorted(shuffled_indices[:n_train])

        all_validation_info.append({
            'file_idx': file_idx,
            'file_path': str(file_path),
            'file_name': Path(file_path).name,
            'epoch_in_first_cycle': file_idx + 1,
            'total_molecules': n_total,
            'train_molecules': n_train,
            'val_molecules': n_val,
            'val_split_ratio': val_split_ratio,
            'split_seed': split_seed,
            'train_indices': train_indices,
            'val_indices': val_indices
        })

        print(f"  File {file_idx + 1}/{len(train_dataset.file_paths)}: {Path(file_path).name}")
        print(f"    Total: {n_total} | Train: {n_train} | Val: {n_val}")

    # Create summary
    total_train_molecules = sum(info['train_molecules'] for info in all_validation_info)
    total_val_molecules = sum(info['val_molecules'] for info in all_validation_info)

    summary = {
        'config': {
            'train_folder': config['data']['train_folder'],
            'val_split_ratio': config['data'].get('val_split_ratio', 0.1),
            'split_seed': config.get('random_seed', 42)
        },
        'summary': {
            'total_files': len(train_dataset.file_paths),
            'total_train_molecules': total_train_molecules,
            'total_val_molecules': total_val_molecules,
            'validation_behavior': (
                "IMPORTANT: Validation set CHANGES across epochs because each file = 1 epoch. "
                "The validation split is deterministic for each file (same molecules are always "
                "in the validation set for a given file), but different files contain different "
                "molecules. Therefore, epoch 1 uses validation molecules from file 0, epoch 2 "
                "uses validation molecules from file 1, etc."
            )
        },
        'files': all_validation_info
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nValidation IDs exported to: {output_path}")
    print(f"Total train molecules (across all files): {total_train_molecules:,}")
    print(f"Total val molecules (across all files): {total_val_molecules:,}")
    print("="*60 + "\n")


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
            patience=params.get('patience', 2)
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


def init_wandb(config):
    """
    Initialize Weights & Biases logging.

    Args:
        config: Configuration dict

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
            'use_bond_types': config['model'].get('use_bond_types', True),
            'num_bond_types': config['model'].get('num_bond_types', 4),
            'bond_embedding_dim': config['model'].get('bond_embedding_dim', 8),
            'val_split_ratio': config['data'].get('val_split_ratio', 0.0),
            'weight_decay': config['training'].get('weight_decay', 0.001),
            'grad_clip': config['training'].get('grad_clip', 1.0),
            'loss_weights': {
                'distance': config['loss']['distance_weight'],
                'geometry': config['loss']['geometry_weight'],
                'theta': config['loss']['theta_weight'],
                'phi': config['loss']['phi_weight']
            }
        }
    )

    print(f"\nWandB initialized: {wandb.run.name}")
    print(f"View at: {wandb.run.url}\n")

    return True


def main():
    """
    Main training function.
    """
    # Load configuration (from environment variable or default)
    config_file = os.environ.get('CONFIG_FILE', 'config.yaml')
    config = load_config(config_file)

    print("\n" + "="*60)
    print("Stereochemistry-Aware 3D Reconstruction GCN")
    print("="*60)
    print(f"Config file: {config_file}")
    print("="*60 + "\n")

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
    use_wandb = init_wandb(config)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Create metrics logger
    metrics_logger = MetricsLogger(
        log_dir=config['logging']['log_dir'],
        use_wandb=use_wandb
    )

    # Resume from checkpoint if specified in config
    start_iteration = 0
    resume_checkpoint = config.get('training', {}).get('resume_checkpoint', None)
    resume_from_latest = config.get('training', {}).get('resume_from_latest', False)

    # If resume_from_latest is True, find the latest checkpoint
    if resume_from_latest and resume_checkpoint is None:
        checkpoint_dir = config['checkpointing']['checkpoint_dir']
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume_checkpoint = latest_checkpoint
            print(f"\nFound latest checkpoint: {resume_checkpoint}")
        else:
            print(f"\nNo checkpoint found in {checkpoint_dir}, starting from scratch")

    if resume_checkpoint:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint_info = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, device)

        # Extract start_iteration from checkpoint metrics
        if 'metrics' in checkpoint_info and 'global_iteration' in checkpoint_info['metrics']:
            start_iteration = checkpoint_info['metrics']['global_iteration']
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No iteration info found in checkpoint, starting from iteration 0")

    # Checkpoint callback
    def checkpoint_callback(epoch, model, optimizer, scheduler, metrics):
        """Save checkpoint every N iterations and cleanup old checkpoints."""
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
        metrics_logger=metrics_logger,
        start_iteration=start_iteration
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()

"""
Training utilities for multi-file streaming dataset training.

This module provides training and validation functions optimized for TB-scale
datasets that don't fit in RAM. Key features:

- Multi-file training: Each file = 1 epoch, cycle through all files
- Streaming: No data caching, everything read from disk
- File cycling: Dataset switches to next file after each epoch
- Checkpoint after each file: Resume training mid-cycle
- Early stopping: Prevent overfitting with validation monitoring

Functions:
- train_multi_file: Main training loop across multiple CSV files
- train_epoch: Train one epoch on current file
- validate_on_file: Validate on separate validation file
- compute_batch_loss: Compute loss for a single batch
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import time

from .loss import total_loss, compute_angle_errors
import json
from pathlib import Path


def compute_batch_loss(
    model: nn.Module,
    data,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for a single batch.

    Args:
        model: The LocalGeometryGCN model
        data: PyG Data batch
        device: Device to run on
        config: Configuration dict with loss weights

    Returns:
        loss_dict: Dictionary with 'total', 'distance', 'geometry_a', 'geometry_b'
    """
    data = data.to(device)

    # Forward pass
    pred_dist, pred_geom_a, pred_geom_b = model(data)

    # Compute loss
    loss_dict = total_loss(pred_dist, pred_geom_a, pred_geom_b, data, config)

    return loss_dict


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict,
    epoch: int,
    file_idx: int,
    total_files: int
) -> Dict[str, float]:
    """
    Train for one epoch (one file).

    Args:
        model: The LocalGeometryGCN model
        train_loader: DataLoader for current training file
        optimizer: Optimizer
        device: Device to run on
        config: Configuration dict
        epoch: Current epoch number
        file_idx: Current file index
        total_files: Total number of files

    Returns:
        metrics: Dictionary with average losses for this epoch
    """
    model.train()

    total_loss = 0.0
    total_distance_loss = 0.0
    total_geometry_loss = 0.0
    num_batches = 0

    # Check for max batches limit (for testing)
    max_batches = config['training'].get('max_batches_per_epoch', None)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch} [File {file_idx+1}/{total_files}] Training",
        leave=False,
        total=max_batches if max_batches else len(train_loader)
    )

    for batch_idx, data in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        optimizer.zero_grad()

        # Compute loss
        loss_dict = compute_batch_loss(model, data, device, config)

        # Backward pass
        loss_dict['total'].backward()

        # Gradient clipping
        if config['training'].get('grad_clip', None):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['grad_clip']
            )

        optimizer.step()

        # Accumulate losses
        total_loss += loss_dict['total'].item()
        total_distance_loss += loss_dict['distance'].item()
        total_geometry_loss += loss_dict['geometry'].item()
        num_batches += 1

        # Update progress bar
        if num_batches > 0:
            pbar.set_postfix({
                'Loss': f"{total_loss/num_batches:.4f}",
                'Dist': f"{total_distance_loss/num_batches:.4f}",
                'Geom': f"{total_geometry_loss/num_batches:.4f}"
            })

    # Compute averages
    metrics = {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'distance_loss': total_distance_loss / num_batches if num_batches > 0 else 0.0,
        'geometry_loss': total_geometry_loss / num_batches if num_batches > 0 else 0.0,
        'num_batches': num_batches
    }

    return metrics


def validate_on_file(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
    compute_detailed_metrics: bool = False
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: The LocalGeometryGCN model
        val_loader: DataLoader for validation data
        device: Device to run on
        config: Configuration dict
        compute_detailed_metrics: If True, compute angle MAE/RMSE

    Returns:
        metrics: Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_distance_loss = 0.0
    total_geometry_loss = 0.0
    num_batches = 0

    # For detailed metrics
    all_pred_distances = []
    all_target_distances = []
    all_pred_theta_a = []
    all_target_theta_a = []
    all_mask_a = []

    # Check for max validation batches limit (for testing)
    max_val_batches = config['training'].get('max_val_batches', None)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False,
                   total=max_val_batches if max_val_batches else len(val_loader))
        for batch_idx, data in enumerate(pbar):
            if max_val_batches and batch_idx >= max_val_batches:
                break
            # Compute loss
            loss_dict = compute_batch_loss(model, data, device, config)

            # Accumulate losses
            total_loss += loss_dict['total'].item()
            total_distance_loss += loss_dict['distance'].item()
            total_geometry_loss += loss_dict['geometry'].item()
            num_batches += 1

            # Store predictions for detailed metrics
            if compute_detailed_metrics:
                data = data.to(device)
                pred_dist, pred_geom_a, pred_geom_b = model(data)

                # Get batch size and max_bonds for reshaping
                batch_size = data.num_graphs
                max_bonds = config['data']['max_bonds']

                # Reshape target_geometry_a from (batch_size * max_bonds, 3) to (batch_size, max_bonds, 3)
                target_geom_a = data.target_geometry_a.view(batch_size, max_bonds, 3)
                bond_mask_a = data.bond_mask_a.view(batch_size, max_bonds)

                all_pred_distances.append(pred_dist.cpu())
                all_target_distances.append(data.target_distance.cpu())
                all_pred_theta_a.append(pred_geom_a[:, :, 1].cpu())
                all_target_theta_a.append(target_geom_a[:, :, 1].cpu())
                all_mask_a.append(bond_mask_a.cpu())

            # Update progress bar
            if num_batches > 0:
                pbar.set_postfix({
                    'Val Loss': f"{total_loss/num_batches:.4f}"
                })

    # Compute averages
    metrics = {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'distance_loss': total_distance_loss / num_batches if num_batches > 0 else 0.0,
        'geometry_loss': total_geometry_loss / num_batches if num_batches > 0 else 0.0,
        'num_batches': num_batches
    }

    # Compute detailed metrics if requested
    if compute_detailed_metrics and len(all_pred_distances) > 0:
        # Distance metrics
        pred_dist = torch.cat(all_pred_distances)
        target_dist = torch.cat(all_target_distances)

        mae_dist = torch.mean(torch.abs(pred_dist - target_dist)).item()
        rmse_dist = torch.sqrt(torch.mean((pred_dist - target_dist) ** 2)).item()

        metrics['distance_mae'] = mae_dist
        metrics['distance_rmse'] = rmse_dist

        # Angle metrics (θ only, for simplicity)
        pred_theta = torch.cat(all_pred_theta_a)
        target_theta = torch.cat(all_target_theta_a)
        mask = torch.cat(all_mask_a)

        angle_stats = compute_angle_errors(
            pred_theta.view(-1),
            target_theta.view(-1),
            mask.view(-1),
            angle_type='regular'
        )

        metrics['theta_mae_deg'] = angle_stats['mae']
        metrics['theta_rmse_deg'] = angle_stats['rmse']

    return metrics


def _get_validation_info_for_current_file(train_dataset, val_dataset, file_idx, config):
    """
    Get validation info for the currently loaded file (lazy evaluation).

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        file_idx: Current file index
        config: Configuration dict

    Returns:
        Dictionary with validation info for this file
    """
    # Get current file path
    current_file = train_dataset.file_paths[train_dataset.current_file_idx]

    # Get validation indices from dataset's internal metadata
    # The dataset has already computed the split, so we just extract it
    val_split_ratio = config['data'].get('val_split_ratio', 0.1)

    # Read validation molecule indices from the dataset
    # The dataset stores valid_molecule_indices as (chunk_idx, row_in_chunk, bond_count)
    # We need to map these back to original row indices

    import pandas as pd
    import random as rand
    from .dataset import parse_xyz_string, parse_bonds_from_dataset

    # Read file to get original indices
    df = pd.read_csv(current_file)
    valid_indices = []

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
    split_seed = config.get('random_seed', 42)
    rng = rand.Random(split_seed + hash(current_file))
    shuffled_indices = list(valid_indices)
    rng.shuffle(shuffled_indices)

    n_total = len(shuffled_indices)
    n_val = int(n_total * val_split_ratio)
    n_train = n_total - n_val

    val_indices = sorted(shuffled_indices[n_train:])
    train_indices = sorted(shuffled_indices[:n_train])

    return {
        'file_idx': file_idx,
        'file_path': str(current_file),
        'file_name': Path(current_file).name,
        'epoch_in_first_cycle': file_idx + 1,
        'total_molecules': n_total,
        'train_molecules': n_train,
        'val_molecules': n_val,
        'val_split_ratio': val_split_ratio,
        'split_seed': split_seed,
        'train_indices': train_indices,
        'val_indices': val_indices
    }


def _save_validation_ids(validation_info_list, config):
    """
    Save validation IDs to JSON file.

    Args:
        validation_info_list: List of validation info dicts
        config: Configuration dict
    """
    output_path = Path(config['logging'].get('validation_ids_file', 'logs/validation_ids.json'))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_train_molecules = sum(info['train_molecules'] for info in validation_info_list)
    total_val_molecules = sum(info['val_molecules'] for info in validation_info_list)

    use_fixed_validation = config['data'].get('use_fixed_validation', False)

    summary = {
        'config': {
            'train_folder': config['data']['train_folder'],
            'val_split_ratio': config['data'].get('val_split_ratio', 0.1),
            'split_seed': config.get('random_seed', 42),
            'use_fixed_validation': use_fixed_validation
        },
        'summary': {
            'total_files': len(validation_info_list),
            'total_train_molecules': total_train_molecules,
            'total_val_molecules': total_val_molecules,
            'validation_behavior': (
                f"FIXED VALIDATION: Validation uses only file 0 across all epochs (same molecules every epoch)."
                if use_fixed_validation else
                "CYCLING VALIDATION: Validation set CHANGES across epochs because each file = 1 epoch. "
                "The validation split is deterministic for each file (same molecules are always "
                "in the validation set for a given file), but different files contain different "
                "molecules. Therefore, epoch 1 uses validation molecules from file 0, epoch 2 "
                "uses validation molecules from file 1, etc."
            )
        },
        'files': validation_info_list
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Validation IDs Saved")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Total train molecules: {total_train_molecules:,}")
    print(f"Total val molecules: {total_val_molecules:,}")
    if use_fixed_validation:
        print(f"\nMode: FIXED validation (same molecules every epoch)")
        print(f"Validation file: {validation_info_list[0]['file_name']}")
    else:
        print(f"\nMode: CYCLING validation (different molecules each epoch)")
    print(f"{'='*60}\n")


def train_multi_file(
    model: nn.Module,
    train_dataset,
    val_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    config: Dict,
    checkpoint_callback = None,
    metrics_logger = None
) -> Dict[str, List[float]]:
    """
    Main training loop for multi-file streaming dataset.

    Trains on multiple files in sequence, where each file = 1 epoch.
    Automatically cycles through files and checkpoints after each.

    Args:
        model: The LocalGeometryGCN model
        train_dataset: MultiFileStreamingDataset for training
        val_dataset: MultiFileStreamingDataset for validation (single file)
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to run on
        config: Configuration dict with:
            - batch_size: Batch size
            - num_epochs: Number of passes through all files
            - num_workers: DataLoader workers (set to 0 for file switching)
            - grad_clip: Gradient clipping max norm (optional)
            - early_stopping_patience: Epochs without improvement before stopping
            - validate_every: Validate every N epochs (default: 1)
        checkpoint_callback: Function(epoch, model, optimizer, metrics) for saving
        metrics_logger: MetricsLogger for logging (optional)

    Returns:
        history: Dictionary with lists of training/validation metrics

    File Cycling Strategy:
        1. Load file 1 → train epoch 1 → checkpoint → switch to file 2
        2. Load file 2 → train epoch 2 → checkpoint → switch to file 3
        3. ... repeat until all files processed (1 full cycle)
        4. Repeat cycle for num_epochs passes

    Example:
        >>> history = train_multi_file(
        ...     model, train_ds, val_ds, optimizer, scheduler, device, config
        ... )
        >>> # Access history['train_loss'], history['val_loss'], etc.
    """
    print(f"\n{'='*60}")
    print("Starting Multi-File Training")
    print(f"{'='*60}")
    print(f"Total files: {len(train_dataset.file_paths)}")
    print(f"Epochs per cycle: {len(train_dataset.file_paths)}")
    print(f"Total cycles: {config['training']['num_epochs']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Training history
    history = {
        'train_loss': [],
        'train_distance_loss': [],
        'train_geometry_loss': [],
        'val_loss': [],
        'val_distance_loss': [],
        'val_geometry_loss': [],
        'learning_rates': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config['training'].get('early_stopping_patience', 5)
    validate_every = config['training'].get('validate_every', 1)

    # DataLoader parameters
    num_workers = config['data'].get('num_workers', 4)
    prefetch_factor = config['data'].get('prefetch_factor', 2) if num_workers > 0 else None

    # File cycling
    total_files = len(train_dataset.file_paths)
    global_epoch = 0

    # Validation ID tracking (built incrementally during first cycle)
    validation_info_list = []
    export_validation_ids = config['logging'].get('export_validation_ids', False)

    for cycle in range(config['training']['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Cycle {cycle + 1}/{config['training']['num_epochs']}")
        print(f"{'='*60}")

        # Iterate through all files
        for file_idx in range(total_files):
            global_epoch += 1

            # Switch to next file
            if file_idx > 0:  # Skip first file (already loaded)
                print(f"\nSwitching to file {file_idx + 1}/{total_files}...")
                train_dataset.next_file()

                # Conditionally switch validation dataset
                use_fixed_validation = config['data'].get('use_fixed_validation', False)
                if not use_fixed_validation:
                    # Cycle validation dataset (validation molecules change each epoch)
                    val_dataset.next_file()
                else:
                    # Keep validation dataset fixed (same molecules every epoch)
                    print(f"  Validation stays on: {val_dataset.file_paths[val_dataset.current_file_idx]}")

            # Track validation info during first cycle
            if cycle == 0 and export_validation_ids:
                val_info = _get_validation_info_for_current_file(
                    train_dataset,
                    val_dataset,
                    file_idx,
                    config
                )
                validation_info_list.append(val_info)

            # Create dataloaders for current file
            # Use multiprocessing for faster data loading
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True if device.type == 'cuda' else False,
                persistent_workers=True if num_workers > 0 else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True if device.type == 'cuda' else False,
                persistent_workers=True if num_workers > 0 else False
            )

            print(f"\nEpoch {global_epoch} - Training on file: {train_dataset.file_paths[train_dataset.current_file_idx]}")
            print(f"Train dataset size: {len(train_dataset)} bonds")
            print(f"Val dataset size: {len(val_dataset)} bonds")

            # Train one epoch
            start_time = time.time()
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, config,
                global_epoch, file_idx, total_files
            )
            train_time = time.time() - start_time

            # Log training metrics
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_distance_loss'].append(train_metrics['distance_loss'])
            history['train_geometry_loss'].append(train_metrics['geometry_loss'])

            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Print training summary
            print(f"\nEpoch {global_epoch} Training Summary:")
            print(f"  Time: {train_time:.1f}s")
            print(f"  Loss: {train_metrics['total_loss']:.6f}")
            print(f"  Distance Loss: {train_metrics['distance_loss']:.6f}")
            print(f"  Geometry Loss: {train_metrics['geometry_loss']:.6f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Validation
            if global_epoch % validate_every == 0:
                print(f"\nRunning validation...")
                val_start_time = time.time()
                val_metrics = validate_on_file(
                    model, val_loader, device, config,
                    compute_detailed_metrics=(global_epoch % 5 == 0)  # Detailed metrics every 5 epochs
                )
                val_time = time.time() - val_start_time

                history['val_loss'].append(val_metrics['total_loss'])
                history['val_distance_loss'].append(val_metrics['distance_loss'])
                history['val_geometry_loss'].append(val_metrics['geometry_loss'])

                print(f"Validation Summary:")
                print(f"  Time: {val_time:.1f}s")
                print(f"  Loss: {val_metrics['total_loss']:.6f}")
                print(f"  Distance Loss: {val_metrics['distance_loss']:.6f}")
                print(f"  Geometry Loss: {val_metrics['geometry_loss']:.6f}")

                # Detailed metrics if computed
                if 'distance_mae' in val_metrics:
                    print(f"  Distance MAE: {val_metrics['distance_mae']:.4f} Å")
                    print(f"  Distance RMSE: {val_metrics['distance_rmse']:.4f} Å")
                    print(f"  Theta MAE: {val_metrics['theta_mae_deg']:.2f}°")

                # Learning rate scheduling
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics['total_loss'])
                    else:
                        scheduler.step()

                # Early stopping check
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                    print(f"  [BEST] New best validation loss: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

                if patience_counter >= early_stopping_patience:
                    print(f"\n{'='*60}")
                    print(f"Early stopping triggered at epoch {global_epoch}")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                    print(f"{'='*60}")
                    return history

            # Checkpoint after each file
            if checkpoint_callback is not None:
                checkpoint_callback(
                    epoch=global_epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={
                        'train_loss': train_metrics['total_loss'],
                        'val_loss': val_metrics['total_loss'] if 'val_loss' in history and len(history['val_loss']) > 0 else None,
                        'best_val_loss': best_val_loss
                    }
                )

            # Log to metrics logger
            if metrics_logger is not None:
                metrics_logger.log(
                    epoch=global_epoch,
                    train_loss=train_metrics['total_loss'],
                    val_loss=val_metrics['total_loss'] if global_epoch % validate_every == 0 else None,
                    metrics={
                        'train_distance_loss': train_metrics['distance_loss'],
                        'train_geometry_loss': train_metrics['geometry_loss'],
                        'val_distance_loss': val_metrics.get('distance_loss'),
                        'val_geometry_loss': val_metrics.get('geometry_loss'),
                    },
                    lr=current_lr
                )

        # End of file loop - save validation IDs after first cycle
        if cycle == 0 and export_validation_ids and len(validation_info_list) > 0:
            _save_validation_ids(validation_info_list, config)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*60}\n")

    return history

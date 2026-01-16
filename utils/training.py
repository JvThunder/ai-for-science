"""
Training utilities for multi-file streaming dataset training.

This module provides training and validation functions optimized for TB-scale
datasets that don't fit in RAM. Key features:

- Multi-file training: 1 epoch = all files, 1 iteration = 1 file
- Streaming: No data caching, everything read from disk
- File iteration: Dataset switches to next file after each iteration
- Checkpoint after each epoch: Resume training easily
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
        desc=f"Epoch {epoch+1} [File {file_idx+1}/{total_files}] Training",
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
    Validate model on validation set (streaming-safe version).
    Computes metrics incrementally to avoid large tensor concatenations.
    """
    import gc, psutil, os
    
    model.eval()

    total_loss = 0.0
    total_distance_loss = 0.0
    total_geometry_loss = 0.0
    num_batches = 0

    # Running sums for detailed metrics
    mae_sum = 0.0
    rmse_sum = 0.0
    mae_angle_sum = 0.0
    rmse_angle_sum = 0.0
    total_samples = 0

    max_val_batches = config['training'].get('max_val_batches', None)

    with torch.no_grad():
        pbar = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
            total=max_val_batches if max_val_batches else len(val_loader)
        )
        for batch_idx, data in enumerate(pbar):
            if max_val_batches and batch_idx >= max_val_batches:
                break

            loss_dict = compute_batch_loss(model, data, device, config)
            total_loss += loss_dict['total'].item()
            total_distance_loss += loss_dict['distance'].item()
            total_geometry_loss += loss_dict['geometry'].item()
            num_batches += 1

            if compute_detailed_metrics:
                data = data.to(device)
                pred_dist, pred_geom_a, pred_geom_b = model(data)

                batch_size = data.num_graphs
                max_bonds = config['data']['max_bonds']

                target_geom_a = data.target_geometry_a.view(batch_size, max_bonds, 2)
                bond_mask_a = data.bond_mask_a.view(batch_size, max_bonds)

                # Compute batch-wise MAE/RMSE for distances
                diff = pred_dist - data.target_distance
                mae_sum += torch.sum(torch.abs(diff)).item()
                rmse_sum += torch.sum(diff ** 2).item()
                total_samples += diff.numel()

                # Compute angle stats incrementally (θ only)
                pred_theta = pred_geom_a[:, :, 1]
                target_theta = target_geom_a[:, :, 1]
                mask = bond_mask_a

                angle_stats = compute_angle_errors(
                    pred_theta.view(-1),
                    target_theta.view(-1),
                    mask.view(-1),
                    angle_type='regular'
                )
                mae_angle_sum += angle_stats['mae'] * mask.sum().item()
                rmse_angle_sum += (angle_stats['rmse'] ** 2) * mask.sum().item()

            pbar.set_postfix({'Val Loss': f"{total_loss/num_batches:.4f}"})

    metrics = {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'distance_loss': total_distance_loss / num_batches if num_batches > 0 else 0.0,
        'geometry_loss': total_geometry_loss / num_batches if num_batches > 0 else 0.0,
        'num_batches': num_batches
    }

    # Compute final aggregated detailed metrics
    if compute_detailed_metrics and total_samples > 0:
        metrics['distance_mae'] = mae_sum / total_samples
        metrics['distance_rmse'] = (rmse_sum / total_samples) ** 0.5
        if mae_angle_sum > 0:
            metrics['theta_mae_deg'] = mae_angle_sum / total_samples
            metrics['theta_rmse_deg'] = (rmse_angle_sum / total_samples) ** 0.5

    # Free memory after validation
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[Validation] Memory usage after cleanup: {mem_gb:.2f} GB")

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

    # Get k-fold configuration
    kfold = config['data'].get('kfold', 5)
    current_fold = config['data'].get('current_fold', 0)
    split_seed = config.get('random_seed', 42)

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

            coordinates, _ = parse_xyz_string(row['geometry_xyz'])
            bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])

            if coordinates and len(coordinates) >= 2 and bonds_data:
                valid_indices.append(idx)
        except Exception:
            continue

    # Apply k-fold split logic (same as dataset)
    rng = rand.Random(split_seed + hash(current_file))
    shuffled_indices = list(valid_indices)
    rng.shuffle(shuffled_indices)

    n_total = len(shuffled_indices)
    fold_size = n_total // kfold

    # Calculate fold boundaries
    val_start = current_fold * fold_size
    if current_fold == kfold - 1:
        val_end = n_total
    else:
        val_end = val_start + fold_size

    val_indices = sorted(shuffled_indices[val_start:val_end])
    train_indices = sorted(shuffled_indices[:val_start] + shuffled_indices[val_end:])
    n_train = len(train_indices)
    n_val = len(val_indices)

    return {
        'file_idx': file_idx,
        'file_path': str(current_file),
        'file_name': Path(current_file).name,
        'epoch_in_first_cycle': file_idx + 1,
        'total_molecules': n_total,
        'train_molecules': n_train,
        'val_molecules': n_val,
        'kfold': kfold,
        'current_fold': current_fold,
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

    kfold = config['data'].get('kfold', 5)
    current_fold = config['data'].get('current_fold', 0)

    summary = {
        'config': {
            'train_folder': config['data']['train_folder'],
            'kfold': kfold,
            'current_fold': current_fold,
            'split_seed': config.get('random_seed', 42)
        },
        'summary': {
            'total_files': len(validation_info_list),
            'total_train_molecules': total_train_molecules,
            'total_val_molecules': total_val_molecules,
            'validation_behavior': (
                f"K-fold cross-validation with k={kfold}, using fold {current_fold} for validation. "
                f"Training uses {kfold-1}/{kfold} of data, validation uses 1/{kfold} of data. "
                "No train/val overlap (data leakage prevented)."
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
    print(f"K-fold: k={kfold}, validation fold={current_fold}")
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
    metrics_logger = None,
    start_iteration: int = 0
) -> Dict[str, List[float]]:
    """
    Main training loop for multi-file streaming dataset.

    Training terminology:
    - 1 EPOCH = Sweep through ALL files in the data folder
    - 1 ITERATION = Process 1 file

    Automatically cycles through files and checkpoints after each epoch.

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
        start_iteration: Global iteration to start from (default: 0, for resuming training)
                        Automatically calculates the correct epoch and file to resume from

    Returns:
        history: Dictionary with lists of training/validation metrics

    Training Strategy:
        1. Load file 1 → train iteration 1 → switch to file 2
        2. Load file 2 → train iteration 2 → switch to file 3
        3. ... repeat until all files processed (1 full epoch) → checkpoint
        4. Repeat for num_epochs

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
    print(f"Iterations per epoch: {len(train_dataset.file_paths)}")
    print(f"Total epochs: {config['training']['num_epochs']}")
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

    # Early stopping (disabled when current_fold = -1, i.e., training on all data)
    best_val_loss = float('inf')
    patience_counter = 0
    current_fold = config['data'].get('current_fold', 0)
    early_stopping_patience = config['training'].get('early_stopping_patience', 5) if current_fold != -1 else None
    validate_every = config['training'].get('validate_every', 1)

    # DataLoader parameters
    num_workers = config['data'].get('num_workers', 4)
    prefetch_factor = config['data'].get('prefetch_factor', 2) if num_workers > 0 else None

    # Training setup: 1 epoch = sweep through all files, 1 iteration = 1 file
    total_files = len(train_dataset.file_paths)
    num_epochs = config['training']['num_epochs']
    total_iterations = num_epochs * total_files

    # Calculate starting epoch and iteration from start_iteration
    start_epoch = start_iteration // total_files
    start_file_idx = start_iteration % total_files

    if start_iteration > 0:
        print(f"\n{'='*60}")
        print(f"Resuming from iteration {start_iteration}/{total_iterations}")
        print(f"Starting at epoch {start_epoch + 1}/{num_epochs}, file {start_file_idx + 1}/{total_files}")
        print(f"{'='*60}\n")

    # Validation ID tracking (built incrementally during first epoch)
    validation_info_list = []
    export_validation_ids = config['logging'].get('export_validation_ids', False)

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs} (Sweeping all {total_files} files)")
        print(f"{'='*60}")

        # Determine starting iteration for this epoch
        iteration_start = start_file_idx if epoch == start_epoch else 0

        # Iterate through all files (1 file = 1 iteration)
        for iteration in range(iteration_start, total_files):
            print(f"\n[Epoch {epoch + 1}, Iteration {iteration + 1}/{total_files}]")

            # Switch to next file
            # Skip file switching on very first iteration (file already loaded at index 0)
            # OR if we're resuming from start_iteration (files already positioned correctly)
            should_switch = iteration > 0 and not (epoch == start_epoch and iteration == iteration_start and start_iteration > 0)

            if should_switch:
                print(f"Switching to file {iteration + 1}/{total_files}...")
                train_dataset.next_file()
                # Validation dataset cycles through same files (uses k-fold split)
                val_dataset.next_file()
            elif start_iteration > 0 and epoch == start_epoch and iteration == iteration_start:
                # When resuming, we need to manually set the dataset to the correct file
                print(f"Resuming at file {iteration + 1}/{total_files}...")
                # Clear cache and set to correct file index
                train_dataset.chunk_cache.clear()
                train_dataset.current_file_idx = iteration
                train_dataset.current_file = train_dataset.file_paths[iteration]
                train_dataset._load_file_metadata()

                val_dataset.chunk_cache.clear()
                val_dataset.current_file_idx = iteration
                val_dataset.current_file = val_dataset.file_paths[iteration]
                val_dataset._load_file_metadata()

            # Track validation info during first epoch
            if epoch == 0 and export_validation_ids:
                val_info = _get_validation_info_for_current_file(
                    train_dataset,
                    val_dataset,
                    iteration,
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

            print(f"Training on file: {train_dataset.file_paths[train_dataset.current_file_idx]}")
            print(f"Train dataset size: {len(train_dataset)} bonds")
            print(f"Val dataset size: {len(val_dataset)} bonds")

            # Train one iteration
            start_time = time.time()
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, config,
                epoch, iteration, total_files
            )
            train_time = time.time() - start_time

            # Calculate global epoch (for logging and display)
            global_iteration = epoch * total_files + iteration + 1  # +1 for 1-indexed
            global_epoch = epoch + 1  # 1-indexed for display

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

            # Initialize val_metrics to None (will be set if validation runs)
            val_metrics = None

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

                # Early stopping check (only if early_stopping_patience is enabled)
                if early_stopping_patience is not None:
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

            # Checkpoint every N iterations (and always save the last iteration)
            save_every_n = config['checkpointing'].get('save_every_n_iterations', 1)
            is_last_iteration = (epoch == num_epochs - 1) and (iteration == total_files - 1)
            should_save = (global_iteration % save_every_n == 0) or is_last_iteration

            if checkpoint_callback is not None and should_save:
                if is_last_iteration:
                    print(f"\n[Checkpoint] Saving final iteration {iteration + 1}/{total_files}, epoch {epoch + 1}")
                else:
                    print(f"\n[Checkpoint] Saving at iteration {iteration + 1}/{total_files}, epoch {epoch + 1}")
                checkpoint_callback(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={
                        'iteration': iteration,
                        'global_iteration': global_iteration,
                        'train_loss': train_metrics['total_loss'],
                        'val_loss': val_metrics['total_loss'] if val_metrics is not None else None,
                        'best_val_loss': best_val_loss
                    }
                )

            # Log to metrics logger
            if metrics_logger is not None:
                metrics_logger.log(
                    iteration=global_iteration,
                    train_loss=train_metrics['total_loss'],
                    val_loss=val_metrics['total_loss'] if val_metrics is not None else None,
                    metrics={
                        'train_distance_loss': train_metrics['distance_loss'],
                        'train_geometry_loss': train_metrics['geometry_loss'],
                        'val_distance_loss': val_metrics.get('distance_loss') if val_metrics is not None else None,
                        'val_geometry_loss': val_metrics.get('geometry_loss') if val_metrics is not None else None,
                    },
                    lr=current_lr,
                    epoch=global_epoch
                )

        # End of iteration loop - save validation IDs after first epoch
        if epoch == 0 and export_validation_ids and len(validation_info_list) > 0:
            _save_validation_ids(validation_info_list, config)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*60}\n")

    return history

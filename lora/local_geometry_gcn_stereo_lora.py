#!/usr/bin/env python3
"""
Stereochemistry-Aware 3D Reconstruction GCN - LoRA Fine-Tuning Script

This script implements LoRA (Low-Rank Adaptation) fine-tuning for the Graph
Convolutional Network, enabling parameter-efficient fine-tuning of the model
for predicting local molecular geometry in Spherical coordinates (r, θ, φ).

Key Features:
- LoRA fine-tuning: Parameter-efficient training with low-rank adapters
- Multi-file streaming: Train on TB-scale datasets without RAM caching
- Spherical coordinates: Rotationally-invariant (r, θ, φ) representation
- Stereochemistry-aware: Ordered neighbors with azimuthal angles
- Percentage-based loss: Better scaling across distance ranges
- Training scheme: 1 epoch = all files, 1 iteration = 1 file

LoRA Configuration:
- Only trains adapter weights (1-5% of total parameters)
- Freezes pre-trained model weights
- Configurable rank and alpha for adapter scaling
- Can target specific layers (GAT, MLP, embeddings)

Usage:
    # Fine-tune with lora_config.yaml
    python local_geometry_gcn_stereo_lora.py

    # Configure LoRA settings in config YAML:
    # lora:
    #   enabled: true
    #   rank: 8                    # Low-rank dimension (4, 8, 16, 32)
    #   alpha: 16                  # Scaling factor (typically 2*rank)
    #   dropout: 0.1               # LoRA dropout
    #   target_modules:            # Modules to apply LoRA
    #     - "gat_layers"
    #     - "final_mlp"
    #   pretrained_checkpoint: "models/final_model.pt"  # Base model

    # To resume from a LoRA checkpoint:
    # training:
    #   resume_checkpoint: "checkpoints/lora_best_model.pt"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import utilities
from utils.config import load_config, validate_config, print_config
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


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for parameter-efficient fine-tuning.

    Implements: h = W0*x + (B*A)*x * (alpha/r)
    where W0 is frozen, A and B are trainable low-rank matrices.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices (A: in->rank, B: rank->out)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA adapter."""
        if self.dropout is not None:
            x = self.dropout(x)

        # Low-rank adaptation: (B @ A) @ x
        result = x @ self.lora_A @ self.lora_B
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines frozen base weights with trainable low-rank adapters.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Freeze base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Add LoRA adapter
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # Move LoRA to same device as base layer
        device = next(base_layer.parameters()).device
        self.lora = self.lora.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base + LoRA adapter."""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output


class LoRAGATConv(nn.Module):
    """
    GATConv layer with LoRA adaptation.

    Applies LoRA by recursively wrapping all internal Linear layers within GATConv.
    This approach is simpler and more robust than manually implementing GAT's
    attention mechanism with LoRA.

    The GATConv's internal Linear layers (lin_src, lin_dst, etc.) will be
    replaced with LoRALinear versions during initialization.
    """
    def __init__(
        self,
        base_gatconv: GATConv,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Store base GATConv (we'll modify it in-place)
        self.base_gatconv = base_gatconv

        # Get device
        device = next(self.base_gatconv.parameters()).device

        # Recursively replace all Linear layers inside GATConv with LoRALinear
        lora_added = 0
        for name, module in list(self.base_gatconv.named_children()):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(self.base_gatconv, name, lora_linear)
                lora_added += 1
                print(f"    Added LoRA to GATConv.{name}")

        if lora_added == 0:
            print(f"    Warning: No Linear layers found in GATConv")

        print(f"    Total LoRA layers in GATConv: {lora_added}")

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=None):
        """
        Forward pass through modified GATConv with LoRA.

        The internal Linear layers have been replaced with LoRALinear,
        so the standard GATConv forward pass now includes LoRA adaptations.
        """
        # Simply call the base GATConv - its internal layers now have LoRA
        return self.base_gatconv(
            x, edge_index, edge_attr=edge_attr, size=size,
            return_attention_weights=return_attention_weights
        )


def add_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Add LoRA adapters to specified modules in the model.

    Supports both Linear layers and GATConv layers (PyTorch Geometric).
    For GATConv, LoRA is applied to internal linear transformation layers.

    Args:
        model: Base model to add LoRA to
        rank: Low-rank dimension
        alpha: Scaling factor
        dropout: LoRA dropout rate
        target_modules: List of module name patterns to apply LoRA
                       (e.g., ["gnn_layers", "distance_head", "geometry_head"])

    Returns:
        Model with LoRA adapters added
    """
    if target_modules is None:
        target_modules = ["gnn_layers", "distance_head", "geometry_head"]

    lora_added_count = 0

    def apply_lora_recursive(module, prefix=""):
        nonlocal lora_added_count

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this module matches target patterns
            should_apply_lora = any(
                target in full_name for target in target_modules
            )

            print(f"Full name: {full_name}")
            print(f"Target modules: {target_modules}")
            print(f"Should apply LoRA: {should_apply_lora}")

            # Apply LoRA to Linear layers in targeted modules
            if should_apply_lora:
                if isinstance(child, nn.Linear):
                    # Replace with LoRA version
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_layer)
                    lora_added_count += 1
                    print(f"  Added LoRA to Linear: {full_name}")
                elif isinstance(child, GATConv):
                    # Replace with LoRA version
                    lora_layer = LoRAGATConv(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_layer)
                    lora_added_count += 1
                    print(f"  Added LoRA to GATConv: {full_name}")
                else:
                    print(f"Child is instance of: {type(child).__name__}")
                    apply_lora_recursive(child, full_name)
            else:
                # Recurse into child modules (including GATConv internals)
                apply_lora_recursive(child, full_name)

    apply_lora_recursive(model)
    print(f"\nTotal LoRA layers added: {lora_added_count}")

    return model


def freeze_base_model(model: nn.Module) -> None:
    """
    Freeze all non-LoRA parameters in the model.

    Args:
        model: Model with LoRA adapters
    """
    total_params = 0
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        # Check if this is a LoRA parameter
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()

    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"  Trainable LoRA parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Reduction factor: {total_params/trainable_params:.1f}x fewer trainable params")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base model for inference.

    Computes: W_merged = W0 + (A @ B)^T * (alpha/r)

    Where:
    - A: (in_features, rank)
    - B: (rank, out_features)
    - A @ B: (in_features, out_features)
    - (A @ B)^T: (out_features, in_features) - matches Linear weight shape

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with merged weights (LoRA adapters removed)
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Merge LoRA into base weights
            base_weight = module.base_layer.weight.data.clone()
            # Correct formula: (A @ B).T because A is (in,r), B is (r,out), weight is (out,in)
            lora_weight = (module.lora.lora_A @ module.lora.lora_B).T * module.lora.scaling
            merged_weight = base_weight + lora_weight

            # Create new linear layer with merged weights
            merged_layer = nn.Linear(
                module.base_layer.in_features,
                module.base_layer.out_features,
                bias=module.base_layer.bias is not None
            )
            merged_layer.weight.data = merged_weight
            if module.base_layer.bias is not None:
                merged_layer.bias.data = module.base_layer.bias.data.clone()

            # Replace LoRA layer with merged layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = model
            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, merged_layer)

    print("LoRA weights merged into base model")
    return model


# ============================================================================
# Training Setup Functions
# ============================================================================

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
    Create training and validation datasets using k-fold cross-validation.

    Uses k-fold splitting (default k=5) to create non-overlapping train/val sets.
    No data leakage: validation fold is completely separate from training folds.

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

    # Get k-fold configuration
    kfold = config['data'].get('kfold', 5)
    current_fold = config['data'].get('current_fold', 0)
    split_seed = config.get('random_seed', 42)

    if current_fold == -1:
        print(f"\nTRAIN ALL MODE (current_fold=-1)")
        print(f"  Training: 100% of data (all molecules)")
        print(f"  Validation: DISABLED (empty)")
    else:
        print(f"\nK-FOLD CROSS-VALIDATION (k={kfold}, fold={current_fold})")
        print(f"  Training: {kfold-1}/{kfold} of data ({(kfold-1)/kfold*100:.1f}%)")
        print(f"  Validation: 1/{kfold} of data ({1/kfold*100:.1f}%)")
        print(f"  No train/val overlap (data leakage prevented)")

    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = MultiFileStreamingDataset(
        file_paths=[str(f) for f in train_files],
        element_to_idx=element_to_idx,
        max_bonds=config['data']['max_bonds'],
        chunk_size=config['data'].get('chunk_size', 10000),
        cache_ram_mb=config['data'].get('cache_ram_mb', 4096),
        training=True,
        split='train',
        split_seed=split_seed,
        kfold=kfold,
        current_fold=current_fold
    )

    print(f"Training dataset created:")
    print(f"  Files: {len(train_dataset.file_paths)}")
    print(f"  Current file: {train_dataset.file_paths[train_dataset.current_file_idx]}")
    print(f"  Bonds in current file: {len(train_dataset)}")

    # Create validation dataset
    print("\nCreating validation dataset...")
    val_dataset = MultiFileStreamingDataset(
        file_paths=[str(f) for f in train_files],
        element_to_idx=element_to_idx,
        max_bonds=config['data']['max_bonds'],
        chunk_size=config['data'].get('chunk_size', 10000),
        cache_ram_mb=config['data'].get('cache_ram_mb', 4096),
        training=False,
        split='val',
        split_seed=split_seed,
        kfold=kfold,
        current_fold=current_fold
    )
    print(f"Validation dataset created:")
    print(f"  Files: {len(val_dataset.file_paths)}")
    print(f"  Current file: {val_dataset.file_paths[val_dataset.current_file_idx]}")
    print(f"  Bonds in current file: {len(val_dataset)}")

    print("="*60 + "\n")

    return train_dataset, val_dataset, element_to_idx


def create_model_with_lora(config, device):
    """
    Create model and add LoRA adapters for fine-tuning.

    Args:
        config: Configuration dict
        device: torch.device

    Returns:
        model: LocalGeometryGCN model with LoRA adapters
    """
    print("\n" + "="*60)
    print("Creating Model with LoRA Adapters")
    print("="*60)

    # Create base model
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

    # Load pre-trained weights if specified
    pretrained_checkpoint = config.get('lora', {}).get('pretrained_checkpoint')
    if pretrained_checkpoint:
        print(f"\nLoading pre-trained weights from: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Pre-trained weights loaded successfully")
    else:
        print("\nWarning: No pre-trained checkpoint specified!")
        print("  Set 'lora.pretrained_checkpoint' in config for fine-tuning")
        print("  Initializing with random weights...")
        initialize_weights(model)

    # Print base model info
    base_params = count_parameters(model)
    print(f"\nBase Model Architecture:")
    print(f"  Hidden dim: {config['model']['hidden_dim']}")
    print(f"  Embedding dim: {config['model']['embedding_dim']}")
    print(f"  Vocabulary size: {config['model']['vocab_size']}")
    print(f"  Max bonds: {config['model']['max_bonds']}")
    print(f"  Depth (GAT layers): {config['model'].get('depth', 1)}")
    print(f"  Total base parameters: {base_params:,}")

    # Add LoRA adapters
    lora_config = config.get('lora', {})
    if lora_config.get('enabled', True):
        print(f"\nAdding LoRA Adapters:")
        print(f"  Rank: {lora_config.get('rank', 8)}")
        print(f"  Alpha: {lora_config.get('alpha', 16.0)}")
        print(f"  Dropout: {lora_config.get('dropout', 0.0)}")
        print(f"  Target modules: {lora_config.get('target_modules', ['gat_layers', 'final_mlp'])}")

        model = add_lora_to_model(
            model,
            rank=lora_config.get('rank', 8),
            alpha=lora_config.get('alpha', 16.0),
            dropout=lora_config.get('dropout', 0.0),
            target_modules=lora_config.get('target_modules', ['gat_layers', 'final_mlp'])
        )

        # Freeze base model parameters
        freeze_base_model(model)
    else:
        print("\nWarning: LoRA disabled in config! Training all parameters...")

    print("="*60 + "\n")

    return model


def create_optimizer_and_scheduler(model, config):
    """
    Create optimizer and learning rate scheduler for LoRA parameters.

    Args:
        model: Model to optimize (with LoRA adapters)
        config: Configuration dict

    Returns:
        optimizer: Optimizer (only for LoRA parameters)
        scheduler: Learning rate scheduler (or None)
    """
    print("Creating optimizer and scheduler...")

    # Get LoRA-specific or default training config
    lora_config = config.get('lora', {})
    lr = lora_config.get('learning_rate', config['training']['learning_rate'])
    weight_decay = lora_config.get('weight_decay', config['training'].get('weight_decay', 0.001))

    # Only optimize LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]

    if not lora_params:
        raise ValueError("No trainable parameters found! Check LoRA configuration.")

    optimizer = optim.Adam(
        lora_params,
        lr=lr,
        weight_decay=weight_decay
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
        print(f"  Scheduler: ReduceLROnPlateau")

    elif scheduler_type == 'step':
        params = config['training'].get('lr_scheduler_params', {})
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 10),
            gamma=params.get('gamma', 0.7)
        )
        print(f"  Scheduler: StepLR")

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

    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Trainable parameters: {sum(p.numel() for p in lora_params):,}")

    return optimizer, scheduler


def init_wandb(config):
    """
    Initialize Weights & Biases logging for LoRA fine-tuning.

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

    # Generate run name
    run_name = config['logging'].get('wandb_run_name')
    if not run_name:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_name = f"lora-{timestamp}"

    # Prepare LoRA config for wandb
    lora_config_dict = config.get('lora', {})

    # Initialize wandb
    wandb.init(
        project=config['logging'].get('wandb_project', 'stereochem-gcn-lora'),
        entity=config['logging'].get('wandb_entity'),
        name=run_name,
        tags=config['logging'].get('wandb_tags', []) + ['lora'],
        notes=config['logging'].get('wandb_notes'),
        config={
            # Training config
            'batch_size': config['data']['batch_size'],
            'learning_rate': lora_config_dict.get('learning_rate', config['training']['learning_rate']),
            'num_epochs': config['training']['num_epochs'],
            'weight_decay': lora_config_dict.get('weight_decay', config['training'].get('weight_decay', 0.001)),
            'grad_clip': config['training'].get('grad_clip', 1.0),

            # Model config
            'hidden_dim': config['model']['hidden_dim'],
            'embedding_dim': config['model']['embedding_dim'],
            'max_bonds': config['model']['max_bonds'],
            'depth': config['model'].get('depth', 1),
            'use_bond_types': config['model'].get('use_bond_types', True),

            # LoRA config
            'lora_enabled': lora_config_dict.get('enabled', True),
            'lora_rank': lora_config_dict.get('rank', 8),
            'lora_alpha': lora_config_dict.get('alpha', 16.0),
            'lora_dropout': lora_config_dict.get('dropout', 0.0),
            'lora_target_modules': lora_config_dict.get('target_modules', ['gat_layers', 'final_mlp']),
            'pretrained_checkpoint': lora_config_dict.get('pretrained_checkpoint', 'none'),

            # Data config
            'kfold': config['data'].get('kfold', 5),
            'current_fold': config['data'].get('current_fold', 0),

            # Loss config
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
    Main LoRA fine-tuning function.
    """
    # Load configuration
    config_file = os.environ.get('CONFIG_FILE', 'lora_config.yaml')
    config = load_config(config_file)

    print("\n" + "="*60)
    print("LoRA Fine-Tuning: Stereochemistry-Aware 3D GCN")
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

    # Create model with LoRA adapters
    model = create_model_with_lora(config, device)

    # Initialize wandb if enabled
    use_wandb = init_wandb(config)

    # Create optimizer and scheduler (only for LoRA params)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Create metrics logger
    metrics_logger = MetricsLogger(
        log_dir=config['logging']['log_dir'],
        use_wandb=use_wandb
    )

    # Resume from checkpoint if specified
    start_iteration = 0
    resume_checkpoint = config.get('training', {}).get('resume_checkpoint', None)
    resume_from_latest = config.get('training', {}).get('resume_from_latest', False)

    # Find latest checkpoint if requested
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

        if 'metrics' in checkpoint_info and 'global_iteration' in checkpoint_info['metrics']:
            start_iteration = checkpoint_info['metrics']['global_iteration']
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No iteration info found in checkpoint, starting from iteration 0")

    # Checkpoint callback
    def checkpoint_callback(epoch, model, optimizer, scheduler, metrics):
        """Save LoRA checkpoint."""
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
    print("Starting LoRA Fine-Tuning")
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

    # Save final LoRA model
    from utils.checkpoint import save_final_model

    # Option 1: Save with LoRA adapters (for further fine-tuning)
    save_final_model(
        model=model,
        config=config,
        metrics={'history': history},
        save_dir="models",
        filename="final_lora_model.pt"
    )
    print("Final LoRA model saved (with adapters)")

    # Option 2: Save merged model (for inference)
    if config.get('lora', {}).get('save_merged', True):
        print("\nMerging LoRA weights into base model...")
        merged_model = merge_lora_weights(model)
        save_final_model(
            model=merged_model,
            config=config,
            metrics={'history': history},
            save_dir="models",
            filename="final_merged_model.pt"
        )
        print("Merged model saved (LoRA weights integrated)")

    # Finish wandb logging
    metrics_logger.finish()

    print("\n" + "="*60)
    print("LoRA Fine-Tuning Complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nFine-tuning interrupted by user.")
    except Exception as e:
        print(f"\n\nError during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

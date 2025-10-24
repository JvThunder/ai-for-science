"""
Configuration utilities for loading and validating YAML configs.

Provides functions for:
- Loading configuration from YAML files
- Validating configuration parameters
- Merging configs with command-line overrides
- Setting default values

Example config.yaml structure:
    data:
      train_folder: "./data"
      val_split_ratio: 0.1
      max_bonds: 7
      batch_size: 32

    model:
      hidden_dim: 64
      embedding_dim: 32
      vocab_size: 12

    training:
      num_epochs: 50
      learning_rate: 0.0001
      weight_decay: 0.001

    loss:
      distance_weight: 1.0
      geometry_weight: 0.3
"""

import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        config: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from: {config_path}")

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {save_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.

    Returns:
        config: Default configuration dictionary
    """
    config = {
        'data': {
            'train_folder': './data',
            'val_split_ratio': 0.1,  # Fraction of each file for validation
            'max_bonds': 7,
            'batch_size': 32,
            'num_workers': 4,  # Parallel data loading (set to 0 to disable)
            'prefetch_factor': 2,  # Batches to prefetch per worker
            'chunk_size': 10000,  # Rows per chunk for streaming
            'cache_ram_mb': 4096,  # RAM for caching chunks (MB, 4GB default)
            'min_element_count': 10  # Minimum occurrences for vocabulary
        },
        'model': {
            'hidden_dim': 64,
            'embedding_dim': 32,
            'vocab_size': None,  # Set during training from data
            'max_bonds': 7
        },
        'training': {
            'num_epochs': 50,
            'learning_rate': 0.0001,
            'weight_decay': 0.001,
            'grad_clip': 1.0,
            'early_stopping_patience': 5,
            'validate_every': 1,  # Validate every N epochs
            'lr_scheduler': 'plateau',  # 'plateau', 'step', 'cosine', or None
            'lr_scheduler_params': {
                'factor': 0.7,
                'patience': 2,
                'mode': 'min'
            }
        },
        'loss': {
            'distance_weight': 1.0,
            'geometry_weight': 0.3,
            'theta_weight': 0.5,
            'phi_weight': 0.5
        },
        'checkpointing': {
            'checkpoint_dir': 'checkpoints',
            'save_every': 1,  # Save every N epochs
            'keep_last_n': 3  # Keep only N most recent checkpoints
        },
        'logging': {
            'log_dir': 'logs',
            'log_every': 10  # Log every N batches
        }
    }

    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.

    Args:
        base_config: Base configuration
        override_config: Config to override with

    Returns:
        merged_config: Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        is_valid: True if config is valid

    Raises:
        ValueError: If config is invalid
    """
    # Check required sections
    required_sections = ['data', 'model', 'training', 'loss']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate data section
    data_config = config['data']
    if 'train_folder' not in data_config or not data_config['train_folder']:
        raise ValueError("data.train_folder must be specified and non-empty")

    # Check that folder exists
    train_folder = Path(data_config['train_folder'])
    if not train_folder.exists():
        raise ValueError(f"Training folder not found: {train_folder}")

    if not train_folder.is_dir():
        raise ValueError(f"train_folder must be a directory: {train_folder}")

    # Check that folder contains at least one CSV file
    csv_files = list(train_folder.glob('*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in training folder: {train_folder}")

    # Validate val_split_ratio if specified
    val_split_ratio = data_config.get('val_split_ratio', 0.1)
    if val_split_ratio < 0.0 or val_split_ratio >= 1.0:
        raise ValueError("val_split_ratio must be in range [0.0, 1.0)")

    # Validate numeric ranges
    if config['data']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")

    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")

    if config['training']['num_epochs'] <= 0:
        raise ValueError("num_epochs must be positive")

    # Validate loss weights
    loss_config = config['loss']
    if any(w < 0 for w in [loss_config['distance_weight'], loss_config['geometry_weight']]):
        raise ValueError("Loss weights must be non-negative")

    print("[OK] Configuration validated successfully")
    return True


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty-print configuration.

    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def create_example_config(save_path: str = "config.yaml"):
    """
    Create an example configuration file.

    Args:
        save_path: Path to save example config
    """
    config = get_default_config()
    save_config(config, save_path)
    print(f"Example configuration created at: {save_path}")

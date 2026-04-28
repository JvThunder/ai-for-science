#!/usr/bin/env python3
"""
Inspect LoRA Model Structure

This script helps you understand which layers in your model will have LoRA applied.
It prints the model structure and shows which modules match your target patterns.

Usage:
    python inspect_lora.py
    python inspect_lora.py --config lora_config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils.config import load_config
from utils.model import LocalGeometryGCN
import argparse


def print_model_structure(model, prefix="", target_modules=None):
    """
    Print model structure with indentation, highlighting targeted modules.

    Args:
        model: PyTorch model
        prefix: Current prefix for nested modules
        target_modules: List of target module patterns
    """
    if target_modules is None:
        target_modules = []

    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Check if this module matches target patterns
        is_targeted = any(target in full_name for target in target_modules)
        marker = " ✓ [LORA TARGET]" if is_targeted else ""

        # Print module type and name
        indent = "  " * (len(prefix.split('.')) if prefix else 0)
        print(f"{indent}{name}: {type(child).__name__}{marker}")

        # Show Linear layer details
        if isinstance(child, nn.Linear):
            print(f"{indent}  → Linear({child.in_features}, {child.out_features})")
            if is_targeted:
                print(f"{indent}    LoRA will be added here!")

        # Recurse into children
        print_model_structure(child, full_name, target_modules)


def count_lora_params(model, rank, target_modules):
    """
    Count how many LoRA parameters will be added.

    Args:
        model: PyTorch model
        rank: LoRA rank
        target_modules: List of target module patterns

    Returns:
        Dict with parameter counts
    """
    base_params = sum(p.numel() for p in model.parameters())
    lora_params = 0
    lora_layers_count = 0

    def count_recursive(module, prefix=""):
        nonlocal lora_params, lora_layers_count

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            is_targeted = any(target in full_name for target in target_modules)

            if is_targeted and isinstance(child, nn.Linear):
                # LoRA adds two matrices: A (in x rank) and B (rank x out)
                in_features = child.in_features
                out_features = child.out_features
                lora_params += in_features * rank + rank * out_features
                lora_layers_count += 1

            count_recursive(child, full_name)

    count_recursive(model)

    return {
        'base_params': base_params,
        'lora_params': lora_params,
        'lora_layers': lora_layers_count,
        'total_params': base_params + lora_params,
        'reduction_factor': base_params / lora_params if lora_params > 0 else 0,
        'lora_percentage': (lora_params / base_params * 100) if lora_params > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect LoRA model structure")
    parser.add_argument('--config', default='lora/lora_config.yaml', help='Config file path')
    args = parser.parse_args()

    # Load config
    print("="*70)
    print("LoRA Model Structure Inspector")
    print("="*70)
    print(f"\nLoading config from: {args.config}")

    config = load_config(args.config)

    # Get LoRA config
    lora_config = config.get('lora', {})
    rank = lora_config.get('rank', 8)
    target_modules = lora_config.get('target_modules', ['gnn_layers', 'distance_head', 'geometry_head'])

    print(f"\nLoRA Configuration:")
    print(f"  Rank: {rank}")
    print(f"  Target modules: {target_modules}")

    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)

    model = LocalGeometryGCN(
        num_node_features=config['model'].get('vocab_size', 70),
        hidden_dim=config['model']['hidden_dim'],
        max_bonds=config['data']['max_bonds'],
        vocab_size=config['model'].get('vocab_size', 70),
        embedding_dim=config['model']['embedding_dim'],
        depth=config['model'].get('depth', 1),
        use_bond_types=config['model'].get('use_bond_types', True),
        num_bond_types=config['model'].get('num_bond_types', 4),
        bond_embedding_dim=config['model'].get('bond_embedding_dim', 8)
    )

    # Print model structure
    print("\n" + "="*70)
    print("Model Structure (✓ = LoRA will be applied)")
    print("="*70)
    print_model_structure(model, target_modules=target_modules)

    # Count parameters
    print("\n" + "="*70)
    print("Parameter Analysis")
    print("="*70)

    stats = count_lora_params(model, rank, target_modules)

    print(f"\nBase model parameters:           {stats['base_params']:>15,}")
    print(f"LoRA parameters to add:          {stats['lora_params']:>15,}")
    print(f"Total with LoRA:                 {stats['total_params']:>15,}")
    print(f"\nNumber of LoRA layers:           {stats['lora_layers']:>15,}")
    print(f"Reduction factor:                {stats['reduction_factor']:>15.1f}x")
    print(f"LoRA percentage:                 {stats['lora_percentage']:>14.2f}%")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    if stats['lora_layers'] == 0:
        print("\n⚠️  WARNING: No layers match the target modules!")
        print("   No LoRA adapters will be added.")
        print("\n   Suggestions:")
        print("   1. Check target_modules in your config")
        print("   2. Available modules to target:")
        print("      - gnn_layers (GAT layers)")
        print("      - distance_head (distance prediction)")
        print("      - geometry_head (geometry prediction)")
        print("      - edge_projections (edge features)")
        print("      - element_embedding (element embeddings)")
    else:
        print(f"\n✓ LoRA will be applied to {stats['lora_layers']} Linear layers")
        print(f"  Training only {stats['lora_percentage']:.2f}% of parameters")
        print(f"  {stats['reduction_factor']:.1f}x fewer parameters than full fine-tuning")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()

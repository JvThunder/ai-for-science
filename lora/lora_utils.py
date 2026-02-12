#!/usr/bin/env python3
"""
LoRA Utilities for Model Management and Analysis

This module provides utility functions for working with LoRA fine-tuned models:
- Compare parameter counts between base and LoRA models
- Merge multiple LoRA adapters
- Extract and save LoRA weights separately
- Load and apply LoRA weights to base model
- Analyze LoRA layer statistics

Usage:
    # Compare models
    python lora_utils.py compare --base models/final_model.pt --lora models/final_lora_model.pt

    # Merge LoRA into base model
    python lora_utils.py merge --lora models/final_lora_model.pt --output models/merged.pt

    # Extract LoRA weights only
    python lora_utils.py extract --lora models/final_lora_model.pt --output lora_weights.pt

    # Analyze LoRA statistics
    python lora_utils.py analyze --lora models/final_lora_model.pt
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
import json


def count_parameters(model: nn.Module, only_trainable: bool = False) -> Dict[str, int]:
    """
    Count parameters in a model.

    Args:
        model: PyTorch model
        only_trainable: Only count trainable parameters

    Returns:
        Dictionary with parameter counts
    """
    total = 0
    trainable = 0
    lora = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params

        if param.requires_grad:
            trainable += num_params

        if 'lora_A' in name or 'lora_B' in name:
            lora += num_params

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'lora': lora,
        'base': total - lora
    }


def compare_models(base_path: str, lora_path: str) -> None:
    """
    Compare parameter counts between base and LoRA models.

    Args:
        base_path: Path to base model checkpoint
        lora_path: Path to LoRA model checkpoint
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON: Base vs LoRA")
    print("="*70)

    # Load models
    print(f"\nLoading base model from: {base_path}")
    base_checkpoint = torch.load(base_path, map_location='cpu')

    print(f"Loading LoRA model from: {lora_path}")
    lora_checkpoint = torch.load(lora_path, map_location='cpu')

    # Extract state dicts
    if isinstance(base_checkpoint, dict) and 'model_state_dict' in base_checkpoint:
        base_state = base_checkpoint['model_state_dict']
    else:
        base_state = base_checkpoint

    if isinstance(lora_checkpoint, dict) and 'model_state_dict' in lora_checkpoint:
        lora_state = lora_checkpoint['model_state_dict']
    else:
        lora_state = lora_checkpoint

    # Count parameters
    base_total = sum(p.numel() for p in base_state.values())
    lora_total = sum(p.numel() for p in lora_state.values())
    lora_adapter_params = sum(
        p.numel() for name, p in lora_state.items()
        if 'lora_A' in name or 'lora_B' in name
    )

    print("\n" + "-"*70)
    print("PARAMETER COUNTS")
    print("-"*70)
    print(f"Base model parameters:        {base_total:>15,}")
    print(f"LoRA model total parameters:  {lora_total:>15,}")
    print(f"LoRA adapter parameters:      {lora_adapter_params:>15,}")
    print(f"LoRA overhead:                {lora_total - base_total:>15,}")

    if lora_adapter_params > 0:
        reduction = base_total / lora_adapter_params
        percentage = (lora_adapter_params / base_total) * 100
        print(f"\nReduction factor:             {reduction:>15.1f}x")
        print(f"LoRA percentage:              {percentage:>14.2f}%")

    # Analyze LoRA layers
    print("\n" + "-"*70)
    print("LORA LAYERS")
    print("-"*70)

    lora_layers = {}
    for name in lora_state.keys():
        if 'lora_A' in name or 'lora_B' in name:
            # Extract layer name (remove .lora_A or .lora_B)
            layer_name = name.replace('.lora_A', '').replace('.lora_B', '')
            if layer_name not in lora_layers:
                lora_layers[layer_name] = {'A': None, 'B': None}

            if 'lora_A' in name:
                lora_layers[layer_name]['A'] = lora_state[name].shape
            elif 'lora_B' in name:
                lora_layers[layer_name]['B'] = lora_state[name].shape

    for layer_name, shapes in sorted(lora_layers.items()):
        if shapes['A'] is not None and shapes['B'] is not None:
            rank = shapes['A'][1]  # Second dimension of A matrix
            in_features = shapes['A'][0]
            out_features = shapes['B'][1]
            lora_params = in_features * rank + rank * out_features
            full_params = in_features * out_features
            reduction = full_params / lora_params

            print(f"\n{layer_name}")
            print(f"  Shape: {in_features} x {out_features}")
            print(f"  Rank: {rank}")
            print(f"  LoRA params: {lora_params:,} (Full: {full_params:,})")
            print(f"  Reduction: {reduction:.1f}x")

    print("\n" + "="*70 + "\n")


def merge_lora_weights_from_checkpoint(lora_checkpoint_path: str, output_path: str) -> None:
    """
    Merge LoRA weights into base model from checkpoint file.

    Args:
        lora_checkpoint_path: Path to LoRA checkpoint
        output_path: Path to save merged model
    """
    print("\n" + "="*70)
    print("MERGING LORA WEIGHTS")
    print("="*70)

    print(f"\nLoading LoRA model from: {lora_checkpoint_path}")
    checkpoint = torch.load(lora_checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    else:
        state_dict = checkpoint
        metadata = {}

    # Find all LoRA layers
    lora_layers = set()
    for name in state_dict.keys():
        if 'lora_A' in name or 'lora_B' in name:
            layer_name = name.replace('.lora_A', '').replace('.lora_B', '')
            lora_layers.add(layer_name)

    print(f"Found {len(lora_layers)} LoRA layers to merge")

    # Merge each LoRA layer
    merged_state = {}
    for name, param in state_dict.items():
        # Skip LoRA matrices
        if 'lora_A' in name or 'lora_B' in name:
            continue

        # Check if this is a base layer with LoRA
        if any(name.startswith(lora_layer) for lora_layer in lora_layers):
            # This is a weight matrix with corresponding LoRA
            base_weight = param.clone()

            # Find corresponding LoRA matrices
            lora_a_name = f"{name}.lora_A"
            lora_b_name = f"{name}.lora_B"

            if lora_a_name in state_dict and lora_b_name in state_dict:
                lora_A = state_dict[lora_a_name]
                lora_B = state_dict[lora_b_name]

                # Calculate scaling (assume alpha = 2 * rank)
                rank = lora_A.shape[1]
                alpha = rank * 2  # Default assumption
                scaling = alpha / rank

                # Merge: W_merged = W_base + (B @ A^T) * scaling
                lora_weight = (lora_B @ lora_A.T) * scaling
                merged_weight = base_weight + lora_weight.T

                merged_state[name] = merged_weight
                print(f"  Merged: {name}")
            else:
                merged_state[name] = param
        else:
            merged_state[name] = param

    # Save merged model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = metadata.copy()
    save_dict['model_state_dict'] = merged_state

    torch.save(save_dict, output_path)
    print(f"\nMerged model saved to: {output_path}")
    print("="*70 + "\n")


def extract_lora_weights(lora_checkpoint_path: str, output_path: str) -> None:
    """
    Extract only LoRA weights from checkpoint.

    Args:
        lora_checkpoint_path: Path to LoRA checkpoint
        output_path: Path to save LoRA weights only
    """
    print("\n" + "="*70)
    print("EXTRACTING LORA WEIGHTS")
    print("="*70)

    print(f"\nLoading LoRA model from: {lora_checkpoint_path}")
    checkpoint = torch.load(lora_checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    else:
        state_dict = checkpoint
        metadata = {}

    # Extract only LoRA parameters
    lora_weights = {}
    for name, param in state_dict.items():
        if 'lora_A' in name or 'lora_B' in name:
            lora_weights[name] = param

    print(f"Extracted {len(lora_weights)} LoRA tensors")

    # Calculate total LoRA parameters
    total_params = sum(p.numel() for p in lora_weights.values())
    print(f"Total LoRA parameters: {total_params:,}")

    # Save LoRA weights
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'lora_weights': lora_weights,
        'metadata': metadata
    }

    torch.save(save_dict, output_path)
    print(f"\nLoRA weights saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("="*70 + "\n")


def analyze_lora_statistics(lora_checkpoint_path: str) -> None:
    """
    Analyze statistics of LoRA weights.

    Args:
        lora_checkpoint_path: Path to LoRA checkpoint
    """
    print("\n" + "="*70)
    print("LORA WEIGHT STATISTICS")
    print("="*70)

    print(f"\nLoading LoRA model from: {lora_checkpoint_path}")
    checkpoint = torch.load(lora_checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Organize LoRA layers
    lora_layers = {}
    for name, param in state_dict.items():
        if 'lora_A' in name or 'lora_B' in name:
            layer_name = name.replace('.lora_A', '').replace('.lora_B', '')
            if layer_name not in lora_layers:
                lora_layers[layer_name] = {'A': None, 'B': None}

            if 'lora_A' in name:
                lora_layers[layer_name]['A'] = param
            elif 'lora_B' in name:
                lora_layers[layer_name]['B'] = param

    print(f"\nFound {len(lora_layers)} LoRA layers")

    # Analyze each layer
    print("\n" + "-"*70)
    for layer_name, matrices in sorted(lora_layers.items()):
        if matrices['A'] is None or matrices['B'] is None:
            continue

        A = matrices['A']
        B = matrices['B']

        print(f"\n{layer_name}")
        print(f"  A matrix shape: {tuple(A.shape)}")
        print(f"  B matrix shape: {tuple(B.shape)}")
        print(f"  Rank: {A.shape[1]}")

        # Statistics for A matrix
        print(f"\n  A matrix statistics:")
        print(f"    Mean: {A.mean().item():.6f}")
        print(f"    Std:  {A.std().item():.6f}")
        print(f"    Min:  {A.min().item():.6f}")
        print(f"    Max:  {A.max().item():.6f}")
        print(f"    Norm: {A.norm().item():.6f}")

        # Statistics for B matrix
        print(f"\n  B matrix statistics:")
        print(f"    Mean: {B.mean().item():.6f}")
        print(f"    Std:  {B.std().item():.6f}")
        print(f"    Min:  {B.min().item():.6f}")
        print(f"    Max:  {B.max().item():.6f}")
        print(f"    Norm: {B.norm().item():.6f}")

        # Compute effective update magnitude
        rank = A.shape[1]
        alpha = rank * 2  # Assume default
        scaling = alpha / rank
        update = (B @ A.T) * scaling
        print(f"\n  Effective update (B @ A^T * {scaling:.1f}):")
        print(f"    Mean: {update.mean().item():.6f}")
        print(f"    Std:  {update.std().item():.6f}")
        print(f"    Norm: {update.norm().item():.6f}")

    print("\n" + "="*70 + "\n")


def main():
    """Command-line interface for LoRA utilities."""
    parser = argparse.ArgumentParser(
        description="LoRA Model Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare base and LoRA models')
    compare_parser.add_argument('--base', required=True, help='Path to base model checkpoint')
    compare_parser.add_argument('--lora', required=True, help='Path to LoRA model checkpoint')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge LoRA weights into base model')
    merge_parser.add_argument('--lora', required=True, help='Path to LoRA checkpoint')
    merge_parser.add_argument('--output', required=True, help='Path to save merged model')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract LoRA weights only')
    extract_parser.add_argument('--lora', required=True, help='Path to LoRA checkpoint')
    extract_parser.add_argument('--output', required=True, help='Path to save LoRA weights')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze LoRA statistics')
    analyze_parser.add_argument('--lora', required=True, help='Path to LoRA checkpoint')

    args = parser.parse_args()

    if args.command == 'compare':
        compare_models(args.base, args.lora)
    elif args.command == 'merge':
        merge_lora_weights_from_checkpoint(args.lora, args.output)
    elif args.command == 'extract':
        extract_lora_weights(args.lora, args.output)
    elif args.command == 'analyze':
        analyze_lora_statistics(args.lora)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

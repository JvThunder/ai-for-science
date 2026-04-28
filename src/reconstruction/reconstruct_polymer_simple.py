#!/usr/bin/env python3
"""
Simple polymer reconstruction - reconstructs the full chain directly
without chunk-based merging.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import argparse
from typing import List, Dict

# Import from reconstruction module
from src.reconstruction.reconstruction import load_trained_model, reconstruct_from_center


def create_polyethylene_chain(n_carbons: int) -> tuple:
    """
    Create a simple polyethylene chain with n carbon atoms.

    Args:
        n_carbons: Number of carbon atoms in the chain

    Returns:
        elements: List of element symbols
        bonds: List of bond dictionaries
    """
    elements = []
    bonds = []

    atom_idx = 0
    for c_idx in range(n_carbons):
        # Add carbon
        c_atom_idx = atom_idx
        elements.append('C')
        atom_idx += 1

        # Add 2 hydrogens per carbon
        h1_idx = atom_idx
        elements.append('H')
        atom_idx += 1

        h2_idx = atom_idx
        elements.append('H')
        atom_idx += 1

        # C-H bonds
        bonds.append({'i': c_atom_idx, 'j': h1_idx, 'order': 1.0})
        bonds.append({'i': c_atom_idx, 'j': h2_idx, 'order': 1.0})

        # C-C backbone bond (except for last carbon)
        if c_idx < n_carbons - 1:
            next_c_idx = c_atom_idx + 3  # Next carbon is 3 atoms away
            bonds.append({'i': c_atom_idx, 'j': next_c_idx, 'order': 1.0})

    print(f"Created polyethylene chain:")
    print(f"  Carbons: {n_carbons}")
    print(f"  Total atoms: {len(elements)} ({n_carbons}C + {len(elements) - n_carbons}H)")
    print(f"  Total bonds: {len(bonds)}")
    print(f"  C-C backbone bonds: {n_carbons - 1}")
    print(f"  C-H bonds: {2 * n_carbons}")

    return elements, bonds


def analyze_structure(elements, positions, bonds):
    """Analyze bond lengths in reconstructed structure."""

    c_c_lengths = []
    c_h_lengths = []

    for bond in bonds:
        i, j = bond['i'], bond['j']
        dist = np.linalg.norm(positions[i] - positions[j])

        if elements[i] == 'C' and elements[j] == 'C':
            c_c_lengths.append(dist)
        elif (elements[i] == 'C' and elements[j] == 'H') or \
             (elements[i] == 'H' and elements[j] == 'C'):
            c_h_lengths.append(dist)

    print("\n[Bond Length Analysis]")
    if c_c_lengths:
        print(f"  C-C bonds: {len(c_c_lengths)}")
        print(f"    Mean: {np.mean(c_c_lengths):.3f} Å")
        print(f"    Std:  {np.std(c_c_lengths):.3f} Å")
        print(f"    Range: [{np.min(c_c_lengths):.3f}, {np.max(c_c_lengths):.3f}] Å")
        print(f"    Expected: ~1.54 Å")

    if c_h_lengths:
        print(f"  C-H bonds: {len(c_h_lengths)}")
        print(f"    Mean: {np.mean(c_h_lengths):.3f} Å")
        print(f"    Std:  {np.std(c_h_lengths):.3f} Å")
        print(f"    Range: [{np.min(c_h_lengths):.3f}, {np.max(c_h_lengths):.3f}] Å")
        print(f"    Expected: ~1.09 Å")


def export_to_xyz(elements, positions, filepath, comment="Polyethylene chain"):
    """Export structure to XYZ format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n")
        f.write(f"{comment}\n")
        for elem, pos in zip(elements, positions):
            f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    print(f"  Exported XYZ: {filepath}")


def visualize_structure(elements, positions, bonds, filepath):
    """Create visualization of the structure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 8))

    # CPK colors
    element_colors = {'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D'}
    element_sizes = {'H': 150, 'C': 400}

    # Two views
    for plot_idx, (elev, azim, title) in enumerate([
        (20, 45, "Perspective View"),
        (30, -60, "Alternate View")
    ], 1):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')

        # Draw bonds
        for bond in bonds:
            i, j = bond['i'], bond['j']
            elem_i, elem_j = elements[i], elements[j]

            if elem_i == 'C' and elem_j == 'C':
                color = '#404040'
                linewidth = 3.5
            elif (elem_i == 'C' and elem_j == 'H') or (elem_i == 'H' and elem_j == 'C'):
                color = '#808080'
                linewidth = 2.5
            else:
                color = '#606060'
                linewidth = 2.0

            ax.plot([positions[i, 0], positions[j, 0]],
                   [positions[i, 1], positions[j, 1]],
                   [positions[i, 2], positions[j, 2]],
                   color=color, linewidth=linewidth, zorder=1)

        # Draw atoms
        for elem in set(elements):
            indices = [i for i, e in enumerate(elements) if e == elem]
            color = element_colors.get(elem, '#FF1493')
            size = element_sizes.get(elem, 200)

            ax.scatter(positions[indices, 0],
                      positions[indices, 1],
                      positions[indices, 2],
                      c=color, s=size, edgecolors='black',
                      linewidths=1.5, depthshade=True, zorder=2,
                      label=elem)

        ax.set_xlabel('X (Å)', fontsize=10)
        ax.set_ylabel('Y (Å)', fontsize=10)
        ax.set_zlabel('Z (Å)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Visualization: {filepath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Simple polymer reconstruction')
    parser.add_argument('--n_carbons', type=int, default=3,
                       help='Number of carbon atoms in chain')
    parser.add_argument('--output_dir', type=str, default='polymer_results',
                       help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Simple Polymer Reconstruction")
    print("="*60)

    # Load model
    print("\n[1] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='src/config.yaml'
    )

    # Load config separately for max_bonds
    import yaml
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = device

    # Create polymer chain
    print(f"\n[2] Creating {args.n_carbons}-carbon polyethylene chain...")
    elements, bonds = create_polyethylene_chain(args.n_carbons)

    # Reconstruct 3D structure
    print("\n[3] Reconstructing 3D structure...")
    # Find a good starting atom (carbon with highest degree)
    from collections import defaultdict
    adjacency = defaultdict(list)
    for bond in bonds:
        i, j = bond['i'], bond['j']
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Find carbon with most bonds
    start_atom = max([i for i, e in enumerate(elements) if e == 'C'],
                    key=lambda x: len(adjacency[x]))

    positions, _, _, _, _ = reconstruct_from_center(
        model=model,
        elements=elements,
        bonds_data=bonds,
        element_to_idx=element_vocab,
        device=device,
        max_bonds=config['data']['max_bonds'],
        start_atom=start_atom
    )

    print(f"  Reconstructed {len(positions)} atom positions")

    # Analyze
    analyze_structure(elements, positions, bonds)

    # Export
    print("\n[4] Exporting files...")
    xyz_file = output_dir / 'polyethylene_simple.xyz'
    png_file = output_dir / 'polyethylene_simple.png'

    export_to_xyz(elements, positions, xyz_file,
                 comment=f"Polyethylene chain - {args.n_carbons} carbons")
    visualize_structure(elements, positions, bonds, png_file)

    print("\n" + "="*60)
    print("Reconstruction complete!")
    print(f"Output files in: {output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()

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
import sys
sys.path.insert(0, 'src')
from reconstruction import load_trained_model, reconstruct_from_center


def create_nylon66_chain(n_repeat_units: int) -> tuple:
    """
    Create a simplified nylon 6,6 chain.

    Repeat unit:
        -NH-(CH2)6-NH-C(=O)-(CH2)4-C(=O)-

    Args:
        n_repeat_units: Number of nylon 6,6 repeat units

    Returns:
        elements: List of element symbols
        bonds: List of bond dictionaries
    """
    elements = []
    bonds = []

    atom_idx = 0
    previous_carbonyl_c = None

    for unit in range(n_repeat_units):
        # Amine nitrogen: -NH-
        n1 = atom_idx
        elements.append("N")
        atom_idx += 1

        if previous_carbonyl_c is not None:
            bonds.append({"i": previous_carbonyl_c, "j": n1, "order": 1.0})

        h_n1 = atom_idx
        elements.append("H")
        atom_idx += 1
        bonds.append({"i": n1, "j": h_n1, "order": 1.0})

        # Hexamethylene segment: -(CH2)6-
        ch2_6 = []
        for _ in range(6):
            c = atom_idx
            elements.append("C")
            atom_idx += 1
            ch2_6.append(c)

        bonds.append({"i": n1, "j": ch2_6[0], "order": 1.0})

        for i in range(5):
            bonds.append({"i": ch2_6[i], "j": ch2_6[i + 1], "order": 1.0})

        for c in ch2_6:
            for _ in range(2):
                h = atom_idx
                elements.append("H")
                atom_idx += 1
                bonds.append({"i": c, "j": h, "order": 1.0})

        # Second amine nitrogen: -NH-
        n2 = atom_idx
        elements.append("N")
        atom_idx += 1
        bonds.append({"i": ch2_6[-1], "j": n2, "order": 1.0})

        h_n2 = atom_idx
        elements.append("H")
        atom_idx += 1
        bonds.append({"i": n2, "j": h_n2, "order": 1.0})

        # First amide carbonyl: -NH-C(=O)-
        c_carbonyl_1 = atom_idx
        elements.append("C")
        atom_idx += 1

        o_carbonyl_1 = atom_idx
        elements.append("O")
        atom_idx += 1

        bonds.append({"i": n2, "j": c_carbonyl_1, "order": 1.0})
        bonds.append({"i": c_carbonyl_1, "j": o_carbonyl_1, "order": 2.0})

        # Adipoyl methylene segment: -(CH2)4-
        ch2_4 = []
        for _ in range(4):
            c = atom_idx
            elements.append("C")
            atom_idx += 1
            ch2_4.append(c)

        bonds.append({"i": c_carbonyl_1, "j": ch2_4[0], "order": 1.0})

        for i in range(3):
            bonds.append({"i": ch2_4[i], "j": ch2_4[i + 1], "order": 1.0})

        for c in ch2_4:
            for _ in range(2):
                h = atom_idx
                elements.append("H")
                atom_idx += 1
                bonds.append({"i": c, "j": h, "order": 1.0})

        # Second carbonyl, used to connect to next repeat unit
        c_carbonyl_2 = atom_idx
        elements.append("C")
        atom_idx += 1

        o_carbonyl_2 = atom_idx
        elements.append("O")
        atom_idx += 1

        bonds.append({"i": ch2_4[-1], "j": c_carbonyl_2, "order": 1.0})
        bonds.append({"i": c_carbonyl_2, "j": o_carbonyl_2, "order": 2.0})

        previous_carbonyl_c = c_carbonyl_2

    # Terminal caps, simplified
    h_start = atom_idx
    elements.append("H")
    atom_idx += 1
    bonds.append({"i": 0, "j": h_start, "order": 1.0})

    h_end = atom_idx
    elements.append("H")
    atom_idx += 1
    bonds.append({"i": previous_carbonyl_c, "j": h_end, "order": 1.0})

    print("Created nylon 6,6 chain:")
    print(f"  Repeat units: {n_repeat_units}")
    print(f"  Total atoms: {len(elements)}")
    print(f"  Total bonds: {len(bonds)}")
    print(f"  Repeat unit: -NH-(CH2)6-NH-C(=O)-(CH2)4-C(=O)-")

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


def export_to_xyz(elements, positions, filepath, comment="Nylon 6,6 chain"):
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
        (10, 45, "Perspective View"),
        (30, -70, "Alternate View")
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
        config_path='config/config.yaml'
    )

    # Load config separately for max_bonds
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = device

    # Create polymer chain
    print(f"\n[2] Creating {args.n_carbons}-carbon Nylon 6,6 chain...")
    elements, bonds = create_nylon66_chain(args.n_carbons)

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
    xyz_file = output_dir / 'Nylon 6,6_simple.xyz'
    png_file = output_dir / 'Nylon 6,6_simple.png'

    export_to_xyz(elements, positions, xyz_file,
                 comment=f"Nylon 6,6 chain - {args.n_carbons} carbons")
    visualize_structure(elements, positions, bonds, png_file)

    print("\n" + "="*60)
    print("Reconstruction complete!")
    print(f"Output files in: {output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()

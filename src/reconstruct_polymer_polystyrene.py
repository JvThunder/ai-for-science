#!/usr/bin/env python3
"""
Polystyrene reconstruction.

Repeat unit: -CH2-CH(C6H5)-
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import argparse
from collections import defaultdict
from typing import List, Dict

import sys
sys.path.insert(0, 'src')
from reconstruction import load_trained_model, reconstruct_from_center


def create_polystyrene_chain(n_repeat_units: int) -> tuple:
    """
    Create a polystyrene chain.

    Repeat unit: -CH2-CH(C6H5)-
    Each unit contains:
      - 1 methylene carbon (CH2): 1C + 2H
      - 1 methine carbon (CH):   1C + 1H
      - 1 phenyl ring (C6H5):    6C + 5H
    Total: 8C + 8H = 16 atoms per repeat unit

    Args:
        n_repeat_units: Number of -CH2-CH(Ph)- repeat units

    Returns:
        elements: List of element symbols
        bonds:    List of bond dicts with 'i', 'j', 'order' keys
    """
    elements = []
    bonds = []
    atom_idx = 0
    prev_methine = None  # index of the CH carbon from the previous unit

    for unit in range(n_repeat_units):
        # ── methylene: -CH2- ──────────────────────────────────────────────────
        c_methylene = atom_idx;  elements.append('C');  atom_idx += 1
        h1 = atom_idx;           elements.append('H');  atom_idx += 1
        h2 = atom_idx;           elements.append('H');  atom_idx += 1

        bonds.append({'i': c_methylene, 'j': h1, 'order': 1.0})
        bonds.append({'i': c_methylene, 'j': h2, 'order': 1.0})

        # backbone bond: previous CH → this CH2
        if prev_methine is not None:
            bonds.append({'i': prev_methine, 'j': c_methylene, 'order': 1.0})

        # ── methine: -CH(Ph)- ─────────────────────────────────────────────────
        c_methine = atom_idx;  elements.append('C');  atom_idx += 1
        h3 = atom_idx;         elements.append('H');  atom_idx += 1

        bonds.append({'i': c_methine, 'j': h3,         'order': 1.0})
        bonds.append({'i': c_methylene, 'j': c_methine, 'order': 1.0})

        # ── phenyl ring: C6H5 ─────────────────────────────────────────────────
        # ph[0] = ipso (bonded to c_methine, no H)
        # ph[1..5] = ortho, meta, para, meta, ortho (each carries one H)
        ph = []
        for _ in range(6):
            c = atom_idx;  elements.append('C');  atom_idx += 1
            ph.append(c)

        # aromatic C-C ring bonds
        for k in range(6):
            bonds.append({'i': ph[k], 'j': ph[(k + 1) % 6], 'order': 1.5})

        # ipso bond to backbone methine
        bonds.append({'i': c_methine, 'j': ph[0], 'order': 1.0})

        # H on ortho/meta/para positions (all except ipso)
        for k in range(1, 6):
            h = atom_idx;  elements.append('H');  atom_idx += 1
            bonds.append({'i': ph[k], 'j': h, 'order': 1.0})

        prev_methine = c_methine

    # ── terminal H caps ───────────────────────────────────────────────────────
    # head: extra H on the first CH2 (atom 0)
    h_head = atom_idx;  elements.append('H');  atom_idx += 1
    bonds.append({'i': 0, 'j': h_head, 'order': 1.0})

    # tail: extra H on the last CH
    h_tail = atom_idx;  elements.append('H');  atom_idx += 1
    bonds.append({'i': prev_methine, 'j': h_tail, 'order': 1.0})

    print("Created polystyrene chain:")
    print(f"  Repeat units : {n_repeat_units}")
    print(f"  Total atoms  : {len(elements)}")
    print(f"  Total bonds  : {len(bonds)}")
    print(f"  Repeat unit  : -CH2-CH(C6H5)-")

    return elements, bonds


def analyze_structure(elements, positions, bonds):
    """Print bond-length statistics for backbone C-C, aromatic C-C, and C-H."""
    backbone_cc = []
    aromatic_cc = []
    ch_lengths  = []

    for bond in bonds:
        i, j = bond['i'], bond['j']
        dist = np.linalg.norm(positions[i] - positions[j])
        ei, ej = elements[i], elements[j]

        if ei == 'C' and ej == 'C':
            if abs(bond['order'] - 1.5) < 0.1:
                aromatic_cc.append(dist)
            else:
                backbone_cc.append(dist)
        elif (ei == 'C' and ej == 'H') or (ei == 'H' and ej == 'C'):
            ch_lengths.append(dist)

    print("\n[Bond Length Analysis]")
    for label, lengths, expected in [
        ("C-C backbone",  backbone_cc,  "~1.54 Å"),
        ("C-C aromatic",  aromatic_cc,  "~1.40 Å"),
        ("C-H",           ch_lengths,   "~1.09 Å"),
    ]:
        if lengths:
            print(f"  {label}: {len(lengths)} bonds")
            print(f"    Mean:  {np.mean(lengths):.3f} Å  (expected {expected})")
            print(f"    Std:   {np.std(lengths):.3f} Å")
            print(f"    Range: [{np.min(lengths):.3f}, {np.max(lengths):.3f}] Å")


def export_to_xyz(elements, positions, filepath, comment="Polystyrene chain"):
    """Write structure to XYZ file."""
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n")
        f.write(f"{comment}\n")
        for elem, pos in zip(elements, positions):
            f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    print(f"  Exported XYZ: {filepath}")


def visualize_structure(elements, positions, bonds, filepath):
    """Save a two-view 3D plot of the structure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    element_colors = {'H': '#FFFFFF', 'C': '#909090'}
    element_sizes  = {'H': 100, 'C': 300}

    fig = plt.figure(figsize=(16, 8))

    for plot_idx, (elev, azim, title) in enumerate([
        (15, 45,  "Perspective View"),
        (30, -70, "Alternate View"),
    ], 1):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')

        for bond in bonds:
            i, j = bond['i'], bond['j']
            ei, ej = elements[i], elements[j]
            if bond['order'] == 1.5:
                color, lw = '#2060C0', 2.0   # aromatic — blue
            elif ei == 'C' and ej == 'C':
                color, lw = '#404040', 3.5   # backbone — dark grey
            else:
                color, lw = '#909090', 1.5   # C-H — light grey

            ax.plot([positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]],
                    color=color, linewidth=lw, zorder=1)

        for elem in set(elements):
            idxs = [k for k, e in enumerate(elements) if e == elem]
            ax.scatter(positions[idxs, 0], positions[idxs, 1], positions[idxs, 2],
                       c=element_colors.get(elem, '#FF1493'),
                       s=element_sizes.get(elem, 200),
                       edgecolors='black', linewidths=1.0,
                       depthshade=True, zorder=2, label=elem)

        ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_zlabel('Z (Å)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Visualization: {filepath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Polystyrene 3D reconstruction')
    parser.add_argument('--n_repeat_units', type=int, default=5,
                        help='Number of -CH2-CH(Ph)- repeat units (default: 5)')
    parser.add_argument('--output_dir', type=str, default='polymer_results',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Polystyrene Reconstruction")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='config/config.yaml'
    )

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n[2] Creating polystyrene chain ({args.n_repeat_units} repeat units)...")
    elements, bonds = create_polystyrene_chain(args.n_repeat_units)

    print("\n[3] Reconstructing 3D structure...")
    adjacency = defaultdict(list)
    for bond in bonds:
        adjacency[bond['i']].append(bond['j'])
        adjacency[bond['j']].append(bond['i'])

    # Start from the backbone methine carbon (highest C-degree = 3 bonds to C)
    start_atom = max(
        [i for i, e in enumerate(elements) if e == 'C'],
        key=lambda i: sum(1 for n in adjacency[i] if elements[n] == 'C')
    )

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

    analyze_structure(elements, positions, bonds)

    print("\n[4] Exporting files...")
    export_to_xyz(elements, positions,
                  output_dir / 'polystyrene.xyz',
                  comment=f"Polystyrene - {args.n_repeat_units} repeat units")
    visualize_structure(elements, positions, bonds,
                        output_dir / 'polystyrene.png')

    print("\n" + "=" * 60)
    print("Reconstruction complete!")
    print(f"Output files in: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

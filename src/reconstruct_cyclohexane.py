#!/usr/bin/env python3
"""
Cyclohexane reconstruction.

Molecule: C6H12 — six-membered saturated ring.
Good test case for ring closure enforcement.
"""

import numpy as np
import yaml
from pathlib import Path
import argparse
from collections import defaultdict

import sys
sys.path.insert(0, 'src')
from reconstruction import load_trained_model, reconstruct_from_center


def create_cyclohexane() -> tuple:
    """
    Build cyclohexane: C6H12.

    Atom layout (18 atoms total):
      For each ring carbon k (k = 0..5):
        atom 3k+0 : C  (ring)
        atom 3k+1 : H
        atom 3k+2 : H

    Bonds:
      C-H  single (×12)
      C-C  ring single (×6, including the ring-closing C5-C0 bond)
    """
    elements = []
    bonds = []
    atom_idx = 0
    c_indices = []

    for k in range(6):
        c = atom_idx;  elements.append('C');  atom_idx += 1
        h1 = atom_idx; elements.append('H');  atom_idx += 1
        h2 = atom_idx; elements.append('H');  atom_idx += 1

        bonds.append({'i': c, 'j': h1, 'order': 1.0})
        bonds.append({'i': c, 'j': h2, 'order': 1.0})
        c_indices.append(c)

    # Ring C-C bonds (including closure C5 → C0)
    for k in range(6):
        bonds.append({'i': c_indices[k], 'j': c_indices[(k + 1) % 6], 'order': 1.0})

    print("Created cyclohexane (C6H12):")
    print(f"  Atoms : {len(elements)}  (6C + 12H)")
    print(f"  Bonds : {len(bonds)}  (6 C-C ring + 12 C-H)")

    return elements, bonds


def analyze_structure(elements, positions, bonds):
    cc, ch = [], []
    for bond in bonds:
        i, j = bond['i'], bond['j']
        d = np.linalg.norm(positions[i] - positions[j])
        ei, ej = elements[i], elements[j]
        if ei == 'C' and ej == 'C':
            cc.append(d)
        else:
            ch.append(d)

    print("\n[Bond Length Analysis]")
    for label, lengths, expected in [
        ("C-C ring", cc, "~1.54 Å"),
        ("C-H",      ch, "~1.09 Å"),
    ]:
        if lengths:
            print(f"  {label}: {len(lengths)} bonds")
            print(f"    Mean:  {np.mean(lengths):.3f} Å  (expected {expected})")
            print(f"    Std:   {np.std(lengths):.3f} Å")
            print(f"    Range: [{np.min(lengths):.3f}, {np.max(lengths):.3f}] Å")


def export_to_xyz(elements, positions, filepath, comment="Cyclohexane"):
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for elem, pos in zip(elements, positions):
            f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    print(f"  Exported XYZ: {filepath}")


def visualize_structure(elements, positions, bonds, filepath):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    element_colors = {'H': '#FFFFFF', 'C': '#909090'}
    element_sizes  = {'H': 100, 'C': 350}

    fig = plt.figure(figsize=(14, 6))
    for plot_idx, (elev, azim, title) in enumerate([
        (20, 30,  "Side view"),
        (80, 30,  "Top view (ring)"),
    ], 1):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')

        for bond in bonds:
            i, j = bond['i'], bond['j']
            ei, ej = elements[i], elements[j]
            color = '#404040' if (ei == 'C' and ej == 'C') else '#a0a0a0'
            lw    = 3.0       if (ei == 'C' and ej == 'C') else 1.5
            ax.plot([positions[i,0], positions[j,0]],
                    [positions[i,1], positions[j,1]],
                    [positions[i,2], positions[j,2]],
                    color=color, linewidth=lw, zorder=1)

        for elem in set(elements):
            idxs = [k for k, e in enumerate(elements) if e == elem]
            ax.scatter(positions[idxs,0], positions[idxs,1], positions[idxs,2],
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
    parser = argparse.ArgumentParser(description='Cyclohexane 3D reconstruction')
    parser.add_argument('--output_dir', type=str, default='polymer_results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Cyclohexane Reconstruction")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='config/config.yaml'
    )
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n[2] Building cyclohexane topology...")
    elements, bonds = create_cyclohexane()

    print("\n[3] Reconstructing 3D structure...")
    adjacency = defaultdict(list)
    for bond in bonds:
        adjacency[bond['i']].append(bond['j'])
        adjacency[bond['j']].append(bond['i'])

    # Any ring carbon is a good start (all have degree 4)
    start_atom = next(i for i, e in enumerate(elements) if e == 'C')

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
                  output_dir / 'cyclohexane.xyz',
                  comment="Cyclohexane C6H12")
    visualize_structure(elements, positions, bonds,
                        output_dir / 'cyclohexane.png')

    print("\n" + "=" * 60)
    print("Reconstruction complete!")
    print(f"Output files in: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

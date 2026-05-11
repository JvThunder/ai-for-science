#!/usr/bin/env python3
"""
Acetylsalicylic acid (aspirin) reconstruction.

Molecule: C9H8O4 — benzene ring with carboxylic acid and acetyloxy substituents.
Good test case for mixed aromatic/polar bonds and multiple functional groups.
"""

import numpy as np
import yaml
from pathlib import Path
import argparse
from collections import defaultdict

import sys
sys.path.insert(0, 'src')
from reconstruction import load_trained_model, reconstruct_from_center


def create_acetylsalicylic_acid() -> tuple:
    """
    Build acetylsalicylic acid (aspirin): C9H8O4.

    Atom layout (21 atoms total):
       0: C  (ring C0 — bears -COOH at position 1)
       1: C  (ring C1 — bears -O-CO-CH3 at position 2)
       2: C  (ring C2 — aromatic CH)
       3: C  (ring C3 — aromatic CH)
       4: C  (ring C4 — aromatic CH)
       5: C  (ring C5 — aromatic CH)
       6: H  (on C2)
       7: H  (on C3)
       8: H  (on C4)
       9: H  (on C5)
      10: C  (COOH carbonyl carbon)
      11: O  (COOH C=O oxygen)
      12: O  (COOH O-H oxygen)
      13: H  (COOH hydroxyl H)
      14: O  (ester oxygen linking ring to acetyl)
      15: C  (acetyl carbonyl carbon)
      16: O  (acetyl C=O oxygen)
      17: C  (methyl carbon)
      18: H  (methyl H)
      19: H  (methyl H)
      20: H  (methyl H)

    Bonds:
      C-C aromatic (×6, ring, order 1.5)
      C-H single   (×7, 4 ring + 3 methyl)
      C-C single   (×2, C0–C10 COOH, C15–C17 methyl)
      C=O double   (×2, C10–O11 carboxyl, C15–O16 acetyl)
      C-O single   (×3, C10–O12, C1–O14, O14–C15)
      O-H single   (×1, O12–H13)
    """
    elements = [
        'C', 'C', 'C', 'C', 'C', 'C',   # ring C0–C5
        'H', 'H', 'H', 'H',              # aromatic H on C2–C5
        'C', 'O', 'O', 'H',              # COOH: C10, O11, O12, H13
        'O', 'C', 'O', 'C',              # ester/acetyl: O14, C15, O16, C17
        'H', 'H', 'H',                   # methyl H: H18–H20
    ]

    bonds = [
        # Benzene ring (aromatic, order 1.5)
        {'i': 0, 'j': 1, 'order': 1.5},
        {'i': 1, 'j': 2, 'order': 1.5},
        {'i': 2, 'j': 3, 'order': 1.5},
        {'i': 3, 'j': 4, 'order': 1.5},
        {'i': 4, 'j': 5, 'order': 1.5},
        {'i': 5, 'j': 0, 'order': 1.5},

        # Aromatic C-H (C2–C5)
        {'i': 2, 'j': 6,  'order': 1.0},
        {'i': 3, 'j': 7,  'order': 1.0},
        {'i': 4, 'j': 8,  'order': 1.0},
        {'i': 5, 'j': 9,  'order': 1.0},

        # Carboxylic acid (-COOH) at C0
        {'i': 0,  'j': 10, 'order': 1.0},  # C-C single
        {'i': 10, 'j': 11, 'order': 2.0},  # C=O double
        {'i': 10, 'j': 12, 'order': 1.0},  # C-O single
        {'i': 12, 'j': 13, 'order': 1.0},  # O-H

        # Acetyloxy (-O-CO-CH3) at C1
        {'i': 1,  'j': 14, 'order': 1.0},  # C-O single (ring to ester O)
        {'i': 14, 'j': 15, 'order': 1.0},  # O-C single (ester O to acetyl C)
        {'i': 15, 'j': 16, 'order': 2.0},  # C=O double (acetyl carbonyl)
        {'i': 15, 'j': 17, 'order': 1.0},  # C-C single (acetyl to methyl)

        # Methyl C-H
        {'i': 17, 'j': 18, 'order': 1.0},
        {'i': 17, 'j': 19, 'order': 1.0},
        {'i': 17, 'j': 20, 'order': 1.0},
    ]

    print("Created acetylsalicylic acid (C9H8O4):")
    print(f"  Atoms : {len(elements)}  (9C + 8H + 4O)")
    print(f"  Bonds : {len(bonds)}  (6 aromatic C-C + 7 C-H + 2 C-C + 2 C=O + 3 C-O + 1 O-H)")

    return elements, bonds


def analyze_structure(elements, positions, bonds):
    categories = {
        'C-C aromatic': [],
        'C-C single':   [],
        'C-H':          [],
        'C=O':          [],
        'C-O single':   [],
        'O-H':          [],
    }

    for bond in bonds:
        i, j = bond['i'], bond['j']
        ei, ej = elements[i], elements[j]
        order = bond['order']
        d = np.linalg.norm(positions[i] - positions[j])
        pair = tuple(sorted([ei, ej]))

        if pair == ('C', 'C') and order == 1.5:
            categories['C-C aromatic'].append(d)
        elif pair == ('C', 'C') and order == 1.0:
            categories['C-C single'].append(d)
        elif pair == ('C', 'H'):
            categories['C-H'].append(d)
        elif pair == ('C', 'O') and order == 2.0:
            categories['C=O'].append(d)
        elif pair == ('C', 'O') and order == 1.0:
            categories['C-O single'].append(d)
        elif pair == ('H', 'O'):
            categories['O-H'].append(d)

    expected = {
        'C-C aromatic': '~1.40 Å',
        'C-C single':   '~1.52 Å',
        'C-H':          '~1.09 Å',
        'C=O':          '~1.22 Å',
        'C-O single':   '~1.36 Å',
        'O-H':          '~0.97 Å',
    }

    print("\n[Bond Length Analysis]")
    for label, lengths in categories.items():
        if lengths:
            print(f"  {label}: {len(lengths)} bonds")
            print(f"    Mean:  {np.mean(lengths):.3f} Å  (expected {expected[label]})")
            print(f"    Std:   {np.std(lengths):.3f} Å")
            print(f"    Range: [{np.min(lengths):.3f}, {np.max(lengths):.3f}] Å")


def export_to_xyz(elements, positions, filepath, comment="Acetylsalicylic acid"):
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for elem, pos in zip(elements, positions):
            f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    print(f"  Exported XYZ: {filepath}")


def visualize_structure(elements, positions, bonds, filepath):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    element_colors = {'H': '#FFFFFF', 'C': '#909090', 'O': '#FF4444'}
    element_sizes  = {'H': 100, 'C': 350, 'O': 300}

    fig = plt.figure(figsize=(14, 6))
    for plot_idx, (elev, azim, title) in enumerate([
        (20, 30, "Side view"),
        (80, 10, "Top view (ring)"),
    ], 1):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')

        for bond in bonds:
            i, j = bond['i'], bond['j']
            ei, ej = elements[i], elements[j]
            order = bond['order']
            pair = tuple(sorted([ei, ej]))

            if pair == ('C', 'C') and order == 1.5:
                color, lw = '#404040', 3.0
            elif pair == ('C', 'C'):
                color, lw = '#606060', 2.5
            elif 'O' in pair:
                color, lw = '#FF8888', 2.0
            else:
                color, lw = '#a0a0a0', 1.5

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
    parser = argparse.ArgumentParser(description='Acetylsalicylic acid 3D reconstruction')
    parser.add_argument('--output_dir', type=str, default='polymer_results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Acetylsalicylic Acid (Aspirin) Reconstruction")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='config/config.yaml'
    )
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n[2] Building acetylsalicylic acid topology...")
    elements, bonds = create_acetylsalicylic_acid()

    print("\n[3] Reconstructing 3D structure...")
    adjacency = defaultdict(list)
    for bond in bonds:
        adjacency[bond['i']].append(bond['j'])
        adjacency[bond['j']].append(bond['i'])

    # C0 is the ring carbon bearing COOH — highest connectivity, good start
    start_atom = 0

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
                  output_dir / 'acetylsalicylic_acid.xyz',
                  comment="Acetylsalicylic acid C9H8O4 (Aspirin)")
    visualize_structure(elements, positions, bonds,
                        output_dir / 'acetylsalicylic_acid.png')

    print("\n" + "=" * 60)
    print("Reconstruction complete!")
    print(f"Output files in: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

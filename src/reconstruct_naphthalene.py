#!/usr/bin/env python3
"""
Naphthalene (C10H8) reconstruction — angle stress test.

Two fused aromatic rings. Expected geometry:
  - All C-C-C angles   ~120°  (sp2 throughout)
  - C-C aromatic bonds ~1.40 Å
  - C-H bonds          ~1.08 Å
  - All heavy atoms coplanar (max out-of-plane deviation ~0 Å)

Harder than cyclohexane because:
  - Both rings must close simultaneously
  - Shared bond C4-C5 is constrained by BOTH ring optimisers
  - 120° (not 109.47°) angles must come from model predictions alone
"""

import math
import sys
from collections import defaultdict
from pathlib import Path
import argparse

import numpy as np
import yaml

sys.path.insert(0, 'src')
from reconstruction import load_trained_model, reconstruct_from_center


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def create_naphthalene():
    """
    Atom indices:
      C0-C1-C2-C3-C4-C5  ring 1 (C4, C5 are junction atoms, no H)
      C5-C6-C7-C8-C9-C4  ring 2 (shares bond C4-C5 with ring 1)
      H10-H17             one H on each of C0-C3, C6-C9

    All C-C bonds are aromatic (order 1.5).
    """
    elements = ['C'] * 10 + ['H'] * 8

    bonds = []
    # Ring 1: C0-C1-C2-C3-C4-C5 (closes via C5-C0)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
        bonds.append({'i': a, 'j': b, 'order': 1.5})
    # Ring 2: C5-C6-C7-C8-C9-C4 (closes via C9-C4; shares C4-C5 with ring 1)
    for a, b in [(5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]:
        bonds.append({'i': a, 'j': b, 'order': 1.5})
    # C-H bonds (non-junction carbons only)
    for c_idx, h_idx in [(0, 10), (1, 11), (2, 12), (3, 13),
                          (6, 14), (7, 15), (8, 16), (9, 17)]:
        bonds.append({'i': c_idx, 'j': h_idx, 'order': 1.0})

    print("Created naphthalene (C10H8):")
    print(f"  Atoms : {len(elements)}  (10C + 8H)")
    print(f"  Bonds : {len(bonds)}  (11 C-C aromatic + 8 C-H)")
    return elements, bonds


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_structure(elements, positions, bonds):
    adjacency = defaultdict(list)
    for bond in bonds:
        i, j = bond['i'], bond['j']
        adjacency[i].append(j)
        adjacency[j].append(i)

    cc, ch = [], []
    for bond in bonds:
        i, j = bond['i'], bond['j']
        d = np.linalg.norm(positions[i] - positions[j])
        if elements[i] == 'C' and elements[j] == 'C':
            cc.append(d)
        else:
            ch.append(d)

    print("\n[Bond Length Analysis]")
    for label, lengths, expected in [
        ("C-C aromatic", cc, "~1.40 Å"),
        ("C-H",          ch, "~1.08 Å"),
    ]:
        if lengths:
            arr = np.array(lengths)
            print(f"  {label}: {len(arr)} bonds")
            print(f"    Mean:  {arr.mean():.3f} Å  (expected {expected})")
            print(f"    Std:   {arr.std():.3f} Å")
            print(f"    Range: [{arr.min():.3f}, {arr.max():.3f}] Å")

    print("\n[Bond Angle Analysis — C-C-C at ring carbons, expected ~120°]")
    angles = []
    for center in range(10):
        cc_nbrs = [n for n in adjacency[center] if elements[n] == 'C']
        for i in range(len(cc_nbrs)):
            for j in range(i + 1, len(cc_nbrs)):
                va = positions[cc_nbrs[i]] - positions[center]
                vb = positions[cc_nbrs[j]] - positions[center]
                cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
                angles.append(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))
    if angles:
        arr = np.array(angles)
        print(f"  Count: {len(arr)}")
        print(f"  Mean:  {arr.mean():.2f}°  (expected 120.00°)")
        print(f"  Std:   {arr.std():.2f}°")
        print(f"  Range: [{arr.min():.2f}°, {arr.max():.2f}°]")

    print("\n[Planarity — ring carbons, expected ~0 Å out-of-plane]")
    ring_pos = positions[:10]
    centroid = ring_pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(ring_pos - centroid)
    normal = Vt[-1]
    deviations = np.abs((ring_pos - centroid) @ normal)
    print(f"  Max deviation : {deviations.max():.3f} Å")
    print(f"  Mean deviation: {deviations.mean():.3f} Å")


# ---------------------------------------------------------------------------
# Export / visualise
# ---------------------------------------------------------------------------

def export_xyz(elements, positions, filepath, comment="Naphthalene C10H8"):
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for elem, pos in zip(elements, positions):
            f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    print(f"  Exported XYZ : {filepath}")


def visualize(elements, positions, bonds, filepath):
    import matplotlib.pyplot as plt

    element_colors = {'H': '#FFFFFF', 'C': '#909090'}
    element_sizes  = {'H': 100,       'C': 350}

    fig = plt.figure(figsize=(14, 6))
    for plot_idx, (elev, azim, title) in enumerate([
        (20, 30, "Side view"),
        (80, 30, "Top view (rings)"),
    ], 1):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')

        for bond in bonds:
            i, j = bond['i'], bond['j']
            ei, ej = elements[i], elements[j]
            color = '#404040' if (ei == 'C' and ej == 'C') else '#a0a0a0'
            lw    = 3.0       if (ei == 'C' and ej == 'C') else 1.5
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Naphthalene 3D reconstruction')
    parser.add_argument('--output_dir', type=str, default='polymer_results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Naphthalene Reconstruction (C10H8)")
    print("Angle stress test — two fused aromatic rings")
    print("=" * 60)

    print("\n[1] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='config/config.yaml'
    )
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    print("\n[2] Building naphthalene topology...")
    elements, bonds = create_naphthalene()

    print("\n[3] Reconstructing 3D structure...")
    # Start from a junction carbon (C4 or C5, each bonded to 3 ring carbons)
    positions, _, _, _, _ = reconstruct_from_center(
        model=model,
        elements=elements,
        bonds_data=bonds,
        element_to_idx=element_vocab,
        device=device,
        max_bonds=config['data']['max_bonds'],
        start_atom=4,
    )

    print(f"  Reconstructed {len(positions)} atom positions")
    analyze_structure(elements, positions, bonds)

    print("\n[4] Exporting files...")
    export_xyz(elements, positions, output_dir / 'naphthalene.xyz')
    visualize(elements, positions, bonds, output_dir / 'naphthalene.png')

    print("\n" + "=" * 60)
    print("Done!  naphthalene.xyz / naphthalene.png")
    print(f"Output in: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

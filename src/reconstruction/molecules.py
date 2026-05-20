#!/usr/bin/env python3
"""
OOP molecule topology and reconstruction classes.

Each class encapsulates:
  - create_topology(*args) — @staticmethod, returns (elements, bonds); call directly
                             from benchmarks without instantiating the class.
  - build_topology()       — instance method that delegates to create_topology using
                             instance parameters set by the CLI (via main()).
  - choose_start_atom()    — returns the atom index to seed reconstruction.
  - analyze()              — prints bond-length statistics; override for extra checks.
  - bond_style()           — returns (color, linewidth) for visualisation; override
                             to customise per-molecule bond colours.
  - run()                  — template method: load → build → reconstruct → export.
  - main()                 — CLI entry point (parses args, calls run()).
"""

import math
import sys
import argparse
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, 'src')
from reconstruction.reconstruction import load_trained_model, reconstruct_from_center

_ELEMENT_COLORS = {
    'H': '#FFFFFF',
    'C': '#909090',
    'O': '#FF4444',
    'F': '#00CC44',
    'N': '#3050F8',
}
_ELEMENT_SIZES = {'H': 100, 'C': 350, 'O': 300, 'F': 280}


# =============================================================================
# Base class
# =============================================================================

class MoleculeReconstructor(ABC):
    """
    Template-method base for single-molecule 3D reconstruction scripts.

    Subclasses must implement build_topology().
    Override choose_start_atom(), analyze(), bond_style(), add_args(), or
    apply_args() as needed.
    """

    name: str = "Molecule"
    output_stem: str = "molecule"

    @abstractmethod
    def build_topology(self) -> tuple:
        """Return (elements: list[str], bonds: list[dict]) for the molecule."""

    def choose_start_atom(self, elements: list, bonds: list) -> int:
        """Return the atom index to seed reconstruction from.

        Default: heavy atom with the highest bond degree.
        """
        adjacency = defaultdict(list)
        for b in bonds:
            adjacency[b['i']].append(b['j'])
            adjacency[b['j']].append(b['i'])
        candidates = [i for i, e in enumerate(elements) if e != 'H']
        if not candidates:
            candidates = list(range(len(elements)))
        return max(candidates, key=lambda x: len(adjacency[x]))

    def analyze(self, elements: list, positions: np.ndarray, bonds: list) -> None:
        """Print bond-length statistics grouped by element pair and bond order."""
        groups = defaultdict(list)
        for bond in bonds:
            i, j = bond['i'], bond['j']
            ei, ej = sorted([elements[i], elements[j]])
            order = bond['order']
            if order == 2.0:
                label = f"{ei}={ej}"
            elif abs(order - 1.5) < 0.1:
                label = f"{ei}~{ej} (aromatic)"
            else:
                label = f"{ei}-{ej}"
            groups[label].append(np.linalg.norm(positions[i] - positions[j]))

        print("\n[Bond Length Analysis]")
        for label in sorted(groups):
            arr = np.array(groups[label])
            print(f"  {label}: {len(arr)} bonds")
            print(f"    Mean: {arr.mean():.3f} Å  Std: {arr.std():.3f} Å"
                  f"  Range: [{arr.min():.3f}, {arr.max():.3f}] Å")

    def bond_style(self, elem_i: str, elem_j: str, order: float) -> tuple:
        """Return (color, linewidth) for a bond in the visualisation."""
        if elem_i == 'C' and elem_j == 'C':
            return '#404040', 3.0
        return '#a0a0a0', 1.5

    def export_xyz(self, elements: list, positions: np.ndarray,
                   filepath: Path, comment: str = None) -> None:
        comment = comment or self.name
        with open(filepath, 'w') as f:
            f.write(f"{len(elements)}\n{comment}\n")
            for elem, pos in zip(elements, positions):
                f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        print(f"  Exported XYZ: {filepath}")

    def visualize(self, elements: list, positions: np.ndarray,
                  bonds: list, filepath: Path) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 6))
        for plot_idx, (elev, azim, title) in enumerate([
            (20, 30, "Perspective view"),
            (80, 30, "Top view"),
        ], 1):
            ax = fig.add_subplot(1, 2, plot_idx, projection='3d')

            for bond in bonds:
                i, j = bond['i'], bond['j']
                color, lw = self.bond_style(elements[i], elements[j], bond['order'])
                ax.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        color=color, linewidth=lw, zorder=1)

            for elem in set(elements):
                idxs = [k for k, e in enumerate(elements) if e == elem]
                ax.scatter(
                    positions[idxs, 0], positions[idxs, 1], positions[idxs, 2],
                    c=_ELEMENT_COLORS.get(elem, '#FF1493'),
                    s=_ELEMENT_SIZES.get(elem, 200),
                    edgecolors='black', linewidths=1.0,
                    depthshade=True, zorder=2, label=elem,
                )

            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Visualization: {filepath}")
        plt.close()

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Subclasses add molecule-specific CLI arguments here."""

    def apply_args(self, args: argparse.Namespace) -> None:
        """Subclasses copy parsed args into instance attributes here."""

    def run(self, output_dir,
            checkpoint_path: str = 'checkpoints/best_model.pt',
            config_path: str = 'config/config.yaml') -> None:
        """Load model → build topology → reconstruct → analyze → export."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("=" * 60)
        print(f"{self.name} Reconstruction")
        print("=" * 60)

        print("\n[1] Loading model...")
        model, element_vocab, device = load_trained_model(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"\n[2] Building {self.name} topology...")
        elements, bonds = self.build_topology()

        print("\n[3] Reconstructing 3D structure...")
        start_atom = self.choose_start_atom(elements, bonds)
        positions, _, _, _, _ = reconstruct_from_center(
            model=model,
            elements=elements,
            bonds_data=bonds,
            element_to_idx=element_vocab,
            device=device,
            max_bonds=config['data']['max_bonds'],
            start_atom=start_atom,
        )
        print(f"  Reconstructed {len(positions)} atom positions")
        self.analyze(elements, positions, bonds)

        print("\n[4] Exporting files...")
        self.export_xyz(elements, positions, output_dir / f"{self.output_stem}.xyz")
        self.visualize(elements, positions, bonds, output_dir / f"{self.output_stem}.png")

        print("\n" + "=" * 60)
        print("Reconstruction complete!")
        print(f"Output files in: {output_dir}/")
        print("=" * 60)

    def main(self) -> None:
        parser = argparse.ArgumentParser(description=f'{self.name} 3D reconstruction')
        parser.add_argument('--output_dir', default='polymer_results')
        parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
        parser.add_argument('--config', default='config/config.yaml')
        self.add_args(parser)
        args = parser.parse_args()
        self.apply_args(args)
        self.run(args.output_dir, args.checkpoint, args.config)


# =============================================================================
# Small molecules
# =============================================================================

class Cyclohexane(MoleculeReconstructor):
    """C6H12 — six-membered saturated ring. Tests ring-closure enforcement."""

    name = "Cyclohexane"
    output_stem = "cyclohexane"

    @staticmethod
    def create_topology() -> tuple:
        elements, bonds = [], []
        atom_idx = 0
        c_indices = []
        for _ in range(6):
            c = atom_idx;  elements.append('C'); atom_idx += 1
            h1 = atom_idx; elements.append('H'); atom_idx += 1
            h2 = atom_idx; elements.append('H'); atom_idx += 1
            bonds.append({'i': c, 'j': h1, 'order': 1.0})
            bonds.append({'i': c, 'j': h2, 'order': 1.0})
            c_indices.append(c)
        for k in range(6):
            bonds.append({'i': c_indices[k], 'j': c_indices[(k + 1) % 6], 'order': 1.0})
        print(f"Created cyclohexane (C6H12): {len(elements)} atoms, {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology()


class Naphthalene(MoleculeReconstructor):
    """C10H8 — two fused aromatic rings. Angle + planarity stress test."""

    name = "Naphthalene"
    output_stem = "naphthalene"

    @staticmethod
    def create_topology() -> tuple:
        elements = ['C'] * 10 + ['H'] * 8
        bonds = []
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
            bonds.append({'i': a, 'j': b, 'order': 1.5})
        for a, b in [(5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]:
            bonds.append({'i': a, 'j': b, 'order': 1.5})
        for c_idx, h_idx in [(0, 10), (1, 11), (2, 12), (3, 13),
                              (6, 14), (7, 15), (8, 16), (9, 17)]:
            bonds.append({'i': c_idx, 'j': h_idx, 'order': 1.0})
        print(f"Created naphthalene (C10H8): {len(elements)} atoms, {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology()

    def choose_start_atom(self, elements, bonds) -> int:
        return 4  # junction carbon bonded to 3 ring carbons

    def analyze(self, elements, positions, bonds) -> None:
        super().analyze(elements, positions, bonds)

        adjacency = defaultdict(list)
        for bond in bonds:
            adjacency[bond['i']].append(bond['j'])
            adjacency[bond['j']].append(bond['i'])

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
            print(f"  Count: {len(arr)}  Mean: {arr.mean():.2f}°  Std: {arr.std():.2f}°"
                  f"  Range: [{arr.min():.2f}°, {arr.max():.2f}°]  (expected 120°)")

        print("\n[Planarity — ring carbons, expected ~0 Å out-of-plane]")
        ring_pos = positions[:10]
        centroid = ring_pos.mean(axis=0)
        _, _, Vt = np.linalg.svd(ring_pos - centroid)
        normal = Vt[-1]
        deviations = np.abs((ring_pos - centroid) @ normal)
        print(f"  Max deviation: {deviations.max():.3f} Å  Mean: {deviations.mean():.3f} Å")


class AcetylsalicylicAcid(MoleculeReconstructor):
    """C9H8O4 (Aspirin) — aromatic ring with carboxylic acid + acetyloxy groups."""

    name = "Acetylsalicylic Acid (Aspirin)"
    output_stem = "acetylsalicylic_acid"

    @staticmethod
    def create_topology() -> tuple:
        elements = [
            'C', 'C', 'C', 'C', 'C', 'C',   # ring C0–C5
            'H', 'H', 'H', 'H',              # aromatic H on C2–C5
            'C', 'O', 'O', 'H',              # COOH: C10, O11, O12, H13
            'O', 'C', 'O', 'C',              # ester/acetyl: O14, C15, O16, C17
            'H', 'H', 'H',                   # methyl H: H18–H20
        ]
        bonds = [
            {'i': 0, 'j': 1, 'order': 1.5}, {'i': 1, 'j': 2, 'order': 1.5},
            {'i': 2, 'j': 3, 'order': 1.5}, {'i': 3, 'j': 4, 'order': 1.5},
            {'i': 4, 'j': 5, 'order': 1.5}, {'i': 5, 'j': 0, 'order': 1.5},
            {'i': 2, 'j':  6, 'order': 1.0}, {'i': 3, 'j':  7, 'order': 1.0},
            {'i': 4, 'j':  8, 'order': 1.0}, {'i': 5, 'j':  9, 'order': 1.0},
            {'i': 0, 'j': 10, 'order': 1.0}, {'i': 10, 'j': 11, 'order': 2.0},
            {'i': 10, 'j': 12, 'order': 1.0}, {'i': 12, 'j': 13, 'order': 1.0},
            {'i': 1, 'j': 14, 'order': 1.0}, {'i': 14, 'j': 15, 'order': 1.0},
            {'i': 15, 'j': 16, 'order': 2.0}, {'i': 15, 'j': 17, 'order': 1.0},
            {'i': 17, 'j': 18, 'order': 1.0}, {'i': 17, 'j': 19, 'order': 1.0},
            {'i': 17, 'j': 20, 'order': 1.0},
        ]
        print(f"Created acetylsalicylic acid (C9H8O4): {len(elements)} atoms, {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology()

    def choose_start_atom(self, elements, bonds) -> int:
        return 0  # ring carbon C0 bearing COOH — well-connected start


# =============================================================================
# Polymers
# =============================================================================

class PEG(MoleculeReconstructor):
    """Polyethylene glycol — repeat unit: -CH2-CH2-O-"""

    name = "PEG"
    output_stem = "peg"

    def __init__(self, n_repeat_units: int = 10):
        self.n_repeat_units = n_repeat_units

    def add_args(self, parser) -> None:
        parser.add_argument('--n_repeat_units', type=int, default=self.n_repeat_units,
                            help='Number of -CH2-CH2-O- repeat units')

    def apply_args(self, args) -> None:
        self.n_repeat_units = args.n_repeat_units

    @staticmethod
    def create_topology(n_repeat_units: int) -> tuple:
        elements, bonds = [], []
        atom_idx = 0
        previous_o = None
        for _ in range(n_repeat_units):
            c1 = atom_idx; elements.append('C'); atom_idx += 1
            c2 = atom_idx; elements.append('C'); atom_idx += 1
            o  = atom_idx; elements.append('O'); atom_idx += 1
            if previous_o is not None:
                bonds.append({'i': previous_o, 'j': c1, 'order': 1.0})
            bonds.append({'i': c1, 'j': c2, 'order': 1.0})
            bonds.append({'i': c2, 'j': o,  'order': 1.0})
            for carbon in [c1, c2]:
                for _h in range(2):
                    h = atom_idx; elements.append('H'); atom_idx += 1
                    bonds.append({'i': carbon, 'j': h, 'order': 1.0})
            previous_o = o
        h_start = atom_idx; elements.append('H'); atom_idx += 1
        bonds.append({'i': 0, 'j': h_start, 'order': 1.0})
        h_end = atom_idx; elements.append('H'); atom_idx += 1
        bonds.append({'i': previous_o, 'j': h_end, 'order': 1.0})
        print(f"Created PEG chain ({n_repeat_units} units): {len(elements)} atoms, {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology(self.n_repeat_units)


class Polyester(MoleculeReconstructor):
    """Aliphatic polyester — repeat unit: -O-CH2-CH2-C(=O)-"""

    name = "Polyester"
    output_stem = "polyester"

    def __init__(self, n_repeat_units: int = 10):
        self.n_repeat_units = n_repeat_units

    def add_args(self, parser) -> None:
        parser.add_argument('--n_repeat_units', type=int, default=self.n_repeat_units,
                            help='Number of -O-CH2-CH2-C(=O)- repeat units')

    def apply_args(self, args) -> None:
        self.n_repeat_units = args.n_repeat_units

    @staticmethod
    def create_topology(n_repeat_units: int) -> tuple:
        elements, bonds = [], []
        atom_idx = 0
        previous_carbonyl_c = None
        for _ in range(n_repeat_units):
            o_ether    = atom_idx; elements.append('O'); atom_idx += 1
            c1         = atom_idx; elements.append('C'); atom_idx += 1
            c2         = atom_idx; elements.append('C'); atom_idx += 1
            c_carbonyl = atom_idx; elements.append('C'); atom_idx += 1
            o_carbonyl = atom_idx; elements.append('O'); atom_idx += 1
            if previous_carbonyl_c is not None:
                bonds.append({'i': previous_carbonyl_c, 'j': o_ether, 'order': 1.0})
            bonds.append({'i': o_ether,    'j': c1,         'order': 1.0})
            bonds.append({'i': c1,         'j': c2,         'order': 1.0})
            bonds.append({'i': c2,         'j': c_carbonyl, 'order': 1.0})
            bonds.append({'i': c_carbonyl, 'j': o_carbonyl, 'order': 2.0})
            for carbon in [c1, c2]:
                for _h in range(2):
                    h = atom_idx; elements.append('H'); atom_idx += 1
                    bonds.append({'i': carbon, 'j': h, 'order': 1.0})
            previous_carbonyl_c = c_carbonyl
        h_start = atom_idx; elements.append('H'); atom_idx += 1
        bonds.append({'i': 0, 'j': h_start, 'order': 1.0})
        h_end = atom_idx; elements.append('H'); atom_idx += 1
        bonds.append({'i': previous_carbonyl_c, 'j': h_end, 'order': 1.0})
        print(f"Created polyester chain ({n_repeat_units} units): {len(elements)} atoms, {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology(self.n_repeat_units)


class Polystyrene(MoleculeReconstructor):
    """Polystyrene — repeat unit: -CH2-CH(C6H5)-"""

    name = "Polystyrene"
    output_stem = "polystyrene"

    def __init__(self, n_repeat_units: int = 3):
        self.n_repeat_units = n_repeat_units

    def add_args(self, parser) -> None:
        parser.add_argument('--n_repeat_units', type=int, default=self.n_repeat_units,
                            help='Number of -CH2-CH(C6H5)- repeat units')

    def apply_args(self, args) -> None:
        self.n_repeat_units = args.n_repeat_units

    @staticmethod
    def create_topology(n_repeat_units: int) -> tuple:
        elements, bonds = [], []
        atom_idx = 0
        prev_methine = None
        for _ in range(n_repeat_units):
            # -CH2- (methylene)
            c_methylene = atom_idx; elements.append('C'); atom_idx += 1
            h1 = atom_idx;          elements.append('H'); atom_idx += 1
            h2 = atom_idx;          elements.append('H'); atom_idx += 1
            bonds.append({'i': c_methylene, 'j': h1, 'order': 1.0})
            bonds.append({'i': c_methylene, 'j': h2, 'order': 1.0})
            if prev_methine is not None:
                bonds.append({'i': prev_methine, 'j': c_methylene, 'order': 1.0})

            # -CH(Ph)- (methine)
            c_methine = atom_idx; elements.append('C'); atom_idx += 1
            h3 = atom_idx;        elements.append('H'); atom_idx += 1
            bonds.append({'i': c_methine,   'j': h3,          'order': 1.0})
            bonds.append({'i': c_methylene, 'j': c_methine,   'order': 1.0})

            # phenyl ring C6H5 (ph[0] = ipso, no H; ph[1..5] carry one H each)
            ph = []
            for _c in range(6):
                c = atom_idx; elements.append('C'); atom_idx += 1
                ph.append(c)
            for k in range(6):
                bonds.append({'i': ph[k], 'j': ph[(k + 1) % 6], 'order': 1.5})
            bonds.append({'i': c_methine, 'j': ph[0], 'order': 1.0})
            for k in range(1, 6):
                h = atom_idx; elements.append('H'); atom_idx += 1
                bonds.append({'i': ph[k], 'j': h, 'order': 1.0})

            prev_methine = c_methine

        # terminal H caps
        h_head = atom_idx; elements.append('H'); atom_idx += 1
        bonds.append({'i': 0, 'j': h_head, 'order': 1.0})
        h_tail = atom_idx; elements.append('H'); atom_idx += 1
        bonds.append({'i': prev_methine, 'j': h_tail, 'order': 1.0})

        print(f"Created polystyrene chain ({n_repeat_units} units): {len(elements)} atoms, {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology(self.n_repeat_units)

    def choose_start_atom(self, elements, bonds) -> int:
        adjacency = defaultdict(list)
        for b in bonds:
            adjacency[b['i']].append(b['j'])
            adjacency[b['j']].append(b['i'])
        # methine carbons have the most C-neighbours (backbone CH2 + backbone CH + ipso)
        return max(
            (i for i, e in enumerate(elements) if e == 'C'),
            key=lambda i: sum(1 for n in adjacency[i] if elements[n] == 'C'),
        )

    def bond_style(self, elem_i, elem_j, order) -> tuple:
        if abs(order - 1.5) < 0.1:
            return '#2060C0', 2.0   # aromatic — blue
        if elem_i == 'C' and elem_j == 'C':
            return '#404040', 3.5   # backbone — dark grey
        return '#909090', 1.5       # C-H — light grey


class PTFE(MoleculeReconstructor):
    """Polytetrafluoroethylene — repeat unit: -CF2-CF2-"""

    name = "Polytetrafluoroethylene (PTFE)"
    output_stem = "polytetrafluoroethylene"

    def __init__(self, n_carbons: int = 20):
        self.n_carbons = n_carbons

    def add_args(self, parser) -> None:
        parser.add_argument('--n_carbons', type=int, default=self.n_carbons,
                            help='Number of carbon atoms in the PTFE chain')

    def apply_args(self, args) -> None:
        self.n_carbons = args.n_carbons

    @staticmethod
    def create_topology(n_carbons: int) -> tuple:
        elements, bonds = [], []
        atom_idx = 0
        for c_idx in range(n_carbons):
            c  = atom_idx; elements.append('C'); atom_idx += 1
            f1 = atom_idx; elements.append('F'); atom_idx += 1
            f2 = atom_idx; elements.append('F'); atom_idx += 1
            bonds.append({'i': c, 'j': f1, 'order': 1.0})
            bonds.append({'i': c, 'j': f2, 'order': 1.0})
            if c_idx < n_carbons - 1:
                bonds.append({'i': c, 'j': c + 3, 'order': 1.0})
        n_f = 2 * n_carbons
        print(f"Created PTFE chain ({n_carbons} carbons): {len(elements)} atoms ({n_carbons}C + {n_f}F), {len(bonds)} bonds")
        return elements, bonds

    def build_topology(self) -> tuple:
        return self.create_topology(self.n_carbons)

    def bond_style(self, elem_i, elem_j, order) -> tuple:
        if elem_i == 'C' and elem_j == 'C':
            return '#404040', 3.5
        if 'F' in (elem_i, elem_j):
            return '#00CC44', 2.0   # C-F — green
        return '#a0a0a0', 1.5

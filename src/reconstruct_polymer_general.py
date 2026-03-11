#!/usr/bin/env python3
"""
Chunk-Based Polymer Reconstruction with Overlapping Segments

Reconstructs long polymers by:
1. Dividing into overlapping chunks
2. Reconstructing each chunk independently
3. Aligning and merging chunks using overlap regions
4. Averaging positions in overlapping regions for smoothness

This approach prevents error accumulation on very long chains.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
import pandas as pd
import re

# Import from reconstruction module
import sys
sys.path.insert(0, 'src')
from reconstruction import (
    load_trained_model,
    reconstruct_from_center,
    align_molecules,
    compute_rmsd
)

def create_polymer_from_smiles(monomer_smiles: str, n_units: int):
    """
    Build a polymer by repeating a monomer SMILES containing two '*' atoms.

    Example:
        *CC(*)c1ccccc1C(=O)OCCCCCC

    The '*' atoms mark polymer connection points.
    """

    monomer = Chem.MolFromSmiles(monomer_smiles)

    if monomer is None:
        raise ValueError("Invalid SMILES")

    # Locate star atoms
    star_atoms = [a for a in monomer.GetAtoms() if a.GetAtomicNum() == 0]

    if len(star_atoms) != 2:
        raise ValueError("Monomer must contain exactly two '*' atoms")

    star1, star2 = star_atoms

    # Find the atoms they connect to
    nbr1 = star1.GetNeighbors()[0].GetIdx()
    nbr2 = star2.GetNeighbors()[0].GetIdx()

    star1_idx = star1.GetIdx()
    star2_idx = star2.GetIdx()

    combined = Chem.RWMol()

    connection_sites = []

    for unit in range(n_units):

        atom_map = {}

        # copy atoms
        for atom in monomer.GetAtoms():
            idx = combined.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            atom_map[atom.GetIdx()] = idx

        # copy bonds
        for bond in monomer.GetBonds():
            i = atom_map[bond.GetBeginAtomIdx()]
            j = atom_map[bond.GetEndAtomIdx()]
            combined.AddBond(i, j, bond.GetBondType())

        # store connection sites (neighbors of stars)
        left = atom_map[nbr1]
        right = atom_map[nbr2]

        connection_sites.append((left, right))

    # connect repeating units
    for i in range(n_units - 1):
        right_current = connection_sites[i][1]
        left_next = connection_sites[i + 1][0]

        combined.AddBond(right_current, left_next, Chem.BondType.SINGLE)

    mol = combined.GetMol()

    # remove star atoms
    editable = Chem.RWMol(mol)

    star_indices = [a.GetIdx() for a in editable.GetAtoms() if a.GetAtomicNum() == 0]

    for idx in sorted(star_indices, reverse=True):
        editable.RemoveAtom(idx)

    mol = editable.GetMol()

    # extract graph
    elements = [a.GetSymbol() for a in mol.GetAtoms()]

    bonds = []
    for bond in mol.GetBonds():
        bonds.append({
            "i": bond.GetBeginAtomIdx(),
            "j": bond.GetEndAtomIdx(),
            "order": bond.GetBondTypeAsDouble()
        })

    print("Created polymer")
    print("Units:", n_units)
    print("Atoms:", len(elements))
    print("Bonds:", len(bonds))

    return elements, bonds


def get_carbon_indices(elements: List[str]) -> List[int]:
    """Get indices of all carbon atoms."""
    return [i for i, elem in enumerate(elements) if elem == 'C']


def create_chunks(n_carbons: int, chunk_size: int = 10, overlap: int = 3) -> List[Tuple[int, int]]:
    """
    Divide polymer into overlapping chunks based on carbon indices.

    Args:
        n_carbons: Total number of carbons
        chunk_size: Number of carbons per chunk
        overlap: Number of carbons overlap between consecutive chunks

    Returns:
        List of (start_carbon, end_carbon) tuples
    """
    chunks = []
    start = 0

    while start < n_carbons:
        end = min(start + chunk_size, n_carbons)
        chunks.append((start, end))

        if end >= n_carbons:
            break

        # Next chunk starts at (current_end - overlap)
        start = end - overlap

    print(f"\nChunk strategy:")
    print(f"  Chunk size: {chunk_size} carbons")
    print(f"  Overlap: {overlap} carbons")
    print(f"  Total chunks: {len(chunks)}")

    for i, (s, e) in enumerate(chunks):
        print(f"  Chunk {i+1}: carbons [{s}:{e}] ({e-s} carbons)")

    return chunks


def extract_chunk_subgraph(
    elements: List[str],
    bonds: List[Dict],
    carbon_indices: List[int],
    chunk_start: int,
    chunk_end: int
) -> Tuple[List[str], List[Dict], Dict[int, int], Dict[int, int]]:
    """
    Extract subgraph for a chunk (carbons in range [chunk_start:chunk_end]).

    Returns:
        chunk_elements: Elements in chunk
        chunk_bonds: Bonds in chunk
        old_to_new: Mapping from original atom indices to chunk indices
        new_to_old: Mapping from chunk indices to original atom indices
    """
    # Get carbons in this chunk
    chunk_carbon_indices = carbon_indices[chunk_start:chunk_end]

    # Build adjacency to find all atoms in chunk (carbons + their hydrogens)
    adjacency = defaultdict(list)
    for bond in bonds:
        i, j = bond['i'], bond['j']
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Include all atoms bonded to chunk carbons
    chunk_atom_set = set(chunk_carbon_indices)
    for c_idx in chunk_carbon_indices:
        chunk_atom_set.update(adjacency[c_idx])

    chunk_atoms = sorted(list(chunk_atom_set))

    # Create mappings
    old_to_new = {old: new for new, old in enumerate(chunk_atoms)}
    new_to_old = {new: old for old, new in old_to_new.items()}

    # Extract elements
    chunk_elements = [elements[i] for i in chunk_atoms]

    # Extract bonds (only bonds within chunk)
    chunk_bonds = []
    for bond in bonds:
        i, j = bond['i'], bond['j']
        if i in old_to_new and j in old_to_new:
            chunk_bonds.append({
                'i': old_to_new[i],
                'j': old_to_new[j],
                'order': bond['order']
            })

    return chunk_elements, chunk_bonds, old_to_new, new_to_old


def reconstruct_chunk(
    model,
    chunk_elements: List[str],
    chunk_bonds: List[Dict],
    element_vocab: Dict[str, int],
    device: torch.device,
    max_bonds: int
) -> np.ndarray:
    """
    Reconstruct a single chunk using BFS from center.

    Returns:
        positions: (N_chunk, 3) array of atom positions
    """
    # Find starting atom (carbon with most bonds in chunk)
    adjacency = defaultdict(list)
    for bond in chunk_bonds:
        i, j = bond['i'], bond['j']
        adjacency[i].append(j)
        adjacency[j].append(i)

    carbon_indices = [i for i, e in enumerate(chunk_elements) if e == 'C']
    if not carbon_indices:
        # No carbons (shouldn't happen)
        return np.zeros((len(chunk_elements), 3))

    start_atom = max(carbon_indices, key=lambda x: len(adjacency[x]))

    positions, _, _, _, _ = reconstruct_from_center(
        model=model,
        elements=chunk_elements,
        bonds_data=chunk_bonds,
        element_to_idx=element_vocab,
        device=device,
        max_bonds=max_bonds,
        start_atom=start_atom
    )

    return positions


def align_chunk_to_reference(
    chunk_positions: np.ndarray,
    reference_positions: np.ndarray,
    chunk_overlap_indices: List[int],
    reference_overlap_indices: List[int]
) -> np.ndarray:
    """
    Align chunk to reference using overlapping atoms (Kabsch algorithm).

    Args:
        chunk_positions: (N_chunk, 3) positions in chunk coordinate system
        reference_positions: (N_ref, 3) positions in reference coordinate system
        chunk_overlap_indices: Indices of overlap atoms in chunk
        reference_overlap_indices: Indices of overlap atoms in reference

    Returns:
        aligned_positions: (N_chunk, 3) chunk positions aligned to reference
    """
    # Extract overlapping positions
    chunk_overlap = chunk_positions[chunk_overlap_indices]
    ref_overlap = reference_positions[reference_overlap_indices]

    # Center both overlap regions
    chunk_center = chunk_overlap.mean(axis=0)
    ref_center = ref_overlap.mean(axis=0)

    chunk_centered = chunk_overlap - chunk_center
    ref_centered = ref_overlap - ref_center

    # Covariance matrix for Kabsch algorithm
    H = chunk_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply transformation to all chunk atoms
    chunk_all_centered = chunk_positions - chunk_center
    aligned = (chunk_all_centered @ R.T) + ref_center

    return aligned


def merge_chunks(
    chunk_positions_list: List[np.ndarray],
    chunk_mappings: List[Dict[int, int]],
    overlap_info: List[Tuple[List[int], List[int]]],
    total_atoms: int
) -> np.ndarray:
    """
    Merge multiple reconstructed chunks into full polymer structure.

    Args:
        chunk_positions_list: List of (N_i, 3) position arrays for each chunk
        chunk_mappings: List of new_to_old mappings for each chunk
        overlap_info: List of (chunk_overlap_indices, ref_overlap_indices) tuples
        total_atoms: Total number of atoms in full polymer

    Returns:
        merged_positions: (total_atoms, 3) merged positions
        atom_counts: (total_atoms,) how many times each atom was predicted
    """
    merged_positions = np.zeros((total_atoms, 3))
    atom_counts = np.zeros(total_atoms)

    # First chunk: use as-is (reference)
    print(f"\n[Merging] Chunk 1: Using as reference")
    for new_idx, old_idx in chunk_mappings[0].items():
        merged_positions[old_idx] = chunk_positions_list[0][new_idx]
        atom_counts[old_idx] += 1

    # Subsequent chunks: align and merge
    for chunk_idx in range(1, len(chunk_positions_list)):
        chunk_pos = chunk_positions_list[chunk_idx]
        chunk_mapping = chunk_mappings[chunk_idx]
        chunk_overlap, ref_overlap = overlap_info[chunk_idx - 1]

        print(f"[Merging] Chunk {chunk_idx + 1}:")
        print(f"  Overlap atoms: {len(chunk_overlap)}")

        # Align chunk to current merged structure
        aligned_chunk = align_chunk_to_reference(
            chunk_positions=chunk_pos,
            reference_positions=merged_positions,
            chunk_overlap_indices=chunk_overlap,
            reference_overlap_indices=ref_overlap
        )

        # Compute RMSD of overlap region (quality check)
        overlap_rmsd = compute_rmsd(
            aligned_chunk[chunk_overlap],
            merged_positions[ref_overlap]
        )
        print(f"  Overlap RMSD: {overlap_rmsd:.4f} Å")

        # Add aligned chunk to merged structure
        for new_idx, old_idx in chunk_mapping.items():
            merged_positions[old_idx] += aligned_chunk[new_idx]
            atom_counts[old_idx] += 1

    # Average positions for atoms predicted multiple times (overlap regions)
    for i in range(total_atoms):
        if atom_counts[i] > 0:
            merged_positions[i] /= atom_counts[i]

    print(f"\n[Merging] Complete:")
    print(f"  Atoms predicted once: {np.sum(atom_counts == 1)}")
    print(f"  Atoms predicted multiple times: {np.sum(atom_counts > 1)}")
    print(f"  Max predictions per atom: {int(np.max(atom_counts))}")

    return merged_positions, atom_counts


def reconstruct_polymer_chunked(
    model,
    elements: List[str],
    bonds: List[Dict],
    element_vocab: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    chunk_size: int = 10,
    overlap: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct polymer using chunk-based approach with overlapping segments.

    Returns:
        positions: (N, 3) merged positions
        atom_counts: (N,) prediction counts per atom
    """
    n_atoms = len(elements)
    carbon_indices = get_carbon_indices(elements)
    n_carbons = len(carbon_indices)

    print(f"\n{'='*60}")
    print("Chunk-Based Reconstruction")
    print(f"{'='*60}")

    # Create chunks
    chunks = create_chunks(n_carbons, chunk_size, overlap)

    # Reconstruct each chunk
    print(f"\n{'='*60}")
    print("Reconstructing Chunks")
    print(f"{'='*60}")

    chunk_positions_list = []
    chunk_mappings = []
    overlap_info = []

    for chunk_idx, (c_start, c_end) in enumerate(chunks):
        print(f"\nChunk {chunk_idx + 1}/{len(chunks)}: Carbons [{c_start}:{c_end}]")

        # Extract subgraph
        chunk_elements, chunk_bonds, old_to_new, new_to_old = extract_chunk_subgraph(
            elements, bonds, carbon_indices, c_start, c_end
        )

        print(f"  Atoms in chunk: {len(chunk_elements)}")
        print(f"  Bonds in chunk: {len(chunk_bonds)}")

        # Reconstruct chunk
        chunk_pos = reconstruct_chunk(
            model, chunk_elements, chunk_bonds,
            element_vocab, device, max_bonds
        )

        chunk_positions_list.append(chunk_pos)
        chunk_mappings.append(new_to_old)

        # Compute overlap info (for next merge)
        if chunk_idx > 0:
            # Overlapping carbons between current and previous chunk
            prev_c_start, prev_c_end = chunks[chunk_idx - 1]
            overlap_carbon_range = range(c_start, min(c_end, prev_c_end))
            overlap_carbon_original = [carbon_indices[i] for i in overlap_carbon_range]

            # Get chunk indices for overlap carbons in current chunk
            prev_chunk_mapping = chunk_mappings[chunk_idx - 1]
            curr_chunk_mapping_inv = {v: k for k, v in new_to_old.items()}

            chunk_overlap = []
            ref_overlap = []

            for orig_carbon_idx in overlap_carbon_original:
                # Find in current chunk
                if orig_carbon_idx in curr_chunk_mapping_inv:
                    chunk_overlap.append(curr_chunk_mapping_inv[orig_carbon_idx])
                # Find in previous chunk (or merged structure)
                ref_overlap.append(orig_carbon_idx)

            overlap_info.append((chunk_overlap, ref_overlap))

    # Merge chunks
    print(f"\n{'='*60}")
    print("Merging Chunks")
    print(f"{'='*60}")

    merged_positions, atom_counts = merge_chunks(
        chunk_positions_list,
        chunk_mappings,
        overlap_info,
        n_atoms
    )

    return merged_positions, atom_counts


def analyze_structure(elements, positions, bonds):
    """Analyze bond lengths using a reference bond-length library."""

    # Reference bond lengths (Å) for common single bonds
    bond_reference = {
        ('C', 'C'): 1.54,
        ('C', 'H'): 1.09,
        ('C', 'N'): 1.47,
        ('C', 'O'): 1.43,
        ('C', 'F'): 1.35,
        ('C', 'Cl'): 1.77,
        ('C', 'Br'): 1.94,
        ('C', 'I'): 2.14,
        ('C', 'S'): 1.82,
        ('N', 'H'): 1.01,
        ('O', 'H'): 0.96,
        ('N', 'N'): 1.45,
        ('O', 'O'): 1.48,
        ('S', 'H'): 1.34
    }

    # storage
    bond_lengths = {}

    for bond in bonds:

        i, j = bond['i'], bond['j']
        elem_i = elements[i]
        elem_j = elements[j]

        pair = tuple(sorted((elem_i, elem_j)))

        dist = np.linalg.norm(positions[i] - positions[j])

        if pair not in bond_lengths:
            bond_lengths[pair] = []

        bond_lengths[pair].append(dist)

    print("\n[Bond Length Analysis]")

    for pair, lengths in bond_lengths.items():

        mean_val = np.mean(lengths)
        std_val = np.std(lengths)
        min_val = np.min(lengths)
        max_val = np.max(lengths)

        ref = bond_reference.get(pair, None)

        print(f"\n  {pair[0]}-{pair[1]} bonds: {len(lengths)}")
        print(f"    Mean: {mean_val:.3f} Å")
        print(f"    Std:  {std_val:.3f} Å")
        print(f"    Range: [{min_val:.3f}, {max_val:.3f}] Å")

        if ref is not None:
            print(f"    Expected: ~{ref:.2f} Å")
            print(f"    Deviation: {mean_val - ref:+.3f} Å")
        else:
            print("    Expected: unknown (not in reference library)")


def export_to_xyz(elements, positions, filepath, comment=""):
    """Export structure to XYZ format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n")
        f.write(f"{comment}\n")
        for elem, pos in zip(elements, positions):
            f.write(f"{elem:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
    print(f"  Exported XYZ: {filepath}")


def visualize_comparison(
    elements,
    positions_simple,
    positions_chunked,
    atom_counts,
    bonds,
    filepath,
    n_carbons
):
    """Create visualization comparing simple vs chunked reconstruction."""

    fig = plt.figure(figsize=(20, 8))

    # Extended CPK colors
    element_colors = {
        'H': '#FFFFFF',
        'C': '#909090',
        'N': '#3050F8',
        'O': '#FF0D0D',
        'F': '#90E050',
        'Cl': '#1FF01F',
        'Br': '#A62929',
        'I': '#940094',
        'S': '#FFFF30',
        'P': '#FF8000'
    }

    element_sizes = {
        'H': 120,
        'C': 350,
        'N': 350,
        'O': 350,
        'F': 320,
        'Cl': 380,
        'Br': 420,
        'I': 450,
        'S': 380,
        'P': 380
    }

    default_color = '#808080'
    default_size = 300

    # Identify carbon atoms if present
    carbon_indices = [i for i, e in enumerate(elements) if e == 'C']

    if len(carbon_indices) == 0:
        carbon_indices = list(range(len(elements)))

    carbon_positions_simple = positions_simple[carbon_indices]
    carbon_positions_chunked = positions_chunked[carbon_indices]

    # -----------------------------
    # 3D Structure Plots
    # -----------------------------
    for plot_idx, (positions, title) in enumerate([
        (positions_simple, "Simple BFS Reconstruction"),
        (positions_chunked, "Chunk-Based Reconstruction"),
    ], 1):

        ax = fig.add_subplot(1, 3, plot_idx, projection='3d')

        # Draw bonds
        for bond in bonds:

            i, j = bond['i'], bond['j']
            elem_i, elem_j = elements[i], elements[j]

            if elem_i == 'C' and elem_j == 'C':
                color, linewidth = '#404040', 3.0
            else:
                color, linewidth = '#808080', 1.8

            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                [positions[i, 2], positions[j, 2]],
                color=color,
                linewidth=linewidth,
                zorder=1
            )

        # Draw atoms
        for elem in set(elements):

            indices = [i for i, e in enumerate(elements) if e == elem]

            color = element_colors.get(elem, default_color)
            size = element_sizes.get(elem, default_size)

            ax.scatter(
                positions[indices, 0],
                positions[indices, 1],
                positions[indices, 2],
                c=color,
                s=size,
                edgecolors='black',
                linewidths=0.8,
                depthshade=True,
                zorder=2
            )

        ax.set_xlabel('X (Å)', fontsize=10)
        ax.set_ylabel('Y (Å)', fontsize=10)
        ax.set_zlabel('Z (Å)', fontsize=10)

        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.view_init(elev=20, azim=45)

        ax.grid(True, alpha=0.25)

    # -----------------------------
    # Overlap Heatmap
    # -----------------------------
    ax3 = fig.add_subplot(1, 3, 3)

    carbon_counts = atom_counts[carbon_indices]

    ax3.barh(
        range(len(carbon_counts)),
        carbon_counts,
        color='steelblue',
        edgecolor='black'
    )

    ax3.set_xlabel('Number of Predictions', fontsize=11)
    ax3.set_ylabel('Backbone Atom Index', fontsize=11)

    ax3.set_title(
        'Prediction Count per Backbone Atom\n(Higher = More Overlap)',
        fontsize=12,
        fontweight='bold'
    )

    ax3.set_ylim(-0.5, len(carbon_counts) - 0.5)
    ax3.invert_yaxis()

    ax3.grid(axis='x', alpha=0.3)

    # Annotate overlap regions
    for i, count in enumerate(carbon_counts):
        if count > 1:
            ax3.text(
                count + 0.05,
                i,
                f'{int(count)}x',
                va='center',
                fontsize=8,
                color='red',
                fontweight='bold'
            )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches='tight',
        facecolor='white'
    )

    print(f"  Visualization: {filepath}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Chunk-based polymer reconstruction using SMILES monomers'
    )

    parser.add_argument('--csv', type=str, default="test.csv",
                        help='CSV file containing SMILES column')

    parser.add_argument('--n_monomers', type=int, default=5,
                        help='Number of monomers in polymer')

    parser.add_argument('--chunk_size', type=int, default=8,
                        help='Chunk size for reconstruction')

    parser.add_argument('--overlap', type=int, default=3,
                        help='Overlap between chunks')

    parser.add_argument('--output_dir', type=str, default='polymer_results',
                        help='Output directory')

    args = parser.parse_args()

    # Validate arguments
    if args.overlap >= args.chunk_size:
        raise ValueError(
            f"Overlap ({args.overlap}) must be less than chunk_size ({args.chunk_size})"
        )

    if args.overlap < 2:
        print(f"Warning: Small overlap ({args.overlap}) may lead to poor alignment")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Chunk-Based Polymer Reconstruction from SMILES")
    print("=" * 60)
    print(f"CSV file: {args.csv}")
    print(f"Monomers per polymer: {args.n_monomers}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")

    # Load CSV
    print("\n[1] Loading CSV...")
    df = pd.read_csv(args.csv)

    if "SMILES" not in df.columns:
        raise ValueError("CSV must contain a column named 'SMILES'")

    smiles_list = df["SMILES"].dropna().tolist()

    print(f"Found {len(smiles_list)} SMILES entries")

    # Load model
    print("\n[2] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='config/config.yaml'
    )

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Process each SMILES
    for smiles in smiles_list:

        print("\n" + "-" * 60)
        print(f"[SMILES] {smiles}")

        # Create polymer
        print("\n[3] Creating polymer...")
        elements, bonds = create_polymer_from_smiles(
            smiles,
            args.n_monomers
        )

        # Build adjacency
        from collections import defaultdict
        adjacency = defaultdict(list)

        for bond in bonds:
            i, j = bond['i'], bond['j']
            adjacency[i].append(j)
            adjacency[j].append(i)

        start_atom = max(range(len(elements)), key=lambda x: len(adjacency[x]))

        # Simple reconstruction
        print("\n[4] Simple reconstruction...")
        positions_simple, _, _, _, _ = reconstruct_from_center(
            model=model,
            elements=elements,
            bonds_data=bonds,
            element_to_idx=element_vocab,
            device=device,
            max_bonds=config['model']['max_bonds'],
            start_atom=start_atom
        )

        # Chunked reconstruction
        print("\n[5] Chunked reconstruction...")
        positions_chunked, atom_counts = reconstruct_polymer_chunked(
            model=model,
            elements=elements,
            bonds=bonds,
            element_vocab=element_vocab,
            device=device,
            max_bonds=config['model']['max_bonds'],
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

        # Align results
        positions_chunked_aligned = align_molecules(
            positions_chunked,
            positions_simple
        )

        rmsd_comparison = compute_rmsd(
            positions_chunked_aligned,
            positions_simple
        )

        print(f"\nRMSD between methods: {rmsd_comparison:.4f} Å")

        # Analysis
        print("\n[6] Simple analysis")
        analyze_structure(elements, positions_simple, bonds)

        print("\n[7] Chunked analysis")
        analyze_structure(elements, positions_chunked, bonds)

        # Export
        print("\n[8] Exporting files...")

        safe_smiles = re.sub(r'[^a-zA-Z0-9]', '_', smiles)
        prefix = f"polymer_{safe_smiles}"

        xyz_simple = output_dir / f"{prefix}_simple.xyz"
        xyz_chunked = output_dir / f"{prefix}_chunked.xyz"
        png_comparison = output_dir / f"{prefix}_comparison.png"

        export_to_xyz(
            elements,
            positions_simple,
            xyz_simple,
            comment=f"Polymer from {smiles} - Simple"
        )

        export_to_xyz(
            elements,
            positions_chunked,
            xyz_chunked,
            comment=f"Polymer from {smiles} - Chunked"
        )

        visualize_comparison(
            elements,
            positions_simple,
            positions_chunked_aligned,
            atom_counts,
            bonds,
            png_comparison,
            len(elements)
        )

    print("\n" + "=" * 60)
    print("All reconstructions complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

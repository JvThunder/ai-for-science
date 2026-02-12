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

# Import from reconstruction module
import sys
sys.path.insert(0, 'src')
from reconstruction import (
    load_trained_model,
    reconstruct_from_center,
    align_molecules,
    compute_rmsd
)


def create_polyethylene_chain(n_carbons: int) -> Tuple[List[str], List[Dict]]:
    """
    Create a polyethylene chain with n carbon atoms.

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

    # CPK colors
    element_colors = {'H': '#FFFFFF', 'C': '#909090'}
    element_sizes = {'H': 150, 'C': 400}

    # Get carbon positions for backbone visualization
    carbon_indices = [i for i, e in enumerate(elements) if e == 'C']
    carbon_positions_simple = positions_simple[carbon_indices]
    carbon_positions_chunked = positions_chunked[carbon_indices]

    # Three subplots: Simple, Chunked, Overlap Heatmap
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
                color, linewidth = '#404040', 3.5
            else:
                color, linewidth = '#808080', 2.0

            ax.plot([positions[i, 0], positions[j, 0]],
                   [positions[i, 1], positions[j, 1]],
                   [positions[i, 2], positions[j, 2]],
                   color=color, linewidth=linewidth, zorder=1)

        # Draw atoms
        for elem in set(elements):
            indices = [i for i, e in enumerate(elements) if e == elem]
            ax.scatter(positions[indices, 0],
                      positions[indices, 1],
                      positions[indices, 2],
                      c=element_colors[elem],
                      s=element_sizes[elem],
                      edgecolors='black',
                      linewidths=1.0,
                      depthshade=True,
                      zorder=2)

        ax.set_xlabel('X (Å)', fontsize=10)
        ax.set_ylabel('Y (Å)', fontsize=10)
        ax.set_zlabel('Z (Å)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)

    # Third subplot: Overlap heatmap
    ax3 = fig.add_subplot(1, 3, 3)

    # Show atom prediction counts as heatmap
    carbon_counts = atom_counts[carbon_indices]

    ax3.barh(range(len(carbon_counts)), carbon_counts, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Number of Predictions', fontsize=11)
    ax3.set_ylabel('Carbon Index', fontsize=11)
    ax3.set_title('Prediction Count per Carbon\n(Higher = More Overlap)',
                  fontsize=12, fontweight='bold')
    ax3.set_ylim(-0.5, len(carbon_counts) - 0.5)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # Add text showing overlap regions
    for i, count in enumerate(carbon_counts):
        if count > 1:
            ax3.text(count + 0.1, i, f'{int(count)}x',
                    va='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Visualization: {filepath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Chunk-based polymer reconstruction with overlapping segments'
    )
    parser.add_argument('--n_carbons', type=int, default=20,
                       help='Number of carbon atoms in chain')
    parser.add_argument('--chunk_size', type=int, default=8,
                       help='Number of carbons per chunk')
    parser.add_argument('--overlap', type=int, default=3,
                       help='Number of carbons overlap between chunks')
    parser.add_argument('--output_dir', type=str, default='polymer_results',
                       help='Output directory')
    args = parser.parse_args()

    # Validate arguments
    if args.overlap >= args.chunk_size:
        raise ValueError(f"Overlap ({args.overlap}) must be less than chunk_size ({args.chunk_size})")

    if args.overlap < 2:
        print(f"Warning: Small overlap ({args.overlap}) may lead to poor alignment")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Chunk-Based Polymer Reconstruction")
    print("="*60)
    print(f"Total carbons: {args.n_carbons}")
    print(f"Chunk size: {args.chunk_size} carbons")
    print(f"Overlap: {args.overlap} carbons")

    # Load model
    print("\n[1] Loading model...")
    model, element_vocab, device = load_trained_model(
        checkpoint_path='checkpoints/best_model.pt',
        config_path='src/config.yaml'
    )

    # Load config separately for max_bonds
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create polymer chain
    print(f"\n[2] Creating {args.n_carbons}-carbon polyethylene chain...")
    elements, bonds = create_polyethylene_chain(args.n_carbons)

    # Reconstruct using simple approach (for comparison)
    print(f"\n[3] Reconstructing with SIMPLE approach (baseline)...")
    from collections import defaultdict
    adjacency = defaultdict(list)
    for bond in bonds:
        i, j = bond['i'], bond['j']
        adjacency[i].append(j)
        adjacency[j].append(i)

    start_atom = max([i for i, e in enumerate(elements) if e == 'C'],
                    key=lambda x: len(adjacency[x]))

    positions_simple, _, _, _, _ = reconstruct_from_center(
        model=model,
        elements=elements,
        bonds_data=bonds,
        element_to_idx=element_vocab,
        device=device,
        max_bonds=config['model']['max_bonds'],
        start_atom=start_atom
    )

    print(f"  Simple reconstruction complete")

    # Reconstruct using chunked approach
    print(f"\n[4] Reconstructing with CHUNKED approach...")
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

    # Align chunked to simple for fair comparison
    positions_chunked_aligned = align_molecules(positions_chunked, positions_simple)

    # Compute RMSD between approaches
    rmsd_comparison = compute_rmsd(positions_chunked_aligned, positions_simple)
    print(f"\n[5] Comparison:")
    print(f"  RMSD between simple and chunked: {rmsd_comparison:.4f} Å")

    # Analyze both
    print(f"\n[6] Analysis - Simple approach:")
    analyze_structure(elements, positions_simple, bonds)

    print(f"\n[7] Analysis - Chunked approach:")
    analyze_structure(elements, positions_chunked, bonds)

    # Export
    print(f"\n[8] Exporting files...")
    xyz_simple = output_dir / f'polyethylene_{args.n_carbons}c_simple.xyz'
    xyz_chunked = output_dir / f'polyethylene_{args.n_carbons}c_chunked.xyz'
    png_comparison = output_dir / f'polyethylene_{args.n_carbons}c_comparison.png'

    export_to_xyz(elements, positions_simple, xyz_simple,
                 comment=f"Polyethylene {args.n_carbons}C - Simple BFS")
    export_to_xyz(elements, positions_chunked, xyz_chunked,
                 comment=f"Polyethylene {args.n_carbons}C - Chunked (size={args.chunk_size}, overlap={args.overlap})")
    visualize_comparison(elements, positions_simple, positions_chunked_aligned,
                        atom_counts, bonds, png_comparison, args.n_carbons)

    print("\n" + "="*60)
    print("Reconstruction complete!")
    print(f"Output files in: {output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
3D Molecular Geometry Reconstruction from Trained GCN Model

This script loads a trained LocalGeometryGCN model and reconstructs 3D molecular
geometries from the predicted local spherical coordinates (r, theta, delta_phi).

Usage:
    python reconstruction.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import math
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import utilities from the project
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.model import LocalGeometryGCN
from utils.dataset import parse_xyz_string, parse_bonds_from_dataset, get_element_vocab, dfs_subgraph
from utils.geometry import compute_local_geometry, spherical_to_cartesian, create_orthonormal_frame, get_neighbor_sort_keys
from utils.config import load_config


def load_trained_model(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    device: str = "auto"
) -> Tuple[LocalGeometryGCN, Dict[str, int], torch.device]:
    """
    Load a trained LocalGeometryGCN model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        config_path: Path to config.yaml for model architecture
        device: Device to load model on ('cuda', 'cpu', or 'auto')

    Returns:
        model: Loaded model in eval mode
        element_to_idx: Element vocabulary mapping
        device: torch.device used
    """
    # Setup device
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load config
    config = load_config(config_path)

    # Load element vocabulary
    elements_txt = config['data'].get('elements_txt', 'elements.txt')
    element_to_idx = get_element_vocab(txt_path=elements_txt)
    vocab_size = len(element_to_idx)

    # Create model with same architecture
    model = LocalGeometryGCN(
        num_node_features=vocab_size,
        hidden_dim=config['model']['hidden_dim'],
        max_bonds=config['model']['max_bonds'],
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        depth=config['model'].get('depth', 1),
        use_bond_types=config['model'].get('use_bond_types', True),
        num_bond_types=config['model'].get('num_bond_types', 4),
        bond_embedding_dim=config['model'].get('bond_embedding_dim', 8)
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max bonds: {config['model']['max_bonds']}")

    return model, element_to_idx, device


def load_molecule_from_csv(
    csv_path: str,
    molecule_idx: int
) -> Tuple[List[List[float]], List[str], List[Dict], str]:
    """
    Load a single molecule from CSV file.

    Args:
        csv_path: Path to CSV file
        molecule_idx: Index of molecule to load

    Returns:
        coordinates: List of [x, y, z] coordinates
        elements: List of element symbols
        bonds_data: List of bond dictionaries
        cid: Compound ID
    """
    df = pd.read_csv(csv_path)

    # Find valid molecules (with bonding_status="OK")
    valid_rows = []
    for idx, row in df.iterrows():
        if row.get('bonding_status') == 'OK':
            if 'geometry_xyz' in row and not pd.isna(row['geometry_xyz']):
                if 'bonds_geom_valence' in row and not pd.isna(row['bonds_geom_valence']):
                    valid_rows.append((idx, row))

    if molecule_idx >= len(valid_rows):
        raise IndexError(f"Molecule index {molecule_idx} out of range (only {len(valid_rows)} valid molecules)")

    idx, row = valid_rows[molecule_idx]

    coordinates, elements = parse_xyz_string(row['geometry_xyz'])
    bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])
    cid = row.get('cid', f'molecule_{idx}')

    return coordinates, elements, bonds_data, cid


def create_graph_data(
    coordinates: List[List[float]],
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    node_a: int,
    node_b: int,
    max_bonds: int = 7,
    subgraph_depth: int = 1,
    device: torch.device = torch.device('cpu')
) -> 'torch_geometric.data.Data':
    """
    Create a PyG Data object for a specific bond.

    Args:
        coordinates: Atom positions
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        node_a, node_b: Bond endpoints
        max_bonds: Maximum bonds per atom
        subgraph_depth: Depth for subgraph extraction
        device: torch device

    Returns:
        PyG Data object ready for model inference
    """
    from torch_geometric.data import Data

    pos = torch.tensor(coordinates, dtype=torch.float)

    # Create element indices
    element_indices = [element_to_idx.get(elem, 0) for elem in elements]
    x = torch.tensor(element_indices, dtype=torch.long)

    # Create adjacency graph with bond types
    adjacency = defaultdict(list)
    bond_types_dict = {}

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < len(coordinates) and j < len(coordinates) and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)

                # Extract bond type
                bond_order = bond.get('order', 1.0)
                if bond_order == 1.0:
                    bond_type = 0
                elif bond_order == 1.5:
                    bond_type = 1
                elif bond_order == 2.0:
                    bond_type = 2
                elif bond_order == 3.0:
                    bond_type = 3
                else:
                    bond_type = 0

                bond_types_dict[(i, j)] = bond_type
                bond_types_dict[(j, i)] = bond_type

    # Extract subgraph around both nodes
    subgraph_a = dfs_subgraph(adjacency, node_a, depth=subgraph_depth)
    subgraph_b = dfs_subgraph(adjacency, node_b, depth=subgraph_depth)
    subgraph_nodes = subgraph_a.union(subgraph_b)
    subgraph_nodes.add(node_a)
    subgraph_nodes.add(node_b)

    # Create node mapping
    node_list = sorted(list(subgraph_nodes))
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_list)}

    # Map target nodes
    new_node_a = node_mapping[node_a]
    new_node_b = node_mapping[node_b]

    # Create subgraph features
    subgraph_x = x[node_list]
    subgraph_pos = pos[node_list]

    # Create edges with bond types
    edge_index = []
    edge_attr = []

    for old_i in node_list:
        for old_j in adjacency[old_i]:
            if old_j in node_mapping:
                new_i = node_mapping[old_i]
                new_j = node_mapping[old_j]
                edge_index.append([new_i, new_j])
                bond_type = bond_types_dict.get((old_i, old_j), 0)
                edge_attr.append(bond_type)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long) if edge_attr else torch.empty((0,), dtype=torch.long)

    # Create Data object
    data = Data(
        x=subgraph_x.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        pos=subgraph_pos.to(device),
        node_a=torch.tensor(new_node_a, dtype=torch.long).to(device),
        node_b=torch.tensor(new_node_b, dtype=torch.long).to(device)
    )

    return data, node_mapping, node_list


def predict_bond_geometry(
    model: LocalGeometryGCN,
    data: 'torch_geometric.data.Data',
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Predict geometry for a single bond.

    Args:
        model: Trained model
        data: PyG Data object for the bond
        device: torch device

    Returns:
        distance: Predicted A-B distance
        geometry_a: (max_bonds, 2) predicted (theta, delta_phi) for A's neighbors
        geometry_b: (max_bonds, 2) predicted (theta, delta_phi) for B's neighbors
    """
    model.eval()
    with torch.no_grad():
        distance, geometry_a, geometry_b = model(data)

    distance = distance.squeeze().cpu().item()
    geometry_a = geometry_a.squeeze().cpu().numpy()  # (max_bonds, 2)
    geometry_b = geometry_b.squeeze().cpu().numpy()  # (max_bonds, 2)

    return distance, geometry_a, geometry_b


def reconstruct_molecule(
    model: LocalGeometryGCN,
    coordinates: List[List[float]],
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int = 7
) -> np.ndarray:
    """
    Reconstruct 3D molecular geometry using model predictions.

    This uses an iterative approach:
    1. Fix the first bond (two atoms)
    2. For each subsequent atom, use predicted geometry to place it
    3. Average over multiple bond predictions for consistency

    Args:
        model: Trained model
        coordinates: Original atom positions (for reference)
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom

    Returns:
        reconstructed_coords: (N, 3) numpy array of reconstructed positions
    """
    n_atoms = len(coordinates)
    original_pos = np.array(coordinates)

    # Build adjacency list and bond orders
    adjacency = defaultdict(list)
    bond_list = []
    bond_orders = {}  # (i, j) -> bond_order

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)
                if i < j:  # Avoid duplicates
                    bond_list.append((i, j))
                # Store bond order for deterministic sorting
                bond_order = bond.get('order', 1.0)
                bond_orders[(i, j)] = bond_order
                bond_orders[(j, i)] = bond_order

    if not bond_list:
        return original_pos.copy()

    # Initialize reconstructed coordinates
    reconstructed = np.zeros((n_atoms, 3))
    placed = np.zeros(n_atoms, dtype=bool)

    # Place first two atoms from the first bond
    first_bond = bond_list[0]
    i, j = first_bond

    # Get predicted distance for the first bond
    data, node_mapping, node_list = create_graph_data(
        coordinates, elements, bonds_data, element_to_idx, i, j, max_bonds, device=device
    )
    pred_dist, _, _ = predict_bond_geometry(model, data, device)

    # Place atom i at origin
    reconstructed[i] = [0, 0, 0]
    placed[i] = True

    # Place atom j along x-axis at predicted distance
    reconstructed[j] = [pred_dist, 0, 0]
    placed[j] = True

    # BFS to place remaining atoms
    queue = [i, j]
    visited_bonds = {(min(i, j), max(i, j))}

    while queue:
        center = queue.pop(0)
        neighbors = adjacency[center]

        if len(neighbors) == 0:
            continue

        # Find a placed neighbor to use as reference
        placed_neighbors = [n for n in neighbors if placed[n]]
        has_unplaced = any(not placed[n] for n in neighbors)

        if not placed_neighbors or not has_unplaced:
            continue

        # Use the first placed neighbor as reference
        ref_atom = placed_neighbors[0]

        # Get predictions for center-ref bond
        data, node_mapping, node_list = create_graph_data(
            coordinates, elements, bonds_data, element_to_idx, center, ref_atom, max_bonds, device=device
        )
        _, geometry_center, _ = predict_bond_geometry(model, data, device)

        # Create local coordinate frame
        center_pos = torch.tensor(reconstructed[center], dtype=torch.float)
        ref_pos = torch.tensor(reconstructed[ref_atom], dtype=torch.float)

        bond_vec = ref_pos - center_pos
        bond_length = torch.norm(bond_vec)

        if bond_length < 1e-6:
            continue

        bond_axis = bond_vec / bond_length
        x_axis, y_axis = create_orthonormal_frame(bond_axis)
        coord_frame = (x_axis, y_axis, bond_axis)

        # Get ALL neighbors with deterministic sorting (same as training)
        all_neighbors = neighbors.copy()

        # Create deterministic sort keys (same as training)
        sort_keys = get_neighbor_sort_keys(all_neighbors, elements, bond_orders, center)

        # Sort neighbors deterministically
        sorted_neighbors_with_keys = sorted(zip(all_neighbors, sort_keys), key=lambda x: x[1])
        sorted_neighbors = [n for n, _ in sorted_neighbors_with_keys]

        # IMPORTANT: During training, the reference atom is EXCLUDED from geometry predictions
        # The model predicts for neighbors EXCEPT the reference (which defines the coordinate frame)
        # So we need to filter out the reference and use slots 0, 1, 2... for the remaining neighbors
        sorted_neighbors_excl_ref = [n for n in sorted_neighbors if n != ref_atom]

        # Place unplaced neighbors using predicted geometry
        # Note: During training, the first neighbor's Δφ is set to 0 (defines reference frame)
        # So we should not use predicted Δφ for slot 0 - it should be 0 by definition
        cumulative_phi = 0.0

        for slot_idx, neighbor in enumerate(sorted_neighbors_excl_ref):
            if slot_idx >= max_bonds:
                break

            if placed[neighbor]:
                # Already placed, but still accumulate phi for subsequent neighbors
                # First neighbor (slot 0) has delta_phi = 0 by definition
                if slot_idx > 0:
                    delta_phi_deg = geometry_center[slot_idx, 1]
                    cumulative_phi += delta_phi_deg
                continue

            # Get predicted geometry for this neighbor slot
            theta_deg = geometry_center[slot_idx, 0]  # degrees

            # First neighbor (slot 0) defines phi=0, others use cumulative delta_phi
            if slot_idx == 0:
                # First neighbor: phi = 0 (defines reference direction in plane)
                phi = 0.0
            else:
                delta_phi_deg = geometry_center[slot_idx, 1]  # degrees
                cumulative_phi += delta_phi_deg
                phi = math.radians(cumulative_phi % 360.0)

            theta = math.radians(theta_deg)

            # Get predicted distance for this bond
            bond_key = (min(center, neighbor), max(center, neighbor))
            if bond_key not in visited_bonds:
                data_neighbor, _, _ = create_graph_data(
                    coordinates, elements, bonds_data, element_to_idx,
                    center, neighbor, max_bonds, device=device
                )
                pred_r, _, _ = predict_bond_geometry(model, data_neighbor, device)
                visited_bonds.add(bond_key)
            else:
                # Use original distance as fallback
                pred_r = np.linalg.norm(np.array(coordinates[center]) - np.array(coordinates[neighbor]))

            # Convert spherical to Cartesian
            new_pos = spherical_to_cartesian(pred_r, theta, phi, coord_frame, center_pos)
            reconstructed[neighbor] = new_pos.numpy()
            placed[neighbor] = True
            queue.append(neighbor)

    # For any atoms still not placed, use original coordinates
    for atom_idx in range(n_atoms):
        if not placed[atom_idx]:
            reconstructed[atom_idx] = original_pos[atom_idx]

    return reconstructed


def align_molecules(coords_pred: np.ndarray, coords_true: np.ndarray) -> np.ndarray:
    """
    Align predicted coordinates to true coordinates using Kabsch algorithm.

    Args:
        coords_pred: (N, 3) predicted coordinates
        coords_true: (N, 3) true coordinates

    Returns:
        aligned_pred: (N, 3) aligned predicted coordinates
    """
    # Center both coordinate sets
    pred_center = coords_pred.mean(axis=0)
    true_center = coords_true.mean(axis=0)

    pred_centered = coords_pred - pred_center
    true_centered = coords_true - true_center

    # Compute covariance matrix
    H = pred_centered.T @ true_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Optimal rotation
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation
    aligned = (pred_centered @ R.T) + true_center

    return aligned


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def visualize_reconstruction(
    original_coords: np.ndarray,
    reconstructed_coords: np.ndarray,
    elements: List[str],
    bonds_data: List[Dict],
    cid: str,
    rmsd_value: float,
    save_path: Optional[str] = None
):
    """
    Visualize original vs reconstructed molecular geometry as 3D plots.

    Args:
        original_coords: (N, 3) original coordinates
        reconstructed_coords: (N, 3) reconstructed coordinates
        elements: Element symbols
        bonds_data: Bond information
        cid: Compound ID
        rmsd_value: RMSD between aligned structures
        save_path: Path to save the figure
    """
    # Element colors
    element_colors = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'S': '#FFFF30', 'Cl': '#1FF01F', 'Br': '#A62929',
        'P': '#FF8000', 'I': '#940094', 'B': '#FFB5B5', 'Si': '#F0C8A0'
    }
    default_color = '#FF1493'

    # Element sizes
    element_sizes = {
        'H': 200, 'C': 400, 'N': 400, 'O': 400, 'F': 350, 'S': 500,
        'Cl': 450, 'Br': 500, 'P': 450, 'I': 550, 'B': 400, 'Si': 500
    }
    default_size = 400

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 7))

    # Get colors and sizes for atoms
    colors = [element_colors.get(e, default_color) for e in elements]
    sizes = [element_sizes.get(e, default_size) for e in elements]

    # Parse bonds
    bond_pairs = []
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < len(elements) and j < len(elements):
                bond_pairs.append((i, j))

    # Original structure
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot atoms
    ax1.scatter(
        original_coords[:, 0], original_coords[:, 1], original_coords[:, 2],
        c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
    )

    # Plot bonds
    for i, j in bond_pairs:
        ax1.plot(
            [original_coords[i, 0], original_coords[j, 0]],
            [original_coords[i, 1], original_coords[j, 1]],
            [original_coords[i, 2], original_coords[j, 2]],
            'k-', linewidth=2, alpha=0.6
        )

    # Add atom labels
    for idx, (elem, coord) in enumerate(zip(elements, original_coords)):
        ax1.text(coord[0], coord[1], coord[2] + 0.3, f"{elem}{idx}", fontsize=8, ha='center')

    ax1.set_xlabel('X (A)')
    ax1.set_ylabel('Y (A)')
    ax1.set_zlabel('Z (A)')
    ax1.set_title(f'Original Geometry\n{cid}', fontsize=12)

    # Reconstructed structure
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot atoms
    ax2.scatter(
        reconstructed_coords[:, 0], reconstructed_coords[:, 1], reconstructed_coords[:, 2],
        c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
    )

    # Plot bonds
    for i, j in bond_pairs:
        ax2.plot(
            [reconstructed_coords[i, 0], reconstructed_coords[j, 0]],
            [reconstructed_coords[i, 1], reconstructed_coords[j, 1]],
            [reconstructed_coords[i, 2], reconstructed_coords[j, 2]],
            'k-', linewidth=2, alpha=0.6
        )

    # Add atom labels
    for idx, (elem, coord) in enumerate(zip(elements, reconstructed_coords)):
        ax2.text(coord[0], coord[1], coord[2] + 0.3, f"{elem}{idx}", fontsize=8, ha='center')

    ax2.set_xlabel('X (A)')
    ax2.set_ylabel('Y (A)')
    ax2.set_zlabel('Z (A)')
    ax2.set_title(f'Reconstructed Geometry\nRMSD: {rmsd_value:.4f} A', fontsize=12)

    # Set equal aspect ratio for both plots
    for ax in [ax1, ax2]:
        all_coords = np.vstack([original_coords, reconstructed_coords])
        max_range = np.max(np.ptp(all_coords, axis=0)) / 2
        mid = all_coords.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def reconstruction_demo(
    checkpoint_path: str = "checkpoints/best_model.pt",
    csv_path: str = "data/Compound_000000001_000025000_props_geom.csv",
    config_path: str = "config.yaml",
    molecule_indices: List[int] = [0, 1, 2],
    output_dir: str = "reconstruction_results"
):
    """
    Run reconstruction demo on example molecules.

    Args:
        checkpoint_path: Path to trained model
        csv_path: Path to data CSV
        config_path: Path to config file
        molecule_indices: List of molecule indices to reconstruct
        output_dir: Directory to save results
    """
    print("="*60)
    print("3D Molecular Geometry Reconstruction Demo")
    print("="*60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1] Loading trained model...")
    model, element_to_idx, device = load_trained_model(checkpoint_path, config_path)

    # Load config for max_bonds
    config = load_config(config_path)
    max_bonds = config['model']['max_bonds']

    print(f"\n[2] Processing {len(molecule_indices)} molecules from {csv_path}...")

    results = []

    for mol_idx in molecule_indices:
        print(f"\n{'='*40}")
        print(f"Molecule {mol_idx + 1}")
        print("="*40)

        try:
            # Load molecule
            coords, elements, bonds, cid = load_molecule_from_csv(csv_path, mol_idx)
            original_coords = np.array(coords)

            print(f"  CID: {cid}")
            print(f"  Atoms: {len(elements)}")
            print(f"  Bonds: {len([b for b in bonds if 'i' in b and 'j' in b])}")
            print(f"  Elements: {set(elements)}")

            # Reconstruct
            print("  Reconstructing...")
            reconstructed_coords = reconstruct_molecule(
                model, coords, elements, bonds, element_to_idx, device, max_bonds
            )

            # Align for comparison
            aligned_coords = align_molecules(reconstructed_coords, original_coords)

            # Compute RMSD
            rmsd_value = compute_rmsd(aligned_coords, original_coords)
            print(f"  RMSD (after alignment): {rmsd_value:.4f} A")

            results.append({
                'cid': cid,
                'n_atoms': len(elements),
                'n_bonds': len([b for b in bonds if 'i' in b and 'j' in b]),
                'rmsd': rmsd_value
            })

            # Visualize
            save_path = output_path / f"reconstruction_{cid}.png"
            visualize_reconstruction(
                original_coords, aligned_coords, elements, bonds, cid, rmsd_value, str(save_path)
            )

        except Exception as e:
            print(f"  Error processing molecule {mol_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if results:
        avg_rmsd = np.mean([r['rmsd'] for r in results])
        print(f"\nProcessed {len(results)} molecules successfully")
        print(f"Average RMSD: {avg_rmsd:.4f} A")
        print("\nIndividual Results:")
        for r in results:
            print(f"  {r['cid']}: {r['n_atoms']} atoms, {r['n_bonds']} bonds, RMSD={r['rmsd']:.4f} A")

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    # Run demo with 3 example molecules
    results = reconstruction_demo(
        checkpoint_path="models/final_model.pt",
        csv_path="data/Compound_000000001_000025000_props_geom_3.csv",
        config_path="config.yaml",
        molecule_indices=[0, 1, 2],  # First 3 valid molecules
        output_dir="reconstruction_results"
    )

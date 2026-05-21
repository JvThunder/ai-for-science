#!/usr/bin/env python3
"""
3D Molecular Geometry Reconstruction from Trained GCN Model

This script loads a trained LocalGeometryGCN model and reconstructs 3D molecular
geometries from the predicted local spherical coordinates (r, theta, delta_phi).

Usage:
    uv run python -m src.reconstruction.reconstruction
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
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import imageio
import io

# Import utilities from the project
from src.utils.model import LocalGeometryGCN
from src.utils.dataset import parse_xyz_string, parse_bonds_from_dataset, parse_pubchem_data, _pubchem_mapping_has_error, get_element_vocab, dfs_subgraph
from src.utils.geometry import compute_local_geometry, spherical_to_cartesian, create_orthonormal_frame, get_neighbor_sort_keys
from src.utils.config import load_config
from src.utils.loss import total_loss, percentage_loss, geometry_loss


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
        max_bonds=config['data']['max_bonds'],
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
    print(f"  Max bonds: {config['data']['max_bonds']}")

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

    # Find valid molecules (geometry_xyz always required for 3D coords)
    valid_rows = []
    for idx, row in df.iterrows():
        if 'geometry_xyz' not in row or pd.isna(row['geometry_xyz']):
            continue
        use_pubchem = (
            'Pubchem_to_xyz_aid_mapping' in row
            and not _pubchem_mapping_has_error(row.get('Pubchem_to_xyz_aid_mapping'))
        )
        if use_pubchem:
            valid_rows.append((idx, row, True))
        elif row.get('bonding_status') == 'OK' and not pd.isna(row.get('bonds_geom_valence', float('nan'))):
            valid_rows.append((idx, row, False))

    if molecule_idx >= len(valid_rows):
        raise IndexError(f"Molecule index {molecule_idx} out of range (only {len(valid_rows)} valid molecules)")

    idx, row, use_pubchem = valid_rows[molecule_idx]

    xyz_coordinates, xyz_elements = parse_xyz_string(row['geometry_xyz'])

    if use_pubchem:
        coordinates, elements, bonds_data = parse_pubchem_data(row, xyz_coordinates)
        if not coordinates or not bonds_data:
            # Fallback if PubChem parsing failed
            coordinates, elements = xyz_coordinates, xyz_elements
            bonds_data = parse_bonds_from_dataset(row['bonds_geom_valence'])
    else:
        coordinates, elements = xyz_coordinates, xyz_elements
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


def find_center_atom(adjacency: Dict[int, List[int]], n_atoms: int) -> int:
    """
    Find the most central atom in the molecule.
    
    Uses degree centrality (number of bonds) as primary criterion.
    For ties, uses the atom with the smallest index (deterministic).
    
    Args:
        adjacency: Adjacency list mapping atom index to list of neighbors
        n_atoms: Total number of atoms
        
    Returns:
        center_idx: Index of the most central atom
    """
    max_degree = -1
    center_candidates = []
    
    for atom_idx in range(n_atoms):
        degree = len(adjacency.get(atom_idx, []))
        if degree > max_degree:
            max_degree = degree
            center_candidates = [atom_idx]
        elif degree == max_degree:
            center_candidates.append(atom_idx)
    
    # Return the atom with smallest index (deterministic)
    return min(center_candidates)


def predict_bond_distance_multiple(
    model: LocalGeometryGCN,
    coordinates: List[List[float]],
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    atom_a: int,
    atom_b: int,
    max_bonds: int,
    device: torch.device,
    n_predictions: int = 3
) -> float:
    """
    Predict bond distance by averaging multiple predictions.
    
    This improves accuracy by averaging over multiple forward passes
    and potentially different subgraph extractions.
    
    Args:
        model: Trained model
        coordinates: Atom positions
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        atom_a, atom_b: Bond endpoints
        max_bonds: Maximum bonds per atom
        device: torch device
        n_predictions: Number of predictions to average
        
    Returns:
        avg_distance: Average predicted distance
    """
    distances = []
    
    for _ in range(n_predictions):
        data, _, _ = create_graph_data(
            coordinates, elements, bonds_data, element_to_idx,
            atom_a, atom_b, max_bonds, device=device
        )
        pred_dist, _, _ = predict_bond_geometry(model, data, device)
        distances.append(pred_dist)
    
    return np.mean(distances)


def distance_geometry_reconstruction(
    model: LocalGeometryGCN,
    coordinates: List[List[float]],
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    max_iterations: int = 500
) -> np.ndarray:
    """
    Reconstruct molecule using pure distance geometry optimization.

    This approach predicts ALL bond distances first, then optimizes all atom
    positions simultaneously to match these distances. This avoids error
    accumulation from sequential BFS placement.

    Args:
        model: Trained model
        coordinates: Original atom positions (for initialization)
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        max_iterations: Maximum optimization iterations

    Returns:
        optimized: (N, 3) optimized positions
    """
    from scipy.optimize import minimize

    n_atoms = len(coordinates)
    original_pos = np.array(coordinates)

    # Build adjacency and collect bonds
    adjacency = defaultdict(list)
    bond_list = []

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)
                if i < j:
                    bond_list.append((i, j))

    if not bond_list:
        return original_pos.copy()

    # Predict ALL bond distances using model
    predicted_distances = {}
    for i, j in bond_list:
        data, _, _ = create_graph_data(
            coordinates, elements, bonds_data, element_to_idx,
            i, j, max_bonds, device=device
        )
        pred_dist, _, _ = predict_bond_geometry(model, data, device)
        predicted_distances[(i, j)] = pred_dist
        predicted_distances[(j, i)] = pred_dist

    # Find anchor atoms (highest degree atoms that are connected)
    atom_degrees = [(i, len(adjacency[i])) for i in range(n_atoms)]
    atom_degrees.sort(key=lambda x: -x[1])

    # Use top 3 connected atoms as anchors (fix their positions)
    anchor1 = atom_degrees[0][0]
    anchor2 = None
    anchor3 = None

    for atom, _ in atom_degrees[1:]:
        if atom in adjacency[anchor1]:
            anchor2 = atom
            break
    if anchor2 is None:
        anchor2 = atom_degrees[1][0]

    # Find third anchor not collinear with first two
    v1 = original_pos[anchor2] - original_pos[anchor1]
    for atom, _ in atom_degrees:
        if atom in [anchor1, anchor2]:
            continue
        if atom in adjacency[anchor1] or atom in adjacency[anchor2]:
            v2 = original_pos[atom] - original_pos[anchor1]
            cross = np.cross(v1, v2)
            if np.linalg.norm(cross) > 0.1:
                anchor3 = atom
                break
    if anchor3 is None:
        anchor3 = atom_degrees[2][0] if len(atom_degrees) > 2 else None

    anchors = [anchor1, anchor2]
    if anchor3 is not None:
        anchors.append(anchor3)

    # Initialize positions from original (good starting point)
    positions = original_pos.copy()

    # Flatten for optimization (exclude anchors)
    free_atoms = [i for i in range(n_atoms) if i not in anchors]
    n_free = len(free_atoms)

    if n_free == 0:
        return original_pos.copy()

    # Map atom indices to optimization variable indices
    atom_to_var = {atom: i for i, atom in enumerate(free_atoms)}

    x0 = positions[free_atoms].flatten()

    def objective(x):
        """Total distance error."""
        pos = np.zeros((n_atoms, 3))
        # Set anchor positions
        for anchor in anchors:
            pos[anchor] = original_pos[anchor]
        # Set free atom positions
        free_pos = x.reshape(-1, 3)
        for i, atom in enumerate(free_atoms):
            pos[atom] = free_pos[i]

        total_loss = 0.0

        # Distance constraints for all bonds
        for i, j in bond_list:
            if (i, j) not in predicted_distances:
                continue

            pred_dist = predicted_distances[(i, j)]
            current_dist = np.linalg.norm(pos[j] - pos[i])

            # Relative squared error
            error = ((current_dist - pred_dist) / (pred_dist + 1e-6)) ** 2
            total_loss += error

        return total_loss

    def gradient(x):
        """Analytical gradient for faster optimization."""
        pos = np.zeros((n_atoms, 3))
        for anchor in anchors:
            pos[anchor] = original_pos[anchor]
        free_pos = x.reshape(-1, 3)
        for i, atom in enumerate(free_atoms):
            pos[atom] = free_pos[i]

        grad = np.zeros((n_free, 3))

        for i, j in bond_list:
            if (i, j) not in predicted_distances:
                continue

            pred_dist = predicted_distances[(i, j)]
            diff = pos[j] - pos[i]
            current_dist = np.linalg.norm(diff)

            if current_dist < 1e-8:
                continue

            # d(error)/d(pos) = 2 * (current - pred) / pred^2 * direction
            coeff = 2.0 * (current_dist - pred_dist) / ((pred_dist + 1e-6) ** 2 * current_dist)
            direction = diff  # j - i

            # Gradient contribution
            if i in atom_to_var:
                grad[atom_to_var[i]] -= coeff * direction
            if j in atom_to_var:
                grad[atom_to_var[j]] += coeff * direction

        return grad.flatten()

    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': max_iterations, 'ftol': 1e-10, 'gtol': 1e-8}
    )

    # Build final positions
    optimized = np.zeros((n_atoms, 3))
    for anchor in anchors:
        optimized[anchor] = original_pos[anchor]
    final_pos = result.x.reshape(-1, 3)
    for i, atom in enumerate(free_atoms):
        optimized[atom] = final_pos[i]

    return optimized


def global_optimization_refinement(
    reconstructed: np.ndarray,
    predicted_distances: Dict[Tuple[int, int], float],
    predicted_angles: Dict[Tuple[int, int, int], float],
    bond_list: List[Tuple[int, int]],
    adjacency: Dict[int, List[int]],
    placed: np.ndarray,
    max_iterations: int = 100,
    distance_weight: float = 1.0,
    angle_weight: float = 0.5
) -> np.ndarray:
    """
    Global optimization to refine all atom positions simultaneously.

    Uses scipy.optimize.minimize with L-BFGS-B to find positions that best
    satisfy all predicted bond distances and angles.

    Args:
        reconstructed: (N, 3) initial positions
        predicted_distances: Dict mapping (i, j) to predicted distance
        predicted_angles: Dict mapping (i, j, k) to predicted angle in radians
        bond_list: List of (i, j) bond pairs
        adjacency: Adjacency list
        placed: Boolean array indicating which atoms are placed
        max_iterations: Maximum optimization iterations
        distance_weight: Weight for distance constraints
        angle_weight: Weight for angle constraints

    Returns:
        optimized: (N, 3) optimized positions
    """
    from scipy.optimize import minimize

    n_atoms = len(reconstructed)
    placed_indices = np.where(placed)[0]
    n_placed = len(placed_indices)

    if n_placed < 3:
        return reconstructed.copy()

    # Create mapping for optimization (only optimize placed atoms)
    idx_to_opt = {idx: i for i, idx in enumerate(placed_indices)}

    # Initial positions (flattened)
    x0 = reconstructed[placed_indices].flatten()

    def objective(x):
        """Compute total error: distance + angle constraints."""
        positions = x.reshape(-1, 3)
        total_loss = 0.0

        # Distance constraints
        for i, j in bond_list:
            if i not in idx_to_opt or j not in idx_to_opt:
                continue
            if (i, j) not in predicted_distances and (j, i) not in predicted_distances:
                continue

            pred_dist = predicted_distances.get((i, j), predicted_distances.get((j, i)))
            pos_i = positions[idx_to_opt[i]]
            pos_j = positions[idx_to_opt[j]]

            current_dist = np.linalg.norm(pos_j - pos_i)
            # Relative error squared
            dist_error = ((current_dist - pred_dist) / (pred_dist + 1e-6)) ** 2
            total_loss += distance_weight * dist_error

        # Angle constraints (bond angles at each atom)
        for center in placed_indices:
            if center not in idx_to_opt:
                continue
            neighbors = [n for n in adjacency.get(center, []) if n in idx_to_opt]
            if len(neighbors) < 2:
                continue

            pos_center = positions[idx_to_opt[center]]

            for i_n, n1 in enumerate(neighbors):
                for n2 in neighbors[i_n+1:]:
                    key = (n1, center, n2)
                    rev_key = (n2, center, n1)

                    if key not in predicted_angles and rev_key not in predicted_angles:
                        continue

                    pred_angle = predicted_angles.get(key, predicted_angles.get(rev_key))

                    pos_n1 = positions[idx_to_opt[n1]]
                    pos_n2 = positions[idx_to_opt[n2]]

                    v1 = pos_n1 - pos_center
                    v2 = pos_n2 - pos_center

                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)

                    if norm1 < 1e-6 or norm2 < 1e-6:
                        continue

                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    current_angle = np.arccos(cos_angle)

                    # Angular error (in radians)
                    angle_error = (current_angle - pred_angle) ** 2
                    total_loss += angle_weight * angle_error

        return total_loss

    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iterations, 'ftol': 1e-8, 'gtol': 1e-6}
    )

    # Update positions
    optimized = reconstructed.copy()
    optimized_positions = result.x.reshape(-1, 3)

    for i, idx in enumerate(placed_indices):
        optimized[idx] = optimized_positions[i]

    return optimized


def reconstruct_from_center(
    model: LocalGeometryGCN,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    start_atom: int
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]], List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Reconstruct molecule starting from a specific atom without using original coordinates.
    
    Places first atom at origin and second along x-axis based on model predictions.
    Translation and rotation alignment is handled during RMSD calculation.

    Returns reconstructed coordinates and metadata for further processing.
    """
    n_atoms = len(elements)

    # Build adjacency list and bond orders
    adjacency = defaultdict(list)
    bond_list = []
    bond_orders = {}

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)
                if i < j:
                    bond_list.append((i, j))
                bond_order = bond.get('order', 1.0)
                bond_orders[(i, j)] = bond_order
                bond_orders[(j, i)] = bond_order

    if not bond_list:
        # No bonds - place all atoms at origin
        return np.zeros((n_atoms, 3)), np.ones(n_atoms, dtype=bool), adjacency, bond_list, {}

    # Initialize
    reconstructed = np.zeros((n_atoms, 3))
    placed = np.zeros(n_atoms, dtype=bool)
    predicted_distances = {}

    # Find a neighbor to establish coordinate frame
    if not adjacency[start_atom]:
        return np.zeros((n_atoms, 3)), placed, adjacency, bond_list, predicted_distances

    # Use first neighbor as reference
    ref_atom = adjacency[start_atom][0]

    # Place first atom at origin
    reconstructed[start_atom] = [0, 0, 0]
    placed[start_atom] = True

    # Get predicted distance for first bond
    # Use temporary coordinates (will be updated as we build)
    temp_coords = reconstructed.copy()
    data, _, _ = create_graph_data(
        temp_coords.tolist(), elements, bonds_data, element_to_idx,
        start_atom, ref_atom, max_bonds, device=device
    )
    pred_dist, _, _ = predict_bond_geometry(model, data, device)
    predicted_distances[(start_atom, ref_atom)] = pred_dist
    predicted_distances[(ref_atom, start_atom)] = pred_dist

    # Place reference atom along x-axis
    reconstructed[ref_atom] = [pred_dist, 0, 0]
    placed[ref_atom] = True

    # BFS to place remaining atoms
    queue = [start_atom, ref_atom]
    visited_bonds = {(min(start_atom, ref_atom), max(start_atom, ref_atom))}

    while queue:
        center = queue.pop(0)
        neighbors = adjacency[center]

        if len(neighbors) == 0:
            continue

        placed_neighbors = [n for n in neighbors if placed[n]]
        has_unplaced = any(not placed[n] for n in neighbors)

        if not placed_neighbors or not has_unplaced:
            continue

        ref_atom_local = placed_neighbors[0]

        # Get geometry predictions using reconstructed positions
        data, node_mapping, node_list = create_graph_data(
            reconstructed.tolist(), elements, bonds_data, element_to_idx,
            center, ref_atom_local, max_bonds, device=device
        )
        _, geometry_center, _ = predict_bond_geometry(model, data, device)

        # Create local coordinate frame
        center_pos = torch.tensor(reconstructed[center], dtype=torch.float)
        ref_pos = torch.tensor(reconstructed[ref_atom_local], dtype=torch.float)

        bond_vec = ref_pos - center_pos
        bond_length = torch.norm(bond_vec)

        if bond_length < 1e-6:
            continue

        bond_axis = bond_vec / bond_length
        x_axis, y_axis = create_orthonormal_frame(bond_axis)
        coord_frame = (x_axis, y_axis, bond_axis)

        # Get sorted neighbors
        all_neighbors = neighbors.copy()
        sort_keys = get_neighbor_sort_keys(all_neighbors, elements, bond_orders, center)
        sorted_neighbors_with_keys = sorted(zip(all_neighbors, sort_keys), key=lambda x: x[1])
        sorted_neighbors = [n for n, _ in sorted_neighbors_with_keys]
        sorted_neighbors_excl_ref = [n for n in sorted_neighbors if n != ref_atom_local]

        # Place unplaced neighbors
        cumulative_phi = 0.0

        for slot_idx, neighbor in enumerate(sorted_neighbors_excl_ref):
            if slot_idx >= max_bonds:
                break

            if placed[neighbor]:
                if slot_idx > 0:
                    delta_phi_deg = geometry_center[slot_idx, 1]
                    cumulative_phi += delta_phi_deg
                continue

            theta_deg = geometry_center[slot_idx, 0]

            if slot_idx == 0:
                phi = 0.0
            else:
                delta_phi_deg = geometry_center[slot_idx, 1]
                cumulative_phi += delta_phi_deg
                phi = math.radians(cumulative_phi % 360.0)

            theta = math.radians(theta_deg)

            # Get predicted distance
            bond_key = (min(center, neighbor), max(center, neighbor))
            if bond_key not in visited_bonds:
                data_dist, _, _ = create_graph_data(
                    reconstructed.tolist(), elements, bonds_data, element_to_idx,
                    center, neighbor, max_bonds, device=device
                )
                pred_r, _, _ = predict_bond_geometry(model, data_dist, device)
                predicted_distances[(center, neighbor)] = pred_r
                predicted_distances[(neighbor, center)] = pred_r
                visited_bonds.add(bond_key)
            else:
                # Use previously predicted distance
                pred_r = predicted_distances[(center, neighbor)]

            # Convert to Cartesian
            new_pos = spherical_to_cartesian(pred_r, theta, phi, coord_frame, center_pos)
            reconstructed[neighbor] = new_pos.numpy()
            placed[neighbor] = True
            queue.append(neighbor)

    return reconstructed, placed, adjacency, bond_list, predicted_distances


def reconstruct_with_bfs_tracking(
    model: LocalGeometryGCN,
    coordinates: List[List[float]],
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    start_atom: Optional[int] = None
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[np.ndarray]]:
    """
    Reconstruct molecule with BFS tracking for animation.

    Returns the reconstruction along with the order of bonds placed
    and intermediate snapshots of atom positions.

    Args:
        model: Trained model
        coordinates: Original atom positions
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        start_atom: Starting atom index (if None, uses most central atom)

    Returns:
        final_positions: (N, 3) final reconstructed positions
        bfs_bond_order: List of (atom_a, atom_b) tuples in BFS order
        position_snapshots: List of (N, 3) arrays showing positions after each bond
    """
    n_atoms = len(coordinates)
    original_pos = np.array(coordinates)

    # Build adjacency list and bond orders
    adjacency = defaultdict(list)
    bond_orders = {}

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)
                bond_order = bond.get('order', 1.0)
                bond_orders[(i, j)] = bond_order
                bond_orders[(j, i)] = bond_order

    # Find center atom if not specified
    if start_atom is None:
        start_atom = find_center_atom(adjacency, n_atoms)

    # Initialize
    reconstructed = np.zeros((n_atoms, 3))
    placed = np.zeros(n_atoms, dtype=bool)
    bfs_bond_order = []  # Tracks bonds in BFS order
    position_snapshots = []  # Snapshots after each bond placement
    predicted_distances = {}

    # Find a neighbor to establish coordinate frame
    if not adjacency[start_atom]:
        return original_pos.copy(), [], [original_pos.copy()]

    # Use first neighbor as reference
    ref_atom = adjacency[start_atom][0]

    # ANCHOR APPROACH: Use original coordinates for first 2-3 atoms
    reconstructed[start_atom] = original_pos[start_atom].copy()
    placed[start_atom] = True

    reconstructed[ref_atom] = original_pos[ref_atom].copy()
    placed[ref_atom] = True

    # Record first bond
    bfs_bond_order.append((start_atom, ref_atom))
    position_snapshots.append(reconstructed.copy())

    # Store original distance
    orig_dist = np.linalg.norm(original_pos[start_atom] - original_pos[ref_atom])
    predicted_distances[(start_atom, ref_atom)] = orig_dist
    predicted_distances[(ref_atom, start_atom)] = orig_dist

    # Try to find a 3rd atom to fully constrain the coordinate frame
    third_atom = None
    for neighbor in adjacency[start_atom]:
        if neighbor != ref_atom:
            v1 = original_pos[ref_atom] - original_pos[start_atom]
            v2 = original_pos[neighbor] - original_pos[start_atom]
            cross = np.cross(v1, v2)
            if np.linalg.norm(cross) > 0.1:
                third_atom = neighbor
                break

    if third_atom is None:
        for neighbor in adjacency[ref_atom]:
            if neighbor != start_atom:
                v1 = original_pos[start_atom] - original_pos[ref_atom]
                v2 = original_pos[neighbor] - original_pos[ref_atom]
                cross = np.cross(v1, v2)
                if np.linalg.norm(cross) > 0.1:
                    third_atom = neighbor
                    break

    if third_atom is not None:
        reconstructed[third_atom] = original_pos[third_atom].copy()
        placed[third_atom] = True

        # Record bond to third atom
        if third_atom in adjacency[start_atom]:
            bfs_bond_order.append((start_atom, third_atom))
        else:
            bfs_bond_order.append((ref_atom, third_atom))
        position_snapshots.append(reconstructed.copy())

        for anchor in [start_atom, ref_atom]:
            if third_atom in adjacency[anchor]:
                dist = np.linalg.norm(original_pos[anchor] - original_pos[third_atom])
                predicted_distances[(anchor, third_atom)] = dist
                predicted_distances[(third_atom, anchor)] = dist

    # BFS to place remaining atoms
    queue = deque([start_atom, ref_atom])
    if third_atom is not None:
        queue.append(third_atom)

    visited_bonds = set()
    visited_bonds.add((min(start_atom, ref_atom), max(start_atom, ref_atom)))
    if third_atom is not None:
        if third_atom in adjacency[start_atom]:
            visited_bonds.add((min(start_atom, third_atom), max(start_atom, third_atom)))
        if third_atom in adjacency[ref_atom]:
            visited_bonds.add((min(ref_atom, third_atom), max(ref_atom, third_atom)))

    while queue:
        center = queue.popleft()
        neighbors = adjacency[center]

        if len(neighbors) == 0:
            continue

        placed_neighbors = [n for n in neighbors if placed[n]]
        has_unplaced = any(not placed[n] for n in neighbors)

        if not placed_neighbors or not has_unplaced:
            continue

        ref_atom_local = placed_neighbors[0]

        # Get geometry predictions
        data, node_mapping, node_list = create_graph_data(
            coordinates, elements, bonds_data, element_to_idx,
            center, ref_atom_local, max_bonds, device=device
        )
        _, geometry_center, _ = predict_bond_geometry(model, data, device)

        # Create local coordinate frame
        center_pos = torch.tensor(reconstructed[center], dtype=torch.float)
        ref_pos = torch.tensor(reconstructed[ref_atom_local], dtype=torch.float)

        bond_vec = ref_pos - center_pos
        bond_length = torch.norm(bond_vec)

        if bond_length < 1e-6:
            continue

        bond_axis = bond_vec / bond_length
        x_axis, y_axis = create_orthonormal_frame(bond_axis)
        coord_frame = (x_axis, y_axis, bond_axis)

        # Get sorted neighbors
        all_neighbors = neighbors.copy()
        sort_keys = get_neighbor_sort_keys(all_neighbors, elements, bond_orders, center)
        sorted_neighbors_with_keys = sorted(zip(all_neighbors, sort_keys), key=lambda x: x[1])
        sorted_neighbors = [n for n, _ in sorted_neighbors_with_keys]
        sorted_neighbors_excl_ref = [n for n in sorted_neighbors if n != ref_atom_local]

        # Place unplaced neighbors
        cumulative_phi = 0.0

        for slot_idx, neighbor in enumerate(sorted_neighbors_excl_ref):
            if slot_idx >= max_bonds:
                break

            if placed[neighbor]:
                if slot_idx > 0:
                    delta_phi_deg = geometry_center[slot_idx, 1]
                    cumulative_phi += delta_phi_deg
                continue

            theta_deg = geometry_center[slot_idx, 0]

            if slot_idx == 0:
                phi = 0.0
            else:
                delta_phi_deg = geometry_center[slot_idx, 1]
                cumulative_phi += delta_phi_deg
                phi = math.radians(cumulative_phi % 360.0)

            theta = math.radians(theta_deg)

            # Get predicted distance
            bond_key = (min(center, neighbor), max(center, neighbor))
            if bond_key not in visited_bonds:
                data_dist, _, _ = create_graph_data(
                    coordinates, elements, bonds_data, element_to_idx,
                    center, neighbor, max_bonds, device=device
                )
                pred_r, _, _ = predict_bond_geometry(model, data_dist, device)
                predicted_distances[(center, neighbor)] = pred_r
                predicted_distances[(neighbor, center)] = pred_r
                visited_bonds.add(bond_key)
            else:
                pred_r = predicted_distances.get((center, neighbor),
                         np.linalg.norm(np.array(coordinates[center]) - np.array(coordinates[neighbor])))

            # Convert to Cartesian
            new_pos = spherical_to_cartesian(pred_r, theta, phi, coord_frame, center_pos)
            reconstructed[neighbor] = new_pos.numpy()
            placed[neighbor] = True
            queue.append(neighbor)

            # Record this bond in BFS order
            bfs_bond_order.append((center, neighbor))
            position_snapshots.append(reconstructed.copy())

    # Fill unplaced atoms with original coordinates
    for atom_idx in range(n_atoms):
        if not placed[atom_idx]:
            reconstructed[atom_idx] = original_pos[atom_idx]

    return reconstructed, bfs_bond_order, position_snapshots


def get_bfs_bond_order(
    bonds_data: List[Dict],
    n_atoms: int,
    start_atom: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Get BFS traversal order of bonds without doing reconstruction.

    Args:
        bonds_data: Bond information
        n_atoms: Number of atoms
        start_atom: Starting atom (if None, uses most connected atom)

    Returns:
        bfs_bond_order: List of (atom_a, atom_b) in BFS traversal order
    """
    # Build adjacency
    adjacency = defaultdict(list)
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)

    if not adjacency:
        return []

    # Find start atom (most connected) if not specified
    if start_atom is None:
        start_atom = max(range(n_atoms), key=lambda x: len(adjacency.get(x, [])))

    # BFS traversal
    visited_atoms = {start_atom}
    visited_bonds = set()
    bfs_bond_order = []
    queue = deque([start_atom])

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            bond_key = (min(current, neighbor), max(current, neighbor))
            if bond_key not in visited_bonds:
                visited_bonds.add(bond_key)
                bfs_bond_order.append((current, neighbor))
                if neighbor not in visited_atoms:
                    visited_atoms.add(neighbor)
                    queue.append(neighbor)

    return bfs_bond_order


def create_bfs_animation_gif(
    original_coords: np.ndarray,
    reconstructed_coords: np.ndarray,
    elements: List[str],
    bonds_data: List[Dict],
    bfs_bond_order: List[Tuple[int, int]],
    cid: str,
    save_path: str,
    fps: int = 2,
    show_original: bool = True
):
    """
    Create an animated GIF showing bonds being highlighted in BFS order on the final structure.

    The GIF shows the complete reconstructed molecule with bonds highlighted one by one
    in BFS traversal order, displaying distance and angle information.

    Args:
        original_coords: (N, 3) original coordinates
        reconstructed_coords: (N, 3) final reconstructed coordinates
        elements: Element symbols
        bonds_data: Bond information
        bfs_bond_order: List of (atom_a, atom_b) in BFS placement order
        cid: Compound ID
        save_path: Path to save the GIF
        fps: Frames per second
        show_original: If True, show original structure on the left
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

    # Get colors and sizes for atoms
    colors = [element_colors.get(e, default_color) for e in elements]
    sizes = [element_sizes.get(e, default_size) for e in elements]

    # Parse all bonds and build adjacency
    all_bonds = set()
    adjacency = defaultdict(list)
    bond_orders_dict = {}
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < len(elements) and j < len(elements):
                all_bonds.add((min(i, j), max(i, j)))
                adjacency[i].append(j)
                adjacency[j].append(i)
                bond_order = bond.get('order', 1.0)
                bond_orders_dict[(i, j)] = bond_order
                bond_orders_dict[(j, i)] = bond_order

    # Compute axis limits from all coordinates
    all_coords_combined = np.vstack([original_coords, reconstructed_coords])
    max_range = np.max(np.ptp(all_coords_combined, axis=0)) / 2 * 1.2
    mid = all_coords_combined.mean(axis=0)

    # Precompute bond distances and angles for annotations
    def compute_bond_angle(pos, atom_i, atom_center, atom_j):
        """Compute angle i-center-j in degrees."""
        v1 = pos[atom_i] - pos[atom_center]
        v2 = pos[atom_j] - pos[atom_center]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    frames = []

    # Create frames for each bond being highlighted
    for frame_idx in range(len(bfs_bond_order)):
        # Current highlighted bond
        hi, hj = bfs_bond_order[frame_idx]

        # Compute distance for this bond
        bond_dist = np.linalg.norm(reconstructed_coords[hi] - reconstructed_coords[hj])

        # Find angles at both atoms
        angles_info = []

        # Angles at atom hi (with hj as one arm)
        neighbors_hi = [n for n in adjacency[hi] if n != hj]
        for neighbor in neighbors_hi[:2]:  # Show up to 2 angles
            angle = compute_bond_angle(reconstructed_coords, neighbor, hi, hj)
            angles_info.append(f"{elements[neighbor]}{neighbor}-{elements[hi]}{hi}-{elements[hj]}{hj}: {angle:.1f}°")

        # Angles at atom hj (with hi as one arm)
        neighbors_hj = [n for n in adjacency[hj] if n != hi]
        for neighbor in neighbors_hj[:2]:  # Show up to 2 angles
            angle = compute_bond_angle(reconstructed_coords, hi, hj, neighbor)
            angles_info.append(f"{elements[hi]}{hi}-{elements[hj]}{hj}-{elements[neighbor]}{neighbor}: {angle:.1f}°")

        # Create figure
        if show_original:
            fig = plt.figure(figsize=(16, 7))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            fig = plt.figure(figsize=(10, 8))
            ax2 = fig.add_subplot(111, projection='3d')
            ax1 = None

        # Plot original structure (static, all bonds shown)
        if ax1 is not None:
            ax1.scatter(
                original_coords[:, 0], original_coords[:, 1], original_coords[:, 2],
                c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
            )
            for i, j in all_bonds:
                ax1.plot(
                    [original_coords[i, 0], original_coords[j, 0]],
                    [original_coords[i, 1], original_coords[j, 1]],
                    [original_coords[i, 2], original_coords[j, 2]],
                    'k-', linewidth=2, alpha=0.6
                )
            ax1.set_xlabel('X (Å)', fontsize=10)
            ax1.set_ylabel('Y (Å)', fontsize=10)
            ax1.set_zlabel('Z (Å)', fontsize=10)
            ax1.set_title(f'Original Structure\n{cid}', fontsize=12, fontweight='bold')
            ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # Determine which bonds and atoms are visible
        visited_bonds = set()
        visible_atoms = set()
        for bond in bfs_bond_order[:frame_idx + 1]:  # Include current bond
            visited_bonds.add((min(bond[0], bond[1]), max(bond[0], bond[1])))
            visible_atoms.add(bond[0])
            visible_atoms.add(bond[1])

        # Plot only visible atoms (those connected by visited bonds)
        for atom_idx in visible_atoms:
            ax2.scatter(
                [reconstructed_coords[atom_idx, 0]],
                [reconstructed_coords[atom_idx, 1]],
                [reconstructed_coords[atom_idx, 2]],
                c=[colors[atom_idx]], s=[sizes[atom_idx]],
                edgecolors='black', linewidths=0.5, alpha=0.9
            )

        # Draw only visited bonds (not yet visited bonds are invisible)
        highlight_key = (min(hi, hj), max(hi, hj))
        for i, j in all_bonds:
            bond_key = (min(i, j), max(i, j))

            if bond_key not in visited_bonds:
                # Not yet visited - don't draw
                continue
            elif bond_key == highlight_key:
                # Current bond - bright red, thick
                ax2.plot(
                    [reconstructed_coords[i, 0], reconstructed_coords[j, 0]],
                    [reconstructed_coords[i, 1], reconstructed_coords[j, 1]],
                    [reconstructed_coords[i, 2], reconstructed_coords[j, 2]],
                    color='#FF0000', linewidth=4, alpha=1.0
                )
            else:
                # Previously visited - green
                ax2.plot(
                    [reconstructed_coords[i, 0], reconstructed_coords[j, 0]],
                    [reconstructed_coords[i, 1], reconstructed_coords[j, 1]],
                    [reconstructed_coords[i, 2], reconstructed_coords[j, 2]],
                    color='#00AA00', linewidth=2.5, alpha=0.8
                )

        # Highlight the two atoms of the current bond
        for atom_idx in [hi, hj]:
            ax2.scatter(
                [reconstructed_coords[atom_idx, 0]],
                [reconstructed_coords[atom_idx, 1]],
                [reconstructed_coords[atom_idx, 2]],
                c='yellow', s=sizes[atom_idx] * 1.5, edgecolors='red',
                linewidths=2, alpha=0.7, zorder=10
            )

        ax2.set_xlabel('X (Å)', fontsize=10)
        ax2.set_ylabel('Y (Å)', fontsize=10)
        ax2.set_zlabel('Z (Å)', fontsize=10)

        # Build info text
        bond_type_str = ""
        bond_order = bond_orders_dict.get((hi, hj), 1.0)
        if bond_order == 1.0:
            bond_type_str = "Single"
        elif bond_order == 1.5:
            bond_type_str = "Aromatic"
        elif bond_order == 2.0:
            bond_type_str = "Double"
        elif bond_order == 3.0:
            bond_type_str = "Triple"

        title_text = f'BFS Reconstruction - Bond {frame_idx + 1}/{len(bfs_bond_order)}\n'
        title_text += f'{elements[hi]}{hi} — {elements[hj]}{hj} ({bond_type_str})\n'
        title_text += f'Distance: {bond_dist:.3f} Å'

        ax2.set_title(title_text, fontsize=11, fontweight='bold')
        ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # Add angle info as text annotation (fixed position)
        if angles_info:
            angle_text = "\n".join(angles_info[:3])  # Max 3 angles
            ax2.text2D(0.02, 0.02, f"Angles:\n{angle_text}", transform=ax2.transAxes,
                      fontsize=8, family='monospace', verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for angle text

        # Save frame to buffer (fixed size, no tight bbox)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Add final frame showing complete structure (hold for longer)
    for _ in range(4):  # Hold final frame for 4x duration
        fig = plt.figure(figsize=(16, 7) if show_original else (10, 8))

        if show_original:
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            ax2 = fig.add_subplot(111, projection='3d')
            ax1 = None

        if ax1 is not None:
            ax1.scatter(
                original_coords[:, 0], original_coords[:, 1], original_coords[:, 2],
                c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
            )
            for i, j in all_bonds:
                ax1.plot(
                    [original_coords[i, 0], original_coords[j, 0]],
                    [original_coords[i, 1], original_coords[j, 1]],
                    [original_coords[i, 2], original_coords[j, 2]],
                    'k-', linewidth=2, alpha=0.6
                )
            ax1.set_xlabel('X (Å)', fontsize=10)
            ax1.set_ylabel('Y (Å)', fontsize=10)
            ax1.set_zlabel('Z (Å)', fontsize=10)
            ax1.set_title(f'Original Structure\n{cid}', fontsize=12, fontweight='bold')
            ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # Complete reconstruction - all bonds green
        ax2.scatter(
            reconstructed_coords[:, 0], reconstructed_coords[:, 1], reconstructed_coords[:, 2],
            c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
        )
        for i, j in all_bonds:
            ax2.plot(
                [reconstructed_coords[i, 0], reconstructed_coords[j, 0]],
                [reconstructed_coords[i, 1], reconstructed_coords[j, 1]],
                [reconstructed_coords[i, 2], reconstructed_coords[j, 2]],
                color='#00AA00', linewidth=2.5, alpha=0.9
            )

        rmsd = compute_rmsd(align_molecules(reconstructed_coords, original_coords), original_coords)
        ax2.set_xlabel('X (Å)', fontsize=10)
        ax2.set_ylabel('Y (Å)', fontsize=10)
        ax2.set_zlabel('Z (Å)', fontsize=10)
        ax2.set_title(f'BFS Reconstruction Complete\nRMSD: {rmsd:.4f} Å\n{len(bfs_bond_order)} bonds traversed',
                     fontsize=11, fontweight='bold')
        ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Save GIF
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"Saved BFS animation GIF to {save_path}")


def reconstruct_molecule(
    model: LocalGeometryGCN,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int = 7,
    use_global_optimization: bool = True,
    n_multi_start: int = 2  ,
    use_distance_geometry: bool = False
) -> np.ndarray:
    """
    Reconstruct 3D molecular geometry using model predictions without original coordinates.

    Pure reconstruction approach:
    1. BFS-based placement: Build structure from scratch using model predictions
    2. Multi-start: Try multiple starting atoms and average results
    3. Global optimization: Refine positions to match predicted distances and angles
    
    Note: Distance geometry approach is disabled by default as it requires original coordinates.
    
    Args:
        model: Trained model
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        use_global_optimization: If True, use scipy.optimize for refinement
        n_multi_start: Number of different starting atoms to try
        use_distance_geometry: Distance geometry not supported without coordinates

    Returns:
        reconstructed_coords: (N, 3) numpy array of reconstructed positions
    """
    # Distance geometry requires original coordinates for initialization
    if use_distance_geometry:
        raise ValueError("Distance geometry reconstruction requires original coordinates. "
                        "Set use_distance_geometry=False to use BFS-based reconstruction.")

    n_atoms = len(elements)

    # Build adjacency list
    adjacency = defaultdict(list)
    bond_list = []
    bond_orders = {}

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)
                if i < j:
                    bond_list.append((i, j))
                bond_order = bond.get('order', 1.0)
                bond_orders[(i, j)] = bond_order
                bond_orders[(j, i)] = bond_order

    if not bond_list:
        # No bonds - place all atoms at origin
        return np.zeros((n_atoms, 3))

    # Find candidate starting atoms (sorted by degree, highest first)
    atom_degrees = [(i, len(adjacency[i])) for i in range(n_atoms) if adjacency[i]]
    atom_degrees.sort(key=lambda x: (-x[1], x[0]))

    # Select starting atoms for multi-start
    start_atoms = [atom_degrees[i][0] for i in range(min(n_multi_start, len(atom_degrees)))]

    # Collect reconstructions from multiple starts
    all_reconstructions = []
    all_predicted_distances = {}
    last_adjacency = None
    last_bond_list = None
    last_placed = None

    for start_atom in start_atoms:
        recon, placed, adj, bl, pred_dists = reconstruct_from_center(
            model, elements, bonds_data, element_to_idx,
            device, max_bonds, start_atom
        )

        # For multi-start: align to first reconstruction for averaging
        if len(all_reconstructions) > 0:
            aligned = align_molecules(recon, all_reconstructions[0])
            all_reconstructions.append(aligned)
        else:
            # First reconstruction - use as reference for others
            all_reconstructions.append(recon)

        # Merge predicted distances
        all_predicted_distances.update(pred_dists)
        last_adjacency = adj
        last_bond_list = bl
        last_placed = placed

    # Average all reconstructions (after alignment)
    if len(all_reconstructions) > 1:
        reconstructed = np.mean(all_reconstructions, axis=0)
    else:
        reconstructed = all_reconstructions[0]

    # Compute predicted angles from model for global optimization
    predicted_angles = {}

    for center in range(n_atoms):
        neighbors = list(adjacency.get(center, []))
        if len(neighbors) < 2:
            continue

        # For each pair of neighbors, predict the angle between them
        for i_n, n1 in enumerate(neighbors):
            for n2 in neighbors[i_n+1:]:
                # Get geometry prediction for center with n1 as reference using reconstructed coords
                try:
                    data, _, _ = create_graph_data(
                        reconstructed.tolist(), elements, bonds_data, element_to_idx,
                        center, n1, max_bonds, device=device
                    )
                    _, geometry_center, _ = predict_bond_geometry(model, data, device)

                    # Find n2 in the sorted neighbors (excluding n1)
                    other_neighbors = [n for n in neighbors if n != n1]
                    sort_keys = get_neighbor_sort_keys(other_neighbors, elements, bond_orders, center)
                    sorted_pairs = sorted(zip(other_neighbors, sort_keys), key=lambda x: x[1])
                    sorted_other = [n for n, _ in sorted_pairs]

                    if n2 in sorted_other:
                        slot_idx = sorted_other.index(n2)
                        if slot_idx < max_bonds:
                            # theta is angle from bond axis (n1-center direction)
                            theta_deg = geometry_center[slot_idx, 0]
                            # This gives us the angle center-n1-n2 indirectly
                            # The actual n1-center-n2 angle = 180 - theta (approximately)
                            # But more accurately, we should use the theta directly
                            angle_rad = math.radians(theta_deg)
                            predicted_angles[(n1, center, n2)] = angle_rad
                            predicted_angles[(n2, center, n1)] = angle_rad
                except Exception:
                    continue

    # Global optimization refinement
    if use_global_optimization and last_placed is not None:
        reconstructed = global_optimization_refinement(
            reconstructed,
            all_predicted_distances,
            predicted_angles,
            last_bond_list,
            last_adjacency,
            last_placed,
            max_iterations=200,
            distance_weight=1.0,
            angle_weight=0.3
        )

    # Return reconstructed coordinates (alignment to ground truth handled by caller)
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


def compute_reconstruction_loss(
    model: LocalGeometryGCN,
    original_coords: np.ndarray,
    reconstructed_coords: np.ndarray,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    config: Dict
) -> Dict[str, float]:
    """
    Compute reconstruction loss comparing predicted vs target geometry.
    
    Args:
        model: Trained model
        original_coords: (N, 3) original/true coordinates
        reconstructed_coords: (N, 3) reconstructed coordinates
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        config: Config dict with loss weights
        
    Returns:
        loss_dict: Dictionary with total loss and component losses
    """
    n_atoms = len(original_coords)
    original_pos = torch.tensor(original_coords, dtype=torch.float, device=device)
    reconstructed_pos = torch.tensor(reconstructed_coords, dtype=torch.float, device=device)
    
    # Build adjacency list and bond orders
    adjacency = defaultdict(list)
    bond_list = []
    bond_orders = {}
    
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)
                if i < j:
                    bond_list.append((i, j))
                bond_order = bond.get('order', 1.0)
                bond_orders[(i, j)] = bond_order
                bond_orders[(j, i)] = bond_order
    
    if not bond_list:
        return {
            'total': 0.0,
            'distance': 0.0,
            'geometry_a': 0.0,
            'geometry_b': 0.0,
            'geometry': 0.0,
            'n_bonds': 0
        }
    
    # Collect all predictions and targets
    all_pred_dists = []
    all_pred_geom_a = []
    all_pred_geom_b = []
    all_target_dists = []
    all_target_geom_a = []
    all_target_geom_b = []
    all_masks_a = []
    all_masks_b = []
    
    for i, j in bond_list:
        # Get model predictions
        data, node_mapping, node_list = create_graph_data(
            [c.tolist() for c in original_coords], elements, bonds_data,
            element_to_idx, i, j, max_bonds, device=device
        )
        
        pred_dist, pred_geom_a, pred_geom_b = predict_bond_geometry(model, data, device)
        
        # Compute target geometry from original coordinates
        # For node A: use A->B as reference, neighbors of A
        neighbors_a = [n for n in adjacency[i] if n != j]
        if neighbors_a:
            sort_keys_a = get_neighbor_sort_keys(neighbors_a, elements, bond_orders, i)
            ordered_a, geom_a, _ = compute_local_geometry(
                original_pos, i, neighbors_a, j,
                neighbor_sort_keys=sort_keys_a
            )
        else:
            ordered_a = []
            geom_a = []
        
        # For node B: use B->A as reference, neighbors of B
        neighbors_b = [n for n in adjacency[j] if n != i]
        if neighbors_b:
            sort_keys_b = get_neighbor_sort_keys(neighbors_b, elements, bond_orders, j)
            ordered_b, geom_b, _ = compute_local_geometry(
                original_pos, j, neighbors_b, i,
                neighbor_sort_keys=sort_keys_b
            )
        else:
            ordered_b = []
            geom_b = []
        
        # Target distance
        target_dist = torch.norm(original_pos[i] - original_pos[j]).item()
        
        # Convert geometry to (theta, delta_phi) format
        # Geometry is (r, theta, phi) - we need (theta, delta_phi) in degrees
        target_geom_a_array = np.zeros((max_bonds, 2))
        target_geom_b_array = np.zeros((max_bonds, 2))
        mask_a_array = np.zeros(max_bonds)
        mask_b_array = np.zeros(max_bonds)
        
        # Process geometry for A
        if geom_a:
            prev_phi = None
            for slot_idx, (r, theta, phi) in enumerate(geom_a[:max_bonds]):
                if slot_idx >= max_bonds:
                    break
                theta_deg = math.degrees(theta)
                phi_deg = math.degrees(phi)
                
                if prev_phi is None:
                    # First neighbor: delta_phi = 0
                    delta_phi_deg = 0.0
                    prev_phi = phi_deg
                else:
                    # Relative difference
                    delta_phi_deg = (phi_deg - prev_phi) % 360.0
                    prev_phi = phi_deg
                
                target_geom_a_array[slot_idx] = [theta_deg, delta_phi_deg]
                mask_a_array[slot_idx] = 1.0
        
        # Process geometry for B
        if geom_b:
            prev_phi = None
            for slot_idx, (r, theta, phi) in enumerate(geom_b[:max_bonds]):
                if slot_idx >= max_bonds:
                    break
                theta_deg = math.degrees(theta)
                phi_deg = math.degrees(phi)
                
                if prev_phi is None:
                    delta_phi_deg = 0.0
                    prev_phi = phi_deg
                else:
                    delta_phi_deg = (phi_deg - prev_phi) % 360.0
                    prev_phi = phi_deg
                
                target_geom_b_array[slot_idx] = [theta_deg, delta_phi_deg]
                mask_b_array[slot_idx] = 1.0
        
        # Store predictions and targets
        all_pred_dists.append(pred_dist)
        all_pred_geom_a.append(pred_geom_a)
        all_pred_geom_b.append(pred_geom_b)
        all_target_dists.append(target_dist)
        all_target_geom_a.append(target_geom_a_array)
        all_target_geom_b.append(target_geom_b_array)
        all_masks_a.append(mask_a_array)
        all_masks_b.append(mask_b_array)
    
    # Convert to tensors
    pred_dists_tensor = torch.tensor(all_pred_dists, dtype=torch.float, device=device).unsqueeze(1)  # (n_bonds, 1)
    target_dists_tensor = torch.tensor(all_target_dists, dtype=torch.float, device=device).unsqueeze(1)  # (n_bonds, 1)
    
    pred_geom_a_tensor = torch.tensor(np.stack(all_pred_geom_a), dtype=torch.float, device=device)  # (n_bonds, max_bonds, 2)
    pred_geom_b_tensor = torch.tensor(np.stack(all_pred_geom_b), dtype=torch.float, device=device)  # (n_bonds, max_bonds, 2)
    target_geom_a_tensor = torch.tensor(np.stack(all_target_geom_a), dtype=torch.float, device=device)  # (n_bonds, max_bonds, 2)
    target_geom_b_tensor = torch.tensor(np.stack(all_target_geom_b), dtype=torch.float, device=device)  # (n_bonds, max_bonds, 2)
    mask_a_tensor = torch.tensor(np.stack(all_masks_a), dtype=torch.float, device=device)  # (n_bonds, max_bonds)
    mask_b_tensor = torch.tensor(np.stack(all_masks_b), dtype=torch.float, device=device)  # (n_bonds, max_bonds)
    
    # Compute losses using same functions as training
    # Distance loss
    loss_distance = percentage_loss(pred_dists_tensor, target_dists_tensor, epsilon=1e-3, reduction='mean')
    
    # Geometry losses
    geom_weights = {
        'theta': config['loss'].get('theta_weight', 1.0),
        'phi': config['loss'].get('phi_weight', 1.0)
    }
    
    loss_geom_a = geometry_loss(
        pred_geom_a_tensor,
        target_geom_a_tensor,
        mask_a_tensor,
        weights=geom_weights,
        reduction='mean'
    )
    
    loss_geom_b = geometry_loss(
        pred_geom_b_tensor,
        target_geom_b_tensor,
        mask_b_tensor,
        weights=geom_weights,
        reduction='mean'
    )
    
    loss_geometry = (loss_geom_a + loss_geom_b) / 2
    
    # Total weighted loss (same formula as training)
    distance_weight = config['loss'].get('distance_weight', 1.0)
    geometry_weight = config['loss'].get('geometry_weight', 1.0)
    num_components = distance_weight + geometry_weight
    
    total = (
        distance_weight * loss_distance +
        geometry_weight * loss_geometry
    ) / num_components
    
    return {
        'total': total.item(),
        'distance': loss_distance.item(),
        'geometry_a': loss_geom_a.item(),
        'geometry_b': loss_geom_b.item(),
        'geometry': loss_geometry.item(),
        'n_bonds': len(bond_list)
    }


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
    max_bonds = config['data']['max_bonds']

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

            # Reconstruct (without using original coordinates)
            print("  Reconstructing...")
            reconstructed_coords = reconstruct_molecule(
                model, elements, bonds, element_to_idx, device, max_bonds
            )

            # Align reconstructed to original for comparison (handles translation/rotation)
            aligned_coords = align_molecules(reconstructed_coords, original_coords)

            # Compute RMSD
            rmsd_value = compute_rmsd(aligned_coords, original_coords)
            print(f"  RMSD (after alignment): {rmsd_value:.4f} A")

            # Compute reconstruction loss
            print("  Computing reconstruction loss...")
            loss_dict = compute_reconstruction_loss(
                model, original_coords, aligned_coords, elements, bonds,
                element_to_idx, device, max_bonds, config
            )
            
            print(f"  Total Loss: {loss_dict['total']:.6f}")
            print(f"    Distance Loss: {loss_dict['distance']:.6f} (weight: {config['loss'].get('distance_weight', 1.0)})")
            print(f"    Geometry Loss: {loss_dict['geometry']:.6f} (weight: {config['loss'].get('geometry_weight', 1.0)})")
            print(f"      Geometry A: {loss_dict['geometry_a']:.6f}")
            print(f"      Geometry B: {loss_dict['geometry_b']:.6f}")
            print(f"    Evaluated on {loss_dict['n_bonds']} bonds")

            results.append({
                'cid': cid,
                'n_atoms': len(elements),
                'n_bonds': len([b for b in bonds if 'i' in b and 'j' in b]),
                'rmsd': rmsd_value,
                'total_loss': loss_dict['total'],
                'distance_loss': loss_dict['distance'],
                'geometry_loss': loss_dict['geometry'],
                'geometry_a_loss': loss_dict['geometry_a'],
                'geometry_b_loss': loss_dict['geometry_b']
            })

            # Visualize static comparison
            save_path = output_path / f"reconstruction_{cid}.png"
            visualize_reconstruction(
                original_coords, aligned_coords, elements, bonds, cid, rmsd_value, str(save_path)
            )

            # Generate BFS animation GIF
            print("  Generating BFS animation GIF...")
            bfs_bond_order = get_bfs_bond_order(bonds, len(elements))

            if bfs_bond_order:
                gif_path = output_path / f"bfs_reconstruction_{cid}.gif"
                create_bfs_animation_gif(
                    original_coords=original_coords,
                    reconstructed_coords=aligned_coords,  # Use the final aligned reconstruction
                    elements=elements,
                    bonds_data=bonds,
                    bfs_bond_order=bfs_bond_order,
                    cid=str(cid),
                    save_path=str(gif_path),
                    fps=2,
                    show_original=True
                )
            else:
                print("  Warning: No bonds found for BFS animation")

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
        avg_total_loss = np.mean([r['total_loss'] for r in results])
        avg_distance_loss = np.mean([r['distance_loss'] for r in results])
        avg_geometry_loss = np.mean([r['geometry_loss'] for r in results])
        
        print(f"\nProcessed {len(results)} molecules successfully")
        print(f"Average RMSD: {avg_rmsd:.4f} A")
        print(f"Average Total Loss: {avg_total_loss:.6f}")
        print(f"  Average Distance Loss: {avg_distance_loss:.6f}")
        print(f"  Average Geometry Loss: {avg_geometry_loss:.6f}")
        print("\nIndividual Results:")
        for r in results:
            print(f"  {r['cid']}: {r['n_atoms']} atoms, {r['n_bonds']} bonds")
            print(f"    RMSD: {r['rmsd']:.4f} A")
            print(f"    Total Loss: {r['total_loss']:.6f} (Distance: {r['distance_loss']:.6f}, Geometry: {r['geometry_loss']:.6f})")

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    # Run demo with 3 example molecules
    results = reconstruction_demo(
        checkpoint_path="checkpoints/best_model.pt",
        csv_path="data/Compound_000000001_000025000_props_geom_3.csv",
        config_path="config.yaml",
        molecule_indices=[0, 1, 2],  # First 3 valid molecules
        output_dir="reconstruction_results"
    )

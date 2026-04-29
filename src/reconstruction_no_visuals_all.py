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
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import io
from tqdm import tqdm

# Import utilities from the project
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.model import LocalGeometryGCN
from utils.dataset import parse_xyz_string, parse_bonds_from_dataset, get_element_vocab, dfs_subgraph
from utils.geometry import compute_local_geometry, spherical_to_cartesian, create_orthonormal_frame, get_neighbor_sort_keys
from utils.config import load_config
from utils.loss import total_loss, percentage_loss, geometry_loss
from utils.dataset import parse_pubchem_data
GEOMETRY_COLUMN = "DFT_PBE_USPP_Optimized_Geometry"

ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
    86: 'Rn',
}

def _pubchem_mapping_has_error(mapping_val) -> bool:
    if mapping_val is None:
        return True
    try:
        if pd.isna(mapping_val):
            return True
    except (TypeError, ValueError):
        pass
    return "error" in str(mapping_val).lower()

def _parse_json_list(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


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

    df = pd.read_csv(csv_path, low_memory=False)

    geometry_col = "DFT_PBE_USPP_Optimized_Geometry"

    valid_rows = []

    for idx, row in df.iterrows():
        if geometry_col not in row or pd.isna(row[geometry_col]):
            continue

        if (
            "Pubchem_to_xyz_aid_mapping" not in row
            or pd.isna(row["Pubchem_to_xyz_aid_mapping"])
            or _pubchem_mapping_has_error(row["Pubchem_to_xyz_aid_mapping"])
        ):
            continue

        xyz_coordinates, xyz_elements = parse_xyz_string(row[geometry_col])
        if not xyz_coordinates or len(xyz_coordinates) < 2:
            continue

        coordinates, elements, bonds_data = parse_pubchem_data(row, xyz_coordinates)

        if not coordinates or not elements or not bonds_data:
            continue

        valid_rows.append((idx, row))

    if molecule_idx >= len(valid_rows):
        raise IndexError(
            f"Molecule index {molecule_idx} out of range "
            f"(only {len(valid_rows)} valid molecules)"
        )

    idx, row = valid_rows[molecule_idx]

    xyz_coordinates, xyz_elements = parse_xyz_string(row[geometry_col])
    coordinates, elements, bonds_data = parse_pubchem_data(row, xyz_coordinates)

    cid = row.get("cid", f"molecule_{idx}")

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


def reconstruct_molecule(
    model: LocalGeometryGCN,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int = 7,
    n_refinement_iterations: int = 3,
    use_global_optimization: bool = True,
    n_multi_start: int = 2,
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
        n_refinement_iterations: Number of refinement iterations (legacy)
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

def heavy_atom_mask(elements):
    return np.array([e.upper() != "H" for e in elements], dtype=bool)


def compute_heavy_atom_rmsd(coords_pred, coords_true, elements):
    mask = heavy_atom_mask(elements)

    if mask.sum() == 0:
        return np.nan

    aligned_heavy = align_molecules(coords_pred[mask], coords_true[mask])
    return compute_rmsd(aligned_heavy, coords_true[mask])


def compute_bond_angle_details(original_coords, reconstructed_coords, bonds_data, elements, cid, csv_name, mol_idx):
    bond_rows = []
    angle_rows = []
    adjacency = defaultdict(list)

    for bond_id, bond in enumerate(bonds_data):
        if "i" not in bond or "j" not in bond:
            continue

        i, j = bond["i"], bond["j"]

        if i >= len(original_coords) or j >= len(original_coords):
            continue

        adjacency[i].append(j)
        adjacency[j].append(i)

        true_dist = np.linalg.norm(original_coords[i] - original_coords[j])
        pred_dist = np.linalg.norm(reconstructed_coords[i] - reconstructed_coords[j])
        error = pred_dist - true_dist

        bond_rows.append({
            "csv": csv_name,
            "mol_idx": mol_idx,
            "cid": cid,
            "bond_id": bond_id,
            "atom_i": i,
            "atom_j": j,
            "element_i": elements[i],
            "element_j": elements[j],
            "bond_type": f"{elements[i]}-{elements[j]}",
            "bond_order": bond.get("order", np.nan),
            "true_distance": true_dist,
            "pred_distance": pred_dist,
            "error": error,
            "abs_error": abs(error),
            "squared_error": error ** 2,
        })

    def angle_value(coords, i, center, j):
        v1 = coords[i] - coords[center]
        v2 = coords[j] - coords[center]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-8 or n2 < 1e-8:
            return np.nan

        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    angle_id = 0

    for center, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue

        for a_idx in range(len(neighbors)):
            for b_idx in range(a_idx + 1, len(neighbors)):
                i = neighbors[a_idx]
                j = neighbors[b_idx]

                true_angle = angle_value(original_coords, i, center, j)
                pred_angle = angle_value(reconstructed_coords, i, center, j)

                if np.isnan(true_angle) or np.isnan(pred_angle):
                    continue

                error = pred_angle - true_angle

                angle_rows.append({
                    "csv": csv_name,
                    "mol_idx": mol_idx,
                    "cid": cid,
                    "angle_id": angle_id,
                    "atom_i": i,
                    "center_atom": center,
                    "atom_j": j,
                    "element_i": elements[i],
                    "center_element": elements[center],
                    "element_j": elements[j],
                    "angle_type": f"{elements[i]}-{elements[center]}-{elements[j]}",
                    "true_angle": true_angle,
                    "pred_angle": pred_angle,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error ** 2,
                })

                angle_id += 1

    bond_errors = np.array([r["error"] for r in bond_rows], dtype=float)
    angle_errors = np.array([r["error"] for r in angle_rows], dtype=float)

    summary = {
        "bond_mae": float(np.mean(np.abs(bond_errors))) if len(bond_errors) else np.nan,
        "bond_mse": float(np.mean(bond_errors ** 2)) if len(bond_errors) else np.nan,
        "angle_mae": float(np.mean(np.abs(angle_errors))) if len(angle_errors) else np.nan,
        "angle_mse": float(np.mean(angle_errors ** 2)) if len(angle_errors) else np.nan,
        "n_bonds_eval": int(len(bond_errors)),
        "n_angles_eval": int(len(angle_errors)),
    }

    return summary, bond_rows, angle_rows

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

def count_valid_molecules_in_csv(csv_path: str) -> int:
    df = pd.read_csv(csv_path, low_memory=False)

    geometry_col = "DFT_PBE_USPP_Optimized_Geometry"

    count = 0

    for _, row in df.iterrows():

        # Need geometry
        if geometry_col not in row or pd.isna(row[geometry_col]):
            continue

        # Need valid mapping
        if (
            "Pubchem_to_xyz_aid_mapping" not in row
            or pd.isna(row["Pubchem_to_xyz_aid_mapping"])
            or _pubchem_mapping_has_error(row["Pubchem_to_xyz_aid_mapping"])
        ):
            continue

        # Parse xyz coordinates
        xyz_coordinates, _ = parse_xyz_string(row[geometry_col])
        if not xyz_coordinates or len(xyz_coordinates) < 2:
            continue

        # Parse PubChem atoms + bonds
        coordinates, elements, bonds_data = parse_pubchem_data(row, xyz_coordinates)

        if not coordinates:
            continue
        if not elements:
            continue
        if not bonds_data:
            continue

        count += 1

    return count

def reconstruction_demo_rmsd_vs_atoms(
    checkpoint_path: str = "checkpoints/best_model.pt",
    csv_dir: str = "data",
    config_path: str = "config.yaml",
    output_dir: str = "reconstruction_results_rmsd_vs_atoms",
    molecule_indices: List[int] = None,
    num_workers: int = 1,
    resume: bool = True,
    flush_every: int = 100,
):
    print("=" * 70)
    print("RMSD vs Atom Count (Batch over CSV folder)")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_csv_dir = output_path / "per_csv"
    per_csv_dir.mkdir(parents=True, exist_ok=True)

    def _safe_stem(p: Path) -> str:
        return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in p.stem)

    def _per_csv_paths(csv_path: Path):
        stem = _safe_stem(csv_path)
        results_csv = per_csv_dir / f"{stem}__results.csv"
        log_txt = per_csv_dir / f"{stem}__log.txt"
        return results_csv, log_txt

    print("\n[1] Loading trained model...")
    model, element_to_idx, device = load_trained_model(checkpoint_path, config_path)
    model.eval()

    config = load_config(config_path)
    max_bonds = config["data"]["max_bonds"]

    csv_dir_path = Path(csv_dir)
    csv_files = sorted(csv_dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {csv_dir_path}")

    print(f"\n[2] Found {len(csv_files)} CSV files")

    def iter_valid_molecules_from_csv(csv_path: Path, chunksize: int = 5000):
        valid_mol_idx = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            for raw_idx, row in chunk.iterrows():
                try:
                    if GEOMETRY_COLUMN not in row or pd.isna(row[GEOMETRY_COLUMN]):
                        continue

                    if (
                        "Pubchem_to_xyz_aid_mapping" not in row
                        or pd.isna(row["Pubchem_to_xyz_aid_mapping"])
                        or _pubchem_mapping_has_error(row["Pubchem_to_xyz_aid_mapping"])
                    ):
                        continue

                    xyz_coordinates, _ = parse_xyz_string(row[GEOMETRY_COLUMN])
                    if not xyz_coordinates or len(xyz_coordinates) < 2:
                        continue

                    coordinates, elements, bonds = parse_pubchem_data(row, xyz_coordinates)

                    if not coordinates or not elements or not bonds:
                        continue

                    cid = row.get("cid", f"molecule_{raw_idx}")

                    yield valid_mol_idx, coordinates, elements, bonds, cid

                    valid_mol_idx += 1

                except Exception:
                    continue

    def _compute_one_from_data(csv_path: Path, mol_idx: int, coords, elements, bonds, cid):
        original_coords = np.array(coords)

        adjacency = defaultdict(list)
        for bond in bonds:
            if "i" in bond and "j" in bond:
                i, j = bond["i"], bond["j"]
                adjacency[i].append(j)
                adjacency[j].append(i)

        start_atom = max(
            range(len(elements)),
            key=lambda x: len(adjacency.get(x, []))
        )

        reconstructed_coords, _, _, _, _ = reconstruct_from_center(
            model=model,
            elements=elements,
            bonds_data=bonds,
            element_to_idx=element_to_idx,
            device=device,
            max_bonds=max_bonds,
            start_atom=start_atom
        )

        aligned_coords = align_molecules(reconstructed_coords, original_coords)

        rmsd_value = compute_heavy_atom_rmsd(
            reconstructed_coords,
            original_coords,
            elements
        )

        geom_metrics, bond_detail_rows, angle_detail_rows = compute_bond_angle_details(
            original_coords=original_coords,
            reconstructed_coords=aligned_coords,
            bonds_data=bonds,
            elements=elements,
            cid=cid,
            csv_name=csv_path.name,
            mol_idx=mol_idx,
        )

        loss_dict = {
            "total": np.nan,
            "distance": np.nan,
            "geometry": np.nan,
        }

        return {
            "csv": csv_path.name,
            "cid": cid,
            "mol_idx": mol_idx,
            "n_atoms": int(len(elements)),
            "heavy_atom_rmsd": float(rmsd_value),
            "bond_mae": geom_metrics["bond_mae"],
            "bond_mse": geom_metrics["bond_mse"],
            "angle_mae": geom_metrics["angle_mae"],
            "angle_mse": geom_metrics["angle_mse"],
            "n_bonds_eval": geom_metrics["n_bonds_eval"],
            "n_angles_eval": geom_metrics["n_angles_eval"],
            "loss_total": float(loss_dict["total"]),
            "loss_distance": float(loss_dict["distance"]),
            "loss_geometry": float(loss_dict["geometry"]),
            "bond_detail_rows": bond_detail_rows,
            "angle_detail_rows": angle_detail_rows,
        }

    if num_workers > 1:
        raise NotImplementedError("GPU runs must use num_workers=1")

    total_success = 0
    failed = 0

    running = {
        "bond_abs_sum": 0.0,
        "bond_sq_sum": 0.0,
        "bond_n": 0,
        "angle_abs_sum": 0.0,
        "angle_sq_sum": 0.0,
        "angle_n": 0,
        "rmsd_sum": 0.0,
        "rmsd_n": 0,
    }

    main_log_path = Path.cwd() / "reconstruction_running_metrics_log.txt"

    with open(main_log_path, "w", encoding="utf-8") as f:
        f.write(
            "csv mol_idx cid n_atoms heavy_atom_rmsd "
            "bond_mae bond_mse angle_mae angle_mse "
            "running_bond_mae running_bond_mse "
            "running_angle_mae running_angle_mse "
            "running_heavy_atom_rmsd\n"
        )

    print("\n[3] Computing RMSD for molecules...")

    for csv_path in tqdm(csv_files, desc="CSV files"):

        results_csv_path, log_txt_path = _per_csv_paths(csv_path)

        if not log_txt_path.exists():
            with open(log_txt_path, "w", encoding="utf-8") as f:
                f.write(f"CSV: {csv_path.name}\n")
                f.write("idx cid n_atoms heavy_atom_rmsd loss_total loss_dist loss_geom\n")

        processed_idx = set()
        if resume and results_csv_path.exists():
            prev = pd.read_csv(results_csv_path, usecols=["mol_idx"])
            processed_idx = set(prev["mol_idx"].astype(int).tolist())
            tqdm.write(f"[RESUME] {csv_path.name}: {len(processed_idx)} done")

        wanted_indices = None if molecule_indices is None else set(molecule_indices)

        log_buffer = []
        new_records = []

        for mol_idx, coords, elements, bonds, cid in tqdm(
            iter_valid_molecules_from_csv(csv_path),
            desc=csv_path.name,
            leave=False
        ):
            if wanted_indices is not None and mol_idx not in wanted_indices:
                continue

            if mol_idx in processed_idx:
                continue

            try:
                rec = _compute_one_from_data(
                    csv_path=csv_path,
                    mol_idx=mol_idx,
                    coords=coords,
                    elements=elements,
                    bonds=bonds,
                    cid=cid
                )

                bond_detail_path = Path.cwd() / "reconstruction_bond_details.csv"
                angle_detail_path = Path.cwd() / "reconstruction_angle_details.csv"

                bond_detail_rows = rec.pop("bond_detail_rows")
                angle_detail_rows = rec.pop("angle_detail_rows")

                if bond_detail_rows:
                    df_bonds = pd.DataFrame(bond_detail_rows)
                    df_bonds.to_csv(
                        bond_detail_path,
                        mode="a",
                        header=not bond_detail_path.exists(),
                        index=False
                    )

                if angle_detail_rows:
                    df_angles = pd.DataFrame(angle_detail_rows)
                    df_angles.to_csv(
                        angle_detail_path,
                        mode="a",
                        header=not angle_detail_path.exists(),
                        index=False
                    )
                    
                if not np.isnan(rec["bond_mae"]):
                    running["bond_abs_sum"] += rec["bond_mae"] * rec["n_bonds_eval"]
                    running["bond_sq_sum"] += rec["bond_mse"] * rec["n_bonds_eval"]
                    running["bond_n"] += rec["n_bonds_eval"]

                if not np.isnan(rec["angle_mae"]):
                    running["angle_abs_sum"] += rec["angle_mae"] * rec["n_angles_eval"]
                    running["angle_sq_sum"] += rec["angle_mse"] * rec["n_angles_eval"]
                    running["angle_n"] += rec["n_angles_eval"]

                if not np.isnan(rec["heavy_atom_rmsd"]):
                    running["rmsd_sum"] += rec["heavy_atom_rmsd"]
                    running["rmsd_n"] += 1

                running_bond_mae = running["bond_abs_sum"] / running["bond_n"] if running["bond_n"] else np.nan
                running_bond_mse = running["bond_sq_sum"] / running["bond_n"] if running["bond_n"] else np.nan
                running_angle_mae = running["angle_abs_sum"] / running["angle_n"] if running["angle_n"] else np.nan
                running_angle_mse = running["angle_sq_sum"] / running["angle_n"] if running["angle_n"] else np.nan
                running_rmsd = running["rmsd_sum"] / running["rmsd_n"] if running["rmsd_n"] else np.nan

                new_records.append(rec)
                total_success += 1

                log_buffer.append(
                    f"{rec['mol_idx']} {rec['cid']} {rec['n_atoms']} "
                    f"{rec['heavy_atom_rmsd']:.6f} {rec['loss_total']:.6f} "
                    f"{rec['loss_distance']:.6f} {rec['loss_geometry']:.6f}\n"
                )

                with open(main_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{rec['csv']} {rec['mol_idx']} {rec['cid']} {rec['n_atoms']} "
                        f"{rec['heavy_atom_rmsd']:.6f} "
                        f"{rec['bond_mae']:.6f} {rec['bond_mse']:.6f} "
                        f"{rec['angle_mae']:.6f} {rec['angle_mse']:.6f} "
                        f"{running_bond_mae:.6f} {running_bond_mse:.6f} "
                        f"{running_angle_mae:.6f} {running_angle_mse:.6f} "
                        f"{running_rmsd:.6f}\n"
                    )

                if flush_every and len(new_records) >= flush_every:
                    df_new = pd.DataFrame.from_records(new_records)

                    if results_csv_path.exists():
                        df_new.to_csv(results_csv_path, mode="a", header=False, index=False)
                    else:
                        df_new.to_csv(results_csv_path, index=False)

                    with open(log_txt_path, "a", encoding="utf-8") as f:
                        f.writelines(log_buffer)

                    new_records.clear()
                    log_buffer.clear()

            except Exception as e:
                failed += 1
                tqdm.write(f"[SKIP] {csv_path.name} mol_idx={mol_idx} cid={cid}: {e}")
                continue

        if new_records:
            df_new = pd.DataFrame.from_records(new_records)

            if results_csv_path.exists():
                df_new.to_csv(results_csv_path, mode="a", header=False, index=False)
            else:
                df_new.to_csv(results_csv_path, index=False)

        if log_buffer:
            with open(log_txt_path, "a", encoding="utf-8") as f:
                f.writelines(log_buffer)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total successful reconstructions: {total_success}")
    print(f"Total failures (skipped): {failed}")

    # --- Aggregate from per-CSV result files ---
    print("\n[4] Aggregating per-CSV results for plotting...")
    all_result_files = sorted(per_csv_dir.glob("*__results.csv"))

    dfs = []
    for f in all_result_files:
        dfs.append(pd.read_csv(f))

    if not dfs:
        raise RuntimeError("No results found to plot.")

    df = pd.concat(dfs, ignore_index=True)

    grouped = (
        df.groupby("n_atoms")["heavy_atom_rmsd"]
        .mean()
        .reset_index()
        .sort_values("n_atoms")
    )

    plt.figure()
    plt.plot(grouped["n_atoms"], grouped["rmsd"], marker="o")
    plt.xlabel("Number of atoms")
    plt.ylabel("Mean RMSD (Å)")
    plt.title("Mean RMSD vs Number of Atoms")
    plt.grid(True)

    fig_path = output_path / "mean_rmsd_vs_atoms.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {fig_path}")

    return df, grouped


if __name__ == "__main__":
    # Example main: process all CSVs in a folder, try first 200 molecule indices per CSV.
    # If you have a specific index list you want to reuse for every CSV, set molecule_indices=[...].
    df, grouped = reconstruction_demo_rmsd_vs_atoms(
        checkpoint_path=r"checkpoints\best_model.pt",
        csv_dir=r"D:\B3YLP Database",
        config_path=r"config\config.yaml",
        output_dir=r"reconstruction_results_rmsd_vs_atoms",
        molecule_indices=None,        # auto-scan indices 0..max_mols_per_csv-1
        num_workers=1,                # keep 1 if using GPU
        resume= True,
        flush_every=100,   # write to txt every N successful molecules
    )
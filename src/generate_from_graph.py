#!/usr/bin/env python3
"""
Generate 3D Molecular Geometry from Bond Graph Only

This script takes a bond graph (connectivity + elements) without any coordinates
and generates 3D geometry using the trained GCN model.

For potentially infinite structures (like polymers), it processes in chunks
with overlapping atoms to merge them together.

Example: Polyethylene -[CH2-CH2]n-
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import math
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.model import LocalGeometryGCN
from utils.dataset import get_element_vocab, dfs_subgraph
from utils.geometry import spherical_to_cartesian, create_orthonormal_frame, get_neighbor_sort_keys
from utils.config import load_config


def create_polyethylene_graph(n_carbons: int = 10, extended_context: bool = False, context_carbons: int = 30) -> Tuple[List[str], List[Dict]]:
    """
    Create a polyethylene chain graph: CH3-[CH2]_(n-2)-CH3
    
    For infinite polymer simulation, can create extended context on both ends.

    Args:
        n_carbons: Number of carbon atoms in the chain to generate
        extended_context: If True, add extended context on both ends (for infinite polymer simulation)
        context_carbons: Number of additional carbons to add on each side (if extended_context=True)

    Returns:
        elements: List of element symbols
        bonds: List of bond dictionaries
    """
    elements = []
    bonds = []
    atom_idx = 0

    carbon_indices = []
    
    # Determine total carbons (with context if needed)
    if extended_context:
        total_carbons = n_carbons + 2 * context_carbons
        print(f"Creating extended polyethylene chain:")
        print(f"  Core segment: {n_carbons} carbons (to visualize)")
        print(f"  Extended context: {context_carbons} carbons on each side")
        print(f"  Total carbons: {total_carbons} (for model predictions)")
    else:
        total_carbons = n_carbons
        print(f"Created polyethylene with {n_carbons} carbons:")

    # Create carbon backbone
    for c_idx in range(total_carbons):
        elements.append('C')
        carbon_indices.append(atom_idx)
        atom_idx += 1

    # Connect carbon backbone (C-C single bonds)
    for i in range(total_carbons - 1):
        bonds.append({'i': carbon_indices[i], 'j': carbon_indices[i + 1], 'order': 1.0})

    # Add hydrogens to each carbon
    for c_idx, c_atom in enumerate(carbon_indices):
        if extended_context:
            # In extended context, only the core segment has terminal CH3 groups
            # Context segments are all CH2 (middle carbons)
            is_core_start = (c_idx == context_carbons)
            is_core_end = (c_idx == context_carbons + n_carbons - 1)
            n_hydrogens = 3 if (is_core_start or is_core_end) else 2
        else:
            # End carbons get 3 H, middle carbons get 2 H
            n_hydrogens = 3 if (c_idx == 0 or c_idx == n_carbons - 1) else 2

        for _ in range(n_hydrogens):
            elements.append('H')
            bonds.append({'i': c_atom, 'j': atom_idx, 'order': 1.0})
            atom_idx += 1

    if extended_context:
        print(f"  Total atoms: {len(elements)}")
        print(f"  Total bonds: {len(bonds)}")
        print(f"  Core formula: CH3-[CH2]_{n_carbons-2}-CH3")
    else:
        print(f"  Total atoms: {len(elements)}")
        print(f"  Total bonds: {len(bonds)}")
        print(f"  Formula: CH3-[CH2]_{n_carbons-2}-CH3")

    return elements, bonds


def create_graph_data_from_scratch(
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    node_a: int,
    node_b: int,
    positions: np.ndarray,  # Current positions (may be partial/estimated)
    max_bonds: int = 7,
    subgraph_depth: int = 1,
    device: torch.device = torch.device('cpu')
):
    """
    Create PyG Data object for model inference from graph structure.
    """
    from torch_geometric.data import Data

    n_atoms = len(elements)
    pos = torch.tensor(positions, dtype=torch.float)

    # Create element indices
    element_indices = [element_to_idx.get(elem, 0) for elem in elements]
    x = torch.tensor(element_indices, dtype=torch.long)

    # Create adjacency graph with bond types
    adjacency = defaultdict(list)
    bond_types_dict = {}

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < n_atoms and j < n_atoms and i != j:
                adjacency[i].append(j)
                adjacency[j].append(i)

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

    # Extract subgraph
    subgraph_a = dfs_subgraph(adjacency, node_a, depth=subgraph_depth)
    subgraph_b = dfs_subgraph(adjacency, node_b, depth=subgraph_depth)
    subgraph_nodes = subgraph_a.union(subgraph_b)
    subgraph_nodes.add(node_a)
    subgraph_nodes.add(node_b)

    node_list = sorted(list(subgraph_nodes))
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_list)}

    new_node_a = node_mapping[node_a]
    new_node_b = node_mapping[node_b]

    subgraph_x = x[node_list]
    subgraph_pos = pos[node_list]

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

    data = Data(
        x=subgraph_x.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        pos=subgraph_pos.to(device),
        node_a=torch.tensor(new_node_a, dtype=torch.long).to(device),
        node_b=torch.tensor(new_node_b, dtype=torch.long).to(device)
    )

    return data


def predict_geometry(model, data, device):
    """Predict bond geometry."""
    model.eval()
    with torch.no_grad():
        distance, geometry_a, geometry_b = model(data)

    distance = distance.squeeze().cpu().item()
    geometry_a = geometry_a.squeeze().cpu().numpy()
    geometry_b = geometry_b.squeeze().cpu().numpy()

    return distance, geometry_a, geometry_b


def identify_equivalent_bonds(
    elements: List[str],
    bonds_data: List[Dict],
    backbone_start: int,
    backbone_end: int
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Identify equivalent bonds in a repeating polymer unit.
    
    For polyethylene, equivalent bonds are:
    - All C-C backbone bonds (except terminal ones)
    - All C-H bonds of the same type (CH2 vs CH3)
    
    Args:
        elements: Element symbols
        bonds_data: Bond information
        backbone_start: First carbon index in backbone
        backbone_end: Last carbon index in backbone (exclusive)
    
    Returns:
        equivalence_groups: Dict mapping a representative bond to list of equivalent bonds
    """
    equivalence_groups = defaultdict(list)
    
    # Group bonds by type and local environment
    bond_groups = defaultdict(list)
    
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            elem_i = elements[i]
            elem_j = elements[j]
            bond_order = bond.get('order', 1.0)
            
            # Create a signature for bond equivalence
            # For C-C bonds: check if both are in backbone and not terminal
            if elem_i == 'C' and elem_j == 'C':
                if backbone_start <= i < backbone_end and backbone_start <= j < backbone_end:
                    # Both are backbone carbons
                    i_pos = min(i, j) - backbone_start
                    j_pos = max(i, j) - backbone_start
                    # Check if it's an internal backbone bond (not terminal)
                    if abs(i_pos - j_pos) == 1:  # Adjacent in backbone
                        # All internal C-C bonds are equivalent
                        signature = ('C-C-backbone-internal', bond_order)
                        bond_groups[signature].append((i, j))
            
            # For C-H bonds: group by whether carbon is terminal or internal
            elif (elem_i == 'C' and elem_j == 'H') or (elem_i == 'H' and elem_j == 'C'):
                carbon_idx = i if elem_i == 'C' else j
                if backbone_start <= carbon_idx < backbone_end:
                    carbon_pos = carbon_idx - backbone_start
                    # Check if terminal carbon (first or last in segment)
                    is_terminal = (carbon_pos == 0) or (carbon_pos == backbone_end - backbone_start - 1)
                    signature = ('C-H', bond_order, 'terminal' if is_terminal else 'internal')
                    bond_groups[signature].append((i, j))
    
    # Create equivalence groups
    for signature, bonds in bond_groups.items():
        if len(bonds) > 1:  # Only group if there are multiple equivalent bonds
            # Use first bond as representative
            representative = bonds[0]
            equivalence_groups[representative] = bonds
    
    return equivalence_groups


def optimize_chunk_positions(
    model: LocalGeometryGCN,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    initial_positions: np.ndarray,
    chunk_atoms: List[int],
    anchor_positions: Optional[Dict[int, np.ndarray]] = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-6,
    max_rounds: int = 100,
    enforce_periodicity: bool = True
) -> np.ndarray:
    """
    Optimize positions using iterative distance geometry until convergence.

    Iteratively:
    1. Predict all bond distances using current positions
    2. Optimize positions to match predicted distances
    3. Repeat until convergence (distance predictions stabilize)

    Args:
        model: Trained model
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        initial_positions: Initial positions from BFS
        chunk_atoms: Atoms in this chunk
        anchor_positions: Fixed anchor positions (from overlap)
        max_iterations: Max optimization iterations per round
        convergence_threshold: Stop when distance change < threshold
        max_rounds: Maximum refinement rounds (safety limit)

    Returns:
        optimized_positions: Optimized positions
    """
    from scipy.optimize import minimize

    chunk_set = set(chunk_atoms)

    # Build bond list for chunk
    chunk_bonds = []
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i in chunk_set and j in chunk_set:
                if i < j:
                    chunk_bonds.append((i, j))

    if not chunk_bonds:
        return initial_positions.copy()

    # Determine anchors (fixed atoms)
    anchors = set()
    if anchor_positions:
        anchors = set(anchor_positions.keys())

    # Free atoms to optimize
    free_atoms = [a for a in chunk_atoms if a not in anchors]

    if not free_atoms:
        return initial_positions.copy()

    # Map free atoms to optimization variables
    atom_to_var = {atom: i for i, atom in enumerate(free_atoms)}

    # Identify equivalent bonds for periodic constraints
    equivalence_groups = {}
    if enforce_periodicity:
        # Find backbone atoms (carbons)
        backbone_carbons = [i for i in chunk_atoms if elements[i] == 'C']
        if len(backbone_carbons) > 2:
            backbone_start = min(backbone_carbons)
            backbone_end = max(backbone_carbons) + 1
            equivalence_groups = identify_equivalent_bonds(
                elements, bonds_data, backbone_start, backbone_end
            )
            if equivalence_groups:
                print(f"    Identified {len(equivalence_groups)} groups of equivalent bonds for periodic constraints")

    # Start with initial positions
    positions = initial_positions.copy()
    prev_distances = None
    prev_positions = positions.copy()  # Store full positions array

    # Iterative refinement until convergence
    for round_idx in range(max_rounds):
        # Step 1: Predict all bond distances using current positions
        predicted_distances = {}
        for i, j in chunk_bonds:
            data = create_graph_data_from_scratch(
                elements, bonds_data, element_to_idx,
                i, j, positions, max_bonds, device=device
            )
            pred_dist, _, _ = predict_geometry(model, data, device)
            predicted_distances[(i, j)] = pred_dist
            predicted_distances[(j, i)] = pred_dist
        
        # Enforce periodicity: average predictions for equivalent bonds
        if enforce_periodicity and equivalence_groups:
            for representative, equivalent_bonds in equivalence_groups.items():
                # Collect predictions for all equivalent bonds
                equiv_predictions = []
                for bond in equivalent_bonds:
                    i, j = bond
                    if (i, j) in predicted_distances:
                        equiv_predictions.append(predicted_distances[(i, j)])
                
                if len(equiv_predictions) > 1:
                    # Average the predictions
                    avg_dist = np.mean(equiv_predictions)
                    # Apply averaged distance to all equivalent bonds
                    for bond in equivalent_bonds:
                        i, j = bond
                        predicted_distances[(i, j)] = avg_dist
                        predicted_distances[(j, i)] = avg_dist

        # Check convergence: have distance predictions stabilized?
        if prev_distances is not None:
            max_change = 0.0
            for key in predicted_distances:
                if key in prev_distances:
                    change = abs(predicted_distances[key] - prev_distances[key])
                    max_change = max(max_change, change)

            # Also check if positions have converged
            if round_idx > 0 and prev_positions is not None:
                pos_change = np.max(np.abs(positions[free_atoms] - prev_positions[free_atoms]))
                if max_change < convergence_threshold and pos_change < 1e-4:
                    print(f"    Converged after {round_idx + 1} rounds (max distance change: {max_change:.2e}, pos change: {pos_change:.2e})")
                    break

        prev_distances = predicted_distances.copy()
        prev_positions = positions.copy()  # Store full positions array

        # Step 2: Optimize positions to match predicted distances
        x0 = positions[free_atoms].flatten()

        def objective(x):
            """Total distance error with regularization."""
            pos = positions.copy()
            free_pos = x.reshape(-1, 3)
            for idx, atom in enumerate(free_atoms):
                pos[atom] = free_pos[idx]

            total_loss = 0.0
            for i, j in chunk_bonds:
                pred_dist = predicted_distances[(i, j)]
                current_dist = np.linalg.norm(pos[j] - pos[i])
                # Use squared error normalized by predicted distance
                error = ((current_dist - pred_dist) / (pred_dist + 1e-6)) ** 2
                total_loss += error

            # Add small regularization to prevent extreme positions
            if len(free_atoms) > 0:
                center_of_mass = pos[free_atoms].mean(axis=0)
                reg = 1e-6 * np.sum((free_pos - center_of_mass) ** 2)
                total_loss += reg

            return total_loss

        def gradient(x):
            """Analytical gradient."""
            pos = positions.copy()
            free_pos = x.reshape(-1, 3)
            for idx, atom in enumerate(free_atoms):
                pos[atom] = free_pos[idx]

            grad = np.zeros((len(free_atoms), 3))

            for i, j in chunk_bonds:
                pred_dist = predicted_distances[(i, j)]
                diff = pos[j] - pos[i]
                current_dist = np.linalg.norm(diff)

                if current_dist < 1e-8:
                    continue

                coeff = 2.0 * (current_dist - pred_dist) / ((pred_dist + 1e-6) ** 2 * current_dist)

                if i in atom_to_var:
                    grad[atom_to_var[i]] -= coeff * diff
                if j in atom_to_var:
                    grad[atom_to_var[j]] += coeff * diff

            # Add regularization gradient
            if len(free_atoms) > 0:
                center_of_mass = pos[free_atoms].mean(axis=0)
                reg_grad = 2e-6 * (free_pos - center_of_mass)
                grad += reg_grad

            return grad.flatten()

        # Use bounds to prevent atoms from moving too far
        bounds = []
        for atom in free_atoms:
            current_pos = positions[atom]
            # Allow movement within ±5 Å of current position
            for coord in current_pos:
                bounds.append((coord - 5.0, coord + 5.0))

        result = minimize(
            objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
        )

        # Update positions for next round
        if result.success:
            final_pos = result.x.reshape(-1, 3)
            for idx, atom in enumerate(free_atoms):
                positions[atom] = final_pos[idx]
        else:
            # If optimization failed, use a small gradient step instead
            grad = gradient(x0)
            step_size = 0.1
            x_new = x0 - step_size * grad
            final_pos = x_new.reshape(-1, 3)
            for idx, atom in enumerate(free_atoms):
                positions[atom] = final_pos[idx]

        # Keep anchor positions fixed
        if anchor_positions:
            for atom, pos in anchor_positions.items():
                positions[atom] = pos

    else:
        print(f"    Max rounds ({max_rounds}) reached without convergence")

    return positions


def generate_chunk_geometry(
    model: LocalGeometryGCN,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    chunk_atoms: List[int],
    anchor_positions: Optional[Dict[int, np.ndarray]] = None,
    use_optimization: bool = True
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate 3D geometry for a chunk of atoms.

    Uses BFS for initial placement, then distance geometry optimization.

    Args:
        model: Trained model
        elements: All element symbols
        bonds_data: All bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        chunk_atoms: List of atom indices in this chunk
        anchor_positions: Known positions for anchor atoms (from overlap)
        use_optimization: If True, apply distance geometry optimization after BFS

    Returns:
        positions: (N, 3) array of positions for all atoms
        bfs_order: BFS traversal order of bonds
    """
    n_atoms = len(elements)
    positions = np.zeros((n_atoms, 3))
    placed = np.zeros(n_atoms, dtype=bool)

    # Build adjacency for chunk
    adjacency = defaultdict(list)
    bond_orders = {}
    chunk_set = set(chunk_atoms)

    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            # Only consider bonds where both atoms are in chunk
            if i in chunk_set and j in chunk_set:
                adjacency[i].append(j)
                adjacency[j].append(i)
                bond_order = bond.get('order', 1.0)
                bond_orders[(i, j)] = bond_order
                bond_orders[(j, i)] = bond_order

    bfs_order = []

    # Use anchor positions if provided
    if anchor_positions:
        for atom_idx, pos in anchor_positions.items():
            if atom_idx in chunk_set:
                positions[atom_idx] = pos
                placed[atom_idx] = True

    # Find starting atom
    if anchor_positions:
        # Start from an anchor atom
        start_atom = next(iter(anchor_positions.keys()))
    else:
        # Start from most connected atom in chunk
        start_atom = max(chunk_atoms, key=lambda x: len(adjacency.get(x, [])))
        # Place at origin
        positions[start_atom] = [0, 0, 0]
        placed[start_atom] = True

    # Find first neighbor to establish coordinate frame
    if not placed[start_atom]:
        positions[start_atom] = [0, 0, 0]
        placed[start_atom] = True

    neighbors = [n for n in adjacency[start_atom] if n in chunk_set]
    if not neighbors:
        return positions, bfs_order

    # Place first neighbor if not already placed
    ref_atom = neighbors[0]
    if not placed[ref_atom]:
        # Predict distance
        data = create_graph_data_from_scratch(
            elements, bonds_data, element_to_idx,
            start_atom, ref_atom, positions, max_bonds, device=device
        )
        pred_dist, _, _ = predict_geometry(model, data, device)
        # Clamp distance to reasonable range
        pred_dist = np.clip(pred_dist, 0.5, 5.0)

        # Place along x-axis
        positions[ref_atom] = positions[start_atom] + np.array([pred_dist, 0, 0])
        placed[ref_atom] = True

    bfs_order.append((start_atom, ref_atom))

    # BFS to place remaining atoms
    queue = deque([start_atom, ref_atom])
    visited_bonds = {(min(start_atom, ref_atom), max(start_atom, ref_atom))}

    while queue:
        center = queue.popleft()
        neighbors = [n for n in adjacency[center] if n in chunk_set]

        if not neighbors:
            continue

        placed_neighbors = [n for n in neighbors if placed[n]]
        unplaced_neighbors = [n for n in neighbors if not placed[n]]

        if not placed_neighbors or not unplaced_neighbors:
            continue

        # Use the first placed neighbor as reference, but prefer one that's not collinear
        ref_atom_local = placed_neighbors[0]
        
        # If we have multiple placed neighbors, try to find a better reference
        # (one that's not collinear with center and first neighbor)
        if len(placed_neighbors) > 1:
            for candidate_ref in placed_neighbors[1:]:
                vec1 = positions[ref_atom_local] - positions[center]
                vec2 = positions[candidate_ref] - positions[center]
                # Check if they're not collinear (cross product should be non-zero)
                cross = np.cross(vec1, vec2)
                if np.linalg.norm(cross) > 0.1:  # Not collinear
                    ref_atom_local = candidate_ref
                    break

        # Get geometry predictions
        data = create_graph_data_from_scratch(
            elements, bonds_data, element_to_idx,
            center, ref_atom_local, positions, max_bonds, device=device
        )
        _, geometry_center, _ = predict_geometry(model, data, device)

        # Create coordinate frame
        center_pos = torch.tensor(positions[center], dtype=torch.float)
        ref_pos = torch.tensor(positions[ref_atom_local], dtype=torch.float)

        bond_vec = ref_pos - center_pos
        bond_length = torch.norm(bond_vec)

        if bond_length < 1e-6:
            continue

        bond_axis = bond_vec / bond_length
        x_axis, y_axis = create_orthonormal_frame(bond_axis)
        coord_frame = (x_axis, y_axis, bond_axis)

        # Sort neighbors - prioritize unplaced neighbors (to extend chain), then placed ones
        all_neighbors = neighbors.copy()
        
        # Separate placed and unplaced neighbors
        placed_neighbor_list = [n for n in all_neighbors if placed[n]]
        unplaced_neighbor_list = [n for n in all_neighbors if not placed[n]]
        
        # Sort unplaced neighbors first (to extend chain), then placed ones
        sort_keys_unplaced = get_neighbor_sort_keys(unplaced_neighbor_list, elements, bond_orders, center)
        sort_keys_placed = get_neighbor_sort_keys(placed_neighbor_list, elements, bond_orders, center)
        
        sorted_unplaced = [n for n, _ in sorted(zip(unplaced_neighbor_list, sort_keys_unplaced), key=lambda x: x[1])]
        sorted_placed = [n for n, _ in sorted(zip(placed_neighbor_list, sort_keys_placed), key=lambda x: x[1])]
        
        # Combine: unplaced first, then placed (excluding reference)
        sorted_neighbors = sorted_unplaced + sorted_placed
        sorted_neighbors_excl_ref = [n for n in sorted_neighbors if n != ref_atom_local]

        cumulative_phi = 0.0
        placed_slot_count = 0  # Track how many placed neighbors we've seen

        for slot_idx, neighbor in enumerate(sorted_neighbors_excl_ref):
            if slot_idx >= max_bonds:
                break

            if placed[neighbor]:
                # Already placed neighbor: increment cumulative_phi to maintain correct angle spacing
                # but don't place it again
                if placed_slot_count > 0:
                    cumulative_phi += geometry_center[slot_idx, 1]
                placed_slot_count += 1
                continue

            theta_deg = geometry_center[slot_idx, 0]
            # Clamp theta to reasonable range (0-180 degrees)
            theta_deg = np.clip(theta_deg, 0.0, 180.0)

            if placed_slot_count == 0:
                # First unplaced neighbor: phi = 0 (relative to reference)
                phi = 0.0
            else:
                # Subsequent neighbors: add delta_phi to cumulative
                delta_phi_deg = geometry_center[slot_idx, 1]
                # Clamp delta_phi to reasonable range
                delta_phi_deg = np.clip(delta_phi_deg, 0.0, 360.0)
                cumulative_phi += delta_phi_deg
                phi = math.radians(cumulative_phi % 360.0)
            
            placed_slot_count += 1

            theta = math.radians(theta_deg)

            # Predict distance
            bond_key = (min(center, neighbor), max(center, neighbor))
            elem_center = elements[center]
            elem_neighbor = elements[neighbor]
            
            if bond_key not in visited_bonds:
                data_dist = create_graph_data_from_scratch(
                    elements, bonds_data, element_to_idx,
                    center, neighbor, positions, max_bonds, device=device
                )
                pred_r, _, _ = predict_geometry(model, data_dist, device)
                # Clamp distance to reasonable range based on element types
                if elem_center == 'C' and elem_neighbor == 'C':
                    pred_r = np.clip(pred_r, 1.3, 1.8)  # C-C single bond range
                elif (elem_center == 'C' and elem_neighbor == 'H') or (elem_neighbor == 'C' and elem_center == 'H'):
                    pred_r = np.clip(pred_r, 0.9, 1.3)  # C-H bond range
                else:
                    pred_r = np.clip(pred_r, 0.5, 5.0)  # General range
                visited_bonds.add(bond_key)
            else:
                # Use element-based default distances (already computed)
                if elem_center == 'C' and elem_neighbor == 'C':
                    pred_r = 1.54  # C-C single bond
                elif (elem_center == 'C' and elem_neighbor == 'H') or (elem_neighbor == 'C' and elem_center == 'H'):
                    pred_r = 1.09  # C-H bond
                else:
                    pred_r = 1.5  # Default

            # Place atom
            new_pos = spherical_to_cartesian(pred_r, theta, phi, coord_frame, center_pos)
            positions[neighbor] = new_pos.numpy()
            placed[neighbor] = True
            queue.append(neighbor)

            bfs_order.append((center, neighbor))

    # Apply distance geometry optimization if enabled
    # For small chunks (< 20 atoms), use lighter optimization
    if use_optimization:
        n_chunk_atoms = len(chunk_atoms)
        if n_chunk_atoms < 20:
            # Lighter optimization for small molecules
            positions = optimize_chunk_positions(
                model, elements, bonds_data, element_to_idx,
                device, max_bonds, positions, chunk_atoms, anchor_positions,
                max_iterations=100, convergence_threshold=1e-5, max_rounds=50,
                enforce_periodicity=True  # Enforce periodic constraints
            )
        else:
            # Full optimization for larger chunks
            positions = optimize_chunk_positions(
                model, elements, bonds_data, element_to_idx,
                device, max_bonds, positions, chunk_atoms, anchor_positions,
                enforce_periodicity=True  # Enforce periodic constraints
            )

    return positions, bfs_order


def generate_geometry_chunked(
    model: LocalGeometryGCN,
    elements: List[str],
    bonds_data: List[Dict],
    element_to_idx: Dict[str, int],
    device: torch.device,
    max_bonds: int,
    chunk_size: int = 12,
    overlap: int = 3
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate 3D geometry by processing in overlapping chunks.

    Args:
        model: Trained model
        elements: Element symbols
        bonds_data: Bond information
        element_to_idx: Element vocabulary
        device: torch device
        max_bonds: Maximum bonds per atom
        chunk_size: Number of atoms per chunk
        overlap: Number of overlapping atoms between chunks

    Returns:
        positions: (N, 3) final positions
        all_bfs_order: Combined BFS order from all chunks
    """
    n_atoms = len(elements)

    # Build adjacency
    adjacency = defaultdict(list)
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            adjacency[i].append(j)
            adjacency[j].append(i)

    # Find backbone atoms (carbons for polyethylene)
    backbone = [i for i, elem in enumerate(elements) if elem == 'C']

    # If small enough or no backbone, process in one go
    # Use chunk_size for backbone atoms, not total atoms
    if len(backbone) <= chunk_size:
        chunk_atoms = list(range(n_atoms))
        print(f"Processing entire molecule in one chunk ({len(chunk_atoms)} atoms)")
        return generate_chunk_geometry(
            model, elements, bonds_data, element_to_idx,
            device, max_bonds, chunk_atoms, None
        )

    # Process in chunks along backbone
    final_positions = np.zeros((n_atoms, 3))
    placed = np.zeros(n_atoms, dtype=bool)
    all_bfs_order = []

    chunk_start = 0
    chunk_idx = 0
    max_chunks = len(backbone) + 10  # Safety limit

    while chunk_start < len(backbone) and chunk_idx < max_chunks:
        chunk_end = min(chunk_start + chunk_size, len(backbone))

        # Get backbone atoms for this chunk
        chunk_backbone = backbone[chunk_start:chunk_end]

        # Add all connected atoms (hydrogens)
        chunk_atoms = set(chunk_backbone)
        for atom in chunk_backbone:
            for neighbor in adjacency[atom]:
                chunk_atoms.add(neighbor)

        chunk_atoms = sorted(list(chunk_atoms))

        print(f"Processing chunk {chunk_idx + 1}: backbone[{chunk_start}:{chunk_end}], {len(chunk_atoms)} atoms")

        # Get anchor positions from overlap with previous chunk
        anchor_positions = {}
        if chunk_idx > 0 and chunk_start > 0:
            # Use overlap atoms from previous chunk
            overlap_start = max(0, chunk_start - overlap)
            overlap_backbone = backbone[overlap_start:chunk_start]
            for atom in overlap_backbone:
                if placed[atom]:
                    anchor_positions[atom] = final_positions[atom].copy()
                # Also include connected hydrogens
                for neighbor in adjacency[atom]:
                    if placed[neighbor]:
                        anchor_positions[neighbor] = final_positions[neighbor].copy()

        # Generate chunk geometry
        chunk_positions, chunk_bfs = generate_chunk_geometry(
            model, elements, bonds_data, element_to_idx,
            device, max_bonds, chunk_atoms, anchor_positions if anchor_positions else None
        )

        # Merge into final positions
        if chunk_idx == 0:
            # First chunk - use directly
            for atom in chunk_atoms:
                final_positions[atom] = chunk_positions[atom]
                placed[atom] = True
        else:
            # Align using overlap atoms
            if anchor_positions:
                overlap_atoms = list(anchor_positions.keys())
                
                # Filter to atoms that exist in both sets
                valid_overlap = [a for a in overlap_atoms if a in chunk_atoms and placed[a]]

                if len(valid_overlap) >= 3:
                    # Use Kabsch algorithm to align
                    src_points = np.array([chunk_positions[a] for a in valid_overlap])
                    dst_points = np.array([anchor_positions[a] for a in valid_overlap])

                    src_center = src_points.mean(axis=0)
                    dst_center = dst_points.mean(axis=0)

                    src_centered = src_points - src_center
                    dst_centered = dst_points - dst_center

                    # Compute covariance matrix
                    H = src_centered.T @ dst_centered
                    U, S, Vt = np.linalg.svd(H)
                    R = Vt.T @ U.T

                    # Ensure proper rotation (det(R) = 1)
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = Vt.T @ U.T

                    # Apply transformation to all chunk atoms
                    for atom in chunk_atoms:
                        if not placed[atom]:
                            transformed = (chunk_positions[atom] - src_center) @ R.T + dst_center
                            final_positions[atom] = transformed
                            placed[atom] = True
                elif len(valid_overlap) == 2:
                    # Two points: use rotation + translation
                    src_vec = chunk_positions[valid_overlap[1]] - chunk_positions[valid_overlap[0]]
                    dst_vec = anchor_positions[valid_overlap[1]] - anchor_positions[valid_overlap[0]]
                    
                    src_norm = np.linalg.norm(src_vec)
                    dst_norm = np.linalg.norm(dst_vec)
                    
                    if src_norm > 1e-6 and dst_norm > 1e-6:
                        # Compute rotation axis and angle
                        src_unit = src_vec / src_norm
                        dst_unit = dst_vec / dst_norm
                        
                        # Cross product gives rotation axis
                        axis = np.cross(src_unit, dst_unit)
                        axis_norm = np.linalg.norm(axis)
                        
                        if axis_norm > 1e-6:
                            axis = axis / axis_norm
                            cos_angle = np.clip(np.dot(src_unit, dst_unit), -1.0, 1.0)
                            angle = np.arccos(cos_angle)
                            
                            # Rodrigues rotation formula
                            K = np.array([[0, -axis[2], axis[1]],
                                         [axis[2], 0, -axis[0]],
                                         [-axis[1], axis[0], 0]])
                            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                        else:
                            # Vectors are parallel or anti-parallel
                            R = np.eye(3)
                        
                        # Apply rotation + translation
                        src_anchor = chunk_positions[valid_overlap[0]]
                        dst_anchor = anchor_positions[valid_overlap[0]]
                        
                        for atom in chunk_atoms:
                            if not placed[atom]:
                                transformed = (chunk_positions[atom] - src_anchor) @ R.T + dst_anchor
                                final_positions[atom] = transformed
                                placed[atom] = True
                    else:
                        # Fallback to translation only
                        offset = anchor_positions[valid_overlap[0]] - chunk_positions[valid_overlap[0]]
                        for atom in chunk_atoms:
                            if not placed[atom]:
                                final_positions[atom] = chunk_positions[atom] + offset
                                placed[atom] = True
                elif len(valid_overlap) == 1:
                    # Single point: translation only
                    offset = anchor_positions[valid_overlap[0]] - chunk_positions[valid_overlap[0]]
                    for atom in chunk_atoms:
                        if not placed[atom]:
                            final_positions[atom] = chunk_positions[atom] + offset
                            placed[atom] = True

        all_bfs_order.extend(chunk_bfs)

        # Move to next chunk - ensure we make forward progress
        if chunk_end >= len(backbone):
            break  # We've processed all backbone atoms

        # Calculate next start ensuring forward progress
        step = max(1, chunk_size - overlap)
        chunk_start = chunk_start + step
        chunk_idx += 1

    return final_positions, all_bfs_order


def visualize_generated_structure(
    positions: np.ndarray,
    elements: List[str],
    bonds_data: List[Dict],
    title: str = "Generated Structure",
    save_path: Optional[str] = None
):
    """Visualize the generated 3D structure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    element_colors = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'S': '#FFFF30', 'Cl': '#1FF01F', 'Br': '#A62929'
    }
    element_sizes = {
        'H': 200, 'C': 400, 'N': 400, 'O': 400, 'F': 350, 'S': 500
    }

    colors = [element_colors.get(e, '#FF1493') for e in elements]
    sizes = [element_sizes.get(e, 400) for e in elements]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot atoms
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
    )

    # Plot bonds
    for bond in bonds_data:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i < len(positions) and j < len(positions):
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]],
                    'k-', linewidth=2, alpha=0.6
                )

    # Add atom labels
    for idx, (elem, pos) in enumerate(zip(elements, positions)):
        ax.text(pos[0], pos[1], pos[2] + 0.3, f"{elem}{idx}", fontsize=7, ha='center')

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Equal aspect ratio
    max_range = np.max(np.ptp(positions, axis=0)) / 2 * 1.2
    mid = positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()

    return fig


def create_bfs_animation(
    positions: np.ndarray,
    elements: List[str],
    bonds_data: List[Dict],
    bfs_order: List[Tuple[int, int]],
    title: str,
    save_path: str,
    fps: int = 2
):
    """Create BFS animation GIF for the generated structure."""
    import matplotlib.pyplot as plt
    import imageio
    import io

    element_colors = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D'
    }
    element_sizes = {
        'H': 200, 'C': 400, 'N': 400, 'O': 400
    }

    colors = [element_colors.get(e, '#FF1493') for e in elements]
    sizes = [element_sizes.get(e, 400) for e in elements]

    # Compute axis limits
    max_range = np.max(np.ptp(positions, axis=0)) / 2 * 1.2
    mid = positions.mean(axis=0)

    frames = []

    for frame_idx in range(len(bfs_order)):
        hi, hj = bfs_order[frame_idx]
        bond_dist = np.linalg.norm(positions[hi] - positions[hj])

        # Determine visible atoms and bonds
        visited_bonds = set()
        visible_atoms = set()
        for bond in bfs_order[:frame_idx + 1]:
            visited_bonds.add((min(bond[0], bond[1]), max(bond[0], bond[1])))
            visible_atoms.add(bond[0])
            visible_atoms.add(bond[1])

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot visible atoms
        for atom_idx in visible_atoms:
            ax.scatter(
                [positions[atom_idx, 0]], [positions[atom_idx, 1]], [positions[atom_idx, 2]],
                c=[colors[atom_idx]], s=[sizes[atom_idx]],
                edgecolors='black', linewidths=0.5, alpha=0.9
            )

        # Draw bonds
        highlight_key = (min(hi, hj), max(hi, hj))
        for bond in bonds_data:
            if 'i' in bond and 'j' in bond:
                i, j = bond['i'], bond['j']
                bond_key = (min(i, j), max(i, j))

                if bond_key not in visited_bonds:
                    continue
                elif bond_key == highlight_key:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        color='#FF0000', linewidth=4, alpha=1.0
                    )
                else:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        color='#00AA00', linewidth=2.5, alpha=0.8
                    )

        # Highlight current atoms
        for atom_idx in [hi, hj]:
            ax.scatter(
                [positions[atom_idx, 0]], [positions[atom_idx, 1]], [positions[atom_idx, 2]],
                c='yellow', s=sizes[atom_idx] * 1.5, edgecolors='red',
                linewidths=2, alpha=0.7, zorder=10
            )

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'{title}\nBond {frame_idx + 1}/{len(bfs_order)}: {elements[hi]}{hi}-{elements[hj]}{hj}\nDistance: {bond_dist:.3f} Å',
                    fontsize=11, fontweight='bold')
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Final frame (hold longer)
    for _ in range(4):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, s=sizes, edgecolors='black', linewidths=0.5, alpha=0.9
        )

        for bond in bonds_data:
            if 'i' in bond and 'j' in bond:
                i, j = bond['i'], bond['j']
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]],
                    color='#00AA00', linewidth=2.5, alpha=0.8
                )

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'{title}\nGeneration Complete - {len(elements)} atoms, {len(bonds_data)} bonds',
                    fontsize=11, fontweight='bold')
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"Saved BFS animation to {save_path}")


def main():
    """Main function to demonstrate graph-to-geometry generation."""
    print("="*60)
    print("Generate 3D Geometry from Bond Graph")
    print("="*60)

    # Configuration
    checkpoint_path = "checkpoints/best_model.pt"
    config_path = "config.yaml"
    output_dir = Path("generation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1] Loading trained model...")
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    elements_txt = config['data'].get('elements_txt', 'elements.txt')
    element_to_idx = get_element_vocab(txt_path=elements_txt)
    vocab_size = len(element_to_idx)

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    max_bonds = config['data']['max_bonds']

    # Create polyethylene graph
    print("\n[2] Creating polyethylene bond graph...")
    n_carbons = 6  # Number of carbons to visualize
    context_carbons = 10  # Additional carbons on each side for infinite polymer context
    
    # Create extended graph for model predictions (simulating infinite polymer)
    elements, bonds = create_polyethylene_graph(
        n_carbons=n_carbons,
        extended_context=True,
        context_carbons=context_carbons
    )

    # Generate 3D geometry for the full extended graph
    print("\n[3] Generating 3D geometry from extended graph (infinite polymer context)...")
    total_carbons = n_carbons + 2 * context_carbons
    positions, bfs_order = generate_geometry_chunked(
        model, elements, bonds, element_to_idx, device, max_bonds,
        chunk_size=max(12, total_carbons + 2), overlap=3
    )
    
    # Extract only the core segment for visualization (first n_carbons + their hydrogens)
    print("\n[3.5] Extracting core segment for visualization...")
    # Find all atoms in the core segment (carbons + their hydrogens)
    core_carbon_start = context_carbons
    core_carbon_end = context_carbons + n_carbons
    
    # Build adjacency to find connected atoms
    adjacency = defaultdict(list)
    for bond in bonds:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            adjacency[i].append(j)
            adjacency[j].append(i)
    
    # Find all carbons in the core segment
    core_carbons = list(range(core_carbon_start, core_carbon_end))
    
    # Find all atoms connected to core carbons (including hydrogens)
    core_atoms = set(core_carbons)
    for carbon_idx in core_carbons:
        for neighbor in adjacency.get(carbon_idx, []):
            core_atoms.add(neighbor)
    
    core_atoms = sorted(list(core_atoms))
    
    # Create mapping from full graph indices to core segment indices
    full_to_core = {old_idx: new_idx for new_idx, old_idx in enumerate(core_atoms)}
    
    # Extract core positions
    core_positions = positions[core_atoms]
    
    # Extract core elements and bonds
    core_elements = [elements[i] for i in core_atoms]
    core_bonds = []
    for bond in bonds:
        if 'i' in bond and 'j' in bond:
            i, j = bond['i'], bond['j']
            if i in core_atoms and j in core_atoms:
                core_bonds.append({
                    'i': full_to_core[i],
                    'j': full_to_core[j],
                    'order': bond.get('order', 1.0)
                })
    
    # Extract core BFS order
    core_bfs_order = []
    for bond in bfs_order:
        i, j = bond
        if i in core_atoms and j in core_atoms:
            core_bfs_order.append((full_to_core[i], full_to_core[j]))
    
    print(f"  Extracted {len(core_atoms)} atoms from core segment")
    print(f"  Core carbons: {len(core_carbons)}")
    
    # Use core segment for visualization
    positions = core_positions
    elements = core_elements
    bonds = core_bonds
    bfs_order = core_bfs_order

    # Print bond lengths and validate structure
    print("\n[4] Generated bond lengths (core segment):")
    bond_lengths = []
    for bond in bonds:
        i, j = bond['i'], bond['j']
        dist = np.linalg.norm(positions[i] - positions[j])
        bond_lengths.append(dist)
        print(f"  {elements[i]}{i}-{elements[j]}{j}: {dist:.3f} Å")
    
    # Validate structure
    print("\n[4.5] Structure validation:")
    print(f"  Total atoms: {len(elements)}")
    print(f"  Total bonds: {len(bonds)}")
    print(f"  Mean bond length: {np.mean(bond_lengths):.3f} Å")
    print(f"  Std bond length: {np.std(bond_lengths):.3f} Å")
    print(f"  Min bond length: {np.min(bond_lengths):.3f} Å")
    print(f"  Max bond length: {np.max(bond_lengths):.3f} Å")
    
    # Check for unreasonable bond lengths
    unreasonable = [d for d in bond_lengths if d < 0.5 or d > 5.0]
    if unreasonable:
        print(f"  WARNING: {len(unreasonable)} bonds have unreasonable lengths!")
    else:
        print("  ✓ All bond lengths are reasonable")
    
    # Check for overlapping atoms
    min_dist = float('inf')
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                min_dist = dist
    
    print(f"  Minimum interatomic distance: {min_dist:.3f} Å")
    if min_dist < 0.5:
        print(f"  WARNING: Some atoms are too close together!")
    else:
        print("  ✓ No overlapping atoms detected")

    # Visualize
    print("\n[5] Creating visualizations...")
    save_path = output_dir / "polyethylene_generated.png"
    visualize_generated_structure(
        positions, elements, bonds,
        title=f"Generated Polyethylene (CH3-[CH2]_{n_carbons-2}-CH3)",
        save_path=str(save_path)
    )

    # Create BFS animation
    gif_path = output_dir / "polyethylene_bfs_generation.gif"
    create_bfs_animation(
        positions, elements, bonds, bfs_order,
        title="Polyethylene Generation",
        save_path=str(gif_path),
        fps=2
    )

    print("\n" + "="*60)
    print("Generation Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

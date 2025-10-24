"""
Geometry utilities for stereochemistry-aware 3D reconstruction.

This module provides functions for computing local Spherical coordinates
(r, θ, φ) in a bond-centered reference frame, enabling rotationally-invariant
and stereochemically-aware molecular geometry representation.

Key Functions:
- compute_local_geometry: Main function for computing ordered neighbor geometry
- create_orthonormal_frame: Construct local coordinate system
- spherical_to_cartesian: Convert (r, θ, φ) back to 3D positions
"""

import torch
import math
import random
import numpy as np
from typing import List, Tuple, Optional


def create_orthonormal_frame(bond_axis: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create orthonormal frame (x_axis, y_axis) perpendicular to bond_axis.

    Uses Gram-Schmidt orthogonalization to construct a right-handed coordinate
    system with bond_axis as the z-axis.

    Args:
        bond_axis: (3,) tensor, normalized bond direction (z-axis)
        epsilon: Small value for numerical stability

    Returns:
        x_axis: (3,) tensor, first perpendicular axis
        y_axis: (3,) tensor, second perpendicular axis (cross product)

    Note:
        The returned frame is right-handed: x_axis × y_axis = bond_axis
    """
    # Find most perpendicular canonical axis for numerical stability
    abs_components = torch.abs(bond_axis)
    min_idx = torch.argmin(abs_components)

    # Create reference direction
    ref_vec = torch.zeros(3, device=bond_axis.device, dtype=bond_axis.dtype)
    ref_vec[min_idx] = 1.0

    # Gram-Schmidt: make perpendicular to bond_axis
    x_axis = ref_vec - torch.dot(ref_vec, bond_axis) * bond_axis
    x_norm = torch.norm(x_axis)

    if x_norm < epsilon:
        # Extremely rare edge case: try different canonical vector
        min_idx = (min_idx + 1) % 3
        ref_vec = torch.zeros(3, device=bond_axis.device, dtype=bond_axis.dtype)
        ref_vec[min_idx] = 1.0
        x_axis = ref_vec - torch.dot(ref_vec, bond_axis) * bond_axis
        x_norm = torch.norm(x_axis)

    x_axis = x_axis / x_norm  # normalize (local x-axis)

    # Complete the orthonormal basis
    y_axis = torch.cross(bond_axis, x_axis, dim = -1)  # local y-axis
    # y_axis is automatically normalized since bond_axis and x_axis are orthonormal

    return x_axis, y_axis


def compute_local_geometry(
    pos: torch.Tensor,
    center_idx: int,
    neighbor_indices: List[int],
    reference_idx: int,
    epsilon: float = 1e-6,
    augment_rotation: bool = False
) -> Tuple[List[int], List[Tuple[float, float, float]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute local Spherical coordinates (r, θ, φ) for neighbors of center_idx,
    using center→reference bond as the coordinate system axis.

    Returns neighbors ordered by azimuthal angle φ (clockwise when viewing down axis).

    This is the CORE function for stereochemistry-aware geometry representation.
    It computes a rotationally-invariant representation of local molecular geometry
    by expressing neighbor positions in a bond-centered Spherical coordinate system.

    Args:
        pos: (N, 3) tensor of atom positions
        center_idx: Index of the central atom
        neighbor_indices: List of neighbor atom indices (excluding reference)
        reference_idx: Index of reference atom (defines local z-axis)
        epsilon: Threshold for numerical stability (default: 1e-6)
        augment_rotation: If True, apply random rotation to φ angles (for training)

    Returns:
        ordered_neighbor_indices: List of neighbor indices ordered by φ (clockwise)
        geometry: List of (r, θ, φ) tuples for each ordered neighbor
            - r: Distance from center to neighbor (Angstroms)
            - θ: Polar angle from bond axis (0 to π radians)
            - φ: Azimuthal angle around bond axis (0 to 2π radians)
        coord_frame: (x_axis, y_axis, bond_axis) orthonormal frame for this bond

    Example:
        >>> pos = torch.tensor([[0,0,0], [1,0,0], [0.5,0.87,0], [0.5,-0.87,0]])
        >>> center_idx = 0
        >>> neighbors = [2, 3]
        >>> reference = 1
        >>> ordered, geom, frame = compute_local_geometry(pos, 0, neighbors, 1)
        >>> # Returns ordered neighbors with (r, θ, φ) in local frame
    """
    center_pos = pos[center_idx]
    reference_pos = pos[reference_idx]

    # 1. Define bond axis (local z-axis)
    bond_vec = reference_pos - center_pos
    bond_length = torch.norm(bond_vec)

    if bond_length < epsilon:
        # Degenerate case: zero-length bond
        # Return default values to avoid NaN propagation
        default_geom = [(0.0, 0.0, 0.0)] * len(neighbor_indices)
        default_frame = (
            torch.tensor([1., 0., 0.], device=pos.device, dtype=pos.dtype),
            torch.tensor([0., 1., 0.], device=pos.device, dtype=pos.dtype),
            torch.tensor([0., 0., 1.], device=pos.device, dtype=pos.dtype)
        )
        return neighbor_indices, default_geom, default_frame

    bond_axis = bond_vec / bond_length  # normalize (local z-axis)

    # 2. Create orthonormal coordinate system
    x_axis, y_axis = create_orthonormal_frame(bond_axis, epsilon)

    # 3. Compute (r, θ, φ) for each neighbor
    neighbor_geometry = []

    for n_idx in neighbor_indices:
        if n_idx == reference_idx or n_idx == center_idx:
            # Skip reference atom and self-loops
            continue

        neighbor_pos = pos[n_idx]
        vec_to_neighbor = neighbor_pos - center_pos

        # Distance (r)
        r = torch.norm(vec_to_neighbor)

        if r < epsilon:
            # Zero-length bond - use default values
            neighbor_geometry.append((n_idx, 0.0, 0.0, 0.0, False))
            continue

        # Normalize
        vec_norm = vec_to_neighbor / r

        # Polar angle (θ): angle from bond axis
        # θ = 0° means along bond, θ = 90° means perpendicular, θ = 180° means opposite
        cos_theta = torch.dot(vec_norm, bond_axis)
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # Clamp for numerical stability

        # Azimuthal angle (φ): rotation around bond axis
        # Project onto plane perpendicular to bond to find φ
        projected = vec_to_neighbor - cos_theta * r * bond_axis
        projected_length = torch.norm(projected)

        if projected_length < epsilon:
            # Collinear with bond axis - φ is undefined
            # This happens when θ ≈ 0° or θ ≈ 180°
            phi = 0.0
            is_degenerate = True
        else:
            # Normalize projection
            projected_norm = projected / projected_length

            # Compute φ in the plane using the orthonormal frame
            cos_phi = torch.dot(projected_norm, x_axis)
            sin_phi = torch.dot(projected_norm, y_axis)
            phi = torch.atan2(sin_phi, cos_phi)  # Returns -π to π

            # Convert to [0, 2π) range
            if phi < 0:
                phi = phi + 2 * math.pi

            is_degenerate = False

        # Convert to Python floats (phi may already be float if collinear)
        r_val = r.item()
        theta_val = theta.item()
        phi_val = phi.item() if isinstance(phi, torch.Tensor) else phi

        neighbor_geometry.append((n_idx, r_val, theta_val, phi_val, is_degenerate))

    # 4. Apply random rotation augmentation if training
    if augment_rotation and len(neighbor_geometry) > 0:
        # Random rotation angle around bond axis
        rotation_angle = random.uniform(0, 2 * math.pi)

        # Apply rotation to φ angles only (r and θ remain unchanged)
        neighbor_geometry = [
            (idx, r, theta, (phi + rotation_angle) % (2 * math.pi), degen)
            for idx, r, theta, phi, degen in neighbor_geometry
        ]

    # 5. Sort by φ (clockwise order when viewing down bond)
    neighbor_geometry.sort(key=lambda x: x[3])  # Sort by phi

    # Extract results
    ordered_indices = [idx for idx, _, _, _, _ in neighbor_geometry]
    geometry = [(r, theta, phi) for _, r, theta, phi, _ in neighbor_geometry]
    coord_frame = (x_axis, y_axis, bond_axis)

    return ordered_indices, geometry, coord_frame


def spherical_to_cartesian(
    r: float,
    theta: float,
    phi: float,
    coord_frame: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    center_pos: torch.Tensor
) -> torch.Tensor:
    """
    Convert Spherical coordinates (r, θ, φ) to Cartesian position.

    Useful for 3D reconstruction from predicted geometry.

    Args:
        r: Distance from center (Angstroms)
        theta: Polar angle from bond axis (radians, 0 to π)
        phi: Azimuthal angle around bond axis (radians, 0 to 2π)
        coord_frame: (x_axis, y_axis, bond_axis) orthonormal frame
        center_pos: (3,) tensor, position of central atom

    Returns:
        position: (3,) tensor, Cartesian position of the neighbor

    Example:
        >>> r, theta, phi = 1.0, math.pi/2, 0.0  # 1Å perpendicular, along x
        >>> frame = (torch.tensor([1,0,0]), torch.tensor([0,1,0]), torch.tensor([0,0,1]))
        >>> center = torch.tensor([0,0,0])
        >>> pos = spherical_to_cartesian(r, theta, phi, frame, center)
        >>> # Returns approximately [1, 0, 0]
    """
    x_axis, y_axis, bond_axis = coord_frame

    # Convert to Cartesian in local frame
    # θ is angle from bond axis, φ is azimuthal rotation
    x_local = r * torch.sin(torch.tensor(theta)) * torch.cos(torch.tensor(phi))
    y_local = r * torch.sin(torch.tensor(theta)) * torch.sin(torch.tensor(phi))
    z_local = r * torch.cos(torch.tensor(theta))  # Component along bond

    # Transform to world coordinates
    position_local = (x_local * x_axis + y_local * y_axis + z_local * bond_axis)

    # Add to center position
    position = center_pos + position_local

    return position


def validate_geometry_consistency(
    geometry: List[Tuple[float, float, float]],
    tolerance: float = 1e-4
) -> Tuple[bool, str]:
    """
    Validate that computed geometry values are physically reasonable.

    Args:
        geometry: List of (r, θ, φ) tuples
        tolerance: Tolerance for validation checks

    Returns:
        is_valid: True if geometry is valid
        message: Description of any issues found

    Checks:
        - r >= 0 (distances must be non-negative)
        - 0 <= θ <= π (polar angle range)
        - 0 <= φ < 2π (azimuthal angle range)
    """
    for i, (r, theta, phi) in enumerate(geometry):
        # Check distance
        if r < -tolerance:
            return False, f"Negative distance at index {i}: r={r}"

        # Check polar angle
        if theta < -tolerance or theta > math.pi + tolerance:
            return False, f"Invalid polar angle at index {i}: θ={theta} (must be in [0, π])"

        # Check azimuthal angle
        if phi < -tolerance or phi >= 2 * math.pi + tolerance:
            return False, f"Invalid azimuthal angle at index {i}: φ={phi} (must be in [0, 2π))"

    return True, "Geometry is valid"


# Utility functions for reconstruction and analysis

def compute_dihedral_angle(
    pos: torch.Tensor,
    idx_a: int,
    idx_b: int,
    idx_c: int,
    idx_d: int,
    epsilon: float = 1e-6
) -> float:
    """
    Compute dihedral angle defined by four atoms A-B-C-D.

    Args:
        pos: (N, 3) tensor of atom positions
        idx_a, idx_b, idx_c, idx_d: Indices of the four atoms
        epsilon: Small value for numerical stability (default: 1e-6)

    Returns:
        dihedral: Dihedral angle in radians (-π to π)
        Returns 0.0 for degenerate geometries (collinear atoms)

    Note:
        Dihedral angle is the angle between planes A-B-C and B-C-D.
        Used for validation and analysis of torsional geometry.
    """
    # Get positions
    p_a = pos[idx_a]
    p_b = pos[idx_b]
    p_c = pos[idx_c]
    p_d = pos[idx_d]

    # Vectors
    b1 = p_b - p_a
    b2 = p_c - p_b
    b3 = p_d - p_c

    # Normal vectors to the planes
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)

    # Check for degenerate geometries (collinear atoms)
    n1_norm = torch.norm(n1)
    n2_norm = torch.norm(n2)
    b2_norm = torch.norm(b2)

    if n1_norm < epsilon or n2_norm < epsilon or b2_norm < epsilon:
        # Degenerate geometry - atoms are collinear
        return 0.0

    # Normalize
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Compute dihedral
    cos_dihedral = torch.dot(n1, n2)
    dihedral = torch.acos(torch.clamp(cos_dihedral, -1.0, 1.0))

    # Determine sign using the triple product
    m1 = torch.cross(n1, b2 / b2_norm)
    sign = torch.dot(m1, n2)

    if sign < 0:
        dihedral = -dihedral

    return dihedral.item()


def rmsd(positions_pred: torch.Tensor, positions_true: torch.Tensor) -> float:
    """
    Compute Root Mean Square Deviation between predicted and true positions.

    Args:
        positions_pred: (N, 3) tensor of predicted positions
        positions_true: (N, 3) tensor of true positions

    Returns:
        rmsd: Root mean square deviation in Angstroms

    Note:
        Does NOT perform alignment (assumes molecules are already aligned).
        For proper RMSD, use Kabsch algorithm to align first.
    """
    diff = positions_pred - positions_true
    squared_dist = torch.sum(diff ** 2, dim=1)
    return torch.sqrt(torch.mean(squared_dist)).item()


def convert_to_relative_azimuthal(
    geometry: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert absolute azimuthal angles (φ) to relative differences (Δφ).

    The first neighbor is set to Δφ=0 by definition (establishes reference frame).
    Subsequent neighbors have Δφ = difference from previous neighbor's φ.

    Args:
        geometry: (max_bonds, 3) array with (r, θ, φ) or (max_bonds, 2) with (θ, φ)
                  φ should be in degrees [0, 360]
        mask: (max_bonds,) optional mask (1=valid, 0=padding)

    Returns:
        geometry_relative: Same shape as input, with φ replaced by Δφ
                          - If input is (r, θ, φ): returns (r, θ, Δφ)
                          - If input is (θ, φ): returns (θ, Δφ)

    Example:
        >>> geom = np.array([[1.5, 90, 45], [1.2, 120, 120], [1.8, 60, 200]])
        >>> geom_rel = convert_to_relative_azimuthal(geom)
        >>> # Returns: [[1.5, 90, 0], [1.2, 120, 75], [1.8, 60, 80]]
        >>> #   Δφ₁=0, Δφ₂=120-45=75, Δφ₃=200-120=80
    """
    geometry_relative = geometry.copy()
    n_components = geometry.shape[1]
    phi_idx = n_components - 1  # φ is always the last component

    # Find valid neighbors
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
    else:
        # Assume all non-zero rows are valid
        valid_indices = np.where(np.any(geometry != 0, axis=1))[0]

    if len(valid_indices) == 0:
        return geometry_relative

    # First valid neighbor: Δφ = 0 (by definition, sets reference frame)
    geometry_relative[valid_indices[0], phi_idx] = 0.0

    # Subsequent neighbors: Δφ = difference from previous neighbor
    for i in range(1, len(valid_indices)):
        curr_idx = valid_indices[i]
        prev_idx = valid_indices[i-1]

        phi_curr = geometry[curr_idx, phi_idx]
        phi_prev = geometry[prev_idx, phi_idx]

        # Compute difference
        delta_phi = phi_curr - phi_prev

        # Normalize to [0, 360) range
        delta_phi = delta_phi % 360.0

        geometry_relative[curr_idx, phi_idx] = delta_phi

    return geometry_relative

"""
Loss functions for stereochemistry-aware 3D reconstruction.

This module implements specialized loss functions for training the model:
- Percentage-based loss for distances (scale invariance)
- Angle loss normalized by 360° (so 36° error = 0.1 loss)
- Circular angle loss for periodic φ angles with wraparound
- Combined geometry loss with masking for variable bond counts
- Total weighted loss combining all components

Key Features:
- Distance: Percentage error loss (relative to target value)
- Angles (θ, φ): Normalized for consistent [0, 1] range
  - θ: Normalized by 180° (range [0, 180]) - Example: 90° error → 90/180 = 0.5 loss
  - φ: Normalized by 360° (range [0, 360]) - Example: 36° error → 36/360 = 0.1 loss
- Circular loss: properly handles φ angle wraparound (0° = 360°)
- All angles in degrees (θ ∈ [0, 180], φ ∈ [0, 360])
- Masking: handles variable number of bonds per atom
- Equal weighting: distance and geometry losses are weighted equally by default
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


def percentage_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute percentage error loss: |pred - target| / (|target| + epsilon).

    This loss function is scale-invariant. Always uses mean reduction to keep
    values normalized across different batch sizes.

    Args:
        pred: (batch_size, *) predicted values
        target: (batch_size, *) target values
        epsilon: Small value to prevent division by zero (default: 1e-3)
        reduction: 'mean' or 'none' (default: 'mean')

    Returns:
        loss: Mean percentage error (if reduction='mean')
              or per-element loss (if reduction='none')

    Example:
        >>> pred = torch.tensor([1.5, 2.0])
        >>> target = torch.tensor([1.0, 2.5])
        >>> loss = percentage_loss(pred, target)
        >>> # Returns mean of [|1.5-1.0|/1.0, |2.0-2.5|/2.5] = mean([0.5, 0.2]) = 0.35
    """
    # Absolute percentage error
    abs_error = torch.abs(pred - target)
    abs_target = torch.abs(target) + epsilon
    percent_error = abs_error / abs_target

    if reduction == 'mean':
        return torch.mean(percent_error)
    elif reduction == 'none':
        return percent_error
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")


def circular_angle_loss(
    pred_phi: torch.Tensor,
    target_phi: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    period: float = 360.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute circular loss for periodic angles normalized by full circle (360°).

    Azimuthal angle φ is periodic: 0° = 360°.
    This loss properly handles the wraparound by computing the shortest
    angular distance, then normalizes by 360° (full circle).

    Example: 36° error = 36/360 = 0.1 loss

    Args:
        pred_phi: (batch_size, max_bonds) predicted φ angles [0, 360] degrees
        target_phi: (batch_size, max_bonds) target φ angles [0, 360] degrees
        mask: (batch_size, max_bonds) optional mask (1 = valid, 0 = padding)
        period: Period of the angle (default: 360.0 for degrees)
        reduction: 'mean' or 'none' (default: 'mean')

    Returns:
        loss: Mean normalized error [0, 1] or per-element loss

    Example:
        >>> pred = torch.tensor([[36.0, 358.0]])
        >>> target = torch.tensor([[0.0, 0.0]])
        >>> loss = circular_angle_loss(pred, target)
        >>> # 36° error -> 36/360 = 0.1, 2° error -> 2/360 ≈ 0.0056
        >>> # Mean = (0.1 + 0.0056) / 2 ≈ 0.053
    """
    # Compute angular difference
    diff = pred_phi - target_phi

    # Wrap to [-period/2, period/2]
    # This finds the shortest angular distance
    diff = torch.remainder(diff + period / 2, period) - period / 2

    # Absolute error in degrees
    abs_diff = torch.abs(diff)

    # Normalize by period to get [0, 1] range
    normalized_error = abs_diff / period

    # Apply mask if provided
    if mask is not None:
        normalized_error = normalized_error * mask
        if reduction == 'mean':
            # Mean over valid elements only
            num_valid = torch.sum(mask) + 1e-8  # Avoid division by zero
            return torch.sum(normalized_error) / num_valid
        else:  # 'none'
            return normalized_error
    else:
        if reduction == 'mean':
            return torch.mean(normalized_error)
        elif reduction == 'none':
            return normalized_error
        else:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")


def geometry_loss(
    pred_geom: torch.Tensor,
    target_geom: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute combined loss for (θ, Δφ) angular geometry with proper handling of each component.
    Always uses averaging over valid elements.

    Args:
        pred_geom: (batch_size, max_bonds, 2) predicted (θ, Δφ) [θ, Δφ in degrees]
        target_geom: (batch_size, max_bonds, 2) target (θ, Δφ) [θ, Δφ in degrees]
        mask: (batch_size, max_bonds) mask for valid bonds (1 = valid, 0 = padding)
        weights: Optional dict with keys 'theta', 'phi' for component weights
                 Default: {'theta': 1.0, 'phi': 1.0} - all equal
        reduction: 'mean' or 'none' (default: 'mean')

    Returns:
        loss: Mean combined geometry loss

    Note:
        - θ: Normalized by 180° (so 90° error = 0.5 loss, θ range is [0, 180])
        - Δφ: Normalized by 360° (so 36° error = 0.1 loss, Δφ range is [0, 360])
              Represents relative azimuthal differences (no circular wraparound needed)
    """
    if weights is None:
        weights = {'theta': 1.0, 'phi': 1.0}

    # Extract components
    pred_theta = pred_geom[:, :, 0]  # (batch_size, max_bonds)
    pred_delta_phi = pred_geom[:, :, 1]

    target_theta = target_geom[:, :, 0]
    target_delta_phi = target_geom[:, :, 1]

    # Loss for θ (polar angle) - normalized by 180 degrees (θ range is [0, 180])
    theta_diff = torch.abs(pred_theta - target_theta)
    loss_theta = (theta_diff / 180.0) * mask  # Normalize by 180°

    # Loss for Δφ (relative azimuthal difference) - normalized by 360°
    # No circular wraparound needed since these are differences, not absolute angles
    delta_phi_diff = torch.abs(pred_delta_phi - target_delta_phi)
    loss_delta_phi = (delta_phi_diff / 360.0) * mask  # Normalize by 360°

    # Combine with weights and AVERAGE (not sum) the two components
    # This ensures the geometry loss stays in a reasonable [0, ~1] range
    num_components = sum(weights.values())  # Typically 2.0 if all weights are 1.0
    total_loss = (
        weights['theta'] * loss_theta +
        weights['phi'] * loss_delta_phi
    ) / num_components  # Average across components

    # Average over valid elements
    if reduction == 'mean':
        num_valid = torch.sum(mask) + 1e-8
        return torch.sum(total_loss) / num_valid
    elif reduction == 'none':
        return total_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")


def total_loss(
    pred_dist: torch.Tensor,
    pred_geom_a: torch.Tensor,
    pred_geom_b: torch.Tensor,
    data,
    config: Optional[Dict] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute total training loss combining all components.

    Args:
        pred_dist: (batch_size, 1) predicted A-B distances
        pred_geom_a: (batch_size, max_bonds, 2) predicted (θ, Δφ) for A's neighbors
        pred_geom_b: (batch_size, max_bonds, 2) predicted (θ, Δφ) for B's neighbors
        data: PyG Data batch containing target values and masks
        config: Optional dict with loss weights. Default:
                {
                    'loss': {
                        'distance_weight': 1.0,
                        'geometry_weight': 1.0,
                        'theta_weight': 1.0,
                        'phi_weight': 1.0
                    }
                }

    Returns:
        loss_dict: Dictionary containing:
            - 'total': Total combined loss (for backprop)
            - 'distance': Distance loss component
            - 'geometry_a': Geometry loss for node A
            - 'geometry_b': Geometry loss for node B

    Example:
        >>> loss_dict = total_loss(pred_dist, pred_geom_a, pred_geom_b, data, config)
        >>> loss_dict['total'].backward()  # Backprop on total loss
        >>> # Also log individual components for monitoring
    """
    if config is None:
        config = {
            'loss': {
                'distance_weight': 1.0,
                'geometry_weight': 1.0,
                'theta_weight': 1.0,
                'phi_weight': 1.0
            }
        }

    # Geometry component weights - all equal
    geom_weights = {
        'theta': config['loss'].get('theta_weight', 1.0),
        'phi': config['loss'].get('phi_weight', 1.0)
    }

    # Distance loss (percentage-based)
    target_dist = data.target_distance.view(-1, 1)
    loss_distance = percentage_loss(pred_dist, target_dist, epsilon=1e-3, reduction='mean')

    # Geometry loss for node A
    batch_size = pred_geom_a.size(0)
    mask_a = data.bond_mask_a.view(batch_size, -1)
    target_geom_a = data.target_geometry_a.view(batch_size, -1, 2)

    loss_geom_a = geometry_loss(
        pred_geom_a,
        target_geom_a,
        mask_a,
        weights=geom_weights,
        reduction='mean'
    )

    # Geometry loss for node B
    mask_b = data.bond_mask_b.view(batch_size, -1)
    target_geom_b = data.target_geometry_b.view(batch_size, -1, 2)

    loss_geom_b = geometry_loss(
        pred_geom_b,
        target_geom_b,
        mask_b,
        weights=geom_weights,
        reduction='mean'
    )

    # Combined geometry loss (average of A and B)
    loss_geometry = (loss_geom_a + loss_geom_b) / 2

    # Total weighted loss - AVERAGE (not sum) distance and geometry components
    num_components = config['loss']['distance_weight'] + config['loss']['geometry_weight']
    total = (
        config['loss']['distance_weight'] * loss_distance +
        config['loss']['geometry_weight'] * loss_geometry
    ) / num_components  # Average across distance and geometry

    return {
        'total': total,
        'distance': loss_distance,
        'geometry_a': loss_geom_a,
        'geometry_b': loss_geom_b,
        'geometry': loss_geometry
    }


# Utility functions for loss analysis

def compute_angle_errors(
    pred_angles: torch.Tensor,
    target_angles: torch.Tensor,
    mask: torch.Tensor,
    angle_type: str = 'regular'
) -> Dict[str, float]:
    """
    Compute angle error statistics (MAE, RMSE) in degrees.

    Args:
        pred_angles: Predicted angles in degrees
        target_angles: Target angles in degrees (converted from radians in dataset)
        mask: Valid angle mask
        angle_type: 'regular' for θ (non-periodic), 'circular' for φ (periodic)

    Returns:
        stats: Dictionary with 'mae' and 'rmse' in degrees
    """
    with torch.no_grad():
        if angle_type == 'circular':
            # Use circular difference (period = 360°)
            diff = pred_angles - target_angles
            diff = torch.remainder(diff + 180.0, 360.0) - 180.0
            abs_diff = torch.abs(diff)
        else:
            # Regular difference
            abs_diff = torch.abs(pred_angles - target_angles)

        # Apply mask (already in degrees)
        abs_diff = abs_diff * mask

        # Compute statistics
        num_valid = torch.sum(mask)
        if num_valid > 0:
            mae = torch.sum(abs_diff) / num_valid
            rmse = torch.sqrt(torch.sum((abs_diff ** 2) * mask) / num_valid)
        else:
            mae = torch.tensor(0.0)
            rmse = torch.tensor(0.0)

        return {
            'mae': mae.item(),
            'rmse': rmse.item()
        }

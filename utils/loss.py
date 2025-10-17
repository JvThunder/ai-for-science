"""
Loss functions for stereochemistry-aware 3D reconstruction.

This module implements specialized loss functions for training the model:
- Percentage-based loss for better scale invariance
- Circular angle loss for periodic φ angles
- Combined geometry loss with masking for variable bond counts
- Total weighted loss combining all components

Key Features:
- Percentage error loss: relative error scales better across different distance ranges
- Circular loss: properly handles φ angle wraparound (0° = 360°)
- Masking: handles variable number of bonds per atom
- Weighted combination: balances distance vs. geometry prediction
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

    This loss function is scale-invariant and prevents the model from focusing
    only on large values. It's especially useful for distances which can vary
    from ~1Å (H-H) to ~4Å (C-C across rings).

    Args:
        pred: (batch_size, *) predicted values
        target: (batch_size, *) target values
        epsilon: Small value to prevent division by zero (default: 1e-3)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')

    Returns:
        loss: Scalar loss value (if reduction='mean' or 'sum')
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
    elif reduction == 'sum':
        return torch.sum(percent_error)
    elif reduction == 'none':
        return percent_error
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")


def circular_angle_loss(
    pred_phi: torch.Tensor,
    target_phi: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    period: float = 2 * math.pi,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute circular loss for periodic angles (handles wraparound).

    Azimuthal angle φ is periodic: 0° = 360° = 2π radians.
    This loss properly handles the wraparound by computing the shortest
    angular distance between predicted and target angles.

    Args:
        pred_phi: (batch_size, max_bonds) predicted φ angles [0, 2π]
        target_phi: (batch_size, max_bonds) target φ angles [0, 2π]
        mask: (batch_size, max_bonds) optional mask (1 = valid, 0 = padding)
        period: Period of the angle (default: 2π for radians)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')

    Returns:
        loss: Scalar loss value or per-element loss

    Example:
        >>> pred = torch.tensor([[0.1, 6.2]])  # 6.2 ≈ 2π (close to 0)
        >>> target = torch.tensor([[0.0, 0.0]])
        >>> loss = circular_angle_loss(pred, target)
        >>> # Uses shortest path: min(|6.2-0|, 2π-|6.2-0|) ≈ 0.1
    """
    # Compute angular difference
    diff = pred_phi - target_phi

    # Wrap to [-period/2, period/2]
    # This finds the shortest angular distance
    diff = torch.remainder(diff + period / 2, period) - period / 2

    # Absolute error
    abs_diff = torch.abs(diff)

    # Apply mask if provided
    if mask is not None:
        abs_diff = abs_diff * mask
        if reduction == 'mean':
            # Mean over valid elements only
            num_valid = torch.sum(mask) + 1e-8  # Avoid division by zero
            return torch.sum(abs_diff) / num_valid
        elif reduction == 'sum':
            return torch.sum(abs_diff)
        else:  # 'none'
            return abs_diff
    else:
        if reduction == 'mean':
            return torch.mean(abs_diff)
        elif reduction == 'sum':
            return torch.sum(abs_diff)
        elif reduction == 'none':
            return abs_diff
        else:
            raise ValueError(f"Invalid reduction: {reduction}")


def geometry_loss(
    pred_geom: torch.Tensor,
    target_geom: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute combined loss for (r, θ, φ) geometry with proper handling of each component.

    Args:
        pred_geom: (batch_size, max_bonds, 3) predicted (r, θ, φ)
        target_geom: (batch_size, max_bonds, 3) target (r, θ, φ)
        mask: (batch_size, max_bonds) mask for valid bonds (1 = valid, 0 = padding)
        weights: Optional dict with keys 'r', 'theta', 'phi' for component weights
                 Default: {'r': 1.0, 'theta': 0.5, 'phi': 0.5}
        reduction: 'mean', 'sum', or 'none' (default: 'mean')

    Returns:
        loss: Combined geometry loss

    Note:
        - r: Uses percentage loss (scale-invariant)
        - θ: Uses MSE loss (bounded [0, π])
        - φ: Uses circular loss (periodic [0, 2π])
    """
    if weights is None:
        weights = {'r': 1.0, 'theta': 0.5, 'phi': 0.5}

    # Extract components
    pred_r = pred_geom[:, :, 0]  # (batch_size, max_bonds)
    pred_theta = pred_geom[:, :, 1]
    pred_phi = pred_geom[:, :, 2]

    target_r = target_geom[:, :, 0]
    target_theta = target_geom[:, :, 1]
    target_phi = target_geom[:, :, 2]

    # Loss for r (distance) - percentage loss
    loss_r = percentage_loss(pred_r, target_r, epsilon=1e-3, reduction='none')
    loss_r = loss_r * mask  # Apply mask

    # Loss for θ (polar angle) - MSE loss
    loss_theta = F.mse_loss(pred_theta, target_theta, reduction='none')
    loss_theta = loss_theta * mask  # Apply mask

    # Loss for φ (azimuthal angle) - circular loss
    loss_phi = circular_angle_loss(pred_phi, target_phi, mask=mask, reduction='none')

    # Combine with weights
    total_loss = (
        weights['r'] * loss_r +
        weights['theta'] * loss_theta +
        weights['phi'] * loss_phi
    )

    # Reduce
    if reduction == 'mean':
        num_valid = torch.sum(mask) + 1e-8
        return torch.sum(total_loss) / num_valid
    elif reduction == 'sum':
        return torch.sum(total_loss)
    elif reduction == 'none':
        return total_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


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
        pred_geom_a: (batch_size, max_bonds, 3) predicted geometry for A's neighbors
        pred_geom_b: (batch_size, max_bonds, 3) predicted geometry for B's neighbors
        data: PyG Data batch containing target values and masks
        config: Optional dict with loss weights. Default:
                {
                    'distance_weight': 1.0,
                    'geometry_weight': 0.3,
                    'r_weight': 1.0,
                    'theta_weight': 0.5,
                    'phi_weight': 0.5
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
            'distance_weight': 1.0,
            'geometry_weight': 0.3,
            'r_weight': 1.0,
            'theta_weight': 0.5,
            'phi_weight': 0.5
        }

    # Geometry component weights
    geom_weights = {
        'r': config['loss'].get('r_weight', 1.0),
        'theta': config['loss'].get('theta_weight', 0.5),
        'phi': config['loss'].get('phi_weight', 0.5)
    }

    # Distance loss (percentage-based)
    target_dist = data.target_distance.view(-1, 1)
    loss_distance = percentage_loss(pred_dist, target_dist, epsilon=1e-3, reduction='mean')

    # Geometry loss for node A
    batch_size = pred_geom_a.size(0)
    mask_a = data.bond_mask_a.view(batch_size, -1)
    target_geom_a = data.target_geometry_a.view(batch_size, -1, 3)

    loss_geom_a = geometry_loss(
        pred_geom_a,
        target_geom_a,
        mask_a,
        weights=geom_weights,
        reduction='mean'
    )

    # Geometry loss for node B
    mask_b = data.bond_mask_b.view(batch_size, -1)
    target_geom_b = data.target_geometry_b.view(batch_size, -1, 3)

    loss_geom_b = geometry_loss(
        pred_geom_b,
        target_geom_b,
        mask_b,
        weights=geom_weights,
        reduction='mean'
    )

    # Combined geometry loss
    loss_geometry = (loss_geom_a + loss_geom_b) / 2

    # Total weighted loss
    total = (
        config['loss']['distance_weight'] * loss_distance +
        config['loss']['geometry_weight'] * loss_geometry
    )

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
        pred_angles: Predicted angles in radians
        target_angles: Target angles in radians
        mask: Valid angle mask
        angle_type: 'regular' for θ, 'circular' for φ

    Returns:
        stats: Dictionary with 'mae' and 'rmse' in degrees
    """
    with torch.no_grad():
        if angle_type == 'circular':
            # Use circular difference
            diff = pred_angles - target_angles
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
            abs_diff = torch.abs(diff)
        else:
            # Regular difference
            abs_diff = torch.abs(pred_angles - target_angles)

        # Apply mask and convert to degrees
        abs_diff = abs_diff * mask
        abs_diff_deg = abs_diff * (180.0 / math.pi)

        # Compute statistics
        num_valid = torch.sum(mask)
        if num_valid > 0:
            mae = torch.sum(abs_diff_deg) / num_valid
            rmse = torch.sqrt(torch.sum((abs_diff_deg ** 2) * mask) / num_valid)
        else:
            mae = torch.tensor(0.0)
            rmse = torch.tensor(0.0)

        return {
            'mae': mae.item(),
            'rmse': rmse.item()
        }

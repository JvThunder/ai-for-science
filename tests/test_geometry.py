"""
Unit tests for geometry utilities.

Tests the core stereochemistry-aware geometry functions including
local coordinate frame construction and Spherical coordinate computation.
"""

import pytest
import torch
import math
import sys
sys.path.append('..')

from utils.geometry import (
    create_orthonormal_frame,
    compute_local_geometry,
    spherical_to_cartesian,
    validate_geometry_consistency,
    compute_dihedral_angle,
    rmsd
)


class TestOrthonormalFrame:
    """Tests for orthonormal frame construction."""

    def test_frame_is_orthonormal(self):
        """Test that constructed frame is orthonormal."""
        bond_axis = torch.tensor([1., 0., 0.])  # Along x-axis
        x_axis, y_axis = create_orthonormal_frame(bond_axis)

        # Check normalization
        assert torch.allclose(torch.norm(x_axis), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(torch.norm(y_axis), torch.tensor(1.0), atol=1e-6)

        # Check orthogonality
        assert torch.allclose(torch.dot(bond_axis, x_axis), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(torch.dot(bond_axis, y_axis), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(torch.dot(x_axis, y_axis), torch.tensor(0.0), atol=1e-6)

        # Check right-handedness: x × y = z
        cross = torch.cross(x_axis, y_axis)
        assert torch.allclose(cross, bond_axis, atol=1e-6)

    def test_different_bond_orientations(self):
        """Test frame construction for various bond orientations."""
        test_axes = [
            torch.tensor([1., 0., 0.]),
            torch.tensor([0., 1., 0.]),
            torch.tensor([0., 0., 1.]),
            torch.tensor([1., 1., 1.]) / math.sqrt(3),  # Diagonal
            torch.tensor([1., 2., 3.]) / math.sqrt(14),  # Random direction
        ]

        for bond_axis in test_axes:
            x_axis, y_axis = create_orthonormal_frame(bond_axis)

            # Verify orthonormality
            assert torch.allclose(torch.norm(x_axis), torch.tensor(1.0), atol=1e-6)
            assert torch.allclose(torch.norm(y_axis), torch.tensor(1.0), atol=1e-6)
            assert torch.allclose(torch.dot(bond_axis, x_axis), torch.tensor(0.0), atol=1e-6)
            assert torch.allclose(torch.dot(bond_axis, y_axis), torch.tensor(0.0), atol=1e-6)


class TestLocalGeometry:
    """Tests for local geometry computation."""

    def test_linear_molecule(self):
        """Test with linear molecule A-B-C."""
        # Linear: A at origin, B at (1,0,0), C at (2,0,0)
        pos = torch.tensor([
            [0., 0., 0.],  # A (center)
            [1., 0., 0.],  # B (reference)
            [2., 0., 0.],  # C (collinear neighbor)
        ])

        ordered, geometry, frame = compute_local_geometry(
            pos, center_idx=0, neighbor_indices=[2], reference_idx=1
        )

        # C is collinear with A-B bond (θ ≈ 0)
        assert len(geometry) == 1  # C is included
        r, theta, phi = geometry[0]
        assert torch.allclose(torch.tensor(r), torch.tensor(2.0), atol=1e-4)
        assert torch.allclose(torch.tensor(theta), torch.tensor(0.0), atol=1e-4)  # Aligned with bond

    def test_tetrahedral_geometry(self):
        """Test with tetrahedral geometry (methane-like)."""
        # Central atom at origin, 4 neighbors at tetrahedral positions
        pos = torch.tensor([
            [0., 0., 0.],  # Center
            [1., 0., 0.],  # Reference (along x)
            [0., 1., 0.],  # Neighbor 1 (90° from reference)
            [0., 0., 1.],  # Neighbor 2 (90° from reference, different plane)
        ])

        ordered, geometry, frame = compute_local_geometry(
            pos, center_idx=0, neighbor_indices=[1, 2, 3], reference_idx=1
        )

        # Should return 2 neighbors (excluding reference itself)
        assert len(geometry) == 2

        # All neighbors should be at distance ~1.0
        for r, theta, phi in geometry:
            assert torch.allclose(torch.tensor(r), torch.tensor(1.0), atol=1e-4)

        # θ should be 90° (π/2) for perpendicular neighbors
        for r, theta, phi in geometry:
            assert torch.allclose(torch.tensor(theta), torch.tensor(math.pi/2), atol=1e-4)

    def test_planar_geometry(self):
        """Test with planar geometry (ethylene-like)."""
        # All atoms in xy-plane
        pos = torch.tensor([
            [0., 0., 0.],      # Center
            [1., 0., 0.],      # Reference
            [0.5, 0.87, 0.],   # 120° ccw
            [0.5, -0.87, 0.],  # 120° cw
        ])

        ordered, geometry, frame = compute_local_geometry(
            pos, center_idx=0, neighbor_indices=[1, 2, 3], reference_idx=1
        )

        # Should return 2 neighbors
        assert len(geometry) == 2

        # Verify ordering by φ
        phi_values = [phi for r, theta, phi in geometry]
        # φ values should be in ascending order (clockwise)
        assert phi_values == sorted(phi_values)

    def test_rotation_augmentation(self):
        """Test that rotation augmentation works correctly."""
        pos = torch.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])

        # Without augmentation
        ordered1, geometry1, _ = compute_local_geometry(
            pos, center_idx=0, neighbor_indices=[1, 2, 3], reference_idx=1,
            augment_rotation=False
        )

        # With augmentation
        ordered2, geometry2, _ = compute_local_geometry(
            pos, center_idx=0, neighbor_indices=[1, 2, 3], reference_idx=1,
            augment_rotation=True
        )

        # r and θ should be unchanged
        for (r1, theta1, _), (r2, theta2, _) in zip(geometry1, geometry2):
            assert torch.allclose(torch.tensor(r1), torch.tensor(r2), atol=1e-6)
            assert torch.allclose(torch.tensor(theta1), torch.tensor(theta2), atol=1e-6)

        # φ values may be different (rotated)
        # But relative differences should be preserved
        if len(geometry1) >= 2:
            phi_diff1 = abs(geometry1[1][2] - geometry1[0][2])
            phi_diff2 = abs(geometry2[1][2] - geometry2[0][2])
            # Handle wraparound
            phi_diff1 = min(phi_diff1, 2*math.pi - phi_diff1)
            phi_diff2 = min(phi_diff2, 2*math.pi - phi_diff2)
            assert torch.allclose(torch.tensor(phi_diff1), torch.tensor(phi_diff2), atol=1e-4)

    def test_zero_length_bond(self):
        """Test handling of degenerate case: zero-length bond."""
        pos = torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],  # Same position as center!
            [1., 0., 0.],
        ])

        ordered, geometry, frame = compute_local_geometry(
            pos, center_idx=0, neighbor_indices=[2], reference_idx=1
        )

        # Should return default values, not crash
        assert isinstance(geometry, list)


class TestSphericalToCartesian:
    """Tests for coordinate conversion."""

    def test_simple_conversion(self):
        """Test basic spherical to Cartesian conversion."""
        # Frame aligned with world axes
        frame = (
            torch.tensor([1., 0., 0.]),
            torch.tensor([0., 1., 0.]),
            torch.tensor([0., 0., 1.])
        )
        center = torch.tensor([0., 0., 0.])

        # Point 1Å along x-axis (θ=90°, φ=0°)
        pos = spherical_to_cartesian(1.0, math.pi/2, 0.0, frame, center)
        expected = torch.tensor([1., 0., 0.])
        assert torch.allclose(pos, expected, atol=1e-4)

        # Point 1Å along y-axis (θ=90°, φ=90°)
        pos = spherical_to_cartesian(1.0, math.pi/2, math.pi/2, frame, center)
        expected = torch.tensor([0., 1., 0.])
        assert torch.allclose(pos, expected, atol=1e-4)

        # Point 1Å along z-axis (θ=0°)
        pos = spherical_to_cartesian(1.0, 0.0, 0.0, frame, center)
        expected = torch.tensor([0., 0., 1.])
        assert torch.allclose(pos, expected, atol=1e-4)


class TestValidation:
    """Tests for geometry validation."""

    def test_valid_geometry(self):
        """Test validation of valid geometry."""
        geometry = [
            (1.0, math.pi/4, math.pi/2),
            (1.5, math.pi/2, 0.0),
            (2.0, math.pi*3/4, math.pi),
        ]

        is_valid, message = validate_geometry_consistency(geometry)
        assert is_valid
        assert "valid" in message.lower()

    def test_invalid_distance(self):
        """Test detection of negative distance."""
        geometry = [(-1.0, math.pi/2, 0.0)]  # Negative r!

        is_valid, message = validate_geometry_consistency(geometry)
        assert not is_valid
        assert "distance" in message.lower()

    def test_invalid_theta(self):
        """Test detection of out-of-range θ."""
        geometry = [(1.0, -0.1, 0.0)]  # θ < 0!

        is_valid, message = validate_geometry_consistency(geometry)
        assert not is_valid
        assert "polar" in message.lower()

    def test_invalid_phi(self):
        """Test detection of out-of-range φ."""
        geometry = [(1.0, math.pi/2, 2*math.pi + 0.1)]  # φ >= 2π!

        is_valid, message = validate_geometry_consistency(geometry)
        assert not is_valid
        assert "azimuthal" in message.lower()


class TestDihedralAngle:
    """Tests for dihedral angle computation."""

    def test_planar_dihedral(self):
        """Test dihedral angle for planar configuration (0°)."""
        # Four atoms in a plane: should give dihedral ≈ 0° or 180°
        pos = torch.tensor([
            [0., 0., 0.],
            [1., 0., 0.],
            [2., 0., 0.],
            [3., 0., 0.],
        ])

        dihedral = compute_dihedral_angle(pos, 0, 1, 2, 3)
        # All in a line, dihedral is degenerate but should not crash
        assert isinstance(dihedral, float)

    def test_perpendicular_dihedral(self):
        """Test dihedral angle for perpendicular configuration (90°)."""
        # Create a proper non-degenerate dihedral
        # A-B-C-D where planes ABC and BCD are perpendicular
        pos = torch.tensor([
            [0., 0., 0.],  # A
            [1., 0., 0.],  # B
            [1., 1., 0.],  # C
            [1., 1., 1.],  # D
        ])

        dihedral = compute_dihedral_angle(pos, 0, 1, 2, 3)
        # Should be close to 90° or -90°
        assert abs(abs(dihedral) - math.pi/2) < 0.1


class TestRMSD:
    """Tests for RMSD calculation."""

    def test_identical_structures(self):
        """Test RMSD for identical structures."""
        pos1 = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        pos2 = pos1.clone()

        rmsd_value = rmsd(pos1, pos2)
        assert torch.allclose(torch.tensor(rmsd_value), torch.tensor(0.0), atol=1e-6)

    def test_translated_structures(self):
        """Test RMSD for translated structures."""
        pos1 = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        pos2 = pos1 + torch.tensor([1., 1., 1.])  # Translate by (1,1,1)

        rmsd_value = rmsd(pos1, pos2)
        expected_rmsd = math.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        assert torch.allclose(torch.tensor(rmsd_value), torch.tensor(expected_rmsd), atol=1e-4)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

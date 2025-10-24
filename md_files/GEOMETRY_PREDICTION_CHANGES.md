# Geometry Prediction Architecture Changes

## Date: 2025-01-24

## Problem Statement

### Issue 1: Redundant Distance Prediction
The current model predicts distance (r) in **two places**:
1. **Distance Head**: Predicts the A-B bond distance
2. **Geometry Head**: Predicts (r, θ, φ) for each neighbor, including r again

This creates **conflicting predictions** for the same quantity (distance to neighbors), leading to:
- Wasted model capacity
- Potential inconsistencies between the two predictions
- More complex training dynamics

### Issue 2: Absolute Azimuthal Angles
Currently predicting **absolute** φ angles (0-360°) for each neighbor:
- The azimuthal angle is relative to an arbitrary reference direction
- All neighbors' φ values are independent
- No inherent structure or constraint in the predictions

## Proposed Solution

### Change 1: Remove r from Geometry Head
**Geometry Head Output: (θ, φ) only** - 2 values per neighbor instead of 3

- Distance information comes ONLY from the distance head (A-B distance)
- For neighbor distances: use the ground truth r values or rely on θ alone for angular information
- Geometry head focuses purely on **angular geometry** (θ, φ)

**Rationale:**
- Eliminates conflicting predictions
- Reduces model complexity (fewer parameters to learn)
- Cleaner separation of concerns: distance head predicts distances, geometry head predicts angles

### Change 2: Relative Azimuthal Angles (Δφ)
**Instead of:** φ₁, φ₂, φ₃, ... (absolute angles)
**Predict:** Δφ₁=0, Δφ₂, Δφ₃, ... (relative angle differences)

**How it works:**
```
Neighbor 1: φ₁ = 0°           (by definition, sets reference frame)
Neighbor 2: φ₂ = φ₁ + Δφ₂    (predict Δφ₂ = angle difference from neighbor 1)
Neighbor 3: φ₃ = φ₂ + Δφ₃    (predict Δφ₃ = angle difference from neighbor 2)
...
```

**Example:**
```
Ground truth absolute φ: [45°, 120°, 200°, 315°]

Convert to relative Δφ:
Δφ₁ = 0°                    (first neighbor by definition)
Δφ₂ = 120° - 45° = 75°     (difference from neighbor 1)
Δφ₃ = 200° - 120° = 80°    (difference from neighbor 2)
Δφ₄ = 315° - 200° = 115°   (difference from neighbor 3)

Model predicts: [0°, 75°, 80°, 115°]
```

**Rationale:**
- **Reduced prediction space**: Δφ values tend to be smaller and more structured than absolute angles
- **Consistent reference frame**: First neighbor always at φ=0 eliminates rotational ambiguity
- **Local structure**: Angle differences between consecutive neighbors may be easier to learn
- **Physically meaningful**: The relative orientation between neighbors is what matters for stereochemistry

## Implementation Details

### Model Architecture Changes

**Before:**
```python
geometry_head: Linear(hidden, max_bonds * 3)  # Outputs (r, θ, φ) for each bond
Output shape: (batch, max_bonds, 3)
- [:, :, 0] = r (distance)
- [:, :, 1] = θ (polar angle)
- [:, :, 2] = φ (azimuthal angle)
```

**After:**
```python
geometry_head: Linear(hidden, max_bonds * 2)  # Outputs (θ, Δφ) for each bond
Output shape: (batch, max_bonds, 2)
- [:, :, 0] = θ (polar angle)
- [:, :, 1] = Δφ (relative azimuthal difference)
```

### Activation Functions

**Before:**
```python
r = softplus(raw[:, :, 0])        # Positive distances
θ = sigmoid(raw[:, :, 1]) * 180   # [0, 180°]
φ = sigmoid(raw[:, :, 2]) * 360   # [0, 360°]
```

**After:**
```python
θ = sigmoid(raw[:, :, 0]) * 180   # [0, 180°]
Δφ = sigmoid(raw[:, :, 1]) * 360  # [0, 360°] for differences
```

Note: Δφ can theoretically be [0, 360°] since consecutive neighbors could wrap around

### Loss Function Changes

**Before:**
```python
geometry_loss computes:
- loss_r: percentage loss on distances
- loss_θ: normalized angle loss (÷180°)
- loss_φ: circular angle loss (÷360°)
```

**After:**
```python
geometry_loss computes:
- loss_θ: normalized angle loss (÷180°)  # UNCHANGED
- loss_Δφ: difference angle loss (÷360°)  # Changed from absolute to relative
```

No need for circular wraparound in Δφ loss since we're predicting differences directly.

### Dataset Changes

Need to compute relative azimuthal differences during data preparation:

```python
# In dataset.py or geometry.py
def compute_relative_azimuthal(phi_absolute: np.ndarray) -> np.ndarray:
    """
    Convert absolute azimuthal angles to relative differences.

    Args:
        phi_absolute: (max_bonds,) absolute φ angles [0, 360°]

    Returns:
        phi_relative: (max_bonds,) relative Δφ [first=0, rest=differences]
    """
    phi_relative = np.zeros_like(phi_absolute)
    phi_relative[0] = 0.0  # First neighbor at reference

    for i in range(1, len(phi_absolute)):
        if phi_absolute[i] > 0:  # Valid neighbor (not padding)
            # Compute difference from previous neighbor
            diff = phi_absolute[i] - phi_absolute[i-1]
            # Normalize to [0, 360)
            diff = diff % 360.0
            phi_relative[i] = diff

    return phi_relative
```

### Reconstruction (Inference)

To reconstruct absolute φ angles from predictions:

```python
def reconstruct_absolute_phi(delta_phi: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct absolute φ angles from relative differences.

    Args:
        delta_phi: (batch, max_bonds) predicted Δφ values

    Returns:
        phi_absolute: (batch, max_bonds) reconstructed absolute φ
    """
    phi_absolute = torch.zeros_like(delta_phi)
    phi_absolute[:, 0] = 0.0  # First neighbor at reference

    # Cumulative sum to get absolute angles
    for i in range(1, delta_phi.shape[1]):
        phi_absolute[:, i] = (phi_absolute[:, i-1] + delta_phi[:, i]) % 360.0

    return phi_absolute
```

## Benefits

1. **Eliminates redundancy**: No conflicting r predictions
2. **Reduces model size**: 2 outputs per neighbor instead of 3
3. **Better structure**: Relative angles may be easier to learn
4. **Consistent reference**: First neighbor defines coordinate frame
5. **Physically motivated**: Captures local angular relationships

## Potential Concerns

1. **Error propagation**: Errors in Δφ predictions accumulate when reconstructing absolute angles
   - Mitigation: This is a valid concern, but the model learns Δφ directly during training

2. **Loss of distance information**: Geometry head no longer predicts r
   - Mitigation: Distance head still predicts A-B distance; neighbor distances from ground truth

3. **Ordering dependency**: Predictions now depend on neighbor ordering
   - Mitigation: Neighbors are already ordered by the dataset (consistent ordering)

## Files to Modify

1. **utils/model.py**:
   - Change geometry_head output size: `max_bonds * 3` → `max_bonds * 2`
   - Update forward pass activations
   - Update docstrings

2. **utils/loss.py**:
   - Modify `geometry_loss()` to handle 2-component (θ, Δφ) instead of 3-component (r, θ, φ)
   - Remove r loss computation
   - Change φ loss to Δφ loss (may not need circular wraparound)

3. **utils/dataset.py** or **utils/geometry.py**:
   - Add function to compute relative azimuthal differences
   - Modify data loading to compute Δφ from absolute φ
   - Store Δφ in target_geometry fields

4. **config.yaml**:
   - Update loss weights (remove r_weight, keep theta_weight and phi_weight)
   - Update comments/documentation

5. **README.md** and other documentation:
   - Update model description
   - Update examples

## Testing Plan

1. **Verify data transformation**: Check that relative Δφ computation is correct
2. **Model output shape**: Ensure geometry head outputs (batch, max_bonds, 2)
3. **Loss computation**: Verify loss handles 2-component geometry
4. **Reconstruction**: Test that cumulative Δφ → absolute φ works correctly
5. **Training**: Run a few iterations to ensure everything works end-to-end

## Status

- [x] Analysis complete
- [x] Implementation complete
- [ ] Testing pending
- [x] Documentation updated

## Implementation Summary

All changes have been successfully implemented:

1. **utils/model.py**:
   - Changed geometry_head output size from `max_bonds * 3` to `max_bonds * 2`
   - Updated forward pass to output (θ, Δφ) instead of (r, θ, φ)
   - Updated all docstrings and comments

2. **utils/loss.py**:
   - Modified `geometry_loss()` to handle 2-component (θ, Δφ) format
   - Removed r loss computation
   - Updated Δφ loss (no circular wraparound needed for differences)
   - Removed r_weight from total_loss configuration

3. **utils/geometry.py**:
   - Added `convert_to_relative_azimuthal()` function
   - Converts absolute φ angles to relative Δφ differences
   - First neighbor always has Δφ=0 (reference frame)

4. **utils/dataset.py**:
   - Updated imports to include `convert_to_relative_azimuthal`
   - Modified geometry processing pipeline:
     - Creates bond masks first
     - Converts (r, θ, φ) to (r, θ, Δφ)
     - Extracts only (θ, Δφ), discards r
   - Updated target_geometry tensors to shape (max_bonds, 2)

5. **config.yaml**:
   - Removed r_weight from loss configuration
   - Updated comments to reflect (θ, Δφ) prediction
   - Kept theta_weight and phi_weight (now for Δφ)

## Next Steps

Run training to verify the changes work correctly. The model should now:
- Predict 2 values per neighbor instead of 3
- Have fewer parameters (reduced model size)
- Learn relative azimuthal differences (potentially easier to learn)

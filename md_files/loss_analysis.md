# Analysis of Circular Loss Logic in loss.py

## Summary

✅ **Circular loss is CORRECTLY applied to the right angles** (phi gets circular, theta doesn't)
❌ **CRITICAL BUG**: Dataset stores angles in RADIANS, but loss expects DEGREES
⚠️ **Theta normalization is incorrect**: Should divide by 180°, not 360°
⚠️ **Minor bug in circular_angle_loss**: Line 114 hardcodes 360.0 instead of using `period` parameter

---

## Detailed Analysis

### 1. Data Flow Through the System

```
Dataset (geometry.py)          Model (model.py)                Loss (loss.py)
─────────────────────         ──────────────────              ───────────────
theta: [0, π] RADIANS    -->  theta: [0, 180] DEGREES    -->  Expects DEGREES
phi:   [0, 2π] RADIANS   -->  phi:   [0, 360] DEGREES    -->  Expects DEGREES
```

**Issue**: Dataset stores in RADIANS (geometry.py:157, 176), but:
- Model outputs in DEGREES (model.py:181-182, 192-193)
- Loss expects DEGREES (loss.py:17)
- **NO CONVERSION** between dataset radians and loss degrees!

### 2. Which Angles Should Use Circular Loss?

| Angle | Range | Periodic? | Should be Circular? | Current Implementation |
|-------|-------|-----------|---------------------|------------------------|
| **θ (theta)** | [0, 180°] or [0, π rad] | ❌ NO | ❌ NO | ✅ Correctly non-circular (line 178) |
| **φ (phi)** | [0, 360°] or [0, 2π rad] | ✅ YES (0° = 360°) | ✅ YES | ✅ Correctly circular (line 182) |

**Verdict**: ✅ The circular loss is correctly applied to phi and NOT to theta.

### 3. Issues with Current Implementation

#### Issue #1: UNITS MISMATCH (CRITICAL BUG)
**Location**: Dataset → Loss

**Problem**:
```python
# geometry.py line 157, 176
theta = torch.acos(...)  # Returns [0, π] RADIANS
phi = torch.atan2(...)   # Returns [-π, π] RADIANS, converted to [0, 2π]

# dataset.py line 682-683
geometry_a_padded.append([r, theta, phi])  # Stored in RADIANS

# loss.py line 170-171
target_theta = target_geom[:, :, 1]  # Still in RADIANS
target_phi = target_geom[:, :, 2]    # Still in RADIANS

# model.py line 181-182
theta_a = torch.sigmoid(...) * 180.0  # Predicted in DEGREES
phi_a = torch.sigmoid(...) * 360.0    # Predicted in DEGREES
```

**Result**: You're comparing DEGREES (predicted) vs RADIANS (target)!
- Example: Predicted theta = 90°, Target theta = π/2 rad ≈ 1.57
- Loss thinks error is |90 - 1.57| / 360 ≈ 0.246, which is nonsense!

**Fix**: Convert target angles from radians to degrees before loss computation:
```python
target_theta_deg = torch.rad2deg(target_theta)
target_phi_deg = torch.rad2deg(target_phi)
```

---

#### Issue #2: Theta Normalization Incorrect
**Location**: loss.py line 179

**Problem**:
```python
loss_theta = (theta_diff / 360.0) * mask  # Normalizes by 360°
```

**Issue**: Theta ranges from [0, 180°], NOT [0, 360°]!
- A 90° error in theta gives: 90/360 = 0.25 loss
- But 90° is actually 50% of the full theta range (180°)!
- Should give: 90/180 = 0.5 loss

**Fix**: Change line 179 to:
```python
loss_theta = (theta_diff / 180.0) * mask  # Normalize by 180°
```

---

#### Issue #3: Circular Loss Hardcoded Normalization
**Location**: loss.py line 114

**Problem**:
```python
def circular_angle_loss(..., period: float = 360.0, ...):
    # ...
    normalized_error = abs_diff / 360.0  # Hardcoded!
```

**Issue**: The function accepts a `period` parameter but ignores it on line 114!

**Fix**: Change line 114 to:
```python
normalized_error = abs_diff / period
```

---

### 4. Circular Loss Logic (Assuming Units Were Correct)

**Current implementation** (lines 103-132):
```python
# Compute difference
diff = pred_phi - target_phi

# Wrap to [-period/2, period/2] to find shortest angular distance
diff = torch.remainder(diff + period / 2, period) - period / 2

# Take absolute value and normalize
abs_diff = torch.abs(diff)
normalized_error = abs_diff / 360.0  # Should be: / period
```

**Is this correct?** ✅ YES! This properly handles wraparound:
- Example 1: pred=10°, target=350° → diff=20°, wraps to -20° → |−20|=20° ✓
- Example 2: pred=350°, target=10° → diff=340°, wraps to -20° → |−20|=20° ✓
- Example 3: pred=180°, target=190° → diff=-10°, wraps to -10° → 10° ✓

The wraparound logic is mathematically correct for periodic angles.

---

### 5. Verification: compute_angle_errors Function

**Location**: lines 309-352

This function has the SAME units mismatch:
```python
if angle_type == 'circular':
    diff = pred_angles - target_angles  # DEGREES vs RADIANS!
    diff = torch.remainder(diff + 180.0, 360.0) - 180.0
```

**Also affected by the radians/degrees bug.**

---

## Recommendations

### Priority 1: FIX UNITS MISMATCH (CRITICAL)

Add conversion in `total_loss()` function (loss.py line 203):

```python
def total_loss(...):
    # ... existing code ...

    # Extract targets and convert from radians to degrees
    target_dist = data.target_distance.view(-1, 1)

    # CONVERT ANGLES FROM RADIANS TO DEGREES
    target_geom_a_rad = data.target_geometry_a.view(batch_size, -1, 3)
    target_geom_b_rad = data.target_geometry_b.view(batch_size, -1, 3)

    target_geom_a = torch.zeros_like(target_geom_a_rad)
    target_geom_a[:, :, 0] = target_geom_a_rad[:, :, 0]  # r (no conversion)
    target_geom_a[:, :, 1] = torch.rad2deg(target_geom_a_rad[:, :, 1])  # theta
    target_geom_a[:, :, 2] = torch.rad2deg(target_geom_a_rad[:, :, 2])  # phi

    target_geom_b = torch.zeros_like(target_geom_b_rad)
    target_geom_b[:, :, 0] = target_geom_b_rad[:, :, 0]  # r (no conversion)
    target_geom_b[:, :, 1] = torch.rad2deg(target_geom_b_rad[:, :, 1])  # theta
    target_geom_b[:, :, 2] = torch.rad2deg(target_geom_b_rad[:, :, 2])  # phi

    # Now use converted targets
    loss_geom_a = geometry_loss(pred_geom_a, target_geom_a, mask_a, ...)
    loss_geom_b = geometry_loss(pred_geom_b, target_geom_b, mask_b, ...)
```

### Priority 2: FIX THETA NORMALIZATION

Change line 179 from:
```python
loss_theta = (theta_diff / 360.0) * mask
```
to:
```python
loss_theta = (theta_diff / 180.0) * mask
```

### Priority 3: FIX HARDCODED PERIOD

Change line 114 from:
```python
normalized_error = abs_diff / 360.0
```
to:
```python
normalized_error = abs_diff / period
```

---

## Testing the Fix

After applying fixes, verify with:

```python
import torch

# Test case: Target in radians, prediction in degrees
target_theta_rad = torch.tensor([[1.5708]])  # π/2 ≈ 90°
target_theta_deg = torch.rad2deg(target_theta_rad)  # 90°
pred_theta = torch.tensor([[85.0]])  # 5° error

# Before fix: |85 - 1.5708| / 360 ≈ 0.232 (wrong!)
# After fix: |85 - 90| / 180 ≈ 0.028 (correct!)

# Test circular phi
target_phi_rad = torch.tensor([[0.1]])  # ~5.7°
target_phi_deg = torch.rad2deg(target_phi_rad)  # 5.7°
pred_phi = torch.tensor([[355.0]])  # Should be ~10.7° error (wraparound)

# Expected: shortest distance is 355° → 5.7° = -10.7° → wraps to 10.7°
# Loss: 10.7 / 360 ≈ 0.030
```

---

## Conclusion

**Circular loss logic**: ✅ Correct (phi is circular, theta is not)
**Implementation bugs**: ❌ Three bugs found (units, theta normalization, hardcoded period)

The conceptual understanding is correct, but the implementation has critical bugs that will cause training to fail or produce nonsensical results.

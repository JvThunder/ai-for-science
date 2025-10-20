# Validation Set Behavior in Multi-File Training

## Summary

You can choose between **two validation modes**:

1. **Fixed Validation (Default)**: Validation stays on the same molecules every epoch
2. **Cycling Validation**: Validation changes across epochs (each file = 1 epoch)

Set this in `config.yaml`:
```yaml
data:
  use_fixed_validation: true  # Fixed (default, recommended)
  # use_fixed_validation: false  # Cycling
```

## Mode 1: Fixed Validation (Default - Recommended)

**Config setting:**
```yaml
data:
  use_fixed_validation: true  # Default
```

### How It Works
- Validation dataset uses **only the first file** across all epochs
- Training dataset cycles through all files normally
- Same molecules are used for validation every epoch

**Example:**
- **Epoch 1**: Train on file 0 (90%), validate on file 0 (10%)
- **Epoch 2**: Train on file 1 (90%), validate on file 0 (10%) ← same validation molecules
- **Epoch 3**: Train on file 2 (90%), validate on file 0 (10%) ← same validation molecules
- ... and so on

### Advantages
✅ **Validation loss is directly comparable** across epochs
✅ **Better for early stopping** - can reliably detect overfitting
✅ **Better for model selection** - can choose best checkpoint based on val loss
✅ **Standard ML practice** - same validation set throughout training

### Trade-offs
⚠️ Validation set is smaller (only from first file)
⚠️ May not be representative if first file is unusual

---

## Mode 2: Cycling Validation

**Config setting:**
```yaml
data:
  use_fixed_validation: false
```

### How It Works
- Both training and validation datasets cycle through files
- Each file = 1 epoch for both train and validation
- Different molecules are used for validation each epoch

**Example:**
- **Epoch 1**: Train on file 0 (90%), validate on file 0 (10%)
- **Epoch 2**: Train on file 1 (90%), validate on file 1 (10%) ← different validation molecules
- **Epoch 3**: Train on file 2 (90%), validate on file 2 (10%) ← different validation molecules
- ... and so on

### Advantages
✅ **More diverse validation** - tests on different molecules across epochs
✅ **Larger effective validation set** - uses 10% of all files over time

### Trade-offs
⚠️ **Validation loss NOT directly comparable** across epochs (different molecules, different difficulty)
⚠️ **Early stopping may be unreliable** - loss can vary due to dataset difficulty, not model performance
⚠️ **Harder to select best model** - can't simply compare validation losses

---

## Deterministic Splits

Regardless of mode, each file has a **deterministic** train/validation split:
- Split seed: `split_seed + hash(file_path)`
- Default `split_seed = 42` (set in config as `random_seed`)
- The hash ensures each file gets a different but reproducible split
- **Same molecules are ALWAYS in the validation set for a given file**

This means if you run training twice with the same config, you'll get the exact same train/val splits.

## Code Implementation

### Dataset Split Logic
From `utils/dataset.py`, line 391-422:

```python
def _apply_train_val_split(self, all_molecule_indices):
    # Use a deterministic random seed for reproducible splits
    rng = random.Random(self.split_seed + hash(self.current_file))

    # Shuffle indices with the deterministic seed
    shuffled_indices = list(all_molecule_indices)
    rng.shuffle(shuffled_indices)

    # Calculate split point
    n_total = len(shuffled_indices)
    n_val = int(n_total * self.val_split_ratio)
    n_train = n_total - n_val

    # Return appropriate split
    if self.split == 'train':
        return shuffled_indices[:n_train]
    elif self.split == 'val':
        return shuffled_indices[n_train:]
```

### Training Loop File Cycling
From `utils/training.py`, line 356-360:

```python
if file_idx > 0:  # Skip first file (already loaded)
    print(f"\nSwitching to file {file_idx + 1}/{total_files}...")
    train_dataset.next_file()
    # Also switch validation dataset to the same file
    val_dataset.next_file()
```

## Which Mode Should I Use?

### Use **Fixed Validation** (default) if:
- ✅ You want reliable early stopping
- ✅ You want to compare validation loss across epochs
- ✅ You want to select the best model checkpoint
- ✅ You're following standard ML practices
- ✅ **Recommended for most use cases**

### Use **Cycling Validation** if:
- ✅ You want to see how the model performs on diverse molecules
- ✅ You want a larger effective validation set
- ✅ You care less about precise model selection
- ✅ You're doing exploratory analysis

## Shared Advantages (Both Modes)
1. **Memory efficient**: Never loads all validation data at once
2. **Scalable**: Works with TB-scale datasets
3. **Reproducible**: Same split for each file every time you run
4. **Streaming**: No RAM caching required

### 📊 Tracking Validation Sets
Use `export_validation_ids.py` to track which molecules are used for validation in each epoch:

```bash
python export_validation_ids.py --config config.yaml --output logs/validation_ids.json
```

This creates a JSON file with:
- Validation molecule indices for each file
- Train/val split statistics
- File-to-epoch mapping

## Alternative Approaches (Not Currently Implemented)

If you need a **fixed validation set** across all epochs:

### Option 1: Dedicated Validation File
Create a separate CSV file with validation molecules and don't cycle it:

```python
# In training loop, DON'T call val_dataset.next_file()
train_dataset.next_file()  # Cycle training data
# val_dataset stays on same file
```

### Option 2: Global Random Split
Modify dataset to split molecules randomly across all files at initialization (requires reading all file metadata upfront).

### Option 3: Pre-split Data
Manually create separate `train/` and `val/` folders with pre-split CSV files.

## Recommendations

### For Fixed Validation (Default)
1. ✅ Use standard early stopping (patience = 5 is fine)
2. ✅ Compare validation losses directly across epochs
3. ✅ Select best model by lowest validation loss
4. ✅ Monitor for overfitting normally

### For Cycling Validation
1. ⚠️ **Monitor trends** rather than absolute validation loss values
2. ⚠️ **Use longer patience** for early stopping (e.g., 10-20 epochs)
3. ⚠️ **Export validation IDs** to understand which molecules are being evaluated
4. ⚠️ **Consider cycle-level metrics** (average validation loss per full cycle through all files)

## Example Validation Tracking

```python
# After training
import json

with open('logs/validation_ids.json', 'r') as f:
    val_info = json.load(f)

# See which molecules were used for validation in epoch 1
epoch_1_info = val_info['files'][0]
print(f"Epoch 1 validation molecules: {epoch_1_info['val_indices']}")
print(f"File: {epoch_1_info['file_name']}")
print(f"Count: {epoch_1_info['val_molecules']}")
```

## Questions?

If you need a fixed validation set across epochs, please let me know and I can modify the training loop to keep the validation dataset on a single file while cycling the training data.

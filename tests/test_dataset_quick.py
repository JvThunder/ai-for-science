"""
Quick test for MultiFileStreamingDataset with real data.
"""

import sys
sys.path.append('utils')

from dataset import MultiFileStreamingDataset, get_element_vocab
import glob

print("Testing MultiFileStreamingDataset...")
print("=" * 60)

# Get file paths
file_paths = sorted(glob.glob("../Compound_*.csv"))
if not file_paths:
    print("No CSV files found!")
    sys.exit(1)

# Step 1: Load element vocabulary
print("\n1. Loading element vocabulary...")
element_to_idx = get_element_vocab(
    txt_path="../elements.txt",
    file_paths=file_paths,
    max_molecules=1000  # Quick test with 1000 molecules
)

print(f"\nElement vocabulary size: {len(element_to_idx)}")
print(f"Sample elements: {list(element_to_idx.keys())[:10]}")

# Step 2: Create dataset
print("\n2. Creating streaming dataset...")
dataset = MultiFileStreamingDataset(
    file_paths=file_paths,
    element_to_idx=element_to_idx,
    current_file_idx=0,
    max_samples=100,  # Test with 100 samples
    training=True
)

print(f"\nDataset size: {len(dataset)} bonds")
print(f"Current file: {dataset.get_current_file_info()}")

# Step 3: Load a sample
print("\n3. Loading sample from disk...")
try:
    sample = dataset[0]
    print(f"Sample loaded successfully!")
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge index shape: {sample.edge_index.shape}")
    print(f"  Position shape: {sample.pos.shape}")
    print(f"  Target distance: {sample.target_distance.item():.4f} Angstrom")
    print(f"  Target geometry A shape: {sample.target_geometry_a.shape}")  # Should be (max_bonds, 3)
    print(f"  Target geometry B shape: {sample.target_geometry_b.shape}")  # Should be (max_bonds, 3)
    print(f"  Bond mask A: {sample.bond_mask_a}")
    print(f"  Bond mask B: {sample.bond_mask_b}")

    # Check geometry values
    print(f"\n  Sample geometry A (first 3 neighbors):")
    for i in range(min(3, sample.target_geometry_a.shape[0])):
        if sample.bond_mask_a[i] > 0:
            r, theta, phi = sample.target_geometry_a[i]
            print(f"    Neighbor {i}: r={r:.4f} A, theta={theta:.4f} rad, phi={phi:.4f} rad")

    print("\n[OK] Basic test passed!")

except Exception as e:
    print(f"\n[ERROR] Error loading sample: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test file cycling
print("\n4. Testing file cycling...")
current_file = dataset.current_file
print(f"  Before: {current_file}")

if len(dataset.file_paths) > 1:
    new_file = dataset.next_file()
    print(f"  After: {new_file}")
    print(f"  New dataset size: {len(dataset)} bonds")
else:
    print(f"  Only one file available, cycling back to same file")
    dataset.next_file()

print("\n" + "=" * 60)
print("Quick test complete!")

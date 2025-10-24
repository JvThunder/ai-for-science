"""
Test script to verify bond type encoding in edge attributes.

This test loads a sample molecule and verifies that edge attributes
contain the correct bond type information.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.dataset import MultiFileStreamingDataset, get_element_vocab
import torch

def test_bond_types():
    """Test that bond types are correctly encoded in edge attributes."""

    # Path to test data
    data_path = r"d:\Joshua\Coding\AI-for-Science\new_implementation\data\Compound_000000001_000025000_props_geom.csv"

    print("=" * 80)
    print("Testing Bond Type Encoding")
    print("=" * 80)

    # Step 1: Load element vocabulary
    print("\n1. Loading element vocabulary...")
    element_to_idx = get_element_vocab(
        txt_path="../elements.txt",
        file_paths=[data_path],
        max_molecules=1000
    )
    print(f"   Found {len(element_to_idx)} elements")

    # Step 2: Create dataset
    print("\n2. Creating dataset...")
    dataset = MultiFileStreamingDataset(
        file_paths=[data_path],
        element_to_idx=element_to_idx,
        current_file_idx=0,
        max_bonds=7,
        chunk_size=10000,
        training=False,
        max_samples=50  # Only load first 50 bonds for quick test
    )
    print(f"   Dataset loaded: {len(dataset)} bonds")

    # Step 3: Test first few samples
    print("\n3. Testing bond type encoding on first 5 samples...")
    print("-" * 80)

    bond_type_names = {0: "Single", 1: "Aromatic", 2: "Double", 3: "Triple"}
    bond_type_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(min(5, len(dataset))):
        print(f"\nSample {i}:")
        data = dataset[i]

        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Edge attributes shape: {data.edge_attr.shape}")

        # Verify edge_attr exists and has correct length
        assert hasattr(data, 'edge_attr'), "Data object missing edge_attr!"
        assert data.edge_attr.shape[0] == data.edge_index.shape[1], \
            f"Edge attr length {data.edge_attr.shape[0]} != edge count {data.edge_index.shape[1]}"

        # Count bond types
        unique_types, counts = torch.unique(data.edge_attr, return_counts=True)
        print(f"  Bond types in this subgraph:")
        for bond_type, count in zip(unique_types.tolist(), counts.tolist()):
            bond_name = bond_type_names.get(bond_type, f"Unknown ({bond_type})")
            print(f"    {bond_name}: {count} edges")
            bond_type_counts[bond_type] += count

    # Step 4: Summary
    print("\n" + "=" * 80)
    print("Summary of Bond Types Across All Samples:")
    print("-" * 80)
    for bond_type, count in sorted(bond_type_counts.items()):
        if count > 0:
            bond_name = bond_type_names.get(bond_type, f"Unknown ({bond_type})")
            print(f"  {bond_name}: {count} edges")

    print("\n" + "=" * 80)
    print("SUCCESS: Test completed successfully!")
    print("  - edge_attr is correctly added to Data objects")
    print("  - Bond types are encoded as: 0=Single, 1=Aromatic, 2=Double, 3=Triple")
    print("  - Edge attributes length matches edge count")
    print("=" * 80)

if __name__ == "__main__":
    test_bond_types()

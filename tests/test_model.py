"""
Unit tests for model architecture.

Tests the LocalGeometryGCN model including:
- Model initialization
- Forward pass with correct output shapes
- Output activation ranges
- Batch processing
- Embedding vs. one-hot features
"""

import torch
from torch_geometric.data import Data, Batch
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model import LocalGeometryGCN, count_parameters, initialize_weights


def create_sample_graph(num_nodes=10, num_edges=15, use_embeddings=True, vocab_size=12):
    """
    Create a sample molecular graph for testing.

    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges (bidirectional, so actual edges = num_edges * 2)
        use_embeddings: If True, use element indices; else use one-hot features
        vocab_size: Size of element vocabulary

    Returns:
        data: PyG Data object
    """
    # Node features
    if use_embeddings:
        # Element indices for embedding layer
        x = torch.randint(0, vocab_size, (num_nodes,), dtype=torch.long)
    else:
        # One-hot encoding (9 common elements)
        x = torch.zeros((num_nodes, 9))
        for i in range(num_nodes):
            x[i, i % 9] = 1  # Assign to one of 9 classes

    # Random edges (bidirectional)
    edge_index = torch.randint(0, num_nodes, (2, num_edges * 2))

    # Random 3D positions (for reference, not used by model)
    pos = torch.randn(num_nodes, 3)

    # Target nodes for prediction
    node_a = torch.tensor([0])  # Predict geometry around node 0
    node_b = torch.tensor([1])  # and node 1

    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        node_a=node_a,
        node_b=node_b
    )

    return data


# Test 1: Model initialization
def test_model_initialization():
    """Test that model initializes correctly with different configurations."""

    # Test with embeddings
    vocab_size = 12
    max_bonds = 7
    hidden_dim = 64
    embedding_dim = 32

    model = LocalGeometryGCN(
        num_node_features=9,  # Not used with embeddings
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    )

    assert model.max_bonds == max_bonds
    assert model.use_embeddings == True
    assert model.hidden_dim == hidden_dim

    # Check that embedding layer exists
    assert hasattr(model, 'element_embedding')
    assert model.element_embedding.num_embeddings == vocab_size
    assert model.element_embedding.embedding_dim == embedding_dim

    # Check GCN layers exist
    assert hasattr(model, 'node_encoder')
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2')

    # Check prediction heads exist
    assert hasattr(model, 'distance_head')
    assert hasattr(model, 'geometry_head')

    # Test without embeddings (fallback to one-hot)
    model_no_embed = LocalGeometryGCN(
        num_node_features=9,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=None  # No embeddings
    )

    assert model_no_embed.use_embeddings == False
    assert not hasattr(model_no_embed, 'element_embedding') or model_no_embed.element_embedding is None

    print("✓ Model initialization test passed")


# Test 2: Forward pass shapes
def test_forward_pass_shapes():
    """Test that forward pass produces correct output shapes."""

    vocab_size = 12
    max_bonds = 7
    hidden_dim = 64
    num_nodes = 10

    model = LocalGeometryGCN(
        num_node_features=9,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=vocab_size,
        embedding_dim=32
    )

    # Single graph
    data = create_sample_graph(num_nodes=num_nodes, use_embeddings=True, vocab_size=vocab_size)

    model.eval()
    with torch.no_grad():
        distance, geometry_a, geometry_b = model(data)

    # Check shapes
    assert distance.shape == (1, 1), f"Expected distance shape (1, 1), got {distance.shape}"
    assert geometry_a.shape == (1, max_bonds, 3), f"Expected geometry_a shape (1, {max_bonds}, 3), got {geometry_a.shape}"
    assert geometry_b.shape == (1, max_bonds, 3), f"Expected geometry_b shape (1, {max_bonds}, 3), got {geometry_b.shape}"

    # Check that outputs are tensors
    assert isinstance(distance, torch.Tensor)
    assert isinstance(geometry_a, torch.Tensor)
    assert isinstance(geometry_b, torch.Tensor)

    print("✓ Forward pass shapes test passed")


# Test 3: Output activation ranges
def test_output_activation_ranges():
    """Test that output activations produce values in correct ranges."""

    vocab_size = 12
    max_bonds = 7
    hidden_dim = 64
    num_nodes = 10

    model = LocalGeometryGCN(
        num_node_features=9,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=vocab_size,
        embedding_dim=32
    )

    # Create sample data
    data = create_sample_graph(num_nodes=num_nodes, use_embeddings=True, vocab_size=vocab_size)

    model.eval()
    with torch.no_grad():
        distance, geometry_a, geometry_b = model(data)

    # Check distance is positive (softplus activation)
    assert torch.all(distance >= 0), "Distance should be non-negative (softplus activation)"

    # Check geometry components
    # geometry shape: (batch_size, max_bonds, 3) where [:, :, 0] = r, [:, :, 1] = theta, [:, :, 2] = phi

    # r (distance): should be positive
    r_a = geometry_a[:, :, 0]
    r_b = geometry_b[:, :, 0]
    assert torch.all(r_a >= 0), "r_a should be non-negative (softplus activation)"
    assert torch.all(r_b >= 0), "r_b should be non-negative (softplus activation)"

    # θ (theta): should be in [0, π]
    theta_a = geometry_a[:, :, 1]
    theta_b = geometry_b[:, :, 1]
    assert torch.all(theta_a >= 0) and torch.all(theta_a <= math.pi + 0.01), \
        f"theta_a should be in [0, π], got range [{theta_a.min():.3f}, {theta_a.max():.3f}]"
    assert torch.all(theta_b >= 0) and torch.all(theta_b <= math.pi + 0.01), \
        f"theta_b should be in [0, π], got range [{theta_b.min():.3f}, {theta_b.max():.3f}]"

    # φ (phi): should be in [0, 2π]
    phi_a = geometry_a[:, :, 2]
    phi_b = geometry_b[:, :, 2]
    assert torch.all(phi_a >= 0) and torch.all(phi_a <= 2 * math.pi + 0.01), \
        f"phi_a should be in [0, 2π], got range [{phi_a.min():.3f}, {phi_a.max():.3f}]"
    assert torch.all(phi_b >= 0) and torch.all(phi_b <= 2 * math.pi + 0.01), \
        f"phi_b should be in [0, 2π], got range [{phi_b.min():.3f}, {phi_b.max():.3f}]"

    print("✓ Output activation ranges test passed")
    print(f"  Distance range: [{distance.min():.3f}, {distance.max():.3f}]")
    print(f"  r range: [{r_a.min():.3f}, {r_a.max():.3f}]")
    print(f"  θ range: [{theta_a.min():.3f}, {theta_a.max():.3f}] (should be [0, {math.pi:.3f}])")
    print(f"  φ range: [{phi_a.min():.3f}, {phi_a.max():.3f}] (should be [0, {2*math.pi:.3f}])")


# Test 4: Batch processing
def test_batch_processing():
    """Test that model correctly handles batched graphs."""

    vocab_size = 12
    max_bonds = 7
    hidden_dim = 64
    batch_size = 4
    num_nodes = 10

    model = LocalGeometryGCN(
        num_node_features=9,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=vocab_size,
        embedding_dim=32
    )

    # Create batch of graphs
    graph_list = [
        create_sample_graph(num_nodes=num_nodes, use_embeddings=True, vocab_size=vocab_size)
        for _ in range(batch_size)
    ]

    batch = Batch.from_data_list(graph_list)

    model.eval()
    with torch.no_grad():
        distance, geometry_a, geometry_b = model(batch)

    # Check batch shapes
    assert distance.shape == (batch_size, 1), f"Expected distance shape ({batch_size}, 1), got {distance.shape}"
    assert geometry_a.shape == (batch_size, max_bonds, 3), \
        f"Expected geometry_a shape ({batch_size}, {max_bonds}, 3), got {geometry_a.shape}"
    assert geometry_b.shape == (batch_size, max_bonds, 3), \
        f"Expected geometry_b shape ({batch_size}, {max_bonds}, 3), got {geometry_b.shape}"

    # Check that all predictions are valid (no NaN or Inf)
    assert torch.all(torch.isfinite(distance)), "Distance contains NaN or Inf"
    assert torch.all(torch.isfinite(geometry_a)), "geometry_a contains NaN or Inf"
    assert torch.all(torch.isfinite(geometry_b)), "geometry_b contains NaN or Inf"

    print("✓ Batch processing test passed")
    print(f"  Processed batch of {batch_size} graphs successfully")


# Test 5: Parameter count
def test_parameter_count():
    """Test that parameter counting utility works correctly."""

    vocab_size = 12
    max_bonds = 7
    hidden_dim = 64
    embedding_dim = 32

    model = LocalGeometryGCN(
        num_node_features=9,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    )

    # Count parameters
    num_params = count_parameters(model)

    # Check that count is reasonable (should be thousands to tens of thousands)
    assert num_params > 1000, f"Model seems too small: {num_params} parameters"
    assert num_params < 1_000_000, f"Model seems too large: {num_params} parameters"

    print(f"✓ Parameter count test passed: {num_params:,} trainable parameters")


# Test 6: Weight initialization
def test_weight_initialization():
    """Test that weight initialization function works correctly."""

    vocab_size = 12
    max_bonds = 7
    hidden_dim = 64

    model = LocalGeometryGCN(
        num_node_features=9,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=vocab_size,
        embedding_dim=32
    )

    # Initialize weights
    initialize_weights(model)

    # Check that weights are not all zero
    for name, param in model.named_parameters():
        if 'weight' in name:
            assert not torch.all(param == 0), f"Weights in {name} are all zero after initialization"

    print("✓ Weight initialization test passed")


# Test 7: Model without embeddings (one-hot fallback)
def test_model_without_embeddings():
    """Test model with one-hot features instead of embeddings."""

    num_features = 9
    max_bonds = 7
    hidden_dim = 64
    num_nodes = 10

    model = LocalGeometryGCN(
        num_node_features=num_features,
        hidden_dim=hidden_dim,
        max_bonds=max_bonds,
        vocab_size=None,  # No embeddings
        embedding_dim=32
    )

    # Create sample data with one-hot features
    data = create_sample_graph(num_nodes=num_nodes, use_embeddings=False)

    model.eval()
    with torch.no_grad():
        distance, geometry_a, geometry_b = model(data)

    # Check shapes
    assert distance.shape == (1, 1)
    assert geometry_a.shape == (1, max_bonds, 3)
    assert geometry_b.shape == (1, max_bonds, 3)

    # Check that outputs are valid
    assert torch.all(torch.isfinite(distance))
    assert torch.all(torch.isfinite(geometry_a))
    assert torch.all(torch.isfinite(geometry_b))

    print("✓ Model without embeddings test passed")


if __name__ == "__main__":
    print("Running model tests...\n")

    test_model_initialization()
    test_forward_pass_shapes()
    test_output_activation_ranges()
    test_batch_processing()
    test_parameter_count()
    test_weight_initialization()
    test_model_without_embeddings()

    print("\n" + "="*60)
    print("All model tests passed! ✓")
    print("="*60)

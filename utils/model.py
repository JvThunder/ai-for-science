"""
Model architecture for stereochemistry-aware 3D reconstruction.

This module implements the LocalGeometryGCN model that predicts:
- Bond distances (A-B distance)
- Local spherical angular geometry (θ, Δφ) for neighbors of both atoms
  where Δφ are relative azimuthal angle differences (first neighbor at Δφ=0)

The model uses element embeddings and GAT (Graph Attention Networks) layers to
encode local molecular structure, then predicts geometry in a rotationally-invariant
representation.

Key Features:
- Element embeddings: Learn representations for different atom types
- Bond type embeddings: Distinguish single, aromatic, double, and triple bonds
- GAT layers: Attention-based message passing with edge features
- Dual prediction: Geometry for both bond endpoints (A and B)
- Bond type integration: Edge features aggregated and concatenated with node features

Bond Type Support:
    The model incorporates bond type information (0=single, 1=aromatic, 2=double, 3=triple)
    from the dataset's edge_attr field. For each GNN layer:
    1. Bond types are embedded into bond_embedding_dim dimensions
    2. Edge embeddings are projected to match node feature dimensions
    3. Edge features are aggregated to target nodes (sum of incoming edges)
    4. Concatenated with node features: [node_features, aggregated_edge_features]
    5. Passed through GAT layer for attention-based message passing

    This allows the model to be chemically aware - e.g., distinguishing that double
    bonds are shorter and more rigid than single bonds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import math


class LocalGeometryGCN(nn.Module):
    """
    Graph Convolutional Network for predicting local molecular geometry.

    Predicts:
    - Distance between target atoms A and B
    - Local spherical geometry (θ, Δφ) for neighbors of A
    - Local spherical geometry (θ, Δφ) for neighbors of B
      where Δφ are relative azimuthal differences (first neighbor at Δφ=0)

    Args:
        num_node_features: Number of input features (not used with embeddings)
        hidden_dim: Hidden dimension for GCN layers (default: 64)
        max_bonds: Maximum number of bonds per atom (default: 7)
        vocab_size: Size of element vocabulary (enables embeddings)
        embedding_dim: Dimension of element embeddings (default: 32)
        depth: Number of GCN layers (graph distance to propagate, default: 1)
        use_bond_types: Whether to use bond type edge features (default: True)
        num_bond_types: Number of bond types (default: 4 - single, aromatic, double, triple)
        bond_embedding_dim: Dimension of bond type embeddings (default: 8)

    Architecture:
        1. Element embedding layer (if vocab_size provided)
        2. Bond type embedding layer (if use_bond_types enabled)
        3. GAT encoder with edge features: configurable depth
        4. Batch normalization and dropout for regularization
        5. Distance head: predicts A-B distance
        6. Geometry head: predicts (θ, Δφ) for each neighbor
           - θ: polar angle [0, 180] degrees via sigmoid
           - Δφ: relative azimuthal difference [0, 360] degrees via sigmoid
                 (first neighbor Δφ=0 by definition)
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        max_bonds: int = 7,
        vocab_size: int = None,
        embedding_dim: int = 32,
        depth: int = 1,
        use_bond_types: bool = True,
        num_bond_types: int = 4,
        bond_embedding_dim: int = 8
    ):
        super().__init__()
        self.max_bonds = max_bonds
        self.use_embeddings = vocab_size is not None
        self.use_bond_types = use_bond_types
        self.hidden_dim = hidden_dim
        self.depth = max(1, depth)  # Ensure at least 1 layer

        # Element embeddings (preferred over one-hot encoding)
        if self.use_embeddings:
            self.element_embedding = nn.Embedding(vocab_size, embedding_dim)
            encoder_input_dim = embedding_dim
        else:
            # Fallback to one-hot features if no vocab provided
            encoder_input_dim = num_node_features

        # Bond type embeddings (for edge features)
        if self.use_bond_types:
            self.bond_embedding = nn.Embedding(num_bond_types, bond_embedding_dim)
            edge_feature_dim = bond_embedding_dim
        else:
            edge_feature_dim = 0

        # GAT layers for subgraph encoding (dynamically created based on depth)
        # Edge features are passed directly through GATConv's edge_dim parameter so
        # each attention score conditions on that specific edge's bond type.
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        gat_edge_dim = edge_feature_dim if self.use_bond_types else None

        # First layer (input -> hidden)
        self.gnn_layers.append(
            GATConv(encoder_input_dim, hidden_dim, heads=4, concat=False, edge_dim=gat_edge_dim)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Additional layers (hidden -> hidden)
        for _ in range(self.depth - 1):
            self.gnn_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=gat_edge_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Residual projection for first layer (if input dim != hidden dim)
        self.input_proj = nn.Linear(encoder_input_dim, hidden_dim) if encoder_input_dim != hidden_dim else nn.Identity()

        # Dropout for regularization (reduced from 0.5 — residuals already regularize)
        self.dropout = nn.Dropout(0.3)

        # Distance prediction head (A-B distance)
        # Input: concatenated embeddings of A and B
        # Output: single distance value (positive)
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive distances
        )

        # Geometry prediction head (θ, Δφ) for each bond
        # Input: single node embedding
        # Output: max_bonds * 2 values (θ, Δφ for each neighbor)
        #   θ: polar angle [0, 180°]
        #   Δφ: relative azimuthal difference [0, 360°]
        #        First neighbor: Δφ=0 (by definition)
        #        Other neighbors: angle difference from previous neighbor
        #
        # Deeper head (3 layers) to better decode node embeddings into geometry.
        # Raw values are clamped in forward pass (avoids sigmoid saturation near extremes).
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, max_bonds * 2)  # Output (θ, Δφ) only, no r
        )

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: PyG Data object containing:
                - x: Node features (element indices if using embeddings)
                - edge_index: Edge connectivity
                - edge_attr: Bond type indices (if use_bond_types enabled)
                - node_a, node_b: Indices of target atoms
                - batch: Batch indices (optional, for batched graphs)

        Returns:
            distance: (batch_size, 1) predicted A-B distances
            geometry_a: (batch_size, max_bonds, 2) predicted (θ, Δφ) for A's neighbors
            geometry_b: (batch_size, max_bonds, 2) predicted (θ, Δφ) for B's neighbors

        Output activations:
            - distance: softplus (positive values)
            - θ: sigmoid * 180 → (0, 180) degrees
            - Δφ: sigmoid * 360 → (0, 360) degrees
        """
        x, edge_index = data.x, data.edge_index
        node_a, node_b = data.node_a, data.node_b
        batch = getattr(data, 'batch', None)
        edge_attr = getattr(data, 'edge_attr', None)

        # Apply element embeddings if using element indices
        if self.use_embeddings:
            x = self.element_embedding(x)

        # Apply bond type embeddings if using bond types
        if self.use_bond_types and edge_attr is not None:
            edge_features = self.bond_embedding(edge_attr)
        else:
            edge_features = None

        # GNN encoding with batch normalization, dropout, and residual connections.
        # Each GATConv receives bond embeddings as edge_attr so attention scores
        # for edge (i→j) condition directly on that edge's bond type.
        # Residual connections prevent over-smoothing across layers.
        for i, (gnn_layer, bn_layer) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            if self.use_bond_types and edge_features is not None:
                h = gnn_layer(x, edge_index, edge_attr=edge_features)
            else:
                h = gnn_layer(x, edge_index)

            h = F.relu(h)
            if h.size(0) > 1:  # Skip batch norm if batch size == 1
                h = bn_layer(h)
            h = self.dropout(h)

            # Residual: project input on first layer, direct add thereafter
            residual = self.input_proj(x) if i == 0 else x
            x = h + residual

        # Extract embeddings for target nodes A and B
        # Handle batched vs. single graph indexing
        if batch is not None:
            # Batched data: adjust node indices by batch offsets
            batch_size = batch.max().item() + 1

            # Calculate cumulative node counts per batch
            batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=batch.device)
            for i in range(batch_size):
                batch_offsets[i + 1] = batch_offsets[i] + (batch == i).sum()

            # Adjust node indices
            batch_indices = torch.arange(batch_size, device=batch.device)
            adjusted_node_a = node_a + batch_offsets[batch_indices]
            adjusted_node_b = node_b + batch_offsets[batch_indices]

            emb_a = x[adjusted_node_a]
            emb_b = x[adjusted_node_b]
        else:
            # Single graph: add batch dimension
            emb_a = x[node_a].unsqueeze(0)
            emb_b = x[node_b].unsqueeze(0)

        # Predict A-B distance from combined embedding
        combined = torch.cat([emb_a, emb_b], dim=-1)
        distance = self.distance_head(combined)

        # Predict geometry for A's neighbors
        geom_a_raw = self.geometry_head(emb_a)  # (batch_size, max_bonds * 2)
        geom_a_raw = geom_a_raw.view(-1, self.max_bonds, 2)  # (batch_size, max_bonds, 2)

        # Sigmoid scaling — gradient flows everywhere (no dead zones unlike clamp)
        theta_a = torch.sigmoid(geom_a_raw[:, :, 0:1]) * 180.0      # (0, 180) degrees
        delta_phi_a = torch.sigmoid(geom_a_raw[:, :, 1:2]) * 360.0  # (0, 360) degrees

        geometry_a = torch.cat([theta_a, delta_phi_a], dim=-1)  # (batch_size, max_bonds, 2)

        # Predict geometry for B's neighbors
        geom_b_raw = self.geometry_head(emb_b)  # (batch_size, max_bonds * 2)
        geom_b_raw = geom_b_raw.view(-1, self.max_bonds, 2)  # (batch_size, max_bonds, 2)

        # Sigmoid scaling
        theta_b = torch.sigmoid(geom_b_raw[:, :, 0:1]) * 180.0      # (0, 180) degrees
        delta_phi_b = torch.sigmoid(geom_b_raw[:, :, 1:2]) * 360.0  # (0, 360) degrees

        geometry_b = torch.cat([theta_b, delta_phi_b], dim=-1)  # (batch_size, max_bonds, 2)

        return distance, geometry_a, geometry_b


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        num_params: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Xavier/Glorot initialization.

    Applies Xavier uniform initialization to linear layers and
    constant initialization to batch normalization layers.

    Args:
        model: PyTorch model to initialize
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)

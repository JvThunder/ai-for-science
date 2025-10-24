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
        # GAT naturally handles edge features through attention mechanism
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Edge feature projection layers (project bond embeddings to match node features)
        self.edge_projections = nn.ModuleList()

        # First layer (input -> hidden)
        if self.use_bond_types:
            # Project edge features to node feature space for concatenation
            self.edge_projections.append(nn.Linear(edge_feature_dim, encoder_input_dim))
            # GAT with concatenated edge features
            self.gnn_layers.append(GATConv(encoder_input_dim * 2, hidden_dim, heads=4, concat=False, edge_dim=None))
        else:
            # Standard GAT without edge features
            self.gnn_layers.append(GATConv(encoder_input_dim, hidden_dim, heads=4, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Additional layers (hidden -> hidden)
        for _ in range(self.depth - 1):
            if self.use_bond_types:
                self.edge_projections.append(nn.Linear(edge_feature_dim, hidden_dim))
                self.gnn_layers.append(GATConv(hidden_dim * 2, hidden_dim, heads=4, concat=False, edge_dim=None))
            else:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Distance prediction head (A-B distance)
        # Input: concatenated embeddings of A and B
        # Output: single distance value (positive)
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        # NOTE: We output raw values and apply activations in forward pass
        # to allow separate handling of each geometry component
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
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
            - θ: sigmoid * 180 (range [0, 180] degrees)
            - Δφ: sigmoid * 360 (range [0, 360] degrees for angular differences)
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

        # GNN encoding with batch normalization and dropout (loop through all layers)
        for i, (gnn_layer, bn_layer) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            if self.use_bond_types and edge_features is not None:
                # Project edge features
                edge_proj = self.edge_projections[i](edge_features)

                # Aggregate edge features to nodes (sum of incoming edge features)
                # edge_index[0] is source, edge_index[1] is target
                num_nodes = x.size(0)
                edge_aggregated = torch.zeros(num_nodes, edge_proj.size(1), device=x.device)
                edge_aggregated.index_add_(0, edge_index[1], edge_proj)

                # Concatenate node features with aggregated edge features
                x_with_edges = torch.cat([x, edge_aggregated], dim=-1)
                x = gnn_layer(x_with_edges, edge_index)
            else:
                x = gnn_layer(x, edge_index)

            x = F.relu(x)
            if x.size(0) > 1:  # Skip batch norm if batch size == 1
                x = bn_layer(x)
            x = self.dropout(x)

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

        # Apply activations for (θ, Δφ)
        theta_a = torch.sigmoid(geom_a_raw[:, :, 0:1]) * 180.0  # (batch_size, max_bonds, 1) - [0, 180] degrees
        delta_phi_a = torch.sigmoid(geom_a_raw[:, :, 1:2]) * 360.0  # (batch_size, max_bonds, 1) - [0, 360] degrees

        geometry_a = torch.cat([theta_a, delta_phi_a], dim=-1)  # (batch_size, max_bonds, 2)

        # Predict geometry for B's neighbors
        geom_b_raw = self.geometry_head(emb_b)  # (batch_size, max_bonds * 2)
        geom_b_raw = geom_b_raw.view(-1, self.max_bonds, 2)  # (batch_size, max_bonds, 2)

        # Apply activations for (θ, Δφ)
        theta_b = torch.sigmoid(geom_b_raw[:, :, 0:1]) * 180.0  # (batch_size, max_bonds, 1) - [0, 180] degrees
        delta_phi_b = torch.sigmoid(geom_b_raw[:, :, 1:2]) * 360.0  # (batch_size, max_bonds, 1) - [0, 360] degrees

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

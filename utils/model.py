"""
Model architecture for stereochemistry-aware 3D reconstruction.

This module implements the LocalGeometryGCN model that predicts:
- Bond distances (A-B distance)
- Local cylindrical geometry (r, θ, φ) for neighbors of both atoms

The model uses element embeddings and GCN layers to encode local molecular
structure, then predicts geometry in a rotationally-invariant representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math


class LocalGeometryGCN(nn.Module):
    """
    Graph Convolutional Network for predicting local molecular geometry.

    Predicts:
    - Distance between target atoms A and B
    - Local cylindrical geometry (r, θ, φ) for neighbors of A
    - Local cylindrical geometry (r, θ, φ) for neighbors of B

    Args:
        num_node_features: Number of input features (not used with embeddings)
        hidden_dim: Hidden dimension for GCN layers (default: 64)
        max_bonds: Maximum number of bonds per atom (default: 7)
        vocab_size: Size of element vocabulary (enables embeddings)
        embedding_dim: Dimension of element embeddings (default: 32)
        depth: Number of GCN layers (graph distance to propagate, default: 1)

    Architecture:
        1. Element embedding layer (if vocab_size provided)
        2. GCN encoder: configurable depth (number of GCN layers)
        3. Batch normalization and dropout for regularization
        4. Distance head: predicts A-B distance
        5. Geometry head: predicts (r, θ, φ) for each neighbor
           - r: distance (Angstroms) via softplus activation
           - θ: polar angle [0, 180] degrees via sigmoid
           - φ: azimuthal angle [0, 360] degrees via sigmoid
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        max_bonds: int = 7,
        vocab_size: int = None,
        embedding_dim: int = 32,
        depth: int = 1
    ):
        super().__init__()
        self.max_bonds = max_bonds
        self.use_embeddings = vocab_size is not None
        self.hidden_dim = hidden_dim
        self.depth = max(1, depth)  # Ensure at least 1 layer

        # Element embeddings (preferred over one-hot encoding)
        if self.use_embeddings:
            self.element_embedding = nn.Embedding(vocab_size, embedding_dim)
            encoder_input_dim = embedding_dim
        else:
            # Fallback to one-hot features if no vocab provided
            encoder_input_dim = num_node_features

        # GCN layers for subgraph encoding (dynamically created based on depth)
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer (input -> hidden)
        self.gcn_layers.append(GCNConv(encoder_input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Additional layers (hidden -> hidden)
        for _ in range(self.depth - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
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

        # Geometry prediction head (r, θ, φ) for each bond
        # Input: single node embedding
        # Output: max_bonds * 3 values (r, θ, φ for each neighbor)
        #
        # NOTE: We output raw values and apply activations in forward pass
        # to allow separate handling of each geometry component
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, max_bonds * 3)  # Changed from max_bonds * 2
        )

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: PyG Data object containing:
                - x: Node features (element indices if using embeddings)
                - edge_index: Edge connectivity
                - node_a, node_b: Indices of target atoms
                - batch: Batch indices (optional, for batched graphs)

        Returns:
            distance: (batch_size, 1) predicted A-B distances
            geometry_a: (batch_size, max_bonds, 3) predicted (r, θ, φ) for A's neighbors
            geometry_b: (batch_size, max_bonds, 3) predicted (r, θ, φ) for B's neighbors

        Output activations:
            - distance: softplus (positive values)
            - r: softplus (positive distances in Angstroms)
            - θ: sigmoid * 180 (range [0, 180] degrees)
            - φ: sigmoid * 360 (range [0, 360] degrees)
        """
        x, edge_index = data.x, data.edge_index
        node_a, node_b = data.node_a, data.node_b
        batch = getattr(data, 'batch', None)

        # Apply element embeddings if using element indices
        if self.use_embeddings:
            x = self.element_embedding(x)

        # GCN encoding with batch normalization and dropout (loop through all layers)
        for i, (gcn_layer, bn_layer) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn_layer(x, edge_index)
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
        geom_a_raw = self.geometry_head(emb_a)  # (batch_size, max_bonds * 3)
        geom_a_raw = geom_a_raw.view(-1, self.max_bonds, 3)  # (batch_size, max_bonds, 3)

        # Apply activations for (r, θ, φ)
        r_a = F.softplus(geom_a_raw[:, :, 0:1])  # (batch_size, max_bonds, 1) - positive distances
        theta_a = torch.sigmoid(geom_a_raw[:, :, 1:2]) * 180.0  # (batch_size, max_bonds, 1) - [0, 180] degrees
        phi_a = torch.sigmoid(geom_a_raw[:, :, 2:3]) * 360.0  # (batch_size, max_bonds, 1) - [0, 360] degrees

        geometry_a = torch.cat([r_a, theta_a, phi_a], dim=-1)  # (batch_size, max_bonds, 3)

        # Predict geometry for B's neighbors
        geom_b_raw = self.geometry_head(emb_b)  # (batch_size, max_bonds * 3)
        geom_b_raw = geom_b_raw.view(-1, self.max_bonds, 3)  # (batch_size, max_bonds, 3)

        # Apply activations for (r, θ, φ)
        r_b = F.softplus(geom_b_raw[:, :, 0:1])  # (batch_size, max_bonds, 1)
        theta_b = torch.sigmoid(geom_b_raw[:, :, 1:2]) * 180.0  # (batch_size, max_bonds, 1) - [0, 180] degrees
        phi_b = torch.sigmoid(geom_b_raw[:, :, 2:3]) * 360.0  # (batch_size, max_bonds, 1) - [0, 360] degrees

        geometry_b = torch.cat([r_b, theta_b, phi_b], dim=-1)  # (batch_size, max_bonds, 3)

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

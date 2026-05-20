"""
Model architecture for stereochemistry-aware 3D reconstruction.

This module implements the LocalGeometryGCN model that predicts:
- Bond distances (A-B distance)
- Local spherical angular geometry (θ, Δφ) for neighbors of both atoms
  where Δφ are relative azimuthal angle differences (first neighbor at Δφ=0)

Architecture improvements over the baseline:
- GATv2Conv: dynamic attention that conditions on both source and target features,
  fixing the static-attention limitation of GAT.
- Native edge features: bond type embeddings flow directly into the GATv2 attention
  mechanism (edge_dim parameter) instead of being manually aggregated and concatenated.
- Residual connections: skip connection from each layer's input to its output,
  with a learned projection only on the first layer (dimension mismatch). Critical
  for stable gradient flow at depth=10.
- Separate geometry heads per atom role: atom A and atom B have different chemical
  roles (central vs neighbour of the bond being evaluated) and benefit from
  independent prediction networks.
- Cross-atom context: each geometry head receives cat(emb_self, emb_partner) so
  the predicted angles around A are conditioned on the nature of B and vice versa.
- Bond-slot positional embeddings: a learnable embedding per bond slot lets the
  network learn slot-specific geometric biases (slot 0 is Δφ=0 by definition;
  sp3 vs sp2 patterns differ across slots).
- Reduced dropout: 0.2 in GNN layers (was 0.5), 0.3 in prediction heads (was 0.4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class LocalGeometryGCN(nn.Module):
    """
    Graph Convolutional Network for predicting local molecular geometry.

    Predicts:
    - Distance between target atoms A and B
    - Local spherical geometry (θ, Δφ) for neighbors of A
    - Local spherical geometry (θ, Δφ) for neighbors of B
      where Δφ are relative azimuthal differences (first neighbor at Δφ=0)

    Args:
        num_node_features: Number of input features (unused when vocab_size is set)
        hidden_dim: Hidden dimension for GNN layers (default: 64)
        max_bonds: Maximum number of bonds per atom (default: 7)
        vocab_size: Size of element vocabulary (enables learned embeddings)
        embedding_dim: Dimension of element embeddings (default: 32)
        depth: Number of GATv2 layers (default: 1)
        use_bond_types: Whether to use bond type edge features (default: True)
        num_bond_types: Number of bond types (default: 4)
        bond_embedding_dim: Dimension of bond type embeddings (default: 8)
        use_chirality: Append signed tetrahedral-volume scalar to atom features (default: True)

    Architecture:
        1. Element embedding + optional chirality scalar
        2. Bond type embedding → native GATv2 edge features
        3. GATv2 encoder (depth layers) with residual connections + BatchNorm
        4. Distance head: cat(emb_a, emb_b) → scalar distance (Softplus)
        5. Per-atom geometry encoder: cat(emb_self, emb_partner) → slot_dim vector
        6. Slot positional embedding added per bond slot
        7. Shared slot decoder: slot_repr → (θ, Δφ) with sigmoid activations
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
        bond_embedding_dim: int = 8,
        use_chirality: bool = True
    ):
        super().__init__()
        self.max_bonds = max_bonds
        self.use_embeddings = vocab_size is not None
        self.use_bond_types = use_bond_types
        self.use_chirality = use_chirality
        self.hidden_dim = hidden_dim
        self.depth = max(1, depth)

        # --- Input features ---
        if self.use_embeddings:
            self.element_embedding = nn.Embedding(vocab_size, embedding_dim)
            encoder_input_dim = embedding_dim + (1 if use_chirality else 0)
        else:
            encoder_input_dim = num_node_features + (1 if use_chirality else 0)

        # Bond type embeddings passed natively into GATv2 attention
        if self.use_bond_types:
            self.bond_embedding = nn.Embedding(num_bond_types, bond_embedding_dim)
            edge_dim = bond_embedding_dim
        else:
            edge_dim = None

        # --- GATv2 encoder ---
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gnn_layers.append(
            GATv2Conv(encoder_input_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(self.depth - 1):
            self.gnn_layers.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=edge_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Projection for the first residual (encoder_input_dim → hidden_dim).
        # Subsequent layers are identity residuals (hidden_dim → hidden_dim).
        self.res_proj = nn.Linear(encoder_input_dim, hidden_dim, bias=False)

        self.dropout_gnn = nn.Dropout(0.2)

        # --- Distance head ---
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # --- Geometry heads ---
        # Each encoder takes cat(emb_self, emb_partner) for cross-atom context.
        slot_dim = hidden_dim // 2
        self.geometry_encoder_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.geometry_encoder_b = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, slot_dim)
        )

        # Positional embedding per bond slot (0 … max_bonds-1).
        # Added to the encoder output before decoding so each slot has its own bias.
        self.slot_embedding = nn.Embedding(max_bonds, slot_dim)

        # Shared slot decoder: same weights for A and B since the encoders already
        # capture atom-role differences.
        self.slot_decoder = nn.Linear(slot_dim, 2)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, node_a, node_b,
                  optionally batch and chirality.

        Returns:
            distance:   (B, 1)             predicted A-B distances
            geometry_a: (B, max_bonds, 2)  predicted (θ, Δφ) for A's neighbors
            geometry_b: (B, max_bonds, 2)  predicted (θ, Δφ) for B's neighbors
        """
        x, edge_index = data.x, data.edge_index
        node_a, node_b = data.node_a, data.node_b
        batch = getattr(data, 'batch', None)
        edge_attr = getattr(data, 'edge_attr', None)
        chirality = getattr(data, 'chirality', None)

        # Element embeddings
        if self.use_embeddings:
            x = self.element_embedding(x)

        # Chirality scalar appended to each atom feature vector
        if self.use_chirality:
            if chirality is not None:
                x = torch.cat([x, chirality.float().unsqueeze(-1)], dim=-1)
            else:
                x = torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=-1)

        # Bond type → edge feature tensor for GATv2
        if self.use_bond_types and edge_attr is not None:
            edge_features = self.bond_embedding(edge_attr)
        else:
            edge_features = None

        # GATv2 encoding with residual connections
        for i, (gnn_layer, bn_layer) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            x_res = x
            if self.use_bond_types and edge_features is not None:
                h = gnn_layer(x, edge_index, edge_attr=edge_features)
            else:
                h = gnn_layer(x, edge_index)
            h = F.relu(h)
            if h.size(0) > 1:
                h = bn_layer(h)
            h = self.dropout_gnn(h)
            # First layer: project residual to match hidden_dim; subsequent: identity
            x = h + (self.res_proj(x_res) if i == 0 else x_res)

        # Extract target-atom embeddings, handling batched graphs
        if batch is not None:
            batch_size = batch.max().item() + 1
            counts = torch.bincount(batch, minlength=batch_size)
            batch_offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=batch.device),
                counts.cumsum(0)
            ])
            batch_indices = torch.arange(batch_size, device=batch.device)
            emb_a = x[node_a + batch_offsets[batch_indices]]
            emb_b = x[node_b + batch_offsets[batch_indices]]
        else:
            emb_a = x[node_a].unsqueeze(0)
            emb_b = x[node_b].unsqueeze(0)

        # Distance prediction
        distance = self.distance_head(torch.cat([emb_a, emb_b], dim=-1))

        # Slot positional embeddings shared by both atoms
        slot_indices = torch.arange(self.max_bonds, device=x.device)
        slot_embs = self.slot_embedding(slot_indices)  # (max_bonds, slot_dim)

        # Geometry A: self=A, partner=B
        base_a = self.geometry_encoder_a(torch.cat([emb_a, emb_b], dim=-1))  # (B, slot_dim)
        slot_repr_a = base_a.unsqueeze(1) + slot_embs.unsqueeze(0)           # (B, max_bonds, slot_dim)
        geom_a_raw = self.slot_decoder(slot_repr_a)                           # (B, max_bonds, 2)

        # Geometry B: self=B, partner=A (reversed)
        base_b = self.geometry_encoder_b(torch.cat([emb_b, emb_a], dim=-1))  # (B, slot_dim)
        slot_repr_b = base_b.unsqueeze(1) + slot_embs.unsqueeze(0)           # (B, max_bonds, slot_dim)
        geom_b_raw = self.slot_decoder(slot_repr_b)                           # (B, max_bonds, 2)

        # Output activations: θ ∈ [0, 180°], Δφ ∈ [0, 360°]
        geometry_a = torch.cat([
            torch.sigmoid(geom_a_raw[:, :, 0:1]) * 180.0,
            torch.sigmoid(geom_a_raw[:, :, 1:2]) * 360.0,
        ], dim=-1)

        geometry_b = torch.cat([
            torch.sigmoid(geom_b_raw[:, :, 0:1]) * 180.0,
            torch.sigmoid(geom_b_raw[:, :, 1:2]) * 360.0,
        ], dim=-1)

        return distance, geometry_a, geometry_b


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module):
    """Xavier init for Linear layers; constant init for BatchNorm; normal for Embeddings."""
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
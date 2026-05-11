# Chirality Feature — Option A Implementation

## Problem

The model input (element indices + bond connectivity + bond types) is identical for
R and S enantiomers.  The training target `(θ, Δφ)` differs between them, so the
model was unable to learn chirality — it predicted an average of both.

---

## What was changed

### 1. `utils/geometry.py`
**Added** the function `compute_chirality_sign` (inserted before `get_neighbor_sort_keys`):

```python
def compute_chirality_sign(
    pos: torch.Tensor,
    center_idx: int,
    sorted_neighbor_indices: List[int],
    epsilon: float = 1e-4
) -> float:
```

Uses the scalar triple product of bond vectors to the **first three neighbours in
canonical CIP-priority order** to give +1.0 (one enantiomer), -1.0 (the other),
or 0.0 (< 3 neighbours / planar centre).

**To revert:** delete the 30-line block from `def compute_chirality_sign(` through
the closing `return 1.0 if vol > 0 else -1.0` line.

---

### 2. `utils/dataset.py`
**Changed** import line (one word added):
```python
# before
from geometry import compute_local_geometry, convert_to_relative_azimuthal, get_neighbor_sort_keys
# after
from geometry import compute_local_geometry, convert_to_relative_azimuthal, get_neighbor_sort_keys, compute_chirality_sign
```

**Added** a 16-line block in `_process_molecule_bond`, between `subgraph_bond_orders`
and the `# **NEW: Compute stereochemistry-aware local geometry**` comment, that
iterates over every subgraph node (`node_list`), retrieves its full-molecule
neighbours from `adjacency`, sorts them with `get_neighbor_sort_keys`, and calls
`compute_chirality_sign(pos, ...)` using the full-molecule `pos` tensor:

```python
chirality_values = []
for node_global in node_list:
    global_neighbors = adjacency[node_global]
    if len(global_neighbors) < 3:
        chirality_values.append(0.0)
        continue
    global_bond_orders = {
        (node_global, n): bond_type_to_order.get(bond_types_dict.get((node_global, n), 0), 1.0)
        for n in global_neighbors
    }
    sort_keys = get_neighbor_sort_keys(global_neighbors, elements, global_bond_orders, node_global)
    sorted_nbrs = [n for n, _ in sorted(zip(global_neighbors, sort_keys), key=lambda kv: kv[1])]
    chirality_values.append(compute_chirality_sign(pos, node_global, sorted_nbrs))
chirality = torch.tensor(chirality_values, dtype=torch.float)
```

**Added** `chirality=chirality,` field to the `Data(...)` constructor call in
`_process_molecule_bond`.

**To revert:** remove the import addition, the 16-line block, and the `chirality=`
line in `Data(...)`.

---

### 3. `utils/model.py`
**Added** `use_chirality: bool = True` parameter to `LocalGeometryGCN.__init__`.

**Changed** `encoder_input_dim`:
```python
# before
encoder_input_dim = embedding_dim          # (or num_node_features for no-embed path)
# after
encoder_input_dim = embedding_dim + (1 if use_chirality else 0)
```
This adds one extra input dimension to the first GAT layer and the first bond-type
projection layer when chirality is enabled.

**Added** in `forward()`, after the element embedding:
```python
if self.use_chirality:
    if chirality is not None:
        x = torch.cat([x, chirality.float().unsqueeze(-1)], dim=-1)
    else:
        x = torch.cat([x, torch.zeros(x.size(0), 1, device=x.device)], dim=-1)
```
The fallback-to-zeros branch keeps the model compatible when loading a checkpoint
that was saved *with* chirality but the incoming batch has no `chirality` field.

**To revert:** remove the `use_chirality` parameter, revert `encoder_input_dim` to
`embedding_dim`, and remove the chirality concatenation block in `forward()`.

---

### 4. `src/reconstruction.py`
**Changed** import:
```python
from utils.geometry import ..., get_neighbor_sort_keys, compute_chirality_sign
```

**Added** `use_chirality=config['model'].get('use_chirality', True)` to the
`LocalGeometryGCN(...)` constructor call in `load_trained_model`.

**Added** in `create_graph_data`, before `Data(...)`:
```python
bond_type_to_order = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
element_list = list(elements)
chirality_values = []
for node_global in node_list:
    global_neighbors = list(adjacency[node_global])
    if len(global_neighbors) < 3:
        chirality_values.append(0.0)
        continue
    global_bond_orders = {
        (node_global, n): bond_type_to_order.get(bond_types_dict.get((node_global, n), 0), 1.0)
        for n in global_neighbors
    }
    sort_keys = get_neighbor_sort_keys(global_neighbors, element_list, global_bond_orders, node_global)
    sorted_nbrs = [n for n, _ in sorted(zip(global_neighbors, sort_keys), key=lambda kv: kv[1])]
    chirality_values.append(compute_chirality_sign(pos, node_global, sorted_nbrs))
chirality = torch.tensor(chirality_values, dtype=torch.float)
```

**Added** `chirality=chirality.to(device),` to the `Data(...)` call.

**Note on inference realism:** At evaluation time, the reconstruction scripts still
have access to the DFT 3D coordinates, so chirality is computed from those (ground
truth).  In a production setting where only a SMILES / connectivity graph is
available, chirality should instead be extracted from the SMILES via RDKit's CIP
assignment and passed in as a pre-computed feature.

**To revert:** remove the import addition, the `use_chirality=` kwarg in the model
constructor, the chirality computation block, and `chirality=` in `Data(...)`.

---

### 5. `src/train_model.py`
**Added** one keyword argument to the `LocalGeometryGCN(...)` call:
```python
use_chirality=config['model'].get('use_chirality', True)
```

**To revert:** remove that line.

---

### 6. `config/config.yaml`
**Added** under the `model:` section (after `bond_embedding_dim`):
```yaml
  use_chirality: true
```

Set to `false` to disable the feature (e.g. to load a pre-chirality checkpoint
without retraining).

**To revert:** delete the `use_chirality: true` line and the comment above it.

---

## How to revert everything atomically

```bash
git diff --stat
git checkout -- utils/geometry.py utils/dataset.py utils/model.py \
                src/reconstruction.py src/train_model.py config/config.yaml
```

(This restores all six files to their state before this change set.)

---

## Checkpoint compatibility

Checkpoints trained **before** this change have a different first-layer weight
shape (`encoder_input_dim` was `embedding_dim`, now `embedding_dim + 1`).
They cannot be loaded with `use_chirality: true`.
Set `use_chirality: false` in `config.yaml` to use old checkpoints unchanged.

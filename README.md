# Stereochemistry-Aware 3D Reconstruction GCN

A Graph Convolutional Network for predicting local molecular geometry using cylindrical coordinates (r, θ, φ), enabling stereochemically-aware 3D structure reconstruction from learned representations.

##  Key Features

- **Stereochemistry-Aware**: Uses cylindrical coordinates (r, θ, φ) in a bond-centered reference frame
- **Rotationally Invariant**: Ordered neighbor representation with azimuthal angles
- **TB-Scale Ready**: Streaming dataset with NO RAM caching, handles terabytes of data
- **File-Based Epochs**: Each CSV file = 1 epoch, automatic cycling through all files
- **Percentage-Based Loss**: Scale-invariant loss functions for better generalization
- **Circular Angle Loss**: Proper handling of periodic φ angles (wraparound)
- **Multi-File Training**: Distributed learning across multiple data files
- **Checkpoint Resume**: Save/load training state with full optimizer state
- **Comprehensive Metrics**: Detailed logging with visualization plots

##  Project Structure

```
new_implementation/
 config.yaml                        # Main configuration file
 local_geometry_gcn_stereo.py      # Main training script
 README.md                          # This file
 IMPLEMENTATION_PROGRESS.md         # Detailed implementation log

 utils/                             # Core modules
    geometry.py                    # Geometry computations (r, θ, φ)
    dataset.py                     # Streaming dataset
    model.py                       # LocalGeometryGCN model
    loss.py                        # Loss functions
    training.py                    # Training loop
    checkpoint.py                  # Checkpointing utilities
    metrics.py                     # Metrics logging
    config.py                      # Configuration utilities

 tests/                             # Unit tests
    test_geometry.py               # Geometry module tests (16/16 )
    test_model.py                  # Model tests (7/7 )
    test_dataset_quick.py          # Dataset quick test

 checkpoints/                       # Training checkpoints (created during training)
 logs/                              # Training logs and plots (created during training)
 models/                            # Final trained models (created during training)
```

##  Quick Start

### 1. Installation

Ensure you have the required dependencies:

```bash
# PyTorch with CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric

# Other dependencies
pip install pandas numpy matplotlib pyyaml tqdm scikit-learn
```

### 2. Data Preparation

Place your CSV files in the parent directory:
```
AI-for-Science/
 Compound_000000001_000025000_props_geom.csv
 Compound_000025001_000050000_props_geom.csv
 ...
```

CSV files should contain:
- `geometry_xyz`: XYZ format atomic coordinates
- `bonds_geom_valence`: JSON list of bonds with 'i' and 'j' indices

### 3. Configuration

Edit `config.yaml` to set your data paths and hyperparameters:

```yaml
data:
  train_files:
    - "../Compound_000000001_000025000_props_geom.csv"
    - "../Compound_000025001_000050000_props_geom.csv"
  val_file: "../Compound_000000001_000025000_props_geom.csv"
  batch_size: 32

model:
  hidden_dim: 64
  embedding_dim: 32

training:
  num_epochs: 50
  learning_rate: 0.0001
```

### 4. Training

Start training with:

```bash
cd new_implementation
python local_geometry_gcn_stereo.py --config config.yaml
```

Override config parameters:
```bash
python local_geometry_gcn_stereo.py --config config.yaml --batch-size 64 --lr 0.001 --epochs 100
```

Resume from checkpoint:
```bash
python local_geometry_gcn_stereo.py --config config.yaml --resume checkpoints/best_model.pt
```

### 5. Monitor Training

Training progress is logged to:
- **Console**: Real-time progress bars and metrics
- **logs/metrics.json**: JSON file with all metrics
- **logs/training_metrics.png**: Visualization plots
- **checkpoints/**: Model checkpoints (saved after each epoch)

##  Model Architecture

### Input
- **Molecular Graph**: Nodes (atoms) with element embeddings, edges (bonds)
- **Target Bond**: Two atoms A and B to predict geometry for
- **Local Subgraph**: 1-hop neighbors around A and B

### Output
For each target bond A-B:
- **Distance**: A-B distance (Å) with softplus activation
- **Geometry A**: (r, θ, φ) for up to 7 neighbors of A
  - r: Distance from A (Å) - softplus
  - θ: Polar angle from A→B axis [0, π] - sigmoid * π
  - φ: Azimuthal angle around A→B axis [0, 2π] - sigmoid * 2π
- **Geometry B**: (r, θ, φ) for up to 7 neighbors of B (same format)

### Architecture
```
Element Embedding (vocab_size → embedding_dim)
    ↓
GCN Layers (embedding_dim → hidden_dim → hidden_dim)
    ↓
Batch Norm + Dropout (regularization)
    ↓
Distance Head: Concatenated embeddings → 1 value (A-B distance)
Geometry Head: Individual embeddings → max_bonds * 3 (r, θ, φ per neighbor)
```

##  Key Concepts

### Cylindrical Coordinates (r, θ, φ)

Instead of predicting absolute 3D positions, we predict geometry in a **bond-centered coordinate system**:

1. **Bond Axis**: A→B bond defines the local z-axis
2. **Orthonormal Frame**: Gram-Schmidt constructs perpendicular x and y axes
3. **Neighbor Coordinates**: Each neighbor's position expressed as:
   - r: Distance from center atom
   - θ: Polar angle from bond axis (0° = along bond, 90° = perpendicular)
   - φ: Azimuthal angle around bond axis (rotation around z-axis)

**Benefits**:
- Rotationally invariant: Same representation regardless of global orientation
- Stereochemistry-aware: φ ordering captures chirality and stereoisomers
- Physically meaningful: Directly interpretable distances and angles

### File-Based Epochs

For TB-scale data that doesn't fit in RAM:
- Each CSV file = 1 epoch
- Dataset streams from disk with chunked reading (10k rows/chunk)
- After each epoch, cycle to next file
- NO persistent caching - pure streaming

Example: 100 files × 25k molecules/file = 100 epochs through 2.5M molecules

### Loss Functions

**Percentage Loss** (distances):
```
L = |pred - target| / (|target| + ε)
```
- Scale-invariant: 0.1Å error on 1Å bond = same penalty as 0.4Å error on 4Å bond
- Better balancing across different bond types

**Circular Loss** (φ angles):
```
L = min(|pred - target|, 2π - |pred - target|)
```
- Handles wraparound: 0° and 360° are equivalent
- Shortest angular distance

**Combined Loss**:
```
L_total = w_dist * L_distance + w_geom * (L_r + L_θ + L_φ)
```
- Configurable weights for each component
- Masking for variable bond counts

##  Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Memory Usage | <4GB | Streaming dataset memory footprint |
| Distance MAE | <0.2 Å | Mean absolute error for bond distances |
| Distance R² | >0.8 | R-squared for distance predictions |
| θ (polar) MAE | <10° | Mean angle error for polar angles |
| φ (azimuthal) MAE | <15° | Mean angle error for azimuthal angles |
| Reconstruction RMSD | <0.5 Å | Full 3D structure reconstruction accuracy |

##  Testing

Run unit tests:
```bash
cd new_implementation
python tests/test_geometry.py      # Geometry functions (16 tests)
python tests/test_model.py         # Model architecture (7 tests)
python tests/test_dataset_quick.py # Dataset streaming
```

##  Configuration Options

### Data
- `train_files`: List of training CSV files
- `val_file`: Validation CSV file
- `batch_size`: Training batch size (default: 32)
- `max_bonds`: Maximum bonds per atom (default: 7)
- `chunk_size`: CSV chunk size for streaming (default: 10000)

### Model
- `hidden_dim`: GCN hidden dimension (default: 64)
- `embedding_dim`: Element embedding dimension (default: 32)
- `vocab_size`: Auto-detected from data

### Training
- `num_epochs`: Number of cycles through all files (default: 50)
- `learning_rate`: Adam learning rate (default: 0.0001)
- `weight_decay`: L2 regularization (default: 0.001)
- `grad_clip`: Gradient clipping max norm (default: 1.0)
- `early_stopping_patience`: Epochs without improvement (default: 5)
- `lr_scheduler`: 'plateau', 'step', 'cosine', or null

### Loss
- `distance_weight`: Weight for A-B distance loss (default: 1.0)
- `geometry_weight`: Weight for (r, θ, φ) loss (default: 0.3)
- `r_weight`, `theta_weight`, `phi_weight`: Individual geometry weights

##  Advanced Usage

### Custom Geometry Functions

The `utils/geometry.py` module provides standalone functions:

```python
from utils.geometry import compute_local_geometry, spherical_to_cartesian

# Compute local geometry for a molecule
pos = torch.tensor([[0,0,0], [1,0,0], [0.5,0.87,0]])  # Atomic positions
ordered_indices, geometry, frame = compute_local_geometry(
    pos,
    center_idx=0,
    neighbor_indices=[2],
    reference_idx=1,
    augment_rotation=True  # Random rotation for training
)

# geometry = [(r, θ, φ), ...] for each neighbor
# frame = (x_axis, y_axis, bond_axis) orthonormal coordinate system

# Reconstruct 3D position from cylindrical coordinates
r, theta, phi = geometry[0]
reconstructed_pos = spherical_to_cartesian(r, theta, phi, frame, pos[0])
```

### Batch Training Script

For multiple experiments:

```python
import subprocess

configs = [
    ("config_small.yaml", 16, 0.001),
    ("config_medium.yaml", 32, 0.0001),
    ("config_large.yaml", 64, 0.00001)
]

for config_file, batch_size, lr in configs:
    subprocess.run([
        "python", "local_geometry_gcn_stereo.py",
        "--config", config_file,
        "--batch-size", str(batch_size),
        "--lr", str(lr)
    ])
```

##  Troubleshooting

### Out of Memory
- Reduce `batch_size` in config.yaml
- Reduce `chunk_size` (dataset reads fewer rows at once)
- Reduce `hidden_dim` and `embedding_dim` (smaller model)

### Slow Training
- Increase `batch_size` (if memory allows)
- Use GPU (CUDA) if available
- Consider converting CSV to Parquet format (10-100x faster reads)

### NaN Loss
- Reduce `learning_rate`
- Check for invalid data in CSV files
- Enable `grad_clip` in training config

### File Not Found
- Verify file paths in config.yaml are correct
- Use absolute paths if relative paths don't work
- Check that CSV files have required columns

##  Citation

If you use this code in your research, please cite:

```bibtex
@software{stereochem_gcn_2025,
  title={Stereochemistry-Aware 3D Reconstruction GCN},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/stereochem-gcn}
}
```

##  License

This project is licensed under the MIT License - see LICENSE file for details.

##  Acknowledgments

- PyTorch Geometric for graph neural network utilities
- Original inspiration from local geometry GCN concepts
- Stereochemistry-aware geometry representation ideas

##  Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Status**:  Core implementation complete (Phase 1-5 done)
**Next**: Integration testing and 3D reconstruction utilities

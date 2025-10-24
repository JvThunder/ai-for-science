# Stereochemistry-Aware 3D Reconstruction GCN

A Graph Convolutional Network for predicting local molecular geometry using Spherical coordinates (r, θ, φ), enabling stereochemically-aware 3D structure reconstruction from learned representations.

##  Key Features

- **Stereochemistry-Aware**: Uses spherical coordinates (r, θ, φ) in a bond-centered reference frame
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
 requirements.txt                   # Python package dependencies with versions
 local_geometry_gcn_stereo.py      # Main training script
 export_validation_ids.py           # Utility to export validation molecule IDs
 README.md                          # This file
 IMPLEMENTATION_PROGRESS.md         # Detailed implementation log
 VALIDATION_BEHAVIOR.md             # Explains validation set behavior across epochs

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

#### Environment Setup

This project was developed and tested with the following package versions:

**Option 1: Quick Install (Recommended)**
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and extensions
pip install torch-geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 --find-links https://data.pyg.org/whl/torch-2.7.0+cu118.html

# Install remaining dependencies
pip install pandas==2.2.3 numpy==1.26.4 pyyaml==6.0.2 tqdm==4.67.1 matplotlib==3.10.1 scikit-learn==1.7.0

# Optional: Weights & Biases for experiment tracking
pip install wandb==0.21.1
```

**Option 2: Using requirements.txt**
```bash
# Install PyTorch first (CUDA version)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install PyG extensions
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 --find-links https://data.pyg.org/whl/torch-2.7.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

**Note:** For CPU-only installation:
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --find-links https://data.pyg.org/whl/torch-2.7.0+cpu.html
pip install -r requirements.txt
```

### 2. Data Preparation

Place your CSV files in th directory:
```
./data/
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
    - "./data/Compound_000000001_000025000_props_geom.csv"
    - "./data/Compound_000025001_000050000_props_geom.csv"
  val_split_ratio: 0.1  # 10% of each file used for validation
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
python local_geometry_gcn_stereo.py
```

The script uses `config.yaml` by default. To specify a different config file:
```bash
python local_geometry_gcn_stereo.py --config custom_config.yaml
```

Override config parameters:
```bash
python local_geometry_gcn_stereo.py --batch-size 64 --lr 0.001 --epochs 100
```

Resume from checkpoint:
```bash
python local_geometry_gcn_stereo.py --resume checkpoints/best_model.pt
```

### 5. Monitor Training

Training progress is logged to:
- **Console**: Real-time progress bars and metrics
- **logs/metrics.json**: JSON file with all metrics
- **logs/training_metrics.png**: Visualization plots
- **checkpoints/**: Model checkpoints (saved after each epoch)
- **Weights & Biases**: Online experiment tracking (optional, see below)

### 6. Understanding Validation Behavior

You can choose between **fixed** or **cycling** validation sets:

#### **Option 1: Fixed Validation (Recommended - Default)**

Set `use_fixed_validation: true` in `config.yaml` (enabled by default):

```yaml
data:
  use_fixed_validation: true  # Validation stays on same molecules every epoch
```

**Behavior:**
- Validation uses only the **first file** and stays fixed across all epochs
- Same molecules are used for validation every epoch
- Validation loss is **directly comparable** across epochs
- Better for early stopping and model selection

Example:
- Epoch 1: Validates on 10% of file 0
- Epoch 2: Validates on 10% of file 0 (same molecules)
- Epoch 3: Validates on 10% of file 0 (same molecules)
- ... and so on

#### **Option 2: Cycling Validation**

Set `use_fixed_validation: false` in `config.yaml`:

```yaml
data:
  use_fixed_validation: false  # Validation cycles through files
```

**Behavior:**
- Validation cycles through files just like training (each file = 1 epoch)
- Different molecules are used for validation each epoch
- Validation loss is **not directly comparable** across epochs
- Provides diverse evaluation across the dataset

Example:
- Epoch 1: Validates on 10% of file 0
- Epoch 2: Validates on 10% of file 1 (different molecules)
- Epoch 3: Validates on 10% of file 2 (different molecules)
- ... and so on

#### **Validation Tracking**

Set `export_validation_ids: true` in `config.yaml` (enabled by default):
```yaml
logging:
  export_validation_ids: true
  validation_ids_file: "logs/validation_ids.json"
```

This automatically creates a JSON file **after the first epoch cycle** showing which molecule indices are used for validation. The tracking happens incrementally during training (not upfront), so there's no startup delay.

**Manual Export (if needed):**
```bash
python export_validation_ids.py --config config.yaml --output logs/validation_ids.json
```

**For more details**, see [`VALIDATION_BEHAVIOR.md`](VALIDATION_BEHAVIOR.md)

##  Weights & Biases Integration

This project includes built-in support for [Weights & Biases (wandb)](https://wandb.ai), a powerful experiment tracking and visualization platform.

### Installation

```bash
pip install wandb
```

### Setup

1. **Create a free wandb account** at [wandb.ai/signup](https://wandb.ai/signup)

2. **Login to wandb** (only needed once):
```bash
wandb login
```
You'll be prompted to enter your API key (found at [wandb.ai/authorize](https://wandb.ai/authorize))

3. **Enable wandb in config.yaml** (enabled by default):
```yaml
logging:
  use_wandb: true
  wandb_project: "stereochem-gcn"
  wandb_entity: null  # Your username or team (optional)
  wandb_run_name: null  # Auto-generated if null
  wandb_tags: ["baseline", "experiment1"]  # Optional tags
  wandb_notes: "Testing new loss weights"  # Optional notes
```

### What Gets Logged

Wandb automatically tracks:
- **Loss curves**: Train/val total loss, distance loss, geometry loss
- **Metrics**: Distance MAE/RMSE, angle errors (θ, φ)
- **Hyperparameters**: Learning rate, batch size, model architecture
- **Learning rate schedule**: Track LR changes over time
- **System metrics**: GPU/CPU usage, memory consumption

### Usage

**Option 1: Use config file (default)**
```bash
python local_geometry_gcn_stereo.py
# Uses config.yaml by default, wandb enabled by default
# View at the URL printed in console
```

**Option 2: Disable wandb temporarily**
```yaml
# In config.yaml
logging:
  use_wandb: false
```

**Option 3: Custom project/run names**
```yaml
logging:
  use_wandb: true
  wandb_project: "my-molecule-project"
  wandb_entity: "my-team"  # Your wandb username or team
  wandb_run_name: "lr-sweep-001"
  wandb_tags: ["sweep", "lr-optimization"]
```

### Viewing Results

After training starts, you'll see:
```
WandB initialized: run-2025-01-15-143022
View at: https://wandb.ai/username/stereochem-gcn/runs/abc123
```

Visit that URL to see:
- **Real-time metrics**: Loss curves update during training
- **Interactive plots**: Compare multiple runs
- **System metrics**: GPU/CPU/memory usage
- **Model artifacts**: Download checkpoints
- **Notes & tags**: Organize experiments

### Advanced Features

**Compare Multiple Runs:**
```bash
# Run 1: Baseline
python local_geometry_gcn_stereo.py --lr 0.0001

# Run 2: Higher LR
python local_geometry_gcn_stereo.py --lr 0.001

# View both runs in wandb dashboard and compare
```

**Sweeps (Hyperparameter Optimization):**

Create a sweep config file `sweep.yaml`:
```yaml
program: local_geometry_gcn_stereo.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.00001
    max: 0.001
  batch_size:
    values: [16, 32, 64]
  hidden_dim:
    values: [32, 64, 128]
```

Run sweep:
```bash
wandb sweep sweep.yaml  # Creates sweep ID
wandb agent USERNAME/PROJECT/SWEEP_ID  # Starts agents
```

**Offline Mode** (no internet):
```bash
export WANDB_MODE=offline
python local_geometry_gcn_stereo.py
# Logs saved locally, sync later with: wandb sync
```

### Troubleshooting

**"wandb not installed" warning:**
```bash
pip install wandb
```

**Authentication issues:**
```bash
wandb login --relogin
```

**Disable wandb for debugging:**
```yaml
logging:
  use_wandb: false
```

**View past runs:**
Visit [wandb.ai/USERNAME/stereochem-gcn](https://wandb.ai) (replace USERNAME with your wandb username)

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

### Spherical Coordinates (r, θ, φ)

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

### File-Based Epochs with Per-Epoch Validation

For TB-scale data that doesn't fit in RAM:
- Each CSV file = 1 epoch
- Dataset streams from disk with chunked reading (10k rows/chunk)
- After each epoch, cycle to next file
- NO persistent caching - pure streaming
- **Validation split**: Each file is randomly split (e.g., 90% train / 10% val)
  - The split is deterministic (same molecules in train/val each time you run)
  - Different files get different random splits for diversity

Example: 100 files × 25k molecules/file = 100 epochs through 2.5M molecules
- Epoch 1: File 1 (90% train, 10% val)
- Epoch 2: File 2 (90% train, 10% val)
- Epoch 3: File 3 (90% train, 10% val)
- ...

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

##  Command-Line Arguments

The training script supports the following command-line arguments:

```bash
python local_geometry_gcn_stereo.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config.yaml` | Path to YAML configuration file |
| `--resume` | str | None | Path to checkpoint file to resume training from |
| `--batch-size` | int | None | Override batch size from config |
| `--lr` | float | None | Override learning rate from config |
| `--epochs` | int | None | Override number of epochs from config |

**Examples:**
```bash
# Basic training (uses config.yaml by default)
python local_geometry_gcn_stereo.py

# Custom config file
python local_geometry_gcn_stereo.py --config my_config.yaml

# Override specific parameters
python local_geometry_gcn_stereo.py --batch-size 64 --lr 0.001 --epochs 100

# Resume from checkpoint
python local_geometry_gcn_stereo.py --resume checkpoints/best_model.pt

# Combine options
python local_geometry_gcn_stereo.py --config my_config.yaml --batch-size 128 --resume checkpoints/checkpoint_0050.pt
```

##  Configuration File (`config.yaml`)

All training parameters can be configured in `config.yaml`. Command-line arguments override config values.

### Data Configuration (`data:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_folder` | str | `"./data"` | Directory containing CSV files. Each file = 1 epoch |
| `val_split_ratio` | float | `0.1` | Fraction of each file for validation (0.0-1.0). E.g., 0.1 = 90% train / 10% val |
| `use_fixed_validation` | bool | `true` | **NEW** If true, validation uses only first file (same molecules every epoch). If false, validation cycles through files |
| `max_bonds` | int | `7` | Maximum bonds per atom (for padding geometry predictions) |
| `batch_size` | int | `512` | Training batch size. Adjust based on available GPU memory |
| `num_workers` | int | `4` | Number of parallel data loading workers (0 to disable multiprocessing) |
| `prefetch_factor` | int | `1` | Number of batches to prefetch per worker |
| `chunk_size` | int | `10000` | Number of CSV rows to read per chunk (memory efficiency) |
| `cache_ram_mb` | int | `2048` | RAM limit for chunk caching in MB (trades RAM for speed) |
| `min_element_count` | int | `10` | Minimum element occurrences to include in vocabulary |

### Model Architecture (`model:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | `32` | Hidden dimension for GCN layers |
| `embedding_dim` | int | `16` | Element embedding dimension |
| `vocab_size` | int | null | Vocabulary size (set automatically from data) |
| `max_bonds` | int | `7` | Maximum bonds (should match `data.max_bonds`) |
| `depth` | int | `1` | Number of GCN layers (1=1-hop neighbors, 2=2-hop, etc.) |

### Training Configuration (`training:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | int | `2` | Number of cycles through all files |
| `learning_rate` | float | `0.0001` | Initial learning rate for Adam optimizer |
| `weight_decay` | float | `0.001` | L2 regularization weight decay |
| `grad_clip` | float | `1.0` | Gradient clipping max norm (prevents exploding gradients) |
| `early_stopping_patience` | int | `5` | Epochs without improvement before stopping |
| `validate_every` | int | `1` | Validate every N epochs |
| `lr_scheduler` | str | `"plateau"` | LR scheduler: `"plateau"`, `"step"`, `"cosine"`, or `null` |
| `lr_scheduler_params` | dict | See below | Parameters for the LR scheduler |

**LR Scheduler Parameters (`lr_scheduler_params:`):**

For `lr_scheduler: "plateau"`:
- `factor` (float, default: 0.7): Reduce LR by this factor
- `patience` (int, default: 2): Epochs without improvement before reducing LR
- `mode` (str, default: "min"): Optimize for minimum loss

For `lr_scheduler: "step"`:
- `step_size` (int, default: 10): Reduce LR every N epochs
- `gamma` (float, default: 0.7): Multiply LR by this factor

For `lr_scheduler: "cosine"`:
- `T_max` (int, default: num_epochs): Period of cosine annealing
- `eta_min` (float, default: 1e-6): Minimum learning rate

### Loss Function Weights (`loss:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance_weight` | float | `1.0` | Weight for A-B distance prediction loss |
| `geometry_weight` | float | `1.0` | Weight for overall geometry (r, θ, φ) loss |
| `r_weight` | float | `1.0` | Weight for radial distance (r) component |
| `theta_weight` | float | `1.0` | Weight for polar angle (θ) component |
| `phi_weight` | float | `1.0` | Weight for azimuthal angle (φ) component |

**Note:** All losses use percentage-based metrics for scale invariance.

### Checkpointing (`checkpointing:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | str | `"checkpoints"` | Directory to save model checkpoints |
| `save_every` | int | `1` | Save checkpoint every N epochs |
| `keep_last_n` | int | `1` | Keep only N most recent checkpoints (saves disk space) |

### Logging (`logging:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | str | `"logs"` | Directory for logs and plots |
| `log_every` | int | `10` | Log metrics every N batches during training |
| `export_validation_ids` | bool | `true` | Export validation molecule IDs to JSON file |
| `validation_ids_file` | str | `"logs/validation_ids.json"` | Path to validation IDs output file |
| `use_wandb` | bool | `true` | Enable Weights & Biases experiment tracking |
| `wandb_project` | str | `"stereochem-gcn"` | WandB project name |
| `wandb_entity` | str | `null` | WandB username or team (optional) |
| `wandb_run_name` | str | `null` | Custom run name (auto-generated if null) |
| `wandb_tags` | list | `[]` | List of tags for organizing runs (e.g., `["baseline", "experiment1"]`) |
| `wandb_notes` | str | `null` | Optional notes about the run |

### Hardware (`hardware:`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | `"cuda"` | Device: `"cuda"`, `"cpu"`, or `"auto"` (auto-detect) |
| `mixed_precision` | bool | `false` | Enable mixed precision training (if supported) |

### Reproducibility

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | int | `42` | Random seed for reproducibility (affects train/val splits, augmentation) |

### Example Configuration

```yaml
# Minimal config for quick testing
data:
  train_folder: "./data"
  val_split_ratio: 0.1
  batch_size: 32
  num_workers: 0  # Disable multiprocessing for debugging

model:
  hidden_dim: 32
  embedding_dim: 16
  depth: 1

training:
  num_epochs: 10
  learning_rate: 0.0001
  lr_scheduler: "plateau"

logging:
  use_wandb: false  # Disable for quick local runs
  export_validation_ids: true

random_seed: 42
```

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

# Reconstruct 3D position from spherical coordinates
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

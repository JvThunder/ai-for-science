# Stereochemistry-Aware 3D Reconstruction GCN

GATv2-based model that predicts local molecular geometry in spherical coordinates (r, θ, φ) for stereochemically-aware 3D structure reconstruction.

The model (`LocalGeometryGCN`) outputs per-bond predictions: distance (r), polar angle (θ), and relative azimuthal angle (Δφ) for each neighbour slot of both endpoint atoms. Key architecture features: dynamic GATv2 attention with native bond-type edge features, residual connections, separate geometry heads per atom role with cross-atom context, bond-slot positional embeddings, and optional chirality scalar input.

## Project Layout

```text
.
|-- config/
|-- data/
|-- scripts/
|-- src/
|   |-- reconstruction/
|   |   |-- reconstruction.py          # Core reconstruct_from_center logic
|   |   |-- molecules.py              # MoleculeReconstructor base class + all molecule classes
|   |   |-- reconstruct_acetylsalicylic_acid.py  # Aspirin (C9H8O4)  ──┐
|   |   |-- reconstruct_cyclohexane.py           # Cyclohexane (C6H12)  |
|   |   |-- reconstruct_naphthalene.py           # Naphthalene (C10H8)  | thin wrappers
|   |   |-- reconstruct_polymer_peg.py           # PEG                  | around molecules.py
|   |   |-- reconstruct_polymer_polyester.py     # Polyester            |
|   |   |-- reconstruct_polymer_polystyrene.py   # Polystyrene          |
|   |   |-- reconstruct_polymer_polytetrafluoroethylene.py  # PTFE    ──┘
|   |   `-- reconstruct_polymer_time.py          # Timing benchmark across polymer sizes
|   |-- simulation/
|   |   |-- geom_relaxation_mass.py   # Batch geometry relaxation via Quantum ESPRESSO
|   |   |-- compound/                 # Per-compound input files (subdirs 3–7)
|   |   `-- pseudo/PBE/USPP/          # PBE ultrasoft pseudopotentials (.UPF)
|   |-- train/
|   |-- utils/
|   |   |-- model.py      # LocalGeometryGCN (GATv2)
|   |   |-- dataset.py
|   |   |-- geometry.py
|   |   `-- config.py
|   `-- metrics_calculation.py  # Bond + angle error metrics vs ground truth
|-- checkpoints/
|-- logs/
`-- pyproject.toml
```

## Table of Contents

- [Project Layout](#project-layout)
- [Setup With uv](#setup-with-uv)
- [Running The Project](#running-the-project)
- [Data](#data)
- [Training](#training)
- [Reconstruction](#reconstruction)
- [Metrics](#metrics)
- [Visualisation](#visualisation)

## Setup With uv

1. Install `uv` if it is not already on your machine.

```bash
# Windows
winget install Astral-sh.uv

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create the project environment and install dependencies.

```bash
uv venv
uv sync
```

`uv sync` installs all dependencies defined in [pyproject.toml](pyproject.toml).

If you need a GPU build of PyTorch or a specific wheel index, configure that before syncing, for example:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running The Project

Use `uv run` to execute commands inside the project environment without manually activating a virtualenv.

### Quick Commands (via scripts)

For Linux/macOS:
```bash
./scripts/train.sh              # Train model
./scripts/reconstruct.sh        # Reconstruct geometry
```

For Windows:
```cmd
scripts\train.bat
scripts\reconstruct.bat
```

### Direct Python Commands

```bash
uv run python -m src.train.train_model
uv run python -m src.reconstruction.reconstruction
```

## Data

Place CSV files in `./data/`. Each file is one epoch. Files must contain:

| Column | Description |
|--------|-------------|
| `geometry_xyz` | XYZ format 3D coordinates (DFT-optimised) |
| `bonds_geom_valence` | JSON bond list with `i`, `j`, `order` keys |
| `atoms.aid_pubchem` | PubChem atom IDs |
| `atoms.element_pubchem` | Atomic numbers |
| `bonds.aid1_pubchem` / `bonds.aid2_pubchem` | PubChem bond endpoints |
| `bonds.order_pubchem` | Bond orders (1/2/3/4) |
| `Pubchem_to_xyz_aid_mapping` | Maps PubChem atom order → XYZ atom order |

PubChem columns are used by default. Falls back to `geometry_xyz` / `bonds_geom_valence` if `Pubchem_to_xyz_aid_mapping` contains `"error"`.

## Training

Edit `config/config.yaml` to set `data.train_folder`, then:

```bash
uv run python -m src.train.train_model
```

Resume from checkpoint:
```bash
# Set in config/config.yaml:
training:
  resume_from_latest: true
```

Checkpoints are saved to `checkpoints/`, logs to `logs/`.
Enable Weights & Biases with `logging.use_wandb: true` and `wandb login`.

## Reconstruction

```bash
# Linux/macOS
./scripts/reconstruct.sh

# Windows
scripts\reconstruct.bat

# Or Python directly
uv run python -m src.reconstruction.reconstruction
```

Requires a trained checkpoint at `checkpoints/best_model.pt`. Edit the `__main__` block to set `csv_path` and `molecule_indices`.

### Molecule-Specific Scripts

Each script hard-codes a molecule's topology, runs `reconstruct_from_center`, exports an XYZ file and a PNG, and prints a bond-length analysis. All accept `--output_dir` (default: `polymer_results/`).

| Script | Molecule | Formula |
|--------|----------|---------|
| `reconstruct_acetylsalicylic_acid.py` | Aspirin | C9H8O4 |
| `reconstruct_cyclohexane.py` | Cyclohexane | C6H12 |
| `reconstruct_naphthalene.py` | Naphthalene | C10H8 |
| `reconstruct_polymer_peg.py` | PEG | -(CH2CH2O)n- |
| `reconstruct_polymer_polyester.py` | Polyester | -(OC-R-COO-R')n- |
| `reconstruct_polymer_polystyrene.py` | Polystyrene | -(CH2CH(C6H5))n- |
| `reconstruct_polymer_polytetrafluoroethylene.py` | PTFE | -(CF2CF2)n- |
| `reconstruct_polymer_time.py` | Polystyrene timing | — |

```bash
uv run python src/reconstruction/reconstruct_cyclohexane.py --output_dir results/
uv run python src/reconstruction/reconstruct_naphthalene.py --output_dir results/
uv run python src/reconstruction/reconstruct_acetylsalicylic_acid.py --output_dir results/
uv run python src/reconstruction/reconstruct_polymer_polystyrene.py --output_dir results/
```

## Metrics

`src/metrics_calculation.py` evaluates a trained model against ground-truth DFT geometries from CSV files. It computes per-bond distance errors and per-slot local angle errors (θ, Δφ), grouped by bond type and order.

```bash
uv run python src/metrics_calculation.py \
  --root path/to/project \
  --csv-dir path/to/B3YLP_Database
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | *(required)* | Project root containing `checkpoints/` and `config/` |
| `--csv-dir` | *(required)* | Directory of input CSV files |
| `--checkpoint` | `<root>/checkpoints/best_model.pt` | Model checkpoint |
| `--config` | `<root>/config/config.yaml` | Config YAML |
| `--out-bond-csv` | `<root>/local_training_style_bond_details.csv` | Per-bond predictions |
| `--out-angle-csv` | `<root>/local_training_style_angle_details.csv` | Per-angle predictions |
| `--out-summary` | `<root>/local_training_style_summary.txt` | Aggregated summary |

Outputs three files: a bond-level CSV, an angle-level CSV, and a summary TXT with MAE/RMSE tables grouped by bond type/order and angle type.

## Visualisation

Visualise the nth row of a CSV as an interactive 3D HTML:

```bash
# Row 0 (default)
uv run python src/visualize_molecule.py

# Specific row
uv run python src/visualize_molecule.py --row 5

# Custom CSV and output
uv run python src/visualize_molecule.py --row 2 --csv path/to/file.csv --out mol.html
```

Opens automatically in the browser. Atoms are coloured by element (CPK), bonds styled by order (solid=single, dashed=double, dotted=aromatic).


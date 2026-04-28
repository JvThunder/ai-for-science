# Stereochemistry-Aware 3D Reconstruction GCN

GAT-based model that predicts local molecular geometry in spherical coordinates (r, θ, φ) for stereochemically-aware 3D structure reconstruction.

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

Use `uv run` so commands execute inside the project environment without manually activating a virtualenv.

```bash
uv run python src/train_model.py
uv run python src/reconstruction.py
uv run python src/visualize_molecule.py --row 5
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
uv run python src/train_model.py
```

Resume from checkpoint:
```bash
# Set in config/config.yaml:
training:
  resume_from_latest: true
```

Checkpoints are saved to `checkpoints/`, logs to `logs/`.
Enable Weights & Biases with `logging.use_wandb: true` and `wandb login`.

## LoRA Fine-Tuning

Parameter-efficient fine-tuning using LoRA adapters (trains ~1-5% of parameters).

Edit `config/lora_config.yaml` to set `lora.pretrained_checkpoint` and `data.train_folder`, then:

```bash
# First-time setup
uv sync
uv run bash lora/setup_lora.sh

# Train
uv run bash lora/run_lora.sh train

# Or with Python directly
uv run python lora/train_lora.py
```

Other utilities:

```bash
# Compare base vs LoRA model
uv run bash lora/run_lora.sh compare

# Merge LoRA weights into base model
uv run bash lora/run_lora.sh merge

# Analyze LoRA statistics
uv run bash lora/run_lora.sh analyze
```

LoRA checkpoints are saved to `checkpoints/lora/`, logs to `logs/lora/`. See [lora/LORA_FINETUNING.md](lora/LORA_FINETUNING.md) for full documentation.

## Reconstruction

```bash
uv run python src/reconstruction.py
```

Requires a trained checkpoint at `checkpoints/best_model.pt`. Edit the `__main__` block to set `csv_path` and `molecule_indices`.

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

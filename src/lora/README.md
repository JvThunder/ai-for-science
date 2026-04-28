# LoRA Fine-Tuning for Stereochemistry-Aware 3D GCN

This folder contains all files needed for LoRA (Low-Rank Adaptation) fine-tuning of the molecular geometry prediction model.

## Contents

### Scripts
- **`train_lora.py`** - Main LoRA training script
- **`lora_utils.py`** - Utility tools for managing LoRA models
- **`run_lora.sh`** - Bash runner script (recommended)
- **`setup_lora.sh`** - First-time setup script

### Configuration & Documentation
- **`lora_config.yaml`** - Configuration template for LoRA fine-tuning
- **`LORA_FINETUNING.md`** - Comprehensive documentation and guide
- **`README.md`** - This file

## Quick Start

### First Time Setup

```bash
uv sync
uv run bash lora/setup_lora.sh  # Creates directories and validates the LoRA workspace
```

### Using uv Directly

```bash
# Train with LoRA (uses lora_config.yaml by default)
uv run python lora/train_lora.py

# Or with custom config
CONFIG_FILE=my_config.yaml uv run python lora/train_lora.py

# Compare base and LoRA models
uv run python lora/lora_utils.py compare --base models/best_model.pt --lora models/final_lora_model.pt

# Merge LoRA weights into base model
uv run python lora/lora_utils.py merge --lora models/final_lora_model.pt --output models/merged.pt

# Analyze LoRA statistics
uv run python lora/lora_utils.py analyze --lora models/final_lora_model.pt

# Extract LoRA weights only
uv run python lora/lora_utils.py extract --lora models/final_lora_model.pt --output models/lora_only.pt

# Show help
uv run bash lora/run_lora.sh help
```

### Using Bash Scripts

```bash
# Train
uv run bash lora/run_lora.sh train

# Or with custom config
uv run bash lora/run_lora.sh train my_config.yaml

# Utilities
uv run bash lora/run_lora.sh compare
uv run bash lora/run_lora.sh merge
uv run bash lora/run_lora.sh analyze
```

## Documentation

See **`LORA_FINETUNING.md`** for:
- Detailed explanation of LoRA
- Hyperparameter tuning guide
- Use cases and examples
- Troubleshooting
- Best practices

## What is LoRA?

LoRA enables parameter-efficient fine-tuning by:
- Freezing pre-trained weights
- Adding trainable low-rank adapters
- Training only 1-5% of parameters
- Achieving 95-100% of full fine-tuning performance

Perfect for domain adaptation, limited data, or resource-constrained environments!

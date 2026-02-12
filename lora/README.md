# LoRA Fine-Tuning for Stereochemistry-Aware 3D GCN

This folder contains all files needed for LoRA (Low-Rank Adaptation) fine-tuning of the molecular geometry prediction model.

## Contents

### Scripts
- **`local_geometry_gcn_stereo_lora.py`** - Main LoRA training script
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
cd lora
./setup_lora.sh  # Checks dependencies, creates directories, configures LoRA
```

### Using Bash Scripts (Recommended)

```bash
cd lora

# Train with LoRA (uses lora_config.yaml by default)
./run_lora.sh train

# Or with custom config
./run_lora.sh train my_config.yaml

# Compare base and LoRA models
./run_lora.sh compare

# Merge LoRA weights into base model
./run_lora.sh merge

# Analyze LoRA statistics
./run_lora.sh analyze

# Extract LoRA weights only
./run_lora.sh extract

# Show help
./run_lora.sh help
```

### Using Python Directly

```bash
cd lora

# Train
python local_geometry_gcn_stereo_lora.py

# Or with custom config
CONFIG_FILE=my_config.yaml python local_geometry_gcn_stereo_lora.py

# Utilities
python lora_utils.py compare --base ../models/final_model.pt --lora ../models/final_lora_model.pt
python lora_utils.py merge --lora ../models/final_lora_model.pt --output ../models/merged.pt
python lora_utils.py analyze --lora ../models/final_lora_model.pt
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

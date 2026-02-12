# LoRA Fine-Tuning for Stereochemistry-Aware 3D GCN

This guide explains how to use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of the molecular geometry prediction model.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:

- **Freezes** the pre-trained model weights
- **Adds** trainable low-rank matrices (adapters) to specific layers
- **Reduces** trainable parameters by 10-100x
- **Maintains** model performance comparable to full fine-tuning
- **Enables** faster training and lower memory usage

### How LoRA Works

Instead of fine-tuning all weights `W`, LoRA decomposes weight updates into low-rank matrices:

```
h = W₀·x + (B·A)·x · (α/r)
```

Where:
- `W₀`: Frozen pre-trained weights
- `A`: Low-rank matrix (input_dim × rank)
- `B`: Low-rank matrix (rank × output_dim)
- `r`: Rank (typically 4-32)
- `α`: Scaling factor (typically 2r)

**Example**: For a 256×256 layer with rank=8:
- Full fine-tuning: 65,536 parameters
- LoRA: 4,096 parameters (16x reduction)

## Quick Start

### 1. Train Base Model

First, train a base model using the standard training script:

```bash
# Train base model
python local_geometry_gcn_stereo.py

# This will save: models/final_model.pt
```

### 2. Configure LoRA Fine-Tuning

Edit `lora_config.yaml` to specify:

```yaml
lora:
  enabled: true
  rank: 8                                         # Low-rank dimension
  alpha: 16.0                                     # Scaling factor
  dropout: 0.1                                    # LoRA dropout

  target_modules:                                 # Modules to apply LoRA
    - "gat_layers"
    - "final_mlp"

  pretrained_checkpoint: "models/final_model.pt"  # Base model
  learning_rate: 0.0001                           # Lower LR for fine-tuning
```

### 3. Run LoRA Fine-Tuning

```bash
# Fine-tune with LoRA
python local_geometry_gcn_stereo_lora.py

# Or specify custom config
CONFIG_FILE=my_lora_config.yaml python local_geometry_gcn_stereo_lora.py
```

### 4. Use Fine-Tuned Model

The script saves two versions:

1. **LoRA model** (`final_lora_model.pt`): Contains LoRA adapters for further fine-tuning
2. **Merged model** (`final_merged_model.pt`): LoRA weights integrated into base model for inference

```python
# For inference, use the merged model
checkpoint = torch.load("models/final_merged_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

## Configuration Guide

### LoRA Hyperparameters

#### Rank (`rank`)
Controls the bottleneck dimension of LoRA adapters.

- **Smaller rank (4-8)**: Fewer parameters, faster training, may limit expressiveness
- **Larger rank (16-32)**: More parameters, slower training, more expressive
- **Recommended**: Start with 8, increase if underfitting

```yaml
lora:
  rank: 8  # Good default for most tasks
```

#### Alpha (`alpha`)
Scaling factor for LoRA updates.

- **Rule of thumb**: `alpha = 2 × rank`
- **Lower alpha**: Smaller updates, more conservative fine-tuning
- **Higher alpha**: Larger updates, more aggressive fine-tuning

```yaml
lora:
  alpha: 16.0  # For rank=8
```

#### Dropout (`dropout`)
Dropout rate applied to LoRA layers.

- **0.0**: No dropout (risk of overfitting on small datasets)
- **0.1-0.2**: Standard dropout (recommended)
- **>0.2**: High dropout (for very small datasets)

```yaml
lora:
  dropout: 0.1
```

#### Target Modules (`target_modules`)
Which layers to apply LoRA to.

**Options**:
- `gat_layers`: Graph attention layers (most important)
- `final_mlp`: Final prediction head
- `embedding`: Element embeddings (use if vocabulary changes)

**Recommendations**:
- **Standard fine-tuning**: `["gat_layers", "final_mlp"]`
- **Light fine-tuning**: `["final_mlp"]` (only prediction head)
- **Heavy fine-tuning**: `["gat_layers", "final_mlp", "embedding"]`
- **New elements**: Must include `"embedding"`

```yaml
lora:
  target_modules:
    - "gat_layers"    # Adapt attention mechanisms
    - "final_mlp"     # Adapt predictions
```

### Learning Rate

Use a **lower learning rate** than the base model:

```yaml
lora:
  learning_rate: 0.0001  # 10x lower than base model
```

**Guidelines**:
- Base model LR: 0.001
- LoRA LR: 0.0001 - 0.0005
- Very small dataset: 0.00005

### Training Epochs

LoRA fine-tuning converges **faster** than full fine-tuning:

```yaml
training:
  num_epochs: 5  # Much fewer than base model training
```

**Guidelines**:
- **Small dataset**: 3-5 epochs
- **Medium dataset**: 5-10 epochs
- **Large dataset**: 10-20 epochs
- Monitor validation loss to avoid overfitting

## Use Cases

### 1. Domain Adaptation

Adapt a model trained on general molecules to a specific chemical domain:

```yaml
# Example: Fine-tune on drug-like molecules
data:
  train_folder: "data/drug_molecules"

lora:
  rank: 8
  learning_rate: 0.0001
  target_modules: ["gat_layers", "final_mlp"]

training:
  num_epochs: 5
```

### 2. New Element Types

Add support for new elements not in the base model:

```yaml
# Example: Add rare earth elements
lora:
  rank: 16  # Higher rank for new features
  target_modules:
    - "embedding"      # Must include embedding
    - "gat_layers"
    - "final_mlp"

data:
  elements_txt: "rare_earth_elements.txt"
```

### 3. Improved Stereochemistry

Fine-tune for better stereochemical predictions:

```yaml
# Example: Focus on chiral molecules
data:
  train_folder: "data/chiral_molecules"

lora:
  rank: 8
  target_modules: ["gat_layers", "final_mlp"]

loss:
  phi_weight: 2.0  # Increase weight on azimuthal angles
```

### 4. Low-Resource Fine-Tuning

Fine-tune on a small dataset:

```yaml
lora:
  rank: 4           # Lower rank to prevent overfitting
  alpha: 8.0
  dropout: 0.2      # Higher dropout
  learning_rate: 0.00005  # Very low LR

training:
  num_epochs: 3     # Fewer epochs

data:
  batch_size: 16    # Smaller batches
```

## Advanced Usage

### Resume from LoRA Checkpoint

```yaml
training:
  resume_checkpoint: "checkpoints/lora/checkpoint_epoch_3.pt"
```

Or auto-resume from latest:

```yaml
training:
  resume_from_latest: true
```

### Merge LoRA Weights for Inference

```python
from local_geometry_gcn_stereo_lora import merge_lora_weights

# Load LoRA model
model_with_lora = torch.load("models/final_lora_model.pt")

# Merge adapters into base weights
merged_model = merge_lora_weights(model_with_lora)

# Save merged model
torch.save(merged_model.state_dict(), "merged_model.pt")
```

### Multiple LoRA Adapters

Train different adapters for different tasks:

```bash
# Train adapter 1: Drug molecules
CONFIG_FILE=lora_drug.yaml python local_geometry_gcn_stereo_lora.py

# Train adapter 2: Polymers
CONFIG_FILE=lora_polymer.yaml python local_geometry_gcn_stereo_lora.py

# Train adapter 3: Organometallics
CONFIG_FILE=lora_organometallic.yaml python local_geometry_gcn_stereo_lora.py
```

Load the appropriate adapter at inference time.

### Weights & Biases Tracking

Enable W&B to track LoRA experiments:

```yaml
logging:
  use_wandb: true
  wandb_project: "stereochem-gcn-lora"
  wandb_tags:
    - "lora"
    - "domain_adaptation"
  wandb_notes: "Fine-tuning for drug-like molecules"
```

## Performance Comparison

### Memory Usage

| Method | Trainable Params | Memory | Training Speed |
|--------|-----------------|---------|----------------|
| Full Fine-tuning | 100% (5M) | 12 GB | 1.0x |
| LoRA (r=8) | 1-2% (50-100K) | 6 GB | 1.5x |
| LoRA (r=16) | 2-5% (100-250K) | 7 GB | 1.3x |

### Accuracy

LoRA typically achieves **95-100%** of full fine-tuning performance with **10-100x** fewer parameters.

## Troubleshooting

### Issue: Model not improving

**Solutions**:
1. Increase rank: `rank: 16` or `rank: 32`
2. Increase learning rate: `learning_rate: 0.0005`
3. Add more target modules
4. Train for more epochs

### Issue: Overfitting quickly

**Solutions**:
1. Decrease rank: `rank: 4`
2. Increase dropout: `dropout: 0.2`
3. Decrease learning rate: `learning_rate: 0.00005`
4. Reduce epochs
5. Increase weight decay

### Issue: Out of memory

**Solutions**:
1. Decrease batch size: `batch_size: 16`
2. Decrease rank: `rank: 4`
3. Remove target modules (only keep essential ones)
4. Use gradient checkpointing (requires code modification)

### Issue: No pre-trained model specified

**Error**: `Warning: No pre-trained checkpoint specified!`

**Solution**: Set the pre-trained checkpoint path:

```yaml
lora:
  pretrained_checkpoint: "models/final_model.pt"
```

### Issue: Checkpoint loading fails

**Error**: `KeyError` or shape mismatch

**Causes**:
1. Model architecture mismatch
2. Vocab size difference
3. Corrupted checkpoint

**Solutions**:
1. Ensure model config matches pre-trained model
2. Use same `elements.txt` file
3. Retrain base model if needed

## Best Practices

1. **Always start from a pre-trained model**: LoRA is for fine-tuning, not training from scratch
2. **Use lower learning rates**: 10-100x lower than base model training
3. **Monitor validation loss closely**: LoRA can overfit quickly on small datasets
4. **Start with conservative hyperparameters**: `rank=8`, `alpha=16`, `dropout=0.1`
5. **Save both versions**: Keep LoRA adapters (for further tuning) and merged model (for inference)
6. **Use W&B or tensorboard**: Track experiments to find optimal hyperparameters
7. **Validate on held-out data**: Ensure improvements generalize

## Citation

If you use LoRA in your research, please cite:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## Support

For issues or questions:
1. Check this documentation
2. Review `lora_config.yaml` comments
3. Examine training logs in `logs/lora/`
4. Open an issue on GitHub

## Next Steps

1. Train a base model with `local_geometry_gcn_stereo.py`
2. Prepare your fine-tuning dataset
3. Configure `lora_config.yaml`
4. Run `local_geometry_gcn_stereo_lora.py`
5. Evaluate the fine-tuned model
6. Use merged model for inference

Happy fine-tuning! 🚀

#!/bin/bash
# Fine-tune the model with LoRA using config/lora_config.yaml
uv run python -m src.lora.train_lora

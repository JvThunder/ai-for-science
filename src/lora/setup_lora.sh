#!/bin/bash
#
# LoRA Fine-Tuning Setup Script
#
# This script helps you set up LoRA fine-tuning for the first time.
# It checks dependencies, creates necessary directories, and guides you
# through the configuration process.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

print_header() {
    echo ""
    print_msg "$BLUE" "========================================"
    print_msg "$BLUE" "$1"
    print_msg "$BLUE" "========================================"
    echo ""
}

# Check Python
check_python() {
    print_header "Checking Python Environment"

    if ! command -v uv &> /dev/null; then
        print_msg "$RED" "✗ uv not found!"
        exit 1
    fi

    local uv_version=$(uv --version | awk '{print $2}')
    print_msg "$GREEN" "✓ uv $uv_version found"

    print_msg "$YELLOW" "Syncing project dependencies..."
    uv sync

    print_msg "$YELLOW" "Optional wandb support is available with:"
    print_msg "$YELLOW" "  uv sync --extra wandb"
}

# Create directories
setup_directories() {
    print_header "Setting Up Directories"

    local dirs=(
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/checkpoints/lora"
        "$PROJECT_ROOT/logs/lora"
    )

    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_msg "$GREEN" "✓ $dir (exists)"
        else
            mkdir -p "$dir"
            print_msg "$GREEN" "✓ $dir (created)"
        fi
    done
}

# Check base model
check_base_model() {
    print_header "Checking Base Model"

    local base_model="$PROJECT_ROOT/models/final_model.pt"

    if [ -f "$base_model" ]; then
        print_msg "$GREEN" "✓ Base model found: $base_model"
        return 0
    else
        print_msg "$YELLOW" "⚠ Base model not found: $base_model"
        print_msg "$YELLOW" ""
        print_msg "$YELLOW" "You need a pre-trained base model for LoRA fine-tuning."
        print_msg "$YELLOW" "Options:"
        print_msg "$YELLOW" "  1. Train a new base model:"
        print_msg "$YELLOW" "     cd $PROJECT_ROOT && uv run python src/train_model.py"
        print_msg "$YELLOW" "  2. Specify existing model path in lora_config.yaml:"
        print_msg "$YELLOW" "     lora.pretrained_checkpoint: '/path/to/your/model.pt'"
        echo ""
        return 1
    fi
}

# Configure LoRA
configure_lora() {
    print_header "Configuring LoRA"

    local config="$SCRIPT_DIR/lora_config.yaml"

    print_msg "$YELLOW" "Current LoRA configuration:"
    echo ""

    # Extract key settings from config
    if [ -f "$config" ]; then
        print_msg "$BLUE" "Rank:          $(grep '^  rank:' "$config" | awk '{print $2}')"
        print_msg "$BLUE" "Alpha:         $(grep '^  alpha:' "$config" | awk '{print $2}')"
        print_msg "$BLUE" "Learning rate: $(grep '^  learning_rate:' "$config" | grep -v '#' | awk '{print $2}' | head -1)"
        print_msg "$BLUE" "Num epochs:    $(grep '^  num_epochs:' "$config" | awk '{print $2}')"
        echo ""
        print_msg "$GREEN" "✓ Config file: $config"
    else
        print_msg "$RED" "✗ Config file not found: $config"
        return 1
    fi

    echo ""
    read -p "Edit config now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} "$config"
    fi
}

# Show next steps
show_next_steps() {
    print_header "Setup Complete!"

    print_msg "$GREEN" "You're ready to start LoRA fine-tuning!"
    echo ""
    print_msg "$BLUE" "Next steps:"
    echo ""
    print_msg "$YELLOW" "1. Review/edit the configuration:"
    print_msg "$GREEN" "   nano $SCRIPT_DIR/lora_config.yaml"
    echo ""
    print_msg "$YELLOW" "2. Start fine-tuning:"
    print_msg "$GREEN" "   cd $PROJECT_ROOT"
    print_msg "$GREEN" "   uv run bash lora/run_lora.sh train"
    echo ""
    print_msg "$YELLOW" "3. Or use the Python script directly:"
    print_msg "$GREEN" "   uv run python $SCRIPT_DIR/train_lora.py"
    echo ""
    print_msg "$YELLOW" "4. Compare models after training:"
    print_msg "$GREEN" "   uv run bash lora/run_lora.sh compare"
    echo ""
    print_msg "$YELLOW" "5. Merge LoRA weights for inference:"
    print_msg "$GREEN" "   uv run bash lora/run_lora.sh merge"
    echo ""
    print_msg "$BLUE" "For more information:"
    print_msg "$GREEN" "   cat $SCRIPT_DIR/LORA_FINETUNING.md"
    print_msg "$GREEN" "   uv run bash lora/run_lora.sh help"
    echo ""
}

# Main setup flow
main() {
    print_header "LoRA Fine-Tuning Setup"

    check_python
    setup_directories
    check_base_model || true
    configure_lora
    show_next_steps

    print_msg "$GREEN" "Setup script finished!"
    echo ""
}

main "$@"

#!/bin/bash
#
# LoRA Fine-Tuning Runner Script
#
# This script provides an easy interface for running LoRA fine-tuning tasks.
#
# Usage:
#   ./run_lora.sh train                    # Train with default config
#   ./run_lora.sh train custom.yaml        # Train with custom config
#   ./run_lora.sh compare                  # Compare base and LoRA models
#   ./run_lora.sh merge                    # Merge LoRA weights
#   ./run_lora.sh analyze                  # Analyze LoRA statistics
#   ./run_lora.sh extract                  # Extract LoRA weights only
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths
DEFAULT_CONFIG="$SCRIPT_DIR/lora_config.yaml"
BASE_MODEL="$PROJECT_ROOT/models/best_model.pt"
LORA_MODEL="$PROJECT_ROOT/models/final_lora_model.pt"
MERGED_MODEL="$PROJECT_ROOT/models/final_merged_model.pt"

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print header
print_header() {
    echo ""
    print_msg "$BLUE" "=================================="
    print_msg "$BLUE" "$1"
    print_msg "$BLUE" "=================================="
    echo ""
}

# Check if file exists
check_file() {
    local file=$1
    local description=$2

    if [ ! -f "$file" ]; then
        print_msg "$RED" "Error: $description not found at: $file"
        return 1
    fi
    return 0
}

# Train command
cmd_train() {
    print_header "LoRA Fine-Tuning"

    local config="${1:-$DEFAULT_CONFIG}"

    # Check if config exists
    if ! check_file "$config" "Config file"; then
        print_msg "$YELLOW" "Using default config: $DEFAULT_CONFIG"
        config="$DEFAULT_CONFIG"
    fi

    # Check if base model exists
    if ! check_file "$BASE_MODEL" "Base model"; then
        print_msg "$YELLOW" "Warning: Base model not found!"
        print_msg "$YELLOW" "Make sure to set 'lora.pretrained_checkpoint' in your config"
        print_msg "$YELLOW" "Or train a base model first with: cd .. && python local_geometry_gcn_stereo.py"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    print_msg "$GREEN" "Config: $config"
    print_msg "$GREEN" "Starting LoRA fine-tuning..."
    echo ""

    # Set config path and run
    export CONFIG_FILE="$config"
    cd "$PROJECT_ROOT"
    python "$SCRIPT_DIR/local_geometry_gcn_stereo_lora.py"

    print_msg "$GREEN" "✓ Training complete!"
}

# Compare command
cmd_compare() {
    print_header "Compare Models"

    local base="${1:-$BASE_MODEL}"
    local lora="${2:-$LORA_MODEL}"

    check_file "$base" "Base model" || exit 1
    check_file "$lora" "LoRA model" || exit 1

    print_msg "$GREEN" "Comparing models..."
    python "$SCRIPT_DIR/lora_utils.py" compare --base "$base" --lora "$lora"
}

# Merge command
cmd_merge() {
    print_header "Merge LoRA Weights"

    local lora="${1:-$LORA_MODEL}"
    local output="${2:-$MERGED_MODEL}"

    check_file "$lora" "LoRA model" || exit 1

    print_msg "$GREEN" "Input:  $lora"
    print_msg "$GREEN" "Output: $output"
    echo ""

    python "$SCRIPT_DIR/lora_utils.py" merge --lora "$lora" --output "$output"

    print_msg "$GREEN" "✓ Merge complete!"
}

# Analyze command
cmd_analyze() {
    print_header "Analyze LoRA Statistics"

    local lora="${1:-$LORA_MODEL}"

    check_file "$lora" "LoRA model" || exit 1

    python "$SCRIPT_DIR/lora_utils.py" analyze --lora "$lora"
}

# Extract command
cmd_extract() {
    print_header "Extract LoRA Weights"

    local lora="${1:-$LORA_MODEL}"
    local output="${2:-$PROJECT_ROOT/models/lora_weights_only.pt}"

    check_file "$lora" "LoRA model" || exit 1

    print_msg "$GREEN" "Input:  $lora"
    print_msg "$GREEN" "Output: $output"
    echo ""

    python "$SCRIPT_DIR/lora_utils.py" extract --lora "$lora" --output "$output"

    print_msg "$GREEN" "✓ Extraction complete!"
}

# Help command
cmd_help() {
    cat << EOF
LoRA Fine-Tuning Runner Script

Usage: $0 <command> [arguments]

Commands:
  train [config.yaml]           Train with LoRA (default: lora_config.yaml)
  compare [base] [lora]         Compare base and LoRA models
  merge [lora] [output]         Merge LoRA weights into base model
  analyze [lora]                Analyze LoRA weight statistics
  extract [lora] [output]       Extract LoRA weights only
  help                          Show this help message

Examples:
  # Train with default config
  $0 train

  # Train with custom config
  $0 train my_config.yaml

  # Compare models
  $0 compare models/base.pt models/lora.pt

  # Merge LoRA weights
  $0 merge models/lora.pt models/merged.pt

  # Analyze LoRA statistics
  $0 analyze models/lora.pt

  # Extract LoRA weights
  $0 extract models/lora.pt models/lora_only.pt

Default Paths:
  Config:       $DEFAULT_CONFIG
  Base model:   $BASE_MODEL
  LoRA model:   $LORA_MODEL
  Merged model: $MERGED_MODEL

For more information, see LORA_FINETUNING.md

EOF
}

# Main command dispatcher
main() {
    local command="${1:-help}"
    shift || true

    case "$command" in
        train)
            cmd_train "$@"
            ;;
        compare)
            cmd_compare "$@"
            ;;
        merge)
            cmd_merge "$@"
            ;;
        analyze)
            cmd_analyze "$@"
            ;;
        extract)
            cmd_extract "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            print_msg "$RED" "Error: Unknown command '$command'"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"

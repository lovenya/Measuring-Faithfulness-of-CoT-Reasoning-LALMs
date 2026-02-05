#!/bin/bash
# generate_all_masked_datasets.sh
#
# Script to generate all pre-processed masked audio datasets.
# Run from project root with an activated environment.
#
# Usage:
#   source qwen_env/bin/activate
#   bash scripts/generate_all_masked_datasets.sh
#
# This will generate 6 variants (2 mask types × 3 modes) × 5 datasets = 30 dataset variants
# Each with 10 percentile levels (10, 20, ..., 100)

set -e

# Defaults - can be overridden
LEVELS="10 20 30 40 50 60 70 80 90 100"
SEED=42
WORKERS=${1:-8}  # Default to 8 workers, or pass as argument

echo "=============================================="
echo "Generating Masked Audio Datasets"
echo "Levels: $LEVELS"
echo "Seed: $SEED"
echo "Workers: $WORKERS"
echo "=============================================="

# Function to generate masked dataset
generate_masked() {
    local SOURCE=$1
    local OUTPUT=$2
    local MASK_TYPE=$3
    local MODE=$4
    
    echo ""
    echo ">>> Generating: $OUTPUT (${MASK_TYPE}_${MODE})"
    python data_processing/mask_audio_dataset.py \
        --source "$SOURCE" \
        --output "$OUTPUT" \
        --mask-type "$MASK_TYPE" \
        --mode "$MODE" \
        --levels $LEVELS \
        --seed $SEED \
        --workers $WORKERS \
        --verbose
}

# ===== MMAR =====
for MASK_TYPE in silence noise; do
    for MODE in random start end; do
        generate_masked "data/mmar" "data/mmar_masked" "$MASK_TYPE" "$MODE"
    done
done

# ===== SAKURA subdatasets =====
for SUBDATASET in animal emotion gender language; do
    for MASK_TYPE in silence noise; do
        for MODE in random start end; do
            generate_masked \
                "data/sakura/${SUBDATASET}" \
                "data/sakura/${SUBDATASET}_masked" \
                "$MASK_TYPE" \
                "$MODE"
        done
    done
done

echo ""
echo "=============================================="
echo "All masked datasets generated successfully!"
echo "=============================================="

#!/bin/bash
# Run probe evaluation across multiple layers to compare performance

set -e

# Data paths
export HCPYA_ROOT="${HCPYA_ROOT:-/teamspace/studios/this_studio/HCPYA_processed}"

# Configuration
MODEL="${MODEL:-flat_mae_base_patch16_16}"
REPR="${REPR:-patch}"
CLASSIFIER="${CLASSIFIER:-linear}"
DATASET="${DATASET:-hcpya_rest1lr_age}"

# Training settings
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
LR="${LR:-1e-3}"
WANDB="${WANDB:-false}"

# Layers to test: early (2), middle (6), last (11)
LAYERS="${LAYERS:-2 6 11}"

# Move to repo root
cd "$(dirname "$0")/.."

for LAYER in $LAYERS; do
    NAME="eval_layer${LAYER}"

    echo "========================================"
    echo "Running probe evaluation for layer $LAYER"
    echo "========================================"
    echo "  Name: $NAME"
    echo "  Model: $MODEL"
    echo "  Layer: $LAYER"
    echo "  Representation: $REPR"
    echo "  Classifier: $CLASSIFIER"
    echo "  Dataset: $DATASET"
    echo ""

    uv run python -m fmri_fm_eval.main_probe \
        "$MODEL" \
        "$REPR" \
        "$CLASSIFIER" \
        "$DATASET" \
        --overrides \
            name_prefix="$NAME" \
            model_kwargs.layer="$LAYER" \
            epochs="$EPOCHS" \
            batch_size="$BATCH_SIZE" \
            steps_per_epoch="$STEPS_PER_EPOCH" \
            lr="$LR" \
            wandb="$WANDB"

    echo ""
done

echo "All layer evaluations complete!"

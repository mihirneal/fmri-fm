#!/bin/bash
set -euo pipefail

# NSD eval: flat_mae_base_patch16_2 only.
# Task: nsd_cococlip | Representation: patch | Classifier: attn
# Uses architecture-matched default pretrained weights from model loader.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="experiments/eval_v1/output"
DATASET="nsd_cococlip"
NSD_ROOT="/teamspace/studios/this_studio/HCPYA_processed"
DEVICE="${DEVICE:-cuda:3}"
WANDB="${WANDB:-false}"
RUN_NAME="${RUN_NAME:-flatmae_patch16_2/nsdcoco}"
EPOCHS="${EPOCHS:-4}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_ITER="${ACCUM_ITER:-1}"
BASE_LR="${BASE_LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"

NSD_ROOT="${NSD_ROOT}" uv run python -m fmri_fm_eval.main_probe \
    flat_mae_base_patch16_2 \
    patch \
    attn \
    ${DATASET} \
    --overrides \
    "name=${RUN_NAME}" \
    wandb=${WANDB} \
    debug=false \
    device=${DEVICE} \
    epochs=${EPOCHS} \
    steps_per_epoch=${STEPS_PER_EPOCH} \
    warmup_epochs=${WARMUP_EPOCHS} \
    batch_size=${BATCH_SIZE} \
    accum_iter=${ACCUM_ITER} \
    base_lr=${BASE_LR} \
    weight_decay=${WEIGHT_DECAY} \
    lr_scale_grid=[0.03,0.1,0.3,1.0,3.0,10.0,30.0] \
    wd_scale_grid=[0.03,0.1,0.3,1.0,3.0,10.0,30.0] \
    num_workers=${NUM_WORKERS} \
    prefetch_factor=${PREFETCH_FACTOR} \
    output_root=${OUT_DIR}

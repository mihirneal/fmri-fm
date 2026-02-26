#!/bin/bash
set -euo pipefail

# Checkpoint sweep: run NSD cococlip + HCP-YA task21 evals on every checkpoint
# in a directory, then collate results into summary tables.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-checkpoints/ukbb_fulltar_pretrain}"
OUT_DIR="experiments/eval_v1/output"
NSD_ROOT="${NSD_ROOT:-/teamspace/studios/this_studio/eval-set}"
HCPYA_ROOT="${HCPYA_ROOT:-/teamspace/studios/this_studio/eval-set}"
DEVICE="${DEVICE:-cuda:3}"
WANDB="${WANDB:-false}"
DATASET="${DATASET:-nsd_cococlip}"   # nsd_cococlip or hcpya_task21
SWEEP_NAME="${SWEEP_NAME:-ukbb_fulltar_pretrain/${DATASET}}"
CLASSIFIER="${CLASSIFIER:-attn}"    # linear or attn

# ── Discover checkpoints (sorted numerically) ──────────────────────────
mapfile -t CKPTS < <(ls "${CKPT_DIR}"/checkpoint-*.pth 2>/dev/null | sort -V)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "ERROR: No checkpoint-*.pth files found in ${CKPT_DIR}"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints in ${CKPT_DIR}:"
printf '  %s\n' "${CKPTS[@]}"
echo ""
echo "Dataset: ${DATASET}"
echo ""

# ── Run evals sequentially ──────────────────────────────────────────────
for CKPT_PATH in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "${CKPT_PATH}" .pth)       # e.g. checkpoint-00010
    EPOCH_TAG="${CKPT_NAME#checkpoint-}"              # e.g. 00010 or last
    RUN_NAME="${SWEEP_NAME}/${EPOCH_TAG}"

    echo "══════════════════════════════════════════════════════════════"
    echo "  ${DATASET} | ${CKPT_NAME}  →  ${RUN_NAME}"
    echo "══════════════════════════════════════════════════════════════"

    NSD_ROOT="${NSD_ROOT}" HCPYA_ROOT="${HCPYA_ROOT}" \
    uv run python -m fmri_fm_eval.main_probe \
        flat_mae \
        patch \
        ${CLASSIFIER} \
        ${DATASET} \
        --overrides \
        "name=${RUN_NAME}" \
        "model_kwargs.ckpt_path=${CKPT_PATH}" \
        wandb=${WANDB} \
        debug=false \
        device=${DEVICE} \
        epochs=4 \
        steps_per_epoch=500 \
        warmup_epochs=2 \
        batch_size=128 \
        base_lr=0.001 \
        weight_decay=0.05 \
        lr_scale_grid=[0.03,0.1,0.3,1.0,3.0,10.0,30.0] \
        wd_scale_grid=[0.03,0.1,0.3,1.0,3.0,10.0,30.0] \
        num_workers=16 \
        prefetch_factor=8 \
        output_root=${OUT_DIR}

    echo ""
done

# ── Collate results ─────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Summary"
echo "══════════════════════════════════════════════════════════════"

uv run python3 -c "
import json, glob, os, sys

out_dir = '${OUT_DIR}/${SWEEP_NAME}'
logs = sorted(glob.glob(os.path.join(out_dir, '*/eval_log.json')))

print()
print(f'### ${DATASET}')
if not logs:
    print(f'  No eval_log.json files found under {out_dir}')
    sys.exit(0)

header = '| checkpoint | test/acc | test/f1 | val/acc | val/f1 | best_lr | best_wd |'
sep    = '|------------|----------|---------|---------|--------|---------|---------|'
print(header)
print(sep)

for path in logs:
    tag = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        d = json.load(f)
    print('| {tag:>10s} | {tacc:8.4f} | {tf1:7.4f} | {vacc:7.4f} | {vf1:6.4f} | {lr:7.4f} | {wd:7.5f} |'.format(
        tag=tag,
        tacc=d.get('eval/test/acc', 0),
        tf1=d.get('eval/test/f1', 0),
        vacc=d.get('eval/validation/acc', 0),
        vf1=d.get('eval/validation/f1', 0),
        lr=d.get('eval/lr_best', 0),
        wd=d.get('eval/wd_best', 0),
    ))
"

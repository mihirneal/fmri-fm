#!/bin/bash
set -euo pipefail

# Checkpoint sweep: run NSD cococlip + HCP-YA task21 evals on every checkpoint
# in a directory, then collate results into summary tables.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-checkpoints/ukbb_baseline}"
OUT_DIR="${OUT_DIR:-experiments/eval_v1/output}"
NSD_ROOT="${NSD_ROOT:-/teamspace/studios/ukbb-pretrain/eval-set}"
HCPYA_ROOT="${HCPYA_ROOT:-/teamspace/studios/ukbb-pretrain/eval-set}"
AABC_ROOT="${AABC_ROOT:-/teamspace/studios/this_studio/fmri-fm-eval/datasets/AABC/data/processed}"
DEVICE="${DEVICE:-cuda:7}"
WANDB="${WANDB:-false}"
DATASET="${DATASET:-hcpya_task21}"   # nsd_cococlip or hcpya_task21
# DATASET="${DATASET:-nsd_cococlip}"   # nsd_cococlip or hcpya_task21
SWEEP_NAME="${SWEEP_NAME:-ukbb_baseline/${DATASET}}"
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

    # if [[ -f "${OUT_DIR}/${RUN_NAME}/eval_log.json" ]]; then
    #     echo "  Skipping ${EPOCH_TAG} (already done)"
    #     continue
    # fi

    echo "══════════════════════════════════════════════════════════════"
    echo "  ${DATASET} | ${CKPT_NAME}  →  ${RUN_NAME}"
    echo "══════════════════════════════════════════════════════════════"

    NSD_ROOT="${NSD_ROOT}" HCPYA_ROOT="${HCPYA_ROOT}" AABC_ROOT="${AABC_ROOT}" \
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
        epochs=10 \
        steps_per_epoch=200 \
        warmup_epochs=4 \
        batch_size=64 \
        accum_iter=2 \
        lr=0.001 \
        weight_decay=0.05 \
        cv_metric=acc \
        "lr_scale_grid=[0.02,0.023,0.028,0.033,0.038,0.045,0.053,0.062,0.074,0.087,0.1,0.12,0.14,0.17,0.2,0.23,0.27,0.32,0.38,0.44,0.52,0.61,0.72,0.85,1.0,1.2,1.4,1.6,1.9,2.3,2.7,3.1,3.7,4.3,5.1,6.0,7.1,8.3,9.8,12.0,14.0,16.0,19.0,22.0,26.0,31.0,36.0,43.0,50.0]" \
        wd_scale_grid=[1.0] \
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

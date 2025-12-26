#!/bin/bash

if [[ -z $1 || $1 == "-h" || $1 == "--help" ]]; then
    echo "launch_eval.sh MODELID TASKID"
    exit
fi

MODELID=$1
TASKID=$2

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="eval_v1"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

# fill with the name of your home folder on lightning
# SHARE_USER=${SHARE_USER:-volunteer}
# SHARE_DIR="/teamspace/gcs_folders/share/fmri-fm/${SHARE_USER}"

# # save output to persistent shared storage
# SHARE_OUT_DIR="${SHARE_DIR}/${OUT_DIR}"
# mkdir -p ${SHARE_OUT_DIR} 2>/dev/null
# ln -sn ${SHARE_OUT_DIR} ${OUT_DIR} 2>/dev/null

# download data locally
export HCPYA_ROOT="/teamspace/studios/this_studio/HCPYA_processed"
export BRAINLM_COORDS_DATASET_PATH="/teamspace/gcs_folders/share/fmri-fm/BrainLM/Brain_Region_Coordinates"
export BRAINLM_REPO_PATH="/teamspace/studios/this_studio/BrainLM"
mkdir -p $HCPYA_ROOT
aws s3 sync s3://medarc/fmri-fm-eval/processed $HCPYA_ROOT \
    --exclude '*' \
    --include 'hcpya-rest1lr.flat.arrow*' \
    --include 'hcpya-rest1lr.schaefer400.arrow*' \
    --include 'hcpya-rest1lr.a424.arrow*' \
    --include 'targets*'

models=(
    connectome_schaefer400
    flat_mae_base_patch16_16
    flat_mae_base_patch16_16
    flat_mae_base_patch16_16
    flat_mae_base_patch16_2
    flat_mae_base_patch16_2
    flat_mae_base_patch16_2
    brainlm_13m
    brainlm_13m
    brainlm_vitmae_111m
    brainlm_vitmae_111m
    brainlm_vitmae_650m
    brainlm_vitmae_650m
)
reprs=(
    cls
    cls
    avg_patch
    patch
    cls
    avg_patch
    patch
    cls
    avg_patch
    cls
    avg_patch
    cls
    avg_patch
)
model=${models[MODELID]}
repr=${reprs[MODELID]}

datasets=(
    hcpya_rest1lr_gender
    hcpya_rest1lr_age
    hcpya_rest1lr_flanker
    hcpya_rest1lr_neofacn
    hcpya_rest1lr_pmat24
)
dataset=${datasets[TASKID]}
name="test_bs8"
overrides="representation=${repr} name=${name} epochs=10 batch_size=8 lr=0.001 num_workers=16"
overrides="${overrides} debug=false wandb=false device=cuda:0"
if [[ $model == brainlm_* ]]; then
    if [[ -z ${BRAINLM_COORDS_DATASET_PATH:-} ]]; then
        echo "BRAINLM_COORDS_DATASET_PATH is required for BrainLM models."
        exit 1
    fi
    if [[ -z ${BRAINLM_REPO_PATH:-} ]]; then
        echo "BRAINLM_REPO_PATH is required for BrainLM models."
        exit 1
    fi
    overrides="${overrides} model_kwargs.coords_dataset_path=${BRAINLM_COORDS_DATASET_PATH}"
fi

notes="initial probe eval run (${model}/${repr}/${dataset})."

uv run python -m fmri_fm_eval.main_probe \
    $model \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    notes="${notes}" \
    $overrides

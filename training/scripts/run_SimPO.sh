#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -N run_SimPO
#PBS -m abe

set -euxo pipefail

# ========== 基本パス ==========
GROUP_NAME=""
TRAINING_DIR=$HOME/efficient-deep-research/training

LOG_DIR="$TRAINING_DIR/logs"
OUTPUT_DIR="$TRAINING_DIR/output/$PBS_JOBID"
ENV_FILE="$TRAINING_DIR/.env"
SIF_PATH="$TRAINING_DIR/container/ms-swift_container.sif"
CONFIG_FILE="$TRAINING_DIR/config.json"

# ========== wandb ==========
export SINGULARITYENV_WANDB_PROJECT=""
export SINGULARITYENV_WANDB_RUN_NAME=""
export SINGULARITYENV_WANDB_DIR=""
mkdir -p "$SINGULARITYENV_WANDB_DIR"

# ========== ログ ==========
LOG_FILE="$LOG_DIR/$PBS_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Error at line $LINENO, exit status $?"' ERR

# ========== JSONから設定を読み込み ==========
readarray -t TRAIN_ARGS < <(
  jq -r 'to_entries[] | "--\(.key) \(.value)"' "$CONFIG_FILE"
)
TRAIN_ARGS+=(--output_dir "$OUTPUT_DIR" --report_to wandb)

# ========== 実行 ==========
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
nvidia-smi

singularity exec \
    --nv \
    --network host \
    --writable-tmpfs \
    --env-file "$ENV_FILE" \
    --bind /groups/$GROUP_NAME/share:/groups/$GROUP_NAME/share \
    "$SIF_PATH" \
    swift rlhf "${TRAIN_ARGS[@]}"

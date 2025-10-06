#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -k oe
#PBS -N run_DPO
#PBS -m abe

set -euxo pipefail

# ========== 基本パス ==========
source $PBS_O_WORKDIR/scripts/config.sh

# ========== wandb ==========
mkdir -p "$SINGULARITYENV_WANDB_DIR"

# ========== ログ ==========
LOG_FILE="$LOG_DIR/$PBS_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Error at line $LINENO, exit status $?"' ERR

# ========== JSONから設定を読み込み ==========
readarray -t TRAIN_ARGS < <(
  jq -r 'to_entries[] | "--\(.key)\n\(.value)"' "$CONFIG_FILE"
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
    --bind /groups/$GROUP_NAME/share:/groups/$GROUP_NAME/share \
    "$PBS_O_WORKDIR/container/$SIF_NAME" \
    python $PBS_O_WORKDIR/src/train_DPO.py "${TRAIN_ARGS[@]}"

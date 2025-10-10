#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P gcd50664
#PBS -k oe
#PBS -N run_DPO
#PBS -m abe

set -euxo pipefail

# ========== 基本パス ==========
source $PBS_O_WORKDIR/scripts/config.sh

# ========== wandb ==========
export WANDB_PROJECT="efficient-deep-research"
export WANDB_NAME="$PBS_JOBID"

# ========== dataset ==========
DATASET_NAME="preferences_data_example1000.jsonl"

# ========== ログ ==========
LOG_FILE="$LOG_DIR/$PBS_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Error at line $LINENO, exit status $?"' ERR

# ========== JSONから設定を読み込み ==========
readarray -t TRAIN_ARGS < <(
  jq -r 'to_entries[] | "--\(.key)\n\(.value)"' "$CONFIG_FILE"
)
TRAIN_ARGS+=(--output_dir "$OUTPUT_DIR" --report_to wandb --dataset "$PBS_O_WORKDIR/data/$DATASET_NAME" --use_hf true)

# ========== 実行 ==========
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PBS_O_WORKDIR/download"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
nvidia-smi

singularity exec \
    --nv \
    --network host \
    --writable-tmpfs \
    --bind $PBS_O_WORKDIR/download:/mnt/workspace \
    "$PBS_O_WORKDIR/container/$SIF_NAME" \
    python $PBS_O_WORKDIR/src/train_DPO.py "${TRAIN_ARGS[@]}"

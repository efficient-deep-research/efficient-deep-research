#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gcd50664
#PBS -k oe
#PBS -N run_score_gap_0_5_lora
#PBS -m abe
#PBS -J 1-2
case $PBS_ARRAY_INDEX in
    1) BETA=0.01 ;;
    2) BETA=0.5 ;;
esac

set -euxo pipefail

# ========== 基本パス ==========
source $PBS_O_WORKDIR/scripts/config.sh

# ========== wandb ==========
export WANDB_PROJECT="efficient-deep-research"
export WANDB_NAME="$PBS_JOBID"

# ========== dataset ==========
DATASET_NAME="train_0_5_processed.jsonl"

# ========== ログ ==========
LOG_FILE="$LOG_DIR/$PBS_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Error at line $LINENO, exit status $?"' ERR

# ========== JSONから設定を読み込み ==========
readarray -t TRAIN_ARGS < <(
  jq -r 'to_entries[] | "--\(.key)\n\(.value)"' "$CONFIG_FILE"
)
TRAIN_ARGS+=(--output_dir "$OUTPUT_DIR")
TRAIN_ARGS+=(--report_to wandb)
TRAIN_ARGS+=(--dataset "$PBS_O_WORKDIR/data/$DATASET_NAME")
TRAIN_ARGS+=(--use_hf true)

# ========== Training meta parameters ==========
TRAIN_ARGS+=(--num_train_epochs 1)
TRAIN_ARGS+=(--per_device_train_batch_size 1)
TRAIN_ARGS+=(--per_device_eval_batch_size 1)
TRAIN_ARGS+=(--max_length 28160)
# TRAIN_ARGS+=(--deepspeed zero2)
TRAIN_ARGS+=(--attn_impl flash_attention_2)

# ========== DPO hyperparameters ==========
TRAIN_ARGS+=(--learning_rate 1e-4)
TRAIN_ARGS+=(--gradient_accumulation_steps 16)
TRAIN_ARGS+=(--warmup_ratio 0.05)
TRAIN_ARGS+=(--beta $BETA)

TRAIN_ARGS+=(--run_name run_score_gap_0_5_lora_beta_$BETA)


# ========== resume from checkpoint ==========
TRAIN_ARGS+=(--resume_from_checkpoint $PBS_O_WORKDIR/output/1278795.pbs1/v0-20251022-145638/checkpoint-40)

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

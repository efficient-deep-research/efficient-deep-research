#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=5:00:00
#PBS -P gcd50664
#PBS -k oe
#PBS -m abe

# NOTE: SETUP .env FILE BEFORE RUNNING THIS SCRIPT

# ========== ログ ==========
LOG_FILE="$PBS_O_WORKDIR/logs/$PBS_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Error at line $LINENO, exit status $?"' ERR


# ========== HF TOKEN ==========
if [ -f "$PBS_O_WORKDIR/.env" ]; then
    set -a
    source "$PBS_O_WORKDIR/.env"
    set +a
else
    echo ".env file not found"
    exit 1
fi

# FORMATTING: JOBID.pbs1/v0-YYYYMMDD-HHMMSS/checkpoint-:CKPT:BETA:GAP
CHECKPOINTS="
1284231[1].pbs1/v0-20251023-180133/checkpoint-:10:0.5:0.7
1284231[1].pbs1/v0-20251023-180133/checkpoint-:20:0.5:0.7
1284231[2].pbs1/v0-20251023-180125/checkpoint-:10:0.5:0.3
1284231[2].pbs1/v0-20251023-180125/checkpoint-:56:0.5:0.3
"


for item in $CHECKPOINTS; do
    JOBID=$(echo "$item" | cut -d: -f1)
    CKPT=$(echo "$item" | cut -d: -f2)
    BETA=$(echo "$item" | cut -d: -f3)
    GAP=$(echo "$item" | cut -d: -f4)

    echo "Merging and pushing checkpoint ${JOBID}${CKPT}"
    ls -la $PBS_O_WORKDIR/output/${JOBID}${CKPT}
    echo "HF_TOKEN: $HF_TOKEN"
    singularity exec \
        --nv \
        --network host \
        --writable-tmpfs \
        --bind $PBS_O_WORKDIR/download:/mnt/workspace \
        "$PBS_O_WORKDIR/container/ms-swift_container.sif" \
        swift export \
            --adapters "$PBS_O_WORKDIR/output/${JOBID}${CKPT}" \
            --merge_lora True \
            --push_to_hub true \
            --use_hf true \
            --exist_ok True \
            --hub_model_id "efficient-deep-research/gap_${GAP}_beta_${BETA}_lora_ckpt_${CKPT}_merged" \
            --hub_token "$HF_TOKEN"

    rm -rf "$PBS_O_WORKDIR/output/${JOBID}${CKPT}-merged"
    echo "Removed checkpoint ${JOBID}${CKPT}-merged"
done
#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gcd50664
#PBS -k oe
#PBS -m abe

# ========== ログ ==========
LOG_FILE="$PBS_O_WORKDIR/logs/$PBS_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Error at line $LINENO, exit status $?"' ERR

echo "Merging and pushing checkpoint 1278795.pbs1/v0-20251022-145638/checkpoint-30"
singularity exec \
    --nv \
    --network host \
    --writable-tmpfs \
    --bind $PBS_O_WORKDIR/download:/mnt/workspace \
    "$PBS_O_WORKDIR/container/ms-swift_container.sif" \
    swift export \
        --adapters $PBS_O_WORKDIR/output/1278795.pbs1/v0-20251022-145638/checkpoint-30 \
        --merge_lora True \
        --push_to_hub true \
        --use_hf true \
        --exist_ok True \
        --hub_model_id "efficient-deep-research/gap_0_5_lora_ckpt_30_merged" \
        --hub_token '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'

rm -rf $PBS_O_WORKDIR/output/1278795.pbs1/v0-20251022-145638/checkpoint-30-merged
echo "Removed checkpoint 1278795.pbs1/v0-20251022-145638/checkpoint-30-merged"

echo "Merging and pushing checkpoint 1278795.pbs1/v0-20251022-145638/checkpoint-40"
singularity exec \
    --nv \
    --network host \
    --writable-tmpfs \
    --bind $PBS_O_WORKDIR/download:/mnt/workspace \
    "$PBS_O_WORKDIR/container/ms-swift_container.sif" \
    swift export \
        --adapters $PBS_O_WORKDIR/output/1278795.pbs1/v0-20251022-145638/checkpoint-40 \
        --merge_lora True \
        --push_to_hub true \
        --use_hf true \
        --exist_ok True \
        --hub_model_id "efficient-deep-research/gap_0_5_lora_ckpt_40_merged" \
        --hub_token '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'

rm -rf $PBS_O_WORKDIR/output/1278795.pbs1/v0-20251022-145638/checkpoint-40-merged
echo "Removed checkpoint 1278795.pbs1/v0-20251022-145638/checkpoint-40-merged"

echo "Merging and pushing checkpoint 1280112.pbs1/v0-20251022-231318/checkpoint-47"
singularity exec \
    --nv \
    --network host \
    --writable-tmpfs \
    --bind $PBS_O_WORKDIR/download:/mnt/workspace \
    "$PBS_O_WORKDIR/container/ms-swift_container.sif" \
    swift export \
        --adapters $PBS_O_WORKDIR/output/1280112.pbs1/v0-20251022-231318/checkpoint-47 \
        --merge_lora True \
        --push_to_hub true \
        --use_hf true \
        --exist_ok True \
        --hub_model_id "efficient-deep-research/gap_0_5_lora_ckpt_47_merged" \
        --hub_token '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'

rm -rf $PBS_O_WORKDIR/output/1280112.pbs1/v0-20251022-231318/checkpoint-47-merged
echo "Removed checkpoint 1280112.pbs1/v0-20251022-231318/checkpoint-47-merged"
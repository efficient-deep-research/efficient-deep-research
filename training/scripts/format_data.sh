#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=5:00
#PBS -j oe
#PBS -N data_format
#PBS -m ae

set -euxo pipefail

# ========== 基本パス ==========
source $PBS_O_WORKDIR/scripts/config.sh

DATASET_NAME="train_0_7.jsonl"
OUTPUT_NAME="train_0_7_processed.jsonl"

singularity exec \
    --nv \
    --network host \
    --writable-tmpfs \
    "$PBS_O_WORKDIR/container/$SIF_NAME" \
    python $PBS_O_WORKDIR/src/format_data.py \
        --dataset_name "$PBS_O_WORKDIR/data/$DATASET_NAME" \
        --output_file "$PBS_O_WORKDIR/data/$OUTPUT_NAME"

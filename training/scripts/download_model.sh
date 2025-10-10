#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N download_model
#PBS -m abe

set -euxo pipefail

# =========== Path setup ===========
source $PBS_O_WORKDIR/scripts/config.sh
SINGULARITYENV_MASTER_PORT=$((29500 + RANDOM % 1000))
# ==================================

singularity exec --nv \
                 --writable-tmpfs \
                 --bind $PBS_O_WORKDIR/download:/mnt/workspace \
                 $PBS_O_WORKDIR/container/$SIF_NAME \
                 python $PBS_O_WORKDIR/src/download_model.py
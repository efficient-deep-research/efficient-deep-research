#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N download_model
#PBS -m abe

set -euxo pipefail

# =========== Path setup ===========
GROUP_NAME=""
TRAINING_DIR=$HOME/efficient-deep-research/training

SIF_PATH="$TRAINING_DIR/container/ms-swift_container.sif"
SINGULARITYENV_MASTER_PORT=$((29500 + RANDOM % 1000))
# ==================================

singularity exec --nv \
                 --writable-tmpfs \
                 --bind /groups/$GROUP_NAME/share:/groups/$GROUP_NAME/share \
                 $SIF_PATH \
                 python $TRAINING_DIR/src/download_model.py

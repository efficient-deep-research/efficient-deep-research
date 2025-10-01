#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N build_ms-swift_container
#PBS -m abe

set -euxo pipefail

# ========== 基本パス ==========
TRAINING_DIR=$HOME/efficient-deep-research/training
CONTAINER_DIR="$TRAINING_DIR/container"
DEF_FILE="ms-swift_container.def"
SIF_FILE="ms-swift_container.sif"

# ========== コンテナビルド ==========
mkdir -p $CONTAINER_DIR
singularity build --fakeroot "$CONTAINER_DIR/$SIF_FILE" "$CONTAINER_DIR/$DEF_FILE"

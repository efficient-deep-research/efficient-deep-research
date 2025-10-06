#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N build_ms-swift_container
#PBS -m abe

set -euxo pipefail

# ========== setup ==========
source $PBS_O_WORKDIR/scripts/config.sh
CONTAINER_DIR="$PBS_O_WORKDIR/container"

# ========== build container ==========
singularity build --fakeroot "$CONTAINER_DIR/$SIF_NAME" "$CONTAINER_DIR/$DEF_NAME"

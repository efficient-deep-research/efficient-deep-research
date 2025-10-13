#!/bin/bash
set -eux

MODEL=Qwen/Qwen3-Next-80B-A3B-Thinking
CACHE_DIR=vllm_server/huggingface_cache

# SingularityでvLLMイメージ内のPython環境を利用
singularity exec \
  --bind ${CACHE_DIR}:/root/.cache/huggingface \
  vllm_server/vllm/vllm-openai_latest.sif \
  huggingface-cli download $MODEL --local-dir /root/.cache/huggingface/${MODEL}

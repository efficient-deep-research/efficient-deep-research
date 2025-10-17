#!/bin/bash
set -euxo pipefail

MODEL_PATH="/home/ko-yoshida/sftp_sync/mmu-rag/vllm_server/huggingface_cache/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"

# GPU 4-7 を使用
export CUDA_VISIBLE_DEVICES=4,5,6,7

singularity exec --nv \
   --bind ~/.cache/huggingface:/root/.cache/huggingface \
   vllm-openai_latest.sif \
   python3 -m vllm.entrypoints.openai.api_server \
   --model "$MODEL_PATH" \
   --tokenizer "$MODEL_PATH" \
   --host 0.0.0.0 \
   --port 8001 \
   --gpu-memory-utilization 0.95 \
   --tensor-parallel-size 4 \
   --enforce-eager \

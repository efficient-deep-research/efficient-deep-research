#!/bin/bash
set -euxo pipefail

MODEL_PATH="/home/ko-yoshida/sftp_sync/mmu-rag/vllm_server/huggingface_cache/Qwen/Qwen3-Next-80B-A3B-Thinking"

singularity exec --nv \
    --bind ~/.cache/huggingface:/root/.cache/huggingface \
    vllm-openai_latest.sif \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --dtype="bfloat16" \
    --max-model-len 10000 \
    --max_num_seqs 1 \
    --enforce-eager \
    --cpu-offload-gb 40 \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 8 \
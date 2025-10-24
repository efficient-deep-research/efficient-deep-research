#!/bin/bash
set -euxo pipefail

MODEL_PATH="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.9

singularity exec --nv \
   --bind ~/.cache/huggingface:/root/.cache/huggingface \
   vllm-openai_v0.11.0.sif \
   python3 -m vllm.entrypoints.openai.api_server \
   --model "$MODEL_PATH" \
   --tokenizer "$MODEL_PATH" \
   --host 0.0.0.0 \
   --port 8000 \
   --gpu-memory-utilization 0.85 \
   --tensor-parallel-size 4 \
   --enable-expert-parallel \
   --max_num_seqs 1 \
   --api-key 12345 \

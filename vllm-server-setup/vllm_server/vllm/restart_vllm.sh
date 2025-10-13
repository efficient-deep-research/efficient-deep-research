#!/bin/bash

while true; do
  bash run_vllm.sh
  echo "vLLM crashed. Restarting in 5s..."
  sleep 5
done

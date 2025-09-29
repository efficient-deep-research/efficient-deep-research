from modelscope import snapshot_download
import os

model_name = "Qwen/Qwen3-Next-80B-A3B-Thinking"

print("Downloading model:", model_name)
model_path = snapshot_download(model_name)

print("Model downloaded to:", model_path)

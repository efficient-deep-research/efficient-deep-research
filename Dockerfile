# FROM python:3.11.13-slim-trixie
FROM nvcr.io/nvidia/pytorch:25.06-py3

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

COPY . /app

WORKDIR /app
RUN uv sync --frozen --no-cache
RUN uv pip install torch
RUN uv pip install vllm --torch-backend=auto

ARG TRANSFORMERS_BASE_MODEL_NAME="RUC-AIBOX/Qwen-7B-SimpleDeepSearcher"
RUN uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN uv run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"

CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]

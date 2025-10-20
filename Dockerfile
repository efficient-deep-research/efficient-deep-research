# Install uv
# FROM python:3.11.13-slim-trixie
FROM nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project
RUN uv pip install torch==2.8.0
RUN uv pip install flash-attn --no-build-isolation

# Copy the project into the image
COPY . /app

# Sync the project
# Here we set --inexact to prevent pip-installed packages from being uninstalled
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --inexact

RUN uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Thinking-2507')"
RUN uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b')"
RUN uv run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b')"

ARG OPENAI_API_BASE="http://host.docker.internal:8001/v1"
ENV OPENAI_API_BASE=$OPENAI_API_BASE

ARG OPENAI_API_KEY=""
ENV OPENAI_API_KEY=$OPENAI_API_KEY

ARG SUMMARIZER_OPENAI_API_BASE=""
ENV SUMMARIZER_OPENAI_API_BASE=$SUMMARIZER_OPENAI_API_BASE

ARG SUMMARIZER_OPENAI_API_KEY=""
ENV SUMMARIZER_OPENAI_API_KEY=$SUMMARIZER_OPENAI_API_KEY

ARG RETRIEVER_API_KEY=""
ENV RETRIEVER_API_KEY=$RETRIEVER_API_KEY

CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]

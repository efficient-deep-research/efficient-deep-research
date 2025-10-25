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
    uv sync --locked --no-install-project && \
    uv pip install torch==2.8.0 && \
    uv pip install flash-attn --no-build-isolation

# Pre-download model tokenizer and weights
ARG MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
ENV MODEL_PATH=$MODEL_PATH

ARG SUMMARIZER_MODEL_PATH=$MODEL_PATH
ENV SUMMARIZER_MODEL_PATH=$SUMMARIZER_MODEL_PATH

ARG RERANKER_MODEL_PATH="Qwen/Qwen3-Reranker-0.6B"
ENV RERANKER_MODEL_PATH=$RERANKER_MODEL_PATH

RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${MODEL_PATH}')" && \
    uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${SUMMARIZER_MODEL_PATH}')" && \
    uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${RERANKER_MODEL_PATH}')" && \
    uv run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${RERANKER_MODEL_PATH}')"
ENV TRANSFORMERS_OFFLINE=1

# Copy the project into the image
COPY . /app

# Sync the project
# Here we set --inexact to prevent pip-installed packages from being uninstalled
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --inexact

# Set environment variables for APIs
ARG OPENAI_API_BASE=""
ENV OPENAI_API_BASE=$OPENAI_API_BASE
ARG OPENAI_API_KEY=""
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ARG OPENAI_API_USERNAME=""
ENV OPENAI_API_USERNAME=$OPENAI_API_USERNAME
ARG OPENAI_API_PASSWORD=""
ENV OPENAI_API_PASSWORD=$OPENAI_API_PASSWORD

ARG SUMMARIZER_OPENAI_API_BASE=$OPENAI_API_BASE
ENV SUMMARIZER_OPENAI_API_BASE=$SUMMARIZER_OPENAI_API_BASE
ARG SUMMARIZER_OPENAI_API_KEY=$OPENAI_API_KEY
ENV SUMMARIZER_OPENAI_API_KEY=$SUMMARIZER_OPENAI_API_KEY
ARG SUMMARIZER_OPENAI_API_USERNAME=$OPENAI_API_USERNAME
ENV SUMMARIZER_OPENAI_API_USERNAME=$SUMMARIZER_OPENAI_API_USERNAME
ARG SUMMARIZER_OPENAI_API_PASSWORD=$OPENAI_API_PASSWORD
ENV SUMMARIZER_OPENAI_API_PASSWORD=$SUMMARIZER_OPENAI_API_PASSWORD

ARG RETRIEVER_API_KEY=""
ENV RETRIEVER_API_KEY=$RETRIEVER_API_KEY

ARG LOGTAIL_HOST=""
ENV LOGTAIL_HOST=$LOGTAIL_HOST
ARG LOGTAIL_TOKEN=""
ENV LOGTAIL_TOKEN=$LOGTAIL_TOKEN

EXPOSE 5027

# Run the FastAPI application
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "5027", "--host", "0.0.0.0"]

FROM python:3.11.13-slim-trixie
# FROM nvcr.io/nvidia/pytorch:25.08-py3

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

COPY . /app

WORKDIR /app
RUN uv sync --frozen --no-cache

CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so code changes don't invalidate this (large) layer
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY . .

# Download the in-process (Hugging Face) models into the image and build the embeddings database at build time.
# Ollama models are skipped here because there's no Ollama server in the container.
RUN python build_db.py

# Run as a non-root user that can still write on-demand embeddings to the database
RUN useradd --create-home app && chown -R app /app/data /models
USER app

# Models are already in the image, so never reach out to Hugging Face at runtime
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["fastapi", "run", "backend/app.py", "--port", "8000"]

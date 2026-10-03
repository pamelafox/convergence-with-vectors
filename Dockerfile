# The base image (Dockerfile.base, built by scripts/build_base.sh) has the dependencies, models, and embeddings database
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY . .

# Run as a non-root user that can still write on-demand embeddings to the database
RUN useradd --create-home app && chown -R app /app/data
USER app

# Models are already in the image, so never reach out to Hugging Face at runtime
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["fastapi", "run", "backend/app.py", "--port", "8000"]

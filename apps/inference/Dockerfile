# apps/inference/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VIRTUALENVS_CREATE=false

# Copy poetry files first for better caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 app
RUN chown -R app:app /app

# Create cache directories with proper permissions
RUN mkdir -p /app/model-cache /app/.cache
RUN chown -R app:app /app/model-cache /app/.cache

USER app

# Expose port
EXPOSE 5002

# Environment variables
ENV PYTHONPATH=/app
ENV INFERENCE_SERVER_HOST=0.0.0.0
ENV INFERENCE_SERVER_PORT=5002

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Start the inference server
CMD ["poetry", "run", "python", "-m", "neuronpedia_inference.server"]

# RunPod Serverless Dockerfile for Neuronpedia Inference
# Llama 3.3 70B AWQ with ChatSpace engine

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies and uv (fast Python package installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with uv (10-100x faster than pip)
# --break-system-packages bypasses PEP 668 protection (safe in Docker)
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# Copy steerllm and chatspace from vendor
COPY vendor/steerllm /app/vendor/steerllm
COPY vendor/chatspace /app/vendor/chatspace

# Install steerllm first, then chatspace (--no-deps to skip chatspace's local steerllm path)
RUN uv pip install --system --break-system-packages --no-cache /app/vendor/steerllm && \
    uv pip install --system --break-system-packages --no-cache --no-deps /app/vendor/chatspace

# Copy application code
COPY src/ /app/src/

# Copy persona data
COPY data/ /app/data/

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Set default environment variables for the model
ENV MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
ENV OVERRIDE_MODEL_ID=casperhansen/llama-3.3-70b-instruct-awq
ENV MODEL_DTYPE=bfloat16
ENV TOKEN_LIMIT=65536
ENV MAX_MODEL_LEN=65536
ENV GPU_MEMORY_UTILIZATION=0.95
ENV TENSOR_PARALLEL_SIZE=1
ENV PERSONA_DATA_PATH=/app/data

# Concurrency settings - allow multiple requests per GPU
# MAX_NUM_SEQS: Max concurrent sequences vLLM can process (default 256, increase with more VRAM)
ENV MAX_NUM_SEQS=32
# MAX_CONCURRENCY: Max HTTP requests the RunPod handler accepts simultaneously
ENV MAX_CONCURRENCY=32

# Run the handler
CMD ["python", "-u", "/app/src/handler.py"]


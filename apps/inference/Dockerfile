# apps/inference/Dockerfile
# Simplified version that bypasses dependency issues
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

# Copy everything first
COPY . .

# Install dependencies manually (bypass poetry issues)
RUN pip install --no-cache-dir \
    transformers \
    torch \
    torchaudio \
    datasets \
    accelerate \
    safetensors \
    huggingface-hub \
    tokenizers \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    requests \
    fastapi \
    uvicorn \
    pydantic \
    typing-extensions

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 app
RUN chown -R app:app /app

# Simple fallback start script
RUN echo '#!/bin/bash\n\
echo "Available files:"\n\
ls -la\n\
echo "Python path: $PYTHONPATH"\n\
if [ -f "server.py" ]; then\n\
    echo "Starting with server.py"\n\
    python server.py\n\
elif [ -f "start.py" ]; then\n\
    echo "Starting with start.py"\n\
    python start.py\n\
elif [ -f "app.py" ]; then\n\
    echo "Starting with app.py"\n\
    python app.py\n\
else\n\
    echo "Trying module import"\n\
    python -m neuronpedia_inference.server || echo "Module not found, keeping container alive"\n\
    sleep 3600\n\
fi\n\
' > /app/start.sh && chmod +x /app/start.sh

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

# Start with the script
CMD ["/app/start.sh"]

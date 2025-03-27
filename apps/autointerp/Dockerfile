ARG BUILD_TYPE
ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04

# NON-CUDA base
FROM python:3.10-slim AS base-nocuda

# CUDA base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base-cuda
# Nvidia container toolkit
RUN apt-get update && apt-get install -y \
    curl gpg
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Pick the correct base based on the BUILD_TYPE
FROM base-${BUILD_TYPE:-nocuda} AS final

WORKDIR /app

ENV HOST=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip \
    && pip install poetry

# Copy only pyproject.toml and poetry.lock first to leverage Docker cache
COPY apps/autointerp/pyproject.toml apps/autointerp/poetry.lock* /app/

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Now copy the actual code
COPY apps/autointerp /app/

# Install the project itself
RUN poetry install --no-interaction --no-ansi

# Expose the port the app runs on
EXPOSE 5003

# Command to run the application
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "1"]
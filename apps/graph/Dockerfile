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

# Main build stage
FROM base-${BUILD_TYPE:-nocuda} AS final

# Set working directory
WORKDIR /app

ENV HOST=0.0.0.0

# Ignore hash sum mismatch for apt-get
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false

COPY apps/graph/pyproject.toml apps/graph/poetry.lock* apps/graph/
WORKDIR /app/apps/graph

# Install dependencies
ENV POETRY_REQUESTS_TIMEOUT=30
RUN poetry install --only main

# Now copy the rest of the application code
COPY apps/graph /app/apps/graph/

# Expose the graph server port
EXPOSE 5004

# The server requires a model argument, defaulting to gemma-2-2b
# This can be overridden when running the container
CMD ["python", "start.py"] 
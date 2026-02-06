ARG BUILD_TYPE
ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04

# Optional custom CA bundle file support
ARG CUSTOM_CA_BUNDLE

# NON-CUDA base
FROM python:3.10-slim AS base-nocuda

# Re-declare ARG after FROM (ARGs don't persist across FROM statements)
ARG CUSTOM_CA_BUNDLE

# Copy the CA bundle file if provided, otherwise copy nothing (using .nocustomca as a no-op)
COPY ${CUSTOM_CA_BUNDLE:-.nocustomca} /tmp/ca-bundle-temp

# Set up CA certificates and environment if bundle was provided
RUN if [ -f /tmp/ca-bundle-temp ] && [ "${CUSTOM_CA_BUNDLE}" != ".nocustomca" ]; then \
    apt-get update && apt-get install -y ca-certificates && \
    mkdir -p /usr/local/share/ca-certificates && \
    mv /tmp/ca-bundle-temp /usr/local/share/ca-certificates/custom-ca.crt && \
    cat /usr/local/share/ca-certificates/custom-ca.crt >> /etc/ssl/certs/ca-certificates.crt && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*; \
    else \
    rm -f /tmp/ca-bundle-temp; \
    fi

# Set SSL environment variables if CA bundle was provided
ENV SSL_CERT_FILE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV SSL_CERT_DIR=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs}
ENV REQUESTS_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV CURL_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV GIT_SSL_CAINFO=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}

# CUDA base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base-cuda

# Re-declare ARG after FROM (ARGs don't persist across FROM statements)
ARG CUSTOM_CA_BUNDLE

# Copy the CA bundle file if provided, otherwise copy nothing (using .nocustomca as a no-op)
COPY ${CUSTOM_CA_BUNDLE:-.nocustomca} /tmp/ca-bundle-temp

# Install dependencies and set up CA certificates, then download NVIDIA toolkit
RUN apt-get update && apt-get install -y curl gpg ca-certificates && \
    if [ -f /tmp/ca-bundle-temp ] && [ "${CUSTOM_CA_BUNDLE}" != ".nocustomca" ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        mv /tmp/ca-bundle-temp /usr/local/share/ca-certificates/custom-ca.crt && \
        cat /usr/local/share/ca-certificates/custom-ca.crt >> /etc/ssl/certs/ca-certificates.crt && \
        update-ca-certificates && \
        export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt && \
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list; \
    else \
        rm -f /tmp/ca-bundle-temp && \
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Set SSL environment variables if CA bundle was provided
ENV SSL_CERT_FILE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV SSL_CERT_DIR=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs}
ENV REQUESTS_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV CURL_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV GIT_SSL_CAINFO=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*
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

# Optional custom CA bundle file support (re-declare for final stage)
ARG CUSTOM_CA_BUNDLE

# Ignore hash sum mismatch for apt-get
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry config virtualenvs.create false

# Set SSL environment variables if CA bundle was provided (for final stage)
ENV SSL_CERT_FILE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV SSL_CERT_DIR=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs}
ENV REQUESTS_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV CURL_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV GIT_SSL_CAINFO=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}

# Configure poetry/pip to use custom CA if provided
RUN if [ -n "${CUSTOM_CA_BUNDLE}" ] && [ "${CUSTOM_CA_BUNDLE}" != ".nocustomca" ]; then \
    pip config set global.cert /etc/ssl/certs/ca-certificates.crt && \
    poetry config certificates.default.cert /etc/ssl/certs/ca-certificates.crt; \
    fi

# Copy the client package first
COPY packages/python/neuronpedia-autointerp-client /app/packages/python/neuronpedia-autointerp-client/

# Copy only poetry files
COPY apps/autointerp/pyproject.toml apps/autointerp/poetry.lock* apps/autointerp/
WORKDIR /app/apps/autointerp

# Install only main dependencies
ENV POETRY_REQUESTS_TIMEOUT=30
RUN poetry install --only main

# Now copy the rest of the application code
COPY apps/autointerp /app/apps/autointerp/

EXPOSE 5003

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "1"]

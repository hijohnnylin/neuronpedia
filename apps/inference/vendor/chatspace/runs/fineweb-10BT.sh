#!/usr/bin/env bash

uv run chatspace embed-hf \
    --dataset HuggingFaceFW/fineweb \
    --subset sample-10BT \
    --split train \
    --model Qwen/Qwen3-Embedding-0.6B \
    --tokens-per-batch 131072 \
    --rows-per-shard 8192 \
    --output-root /workspace \
    --device cuda \
    --dtype bfloat16 \
    --compile-model \
    # --max-rows 4096 \
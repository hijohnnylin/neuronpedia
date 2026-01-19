#!/usr/bin/env bash

# WildChat-4.8M: ~3.2M conversations (non-toxic) from GPT-3.5 and GPT-4
# Extracts first assistant response from each conversation
# Embedded text will be in 'text' field in output parquet

uv run chatspace embed-hf \
    --dataset allenai/WildChat-4.8M \
    --split train \
    --model Qwen/Qwen3-Embedding-0.6B \
    --text-field conversation \
    --extract-first-assistant \
    --tokens-per-batch 131072 \
    --rows-per-shard 8192 \
    --output-root /workspace \
    --device cuda \
    --dtype bfloat16 \
    --compile-model \
    # --max-rows 4096 \
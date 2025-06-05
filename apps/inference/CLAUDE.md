# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Inference Service Overview

The inference service is a FastAPI server that provides neural network interpretability capabilities for Neuronpedia. It handles model steering, feature activation testing, and tokenization using Sparse Autoencoders (SAEs).

## Architecture

- **FastAPI** server with async endpoints
- **Singleton pattern** for Config, Model, and SAEManager
- **Request locking** to prevent concurrent model operations
- **Lazy loading** of models after server startup
- **Type safety** using Pydantic models from auto-generated client library

## Key Components

- `server.py`: Main FastAPI application and endpoint definitions
- `config.py`: Global configuration singleton
- `sae_manager.py`: Manages loading and caching of SAEs
- `endpoints/`: Individual endpoint implementations
- `saes/`: SAE implementations (base class and SAELens adapter)
- `inference_utils/`: Core logic for steering and inference

## Development Commands

```bash
# Install dependencies
poetry lock && poetry install

# Run server locally
poetry run python start.py

# Run with specific model
poetry run python start.py --model_id gemma-2-2b --sae_sets gemmascope-res-16k

# Run all tests
make test

# Run specific test
poetry run pytest tests/unit/test_server.py -v

# Format code
make format

# Type check
make check-type

# Full CI checks
make check-ci
```

## Testing Approach

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test full API endpoints with real models
- Use `pytest` with fixtures defined in `conftest.py`
- Mock external dependencies when appropriate
- Always run `make check-ci` before committing
- ALWAYS use the `make` commands defined in @Makefile to run/validate tests (e.g, make check-format)

## API Endpoints

- **Activation**: `/v1/activation/{single,all,topk-by-token}`
- **Steering**: `/v1/steer/{completion,completion-chat}`
- **Utilities**: `/v1/{tokenize,util/*}`
- **System**: `/health`, `/initialize`

All endpoints use Pydantic models from `neuronpedia_interface` for request/response validation.

## Environment Variables

Critical variables for local development:
- `MODEL_ID`: Base model to use (default: gpt2-small)
- `SAE_SETS`: JSON array of SAE sets to load
- `DEVICE`: cpu, cuda, or mps
- `TOKEN_LIMIT`: Maximum tokens to process (default: 200)
- `MAX_LOADED_SAES`: SAE cache size (default: 300)

## Docker & Deployment

```bash
# Build CPU image
docker build --platform=linux/amd64 -t neuronpedia-inference:cpu -f Dockerfile --build-arg BUILD_TYPE=nocuda .

# Build GPU image
docker build --platform=linux/amd64 -t neuronpedia-inference:gpu -f Dockerfile --build-arg BUILD_TYPE=cuda .
```

Kubernetes deployments use Kustomize with overlays for different models and resource configurations.

## Common Tasks

### Adding a New Endpoint
1. Create endpoint file in `endpoints/` following existing patterns
2. Add route to `server.py`
3. Update OpenAPI spec if needed
4. Write unit and integration tests
5. Run `make check-ci` to ensure all checks pass

### Debugging Model Loading
- Check logs for initialization messages
- Verify environment variables are set correctly
- Use `/health` endpoint to check server status
- Models are loaded lazily after server starts

### Performance Optimization
- SAEs are cached with LRU eviction
- Use request locking to prevent memory issues
- Monitor `MAX_LOADED_SAES` for memory usage
- Consider batch processing for multiple activations

## Benchmarking Guidance
- When doing benchmarking on speeds, ALWAYS need to use actual, manually-run results - should never try to "stub" or "simulate demo" performance.
.PHONY: test test-all test-integration install lint

# Default target: Run fast unit tests (skip marked slow tests)
test:
	uv run pytest

# Run all tests including slow integration tests
test-all:
	uv run pytest --run-slow

# Run only the slow integration tests
test-integration:
	uv run pytest --run-slow tests/test_vllm_comprehensive_integration.py

# Install dependencies
install:
	uv sync

# Lint/check code (if ruff or similar is added later, currently just a placeholder)
lint:
	@echo "No linter configured yet"


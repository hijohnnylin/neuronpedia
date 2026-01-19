.PHONY: test test-all test-integration install lint

# Default target: Run fast unit tests (skip marked slow tests)
test:
	uv run pytest

# Run all tests including slow integration tests
test-all:
	uv run pytest --run-slow

# Run only the slow integration tests
test-integration:
	uv run pytest --run-slow -m slow

# Install dependencies
install:
	uv sync

# Lint/check code (placeholder)
lint:
	@echo "No linter configured yet"

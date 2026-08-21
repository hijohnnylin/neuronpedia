.PHONY: test test-verbose format check-format check-type check-ci install install-dev openapi

# Variables
UV = uv
PYTEST = $(UV) run pytest
COVERAGE = $(UV) run coverage
RUFF = $(UV) run ruff

# Targets
test:
	$(PYTEST) -v --cov=neuronpedia_autointerp/ --cov-report=term-missing --cov-branch tests

test-verbose:
	$(PYTEST) -vv --no-header

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(UV) run pyright .

check-ci: check-format check-type test

# Regenerates the committed openapi.json from the pydantic models. The wire format is
# defined in neuronpedia_autointerp/schemas/, so this is derived output, never edited.
openapi:
	$(UV) run python dump_openapi.py

install:
	$(UV) sync --no-dev

install-dev:
	$(UV) sync

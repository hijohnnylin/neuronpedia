.PHONY: test-verbose coverage format test check-format check-ci install install-dev all run-server

# Variables
PYTHON = python
UV = uv
PYTEST = $(UV) run pytest
COVERAGE = $(UV) run coverage
RUFF = $(UV) run ruff

# Targets
test:
	$(PYTEST) -v --cov=neuronpedia_inference/ --cov-report=term-missing --cov-branch tests

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

install:
	$(UV) sync --no-dev

install-dev:
	$(UV) sync

all: install-dev format check-format test coverage

run-server:
	$(UV) run python start.py

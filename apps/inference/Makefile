.PHONY: test-verbose coverage format unit-test check-format check-ci install install-dev all run-server

# Variables
PYTHON = python
POETRY = poetry
PYTEST = $(POETRY) run pytest
COVERAGE = $(POETRY) run coverage
RUFF = $(POETRY) run ruff

# Targets
unit-test:
	$(POETRY) run $(PYTEST) -v --cov=neuronpedia_inference/ --cov-report=term-missing --cov-branch tests/unit

integration-test:
	$(POETRY) run $(PYTEST) $(PYTEST_ARGS) tests/integration

test-verbose:
	$(POETRY) run $(PYTEST) -vv --no-header

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(POETRY) run pyright .

check-ci: check-format check-type unit-test

install:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install

install-dev:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install --with dev

all: install-dev format check-format test coverage

run-server:
	$(POETRY) run python server.py

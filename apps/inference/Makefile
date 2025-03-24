.PHONY: test test-verbose coverage format unit-test integration-test check-format check-ci install install-dev all run-server

# Variables
PYTHON = python
POETRY = poetry
PYTEST = $(POETRY) run pytest
PYTEST_ARGS = -v -s
COVERAGE = $(POETRY) run coverage
RUFF = $(POETRY) run ruff

# Targets
test: unit-test integration-test

unit-test:
	$(POETRY) run $(PYTEST) $(PYTEST_ARGS) tests/unit

integration-test:
	$(POETRY) run $(PYTEST) $(PYTEST_ARGS) tests/integration

test-verbose:
	$(POETRY) run $(PYTEST) $(PYTEST_ARGS) -vv

coverage:
	$(COVERAGE) run -m pytest
	$(COVERAGE) report -m

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(POETRY) run pyright .

check-ci: check-format unit-test

install:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install

install-dev:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install --with dev

all: install-dev format check-format test coverage

run-server:
	$(POETRY) run python server.py

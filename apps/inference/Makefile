.PHONY: test-verbose coverage format unit-test check-format check-ci install install-dev all run-server

# Variables
PYTHON = python
POETRY = poetry
PYTEST = $(POETRY) run pytest
PYTEST_ARGS = -v -s
COVERAGE = $(POETRY) run coverage
RUFF = $(POETRY) run ruff

# Targets
unit-test:
	$(POETRY) run $(PYTEST) $(PYTEST_ARGS) tests/unit

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

check-ci: check-format check-type unit-test

install:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install

install-dev:
	$(POETRY) config virtualenvs.create true
	$(POETRY) install --with dev

all: install-dev format check-format test coverage

run-server:
	$(POETRY) run python server.py --reload

run-server-dev:
	$(POETRY) run python start.py --reload --reload-dir neuronpedia_inference --include_sae 7-res-jb --max_loaded_saes 1

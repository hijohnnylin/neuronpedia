.PHONY: test format check-format

# Variables
POETRY = poetry
PYTEST = $(POETRY) run pytest
PYTEST_ARGS = -v -s
RUFF = $(POETRY) run ruff

test:
	$(POETRY) run $(PYTEST) -v --cov=neuronpedia_autointerp/ --cov-report=term-missing --cov-branch tests

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(POETRY) run pyright .

check-ci: check-format check-type test

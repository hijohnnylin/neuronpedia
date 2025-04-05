.PHONY: unit-test format check-format

# Variables
POETRY = poetry
PYTEST = $(POETRY) run pytest
PYTEST_ARGS = -v -s
RUFF = $(POETRY) run ruff

unit-test:
	$(PYTEST) $(PYTEST_ARGS) tests/unit

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(POETRY) run pyright .

check-ci: check-format check-type unit-test

.PHONY: format check-format

# Variables
POETRY = poetry
RUFF = $(POETRY) run ruff

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(POETRY) run pyright .

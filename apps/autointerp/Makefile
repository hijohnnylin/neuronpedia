.PHONY: format check-format

# Variables
PYTHON = python
RUFF = $(PYTHON) -m ruff

format:
	$(RUFF) format .
	$(RUFF) check --fix-only .

check-format:
	$(RUFF) check .
	$(RUFF) format --check .

check-type:
	$(PYTHON) -m pyright .

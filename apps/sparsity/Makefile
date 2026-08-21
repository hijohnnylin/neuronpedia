.PHONY: install install-dev run test openapi format check-format

UV = uv
PYTEST = $(UV) run pytest

install:
	$(UV) sync --no-dev

install-dev:
	$(UV) sync

run:
	$(UV) run uvicorn server:app --port 5005

test:
	$(PYTEST) -v

# Regenerates the committed openapi.json from the pydantic models. The wire format is defined
# in schemas.py, so this is derived output, never edited. Needs no weights and no GPU.
openapi:
	$(UV) run python dump_openapi.py

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

check-format:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

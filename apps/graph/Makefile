.PHONY: install install-dev test openapi format check-format

UV = uv
PYTEST = $(UV) run pytest

# The attribution backends are optional extras that share no code, so pick one. `openapi` and
# the tests need whichever one is synced to be importable; see pyproject.toml.
install:
	$(UV) sync --extra circuit-tracer

install-dev:
	$(UV) sync --extra circuit-tracer

test:
	$(PYTEST) -v

# Regenerates the committed openapi.json from the pydantic models. The wire format is defined
# in neuronpedia_graph/schemas.py, so this is derived output, never edited. Needs no weights
# and no GPU -- the model is loaded by the lifespan handler, which importing does not run.
openapi:
	$(UV) run python dump_openapi.py

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

check-format:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

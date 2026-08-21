.PHONY: install install-dev run test openapi format check-format

install:
	uv sync

install-dev:
	uv sync --extra dev

run:
	uv run python server.py

test:
	uv run pytest tests/ -v

openapi:
	uv run python dump_openapi.py

format:
	uv run ruff check --fix . && uv run ruff format .

check-format:
	uv run ruff check . && uv run ruff format --check .

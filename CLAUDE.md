# Neuronpedia Development Guide

## Build and Run Commands

- **Webapp Dev**:
  - Local DB: `cd apps/webapp && npm run dev:localhost`
  - Demo: `cd apps/webapp && npm run dev:demo`
- **Webapp Build**:
  - Local DB: `cd apps/webapp && npm run build:localhost`
  - Simple build: `cd apps/webapp && npm run build:simple`
- **Inference**: `cd apps/inference && poetry install && poetry run python start.py`
- **Autointerp**: `cd apps/autointerp && poetry install . && poetry run python server.py`

## Testing

- **Webapp Tests**: `cd apps/webapp && npm run test:vitest`
- **Webapp E2E**: `cd apps/webapp && npm run test:playwright`
- **Single Test**: `cd apps/webapp && npx vitest components/path/to/file.test.tsx`
- **Inference Tests**: `cd apps/inference && make test` or `poetry run pytest tests/path/to/test_file.py -v`

## Linting

- **Webapp**: `cd apps/webapp && npm run lint:fix && npm run format:write`
- **Inference/Python**: `cd apps/inference && poetry run ruff check --fix . && poetry run ruff format .`

## Style Guidelines

- **TS/React**: Use functional components, strict types, `@/` path aliases, 2-space indent, 120 char line width
- **Python**: Ruff for linting/formatting, docstrings for functions, proper exception handling
- **Naming**: TS/React: PascalCase components, camelCase functions, kebab-case files
- **Naming**: Python: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- **Imports**: Group by standard lib, third-party, internal; keep organized and use absolute imports

# Neuronpedia Development Guide

> Comprehensive guide for AI assistants working on the Neuronpedia codebase

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Build and Run Commands](#build-and-run-commands)
- [Testing](#testing)
- [Linting and Formatting](#linting-and-formatting)
- [Style Guidelines](#style-guidelines)
- [Development Workflows](#development-workflows)
- [Common Patterns](#common-patterns)
- [Important File Locations](#important-file-locations)

## Project Overview

Neuronpedia is an open-source interpretability platform for neural networks. It's a monorepo containing multiple services that work together:

- **Webapp**: Next.js frontend and API (TypeScript/React)
- **Inference**: PyTorch-based inference server for model activations, steering, and testing
- **Autointerp**: Automatic interpretation and scoring using EleutherAI's Delphi
- **Graph**: Circuit tracer for attribution graph generation (based on circuit-tracer)
- **Database**: PostgreSQL with Prisma ORM

### Architecture Principles

- **Service Independence**: Each service can run standalone and be independently deployed
- **Schema-Driven Development**: OpenAPI schemas define service contracts
- **Type Safety**: Strict TypeScript and Python type checking throughout
- **Monorepo Structure**: Shared schemas and generated clients in `packages/`

## Repository Structure

```
neuronpedia/
├── apps/                          # Main applications
│   ├── webapp/                    # Next.js frontend + API
│   │   ├── app/                   # Next.js App Router pages
│   │   ├── components/            # React components
│   │   ├── lib/                   # Utilities, hooks, external clients
│   │   ├── prisma/                # Database schema and migrations
│   │   └── package.json
│   ├── inference/                 # PyTorch inference server
│   │   ├── neuronpedia_inference/ # Main Python package
│   │   ├── tests/                 # Pytest tests
│   │   └── pyproject.toml
│   ├── autointerp/                # Auto-interpretation server
│   │   ├── neuronpedia_autointerp/
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── graph/                     # Graph/circuit tracer server
│   │   └── pyproject.toml
│   └── experiments/               # Experimental projects
├── schemas/                       # OpenAPI schemas
│   ├── openapi/                   # Schema definitions
│   │   ├── inference-server.yaml
│   │   └── autointerp-server.yaml
│   └── Makefile                   # Schema generation commands
├── packages/                      # Generated clients
│   ├── typescript/                # TypeScript clients
│   │   ├── neuronpedia-inference-client/
│   │   └── neuronpedia-autointerp-client/
│   └── python/                    # Python clients
│       ├── neuronpedia-inference-client/
│       ├── neuronpedia-autointerp-client/
│       └── neuronpedia-webapp-client/
├── utils/                         # Utility scripts
│   └── neuronpedia-utils/         # Batch processing, data conversion
├── k8s/                          # Kubernetes configurations
├── docker/                       # Docker configurations
└── docs/                         # Additional documentation
```

## Technology Stack

### Webapp (TypeScript/React)
- **Framework**: Next.js 14 (App Router)
- **UI**: React 18, Radix UI, TailwindCSS
- **State**: TanStack Query (React Query)
- **Forms**: Formik, Yup validation
- **Database**: Prisma ORM + PostgreSQL
- **Auth**: NextAuth.js
- **Testing**: Vitest (unit), Playwright (e2e)
- **Linting**: ESLint (Airbnb config), Prettier

### Inference & Autointerp (Python)
- **ML Framework**: PyTorch, TransformerLens, SAELens
- **API**: FastAPI, Uvicorn
- **Testing**: Pytest, Coverage
- **Linting**: Ruff (linting + formatting), Pyright (type checking)
- **Package Manager**: Poetry

### Database
- **RDBMS**: PostgreSQL
- **ORM**: Prisma
- **Vector Search**: pgvector extension

## Build and Run Commands

### Webapp

**Development** (auto-reload, debug info):
```bash
# Demo environment (remote read-only DB + inference)
cd apps/webapp && npm run dev:demo

# Local environment (local DB)
cd apps/webapp && npm run dev:localhost

# Remote environment
cd apps/webapp && npm run dev:remote
```

**Production Build**:
```bash
# With localhost config
cd apps/webapp && npm run build:localhost

# Simple build (uses .env)
cd apps/webapp && npm run build:simple

# Demo build
cd apps/webapp && npm run build:demo
```

**Database Operations**:
```bash
# Run migrations (localhost)
cd apps/webapp && npm run migrate:localhost

# Push schema without migration
cd apps/webapp && npm run db:push

# Seed database
cd apps/webapp && npm run db:seed
```

### Inference Server

```bash
# Install dependencies
cd apps/inference && poetry install

# Run server (development)
cd apps/inference && poetry run python start.py

# Run tests
cd apps/inference && make test
# or
cd apps/inference && poetry run pytest tests/path/to/test_file.py -v
```

### Autointerp Server

```bash
# Install dependencies
cd apps/autointerp && poetry install .

# Run server
cd apps/autointerp && poetry run python server.py

# Run tests
cd apps/autointerp && poetry run pytest
```

### Graph Server

```bash
# Install dependencies
cd apps/graph && poetry install

# Run server
cd apps/graph && poetry run python server.py
```

### Make Commands

The root `Makefile` provides convenience commands:

```bash
make help                          # List all available commands
make init-env                      # Generate local .env files
make webapp-demo-build             # Build webapp with demo config
make webapp-demo-run               # Run webapp demo environment
make webapp-localhost-build        # Build webapp with localhost config
make webapp-localhost-run          # Run webapp with local DB
make webapp-localhost-dev          # Run webapp in dev mode
make install-nodejs                # Install Node.js via nvm
```

## Testing

### Webapp Tests

**Unit Tests (Vitest)**:
```bash
cd apps/webapp && npm run test:vitest

# Single test file
cd apps/webapp && npx vitest components/path/to/file.test.tsx

# Coverage report
cd apps/webapp && npm run test:coverage
```

**E2E Tests (Playwright)**:
```bash
cd apps/webapp && npm run test:playwright
```

### Python Tests

**Inference**:
```bash
cd apps/inference && make test
cd apps/inference && poetry run pytest -v
cd apps/inference && poetry run pytest tests/specific_test.py -v

# With coverage
cd apps/inference && poetry run pytest --cov
```

**Autointerp**:
```bash
cd apps/autointerp && poetry run pytest
cd apps/autointerp && poetry run pytest --cov
```

## Linting and Formatting

### Webapp (TypeScript/React)

```bash
cd apps/webapp

# Lint check
npm run lint

# Lint and auto-fix
npm run lint:fix

# Format check
npm run format

# Format and write
npm run format:write

# Run both (recommended workflow)
npm run lint:fix && npm run format:write
```

**ESLint Config**: Airbnb TypeScript, with Prettier integration
**Prettier Config**: 2-space indent, 120 char line width, semicolons, single quotes

### Python (Inference/Autointerp)

```bash
cd apps/inference  # or apps/autointerp

# Lint check
poetry run ruff check .

# Lint and auto-fix
poetry run ruff check --fix .

# Format code
poetry run ruff format .

# Type check
poetry run pyright

# Run all (recommended workflow)
poetry run ruff check --fix . && poetry run ruff format .
```

**Ruff Config**: Modern Python linter/formatter (replaces flake8, black, isort)
**Pyright Config**: Strict type checking with some exceptions

## Style Guidelines

### TypeScript/React

**Component Structure**:
- ✅ Use functional components with hooks
- ✅ Prefer named exports over default exports
- ✅ Use TypeScript interfaces for props
- ✅ Destructure props in function signature

**Naming Conventions**:
- `PascalCase` for components: `ActivationsList`, `FeatureModal`
- `camelCase` for functions/variables: `fetchActivations`, `userData`
- `kebab-case` for files: `activation-item.tsx`, `feature-stats.tsx`
- `UPPER_SNAKE_CASE` for constants: `MAX_FEATURES`, `API_ENDPOINT`

**Import Order** (auto-sorted by Prettier):
1. React/Next.js imports
2. Third-party libraries
3. Internal absolute imports (`@/...`)
4. Relative imports
5. Type imports (separate)

**Path Aliases**:
```typescript
import Button from '@/components/ui/button'  // instead of '../../../components/ui/button'
import { db } from '@/lib/db'
import { useAuth } from '@/lib/hooks/use-auth'
```

**TypeScript Guidelines**:
- Use strict mode (`strict: true`)
- Prefer `interface` over `type` for object shapes
- Use `type` for unions, intersections, mapped types
- Avoid `any` - use `unknown` or proper types
- Use optional chaining (`?.`) and nullish coalescing (`??`)

**React Patterns**:
```typescript
// Good
interface FeatureCardProps {
  featureId: string
  modelId: string
  onSelect?: (id: string) => void
}

export function FeatureCard({ featureId, modelId, onSelect }: FeatureCardProps) {
  // Component logic
}

// Avoid
export default function FeatureCard(props: any) {
  // ...
}
```

### Python

**Naming Conventions**:
- `PascalCase` for classes: `InferenceServer`, `SAEManager`
- `snake_case` for functions/variables: `load_model`, `activation_data`
- `UPPER_SNAKE_CASE` for constants: `MAX_BATCH_SIZE`, `DEFAULT_TIMEOUT`
- Private members: `_internal_method`, `_cache`

**Type Hints**:
```python
# Required for all function signatures
def process_activations(
    features: list[int],
    model_id: str,
    layer: int | None = None
) -> dict[str, Any]:
    """Process feature activations for a model layer."""
    pass

# Use modern type syntax (Python 3.10+)
# ✅ list[str], dict[str, int], int | None
# ❌ List[str], Dict[str, int], Optional[int]
```

**Docstrings**:
```python
def calculate_scores(activations: list[float], threshold: float = 0.5) -> dict[str, float]:
    """
    Calculate activation scores above threshold.

    Args:
        activations: List of activation values
        threshold: Minimum activation threshold

    Returns:
        Dictionary mapping score names to values

    Raises:
        ValueError: If activations list is empty
    """
    pass
```

**Import Order**:
1. Standard library imports
2. Third-party imports
3. Local application imports

**Error Handling**:
```python
# Use specific exceptions
raise ValueError(f"Invalid model_id: {model_id}")

# Catch specific exceptions
try:
    result = process_data(input)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error in processing")
    raise
```

## Development Workflows

### Schema-Driven Development (Inference/Autointerp)

When adding or modifying API endpoints:

1. **Update OpenAPI Schema**:
   ```bash
   # Edit the schema
   vim schemas/openapi/inference-server.yaml
   # or
   vim schemas/openapi/inference/paths/new-endpoint.yaml
   ```

2. **Regenerate Clients**:
   ```bash
   cd schemas
   make setup-all-inference VERSION=1.x.x
   # or
   make setup-all-autointerp VERSION=1.x.x
   ```

3. **Update Server Implementation**:
   ```bash
   # Implement the endpoint
   vim apps/inference/neuronpedia_inference/endpoints/new_endpoint.py
   ```

4. **Update Webapp Client Usage**:
   ```bash
   # Use the generated client
   vim apps/webapp/lib/utils/inference.ts
   ```

5. **Test Locally**: Run both services and test the integration

### Database Migrations (Prisma)

**Creating a Migration**:
```bash
cd apps/webapp

# 1. Modify schema
vim prisma/schema.prisma

# 2. Create migration (auto-generates SQL)
npm run migrate:localhost

# 3. Migration files created in prisma/migrations/
```

**Applying Migrations**:
```bash
# Development
npm run migrate:localhost

# Production
npm run db:migrate:deploy
```

**Prisma Studio** (Database GUI):
```bash
cd apps/webapp
npx prisma studio
```

### Common Development Tasks

**Adding a New React Component**:
1. Create file: `apps/webapp/components/my-component.tsx`
2. Use kebab-case for filename
3. Export named function component (PascalCase)
4. Add tests: `apps/webapp/components/__tests__/my-component.test.tsx`

**Adding a New API Route**:
1. Create route: `apps/webapp/app/api/my-route/route.ts`
2. Export handler functions: `GET`, `POST`, etc.
3. Use Prisma client from `@/lib/db`
4. Add proper error handling and validation

**Adding a Python Endpoint (Inference)**:
1. Update schema: `schemas/openapi/inference/paths/my-endpoint.yaml`
2. Regenerate: `cd schemas && make setup-all-inference VERSION=x.x.x`
3. Implement: `apps/inference/neuronpedia_inference/my_endpoint.py`
4. Add tests: `apps/inference/tests/test_my_endpoint.py`

## Common Patterns

### Webapp Patterns

**Data Fetching with React Query**:
```typescript
import { useQuery } from '@tanstack/react-query'

export function useFeatureData(modelId: string, featureId: string) {
  return useQuery({
    queryKey: ['feature', modelId, featureId],
    queryFn: () => fetchFeature(modelId, featureId),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}
```

**API Routes**:
```typescript
// apps/webapp/app/api/features/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'

export async function GET(req: NextRequest) {
  try {
    const features = await db.feature.findMany({ take: 10 })
    return NextResponse.json(features)
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch features' },
      { status: 500 }
    )
  }
}
```

**Using Inference Client**:
```typescript
import { getInferenceClient } from '@/lib/utils/inference'

const client = getInferenceClient()
const result = await client.POST('/v1/activations/test', {
  body: {
    modelId: 'gpt2-small',
    sourceId: '10-res-jb',
    featureIndex: 123,
    testText: 'Hello world'
  }
})
```

### Python Patterns

**FastAPI Endpoint**:
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ActivationRequest(BaseModel):
    model_id: str
    feature_index: int
    text: str

@router.post("/v1/activations/test")
async def test_activation(request: ActivationRequest):
    try:
        result = process_activation(
            request.model_id,
            request.feature_index,
            request.text
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Error Handling**:
```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def process_data(input: str) -> dict:
    if not input:
        raise ValueError("Input cannot be empty")

    try:
        result = expensive_operation(input)
        return result
    except Exception as e:
        logger.exception(f"Failed to process: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")
```

## Important File Locations

### Configuration Files

- `.env.localhost` - Local development environment variables
- `.env.demo` - Demo environment configuration
- `.env.inference.*` - Inference server configurations for specific models
- `apps/webapp/tsconfig.json` - TypeScript configuration
- `apps/webapp/next.config.js` - Next.js configuration
- `apps/webapp/tailwind.config.js` - TailwindCSS configuration
- `apps/webapp/.eslintrc` - ESLint configuration
- `apps/webapp/.prettierrc` - Prettier configuration
- `apps/inference/pyproject.toml` - Inference Python configuration
- `apps/autointerp/pyproject.toml` - Autointerp Python configuration

### Schema and Types

- `schemas/openapi/` - OpenAPI schema definitions
- `apps/webapp/prisma/schema.prisma` - Database schema
- `apps/webapp/types/` - TypeScript type definitions
- `packages/typescript/` - Generated TypeScript clients
- `packages/python/` - Generated Python clients

### Key Source Directories

**Webapp**:
- `apps/webapp/app/` - Next.js pages and API routes
- `apps/webapp/components/` - React components
- `apps/webapp/lib/db/` - Database utilities
- `apps/webapp/lib/utils/` - Utility functions and client wrappers
- `apps/webapp/lib/hooks/` - Custom React hooks
- `apps/webapp/lib/external/` - External service clients

**Inference**:
- `apps/inference/neuronpedia_inference/` - Main package
- `apps/inference/tests/` - Test suite

**Autointerp**:
- `apps/autointerp/neuronpedia_autointerp/` - Main package
- `apps/autointerp/tests/` - Test suite

### Documentation

- `README.md` - Main repository documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `apps/*/README.md` - Service-specific documentation
- `schemas/README.md` - OpenAPI schema workflow
- `docs/` - Additional documentation

## Environment Variables

### Webapp Key Variables

- `DATABASE_URL` - PostgreSQL connection string
- `NEXTAUTH_SECRET` - NextAuth.js secret
- `NEXTAUTH_URL` - Application URL
- `OPENAI_API_KEY` - For semantic search embeddings
- `INFERENCE_SERVER_URL` - Inference server endpoint
- `INFERENCE_SERVER_SECRET` - Inference server auth
- `AUTOINTERP_SERVER_URL` - Autointerp server endpoint
- `AUTOINTERP_SERVER_SECRET` - Autointerp server auth

### Inference Server Variables

- `MODEL_ID` - TransformerLens model identifier
- `SAE_SETS` - JSON array of SAE sets to load
- `PORT` - Server port (default: 5002)
- `SECRET` - API authentication secret
- `HF_TOKEN` - HuggingFace token for model access

## Quick Reference

### Path Aliases (Webapp)
```typescript
@/*           → apps/webapp/*
@/components  → apps/webapp/components
@/lib         → apps/webapp/lib
@/app         → apps/webapp/app
```

### Code Formatting
```bash
# Webapp
cd apps/webapp && npm run lint:fix && npm run format:write

# Python
cd apps/inference && poetry run ruff check --fix . && poetry run ruff format .
```

### Running Full Stack Locally
```bash
# Terminal 1: Database (via Docker)
make webapp-localhost-run

# Terminal 2: Webapp
cd apps/webapp && npm run dev:localhost

# Terminal 3: Inference (optional)
cd apps/inference && poetry run python start.py

# Terminal 4: Autointerp (optional)
cd apps/autointerp && poetry run python server.py
```

## Additional Resources

- [Main README](README.md) - Setup instructions and architecture
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [OpenAPI Workflow](schemas/README.md) - Schema-driven development
- [Inference Server README](apps/inference/README.md)
- [Autointerp Server README](apps/autointerp/README.md)
- [Neuronpedia Blog](https://neuronpedia.org/blog) - Updates and announcements

---

**Remember**: When in doubt, check the existing codebase for patterns and conventions. The codebase is designed to be consistent and predictable.

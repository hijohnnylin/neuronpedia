#### neuronpedia 🧠🔍 autointerp server

- [repo status](#repo-status)
- [what this is](#what-this-is)
- [setup](#setup)
- [Documentation / Usage (Swagger)](#documentation--usage-swagger)
- [Testing, Linting, and Formatting](#testing-linting-and-formatting)

## repo status

- we haven't had as much time to work on this, but we'd like to collaborate with eleuther to add more explainers, scorers, and include the openai auto-interp types.
- it would be fantastic to standardize on auto-interp formats (eg an `explainerType` should have xyz fields, a `scorerType` should have abc fields, etc)

## what this is

auto-interp explanations and scoring, using eleutherAI's [delphi](https://github.com/EleutherAI/delphi) (formerly `sae-auto-interp`)

the wire format lives in `neuronpedia_autointerp/schemas/` as plain pydantic models, and it is the source of truth: fastapi derives the openapi document from those models, and everything downstream is generated from it. after changing a request or response shape, run `make openapi` to refresh the committed `openapi.json` — a test fails if you forget — then `make webapp-openapi` from the repo root to refresh the typescript types the webapp compiles against.

> ⚠️ **warning:** this is draft documentation. we expect to either have better readmes or use a hosted documentation website.

> ⚠️ **warning:** the eleuther embedding scorer uses an embedding model only supported on CUDA (it won't work on mac mps or cpu)

## setup

1. `uv sync`

2. launch local server

   ```
   # no auto-reload
   uv run uvicorn server:app --host 0.0.0.0 --port 5003 --workers 1
   # with auto-reload
   uv run uvicorn server:app --host 0.0.0.0 --port 5003 --workers 1 --reload
   ```

## Documentation / Usage (Swagger)

FastAPI has a built-in docs + endpoint tester. After running the server, to see interactive docs, go to [http://localhost:5003/docs](http://localhost:5003/docs)

Notes/Caveats:

- You will need to set the YOUR_OPENROUTER_KEY in your test requests.
- If you set a SECRET (not set by default) in your `.env` file, you'll need to add a `x-secret-key` header.

## Testing, Linting, and Formatting

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing, [pyright](https://github.com/microsoft/pyright) for type-checking, and [Ruff](https://docs.astral.sh/ruff/) for formatting and linting.

If you add new code, it would be greatly appreciated if you could add tests in the `tests` directory. You can run the tests with:

```bash
make test
```

Before committing, make sure you format the code with:

```bash
make format
```

Finally, run all CI checks locally with:

```bash
make check-ci
```

If these pass, you're good to go! Open a pull request with your changes.

"""Write the OpenAPI document this server derives from its pydantic models.

``openapi.json`` is committed because the webapp's TypeScript types are generated from it, so a
webapp developer can regenerate them without a python environment -- which matters here, since
installing this one pulls torch and a circuit-tracer or CRM backend.

Run this after changing anything in ``neuronpedia_graph/schemas.py`` or a route signature --
``make openapi`` from this directory. It needs no weights and no GPU: ``app.openapi()`` only
reads route signatures.

Output is sorted with a trailing newline so the diff shows the fields that changed rather than
churn from dict ordering.
"""

import json
import os
from pathlib import Path

# The server refuses to import without these, since serving without a secret would be an open
# endpoint and it fetches weights on startup. Dumping the spec serves nothing and fetches
# nothing, so placeholders keep this runnable in a checkout with no .env. They must be set
# before the import below, which is what reads them.
os.environ.setdefault("SECRET", "openapi-dump-placeholder")
os.environ.setdefault("HF_TOKEN", "openapi-dump-placeholder")
# Any name from TLENS_MODEL_ID_TO_NP_MODEL_ID will do -- which model is loaded does not appear
# anywhere in the spec, and nothing is loaded here regardless.
os.environ.setdefault("MODEL_ID", "google/gemma-2-2b")

from neuronpedia_graph.server import app  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "openapi.json"


def render() -> str:
    """The exact contents the committed file should have.

    Shared with the staleness test so the writer and the check cannot disagree about
    formatting and report drift that isn't there.
    """
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(render())
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

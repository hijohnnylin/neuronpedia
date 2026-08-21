"""Write the OpenAPI document this server derives from its pydantic models.

``openapi.json`` is committed because the webapp's TypeScript types are generated from it, so a
webapp developer can regenerate them without a python environment -- which matters here, since
installing this one pulls torch and vLLM.

Run this after changing a wire model or a route signature -- ``make openapi`` from this
directory. It needs no weights and no GPU: the models are loaded by the lifespan handler, which
importing does not run, and ``app.openapi()`` only reads route signatures.

Note this covers the eight JSON endpoints only. The SSE frames from ``/completion``,
``/describe`` and ``/explain`` are not response bodies and cannot appear in a spec; they are
pinned separately by ``tests/test_frame_contract.py``.

Output is sorted with a trailing newline so the diff shows the fields that changed rather than
churn from dict ordering.
"""

import json
from pathlib import Path

from server import app

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

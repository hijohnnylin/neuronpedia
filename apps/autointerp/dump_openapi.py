"""Write the OpenAPI document this server derives from its pydantic models.

``openapi.json`` is committed because everything downstream is generated from it: the
TypeScript types the webapp compiles against, and the client SDKs the publish job builds.
Committing it means a webapp developer can regenerate those types without a Python
environment.

Run this after changing anything under ``neuronpedia_autointerp/schemas/`` or a route
signature -- ``make openapi`` from this directory.

Output is sorted with a trailing newline so the diff shows the fields that changed rather
than churn from dict ordering.
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

"""Write the OpenAPI document this server derives from its pydantic models.

``openapi.json`` is committed because the webapp's TypeScript types are generated from it,
so a webapp developer can regenerate them without a python environment -- which matters more
here than for the other apps, since installing this one pulls torch.

Run this after changing anything in ``schemas.py`` or a route signature -- ``make openapi``
from this directory. It needs no model loaded and no GPU: ``app.openapi()`` only reads route
signatures, and the weights are only fetched by the lifespan handler on startup.

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

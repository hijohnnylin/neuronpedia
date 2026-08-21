"""Shared guard for the per-request token budget.

Every endpoint that tokenizes caller-supplied text has to reject inputs that exceed
its budget, and each one has to phrase that rejection identically -- clients match on
the message. Keeping the check in one place is what stops the wording, the log line,
and the status code from drifting apart across a dozen endpoints.

Which budget applies is the caller's business: activation endpoints bound against
``activation_token_limit``, steering endpoints against ``token_limit``, and the batch
variants against a per-request share of the former.
"""

import logging

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def reject_if_over_token_limit(n_tokens: int, limit: float, *, suffix: str = "") -> JSONResponse | None:
    """Return a 400 response when ``n_tokens`` exceeds ``limit``, otherwise ``None``.

    ``suffix`` is appended to the error message so batch endpoints can spell out that
    the limit they enforce is per-request rather than the documented total.

    ``limit`` is a float because the batch endpoints divide the configured budget by
    the batch size; the message reports whatever was passed, fraction and all.
    """
    if n_tokens <= limit:
        return None
    logger.error("Text too long: %s tokens, max is %s", n_tokens, limit)
    return JSONResponse(
        content={"error": f"Text too long: {n_tokens} tokens, max is {limit}{suffix}"},
        status_code=400,
    )

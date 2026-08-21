"""How the process reads its runtime knobs: which device, which dtype, which model engine.

These three used to be duplicated between `server.py` and `crm_backend.py`, once per attribution
engine, because `server.py` cannot import `crm_backend` unconditionally -- that module pulls in
llamascopium, which is only installed for the `lm-saes-crm` extra. The copies drifted: both read
`MODEL_DTYPE`, but one fell back to bfloat16 for an unrecognized value while the other returned
None, which reached `ReplacementModel.from_pretrained` as `dtype=None` and silently meant "the
checkpoint's own dtype". Same variable, two meanings, decided by which engine you were running.

Nothing here imports either backend, so both can depend on it, and nothing here reads the
environment at import time -- callers run `load_dotenv()` after their own imports, so a constant
resolved here would miss anything set in `.env`.
"""

import os
from typing import Literal

import torch

# The three engines `ReplacementModel.from_pretrained` accepts, and the single place that list is
# written down. Not every one is valid for every attribution engine: the CRM backend rejects
# `nnsight` separately, because llamascopium reaches the model through methods nnsight's proxies
# do not provide. That check belongs there, next to the code with the requirement.
MODEL_ENGINES = ("nnsight", "transformerlens", "interp_engine")

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_model_engine() -> Literal["nnsight", "transformerlens", "interp_engine"]:
    """Resolve and validate MODEL_ENGINE. Defaults to interp_engine.

    Rejecting an unknown value here means failing before `load_transcoder_from_hub` spends
    several minutes downloading weights -- `from_pretrained` would raise the same error, but only
    after that. It is also what lets the result be passed as the Literal that call wants; an
    unchecked `str` from the environment is not one.

    Read on call rather than at import, because callers run `load_dotenv()` themselves and a
    module-level constant here would be resolved before they got the chance.
    """
    engine = os.getenv("MODEL_ENGINE", "interp_engine")
    if engine not in MODEL_ENGINES:
        raise ValueError(f"MODEL_ENGINE must be one of {MODEL_ENGINES}, got {engine!r}")
    return engine


def get_device() -> torch.device:
    """Resolve DEVICE, else the best device available."""
    device_env = os.environ.get("DEVICE")
    if device_env:
        return torch.device(device_env)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_dtype() -> torch.dtype:
    """Parse MODEL_DTYPE into a torch dtype. Defaults to bfloat16, including for junk values."""
    return _DTYPES.get(os.environ.get("MODEL_DTYPE", "bfloat16"), torch.bfloat16)

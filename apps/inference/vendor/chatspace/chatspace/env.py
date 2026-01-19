from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv


def load_environment(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file if present, without overriding
    variables that are already set in the environment.

    If env_file is not provided, python-dotenv will search for .env in
    the current working directory and parent directories.
    """
    load_dotenv(dotenv_path=env_file, override=False)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value



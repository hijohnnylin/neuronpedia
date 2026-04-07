import argparse
import os
from dataclasses import dataclass

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5010
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_RELOAD_DIR = "neuronpedia_activations"
VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_DEVICES = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class ActivationsConfig:
    host: str
    port: int
    model_id: str
    model_dtype: str | None
    device: str | None
    secret: str | None


@dataclass(frozen=True)
class ActivationsStartConfig:
    host: str
    port: int
    model_id: str
    model_dtype: str | None
    device: str | None
    reload: bool
    reload_dir: str


def parse_start_args() -> ActivationsStartConfig:
    parser = argparse.ArgumentParser(
        description="Start the Neuronpedia activations server."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help="Startup model ID to download/load eagerly.",
    )
    parser.add_argument(
        "--model_dtype",
        default=None,
        choices=list(VALID_DTYPES),
        help="Optional startup dtype override.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=list(VALID_DEVICES),
        help="Optional startup device override.",
    )
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--reload-dir",
        default=DEFAULT_RELOAD_DIR,
        help="Directory to watch for changes when reload is enabled.",
    )
    args = parser.parse_args()
    return ActivationsStartConfig(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        model_dtype=args.model_dtype,
        device=args.device,
        reload=args.reload,
        reload_dir=args.reload_dir,
    )


def apply_start_config_to_env(config: ActivationsStartConfig) -> None:
    if "HOST" not in os.environ:
        os.environ["HOST"] = config.host
    if "PORT" not in os.environ:
        os.environ["PORT"] = str(config.port)
    if "MODEL_ID" not in os.environ:
        os.environ["MODEL_ID"] = config.model_id
    if "MODEL_DTYPE" not in os.environ and config.model_dtype is not None:
        os.environ["MODEL_DTYPE"] = config.model_dtype
    if "DEVICE" not in os.environ and config.device is not None:
        os.environ["DEVICE"] = config.device


def get_config_from_env() -> ActivationsConfig:
    return ActivationsConfig(
        host=os.environ.get("HOST", DEFAULT_HOST),
        port=int(os.environ.get("PORT", str(DEFAULT_PORT))),
        model_id=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        model_dtype=os.environ.get("MODEL_DTYPE"),
        device=os.environ.get("DEVICE"),
        secret=os.environ.get("SECRET"),
    )

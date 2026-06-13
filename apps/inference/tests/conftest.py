import asyncio
import gc
import json
import os
from dataclasses import dataclass

import pytest
import torch
from fastapi.testclient import TestClient

import neuronpedia_inference.server as server
from neuronpedia_inference.args import parse_env_and_args
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.server import app, initialize
from neuronpedia_inference.shared import Model


@dataclass
class ModelTestConfig:
    """Configuration for a model under test."""

    model_id: str
    sae_source_set: str
    sae_selected_sources: list[str]
    bos_token_str: str
    # Feature index known to exist in the selected SAE
    steer_feature_index: int
    # Model embedding dimension (residual stream width)
    dim_model: int


# Model configurations for testing
MODEL_CONFIGS = {
    "gpt2-small": ModelTestConfig(
        model_id="gpt2-small",
        sae_source_set="res-jb",
        sae_selected_sources=["7-res-jb"],
        bos_token_str="<|endoftext|>",
        steer_feature_index=5,
        dim_model=768,
    ),
    "gemma-3-270m": ModelTestConfig(
        model_id="google/gemma-3-270m",
        sae_source_set="gemmascope-2-res-16k",
        sae_selected_sources=["5-gemmascope-2-res-16k"],
        bos_token_str="<bos>",
        steer_feature_index=5,
        dim_model=1152,
    ),
}

# Select which model to test via environment variable, default to gpt2-small
_model_key = os.environ.get("TEST_MODEL", "gpt2-small")
ACTIVE_MODEL_CONFIG = MODEL_CONFIGS[_model_key]

# Export constants for backward compatibility with existing tests
BOS_TOKEN_STR = ACTIVE_MODEL_CONFIG.bos_token_str
TEST_PROMPT = "Hello, world!"
X_SECRET_KEY = "cat"
MODEL_ID = ACTIVE_MODEL_CONFIG.model_id
SAE_SOURCE_SET = ACTIVE_MODEL_CONFIG.sae_source_set
SAE_SELECTED_SOURCES = ACTIVE_MODEL_CONFIG.sae_selected_sources
DIM_MODEL = ACTIVE_MODEL_CONFIG.dim_model
ABS_TOLERANCE = 0.1
N_COMPLETION_TOKENS = 10
TEMPERATURE = 0
STRENGTH = 10.0  # Steering mechanism (feature or vector) specific strength
STRENGTH_MULTIPLIER = 10.0  # Multiplier across all steering mechanisms
FREQ_PENALTY = 0.0
SEED = 42
STEER_SPECIAL_TOKENS = False
STEER_FEATURE_INDEX = ACTIVE_MODEL_CONFIG.steer_feature_index
INVALID_SAE_SOURCE = "fake-source"


@pytest.fixture(scope="session")
def initialize_models():
    """
    Defining the global state of the app with a session-scoped fixture that initializes the model and SAEs.

    This fixture will be run once per test session and will be available to all tests
    that need an initialized model. It uses the same initialization logic as the
    /initialize endpoint.

    The model to test can be selected via TEST_MODEL environment variable:
        TEST_MODEL=gpt2-small pytest ...   (default)
        TEST_MODEL=gemma-3-270m pytest ...
    """
    # Set environment variables for testing using the active model config
    os.environ.update(
        {
            "MODEL_ID": ACTIVE_MODEL_CONFIG.model_id,
            "SAE_SETS": json.dumps([ACTIVE_MODEL_CONFIG.sae_source_set]),
            "MODEL_DTYPE": "float16",
            "SAE_DTYPE": "float32",
            "TOKEN_LIMIT": "500",
            "DEVICE": "cpu",
            "INCLUDE_SAE": json.dumps(
                ACTIVE_MODEL_CONFIG.sae_selected_sources
            ),  # Only load the specific SAE we want
            "EXCLUDE_SAE": json.dumps([]),
            "MAX_LOADED_SAES": "1",
            "SECRET": X_SECRET_KEY,
        }
    )

    # Re-parse args after setting environment variables
    # This is important to refresh the module-level args in the server module
    server.args = parse_env_and_args()

    # Initialize the model and SAEs
    asyncio.run(initialize())

    yield

    # Cleanup
    Config._instance = None
    SAEManager._instance = None
    Model._instance = None  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="session")
def client(initialize_models: None):  # noqa: ARG001
    return TestClient(app)

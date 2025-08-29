import asyncio
import gc
import json
import os

import pytest
import torch
from fastapi.testclient import TestClient

import neuronpedia_inference.server as server
from neuronpedia_inference.args import parse_env_and_args
from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.server import app, initialize
from neuronpedia_inference.shared import Model

BOS_TOKEN_STR = "<|endoftext|>"
TEST_PROMPT = "Hello, world!"
X_SECRET_KEY = "cat"
SAE_SOURCE_SET = "res-jb"
SAE_SELECTED_SOURCES = ["7-res-jb"]
ABS_TOLERANCE = 0.15
N_COMPLETION_TOKENS = 10
TEMPERATURE = 0
STRENGTH = 10.0  # Steering mechanism (feature or vector) specific strength
STRENGTH_MULTIPLIER = 10.0  # Multiplier across all steering mechanisms
FREQ_PENALTY = 0.0
SEED = 42
STEER_SPECIAL_TOKENS = False
STEER_FEATURE_INDEX = 5
INVALID_SAE_SOURCE = "fake-source"

MODEL_ID = "gpt2-small"
TOKEN_LIMIT = "500"
DEVICE = "mps"
MAX_LOADED_SAES = "1"
MODEL_DTYPE = "float16"
SAE_DTYPE = "float32"



@pytest.fixture(scope="session")
def initialize_models():
    """
    Defining the global state of the app with a session-scoped fixture that initializes the model and SAEs.

    This fixture will be run once per test session and will be available to all tests
    that need an initialized model. It uses the same initialization logic as the
    /initialize endpoint.
    """
    # Set environment variables for testing
    os.environ.update(
        {
            "MODEL_ID": MODEL_ID,
            "SAE_SETS": json.dumps([SAE_SOURCE_SET]),
            "MODEL_DTYPE": MODEL_DTYPE,
            "SAE_DTYPE": SAE_DTYPE,
            "TOKEN_LIMIT": TOKEN_LIMIT,
            "DEVICE": DEVICE,
            "INCLUDE_SAE": json.dumps(
                SAE_SELECTED_SOURCES
            ),  # Only load the specific SAE we want
            "EXCLUDE_SAE": json.dumps([]),
            "MAX_LOADED_SAES": MAX_LOADED_SAES,
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

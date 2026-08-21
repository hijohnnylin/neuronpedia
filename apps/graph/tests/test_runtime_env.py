"""Pins the two properties of `runtime_env` that are easy to undo by tidying it up."""

import ast
from pathlib import Path

import pytest
import torch

from neuronpedia_graph.runtime_env import MODEL_ENGINES, get_model_dtype, get_model_engine

MODULE_PATH = Path(__file__).parents[1] / "neuronpedia_graph" / "runtime_env.py"


@pytest.mark.parametrize("engine", MODEL_ENGINES)
def test_every_listed_engine_is_accepted(engine, monkeypatch):
    monkeypatch.setenv("MODEL_ENGINE", engine)
    assert get_model_engine() == engine


def test_unknown_engine_is_rejected(monkeypatch):
    """Otherwise `from_pretrained` raises the same error, but only after downloading weights."""
    monkeypatch.setenv("MODEL_ENGINE", "transfomerlens")
    with pytest.raises(ValueError, match="MODEL_ENGINE must be one of"):
        get_model_engine()


def test_unknown_dtype_falls_back_rather_than_returning_none(monkeypatch):
    """This returned None once, which reached `from_pretrained` as dtype=None and quietly meant
    "the checkpoint's own dtype" -- disagreeing with what the CRM path did with the same value."""
    monkeypatch.setenv("MODEL_DTYPE", "fp16")
    assert get_model_dtype() is torch.bfloat16
    monkeypatch.setenv("MODEL_DTYPE", "float32")
    assert get_model_dtype() is torch.float32


def test_nothing_is_resolved_at_import_time():
    """Both callers run `load_dotenv()` after their imports, so a module-level `os.getenv` here
    would be resolved first and miss anything set in `.env`. Checked structurally because the
    symptom -- one variable silently ignored, only when set that one way -- is invisible in a
    normal test run, which inherits an already-populated environment."""
    module = ast.parse(MODULE_PATH.read_text())
    top_level_env_reads = [
        node
        for statement in module.body
        if not isinstance(statement, ast.FunctionDef)
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute) and node.attr in {"getenv", "environ"}
    ]
    assert top_level_env_reads == [], "runtime_env must read the environment inside functions only"

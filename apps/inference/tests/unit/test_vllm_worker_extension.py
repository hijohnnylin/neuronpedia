"""The engine's vLLM worker extension must be installable in the vLLM we actually run.

vLLM asserts, at worker init, that a ``worker_extension_cls`` shares NO attribute name with
the worker class it is mixed into -- and it checks single-underscore names too, not just
public ones. A collision therefore fails every vLLM boot rather than one endpoint, and it can
appear from a vLLM upgrade alone without anything here changing.

The second half of the file guards the calling convention across the whole repo: worker
functions must be invoked BY NAME, because a callable argument needs
``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` and the engine no longer sets it. That failure only
reproduces once an engine is up, so without these checks it is invisible until GPU CI.

This lives in the inference suite rather than the engine's because it needs vLLM importable,
which the engine's own venv deliberately does not have (vLLM is a Linux/CUDA-only extra).
Importing the worker module needs no GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import interp_engine
import pytest
from interp_engine import WORKER_EXTENSION_CLS, InterpWorkerExtension

# Mirrors vLLM's own check in WorkerBase.init_worker: every non-dunder attribute.
_EXTENSION_ATTRS = sorted(name for name in dir(InterpWorkerExtension) if not name.startswith("__"))


def _worker_classes() -> list[type]:
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import WorkerBase

    return [Worker, WorkerBase]


def test_extension_has_attributes_to_check() -> None:
    """Guards the tests below from passing because the surface went empty."""
    assert len(_EXTENSION_ATTRS) > 10


@pytest.mark.parametrize("worker_cls", _worker_classes(), ids=lambda c: c.__name__)
def test_extension_names_do_not_collide_with_the_worker(worker_cls: type) -> None:
    clashes = [name for name in _EXTENSION_ATTRS if hasattr(worker_cls, name)]
    assert not clashes, (
        f"{worker_cls.__name__} already defines {clashes}; vLLM refuses the extension class. "
        "Rename the method(s) in interp_engine.vllm_plugin and the matching "
        'collective_rpc("...") strings in interp_engine.vllm_backend.'
    )


def test_vllm_accepts_the_extension_path_as_a_config_field() -> None:
    """``worker_extension_cls`` is the mechanism this all rests on; pin that it exists."""
    import dataclasses

    from vllm.config import ParallelConfig

    fields = {f.name for f in dataclasses.fields(ParallelConfig)}
    assert "worker_extension_cls" in fields


def test_extension_path_resolves_the_way_vllm_resolves_it() -> None:
    """vLLM turns the string into the class with ``resolve_obj_by_qualname`` in each worker."""
    from vllm.utils.import_utils import resolve_obj_by_qualname

    assert resolve_obj_by_qualname(WORKER_EXTENSION_CLS) is InterpWorkerExtension


# --- repo-wide calling convention ------------------------------------------------------

_INFERENCE_ROOT = Path(__file__).parents[2]
_ENGINE_ROOT = Path(interp_engine.__file__).parent

# Both deliberately hand a function object to collective_rpc and document the flag in their
# module docstring: one exists to exercise that bare mechanism, the other dumps arbitrary
# worker internals and so cannot be a fixed named method.
_CALLABLE_RPC_ALLOWED = {"vllm_capture_check.py", "vllm_introspect.py"}


def _python_sources() -> list[Path]:
    """Every .py under the inference app and the engine, minus build artifacts."""
    roots = [_INFERENCE_ROOT, _ENGINE_ROOT]
    return [
        p
        for root in roots
        for p in root.rglob("*.py")
        if not any(part.startswith((".", "__pycache__")) or part == "build" for part in p.parts)
    ]


def _collective_rpc_first_args() -> list[tuple[Path, int, ast.expr]]:
    """Every ``*.collective_rpc(first_arg, ...)`` in the repo, as AST (multi-line safe)."""
    found: list[tuple[Path, int, ast.expr]] = []
    for path in _python_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "collective_rpc"
                and node.args
            ):
                found.append((path, node.lineno, node.args[0]))
    return found


def test_there_are_collective_rpc_calls_to_check() -> None:
    """Guards the two tests below from passing because the scan found nothing."""
    assert len(_collective_rpc_first_args()) > 5


def test_no_collective_rpc_passes_a_callable() -> None:
    """A function argument raises TypeError at call time without the insecure-serialization flag.

    vLLM v1 msgpack-encodes the call to an out-of-process engine core and refuses function
    objects. Since the engine stopped setting that flag, passing a callable is a latent
    GPU-only break -- which is exactly how it reached GPU CI once already.
    """
    offenders = [
        f"{path.name}:{lineno}"
        for path, lineno, arg in _collective_rpc_first_args()
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)) and path.name not in _CALLABLE_RPC_ALLOWED
    ]
    assert not offenders, (
        f"collective_rpc called with a non-string first argument at {offenders}. Pass the "
        "worker method NAME instead (see interp_engine.vllm_plugin.InterpWorkerExtension); a "
        "callable needs VLLM_ALLOW_INSECURE_SERIALIZATION=1, which the engine no longer sets."
    )


def test_every_collective_rpc_name_exists_on_a_worker() -> None:
    """A typo'd method name is a GPU-only AttributeError, so resolve them all here."""
    from vllm.v1.worker.gpu_worker import Worker

    unknown = [
        f"{path.name}:{lineno} -> {arg.value!r}"
        for path, lineno, arg in _collective_rpc_first_args()
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and not hasattr(InterpWorkerExtension, arg.value)
        and not hasattr(Worker, arg.value)
    ]
    assert not unknown, (
        f"collective_rpc names that are neither an InterpWorkerExtension method nor a vLLM "
        f"Worker attribute: {unknown}. Fix the name or add the method to "
        "interp_engine.vllm_plugin."
    )

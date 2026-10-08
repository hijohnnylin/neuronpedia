"""``--backend``: one flag names the backend, and the engine checks it against the machine.

The app used to carry one boolean flag per backend, all of which set the same ``FORCE_BACKEND``
string. A third backend would have meant a third flag for the same variable, so the flag now takes
the name -- and a name this machine cannot run is refused by ``select_backend`` before any load,
which is what makes a wrong choice a startup error with the fix in it rather than a late crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neuronpedia_inference import server

START_PY = Path(server.__file__).resolve().parents[1] / "start.py"
_BACKEND_VARS = ("FORCE_BACKEND",)


def _run_start(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run start.py's parse -> env step the way a pod does, and print what it exported."""
    env = {k: v for k, v in os.environ.items() if k not in _BACKEND_VARS}
    code = (
        f"import os, sys; sys.argv = ['start.py', *{list(argv)!r}, '--list_models']; import start; "
        "import neuronpedia_inference.args as a; a.list_available_options = lambda: None; "
        "start.main(); print(repr(os.environ.get('FORCE_BACKEND')))"
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=START_PY.parent, env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("name", ["eager", "vllm", "mlx"])
def test_a_named_backend_reaches_args_py_as_force_backend(name: str) -> None:
    out = _run_start("--backend", name)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == repr(name)


@pytest.mark.parametrize("argv", [(), ("--backend", "auto")])
def test_auto_leaves_the_choice_to_the_engine(argv: tuple[str, ...]) -> None:
    out = _run_start(*argv)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "None"


def test_an_unknown_name_is_refused_by_the_parser() -> None:
    out = _run_start("--backend", "npengine")
    assert out.returncode != 0
    assert "invalid choice" in out.stderr


@pytest.mark.parametrize(("old", "name"), [("--force-vllm", "vllm"), ("--force-eager", "eager")])
def test_the_old_flags_still_start_a_pod_and_say_what_replaced_them(old: str, name: str) -> None:
    out = _run_start(old)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == repr(name)
    assert f"--backend {name}" in out.stderr


def test_naming_a_backend_two_ways_is_refused() -> None:
    out = _run_start("--backend", "eager", "--force-vllm")
    assert out.returncode != 0
    assert "not allowed with" in out.stderr

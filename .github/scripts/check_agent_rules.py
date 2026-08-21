#!/usr/bin/env python3
"""Fail if agent instructions have drifted back into a single vendor's format.

Every coding agent used against this repo should see the same rules. That holds only while
`AGENTS.md` stays the one place project instructions live, and the per-tool files stay pointers at
it rather than content of their own. Both halves are easy to undo by accident: a rule dropped into
`.cursor/rules/` is invisible to Claude Code and Codex, and a nested `AGENTS.md` without a
`CLAUDE.md` beside it is invisible to Claude Code specifically.

Scope is deliberately narrow: every check here guards a way instructions or hooks can go *silently*
missing, because nothing else in the repo notices those. Violations a person has to commit on
purpose, and hazards no checkout here has ever hit, are not worth a rule that someone later has to
read and keep true.

This script checks the mechanical half of the convention described in the "Agent instruction files"
section of AGENTS.md. Run it from the repo root:

    python3 .github/scripts/check_agent_rules.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that never hold instructions, but do hold thousands of files. Only used by the
# non-git fallback; `git ls-files` skips all of them for free.
PRUNE = {
    ".git",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".lint-venvs",
    "__pycache__",
    "dist",
    "node_modules",
    "site-packages",
}

# The whole point of the convention: one real file, read by every harness.
CANONICAL = "AGENTS.md"

# A CLAUDE.md may only import its sibling AGENTS.md. Claude Code resolves a relative import
# against the importing file, so both spellings land on the same place.
ALLOWED_IMPORTS = {f"@{CANONICAL}", f"@./{CANONICAL}"}

# Per-tool files that must exist and must defer to AGENTS.md rather than restate it.
POINTERS = [
    ".github/copilot-instructions.md",
    ".gemini/settings.json",
    ".aider.conf.yml",
]

# Hooks are genuinely tool-specific, but the scripts they run are not, so they live in one
# neutral directory that both harness configs point into.
HOOK_DIR = "utils/agent-hooks"
HOOK_CONFIGS = {
    ".cursor/hooks.json": "Cursor",
    ".claude/settings.json": "Claude Code",
}


def _walk(root: Path):
    """Yield every file under `root`, skipping vendored and generated trees."""
    for path in root.iterdir():
        if path.name in PRUNE:
            continue
        if path.is_dir():
            yield from _walk(path)
        else:
            yield path


@cache
def walk(root: Path) -> tuple[Path, ...]:
    """Every file worth checking. Tracked files only, since untracked ones cannot break CI.

    Falls back to a filesystem walk outside a git checkout, which is slower here only because
    `utils/` carries a large ignored data tree.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return tuple(_walk(root))
    # git still lists a file deleted from the working tree, so stat before handing it back.
    return tuple(
        path for name in out.split("\0") if name and (path := root / name).exists()
    )


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def check_nested_pairs(errors: list[str]) -> None:
    """Every AGENTS.md needs a one-line CLAUDE.md beside it, and vice versa."""
    agents_dirs = {p.parent for p in walk(REPO_ROOT) if p.name == CANONICAL}
    claude_files = {p for p in walk(REPO_ROOT) if p.name == "CLAUDE.md"}

    if REPO_ROOT not in agents_dirs:
        errors.append(
            f"{CANONICAL} is missing from the repo root; it is the source of truth."
        )

    for directory in sorted(agents_dirs, key=rel):
        claude = directory / "CLAUDE.md"
        if not claude.exists():
            errors.append(
                f"{rel(directory / CANONICAL)} has no CLAUDE.md beside it, so Claude Code will "
                f"never load it. Add one containing only `@./{CANONICAL}`."
            )

    for claude in sorted(claude_files, key=rel):
        body = [
            line.strip() for line in claude.read_text().splitlines() if line.strip()
        ]
        if body and set(body) <= ALLOWED_IMPORTS:
            continue
        errors.append(
            f"{rel(claude)} should contain only an import of its sibling {CANONICAL} "
            f"(`@./{CANONICAL}`). Anything else is a rule Claude Code can see and no other "
            f"agent can. Move it into {rel(claude.parent / CANONICAL)}."
        )


def check_cursor_rules(errors: list[str]) -> None:
    """`.cursor/rules` may only hold genuinely Cursor-specific, narrowly scoped rules."""
    for path in walk(REPO_ROOT):
        if path.parent.name == ".cursor" and path.name == "rules" and path.is_file():
            errors.append(
                f"{rel(path)} is a file, but Cursor expects `.cursor/rules/` to be a directory of "
                f"`.mdc` files — so nothing reads it. Move its contents to "
                f"{rel(path.parent.parent / CANONICAL)}."
            )
        if path.suffix != ".mdc":
            continue
        text = path.read_text()
        if "alwaysApply: true" in text:
            errors.append(
                f"{rel(path)} sets `alwaysApply: true`, which makes it always-on project context "
                f"that only Cursor can see. Move it into {CANONICAL}; use a nested {CANONICAL} if "
                f"it should apply to one subtree."
            )


def check_pointers(errors: list[str]) -> None:
    for name in POINTERS:
        path = REPO_ROOT / name
        if not path.exists():
            errors.append(
                f"{name} is missing; it is what points its harness at {CANONICAL}."
            )
        elif CANONICAL not in path.read_text():
            errors.append(
                f"{name} no longer references {CANONICAL}, so its harness reads nothing."
            )


def hook_commands(config: Path) -> list[str]:
    """Pull every command string out of either harness's hook config."""
    data = json.loads(config.read_text())
    found: list[str] = []
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            command = node.get("command")
            if isinstance(command, str):
                found.append(command)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def check_hooks(errors: list[str]) -> None:
    """Every hook a harness declares must be a real, runnable script in the shared directory.

    Deliberately not asserted: that both configs run the same set. The two harnesses have
    different event models -- Cursor's `afterFileEdit`/`stop` against Claude Code's matcher-scoped
    `PostToolUse`/`Stop` -- so a hook that exists for one and not the other can be correct, and a
    parity rule would only get edited away the first time it was.
    """
    for name, harness in HOOK_CONFIGS.items():
        path = REPO_ROOT / name
        if not path.exists():
            errors.append(
                f"{name} is missing, so {harness} runs none of the {HOOK_DIR} hooks."
            )
            continue
        try:
            commands = hook_commands(path)
        except json.JSONDecodeError as exc:
            errors.append(f"{name} is not valid JSON: {exc}")
            continue

        for command in commands:
            # Claude Code needs an absolute path, hence the variable; strip it to compare.
            cleaned = command.replace("$CLAUDE_PROJECT_DIR/", "").replace(
                "${CLAUDE_PROJECT_DIR}/", ""
            )
            if not cleaned.startswith(f"{HOOK_DIR}/"):
                errors.append(
                    f"{name} runs `{command}`, which is outside {HOOK_DIR}/. Hook scripts are "
                    f"shared between harnesses and belong there."
                )
                continue
            script = REPO_ROOT / cleaned
            if not script.exists():
                errors.append(f"{name} references {cleaned}, which does not exist.")
            elif not script.stat().st_mode & 0o111:
                errors.append(
                    f"{cleaned} is not executable, so the hook will silently never run."
                )


def main() -> int:
    errors: list[str] = []
    check_nested_pairs(errors)
    check_cursor_rules(errors)
    check_pointers(errors)
    check_hooks(errors)

    if errors:
        print(
            "Agent instructions are not portable across harnesses:\n", file=sys.stderr
        )
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            f"\nSee the 'Agent instruction files' section of {CANONICAL} for the convention.",
            file=sys.stderr,
        )
        return 1

    print(
        "Agent instructions are portable: one AGENTS.md per scope, every harness pointed at it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

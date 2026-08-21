#!/usr/bin/env bash
# After a file edit, run ruff over Python files written by the agent or by Tab.
# Wired to Cursor's `afterFileEdit` and Claude Code's `PostToolUse`; it reads either
# harness's payload shape, so it lives here rather than under one tool's directory.
#
# Format-on-save in .vscode/settings.json only fires on a human save, so files an agent
# creates or edits reach CI unformatted and fail the `ruff format --check` gate in
# inference-tests.yml. This closes that gap.
#
# Each Python project pins ruff in its own pyproject.toml (all six are on the same range
# today, but they are separate pins and formatter output changes between releases), so
# resolve the binary from the project that owns the file rather than from a single global
# install. Always exits 0: a formatter is not worth interrupting an edit over.
#
# Only `*.py`, matching `extend-exclude = ["*.ipynb", "*.md"]` in every project. ruff 0.16
# formats Python code blocks in Markdown, and the reasons that stays off are written out in
# those pyproject.toml files -- widen this filter only together with them.

set -uo pipefail

input=$(cat)

command -v jq >/dev/null 2>&1 || exit 0

# Cursor puts the path at the top level; Claude Code nests it under the tool's input.
file=$(printf '%s' "$input" | jq -r '.file_path // .tool_input.file_path // empty' 2>/dev/null)
[ -n "$file" ] || exit 0

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac

repo_root=$(cd -- "$(dirname -- "$0")/../.." && pwd) || exit 0
case "$file" in
  /*) ;;
  *) file="$repo_root/$file" ;;
esac
[ -f "$file" ] || exit 0

# Nearest enclosing pyproject.toml is the owning project; its .venv holds the pinned ruff.
ruff=""
dir=$(cd -- "$(dirname -- "$file")" && pwd) || exit 0
while [ -n "$dir" ] && [ "$dir" != "/" ]; do
  if [ -f "$dir/pyproject.toml" ]; then
    [ -x "$dir/.venv/bin/ruff" ] && ruff="$dir/.venv/bin/ruff"
    break
  fi
  [ "$dir" = "$repo_root" ] && break
  dir=$(dirname -- "$dir")
done

if [ -z "$ruff" ]; then
  ruff=$(command -v ruff 2>/dev/null) || exit 0
fi
[ -n "$ruff" ] || exit 0

# --force-exclude makes ruff honor each project's `exclude` settings even for an explicitly
# named path.
"$ruff" format --force-exclude -- "$file" >/dev/null 2>&1

# Import sorting (I) only, rather than the project's full fix set as `make format` uses. This
# fires after every individual edit, so an agent that writes an import before the code using it
# would otherwise have that import deleted (F401) from under it. Sorting cannot remove code.
"$ruff" check --fix-only --select I --force-exclude -- "$file" >/dev/null 2>&1

exit 0

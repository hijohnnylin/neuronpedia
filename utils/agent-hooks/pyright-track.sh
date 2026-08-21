#!/usr/bin/env bash
# After a file edit, record which Python packages the agent touched so the stop hook can
# typecheck them once at end-of-turn (pyright cannot auto-fix the way ruff can).
#
# Wired to Cursor's `afterFileEdit` and Claude Code's `PostToolUse`; it reads either
# harness's payload shape. Always exits 0. State lives under /tmp keyed by session id.

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

# Only packages that pin pyright (and that CI gates) are worth tracking.
conv=$(printf '%s' "$input" | jq -r '.conversation_id // .session_id // "default"' 2>/dev/null)
state_dir="${TMPDIR:-/tmp}/np-pyright-${conv}"
mkdir -p "$state_dir" || exit 0

dir=$(cd -- "$(dirname -- "$file")" && pwd) || exit 0
project=""
while [ -n "$dir" ] && [ "$dir" != "/" ]; do
  if [ -f "$dir/pyproject.toml" ]; then
    # Track if the package configures pyright; the stop hook needs uv or a local venv.
    if grep -q '^\[tool\.pyright\]' "$dir/pyproject.toml" 2>/dev/null; then
      project="$dir"
    fi
    break
  fi
  [ "$dir" = "$repo_root" ] && break
  dir=$(dirname -- "$dir")
done

[ -n "$project" ] || exit 0

# One path per line; duplicates are fine (stop hook uniquifies).
printf '%s\n' "$project" >>"$state_dir/projects"
exit 0

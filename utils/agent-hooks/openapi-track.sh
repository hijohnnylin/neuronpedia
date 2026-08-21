#!/usr/bin/env bash
# After a file edit, record which FastAPI services the agent touched, so the stop hook can
# regenerate their openapi.json once at end-of-turn.
#
# Regeneration imports the app (torch, sentence-transformers, sae-auto-interp) and takes a
# few seconds, which is far too slow to run per edit the way ruff-format.sh does. Hence the
# same track/stop split as the pyright hooks.
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

# Only apps that derive their spec from pydantic have a dump script, which makes its
# presence the discriminator: an app gains this behavior by being converted, not by being
# listed here.
rel=${file#"$repo_root"/}
case "$rel" in
  apps/*) app="apps/$(printf '%s' "${rel#apps/}" | cut -d/ -f1)" ;;
  *) exit 0 ;;
esac
[ -f "$repo_root/$app/dump_openapi.py" ] || exit 0

conv=$(printf '%s' "$input" | jq -r '.conversation_id // .session_id // "default"' 2>/dev/null)
state_dir="${TMPDIR:-/tmp}/np-openapi-${conv}"
mkdir -p "$state_dir" || exit 0

# One path per line; duplicates are fine (stop hook uniquifies).
printf '%s\n' "$app" >>"$state_dir/apps"
exit 0

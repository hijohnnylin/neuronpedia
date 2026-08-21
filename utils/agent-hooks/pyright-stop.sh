#!/usr/bin/env bash
# At end-of-turn, if the agent edited Python in a pyright-gated package, run pyright once
# and auto-follow-up with the failures so they get fixed before the session ends.
#
# Unlike ruff-format.sh (which mutates the file), pyright can only report — so this asks the
# harness to send the agent back to work. Wired to Cursor's `stop` and Claude Code's `Stop`;
# it reads either payload shape and emits both harnesses' keys in one object, since each
# ignores the other's. Retries are capped so a package that cannot reach zero does not loop.

set -uo pipefail

input=$(cat)

command -v jq >/dev/null 2>&1 || exit 0

# Only nudge after a normal completion; never fight an abort/error. Claude Code sends no
# status at all, so absent means completed there.
status=$(printf '%s' "$input" | jq -r '.status // "completed"' 2>/dev/null)
[ "$status" = "completed" ] || exit 0

# Cursor counts the follow-ups for us; Claude Code only says whether this stop is itself the
# result of one, which is the same guard at a coarser resolution.
loop_count=$(printf '%s' "$input" |
  jq -r '.loop_count // (if .stop_hook_active == true then 2 else 0 end)' 2>/dev/null)
# Already followed up once for type errors this conversation — don't thrash.
[ "${loop_count:-0}" -lt 2 ] || exit 0

conv=$(printf '%s' "$input" | jq -r '.conversation_id // .session_id // "default"' 2>/dev/null)
state_dir="${TMPDIR:-/tmp}/np-pyright-${conv}"
projects_file="$state_dir/projects"
[ -f "$projects_file" ] || exit 0

mapfile -t projects < <(sort -u "$projects_file")
rm -f "$projects_file"
rmdir "$state_dir" 2>/dev/null || true
[ "${#projects[@]}" -gt 0 ] || exit 0

report=""
failed=0
repo_root=$(cd -- "$(dirname -- "$0")/../.." && pwd) || exit 0
for project in "${projects[@]}"; do
  [ -n "$project" ] || continue
  # Prefer `uv run pyright` so the project's venv (torch, pytest, clients) is on
  # sys.path — bare `.venv/bin/pyright` often resolves the wrong interpreter.
  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$project/.venv/bin/pyright" ]; then
    continue
  fi
  # No path args: honors each package's [tool.pyright] include/exclude (engine
  # scopes to interp_engine; inference excludes local_scripts / .venv).
  if command -v uv >/dev/null 2>&1; then
    out=$(cd -- "$project" && uv run pyright 2>&1) || true
  else
    out=$(cd -- "$project" && .venv/bin/pyright --pythonpath .venv/bin/python 2>&1) || true
  fi
  if printf '%s\n' "$out" | grep -qE '[1-9][0-9]* errors?'; then
    failed=1
    rel=${project#"$repo_root/"}
    report+="### ${rel:-$project}"$'\n'
    # Keep the follow-up bounded; full dumps blow the context window.
    report+=$(printf '%s\n' "$out" | tail -n 80)$'\n\n'
  fi
done

[ "$failed" -eq 1 ] || exit 0

jq -n --arg r "$report" '
  ("Pyright failed on Python packages edited this turn. Fix these type errors (do not skip with blanket type: ignore unless unavoidable), then re-run `uv run pyright` in each affected package until clean:\n\n" + $r) as $m
  | { followup_message: $m, decision: "block", reason: $m }
'
exit 0

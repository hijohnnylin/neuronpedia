#!/usr/bin/env bash
# At end-of-turn, if the agent edited a pydantic-derived FastAPI service, rewrite that
# service's openapi.json and the typescript types the webapp compiles against.
#
# The spec and the .d.ts are committed build outputs, so leaving them stale is what turns
# "edit a model" back into a multi-step chore -- and a stale spec fails both the drift test
# in the service's own suite and openapi-drift.yml. This hook mutates rather than reporting,
# like ruff-format.sh, so a clean run is silent; only a generator that actually fails is
# worth an agent's turn.
#
# Wired to Cursor's `stop` and Claude Code's `Stop`; it reads either payload shape and emits
# both harnesses' keys in one object, since each ignores the other's.

set -uo pipefail

input=$(cat)

command -v jq >/dev/null 2>&1 || exit 0

# Only regenerate after a normal completion; never fight an abort/error. Claude Code sends no
# status at all, so absent means completed there.
status=$(printf '%s' "$input" | jq -r '.status // "completed"' 2>/dev/null)
[ "$status" = "completed" ] || exit 0

conv=$(printf '%s' "$input" | jq -r '.conversation_id // .session_id // "default"' 2>/dev/null)
state_dir="${TMPDIR:-/tmp}/np-openapi-${conv}"
apps_file="$state_dir/apps"
[ -f "$apps_file" ] || exit 0

mapfile -t apps < <(sort -u "$apps_file")
rm -f "$apps_file"
rmdir "$state_dir" 2>/dev/null || true
[ "${#apps[@]}" -gt 0 ] || exit 0

command -v uv >/dev/null 2>&1 || exit 0
repo_root=$(cd -- "$(dirname -- "$0")/../.." && pwd) || exit 0

report=""
failed=0
regenerated=0
for app in "${apps[@]}"; do
  [ -n "$app" ] || continue
  [ -f "$repo_root/$app/dump_openapi.py" ] || continue
  if out=$(cd -- "$repo_root/$app" && uv run python dump_openapi.py 2>&1); then
    regenerated=1
  else
    failed=1
    report+="### $app openapi.json"$'\n'
    report+=$(printf '%s\n' "$out" | tail -n 40)$'\n\n'
  fi
done

# The webapp's types are generated from those specs, so they go stale together. Skipped
# rather than failed when node_modules is absent: plenty of python-only work never installs it.
if [ "$regenerated" -eq 1 ] && [ -d "$repo_root/apps/webapp/node_modules" ]; then
  if ! out=$(cd -- "$repo_root/apps/webapp" && npm run --silent openapi 2>&1); then
    failed=1
    report+="### apps/webapp/lib/api/*.d.ts"$'\n'
    report+=$(printf '%s\n' "$out" | tail -n 40)$'\n\n'
  fi
fi

[ "$failed" -eq 1 ] || exit 0

jq -n --arg r "$report" '
  ("Regenerating the OpenAPI artifacts failed for a service edited this turn. The committed openapi.json and lib/api/*.d.ts are now stale, which fails the drift test and openapi-drift.yml. Fix the error below, then re-run `make <service>-openapi` and `make webapp-openapi`:\n\n" + $r) as $m
  | { followup_message: $m, decision: "block", reason: $m }
'
exit 0

#!/usr/bin/env bash
#
# Hand a `prisma db push`-managed database over to `prisma migrate deploy`.
#
# `db push` writes no migration history, so `migrate deploy` would start from
# the first migration and fail on it ("type ... already exists"), leaving a
# failed row that blocks every later migration. This marks the migrations the
# database already reflects as applied, so only the genuinely new ones run.
#
# Everything sorting before CUTOFF is marked applied; CUTOFF and later are left
# for `migrate deploy`. Nothing here alters a table -- `migrate resolve` only
# writes to _prisma_migrations.
#
# Usage: scripts/baseline-migrations.sh <cutoff-migration-name>

set -euo pipefail

cd "$(dirname "$0")/.."

CUTOFF="${1:-}"
if [[ -z "$CUTOFF" ]]; then
  echo "Usage: scripts/baseline-migrations.sh <cutoff-migration-name>" >&2
  echo "The cutoff and everything after it stay pending. Migrations:" >&2
  ls -1 prisma/migrations | grep -v migration_lock | tail -5 | sed 's/^/  /' >&2
  exit 1
fi

if [[ ! -d "prisma/migrations/$CUTOFF" ]]; then
  echo "No such migration: $CUTOFF" >&2
  exit 1
fi

# Migrations must bypass the connection pooler, so insist on the direct URL.
if [[ -z "${POSTGRES_URL_NON_POOLING:-}" ]]; then
  echo "POSTGRES_URL_NON_POOLING is not set. Migrations must not run through pgbouncer." >&2
  exit 1
fi

# Show which database this is about to touch, with the password stripped.
TARGET=$(printf '%s' "$POSTGRES_URL_NON_POOLING" | sed -E 's#(//[^:]+):[^@]*@#\1:***@#')
echo "Target: $TARGET"
echo

ALL=$(ls -1 prisma/migrations | grep -v migration_lock | sort)
TO_BASELINE=$(printf '%s\n' "$ALL" | awk -v cutoff="$CUTOFF" '$0 < cutoff')
PENDING=$(printf '%s\n' "$ALL" | awk -v cutoff="$CUTOFF" '$0 >= cutoff')

echo "Will mark as already applied ($(printf '%s\n' "$TO_BASELINE" | grep -c .)):"
printf '%s\n' "$TO_BASELINE" | sed 's/^/  /' | head -3
echo "  ... through $(printf '%s\n' "$TO_BASELINE" | tail -1)"
echo
echo "Will be left for 'prisma migrate deploy' ($(printf '%s\n' "$PENDING" | grep -c .)):"
printf '%s\n' "$PENDING" | sed 's/^/  /'
echo

read -r -p "Type the database name to confirm: " CONFIRM
EXPECTED=$(printf '%s' "$POSTGRES_URL_NON_POOLING" | sed -E 's#.*/([^/?]+)(\?.*)?$#\1#')
if [[ "$CONFIRM" != "$EXPECTED" ]]; then
  echo "Got \"$CONFIRM\", expected \"$EXPECTED\". Nothing changed." >&2
  exit 1
fi
echo

marked=0
already=0
for migration in $TO_BASELINE; do
  if npx --no-install prisma migrate resolve --applied "$migration" >/dev/null 2>&1; then
    marked=$((marked + 1))
  else
    # Already in _prisma_migrations, which is the other thing resolve can mean.
    already=$((already + 1))
  fi
done

echo "Marked applied: $marked. Already recorded: $already."
echo
echo "Now review the pending changes, then apply them:"
echo "  npx prisma migrate diff --from-url \"\$POSTGRES_URL_NON_POOLING\" --to-schema-datamodel prisma/schema.prisma --script"
echo "  npx prisma migrate deploy"

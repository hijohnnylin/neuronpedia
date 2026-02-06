#!/bin/bash

set -e  # Exit on any error

echo "=== Environment Variables Debug ==="
echo "POSTGRES_PRISMA_URL: ${POSTGRES_PRISMA_URL:-NOT_SET}"
echo "POSTGRES_URL_NON_POOLING: ${POSTGRES_URL_NON_POOLING:-NOT_SET}"
echo "===================================="

echo "Generating Prisma Client..."
./node_modules/.bin/prisma generate

echo "Checking database migration status..."
# Check if _prisma_migrations table exists to determine if we need to baseline
if ! ./node_modules/.bin/prisma db execute --stdin <<< "SELECT 1 FROM _prisma_migrations LIMIT 1;" 2>/dev/null; then
  echo "Migration history table missing - checking if database is empty..."

  # Check if any tables exist (excluding pg_* system tables)
  TABLE_COUNT=$(./node_modules/.bin/prisma db execute --stdin <<< "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';" 2>/dev/null | tail -n 1 | tr -d '[:space:]')

  if [ "$TABLE_COUNT" != "0" ] && [ -n "$TABLE_COUNT" ]; then
    echo "ERROR: Database has $TABLE_COUNT tables but no migration history."
    echo "This database was likely initialized with 'prisma db push' instead of migrations."
    echo ""
    echo "To fix this issue, you have two options:"
    echo "1. Drop and recreate the database (recommended for dev/staging)"
    echo "2. Baseline the existing database with: prisma migrate resolve --applied <migration_name>"
    echo ""
    echo "For a fresh start, run: kubectl exec -n superpod postgres-0 -- psql -U postgres -c 'DROP DATABASE IF EXISTS postgres; CREATE DATABASE postgres;'"
    exit 1
  fi

  echo "Database is empty, proceeding with migrations..."
fi

echo "Running database migrations..."
./node_modules/.bin/prisma migrate deploy

echo "Running database seed..."
# Allow seed to fail gracefully if already seeded (non-critical)
./node_modules/.bin/prisma db seed || {
    echo "Seed failed or already seeded. Continuing..."
    exit_code=$?
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 1 ]; then
        echo "Seed command failed with unexpected error code: $exit_code"
        exit $exit_code
    fi
}

echo "Database initialization completed successfully!"
#!/bin/bash

set -e  # Exit on any error
set -a
source .env
set +a

echo "Running database migrations..."
npx prisma migrate deploy

echo "Running database seed..."
npx prisma db seed

echo "Database initialization completed successfully!"
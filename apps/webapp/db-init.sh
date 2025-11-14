#!/bin/bash

set -e  # Exit on any error
set -a
source .env
set +a

echo "Running Prisma db push..."
./node_modules/.bin/prisma db push

echo "Running database migrations..."
./node_modules/.bin/prisma migrate deploy

echo "Running database seed..."
./node_modules/.bin/prisma db seed

echo "Database initialization completed successfully!"
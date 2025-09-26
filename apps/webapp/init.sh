#!/bin/bash

set -a
source .env
set +a

# Start the Next.js application (database operations handled by db-init container)
echo "Starting Next.js application..."
node server.js
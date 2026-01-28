#!/bin/bash

# Environment variables are injected at runtime via docker compose --env-file or Kubernetes ConfigMaps/Secrets
# No .env file sourcing needed - all variables are already available in the container environment

echo "Starting Next.js application..."
node server.js
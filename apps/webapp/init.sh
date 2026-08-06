#!/bin/bash

set -a
source .env
set +a

# Wait for database to be ready and run database operations at runtime
echo "Waiting for database and running Prisma db push..."
if [ -x ./node_modules/.bin/prisma ]; then
	./node_modules/.bin/prisma db push
else
	echo "Prisma CLI not present in runtime image; skipping db push because db-init handles database setup."
fi

# Start the Next.js application
echo "Starting Next.js application..."
node server.js
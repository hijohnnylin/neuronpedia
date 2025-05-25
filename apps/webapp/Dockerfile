# apps/webapp/Dockerfile
# Neuronpedia Webapp Dockerfile - Fixed version following inference pattern
FROM node:18-alpine

# Install system dependencies and curl for health checks
RUN apk add --no-cache curl git

WORKDIR /app

# Copy the entire source first (needed for any local dependencies)
COPY . .

# Install dependencies
RUN if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    else npm install; fi

# Build the application
RUN npm run build

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Change ownership of the app directory
RUN chown -R nextjs:nodejs /app

# Simple fallback start script
RUN echo '#!/bin/sh\n\
echo "Available files:"\n\
ls -la\n\
echo "Node version: $(node --version)"\n\
echo "NPM version: $(npm --version)"\n\
if [ -f "package.json" ]; then\n\
    echo "Package.json found, checking start script..."\n\
    if npm run start 2>/dev/null; then\n\
        echo "Started with npm start"\n\
    elif npm run dev 2>/dev/null; then\n\
        echo "Started with npm dev"\n\
    elif node server.js 2>/dev/null; then\n\
        echo "Started with node server.js"\n\
    elif node index.js 2>/dev/null; then\n\
        echo "Started with node index.js"\n\
    elif node app.js 2>/dev/null; then\n\
        echo "Started with node app.js"\n\
    else\n\
        echo "Could not determine start method, keeping container alive for debugging"\n\
        sleep 3600\n\
    fi\n\
else\n\
    echo "No package.json found, keeping container alive for debugging"\n\
    sleep 3600\n\
fi\n\
' > /app/start.sh && chmod +x /app/start.sh

USER nextjs

# Expose port
EXPOSE 3000

# Environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || \
    curl -f http://localhost:3000/health || \
    curl -f http://localhost:3000/ || exit 1

# Start with the script
CMD ["/app/start.sh"]

# syntax=docker.io/docker/dockerfile:1
FROM node:22-alpine AS base

# Optional custom CA bundle file support
ARG CUSTOM_CA_BUNDLE
# Copy the CA bundle file if provided, otherwise copy nothing (using .dockerignore as a no-op)
COPY ${CUSTOM_CA_BUNDLE:-.dockerignore} /tmp/ca-bundle-temp

# Set up CA certificates and environment if bundle was provided
RUN if [ -f /tmp/ca-bundle-temp ] && [ "${CUSTOM_CA_BUNDLE}" != ".dockerignore" ]; then \
    mkdir -p /usr/local/share/ca-certificates && \
    mv /tmp/ca-bundle-temp /usr/local/share/ca-certificates/custom-ca.crt && \
    cat /usr/local/share/ca-certificates/custom-ca.crt >> /etc/ssl/certs/ca-certificates.crt; \
    else \
    rm -f /tmp/ca-bundle-temp; \
    fi

# Set SSL environment variables if CA bundle was provided
ENV SSL_CERT_FILE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV REQUESTS_CA_BUNDLE=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV GIT_SSL_CAINFO=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}
ENV NODE_TLS_REJECT_UNAUTHORIZED=${CUSTOM_CA_BUNDLE:+0}

# Configure npm if CA bundle was provided
RUN if [ -n "${CUSTOM_CA_BUNDLE}" ] && [ "${CUSTOM_CA_BUNDLE}" != ".dockerignore" ]; then \
    npm config set cafile "/etc/ssl/certs/ca-certificates.crt"; \
    fi

###############################################################################################
# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat ca-certificates openssl && \
    update-ca-certificates

WORKDIR /app
COPY apps/webapp/package*.json ./
RUN npm ci

###############################################################################################
# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app

# Install OpenSSL and bash for the build stage
RUN apk add --no-cache openssl bash

COPY --from=deps /app/node_modules ./node_modules
COPY apps/webapp ./

# Ensure startup script is executable
RUN chmod +x ./init.sh

# Load environment variables from dotenv file
ARG ENV_FILE=.env.localhost
COPY ${ENV_FILE} .env

# Build with environment variables loaded from dotenv
# Force regenerate Prisma client with correct binary target
RUN bash -c 'set -a && source .env && set +a && npx prisma generate && npm run build'

###############################################################################################
# Database initialization image (has access to ts-node and dev dependencies)
FROM base AS db-init
WORKDIR /app

# Install OpenSSL and bash
RUN apk add --no-cache openssl bash

# Copy all dependencies (including devDependencies for ts-node)
COPY --from=deps /app/node_modules ./node_modules
COPY apps/webapp ./

# Install ts-node globally for seeding
RUN npm install -g ts-node typescript

# Load environment variables from dotenv file
ARG ENV_FILE=.env.localhost
COPY ${ENV_FILE} .env

# Generate Prisma client for db operations
RUN bash -c 'set -a && source .env && set +a && npx prisma generate'

# Make db-init script executable
RUN chmod +x db-init.sh

CMD ["./db-init.sh"]

###############################################################################################
# Production image (minimal runtime)
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

# Install OpenSSL for runtime
RUN apk add --no-cache openssl bash

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

# Copy Prisma files for runtime database operations
COPY --from=builder --chown=nextjs:nodejs /app/prisma ./prisma

# Copy startup script for runtime
COPY --from=builder --chown=nextjs:nodejs /app/init.sh ./init.sh

# Copy environment file for runtime
COPY --from=builder --chown=nextjs:nodejs /app/.env ./.env

USER nextjs

EXPOSE 3000
ENV HOSTNAME="0.0.0.0" PORT=3000

CMD ["/app/init.sh"]
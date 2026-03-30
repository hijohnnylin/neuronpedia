# syntax=docker.io/docker/dockerfile:1

FROM node:22-alpine3.22 AS base

# The ``RUN apk add --no-cache openssl bash`` step might fail without the following ``sed`` command
# Details: https://github.com/alpinelinux/docker-alpine/issues/98#issuecomment-679278499
RUN sed -i 's/https/http/' /etc/apk/repositories

# Optional custom CA bundle file support
ARG CUSTOM_CA_BUNDLE
# Copy the CA bundle file if provided, otherwise copy nothing (using .nocustomca as a no-op)
COPY ${CUSTOM_CA_BUNDLE:-.nocustomca} /tmp/ca-bundle-temp

# Set up CA certificates and environment if bundle was provided
RUN if [ -f /tmp/ca-bundle-temp ] && [ "${CUSTOM_CA_BUNDLE}" != ".nocustomca" ]; then \
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
# ENV NODE_TLS_REJECT_UNAUTHORIZED=${CUSTOM_CA_BUNDLE:+0}
ENV NODE_EXTRA_CA_CERTS=${CUSTOM_CA_BUNDLE:+/etc/ssl/certs/ca-certificates.crt}

# Configure npm if CA bundle was provided
RUN if [ -n "${CUSTOM_CA_BUNDLE}" ] && [ "${CUSTOM_CA_BUNDLE}" != ".nocustomca" ]; then \
    npm config set cafile "/etc/ssl/certs/ca-certificates.crt"; \
    fi

# Install OpenSSL and bash
RUN apk add --no-cache openssl bash

# Build-time environment variables (only NEXT_PUBLIC_* are inlined into JS bundle)
# Reference: https://nextjs.org/docs/pages/building-your-application/configuring/environment-variables
# For local/Docker Compose: Uses env_file at runtime (simpler)
# For K8s/production: Pass these as build args in CI, inject others via ConfigMaps/Secrets at runtime
ARG NEXT_PUBLIC_URL
ARG NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS
ARG NEXT_PUBLIC_DEFAULT_MODELID
ARG NEXT_PUBLIC_DEFAULT_SOURCESET
ARG NEXT_PUBLIC_DEFAULT_SOURCE
ARG NEXT_PUBLIC_DEFAULT_RELEASE_NAME
ARG NEXT_PUBLIC_DEFAULT_STEER_MODEL
ARG NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS
ARG NEXT_PUBLIC_ENABLE_SIGNIN
ARG NEXT_PUBLIC_DEMO_MODE
ARG NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH
ARG NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY

# Convert ARGs to ENVs so they're available in child stages during build
# Note: All other (non-NEXT_PUBLIC_*) variables should be injected at runtime
ENV NEXT_PUBLIC_URL=${NEXT_PUBLIC_URL} \
    NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS=${NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS} \
    NEXT_PUBLIC_DEFAULT_MODELID=${NEXT_PUBLIC_DEFAULT_MODELID} \
    NEXT_PUBLIC_DEFAULT_SOURCESET=${NEXT_PUBLIC_DEFAULT_SOURCESET} \
    NEXT_PUBLIC_DEFAULT_SOURCE=${NEXT_PUBLIC_DEFAULT_SOURCE} \
    NEXT_PUBLIC_DEFAULT_RELEASE_NAME=${NEXT_PUBLIC_DEFAULT_RELEASE_NAME} \
    NEXT_PUBLIC_DEFAULT_STEER_MODEL=${NEXT_PUBLIC_DEFAULT_STEER_MODEL} \
    NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS=${NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS} \
    NEXT_PUBLIC_ENABLE_SIGNIN=${NEXT_PUBLIC_ENABLE_SIGNIN} \
    NEXT_PUBLIC_DEMO_MODE=${NEXT_PUBLIC_DEMO_MODE} \
    NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH=${NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH} \
    NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY=${NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY}

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

COPY --from=deps /app/node_modules ./node_modules
COPY apps/webapp ./

# Ensure startup script is executable
RUN chmod +x ./init.sh

# Build without database operations - only generate Prisma client and build Next.js
# Environment variables are inherited from base stage
RUN npm run build:simple

###############################################################################################
# Database initialization image (has access to ts-node and dev dependencies)
FROM base AS db-init
WORKDIR /app

# Copy all dependencies (including devDependencies for ts-node)
COPY --from=deps /app/node_modules ./node_modules
COPY apps/webapp ./

# Install ts-node globally for seeding
RUN npm install -g ts-node typescript

# Make db-init script executable
RUN chmod +x db-init.sh

# Note: Prisma client is already generated in builder stage (via npm run build:simple)
# This stage only runs db push, migrations, and seeding against live database
CMD ["./db-init.sh"]

###############################################################################################
# Production image (minimal runtime)
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

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

# NOTE: .env files are NOT copied to production image
# Environment variables are injected at runtime via:
# - Kubernetes: ConfigMaps/Secrets
# - Docker Compose: --env-file flag (e.g., docker compose --env-file .env.localhost up)

USER nextjs

EXPOSE 3000
ENV HOSTNAME="0.0.0.0" PORT=3000

CMD ["/app/init.sh"]
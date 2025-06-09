FROM node:22
RUN mkdir /app
WORKDIR /app

# Copy package files first
COPY apps/webapp/package.json apps/webapp/package-lock.json* /app/apps/webapp/


# Install dependencies in a seperate layer
RUN cd apps/webapp && npm install && \
    npm install -g prisma # needed for prisma in build command

# Copy application code after dependencies
COPY apps/webapp apps/webapp

WORKDIR /app/apps/webapp

# Ensure we load the environment variables so that `next start` picks them up
ARG ENV_FILE=.env.demo
COPY ${ENV_FILE} /app/apps/webapp/.env.production
SHELL ["/bin/bash", "-c"]
RUN set -a && source /app/apps/webapp/.env.production && set +a && \
    npm run build:simple

EXPOSE 3000

ENV HOSTNAME=0.0.0.0
ENV PORT=3000

CMD ["npm", "run","start"]

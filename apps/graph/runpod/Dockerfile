FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /

# Install Poetry
RUN pip install --upgrade pip \
    && pip install poetry

# Copy only pyproject.toml and poetry.lock first to leverage Docker cache
COPY pyproject.toml poetry.lock* /

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry lock && poetry install --no-interaction --no-ansi --no-root

# Copy your handler file
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
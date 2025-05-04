# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install curl and needed dependencies
RUN apt-get update && apt-get install -y curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the local script into the container
COPY . /app

# Install dependencies using uv
RUN uv venv
RUN . .venv/bin/activate && uv pip install -e .

ARG BUILD_COMMIT="unknown"
ARG BUILD_BRANCH="main"

ENV BUILD_COMMIT=${BUILD_COMMIT} \
    BUILD_BRANCH=${BUILD_BRANCH}
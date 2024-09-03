# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install curl to download Poetry
RUN apt-get update && apt-get install -y curl

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the local script into the container
COPY . /app

# Change directory 
RUN cd kfp-env

# Install dependencies defined in pyproject.toml
RUN poetry install --no-root

# Change directory 
RUN cd..

# Install any needed packages specified in requirements.txt
RUN pip install -e . --no-deps

ARG BUILD_COMMIT="unknown"
ARG BUILD_BRANCH="main"

ENV BUILD_COMMIT=${BUILD_COMMIT} \
    BUILD_BRANCH=${BUILD_BRANCH}
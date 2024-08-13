FROM gcr.io/shopify-base-images/ubuntu/jammy:latest

ARG LOG_CONFIG_FILE
ENV log_config=$LOG_CONFIG_FILE
# Install python - For now we use the jammy's default Python version 3.10
# This will change when we move to poetry
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# Install Cython directly because of this issue https://github.com/yaml/pyyaml/issues/601
RUN \
    --mount=type=secret,id=pip,dst=/etc/pip.conf \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry poetry-plugin-export && \
    poetry export --without dev --without-urls --output requirements.txt && \
    pip uninstall -y poetry poetry-plugin-export && \
    pip install "Cython<3.0" "pyyaml==5.4.1" --no-build-isolation && \
    pip install --no-cache-dir -r requirements.txt


WORKDIR /app
COPY . /app

# create revision file to store git commit hash
RUN --mount=source=.git,target=.git,type=bind bash ./scripts/ci/create-revision-file.sh
RUN cat /app/src/workflows/REVISION
RUN rm -Rf .git/


ENTRYPOINT uvicorn src.combiner.main:app --host 0.0.0.0 --port 8080 --log-config $log_config --workers 4

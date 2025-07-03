#!/bin/bash

# Debug script for testing deployment locally
# This simulates the Vertex AI deployment environment

set -e

echo "=== ML Pipeline Deployment Debug Script ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Test Python environment
print_status "Testing Python environment..."
python3 -c "import sys; print(f'Python version: {sys.version}')"

# Step 2: Test local server
print_status "Running local server tests..."
python3 test_server_locally.py

# Step 3: Build Docker image
print_status "Building Docker image..."
if docker build -t ml-pipelines-debug .; then
    print_status "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Step 4: Test Docker container startup
print_status "Testing Docker container startup..."

# Create test model directory
TEST_MODEL_DIR="/tmp/ml-pipeline-test-model"
mkdir -p "$TEST_MODEL_DIR"

# Check if we have a real model file from pipeline outputs
if [ -f "model.joblib" ]; then
    cp model.joblib "$TEST_MODEL_DIR/"
    print_status "Using existing model.joblib"
else
    print_warning "No model.joblib found. Creating dummy model..."
    python3 -c "
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
model = RandomForestClassifier(random_state=42)
model.fit(iris.data, iris.target)
joblib.dump(model, '$TEST_MODEL_DIR/model.joblib')
print('Dummy model created')
"
fi

# Step 5: Run container with health check
print_status "Starting Docker container..."

# Kill any existing container
docker rm -f ml-pipeline-test 2>/dev/null || true

# Start container in background
docker run -d \
    --name ml-pipeline-test \
    -p 8080:8080 \
    -e AIP_STORAGE_URI=/app/model \
    -v "$TEST_MODEL_DIR:/app/model" \
    ml-pipelines-debug

# Wait for container to start
sleep 5

# Step 6: Test endpoints
print_status "Testing container endpoints..."

# Check if container is running
if ! docker ps | grep ml-pipeline-test; then
    print_error "Container is not running!"
    docker logs ml-pipeline-test
    exit 1
fi

# Test health endpoint
print_status "Testing health endpoint..."
if curl -f http://localhost:8080/health/live; then
    print_status "Health endpoint working"
else
    print_error "Health endpoint failed"
    docker logs ml-pipeline-test
    exit 1
fi

# Test prediction endpoint
print_status "Testing prediction endpoint..."
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{
        "instances": [{
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }]
    }' || {
    print_error "Prediction endpoint failed"
    docker logs ml-pipeline-test
    exit 1
}

print_status "All tests passed! Container is working correctly."

# Step 7: Show logs for debugging
print_status "Container logs:"
docker logs ml-pipeline-test

# Cleanup
print_status "Cleaning up..."
docker stop ml-pipeline-test
docker rm ml-pipeline-test

print_status "Debug script completed successfully!"
print_status "If this works but Vertex AI deployment fails, check:"
print_status "1. Service account permissions"
print_status "2. VPC/firewall settings"  
print_status "3. Resource quotas"
print_status "4. Container registry access"
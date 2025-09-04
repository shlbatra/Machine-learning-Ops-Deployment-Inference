#!/bin/bash

# Kafka Setup Script for ML Pipelines
echo "Setting up Kafka for ML Pipeline inference..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Start Kafka services
echo "Starting Kafka services..."
docker-compose -f docker-compose.kafka.yml up -d

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 30

# Create the iris-inference-data topic
echo "Creating Kafka topic: iris-inference-data"
docker exec kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --replication-factor 1 \
    --partitions 3 \
    --topic iris-inference-data

# List topics to verify creation
echo "Verifying topic creation..."
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo "Kafka setup complete!"
echo "Services running:"
echo "- Zookeeper: localhost:2181"
echo "- Kafka: localhost:9092"
echo "- Kafka UI: http://localhost:8080"
echo ""
echo "To start the data producer:"
echo "docker-compose -f docker-compose.kafka.yml up iris-data-producer"
echo ""
echo "To stop all services:"
echo "docker-compose -f docker-compose.kafka.yml down"
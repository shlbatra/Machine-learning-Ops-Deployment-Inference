#!/usr/bin/env python3
"""
Test script for GCP Managed Kafka configuration.
This script validates that the Kafka configuration is working correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ml_pipelines_kfp.iris_xgboost.constants import (
    KAFKA_BOOTSTRAP_SERVERS_LOCAL,
    KAFKA_TOPIC,
    PROJECT_ID
)
from ml_pipelines_kfp.iris_xgboost.kafka_producer import IrisDataProducer


def test_kafka_connection():
    """Test connection to GCP Managed Kafka cluster."""
    print("Testing GCP Managed Kafka configuration...")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS_LOCAL}")
    print(f"Topic: {KAFKA_TOPIC}")
    
    try:
        # Test producer connection
        print("\nTesting Kafka producer connection...")
        producer = IrisDataProducer(
            kafka_servers=KAFKA_BOOTSTRAP_SERVERS_LOCAL,
            topic=KAFKA_TOPIC,
            batch_size=1,
            delay_seconds=1.0
        )
        
        # Send a single test message
        print("Sending test message...")
        producer.send_batch()
        
        print("✓ Producer test successful!")
        producer.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Kafka connection test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the Kafka cluster is running: ./setup_managed_kafka.sh")
        print("2. Check that your GCP credentials are configured")
        print("3. Verify the bootstrap servers endpoint in constants.py")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("GCP Managed Kafka Configuration Test")
    print("=" * 60)
    
    success = test_kafka_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! Your GCP Managed Kafka setup is ready.")
        print("\nNext steps:")
        print("1. Run the producer: python src/ml_pipelines_kfp/iris_xgboost/kafka_producer.py --kafka-servers=\"{}\"".format(KAFKA_BOOTSTRAP_SERVERS_LOCAL))
        print("2. Test the KFP pipeline with Kafka data source")
    else:
        print("✗ Tests failed. Please check the configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
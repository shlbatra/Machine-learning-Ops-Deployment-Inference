#!/usr/bin/env python3
"""
Test script for Cloud Pub/Sub configuration.
This script validates that the Pub/Sub setup is working correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ml_pipelines_kfp.iris_xgboost.constants import (
    PUBSUB_TOPIC,
    PUBSUB_SUBSCRIPTION,
    PROJECT_ID
)
from ml_pipelines_kfp.iris_xgboost.pubsub_producer import IrisDataPubSubProducer


def test_pubsub_connection():
    """Test connection to Cloud Pub/Sub."""
    print("Testing Cloud Pub/Sub configuration...")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Topic: {PUBSUB_TOPIC}")
    print(f"Subscription: {PUBSUB_SUBSCRIPTION}")
    
    try:
        # Test producer connection
        print("\nTesting Pub/Sub producer connection...")
        producer = IrisDataPubSubProducer(
            project_id=PROJECT_ID,
            topic_name=PUBSUB_TOPIC,
            batch_size=3,
            delay_seconds=1.0
        )
        
        # Send a test batch
        print("Sending test batch...")
        producer.send_batch()
        
        print("✓ Producer test successful!")
        producer.close()
        
        # Test subscription creation
        print("\nTesting subscription creation...")
        from google.cloud import pubsub_v1
        
        # Use default credentials
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(PROJECT_ID, PUBSUB_SUBSCRIPTION)
        topic_path = subscriber.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        
        try:
            subscriber.get_subscription(request={"subscription": subscription_path})
            print(f"✓ Subscription {PUBSUB_SUBSCRIPTION} already exists")
        except Exception:
            subscriber.create_subscription(
                request={
                    "name": subscription_path,
                    "topic": topic_path,
                    "ack_deadline_seconds": 60
                }
            )
            print(f"✓ Created subscription {PUBSUB_SUBSCRIPTION}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pub/Sub connection test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have the correct GCP credentials configured")
        print("2. Check that the Pub/Sub API is enabled")
        print("3. Verify your service account has Pub/Sub permissions")
        return False


def test_message_flow():
    """Test end-to-end message flow."""
    print("\n" + "="*50)
    print("Testing Message Flow")
    print("="*50)
    
    try:
        from google.cloud import pubsub_v1
        import json
        
        # Initialize clients with default credentials
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(PROJECT_ID, PUBSUB_SUBSCRIPTION)
        
        messages_received = []
        
        def callback(message):
            try:
                data = json.loads(message.data.decode('utf-8'))
                messages_received.append(data)
                print(f"✓ Received message: Sample ID {data['sample_id']}")
                message.ack()
            except Exception as e:
                print(f"✗ Error processing message: {e}")
                message.nack()
        
        # Start subscriber
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Listening for messages on {subscription_path}...")
        
        # Send some test messages
        producer = IrisDataPubSubProducer(
            project_id=PROJECT_ID,
            topic_name=PUBSUB_TOPIC,
            batch_size=2
        )
        
        print("Sending test messages...")
        producer.send_batch()
        
        # Wait for messages
        print("Waiting for messages (10 seconds)...")
        try:
            streaming_pull_future.result(timeout=10)
        except:
            streaming_pull_future.cancel()
        
        print(f"✓ Received {len(messages_received)} messages")
        
        if len(messages_received) > 0:
            print("✓ Message flow test successful!")
            return True
        else:
            print("⚠ No messages received - check topic/subscription configuration")
            return False
            
    except Exception as e:
        print(f"✗ Message flow test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Cloud Pub/Sub Configuration Test")
    print("=" * 60)
    
    # Test basic connection
    connection_success = test_pubsub_connection()
    
    if connection_success:
        # Test message flow
        flow_success = test_message_flow()
        
        print("\n" + "=" * 60)
        if connection_success and flow_success:
            print("✓ All tests passed! Your Cloud Pub/Sub setup is ready.")
            print("\nNext steps:")
            print(f"1. Run the producer: python src/ml_pipelines_kfp/iris_xgboost/pubsub_producer.py --project-id={PROJECT_ID}")
            print("2. Test the KFP pipeline with Pub/Sub data source")
            print("3. Monitor messages in GCP Console: https://console.cloud.google.com/cloudpubsub")
        else:
            print("⚠ Some tests failed. Check the configuration.")
    else:
        print("\n" + "=" * 60)
        print("✗ Connection test failed. Please check the configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
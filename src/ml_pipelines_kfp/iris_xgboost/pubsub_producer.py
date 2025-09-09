import json
import random
import time
from datetime import datetime
from typing import Dict, Any
import logging
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.publisher.futures import Future

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisDataPubSubProducer:
    def __init__(self, 
                 project_id: str,
                 topic_name: str,
                 batch_size: int = 10,
                 delay_seconds: float = 5.0):
        self.project_id = project_id
        self.topic_name = topic_name
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        
        # Initialize Pub/Sub publisher using default credentials
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_name)
        
        # Create topic if it doesn't exist
        self._create_topic_if_not_exists()
        
    def _create_topic_if_not_exists(self):
        """Create the Pub/Sub topic if it doesn't exist."""
        try:
            self.publisher.get_topic(request={"topic": self.topic_path})
            logger.info(f"Topic {self.topic_path} already exists")
        except Exception:
            try:
                self.publisher.create_topic(request={"name": self.topic_path})
                logger.info(f"Created topic {self.topic_path}")
            except Exception as e:
                logger.error(f"Failed to create topic: {e}")
                raise
    
    def generate_iris_sample(self) -> Dict[str, Any]:
        """Generate a random Iris data sample."""
        return {
            "sepal_length": round(random.uniform(4.0, 8.0), 1),
            "sepal_width": round(random.uniform(2.0, 4.5), 1),
            "petal_length": round(random.uniform(1.0, 7.0), 1),
            "petal_width": round(random.uniform(0.1, 2.5), 1),
            "timestamp": datetime.utcnow().isoformat(),
            "sample_id": random.randint(1000, 9999)
        }
    
    def _publish_callback(self, future: Future):
        """Callback for publish operations."""
        try:
            message_id = future.result()
            logger.debug(f"Published message with ID: {message_id}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
    
    def send_batch(self) -> None:
        """Send a batch of Iris samples to Pub/Sub."""
        batch_data = []
        futures = []
        
        for i in range(self.batch_size):
            sample = self.generate_iris_sample()
            batch_data.append(sample)
            
            # Convert to JSON and publish
            message_data = json.dumps(sample).encode('utf-8')
            
            # Add attributes for message routing/filtering
            attributes = {
                "sample_id": str(sample["sample_id"]),
                "source": "iris-data-generator",
                "timestamp": sample["timestamp"]
            }
            
            try:
                future = self.publisher.publish(
                    self.topic_path,
                    data=message_data,
                    **attributes
                )
                future.add_done_callback(self._publish_callback)
                futures.append(future)
                
            except Exception as e:
                logger.error(f"Error publishing message: {e}")
        
        # Wait for all messages to be published
        for future in futures:
            try:
                future.result(timeout=30)  # 30 second timeout
            except Exception as e:
                logger.error(f"Message publish failed: {e}")
        
        logger.info(f"Sent batch of {len(batch_data)} samples to topic {self.topic_name}")
    
    def start_continuous_production(self, duration_minutes: int = None):
        """Start continuous data production to Pub/Sub."""
        logger.info(f"Starting continuous data production to topic: {self.topic_name}")
        logger.info(f"Batch size: {self.batch_size}, Delay: {self.delay_seconds}s")
        
        start_time = time.time()
        batch_count = 0
        
        try:
            while True:
                self.send_batch()
                batch_count += 1
                
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        logger.info(f"Production completed after {duration_minutes} minutes")
                        break
                
                time.sleep(self.delay_seconds)
                
        except KeyboardInterrupt:
            logger.info("Production stopped by user")
        except Exception as e:
            logger.error(f"Production error: {e}")
        finally:
            logger.info(f"Total batches sent: {batch_count}")
            self.close()
    
    def close(self):
        """Close the publisher."""
        if self.publisher:
            # Pub/Sub publisher doesn't need explicit closing
            logger.info("Pub/Sub producer closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random Iris data for Pub/Sub')
    parser.add_argument('--project-id', required=True,
                       help='GCP Project ID')
    parser.add_argument('--topic', default='iris-inference-data',
                       help='Pub/Sub topic name')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of samples per batch')
    parser.add_argument('--delay', type=float, default=5.0,
                       help='Delay between batches in seconds')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in minutes (infinite if not specified)')
    
    args = parser.parse_args()
    
    producer = IrisDataPubSubProducer(
        project_id=args.project_id,
        topic_name=args.topic,
        batch_size=args.batch_size,
        delay_seconds=args.delay
    )
    
    producer.start_continuous_production(duration_minutes=args.duration)


if __name__ == "__main__":
    main()
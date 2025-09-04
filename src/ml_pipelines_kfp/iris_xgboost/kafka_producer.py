import json
import random
import time
from datetime import datetime
from typing import Dict, Any
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisDataProducer:
    def __init__(self, 
                 kafka_servers: str = "localhost:9092",
                 topic: str = "iris-inference-data",
                 batch_size: int = 10,
                 delay_seconds: float = 5.0):
        self.kafka_servers = kafka_servers
        self.topic = topic
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None,
            retries=5,
            retry_backoff_ms=100,
            request_timeout_ms=30000
        )
        
    def generate_iris_sample(self) -> Dict[str, Any]:
        return {
            "sepal_length": round(random.uniform(4.0, 8.0), 1),
            "sepal_width": round(random.uniform(2.0, 4.5), 1),
            "petal_length": round(random.uniform(1.0, 7.0), 1),
            "petal_width": round(random.uniform(0.1, 2.5), 1),
            "timestamp": datetime.utcnow().isoformat(),
            "sample_id": random.randint(1000, 9999)
        }
    
    def send_batch(self) -> None:
        batch_data = []
        for i in range(self.batch_size):
            sample = self.generate_iris_sample()
            batch_data.append(sample)
            
            try:
                future = self.producer.send(
                    topic=self.topic,
                    key=str(sample["sample_id"]),
                    value=sample
                )
                future.add_callback(self._on_send_success)
                future.add_errback(self._on_send_error)
                
            except Exception as e:
                logger.error(f"Error sending message: {e}")
        
        self.producer.flush()
        logger.info(f"Sent batch of {len(batch_data)} samples to topic {self.topic}")
    
    def _on_send_success(self, record_metadata):
        logger.debug(f"Message sent to {record_metadata.topic} partition {record_metadata.partition}")
    
    def _on_send_error(self, excp):
        logger.error(f"Error sending message: {excp}")
    
    def start_continuous_production(self, duration_minutes: int = None):
        logger.info(f"Starting continuous data production to topic: {self.topic}")
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
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random Iris data for Kafka')
    parser.add_argument('--kafka-servers', default='localhost:9092', 
                       help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='iris-inference-data',
                       help='Kafka topic name')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of samples per batch')
    parser.add_argument('--delay', type=float, default=5.0,
                       help='Delay between batches in seconds')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in minutes (infinite if not specified)')
    
    args = parser.parse_args()
    
    producer = IrisDataProducer(
        kafka_servers=args.kafka_servers,
        topic=args.topic,
        batch_size=args.batch_size,
        delay_seconds=args.delay
    )
    
    producer.start_continuous_production(duration_minutes=args.duration)


if __name__ == "__main__":
    main()
import kfp
from kfp import dsl
from typing import NamedTuple


def pubsub_data_source(
    project_id: str,
    topic_name: str,
    subscription_name: str,
    bq_dataset: str,
    bq_table: str,
    batch_size: int = 100,
    timeout_seconds: int = 300
) -> NamedTuple("PubSubOutputs", [("dataset", kfp.dsl.Dataset)]):
    """
    Kubeflow component that consumes data from Pub/Sub topic and stores in BigQuery.
    """
    
    @dsl.component(
        base_image="python:3.10-slim",
        packages_to_install=[
            "google-cloud-pubsub==2.18.1",
            "google-cloud-bigquery==3.11.4",
            "numpy==1.24.3",
            "pandas==2.0.3",
            "pyarrow==12.0.1",
            "google-auth==2.23.3"
        ]
    )
    def pubsub_consumer_op(
        project_id: str,
        topic_name: str,
        subscription_name: str,
        bq_dataset: str,
        bq_table: str,
        batch_size: int,
        timeout_seconds: int,
        dataset: dsl.Output[dsl.Dataset]
    ):
        import json
        import time
        from datetime import datetime
        import logging
        import pandas as pd
        from google.cloud import pubsub_v1, bigquery
        from google.cloud.exceptions import NotFound
        from concurrent.futures import ThreadPoolExecutor
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Initialize clients
        subscriber = pubsub_v1.SubscriberClient()
        bq_client = bigquery.Client(project=project_id)
        
        # Create subscription path
        subscription_path = subscriber.subscription_path(project_id, subscription_name)
        topic_path = subscriber.topic_path(project_id, topic_name)
        
        # Create subscription if it doesn't exist
        try:
            subscriber.get_subscription(request={"subscription": subscription_path})
            logger.info(f"Subscription {subscription_path} already exists")
        except Exception:
            try:
                subscriber.create_subscription(
                    request={
                        "name": subscription_path,
                        "topic": topic_path,
                        "ack_deadline_seconds": 60
                    }
                )
                logger.info(f"Created subscription {subscription_path}")
            except Exception as e:
                logger.error(f"Failed to create subscription: {e}")
                raise
        
        # BigQuery setup
        table_id = f"{project_id}.{bq_dataset}.{bq_table}"
        
        # Create BigQuery table if it doesn't exist
        schema = [
            bigquery.SchemaField("SepalLengthCm", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("SepalWidthCm", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("PetalLengthCm", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("PetalWidthCm", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("sample_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("message_id", "STRING", mode="REQUIRED")
        ]
        
        try:
            table = bq_client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            table = bigquery.Table(table_id, schema=schema)
            table = bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
        
        # Message processing
        consumed_data = []
        start_time = time.time()
        
        def callback(message):
            """Process individual Pub/Sub message."""
            try:
                # Parse message data
                data = json.loads(message.data.decode('utf-8'))
                
                # Transform column names to match inference component expectations
                transformed_data = {
                    'SepalLengthCm': data.get('sepal_length'),
                    'SepalWidthCm': data.get('sepal_width'), 
                    'PetalLengthCm': data.get('petal_length'),
                    'PetalWidthCm': data.get('petal_width'),
                    'timestamp': data.get('timestamp'),
                    'sample_id': data.get('sample_id'),
                    'ingestion_time': datetime.utcnow().isoformat(),
                    'message_id': message.message_id
                }
                
                consumed_data.append(transformed_data)
                
                logger.info(f"Consumed message: {data['sample_id']} (ID: {message.message_id})")
                
                # Acknowledge the message
                message.ack()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                message.nack()
        
        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(max_messages=batch_size * 2)
        
        logger.info(f"Starting Pub/Sub consumer for topic: {topic_name}")
        
        try:
            # Start pulling messages
            streaming_pull_future = subscriber.subscribe(
                subscription_path,
                callback=callback,
                flow_control=flow_control
            )
            
            logger.info(f"Listening for messages on {subscription_path}...")
            
            # Process messages until batch size or timeout
            while len(consumed_data) < batch_size and (time.time() - start_time) < timeout_seconds:
                time.sleep(1)  # Check every second
            
            # Cancel the subscriber
            streaming_pull_future.cancel()
            
            # Process collected data
            if consumed_data:
                df = pd.DataFrame(consumed_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['ingestion_time'] = pd.to_datetime(df['ingestion_time'])
                
                # Load to BigQuery
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND",
                    schema=schema
                )
                
                job = bq_client.load_table_from_dataframe(
                    df, table_id, job_config=job_config
                )
                job.result()
                
                logger.info(f"Loaded {len(consumed_data)} records to BigQuery")
            else:
                logger.warning("No messages received within timeout period")
        
        except Exception as e:
            logger.error(f"Error in Pub/Sub consumer: {e}")
            raise
        
        finally:
            logger.info("Pub/Sub consumer finished")
        
        # Set output dataset metadata
        dataset.uri = f"bq://{table_id}"
        dataset.metadata = {
            "total_records": len(consumed_data),
            "table_id": table_id,
            "topic": topic_name,
            "subscription": subscription_name
        }
    
    return pubsub_consumer_op(
        project_id=project_id,
        topic_name=topic_name,
        subscription_name=subscription_name,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds
    )
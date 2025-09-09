import kfp
from kfp import dsl
from typing import NamedTuple


def kafka_data_source(
    kafka_servers: str,
    topic: str,
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    batch_size: int = 100,
    timeout_seconds: int = 300
) -> NamedTuple("KafkaOutputs", [("dataset", kfp.dsl.Dataset)]):
    """
    Kubeflow component that consumes data from Kafka topic and stores in BigQuery.
    """
    
    @dsl.component(
        base_image="python:3.9-slim",
        packages_to_install=[
            "kafka-python==2.0.2",
            "google-cloud-bigquery==3.11.4",
            "pandas==2.0.3",
            "pyarrow==12.0.1",
            "google-auth==2.23.3",
            "google-cloud-secret-manager==2.16.4"
        ]
    )
    def kafka_consumer_op(
        kafka_servers: str,
        topic: str,
        project_id: str,
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
        from kafka import KafkaConsumer, TopicPartition
        from kafka.errors import KafkaError
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Initialize BigQuery client
        bq_client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.{bq_dataset}.{bq_table}"
        
        # Create BigQuery table if it doesn't exist
        schema = [
            bigquery.SchemaField("sepal_length", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("sepal_width", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("petal_length", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("petal_width", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("sample_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP", mode="REQUIRED")
        ]
        
        try:
            table = bq_client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            table = bigquery.Table(table_id, schema=schema)
            table = bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
        
        # Load GCP service account for SASL authentication
        import base64
        import os
        from pathlib import Path
        
        # Path to service account key (adjust path for container environment)
        key_path = "/gcp-sa-key/deeplearning-sahil-e50332de6687.json"  # Mounted in container
        
        if os.path.exists(key_path):
            with open(key_path, 'r') as f:
                key_data = f.read()
            sasl_password = base64.b64encode(key_data.encode()).decode()
            sasl_username = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
        else:
            raise FileNotFoundError(f"Service account key not found at {key_path}")
        
        # Initialize Kafka consumer with SASL authentication for GCP Managed Kafka
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            consumer_timeout_ms=timeout_seconds * 1000,
            group_id=f"kfp-consumer-{int(time.time())}",
            # SASL authentication for GCP Managed Kafka
            security_protocol='SASL_SSL',
            sasl_mechanism='PLAIN',
            sasl_plain_username=sasl_username,
            sasl_plain_password=sasl_password,
            ssl_check_hostname=False,  # Disable for tunnel connections
            ssl_cafile=None  # Uses system CA certificates
        )
        
        logger.info(f"Starting Kafka consumer for topic: {topic}")
        
        consumed_data = []
        start_time = time.time()
        
        try:
            for message in consumer:
                if message.value:
                    # Add ingestion timestamp
                    record = message.value
                    record['ingestion_time'] = datetime.utcnow().isoformat()
                    consumed_data.append(record)
                    
                    logger.info(f"Consumed message: {record['sample_id']}")
                    
                    # Process batch when reached batch_size
                    if len(consumed_data) >= batch_size:
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
                        consumed_data = []
                    
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        logger.info(f"Timeout reached after {timeout_seconds} seconds")
                        break
        
        except Exception as e:
            logger.error(f"Error in Kafka consumer: {e}")
            raise
        
        finally:
            # Process remaining data
            if consumed_data:
                df = pd.DataFrame(consumed_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['ingestion_time'] = pd.to_datetime(df['ingestion_time'])
                
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND",
                    schema=schema
                )
                
                job = bq_client.load_table_from_dataframe(
                    df, table_id, job_config=job_config
                )
                job.result()
                
                logger.info(f"Loaded final {len(consumed_data)} records to BigQuery")
            
            consumer.close()
            logger.info("Kafka consumer closed")
        
        # Set output dataset metadata
        dataset.uri = f"bq://{table_id}"
        dataset.metadata = {
            "total_records": len(consumed_data),
            "table_id": table_id,
            "topic": topic
        }
    
    return kafka_consumer_op(
        kafka_servers=kafka_servers,
        topic=topic,
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds
    )
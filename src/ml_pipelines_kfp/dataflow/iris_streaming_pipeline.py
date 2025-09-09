"""
Dataflow streaming pipeline for real-time Iris inference.
Reads from Pub/Sub, calls Vertex AI endpoint, writes predictions to BigQuery.
"""
import json
import logging
import argparse
from typing import Any, Dict, List

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
from google.cloud import aiplatform
from google.oauth2 import service_account

# Constants
PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
MODEL_NAME = "Iris-Classifier-XGBoost"
ENDPOINT_NAME = "Iris-Classifier-XGBoost"

# BigQuery schema for predictions
PREDICTION_SCHEMA = {
    'fields': [
        {'name': 'sepal_length', 'type': 'FLOAT', 'mode': 'REQUIRED'},
        {'name': 'sepal_width', 'type': 'FLOAT', 'mode': 'REQUIRED'},
        {'name': 'petal_length', 'type': 'FLOAT', 'mode': 'REQUIRED'},
        {'name': 'petal_width', 'type': 'FLOAT', 'mode': 'REQUIRED'},
        {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
        {'name': 'sample_id', 'type': 'INTEGER', 'mode': 'REQUIRED'},
        {'name': 'prediction', 'type': 'STRING', 'mode': 'REQUIRED'},
        {'name': 'prediction_confidence', 'type': 'FLOAT', 'mode': 'NULLABLE'},
        {'name': 'prediction_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
        {'name': 'model_endpoint', 'type': 'STRING', 'mode': 'REQUIRED'},
        {'name': 'processing_time', 'type': 'FLOAT', 'mode': 'NULLABLE'}
    ]
}


class ParsePubSubMessage(beam.DoFn):
    """Parse JSON message from Pub/Sub."""
    
    def process(self, element):
        try:
            # Parse the Pub/Sub message
            message_data = json.loads(element.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            if all(field in message_data for field in required_fields):
                yield message_data
            else:
                logging.warning(f"Missing required fields in message: {message_data}")
                
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Error parsing message: {e}, message: {element}")


class CallVertexAIEndpoint(beam.DoFn):
    """Call Vertex AI model endpoint for inference."""
    
    def __init__(self, project_id: str, region: str, endpoint_name: str):
        self.project_id = project_id
        self.region = region
        self.endpoint_name = endpoint_name
        self.client = None
        self.endpoint = None
        
    def setup(self):
        """Initialize Vertex AI client."""
        aiplatform.init(project=self.project_id, location=self.region)
        
        # Get the endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{self.endpoint_name}"'
        )
        
        if endpoints:
            self.endpoint = endpoints[0]
            logging.info(f"Found endpoint: {self.endpoint.display_name}")
        else:
            raise RuntimeError(f"Endpoint '{self.endpoint_name}' not found")
    
    def process(self, element):
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        try:
            # Prepare features for prediction
            features = [
                element['sepal_length'],
                element['sepal_width'], 
                element['petal_length'],
                element['petal_width']
            ]
            
            # Call the endpoint
            predictions = self.endpoint.predict(instances=[features])
            
            # Extract prediction result
            prediction_result = predictions.predictions[0]
            
            # Handle different prediction formats
            if isinstance(prediction_result, list):
                predicted_class = prediction_result[0]
                confidence = max(prediction_result) if len(prediction_result) > 1 else None
            else:
                predicted_class = str(prediction_result)
                confidence = None
            
            # Map numeric prediction to class name if needed
            class_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            if str(predicted_class).isdigit():
                predicted_class = class_mapping.get(int(predicted_class), str(predicted_class))
            
            processing_time = time.time() - start_time
            
            # Create result record
            result = {
                'sepal_length': element['sepal_length'],
                'sepal_width': element['sepal_width'],
                'petal_length': element['petal_length'],
                'petal_width': element['petal_width'],
                'timestamp': element.get('timestamp', datetime.utcnow().isoformat()),
                'sample_id': element.get('sample_id', 0),
                'prediction': str(predicted_class),
                'prediction_confidence': confidence,
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'model_endpoint': f"{self.project_id}/{self.region}/{self.endpoint_name}",
                'processing_time': processing_time
            }
            
            logging.info(f"Prediction for sample {element.get('sample_id')}: {predicted_class}")
            yield result
            
        except Exception as e:
            logging.error(f"Error calling endpoint: {e}, element: {element}")
            # Yield error record for monitoring
            yield {
                'sepal_length': element.get('sepal_length', 0.0),
                'sepal_width': element.get('sepal_width', 0.0),
                'petal_length': element.get('petal_length', 0.0),
                'petal_width': element.get('petal_width', 0.0),
                'timestamp': element.get('timestamp', datetime.utcnow().isoformat()),
                'sample_id': element.get('sample_id', 0),
                'prediction': 'ERROR',
                'prediction_confidence': None,
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'model_endpoint': f"ERROR: {str(e)}",
                'processing_time': time.time() - start_time
            }


class AddProcessingMetadata(beam.DoFn):
    """Add processing metadata to records."""
    
    def process(self, element):
        from datetime import datetime
        
        # Add additional metadata
        element['dataflow_processing_time'] = datetime.utcnow().isoformat()
        element['pipeline_version'] = '1.0.0'
        
        yield element


def run_pipeline(argv=None):
    """Run the Dataflow streaming pipeline."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_topic',
        required=True,
        help='Pub/Sub topic to read from (projects/PROJECT/topics/TOPIC)'
    )
    parser.add_argument(
        '--output_table',
        required=True,
        help='BigQuery table to write to (PROJECT:DATASET.TABLE)'
    )
    parser.add_argument(
        '--project_id',
        default=PROJECT_ID,
        help='GCP Project ID'
    )
    parser.add_argument(
        '--region',
        default=REGION,
        help='GCP Region'
    )
    parser.add_argument(
        '--endpoint_name',
        default=ENDPOINT_NAME,
        help='Vertex AI endpoint name'
    )
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    # Pipeline options
    pipeline_options = PipelineOptions(pipeline_args)
    
    # Create pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        predictions = (
            pipeline
            | 'Read from Pub/Sub' >> ReadFromPubSub(topic=known_args.input_topic)
            | 'Parse JSON' >> beam.ParDo(ParsePubSubMessage())
            | 'Add Window' >> beam.WindowInto(window.FixedWindows(60))  # 1-minute windows
            | 'Call Vertex AI' >> beam.ParDo(CallVertexAIEndpoint(
                known_args.project_id, 
                known_args.region, 
                known_args.endpoint_name))
            | 'Add Metadata' >> beam.ParDo(AddProcessingMetadata())
            | 'Write to BigQuery' >> WriteToBigQuery(
                table=known_args.output_table,
                schema=PREDICTION_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                additional_bq_parameters={
                    'timePartitioning': {
                        'type': 'DAY',
                        'field': 'prediction_timestamp'
                    }
                }
            )
        )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run_pipeline()
import json
import logging
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.io import WriteToBigQuery

logger = logging.getLogger(__name__)

DEAD_LETTER_TAG = "dead_letters"

DEAD_LETTER_SCHEMA = {
    "fields": [
        {"name": "entity_id", "type": "STRING", "mode": "NULLABLE"},
        {"name": "pipeline", "type": "STRING", "mode": "REQUIRED"},
        {"name": "stage", "type": "STRING", "mode": "REQUIRED"},
        {"name": "error_type", "type": "STRING", "mode": "REQUIRED"},
        {"name": "error_message", "type": "STRING", "mode": "REQUIRED"},
        {"name": "original_message", "type": "STRING", "mode": "NULLABLE"},
        {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "retry_count", "type": "INTEGER", "mode": "NULLABLE"},
    ]
}


def build_dead_letter(
    pipeline, stage, error_type, error_message,
    entity_id=None, original_message=None, retry_count=None,
):
    row = {
        "entity_id": entity_id,
        "pipeline": pipeline,
        "stage": stage,
        "error_type": error_type,
        "error_message": str(error_message)[:2048],
        "original_message": _safe_serialize(original_message),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retry_count": retry_count,
    }
    return row


def _safe_serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return repr(obj)
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return repr(obj)


def write_dead_letters(pcollection, table, label_prefix=""):
    prefix = f"{label_prefix} " if label_prefix else ""
    return pcollection | f"{prefix}Write Dead Letters" >> WriteToBigQuery(
        table=table,
        schema=DEAD_LETTER_SCHEMA,
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
    )

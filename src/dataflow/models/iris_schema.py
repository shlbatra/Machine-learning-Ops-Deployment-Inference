from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PubSubIrisMessage(BaseModel):
    """Validates incoming Pub/Sub messages for the Iris feature pipeline."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    timestamp: Optional[str] = None
    sample_id: Optional[int] = None

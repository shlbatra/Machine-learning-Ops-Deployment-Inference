from __future__ import annotations

from typing import List

from pydantic import BaseModel
from pydantic import Field


class Prediction(BaseModel):
    class Config:
        allow_population_by_field_name = True

    class_: int = Field(..., alias='class')
    class_probabilities: List[float]
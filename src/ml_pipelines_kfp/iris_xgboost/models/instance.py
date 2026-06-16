from __future__ import annotations

from pydantic import BaseModel


class Instance(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

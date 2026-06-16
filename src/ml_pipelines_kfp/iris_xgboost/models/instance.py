from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Instance(BaseModel):
    # Accept both canonical names (sepal_length_cm) and aliases (SepalLengthCm) in input
    model_config = ConfigDict(populate_by_name=True)

    sepal_length_cm: float = Field(alias="SepalLengthCm")
    sepal_width_cm: float = Field(alias="SepalWidthCm")
    petal_length_cm: float = Field(alias="PetalLengthCm")
    petal_width_cm: float = Field(alias="PetalWidthCm")

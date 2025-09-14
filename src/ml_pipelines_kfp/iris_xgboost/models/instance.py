from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Instance(BaseModel):
    SepalLengthCm: Optional[float] = None
    SepalWidthCm: Optional[float] = None
    PetalLengthCm: Optional[float] = None
    PetalWidthCm: Optional[float] = None

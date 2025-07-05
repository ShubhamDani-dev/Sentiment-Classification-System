from pydantic import BaseModel
from typing import Literal

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: Literal["positive", "negative"]
    score: float

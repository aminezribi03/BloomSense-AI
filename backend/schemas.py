"""
Pydantic schemas used for request validation and response formatting.

These classes define the shape of incoming and outgoing JSON payloads.
Using Pydantic ensures that the API receives correctly typed data and
produces predictable responses.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    

    sepal_length: float = Field(..., description="Length of the sepal in centimetres")
    sepal_width: float = Field(..., description="Width of the sepal in centimetres")
    petal_length: float = Field(..., description="Length of the petal in centimetres")
    petal_width: float = Field(..., description="Width of the petal in centimetres")


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""

    predicted_class: int = Field(..., description="Index of the predicted iris species")
    probability: float = Field(
        ..., description="Probability associated with the predicted class"
    )


class MetricsResponse(BaseModel):
    """Schema for metrics returned by the `/metrics` endpoint."""

    cv_results: Dict[str, float]
    test_accuracy: float
    classification_report: Dict[str, Any]


class HistoryItem(BaseModel):
    """Represents an aggregated prediction count for a single class."""

    class_label: str
    count: int


class HistoryResponse(BaseModel):
    """Schema for the history endpoint returning aggregated prediction counts."""

    history: List[HistoryItem]

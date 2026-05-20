"""Schemas for classification prediction responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionInputMetadata(BaseModel):
    """Metadata about the uploaded prediction input."""

    filename: str
    content_type: str
    file_size_bytes: int = Field(ge=0)


class ClassificationPredictionResponse(BaseModel):
    """Response schema for Track A classification prediction."""

    request_id: str
    model_name: str
    model_version: str
    run_id: str
    threshold: float
    predicted_label: str
    predicted_label_id: int
    probability_good: float
    probability_defect: float
    decision: str
    production_ready: bool
    deployment_safe: bool
    input: PredictionInputMetadata
    live_prediction_enabled: bool
    upload_predict_enabled: bool
    limitations: list[str]

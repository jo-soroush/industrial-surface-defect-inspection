"""Pydantic schemas for API health and metadata responses."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Schema for the API health endpoint."""

    status: str
    service: str
    api_ready: bool
    prediction_ready: bool


class MetadataResponse(BaseModel):
    """Schema for the API metadata endpoint."""

    project_name: str
    api_stage: str
    supported_tracks: list[str]
    implemented_endpoints: list[str]
    planned_endpoints: list[str]
    production_ready: bool
    deployment_safe: bool
    live_prediction_enabled: bool
    upload_predict_enabled: bool

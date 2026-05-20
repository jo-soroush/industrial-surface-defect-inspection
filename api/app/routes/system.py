"""System-level API routes for health and metadata."""

from __future__ import annotations

from fastapi import APIRouter

from api.app.schemas.system import HealthResponse, MetadataResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a minimal readiness signal for the API service."""
    return HealthResponse(
        status="ok",
        service="industrial-surface-defect-api",
        api_ready=True,
        prediction_ready=False,
    )


@router.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return the current API stage and supported tracks."""
    return MetadataResponse(
        project_name="Industrial Surface Defect Inspection Platform",
        api_stage="health_metadata_only",
        supported_tracks=[
            "track_a_classification",
            "track_b_anomaly",
            "yolo_detection",
        ],
        implemented_endpoints=["/health", "/metadata"],
        planned_endpoints=[
            "/predict/classification",
            "/predict/detection",
            "/predict/anomaly",
        ],
        production_ready=False,
        deployment_safe=False,
        live_prediction_enabled=False,
        upload_predict_enabled=False,
    )

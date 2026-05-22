"""API tests for system metadata."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app.main import app


client = TestClient(app)


def test_metadata_includes_unified_inspection_endpoint() -> None:
    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert "/inspect/image" in payload["implemented_endpoints"]
    assert "/inspect/image" not in payload["planned_endpoints"]
    assert "/predict/classification" in payload["implemented_endpoints"]
    assert "/predict/detection" in payload["implemented_endpoints"]
    assert "/predict/anomaly" in payload["implemented_endpoints"]
    assert payload["production_ready"] is False
    assert payload["deployment_safe"] is False
    assert payload["api_stage"] == "health_metadata_plus_unified_inspection"
    assert "unified_image_inspection" in payload["supported_tracks"]

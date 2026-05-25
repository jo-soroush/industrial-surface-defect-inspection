from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"


def test_dockerfile_contains_api_source_and_staged_assets() -> None:
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "COPY api/ ./api/" in text
    assert "COPY src/ ./src/" in text
    assert "COPY frontend/ ./frontend/" in text
    assert "COPY configs/ ./configs/" in text
    assert "COPY runtime_assets/artifacts/ ./artifacts/" in text
    assert "COPY runtime_assets/configs/ ./configs/" in text
    assert "COPY runtime_assets/data/ ./data/" in text
    assert "PYTHONPATH=/app" in text
    assert 'CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in text


def test_dockerfile_does_not_copy_raw_artifacts_tree() -> None:
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "COPY artifacts/" not in text
    assert "COPY data/" not in text
    assert "COPY notebooks/" not in text

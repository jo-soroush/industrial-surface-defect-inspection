from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
API_DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.api"
FRONTEND_DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.frontend"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_api_dockerfile_uses_api_requirements_and_runtime_assets() -> None:
    text = _read(API_DOCKERFILE_PATH)

    assert "COPY requirements-api.txt" in text
    assert "pip install --no-cache-dir -r requirements-api.txt" in text
    assert "COPY api/ ./api/" in text
    assert "COPY src/ ./src/" in text
    assert "COPY configs/ ./configs/" in text
    assert "COPY runtime_assets/artifacts/ ./artifacts/" in text
    assert "COPY runtime_assets/configs/ ./configs/" in text
    assert "COPY runtime_assets/data/ ./data/" in text
    assert "COPY frontend/" not in text
    assert "COPY artifacts/" not in text
    assert "COPY data/" not in text
    assert 'CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in text


def test_frontend_dockerfile_uses_frontend_requirements_and_bundle_assets() -> None:
    text = _read(FRONTEND_DOCKERFILE_PATH)

    assert "COPY requirements-frontend.txt" in text
    assert "pip install --no-cache-dir -r requirements-frontend.txt" in text
    assert "COPY frontend/ ./frontend/" in text
    assert "COPY runtime_assets/artifacts/frontend/ ./artifacts/frontend/" in text
    assert "COPY api/" not in text
    assert "COPY src/" not in text
    assert "COPY runtime_assets/artifacts/ ./artifacts/" not in text
    assert "COPY runtime_assets/data/" not in text
    assert 'CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]' in text


def test_frontend_dockerfile_sets_streamlit_api_base_url() -> None:
    text = _read(FRONTEND_DOCKERFILE_PATH)

    assert "STREAMLIT_API_BASE_URL=http://api:8000" in text

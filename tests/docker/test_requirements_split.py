from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_API_PATH = REPO_ROOT / "requirements-api.txt"
REQUIREMENTS_FRONTEND_PATH = REPO_ROOT / "requirements-frontend.txt"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
REQUIREMENTS_DEV_PATH = REPO_ROOT / "requirements-dev.txt"


def _read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_requirements_txt_is_compatibility_aggregate() -> None:
    lines = _read_lines(REQUIREMENTS_PATH)

    assert lines == [
        "-r requirements-api.txt",
        "-r requirements-frontend.txt",
    ]


def test_api_requirements_contain_backend_runtime_only() -> None:
    lines = _read_lines(REQUIREMENTS_API_PATH)

    assert "fastapi>=0.115,<1.0" in lines
    assert "uvicorn[standard]>=0.30,<1.0" in lines
    assert "pydantic>=2.7,<3.0" in lines
    assert "PyYAML>=6.0,<7.0" in lines
    assert "torch==2.11.0" in lines
    assert "torchvision==0.26.0" in lines
    assert "Pillow==12.2.0" in lines
    assert "numpy==2.4.4" in lines
    assert "ultralytics>=8.0,<9.0" in lines
    assert "python-multipart>=0.0.9" in lines
    assert not any(line.startswith("streamlit") for line in lines)
    assert not any(line.startswith("plotly") for line in lines)
    assert not any(line.startswith("requests") for line in lines)


def test_frontend_requirements_contain_dashboard_runtime_only() -> None:
    lines = _read_lines(REQUIREMENTS_FRONTEND_PATH)

    assert "streamlit>=1.36,<2.0" in lines
    assert "plotly>=5.24,<6.0" in lines
    assert "requests>=2.32,<3.0" in lines
    assert "Pillow==12.2.0" in lines
    assert not any(line.startswith("torch==") for line in lines)
    assert not any(line.startswith("torchvision==") for line in lines)
    assert not any(line.startswith("ultralytics") for line in lines)


def test_dev_requirements_include_both_runtime_sets_and_test_tools() -> None:
    lines = _read_lines(REQUIREMENTS_DEV_PATH)

    assert lines[:2] == [
        "-r requirements-api.txt",
        "-r requirements-frontend.txt",
    ]
    assert "pytest>=8.2,<9.0" in lines
    assert "pytest-cov>=5.0,<6.0" in lines
    assert "ruff>=0.5,<1.0" in lines
    assert "black>=24.4,<25.0" in lines
    assert "mypy>=1.10,<2.0" in lines
    assert "httpx>=0.28,<1.0" in lines

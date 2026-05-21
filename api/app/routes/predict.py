"""Prediction routes for the first governed upload endpoint."""

from __future__ import annotations

from io import BytesIO
from functools import lru_cache
from tempfile import NamedTemporaryFile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from api.app.schemas.inspection import AnomalyResult, DetectionResult
from api.app.schemas.prediction import ClassificationPredictionResponse, PredictionInputMetadata
from src.inspection_ai.inference.anomaly_detector import AnomalyDetector
from src.inspection_ai.inference.track_a_classifier import TrackAClassifier
from src.inspection_ai.inference.yolo_detector import YOLODetector


router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
LIMITATIONS = [
    "not production-ready",
    "not deployment-safe",
    "local prototype endpoint",
]


@lru_cache(maxsize=1)
def get_track_a_classifier() -> TrackAClassifier:
    """Return a cached Track A classifier instance."""
    return TrackAClassifier(device="cpu")


@lru_cache(maxsize=1)
def get_yolo_detector() -> YOLODetector:
    """Return a cached YOLO detector wrapper without loading the model at import time."""
    return YOLODetector(device="cpu")


@lru_cache(maxsize=1)
def get_anomaly_detector() -> AnomalyDetector:
    """Return a cached anomaly detector wrapper without loading the checkpoint at import time."""
    return AnomalyDetector(device="cpu")


@router.post("/predict/classification", response_model=ClassificationPredictionResponse)
async def predict_classification(
    file: UploadFile | None = File(default=None),
) -> ClassificationPredictionResponse:
    """Predict Track A classification from a single uploaded image."""
    request_id = str(uuid4())

    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing image upload. Use multipart field 'file'.",
        )

    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {content_type!r}.",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty or missing.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte limit.",
        )

    suffix = _resolve_suffix(file.filename, content_type)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            temp_path = Path(tmp.name)

        try:
            classifier = get_track_a_classifier()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Track A classifier checkpoint is unavailable: {exc}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Track A classifier could not be initialized: {exc}",
            ) from exc

        try:
            result = classifier.predict(temp_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Track A inference path is unavailable: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Track A inference failed: {exc}",
            ) from exc

        prediction = result.as_dict() if hasattr(result, "as_dict") else result.__dict__
        return ClassificationPredictionResponse(
            request_id=request_id,
            **prediction,
            input=PredictionInputMetadata(
                filename=file.filename or "",
                content_type=content_type,
                file_size_bytes=len(payload),
            ),
            live_prediction_enabled=True,
            upload_predict_enabled=True,
            limitations=LIMITATIONS,
        )
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/predict/detection", response_model=DetectionResult)
async def predict_detection(
    file: UploadFile | None = File(default=None),
) -> DetectionResult:
    """Run live defect detection/localization on a single uploaded image."""
    image = await _read_uploaded_image(file)

    try:
        detector = get_yolo_detector()
        return detector.predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"YOLO detection dependency is unavailable: {exc}",
        ) from exc
    except RuntimeError as exc:
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "ultralytics" in str(exc).lower()
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(
            status_code=status_code,
            detail=f"YOLO detection failed: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"YOLO detection returned invalid output: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="YOLO detection failed unexpectedly.",
        ) from exc


@router.post("/predict/anomaly", response_model=AnomalyResult)
async def predict_anomaly(
    file: UploadFile | None = File(default=None),
) -> AnomalyResult:
    """Run live surface anomaly detection on a single uploaded image."""
    image = await _read_uploaded_image(file)

    try:
        detector = get_anomaly_detector()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Anomaly detection dependency is unavailable: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Anomaly detection dependency is unavailable: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Anomaly detection dependency is unavailable: {exc}",
        ) from exc

    try:
        return detector.predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Anomaly detection dependency is unavailable: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection returned invalid output: {exc}",
        ) from exc
    except RuntimeError as exc:
        detail = str(exc)
        if "checkpoint loading" in detail or "model construction" in detail:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Anomaly detection dependency is unavailable: {exc}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Anomaly detection failed unexpectedly.",
        ) from exc


async def _read_uploaded_image(file: UploadFile | None) -> Image.Image:
    """Validate and decode an uploaded image for live prediction endpoints."""
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing image upload. Use multipart field 'file'.",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image. Received content type: {content_type!r}.",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty or missing.",
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds the {MAX_UPLOAD_BYTES} byte limit.",
        )

    try:
        with Image.open(BytesIO(payload)) as opened:
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be decoded as an image.",
        ) from exc


def _resolve_suffix(filename: str | None, content_type: str) -> str:
    """Derive a safe temporary filename suffix from the upload metadata."""
    suffix_from_name = Path(filename).suffix.lower() if filename else ""
    if suffix_from_name in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix_from_name

    suffix_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }
    return suffix_map.get(content_type, ".img")

"""Orchestration service for the unified image inspection endpoint."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from .contracts.inspection import (
    AnomalyResult,
    AnomalyTraceability,
    ClassificationResult,
    ClassificationTraceability,
    DetectionResult,
    DetectionTraceability,
    ExplanationContext,
    ImageInspectionResponse,
    InspectionError,
    InputMetadata,
    TopLevelTraceability,
)
from .decision import build_inspection_decision
from .inference import anomaly_detector as anomaly_module
from .inference import track_a_classifier as track_a_module
from .inference import yolo_detector as yolo_module
from .training.checkpointing import resolve_model_checkpoint_path


CONTRACT_VERSION = "image_inspection_response_v0_1"
API_VERSION = "inspect_image_api_v0_1"
DEFAULT_SOURCE_ENDPOINT = "/inspect/image"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

LIMITATIONS = [
    "This response is a governed inspection aggregation of model outputs.",
    "This response does not claim production readiness.",
    "This response does not claim deployment safety.",
]


@lru_cache(maxsize=1)
def get_track_a_classifier() -> track_a_module.TrackAClassifier:
    """Return a cached Track A classifier instance."""
    return track_a_module.TrackAClassifier(device="cpu")


@lru_cache(maxsize=1)
def get_yolo_detector() -> yolo_module.YOLODetector:
    """Return a cached YOLO detector instance."""
    return yolo_module.YOLODetector(device="cpu")


@lru_cache(maxsize=1)
def get_anomaly_detector() -> anomaly_module.AnomalyDetector:
    """Return a cached anomaly detector instance."""
    return anomaly_module.AnomalyDetector(device="cpu")


def inspect_image(
    *,
    image_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    request_id: str | None = None,
    source_endpoint: str = DEFAULT_SOURCE_ENDPOINT,
    classifier: Any | None = None,
    detector: Any | None = None,
    anomaly_detector: Any | None = None,
) -> ImageInspectionResponse:
    """Run the full governed inspection pipeline on one uploaded image."""
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError("image_bytes must be bytes.")
    if not image_bytes:
        raise ValueError("Uploaded image is empty or missing.")

    raw_bytes = bytes(image_bytes)
    image = _decode_image(raw_bytes)
    request_id = request_id or str(uuid4())
    content_type_value = content_type or ""
    filename_value = filename or ""
    preprocessing_notes = [
        "decoded_to_rgb_with_pillow",
        "classification_uses_temporary_file_path",
    ]

    input_metadata = InputMetadata(
        filename=filename_value,
        content_type=content_type_value,
        file_size_bytes=len(raw_bytes),
        image_width=image.width,
        image_height=image.height,
        image_mode=image.mode,
        preprocessing_notes=preprocessing_notes,
    )

    classification_result, classification_error = _run_classification(
        image_bytes=raw_bytes,
        filename=filename_value,
        content_type=content_type_value,
        source_endpoint=source_endpoint,
        classifier=classifier,
    )
    detection_result, detection_error = _run_detection(
        image=image,
        source_endpoint=source_endpoint,
        detector=detector,
    )
    anomaly_result, anomaly_error = _run_anomaly(
        image=image,
        source_endpoint=source_endpoint,
        anomaly_detector=anomaly_detector,
    )

    errors = [error for error in (classification_error, detection_error, anomaly_error) if error is not None]
    warnings = _build_warnings(errors=errors, anomaly_result=anomaly_result)

    try:
        decision = build_inspection_decision(
            classification=classification_result,
            detection=detection_result,
            anomaly=anomaly_result,
            source_endpoint=source_endpoint,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Inspection decision aggregation failed: {exc}") from exc

    if decision.rule_id is None:
        raise RuntimeError("Inspection decision did not produce a rule identifier.")

    if errors:
        warnings.append("One or more model subsystems failed; the inspection response is partial.")
    if anomaly_result.quality_status == "review_required_weak_evidence":
        warnings.append("Anomaly evidence is weak and review-only.")
    warnings = _deduplicate_messages(warnings)

    limitations = _build_limitations(
        classification_result=classification_result,
        detection_result=detection_result,
        anomaly_result=anomaly_result,
        errors=errors,
    )

    traceability = TopLevelTraceability(
        contract_version=CONTRACT_VERSION,
        api_version=API_VERSION,
        source_endpoint=source_endpoint,
        classification=classification_result.model_dump(),
        detection=detection_result.model_dump(),
        anomaly=anomaly_result.model_dump(),
        decision=decision.model_dump(),
        frontend_evidence_sources=_frontend_evidence_sources(),
    )

    explanation_context = ExplanationContext(
        status="available",
        context_version="image_inspection_explanation_context_v0_1",
        allowed_sources=[
            "classification",
            "detection",
            "anomaly",
            "decision",
            "traceability",
            "limitations",
        ],
        summary_inputs={
            "classification_label": classification_result.predicted_label,
            "classification_status": classification_result.status,
            "detection_box_count": detection_result.predicted_box_count,
            "detection_status": detection_result.status,
            "anomaly_label": anomaly_result.predicted_label,
            "anomaly_status": anomaly_result.status,
            "anomaly_quality_status": anomaly_result.quality_status,
            "final_decision": decision.final_decision,
            "decision_level": decision.decision_level,
            "model_agreement_status": decision.model_agreement_status,
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        safety_boundaries=[
            "No production-ready claim.",
            "No deployment-safe claim.",
            "No fake model outputs.",
        ],
        forbidden_claims=[
            "production-ready",
            "deployment-safe",
            "autonomous AI agent decision",
            "fake prediction",
        ],
    )

    return ImageInspectionResponse(
        request_id=request_id,
        timestamp_utc=_utc_timestamp(),
        input=input_metadata,
        classification=classification_result,
        detection=detection_result,
        anomaly=anomaly_result,
        decision=decision,
        traceability=traceability,
        limitations=limitations,
        errors=errors,
        warnings=warnings,
        explanation_context=explanation_context,
    )


def _run_classification(
    *,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    source_endpoint: str,
    classifier: Any | None,
) -> tuple[ClassificationResult, InspectionError | None]:
    suffix = _resolve_suffix(filename, content_type)
    temp_path: Path | None = None
    try:
        helper = classifier or get_track_a_classifier()
        with NamedTemporaryFile(mode="wb", delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            temp_path = Path(tmp.name)

        try:
            prediction = helper.predict(temp_path)
            return _classification_result_from_prediction(
                prediction=prediction,
                source_endpoint=source_endpoint,
            ), None
        except FileNotFoundError:
            return (
                _failed_classification_result(
                    source_endpoint=source_endpoint,
                    error_message="Track A classifier dependency is unavailable.",
                    helper=helper,
                ),
                _inspection_error(
                    component="classification",
                    code="dependency_unavailable",
                    message="Track A classifier dependency is unavailable.",
                    recoverable=False,
                ),
            )
        except ValueError:
            return (
                _failed_classification_result(
                    source_endpoint=source_endpoint,
                    error_message="Track A classification returned invalid output.",
                    helper=helper,
                ),
                _inspection_error(
                    component="classification",
                    code="invalid_output",
                    message="Track A classification returned invalid output.",
                    recoverable=False,
                ),
            )
        except RuntimeError:
            return (
                _failed_classification_result(
                    source_endpoint=source_endpoint,
                    error_message="Track A classification failed.",
                    helper=helper,
                ),
                _inspection_error(
                    component="classification",
                    code="prediction_failed",
                    message="Track A classification failed.",
                    recoverable=False,
                ),
            )
        except Exception:
            return (
                _failed_classification_result(
                    source_endpoint=source_endpoint,
                    error_message="Track A classification failed.",
                    helper=helper,
                ),
                _inspection_error(
                    component="classification",
                    code="prediction_failed",
                    message="Track A classification failed.",
                    recoverable=False,
                ),
            )
    except (FileNotFoundError, ValueError, RuntimeError):
        return (
            _failed_classification_result(
                source_endpoint=source_endpoint,
                error_message="Track A classifier dependency is unavailable.",
                helper=classifier,
            ),
            _inspection_error(
                component="classification",
                code="dependency_unavailable",
                message="Track A classifier dependency is unavailable.",
                recoverable=False,
            ),
        )
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _run_detection(
    *,
    image: Image.Image,
    source_endpoint: str,
    detector: Any | None,
) -> tuple[DetectionResult, InspectionError | None]:
    try:
        helper = detector or get_yolo_detector()
    except (FileNotFoundError, ValueError, RuntimeError):
        return (
            _failed_detection_result(
                image=image,
                source_endpoint=source_endpoint,
                error_message="YOLO detection dependency is unavailable.",
                helper=detector,
            ),
            _inspection_error(
                component="detection",
                code="dependency_unavailable",
                message="YOLO detection dependency is unavailable.",
                recoverable=False,
            ),
        )

    try:
        result = helper.predict(image)
        if isinstance(result, DetectionResult):
            result.traceability.source_endpoint = source_endpoint
            return result, None
        if hasattr(result, "model_dump"):
            coerced = DetectionResult.model_validate(result.model_dump())
            coerced.traceability.source_endpoint = source_endpoint
            return coerced, None
        if isinstance(result, dict):
            coerced = DetectionResult.model_validate(result)
            coerced.traceability.source_endpoint = source_endpoint
            return coerced, None
        raise ValueError("Detection helper returned an unsupported result type.")
    except FileNotFoundError:
        return (
            _failed_detection_result(
                image=image,
                source_endpoint=source_endpoint,
                error_message="YOLO detection dependency is unavailable.",
                helper=helper,
            ),
            _inspection_error(
                component="detection",
                code="dependency_unavailable",
                message="YOLO detection dependency is unavailable.",
                recoverable=False,
            ),
        )
    except ValueError:
        return (
            _failed_detection_result(
                image=image,
                source_endpoint=source_endpoint,
                error_message="YOLO detection returned invalid output.",
                helper=helper,
            ),
            _inspection_error(
                component="detection",
                code="invalid_output",
                message="YOLO detection returned invalid output.",
                recoverable=False,
            ),
        )
    except RuntimeError:
        return (
            _failed_detection_result(
                image=image,
                source_endpoint=source_endpoint,
                error_message="YOLO detection failed.",
                helper=helper,
            ),
            _inspection_error(
                component="detection",
                code="prediction_failed",
                message="YOLO detection failed.",
                recoverable=False,
            ),
        )
    except Exception:
        return (
            _failed_detection_result(
                image=image,
                source_endpoint=source_endpoint,
                error_message="YOLO detection failed unexpectedly.",
                helper=helper,
            ),
            _inspection_error(
                component="detection",
                code="unexpected_failure",
                message="YOLO detection failed unexpectedly.",
                recoverable=False,
            ),
        )


def _run_anomaly(
    *,
    image: Image.Image,
    source_endpoint: str,
    anomaly_detector: Any | None,
) -> tuple[AnomalyResult, InspectionError | None]:
    try:
        helper = anomaly_detector or get_anomaly_detector()
    except (FileNotFoundError, ValueError, RuntimeError):
        return (
            _failed_anomaly_result(
                source_endpoint=source_endpoint,
                error_message="Anomaly detection dependency is unavailable.",
                helper=anomaly_detector,
            ),
            _inspection_error(
                component="anomaly",
                code="dependency_unavailable",
                message="Anomaly detection dependency is unavailable.",
                recoverable=False,
            ),
        )

    try:
        result = helper.predict(image)
        if isinstance(result, AnomalyResult):
            result.traceability.source_endpoint = source_endpoint
            return result, None
        if hasattr(result, "model_dump"):
            coerced = AnomalyResult.model_validate(result.model_dump())
            coerced.traceability.source_endpoint = source_endpoint
            return coerced, None
        if isinstance(result, dict):
            coerced = AnomalyResult.model_validate(result)
            coerced.traceability.source_endpoint = source_endpoint
            return coerced, None
        raise ValueError("Anomaly helper returned an unsupported result type.")
    except FileNotFoundError:
        return (
            _failed_anomaly_result(
                source_endpoint=source_endpoint,
                error_message="Anomaly detection dependency is unavailable.",
                helper=helper,
            ),
            _inspection_error(
                component="anomaly",
                code="dependency_unavailable",
                message="Anomaly detection dependency is unavailable.",
                recoverable=False,
            ),
        )
    except ValueError:
        return (
            _failed_anomaly_result(
                source_endpoint=source_endpoint,
                error_message="Anomaly detection returned invalid output.",
                helper=helper,
            ),
            _inspection_error(
                component="anomaly",
                code="invalid_output",
                message="Anomaly detection returned invalid output.",
                recoverable=False,
            ),
        )
    except RuntimeError:
        return (
            _failed_anomaly_result(
                source_endpoint=source_endpoint,
                error_message="Anomaly detection failed.",
                helper=helper,
            ),
            _inspection_error(
                component="anomaly",
                code="prediction_failed",
                message="Anomaly detection failed.",
                recoverable=False,
            ),
        )
    except Exception:
        return (
            _failed_anomaly_result(
                source_endpoint=source_endpoint,
                error_message="Anomaly detection failed unexpectedly.",
                helper=helper,
            ),
            _inspection_error(
                component="anomaly",
                code="unexpected_failure",
                message="Anomaly detection failed unexpectedly.",
                recoverable=False,
            ),
        )


def _classification_result_from_prediction(*, prediction: Any, source_endpoint: str) -> ClassificationResult:
    model_name = _get_any(prediction, "model_name", track_a_module.EXPECTED_MODEL_NAME)
    model_version = _get_any(prediction, "model_version", track_a_module.EXPECTED_MODEL_VERSION)
    run_id = _get_any(prediction, "run_id", track_a_module.EXPECTED_RUN_ID)
    threshold = _get_any(prediction, "threshold", track_a_module.EXPECTED_THRESHOLD)
    predicted_label = _get_any(prediction, "predicted_label", None)
    predicted_label_id = _get_any(prediction, "predicted_label_id", None)
    probability_good = _get_any(prediction, "probability_good", None)
    probability_defect = _get_any(prediction, "probability_defect", None)
    decision = _get_any(prediction, "decision", predicted_label)

    checkpoint_path = _repo_relative(
        Path(track_a_module.REPO_ROOT) / resolve_model_checkpoint_path(track_a_module.EXPECTED_RUN_ID)
    )
    return ClassificationResult(
        status="success",
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        threshold=_float_or_none(threshold),
        predicted_label=_string_or_none(predicted_label),
        predicted_label_id=_int_or_none(predicted_label_id),
        probability_good=_float_or_none(probability_good),
        probability_defect=_float_or_none(probability_defect),
        decision=_string_or_none(decision),
        production_ready=False,
        deployment_safe=False,
        limitations=[
            "Classification output is local model output and not production-ready.",
            "Classification output is not deployment-safe.",
        ],
        traceability=ClassificationTraceability(
            checkpoint_path=checkpoint_path,
            run_config_path=_repo_relative(track_a_module.RUN_CONFIG_PATH),
            model_config_path=_repo_relative(track_a_module.MODEL_CONFIG_PATH),
            preprocessing_config_path=_repo_relative(track_a_module.PREPROCESSING_CONFIG_PATH),
            class_mapping_config_path=_repo_relative(track_a_module.CLASS_MAPPING_CONFIG_PATH),
            quality_decision_path=_repo_relative(track_a_module.QUALITY_DECISION_PATH),
            source_endpoint=source_endpoint,
        ),
    )


def _failed_classification_result(
    *,
    source_endpoint: str,
    error_message: str,
    helper: Any,
) -> ClassificationResult:
    checkpoint_path = _repo_relative(
        Path(track_a_module.REPO_ROOT) / resolve_model_checkpoint_path(track_a_module.EXPECTED_RUN_ID)
    )
    return ClassificationResult(
        status="failed",
        model_name=getattr(helper, "model_name", track_a_module.EXPECTED_MODEL_NAME),
        model_version=getattr(helper, "model_version", track_a_module.EXPECTED_MODEL_VERSION),
        run_id=getattr(helper, "run_id", track_a_module.EXPECTED_RUN_ID),
        threshold=_float_or_none(getattr(helper, "threshold", track_a_module.EXPECTED_THRESHOLD)),
        predicted_label=None,
        predicted_label_id=None,
        probability_good=None,
        probability_defect=None,
        decision=None,
        production_ready=False,
        deployment_safe=False,
        limitations=[
            error_message,
            "Classification output is not production-ready.",
            "Classification output is not deployment-safe.",
        ],
        traceability=ClassificationTraceability(
            checkpoint_path=checkpoint_path,
            run_config_path=_repo_relative(track_a_module.RUN_CONFIG_PATH),
            model_config_path=_repo_relative(track_a_module.MODEL_CONFIG_PATH),
            preprocessing_config_path=_repo_relative(track_a_module.PREPROCESSING_CONFIG_PATH),
            class_mapping_config_path=_repo_relative(track_a_module.CLASS_MAPPING_CONFIG_PATH),
            quality_decision_path=_repo_relative(track_a_module.QUALITY_DECISION_PATH),
            source_endpoint=source_endpoint,
        ),
    )


def _failed_detection_result(
    *,
    image: Image.Image,
    source_endpoint: str,
    error_message: str,
    helper: Any,
) -> DetectionResult:
    return DetectionResult(
        status="failed",
        model_name=getattr(helper, "model_name", yolo_module.DEFAULT_MODEL_NAME),
        model_version=getattr(helper, "model_version", yolo_module.DEFAULT_MODEL_VERSION),
        run_id=getattr(helper, "run_id", yolo_module.DEFAULT_RUN_ID),
        confidence_threshold=_float_or_none(getattr(helper, "confidence_threshold", yolo_module.DEFAULT_CONFIDENCE_THRESHOLD)),
        iou_threshold=_float_or_none(getattr(helper, "iou_threshold", yolo_module.DEFAULT_IOU_THRESHOLD)),
        image_width=image.width,
        image_height=image.height,
        predicted_box_count=0,
        defect_count=0,
        detections=[],
        best_detection=None,
        review_status="failed",
        production_ready=False,
        deployment_safe=False,
        limitations=[
            error_message,
            "Detection output is not production-ready.",
            "Detection output is not deployment-safe.",
        ],
        traceability=yolo_module.DetectionTraceability(
            checkpoint_path=_repo_relative(yolo_module.WEIGHTS_PATH),
            run_config_path=_repo_relative(yolo_module.RUN_CONFIG_PATH),
            model_config_path=_repo_relative(yolo_module.MODEL_CONFIG_PATH),
            source_contract=_repo_relative(yolo_module.SOURCE_CONTRACT_PATH),
            source_endpoint=source_endpoint,
        ),
    )


def _failed_anomaly_result(
    *,
    source_endpoint: str,
    error_message: str,
    helper: Any,
) -> AnomalyResult:
    return AnomalyResult(
        status="failed",
        model_name=getattr(helper, "model_name", anomaly_module.DEFAULT_MODEL_NAME),
        model_version=getattr(helper, "model_version", anomaly_module.DEFAULT_MODEL_VERSION),
        run_id=getattr(helper, "run_id", anomaly_module.EXPECTED_RUN_ID),
        anomaly_score=None,
        reconstruction_loss=None,
        threshold=_float_or_none(getattr(helper, "threshold", None)),
        predicted_label=None,
        decision=None,
        quality_status="unavailable",
        production_ready=False,
        deployment_safe=False,
        limitations=[
            error_message,
            "Anomaly output is not production-ready.",
            "Anomaly output is not deployment-safe.",
        ],
        traceability=AnomalyTraceability(
            checkpoint_path=_repo_relative(anomaly_module.CHECKPOINT_PATH),
            run_config_path=_repo_relative(anomaly_module.RUN_CONFIG_PATH),
            model_config_path=_repo_relative(anomaly_module.MODEL_CONFIG_PATH),
            evaluation_path=_repo_relative(anomaly_module.EVALUATION_PATH),
            source_endpoint=source_endpoint,
        ),
        optional_reconstruction_artifacts=None,
    )


def _build_warnings(*, errors: list[InspectionError], anomaly_result: AnomalyResult) -> list[str]:
    warnings: list[str] = []
    for error in errors:
        warnings.append(f"{error.component} subsystem failed: {error.message}")
    if anomaly_result.quality_status == "review_required_weak_evidence":
        warnings.append("Anomaly evidence is weak and review-only.")
    return warnings


def _deduplicate_messages(messages: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        deduplicated.append(message)
    return deduplicated


def _build_limitations(
    *,
    classification_result: ClassificationResult,
    detection_result: DetectionResult,
    anomaly_result: AnomalyResult,
    errors: list[InspectionError],
) -> list[str]:
    limitations = list(LIMITATIONS)
    if errors:
        limitations.append(
            "One or more model subsystems failed, so the inspection response is partial."
        )
    if anomaly_result.quality_status == "review_required_weak_evidence":
        limitations.append("Anomaly evidence is review-only because the governed anomaly quality is weak.")
    if classification_result.status != "success" or detection_result.status != "success" or anomaly_result.status != "success":
        limitations.append("Missing or failed model signals were handled conservatively.")
    return limitations


def _frontend_evidence_sources() -> list[str]:
    return [
        "artifacts/models/analysis/track_a_resnet18_v0_4_0_quality_decision__1bc92561-c5bf-48f2-8246-b8f3d5718ffe.json",
        "artifacts/models/predictions/detection_bbox_predictions__yolo_train_v0_2_0__validation.json",
        "artifacts/models/metrics/anomaly_pr_curve__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json",
        "artifacts/models/metrics/anomaly_threshold_sweep__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json",
        "artifacts/models/metrics/anomaly_score_distribution__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json",
        "artifacts/models/predictions/anomaly_sample_predictions__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json",
        "artifacts/models/metrics/anomaly_quality_decision__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json",
        "artifacts/models/inventory/anomaly_governed_evidence_inventory__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json",
    ]


def _inspection_error(*, component: str, code: str, message: str, recoverable: bool) -> InspectionError:
    return InspectionError(
        component=component,
        code=code,
        message=message,
        recoverable=recoverable,
    )


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file could not be decoded as an image.") from exc


def _resolve_suffix(filename: str | None, content_type: str) -> str:
    suffix_from_name = Path(filename).suffix.lower() if filename else ""
    if suffix_from_name in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix_from_name
    suffix_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }
    return suffix_map.get(content_type, ".img")


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_relative(path: Path | str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        try:
            return str(path_obj.relative_to(track_a_module.REPO_ROOT))
        except ValueError:
            return path_obj.as_posix()
    return path_obj.as_posix()


def _as_dict(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(payload, "__dict__"):
        return {key: value for key, value in vars(payload).items() if not key.startswith("_")}
    return {}


def _get_any(payload: Any, key: str, default: Any = None) -> Any:
    data = _as_dict(payload)
    return data.get(key, default)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

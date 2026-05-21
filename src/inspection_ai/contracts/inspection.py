"""Shared contracts for unified image inspection responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SubResultStatus = Literal["success", "skipped", "failed", "unavailable"]
FinalDecision = Literal[
    "good",
    "defective",
    "anomalous",
    "needs_manual_review",
    "inconclusive",
]
BBoxFormat = Literal["xyxy"]


class InputMetadata(BaseModel):
    """Metadata about the uploaded image inspection input."""

    filename: str
    content_type: str
    file_size_bytes: int = Field(ge=0)
    image_width: int = Field(ge=0)
    image_height: int = Field(ge=0)
    image_mode: str
    preprocessing_notes: list[str] = Field(default_factory=list)


class ClassificationTraceability(BaseModel):
    """Traceability fields for the classification result."""

    checkpoint_path: str | None = None
    run_config_path: str | None = None
    model_config_path: str | None = None
    preprocessing_config_path: str | None = None
    class_mapping_config_path: str | None = None
    quality_decision_path: str | None = None
    source_endpoint: str | None = None


class ClassificationResult(BaseModel):
    """Unified inspection classification sub-result."""

    status: SubResultStatus
    model_name: str | None = None
    model_version: str | None = None
    run_id: str | None = None
    threshold: float | None = None
    predicted_label: str | None = None
    predicted_label_id: int | None = None
    probability_good: float | None = None
    probability_defect: float | None = None
    decision: str | None = None
    production_ready: bool = False
    deployment_safe: bool = False
    limitations: list[str] = Field(default_factory=list)
    traceability: ClassificationTraceability = Field(default_factory=ClassificationTraceability)


class DetectionBox(BaseModel):
    """One detection/localization bounding box."""

    box_id: int = Field(ge=0)
    class_id: int = Field(ge=0)
    class_label: str
    display_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_format: BBoxFormat
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    score_rank: int = Field(ge=1)
    is_best_prediction: bool
    warnings: list[str] = Field(default_factory=list)


class DetectionTraceability(BaseModel):
    """Traceability fields for detection/localization."""

    checkpoint_path: str | None = None
    run_config_path: str | None = None
    model_config_path: str | None = None
    source_contract: str | None = None
    source_endpoint: str | None = None


class DetectionResult(BaseModel):
    """Unified inspection detection/localization sub-result."""

    status: SubResultStatus
    model_name: str | None = None
    model_version: str | None = None
    run_id: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    image_width: int | None = Field(default=None, ge=0)
    image_height: int | None = Field(default=None, ge=0)
    predicted_box_count: int | None = Field(default=None, ge=0)
    defect_count: int | None = Field(default=None, ge=0)
    detections: list[DetectionBox] = Field(default_factory=list)
    best_detection: DetectionBox | None = None
    review_status: str | None = None
    production_ready: bool = False
    deployment_safe: bool = False
    limitations: list[str] = Field(default_factory=list)
    traceability: DetectionTraceability = Field(default_factory=DetectionTraceability)


class AnomalyTraceability(BaseModel):
    """Traceability fields for anomaly detection."""

    checkpoint_path: str | None = None
    run_config_path: str | None = None
    model_config_path: str | None = None
    evaluation_path: str | None = None
    source_endpoint: str | None = None


class AnomalyResult(BaseModel):
    """Unified inspection anomaly detection sub-result."""

    status: SubResultStatus
    model_name: str | None = None
    model_version: str | None = None
    run_id: str | None = None
    anomaly_score: float | None = None
    reconstruction_loss: float | None = None
    threshold: float | None = None
    predicted_label: str | None = None
    decision: str | None = None
    quality_status: str | None = None
    production_ready: bool = False
    deployment_safe: bool = False
    limitations: list[str] = Field(default_factory=list)
    traceability: AnomalyTraceability = Field(default_factory=AnomalyTraceability)
    optional_reconstruction_artifacts: dict[str, Any] | None = None


class DecisionTraceability(BaseModel):
    """Traceability fields for the smart decision result."""

    rules_config_path: str | None = None
    rule_source: str | None = None
    source_endpoint: str | None = None


class DecisionResult(BaseModel):
    """Unified inspection smart decision result."""

    status: SubResultStatus
    final_decision: FinalDecision
    decision_level: str | None = None
    model_agreement_status: str | None = None
    primary_signal: str | None = None
    supporting_signals: list[str] = Field(default_factory=list)
    conflict_reason: str | None = None
    recommended_action: str | None = None
    rule_id: str | None = None
    rule_summary: str | None = None
    limitations: list[str] = Field(default_factory=list)
    traceability: DecisionTraceability = Field(default_factory=DecisionTraceability)


class TopLevelTraceability(BaseModel):
    """Top-level traceability for the unified inspection response."""

    contract_version: str
    api_version: str
    source_endpoint: str
    classification: dict[str, Any] = Field(default_factory=dict)
    detection: dict[str, Any] = Field(default_factory=dict)
    anomaly: dict[str, Any] = Field(default_factory=dict)
    decision: dict[str, Any] = Field(default_factory=dict)
    frontend_evidence_sources: list[str] = Field(default_factory=list)


class InspectionError(BaseModel):
    """Structured partial-failure or inspection error."""

    component: str
    code: str
    message: str
    recoverable: bool


class ExplanationContext(BaseModel):
    """Grounded context for future evidence explanation features."""

    status: str
    context_version: str
    allowed_sources: list[str] = Field(default_factory=list)
    summary_inputs: dict[str, Any] = Field(default_factory=dict)
    safety_boundaries: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


class ImageInspectionResponse(BaseModel):
    """Unified response schema for future image inspection."""

    request_id: str
    timestamp_utc: str
    input: InputMetadata
    classification: ClassificationResult
    detection: DetectionResult
    anomaly: AnomalyResult
    decision: DecisionResult
    traceability: TopLevelTraceability
    limitations: list[str] = Field(default_factory=list)
    errors: list[InspectionError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    explanation_context: ExplanationContext

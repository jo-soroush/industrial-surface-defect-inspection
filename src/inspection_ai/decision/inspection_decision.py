"""Deterministic smart decision layer for unified image inspection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..contracts.inspection import DecisionResult, DecisionTraceability


DEFAULT_RULES_CONFIG_PATH = "configs/decision/thresholds.yaml"
DEFAULT_SOURCE_ENDPOINT = "/inspect/image"
RULE_SOURCE = "src.inspection_ai.decision.inspection_decision.build_inspection_decision"


def build_inspection_decision(
    *,
    classification: Any | None = None,
    detection: Any | None = None,
    anomaly: Any | None = None,
    source_endpoint: str = DEFAULT_SOURCE_ENDPOINT,
    rules_config_path: str = DEFAULT_RULES_CONFIG_PATH,
) -> DecisionResult:
    """Combine classification, detection, and anomaly signals into one conservative decision."""

    _rules_config = _load_rules_config(rules_config_path)
    classification_state = _normalize_classification(classification)
    detection_state = _normalize_detection(detection)
    anomaly_state = _normalize_anomaly(anomaly)

    supporting_signals = _build_supporting_signals(
        classification_state=classification_state,
        detection_state=detection_state,
        anomaly_state=anomaly_state,
    )
    limitations = _build_limitations(
        classification_state=classification_state,
        detection_state=detection_state,
        anomaly_state=anomaly_state,
    )

    if classification_state.available is False and detection_state.available is False and anomaly_state.available is False:
        # No usable signals were available, so the safest contract-compatible status is unavailable.
        return DecisionResult(
            status="unavailable",
            final_decision="inconclusive",
            decision_level="inconclusive",
            model_agreement_status="no_signal",
            primary_signal=None,
            supporting_signals=supporting_signals,
            conflict_reason="No usable model signals were available.",
            recommended_action="Manual review or retry inspection.",
            rule_id="all_signals_unavailable",
            rule_summary="No usable model signals were available, so the inspection remains inconclusive.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    classification_good = classification_state.available and classification_state.predicted_label == "good"
    classification_defect = classification_state.available and classification_state.predicted_label in {"defect", "defective"}
    detection_has_boxes = detection_state.available and detection_state.predicted_box_count > 0
    detection_no_boxes = detection_state.available and detection_state.predicted_box_count == 0
    anomaly_normal = anomaly_state.available and anomaly_state.predicted_label == "normal"
    anomaly_anomaly = anomaly_state.available and anomaly_state.predicted_label == "anomaly"
    anomaly_weak = anomaly_state.available and anomaly_state.quality_status == "review_required_weak_evidence"

    # Rule 2 - aligned good evidence.
    if classification_good and detection_no_boxes and (anomaly_normal or not anomaly_state.available):
        return DecisionResult(
            status="success",
            final_decision="good",
            decision_level="auto_review_safe",
            model_agreement_status="all_available_signals_agree",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Accept as likely good, with safety limitations.",
            rule_id="good_all_signals_agree_v0",
            rule_summary="Classification is good, detection found no boxes, and anomaly is normal or unavailable.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # Rule 3 - classification defect plus localization.
    if classification_defect and detection_has_boxes:
        return DecisionResult(
            status="success",
            final_decision="defective",
            decision_level="evidence_supported",
            model_agreement_status="classification_detection_agree",
            primary_signal="detection",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Inspect localized defect boxes.",
            rule_id="classification_detection_agree_v0",
            rule_summary="Classification indicates defect and detection returned localized defect boxes.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # Rule 4 - detection boxes versus classification good.
    if classification_good and detection_has_boxes:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="conflict",
            primary_signal="detection",
            supporting_signals=supporting_signals,
            conflict_reason="Classification predicted good, but detection found localized defect boxes.",
            recommended_action="Manual review required.",
            rule_id="classification_good_detection_boxes_v0",
            rule_summary="Detection found boxes while classification predicted good.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # Rule 5 - classification defect without localization.
    if classification_defect and detection_state.available and detection_no_boxes:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="partial_or_conflict",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason="Classification predicted defect, but detection found no localized boxes.",
            recommended_action="Manual review required.",
            rule_id="classification_defect_no_detection_boxes_v0",
            rule_summary="Classification predicted defect without supporting localization evidence.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # Rule 6 and Rule 7 - weak anomaly evidence never finalizes the decision on its own.
    if anomaly_weak and anomaly_anomaly:
        if classification_good and (detection_no_boxes or not detection_state.available):
            return DecisionResult(
                status="success",
                final_decision="needs_manual_review",
                decision_level="review",
                model_agreement_status="conflict",
                primary_signal="anomaly",
                supporting_signals=supporting_signals,
                conflict_reason=(
                    "Weak anomaly evidence indicates anomaly while classification is good and "
                    "detection does not provide corroborating defect boxes."
                ),
                recommended_action="Manual review because anomaly evidence is weak/review-only.",
                rule_id="weak_anomaly_conflict_with_good_evidence_v0",
                rule_summary="Weak anomaly evidence conflicts with otherwise good or non-localized evidence.",
                limitations=limitations,
                traceability=_decision_traceability(rules_config_path, source_endpoint),
            )
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="supporting_only",
            primary_signal="anomaly",
            supporting_signals=supporting_signals,
            conflict_reason="Weak anomaly evidence is review-only and should not finalize the decision.",
            recommended_action="Manual review because anomaly evidence is weak/review-only.",
            rule_id="weak_anomaly_supporting_only_v0",
            rule_summary="Weak anomaly evidence is treated as supporting evidence only.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # Rule 8 - detection-only boxes.
    if detection_has_boxes and not classification_state.available and not anomaly_state.available:
        return DecisionResult(
            status="success",
            final_decision="defective",
            decision_level="evidence_supported",
            model_agreement_status="partial_signal",
            primary_signal="detection",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Inspect localized defect boxes.",
            rule_id="detection_only_boxes_v0",
            rule_summary="Detection returned localized boxes and no other model signals were available.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # Rule 9 - classification-only signals should not finalize good or defective.
    if classification_state.available and not detection_state.available and not anomaly_state.available:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="partial_signal",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason="Only classification was available; localization and anomaly corroboration were missing.",
            recommended_action="Manual review required.",
            rule_id="classification_only_review_v0",
            rule_summary="Classification alone is not sufficient to finalize the inspection decision.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if classification_state.available and not detection_state.available and anomaly_state.available:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="partial_signal",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason="Classification is available, but localization is missing and anomaly evidence is not sufficient to finalize.",
            recommended_action="Manual review required.",
            rule_id="classification_with_missing_localization_v0",
            rule_summary="Classification alone does not finalize the decision when localization is missing.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if detection_has_boxes and classification_state.available and not classification_good:
        # Covers classification defect plus boxes, but also any other non-good label that should still be conservative.
        return DecisionResult(
            status="success",
            final_decision="defective",
            decision_level="evidence_supported",
            model_agreement_status="classification_detection_agree",
            primary_signal="detection",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Inspect localized defect boxes.",
            rule_id="classification_detection_agree_v0",
            rule_summary="Classification and detection both support a defect decision.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if classification_good and detection_no_boxes and anomaly_state.available and anomaly_normal:
        return DecisionResult(
            status="success",
            final_decision="good",
            decision_level="auto_review_safe",
            model_agreement_status="all_available_signals_agree",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Accept as likely good, with safety limitations.",
            rule_id="good_all_signals_agree_v0",
            rule_summary="All available signals agree that the image is likely good.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if detection_has_boxes and not classification_state.available and anomaly_state.available and anomaly_normal:
        return DecisionResult(
            status="success",
            final_decision="defective",
            decision_level="evidence_supported",
            model_agreement_status="partial_signal",
            primary_signal="detection",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Inspect localized defect boxes.",
            rule_id="detection_only_boxes_v0",
            rule_summary="Detection returned localized boxes and the remaining available anomaly signal does not overturn them.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    # If a single weak/non-corroborated signal remains, stay conservative.
    if classification_good:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="partial_signal",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason="Classification predicted good, but the remaining signals did not provide enough corroboration.",
            recommended_action="Manual review required.",
            rule_id="classification_only_review_v0",
            rule_summary="Classification alone cannot finalize the inspection decision.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if classification_defect:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="partial_or_conflict",
            primary_signal="classification",
            supporting_signals=supporting_signals,
            conflict_reason="Classification predicted defect, but localization evidence was missing or non-corroborating.",
            recommended_action="Manual review required.",
            rule_id="classification_defect_review_v0",
            rule_summary="Classification predicted defect, but the decision does not have enough corroborating evidence.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if detection_has_boxes:
        return DecisionResult(
            status="success",
            final_decision="defective",
            decision_level="evidence_supported",
            model_agreement_status="partial_signal",
            primary_signal="detection",
            supporting_signals=supporting_signals,
            conflict_reason=None,
            recommended_action="Inspect localized defect boxes.",
            rule_id="detection_only_boxes_v0",
            rule_summary="Detection returned localized boxes and is treated as the strongest available signal.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if anomaly_weak and anomaly_anomaly:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="supporting_only",
            primary_signal="anomaly",
            supporting_signals=supporting_signals,
            conflict_reason="Weak anomaly evidence is review-only and not sufficient on its own.",
            recommended_action="Manual review because anomaly evidence is weak/review-only.",
            rule_id="weak_anomaly_supporting_only_v0",
            rule_summary="Weak anomaly evidence never finalizes the decision on its own.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    if anomaly_state.available:
        return DecisionResult(
            status="success",
            final_decision="needs_manual_review",
            decision_level="review",
            model_agreement_status="supporting_only",
            primary_signal="anomaly",
            supporting_signals=supporting_signals,
            conflict_reason="Anomaly output alone is not sufficient to finalize the inspection decision.",
            recommended_action="Manual review required.",
            rule_id="anomaly_only_review_v0",
            rule_summary="Anomaly evidence is treated as supporting evidence only in the first decision layer.",
            limitations=limitations,
            traceability=_decision_traceability(rules_config_path, source_endpoint),
        )

    return DecisionResult(
        status="unavailable",
        final_decision="inconclusive",
        decision_level="inconclusive",
        model_agreement_status="no_signal",
        primary_signal=None,
        supporting_signals=supporting_signals,
        conflict_reason="No usable model signals were available.",
        recommended_action="Manual review or retry inspection.",
        rule_id="all_signals_unavailable",
        rule_summary="No usable model signals were available, so the inspection remains inconclusive.",
        limitations=limitations,
        traceability=_decision_traceability(rules_config_path, source_endpoint),
    )


def _load_rules_config(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Decision rules config not found: {path.as_posix()}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Decision rules config is invalid: {path.as_posix()}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Decision rules config must contain a mapping.")
    return payload


def _normalize_classification(payload: Any | None) -> _SignalState:
    data = _as_dict(payload)
    status = _string_or_none(_get(data, "status"))
    if status != "success":
        return _SignalState(available=False, status=status or "unavailable")
    predicted_label = _normalize_label(
        _string_or_none(_get(data, "predicted_label")) or _string_or_none(_get(data, "decision"))
    )
    return _SignalState(
        available=True,
        status="success",
        predicted_label=predicted_label,
        threshold=_float_or_none(_get(data, "threshold")),
        probability_good=_float_or_none(_get(data, "probability_good")),
        probability_defect=_float_or_none(_get(data, "probability_defect")),
    )


def _normalize_detection(payload: Any | None) -> _SignalState:
    data = _as_dict(payload)
    status = _string_or_none(_get(data, "status"))
    if status != "success":
        return _SignalState(available=False, status=status or "unavailable")
    detections = _get(data, "detections")
    predicted_box_count = _int_or_none(_get(data, "predicted_box_count"))
    if predicted_box_count is None:
        predicted_box_count = len(detections) if isinstance(detections, list) else 0
    defect_count = _int_or_none(_get(data, "defect_count"))
    if defect_count is None:
        defect_count = predicted_box_count
    best_detection = _get(data, "best_detection")
    return _SignalState(
        available=True,
        status="success",
        predicted_box_count=predicted_box_count,
        defect_count=defect_count,
        best_detection=best_detection,
        threshold=_float_or_none(_get(data, "confidence_threshold")),
    )


def _normalize_anomaly(payload: Any | None) -> _SignalState:
    data = _as_dict(payload)
    status = _string_or_none(_get(data, "status"))
    if status != "success":
        return _SignalState(available=False, status=status or "unavailable")
    predicted_label = _normalize_label(
        _string_or_none(_get(data, "predicted_label")) or _string_or_none(_get(data, "decision"))
    )
    return _SignalState(
        available=True,
        status="success",
        predicted_label=predicted_label,
        threshold=_float_or_none(_get(data, "threshold")),
        quality_status=_string_or_none(_get(data, "quality_status")),
        anomaly_score=_float_or_none(_get(data, "anomaly_score")),
        reconstruction_loss=_float_or_none(_get(data, "reconstruction_loss")),
    )


def _build_supporting_signals(
    *,
    classification_state: _SignalState,
    detection_state: _SignalState,
    anomaly_state: _SignalState,
) -> list[str]:
    signals = [
        _classification_summary(classification_state),
        _detection_summary(detection_state),
        _anomaly_summary(anomaly_state),
    ]
    return [signal for signal in signals if signal is not None]


def _build_limitations(
    *,
    classification_state: _SignalState,
    detection_state: _SignalState,
    anomaly_state: _SignalState,
) -> list[str]:
    limitations = [
        "This decision is a deterministic rule-based aggregation of model outputs.",
        "This decision does not claim production readiness.",
        "This decision does not claim deployment safety.",
    ]
    if not (classification_state.available and detection_state.available and anomaly_state.available):
        limitations.append(
            "One or more model signals were missing, failed, or unavailable; the decision is conservative."
        )
    if anomaly_state.available and anomaly_state.quality_status == "review_required_weak_evidence":
        limitations.append(
            "Anomaly evidence is review-only because the governed anomaly quality status is weak."
        )
    return limitations


def _classification_summary(state: _SignalState) -> str:
    if not state.available:
        return "classification=unavailable"
    label = state.predicted_label or "unknown"
    summary = [f"classification={label}"]
    if state.threshold is not None:
        summary.append(f"threshold={state.threshold}")
    if state.probability_good is not None and state.probability_defect is not None:
        summary.append(
            f"probabilities=good:{state.probability_good:.6f},defect:{state.probability_defect:.6f}"
        )
    return ", ".join(summary)


def _detection_summary(state: _SignalState) -> str:
    if not state.available:
        return "detection=unavailable"
    summary = [
        f"detection_boxes={state.predicted_box_count}",
        f"detection_defects={state.defect_count}",
    ]
    if state.best_detection is not None:
        summary.append("detection_best_detection=present")
    return ", ".join(summary)


def _anomaly_summary(state: _SignalState) -> str:
    if not state.available:
        return "anomaly=unavailable"
    label = state.predicted_label or "unknown"
    summary = [f"anomaly={label}"]
    if state.quality_status is not None:
        summary.append(f"anomaly_quality={state.quality_status}")
    if state.threshold is not None:
        summary.append(f"threshold={state.threshold}")
    return ", ".join(summary)


def _decision_traceability(rules_config_path: str, source_endpoint: str) -> DecisionTraceability:
    return DecisionTraceability(
        rules_config_path=rules_config_path,
        rule_source=RULE_SOURCE,
        source_endpoint=source_endpoint,
    )


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


def _get(payload: dict[str, Any], key: str) -> Any:
    return payload.get(key)


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"defect", "defective"}:
        return "defect"
    if normalized in {"good", "normal"}:
        return normalized
    if normalized == "anomaly":
        return "anomaly"
    return normalized or None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _SignalState:
    """Minimal normalized signal state used by the deterministic decision layer."""

    def __init__(
        self,
        *,
        available: bool,
        status: str,
        predicted_label: str | None = None,
        threshold: float | None = None,
        probability_good: float | None = None,
        probability_defect: float | None = None,
        predicted_box_count: int = 0,
        defect_count: int = 0,
        best_detection: Any | None = None,
        quality_status: str | None = None,
        anomaly_score: float | None = None,
        reconstruction_loss: float | None = None,
    ) -> None:
        self.available = available
        self.status = status
        self.predicted_label = predicted_label
        self.threshold = threshold
        self.probability_good = probability_good
        self.probability_defect = probability_defect
        self.predicted_box_count = predicted_box_count
        self.defect_count = defect_count
        self.best_detection = best_detection
        self.quality_status = quality_status
        self.anomaly_score = anomaly_score
        self.reconstruction_loss = reconstruction_loss

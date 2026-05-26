"""Tests for the agent grounding context builder."""

from __future__ import annotations

import pytest

from src.inspection_ai.agent.context_builder import (
    build_grounding_context,
    get_allowed_page_ids,
    get_allowed_section_ids,
    validate_page_section,
)
from src.inspection_ai.agent.evidence_loader import EvidenceLoadResult


def test_supported_pages_and_sections_are_loaded() -> None:
    pages = get_allowed_page_ids()
    assert "image_inspection" in pages
    assert "overview" in pages

    sections = get_allowed_section_ids("image_inspection")
    assert "final_decision" in sections
    assert "technical_evidence" in sections


def test_validate_page_section_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="Unsupported section_id"):
        validate_page_section("image_inspection", "unknown_section")


def test_build_grounding_context_extracts_image_inspection_evidence() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "decision_level": "review",
                "rule_id": "classification_detection_agree_v0",
            },
            "classification": {
                "predicted_label": "defect",
                "probability_defect": 0.91,
            },
            "detection": {
                "predicted_box_count": 2,
            },
            "anomaly": {
                "quality_status": "review_required_weak_evidence",
            },
            "traceability": {"source_endpoint": "/inspect/image"},
            "warnings": ["review-only"],
            "limitations": ["manual review required"],
        },
        include_raw_evidence=True,
    )

    sources = [item["source"] for item in context.evidence_used]
    assert "inspection_response.decision.final_decision" in sources
    assert "inspection_response.classification.predicted_label" in sources
    assert "inspection_response.detection.predicted_box_count" in sources
    assert "inspection_response.anomaly.quality_status" in sources
    assert "inspection_response.traceability.source_endpoint" in sources
    assert "inspection_response.warnings" in sources
    assert "inspection_response.limitations" in sources
    assert "inspection_response.raw" in sources
    assert context.grounding_status == "grounded"


def test_empty_context_is_marked_insufficient_evidence() -> None:
    context = build_grounding_context(
        page_id="overview",
        section_id="status_summary",
        question="What should I know?",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    assert context.grounding_status == "insufficient_evidence"
    assert any(item["source"] == "request.page_id" for item in context.evidence_used)
    assert any(item["source"] == "request.section_id" for item in context.evidence_used)


def test_component_context_loads_classification_threshold_curve_evidence() -> None:
    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question="What does this chart mean?",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    sources = [item["source"] for item in context.evidence_used]
    assert "request.component_id" in sources
    assert "component.user_facing_label" in sources
    assert "artifacts/frontend/track_a/threshold_curve_chart_data.json#recommended_threshold" in sources
    assert "artifacts/frontend/track_a/threshold_curve_chart_data.json#rows.threshold" in sources
    assert context.grounding_status == "grounded"


def test_component_context_loads_detection_confidence_evidence() -> None:
    context = build_grounding_context(
        page_id="detection",
        section_id="visual_evidence",
        component_id="detection_confidence_chart",
        question="What does confidence mean?",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    sources = [item["source"] for item in context.evidence_used]
    assert (
        "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json#confidence_bins.count"
        in sources
    )
    assert context.grounding_status == "grounded"


def test_component_context_preserves_anomaly_review_only_limitation() -> None:
    context = build_grounding_context(
        page_id="anomaly",
        section_id="visual_evidence",
        component_id="anomaly_threshold_behavior_chart",
        question="What does this threshold mean?",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    joined_limitations = " ".join(context.limitations).lower()
    assert "weak" in joined_limitations
    assert "review-only" in joined_limitations
    assert context.grounding_status == "grounded"


def test_component_context_loads_image_inspection_runtime_evidence() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        component_id="image_inspection_final_decision_card",
        question="Why is this result good?",
        visible_context={},
        inspection_response={
            "request_id": "request-123",
            "decision": {
                "final_decision": "good",
                "rule_id": "manual_check_rule",
                "recommended_action": "manual_review",
            },
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    sources = [item["source"] for item in context.evidence_used]
    assert "inspection_response#decision.final_decision" in sources
    assert "inspection_response#request_id" in sources
    assert context.grounding_status == "grounded"


def test_component_context_rejects_invalid_component_id() -> None:
    with pytest.raises(ValueError, match="Unknown component definition"):
        build_grounding_context(
            page_id="classification",
            section_id="detailed_metrics",
            component_id="not_a_real_component",
            question="Explain this.",
            visible_context={},
            inspection_response={},
            include_raw_evidence=False,
        )


def test_component_context_blocks_raw_evidence_when_component_disallows_it() -> None:
    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question="Show raw evidence.",
        visible_context={},
        inspection_response={"decision": {"final_decision": "good"}},
        include_raw_evidence=True,
    )

    sources = [item["source"] for item in context.evidence_used]
    assert "inspection_response.raw" not in sources
    assert context.raw_evidence_included is False
    assert any("Raw evidence is disabled" in limitation for limitation in context.limitations)


def test_component_context_missing_evidence_does_not_crash(monkeypatch) -> None:
    def fake_load_component_evidence(*args, **kwargs):
        return EvidenceLoadResult(
            evidence_items=[],
            limitations=["Evidence file unavailable: artifacts/frontend/missing.json."],
            missing_fields=[],
            missing_files=["artifacts/frontend/missing.json"],
            raw_evidence_included=False,
        )

    monkeypatch.setattr(
        "src.inspection_ai.agent.context_builder.load_component_evidence",
        fake_load_component_evidence,
    )

    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question="Explain this.",
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )

    assert context.grounding_status == "insufficient_evidence"
    assert any("Evidence file unavailable" in limitation for limitation in context.limitations)

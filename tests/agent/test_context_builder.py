"""Tests for the agent grounding context builder."""

from __future__ import annotations

import pytest

from src.inspection_ai.agent.context_builder import (
    build_grounding_context,
    get_allowed_page_ids,
    get_allowed_section_ids,
    validate_page_section,
)


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

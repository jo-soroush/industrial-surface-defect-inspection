"""Tests for the deterministic Agent safety guard."""

from __future__ import annotations

from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.safety_guard import (
    guard_post_generation_text,
    guard_pre_generation_context,
)


def test_pre_generation_guard_passes_safe_compact_context() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "request_id": "request-001",
            "decision": {
                "final_decision": "good",
                "rule_id": "manual_check_rule",
                "recommended_action": "manual_review",
            },
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    result = guard_pre_generation_context(context)

    assert result.status == "pass"
    assert result.blocked is False
    assert result.safe_to_send is True
    assert result.safe_to_display is True
    assert result.sanitized_context["page_id"] == "image_inspection"
    assert result.sanitized_context["section_id"] == "final_decision"
    assert any(item["source"] == "inspection_response.decision.final_decision" for item in result.sanitized_context["evidence_used"])
    assert any(item["source"] == "inspection_response.traceability.source_endpoint" for item in result.sanitized_context["evidence_used"])


def test_pre_generation_guard_blocks_raw_evidence_when_present() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Show raw evidence.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "request_id": "request-raw",
            "decision": {
                "final_decision": "good",
                "rule_id": "manual_check_rule",
            },
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=True,
    )

    result = guard_pre_generation_context(context)

    assert result.blocked is True
    assert result.status == "blocked"
    assert any("raw evidence" in reason.lower() for reason in result.reasons)
    assert all("inspection_response.raw" not in item["source"] for item in result.sanitized_context["evidence_used"])


def test_pre_generation_guard_redacts_secret_like_and_absolute_path_values() -> None:
    context = build_grounding_context(
        page_id="overview",
        section_id="status_summary",
        question="What does this mean?",
        visible_context={
            "page_title": "Overview",
            "api_note": "sk-test-0123456789abcdef",
            "artifact_path": "/Users/jo.soroush/private/model.png",
        },
        inspection_response={},
        include_raw_evidence=False,
    )

    result = guard_pre_generation_context(context)

    assert result.blocked is False
    assert result.status == "limited"
    assert result.sanitized_context["visible_context"]["api_note"] == "[REDACTED_SECRET]"
    assert result.sanitized_context["visible_context"]["artifact_path"] == "[REDACTED_PATH]"


def test_pre_generation_guard_preserves_manual_review_and_traceability() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this result defective?",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "request_id": "request-002",
            "decision": {
                "final_decision": "defective",
                "rule_id": "manual_check_rule",
                "recommended_action": "manual_review",
            },
            "limitations": ["manual review still applies"],
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    result = guard_pre_generation_context(context)

    assert result.blocked is False
    assert "manual review still applies" in " ".join(result.limitations).lower()
    assert any(item["source"] == "inspection_response.traceability.source_endpoint" for item in result.sanitized_context["evidence_used"])


def test_post_generation_guard_blocks_prohibited_claims() -> None:
    unsafe_answers = [
        "This dashboard is production-ready and deployment-safe.",
        "Manual review is not required because the Agent replaces human review.",
        "The Agent makes the autonomous final decision.",
        "Gemini is connected and active.",
        "OpenAI is integrated and running.",
        "The threshold is 0.97 and accuracy is 99% even though the evidence does not support that.",
    ]

    for answer in unsafe_answers:
        result = guard_post_generation_text(
            answer,
            allowed_evidence_values=["0.42", "0.65", "0.73"],
        )
        assert result.blocked is True
        assert result.status == "blocked"


def test_post_generation_guard_allows_not_ready_disclaimers() -> None:
    answer = "This dashboard is not production-ready and not deployment-safe. Manual review still applies."

    result = guard_post_generation_text(
        answer,
        allowed_evidence_values=["0.42", "0.65", "0.73"],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}


def test_post_generation_guard_blocks_mixed_readiness_claims() -> None:
    mixed_answers = [
        "This dashboard is not production-ready, but it is deployment-safe.",
        "This dashboard is not deployment-safe, but it is production-ready.",
    ]

    for answer in mixed_answers:
        result = guard_post_generation_text(
            answer,
            allowed_evidence_values=["0.42", "0.65", "0.73"],
        )
        assert result.blocked is True
        assert result.status == "blocked"


def test_post_generation_guard_allows_provider_disabled_statements() -> None:
    answer = "Gemini/Grok/OpenAI are not active."

    result = guard_post_generation_text(
        answer,
        allowed_evidence_values=["0.42", "0.65", "0.73"],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}


def test_post_generation_guard_blocks_mixed_provider_claims() -> None:
    mixed_answers = [
        "Gemini is not active, but OpenAI is connected and running.",
        "External LLM providers are not connected, but Gemini is integrated.",
    ]

    for answer in mixed_answers:
        result = guard_post_generation_text(
            answer,
            allowed_evidence_values=["0.42", "0.65", "0.73"],
        )
        assert result.blocked is True
        assert result.status == "blocked"


def test_post_generation_guard_allows_safe_mock_answers() -> None:
    answer = (
        "This is a mock evidence-grounded explanation. "
        "External LLM not connected. Manual review still applies. "
        "The result is not production-ready and not deployment-safe."
    )

    result = guard_post_generation_text(
        answer,
        allowed_evidence_values=["0.42", "0.65", "0.73"],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}
    assert "manual review" in answer.lower()


def test_post_generation_guard_allows_grounded_probability_percentage_equivalent() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "decision_level": "evidence_supported",
                "rule_id": "manual_check_rule",
                "recommended_action": "Review the inspection evidence before taking action.",
            },
            "classification": {
                "predicted_label": "defect",
                "probability_defect": 1.0,
            },
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    result = guard_post_generation_text(
        "The final decision is defective. The defect probability is 100%. Manual review still applies.",
        grounding_context=context,
        allowed_evidence_values=["defective", "evidence_supported", "defect", 1.0, "Review the inspection evidence before taking action."],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}


def test_post_generation_guard_allows_grounded_detection_percentage_display_equivalents() -> None:
    context = build_grounding_context(
        page_id="detection",
        section_id="visual_evidence",
        component_id="detection_confidence_chart",
        question="Explain only what this detection confidence distribution chart means using the chart evidence.",
        visible_context={
            "page_title": "Defect Detection & Localization",
            "component_label": "Detection confidence distribution",
            "explanation_scope": "confidence_distribution_chart_only",
            "forbidden_summary_scope": "Do not summarize final image decisions or live image inspection results.",
            "manual_review_required": True,
            "chart_title": "Detection confidence distribution",
            "chart_explanation": "Counts of predicted boxes by confidence band on the validation split.",
            "run_id": "yolo_train_v0_2_0",
            "model_name": "YOLOv8",
            "model_version": "0.2.0",
            "image_count": 345,
            "total_bbox_count": 573,
            "confidence_bin_count": 4,
        },
        inspection_response={},
        include_raw_evidence=False,
    )

    result = guard_post_generation_text(
        (
            "The Detection confidence distribution chart explains the counts of predicted boxes by confidence band on the validation split for the yolo model, version 0.2.0, from run yolo_train_v0_2_0. "
            "For example, 226 predicted boxes (39.44%) had a confidence between 0.25 and 0.50, while 218 boxes (38.05%) were in the 0.50-0.75 confidence range. "
            "No predicted boxes fell into the 0.00-0.25 confidence band, and 129 boxes (22.51%) had a confidence between 0.75 and 1.00. "
            "Manual review still applies."
        ),
        grounding_context=context,
        allowed_evidence_values=[
            "Detection confidence distribution",
            "Counts of predicted boxes by confidence band on the validation split.",
            "0.00-0.25",
            "0.25-0.50",
            "0.50-0.75",
            "0.75-1.00",
            0,
            226,
            218,
            129,
            0.0,
            39.44153577661431,
            38.045375218150085,
            22.5130890052356,
            "yolo_train_v0_2_0",
            "YOLOv8",
            "0.2.0",
        ],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}


def test_post_generation_guard_ignores_semantic_version_numbers_in_generated_text() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "decision_level": "evidence_supported",
                "rule_id": "manual_check_rule",
                "recommended_action": "Review the inspection evidence before taking action.",
            },
            "classification": {
                "predicted_label": "defect",
                "model_version": "resnet18 v0.4.0",
            },
            "detection": {
                "predicted_box_count": 19,
                "model_version": "yolo v0.2.0",
            },
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    result = guard_post_generation_text(
        "The models are yolo v0.2.0 and resnet18 v0.4.0. Confidence remains grounded. Manual review still applies.",
        grounding_context=context,
        allowed_evidence_values=[
            "defective",
            "evidence_supported",
            "defect",
            19,
            "resnet18 v0.4.0",
            "yolo v0.2.0",
            "Review the inspection evidence before taking action.",
        ],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}


def test_post_generation_guard_blocks_invented_metrics_even_when_other_values_are_grounded() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "decision_level": "evidence_supported",
                "rule_id": "manual_check_rule",
                "recommended_action": "Review the inspection evidence before taking action.",
            },
            "classification": {
                "predicted_label": "defect",
                "probability_defect": 1.0,
            },
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    result = guard_post_generation_text(
        "The accuracy is 98%. Manual review still applies.",
        grounding_context=context,
        allowed_evidence_values=["defective", "evidence_supported", "defect", 1.0],
    )

    assert result.blocked is True
    assert result.status == "blocked"


def test_post_generation_guard_blocks_invented_percentage_when_not_grounded() -> None:
    result = guard_post_generation_text(
        "The accuracy is 98%. Manual review still applies.",
        allowed_evidence_values=["defective", "evidence_supported", "defect"],
    )

    assert result.blocked is True
    assert result.status == "blocked"


def test_post_generation_guard_allows_grounded_image_inspection_answer_with_probability_confidence_and_versions() -> None:
    context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Explain this inspection result safely for manual review.",
        visible_context={"page_title": "Image Inspection"},
        inspection_response={
            "decision": {
                "final_decision": "defective",
                "decision_level": "evidence_supported",
                "rule_id": "local_gated_runtime_validation_rule",
                "recommended_action": "Review the inspection evidence before taking action.",
            },
            "classification": {
                "predicted_label": "defect",
                "probability_defect": 1.0,
                "model_version": "resnet18 v0.4.0",
            },
            "detection": {
                "predicted_box_count": 19,
                "best_detection": {
                    "class_label": "oil_spot",
                    "display_label": "Oil spot",
                    "confidence": 0.890102744102478,
                },
                "model_version": "yolo v0.2.0",
            },
            "traceability": {
                "source_endpoint": "local_gated_agent_endpoint_validation",
                "contract_version": "local_validation_v1",
            },
        },
        include_raw_evidence=False,
    )

    answer = (
        "The final decision for this image is defective. "
        "This decision is evidence_supported because both the classification model and the detection model identified defects, "
        "with the detection model finding 19 localized defect boxes. "
        "The classification model reports 100% probability of defect. "
        "The models are yolo v0.2.0 and resnet18 v0.4.0. "
        "The most confident detection was an Oil spot with 0.89 confidence. "
        "The recommended action is to Review the inspection evidence before taking action. "
        "Manual review still applies."
    )

    result = guard_post_generation_text(
        answer,
        grounding_context=context,
        allowed_evidence_values=[
            "defective",
            "evidence_supported",
            "defect",
            19,
            1.0,
            0.890102744102478,
            "resnet18 v0.4.0",
            "yolo v0.2.0",
            "Oil spot",
            "Review the inspection evidence before taking action.",
        ],
    )

    assert result.blocked is False
    assert result.status in {"pass", "limited"}
    assert "manual review still applies" in result.sanitized_text.lower()

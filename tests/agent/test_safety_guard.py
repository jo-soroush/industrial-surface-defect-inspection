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

"""Tests for the Gemini Phase G2 mocked client seam."""

from __future__ import annotations

import ast
from pathlib import Path

from src.inspection_ai.agent.context_builder import build_grounding_context
from src.inspection_ai.agent.gemini_provider import (
    GeminiClientResult,
    GeminiProviderStub,
)
from src.inspection_ai.agent.provider_contracts import build_provider_request
from src.inspection_ai.agent.provider_router import AgentProviderRouter


REPO_ROOT = Path(__file__).resolve().parents[2]
GEMINI_PROVIDER_PATH = REPO_ROOT / "src/inspection_ai/agent/gemini_provider.py"


def test_gemini_mocked_client_success_translates_to_gemini_response() -> None:
    stub = GeminiProviderStub()
    request = _build_gemini_provider_request(
        question="What does this classification threshold chart mean?",
    )

    evaluation = stub.evaluate_mocked_client_result(
        request,
        GeminiClientResult(
            text=(
                "This threshold chart summarizes validation evidence and threshold behavior. "
                "Manual review still applies."
            ),
        ),
    )

    assert evaluation.provider_response.provider_used == "gemini"
    assert evaluation.provider_response.fallback_used is False
    assert evaluation.provider_response.raw_provider_response_allowed is False
    assert evaluation.provider_response.safety_status in {"pass", "limited"}
    assert evaluation.safe_to_send is True
    assert evaluation.safe_to_display is True
    assert "manual review" in evaluation.provider_response.answer.lower()
    assert "validation evidence" in evaluation.provider_response.answer.lower()


def test_gemini_mocked_client_timeout_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(error_kind="timeout"),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.raw_provider_response_allowed is False
    assert evaluation.provider_error is not None
    assert "timeout" in evaluation.provider_error.lower()
    assert evaluation.fallback_reason is not None
    assert "mock fallback" in evaluation.fallback_reason.lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_provider_error_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(error_kind="provider_error"),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "provider error" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_rate_limit_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(error_kind="rate_limit"),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "rate limited" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_empty_response_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(text=""),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "empty response" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_malformed_response_falls_back_to_mock() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(payload=["not", "a", "dict"]),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert "malformed" in (evaluation.provider_error or "").lower()
    assert evaluation.safe_to_display is True


def test_gemini_mocked_client_unsafe_output_is_blocked_by_safety_guard() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(
            text=(
                "This model is production-ready, deployment-safe, and Gemini is active. "
                "Manual review is not required."
            ),
        ),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.safety_status == "blocked"
    assert evaluation.safe_to_display is False
    assert "manual review" in evaluation.provider_response.answer.lower()
    assert "production-ready" not in evaluation.provider_response.answer.lower()


def test_gemini_mocked_client_invented_metric_like_output_is_blocked_by_safety_guard() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(
            text="The threshold is 0.99 and the F1 score is 0.87. Manual review still applies.",
        ),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.safety_status == "blocked"
    assert evaluation.safe_to_display is False
    assert evaluation.provider_error is not None


def test_gemini_mocked_client_readiness_claims_are_blocked_separately() -> None:
    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        _build_gemini_provider_request(),
        GeminiClientResult(
            text=(
                "This model is production-ready, deployment-safe, and Gemini is active. "
                "Manual review is not required."
            ),
        ),
    )

    assert evaluation.provider_response.provider_used == "mock"
    assert evaluation.provider_response.fallback_used is True
    assert evaluation.provider_response.safety_status == "blocked"
    assert evaluation.safe_to_display is False
    assert "production-ready" not in evaluation.provider_response.answer.lower()


def test_gemini_mocked_client_sanitizes_secret_like_questions_before_handling() -> None:
    request = _build_gemini_provider_request(
        question="Explain /Users/jo.soroush/secret.key",
    )

    evaluation = GeminiProviderStub().evaluate_mocked_client_result(
        request,
        GeminiClientResult(
            text="This is a safe mocked Gemini answer. Manual review still applies.",
        ),
    )

    assert request.provider_name == "gemini"
    assert request.question == "[REDACTED_SECRET]"
    assert request.sanitized_context["question"] == "[REDACTED_SECRET]"
    assert request.grounding_context["question"] == "[REDACTED_SECRET]"
    assert "/Users/jo.soroush/secret.key" not in request.question
    assert "/Users/jo.soroush/secret.key" not in evaluation.provider_response.answer
    assert evaluation.provider_response.provider_used == "gemini"
    assert evaluation.safe_to_display is True


def test_gemini_provider_module_does_not_import_provider_sdks_or_network_libraries() -> None:
    tree = ast.parse(GEMINI_PROVIDER_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    banned_roots = {"google", "openai", "requests", "httpx", "urllib"}
    assert imported_roots.isdisjoint(banned_roots)


def test_existing_agent_provider_router_normal_mock_path_remains_unchanged() -> None:
    router = AgentProviderRouter()
    grounding_context = build_grounding_context(
        page_id="image_inspection",
        section_id="final_decision",
        question="Why is this image defective?",
        visible_context={},
        inspection_response={
            "decision": {"final_decision": "defective", "rule_id": "manual_check_rule"},
            "classification": {"predicted_label": "defect"},
            "detection": {"predicted_box_count": 1},
            "anomaly": {"predicted_label": "anomaly"},
            "traceability": {"source_endpoint": "/inspect/image"},
        },
        include_raw_evidence=False,
    )

    response = router.explain(grounding_context)

    assert response.provider_used == "mock"
    assert response.fallback_used is True
    assert "manual review" in response.answer.lower()


def _build_gemini_provider_request(*, question: str = "What does this chart mean?"):
    context = build_grounding_context(
        page_id="classification",
        section_id="detailed_metrics",
        component_id="classification_threshold_curve_chart",
        question=question,
        visible_context={},
        inspection_response={},
        include_raw_evidence=False,
    )
    sanitized_context = {
        "page_id": context.page_id,
        "section_id": context.section_id,
        "component_id": context.component_id,
        "question": "[REDACTED_SECRET]"
        if "secret.key" in question
        else context.question,
        "evidence_used": context.evidence_used,
        "limitations": context.limitations,
        "grounding_status": context.grounding_status,
    }
    return build_provider_request(
        provider_name="gemini",
        grounding_context=context,
        sanitized_context=sanitized_context,
        safety_status="pass",
        llm_enabled=True,
    )

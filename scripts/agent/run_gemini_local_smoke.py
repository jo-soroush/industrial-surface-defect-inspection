"""Disabled-by-default local Gemini smoke harness.

This module exposes the explicit local-only manual real-smoke path behind a
strict approval gate. Default behavior remains dry-run, and normal runtime
remains mock-first. Importing the module does not run a real Gemini call.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SMOKE_MODEL_NAME = "gemini-2.5-flash"


def _ensure_repo_root_importable() -> None:
    repo_root_str = str(REPO_ROOT)
    src_root_str = str(REPO_ROOT / "src")
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


_ensure_repo_root_importable()

from src.inspection_ai.agent.context_builder import (
    AgentGroundingContext,
    build_grounding_context,
)
from src.inspection_ai.agent.gemini_provider import (
    GeminiRealGenerationResult,
    GeminiRealProviderConfig,
    GeminiSdkLoadResult,
    _load_google_genai_module,
    generate_with_real_gemini_provider,
)
from src.inspection_ai.agent.provider_contracts import (
    AgentProviderRequest,
    ProviderRuntimeSettings,
    build_provider_request,
)
from src.inspection_ai.agent.safety_guard import guard_pre_generation_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disabled-by-default local Gemini smoke harness skeleton."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned checks only. This is the default behavior.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the approved local-only real-smoke path.",
    )
    parser.add_argument(
        "--i-understand-this-calls-gemini",
        action="store_true",
        help="Required for any future non-dry-run path.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Future sanitized question placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--page-id",
        default="",
        help="Future page identifier placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--section-id",
        default="",
        help="Future section identifier placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--component-id",
        default="",
        help="Future component identifier placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SMOKE_MODEL_NAME,
        help="Smoke-only Gemini model name. This does not change normal runtime defaults.",
    )
    return parser


def build_dry_run_lines(*, smoke_model_name: str) -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=DRY_RUN",
        f"smoke_model={smoke_model_name}",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
        "planned_checks=lazy_sdk_import,key_presence,safety_guard,fallback,normal_route_unchanged",
    )


def build_blocked_lines(*, smoke_model_name: str) -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=BLOCKED",
        "reason=missing_confirmation_flag",
        f"smoke_model={smoke_model_name}",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
    )


def build_missing_fields_lines(missing_fields: tuple[str, ...], *, smoke_model_name: str) -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=BLOCKED",
        "reason=missing_required_smoke_fields",
        f"missing_fields={','.join(missing_fields)}",
        f"smoke_model={smoke_model_name}",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
    )


def build_success_lines(
    *,
    request: AgentProviderRequest,
    result: GeminiRealGenerationResult,
    smoke_model_name: str,
) -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=SUCCESS",
        f"smoke_model={smoke_model_name}",
        f"result_status={result.status}",
        f"provider_used={result.provider_response.provider_used}",
        f"fallback_used={str(result.provider_response.fallback_used).lower()}",
        f"grounding_status={result.provider_response.grounding_status}",
        f"safety_status={result.provider_response.safety_status}",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        _request_summary_line(request),
        _response_summary_line(result),
        "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock",
    )


def build_failure_lines(
    *,
    request: AgentProviderRequest,
    result: GeminiRealGenerationResult,
    smoke_model_name: str,
) -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=FAILED",
        f"smoke_model={smoke_model_name}",
        f"result_status={result.status}",
        f"error_category={_safe_error_category(result)}",
        f"provider_used={result.provider_response.provider_used}",
        f"fallback_used={str(result.provider_response.fallback_used).lower()}",
        f"grounding_status={result.provider_response.grounding_status}",
        f"safety_status={result.provider_response.safety_status}",
        *_safe_provider_error_diagnostic_lines(result),
        *_safe_safety_diagnostic_lines(result),
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        _request_summary_line(request),
        _response_summary_line(result),
        "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smoke_model_name = _resolve_smoke_model_name(args.model_name)

    if smoke_model_name is None:
        for line in build_invalid_model_lines():
            print(line)
        return 2

    if not args.execute or args.dry_run:
        for line in build_dry_run_lines(smoke_model_name=smoke_model_name):
            print(line)
        return 0

    if not args.i_understand_this_calls_gemini:
        for line in build_blocked_lines(smoke_model_name=smoke_model_name):
            print(line)
        return 2

    exit_code, lines = run_explicit_real_smoke_attempt(args, smoke_model_name=smoke_model_name)
    for line in lines:
        print(line)
    return exit_code


def run_explicit_real_smoke_attempt(args: argparse.Namespace, *, smoke_model_name: str) -> tuple[int, tuple[str, ...]]:
    missing_fields = _missing_execute_fields(args)
    if missing_fields:
        return 2, build_missing_fields_lines(missing_fields, smoke_model_name=smoke_model_name)

    try:
        visible_context = _minimal_smoke_visible_context()
        inspection_response = _minimal_smoke_inspection_response()
        smoke_question = _smoke_question_with_evidence_only_instructions(args.question)
        grounding_context = build_grounding_context(
            page_id=args.page_id,
            section_id=args.section_id,
            component_id=args.component_id,
            question=smoke_question,
            visible_context=visible_context,
            inspection_response=inspection_response,
            include_raw_evidence=False,
        )
    except Exception:  # pragma: no cover - defensive CLI guard
        return 2, (
            "gemini_local_smoke_status=BLOCKED",
            "reason=grounding_context_validation_failed",
            f"smoke_model={smoke_model_name}",
            "no_real_gemini_api_call_was_made=true",
            "gemini_api_key_read=false",
            "normal_agent_route=mock_first",
            "provider_routing_activation=disabled",
            "future_real_smoke_requires_explicit_user_approval=true",
        )

    pre_guard = guard_pre_generation_context(grounding_context)
    if pre_guard.blocked:
        return 2, (
            "gemini_local_smoke_status=BLOCKED",
            "reason=pre_generation_safety_guard_blocked",
            f"smoke_model={smoke_model_name}",
            "no_real_gemini_api_call_was_made=true",
            "gemini_api_key_read=false",
            "normal_agent_route=mock_first",
            "provider_routing_activation=disabled",
            "future_real_smoke_requires_explicit_user_approval=true",
        )

    provider_request = build_provider_request(
        provider_name="gemini",
        grounding_context=grounding_context,
        sanitized_context=pre_guard.sanitized_context,
        safety_status=pre_guard.status,
        llm_enabled=True,
    )

    settings = ProviderRuntimeSettings(
        enable_llm=True,
        default_provider="gemini",
        provider_order=("gemini", "mock"),
        enable_fallback=True,
        gemini_key_present=_gemini_api_key_present(),
        grok_key_present=False,
        openai_key_present=False,
    )
    config = GeminiRealProviderConfig(
        model_name=smoke_model_name,
        real_provider_implemented=True,
        sdk_import_allowed=True,
        api_key_resolver=_resolve_gemini_api_key,
    )

    try:
        result = generate_with_real_gemini_provider(
            provider_request,
            settings=settings,
            config=config,
            sdk_loader=_load_gemini_sdk_readiness_result,
            sdk_module_loader=_load_google_genai_module,
            allowed_evidence_values=_allowed_evidence_values(grounding_context),
        )
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        return 1, (
            "gemini_local_smoke_status=FAILED",
            "result_status=exception",
            f"smoke_model={smoke_model_name}",
            "provider_used=unknown",
            "fallback_used=unknown",
            "grounding_status=unknown",
            "safety_status=unknown",
            "normal_agent_route=mock_first",
            "provider_routing_activation=disabled",
            _request_summary_line(provider_request),
            "response_summary=manual_review_visible=true;sanitized=true;raw_response_hidden=true",
            "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock",
            f"failure_reason={type(exc).__name__};sanitized=true",
        )

    if not isinstance(result, GeminiRealGenerationResult):
        return 1, (
            "gemini_local_smoke_status=FAILED",
            "result_status=malformed_result",
            f"smoke_model={smoke_model_name}",
            "provider_used=unknown",
            "fallback_used=unknown",
            "grounding_status=unknown",
            "safety_status=unknown",
            "normal_agent_route=mock_first",
            "provider_routing_activation=disabled",
            _request_summary_line(provider_request),
            "response_summary=manual_review_visible=true;sanitized=true;raw_response_hidden=true",
            "cleanup_reminder=unset_temporary_key_and_restore_mock_defaults;export AGENT_ENABLE_LLM=false;export AGENT_DEFAULT_PROVIDER=mock",
            "failure_reason=unexpected_result_type;sanitized=true",
        )

    if _is_successful_smoke_result(result):
        return 0, build_success_lines(
            request=provider_request,
            result=result,
            smoke_model_name=smoke_model_name,
        )

    return 1, build_failure_lines(
        request=provider_request,
        result=result,
        smoke_model_name=smoke_model_name,
    )


def _missing_execute_fields(args: argparse.Namespace) -> tuple[str, ...]:
    missing = []
    for field_name in ("question", "page_id", "section_id", "component_id"):
        value = getattr(args, field_name, "")
        if not isinstance(value, str) or not value.strip():
            missing.append(field_name)
    return tuple(missing)


def _resolve_smoke_model_name(model_name: str) -> str | None:
    candidate = model_name.strip()
    if not candidate:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate):
        return None
    return candidate


def build_invalid_model_lines() -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=BLOCKED",
        "reason=invalid_smoke_model_name",
        "smoke_model_valid=false",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
    )


def _minimal_smoke_visible_context() -> dict[str, object]:
    return {
        "page_title": "Image Inspection",
        "page_summary": "Local smoke context for the image inspection explanation panel.",
        "section_summary": "Manual review is required before taking action.",
        "visible_summary": "This synthetic context keeps the smoke local-only and minimal.",
    }


def _minimal_smoke_inspection_response() -> dict[str, object]:
    return {
        "decision": {
            "final_decision": "manual_review_required",
            "decision_level": "review",
            "rule_id": "local_smoke_manual_review_rule",
            "recommended_action": "Review the inspection evidence before taking action.",
        },
        "classification": {
            "predicted_label": "surface_defect_candidate",
            "probability_defect": 0.72,
            "threshold": 0.50,
        },
        "detection": {
            "predicted_box_count": 1,
            "defect_count": 1,
        },
        "anomaly": {
            "anomaly_score": 0.21,
            "threshold": 0.20,
        },
        "traceability": {
            "source_endpoint": "local_smoke_synthetic_context",
            "contract_version": "local_smoke_synthetic_context_v1",
        },
        "warnings": ["local smoke only"],
        "limitations": ["manual review still applies"],
        "request_id": "local-smoke-request",
        "explanation_context": {
            "context_version": "local_smoke_synthetic_context_v1",
        },
    }


def _smoke_question_with_evidence_only_instructions(question: str) -> str:
    clean_question = question.strip()
    return (
        "Use only the provided evidence. "
        "Do not invent metrics. "
        "Do not convert numeric values to other formats. "
        "Do not add percentages unless the exact percentage exists in the evidence. "
        "Do not claim production readiness, deployment readiness, HTTPS/domain readiness, or real LLM readiness. "
        "Keep the answer short. "
        "If evidence is insufficient, say manual review is required. "
        f"User question: {clean_question}"
    )


def _resolve_gemini_api_key() -> str | None:
    value = os.getenv("GEMINI_API_KEY")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _gemini_api_key_present() -> bool:
    return _resolve_gemini_api_key() is not None


def _load_gemini_sdk_readiness_result() -> GeminiSdkLoadResult:
    try:
        sdk_module = _load_google_genai_module()
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        return GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="load_error",
            reason="google.genai import failed during local smoke readiness.",
            error_category=type(exc).__name__,
            sdk_name="google-genai",
            import_style="from google import genai",
        )

    has_client = hasattr(sdk_module, "Client")
    if has_client:
        return GeminiSdkLoadResult(
            checked=True,
            sdk_available=True,
            status="available",
            reason="google.genai import succeeded locally.",
            sdk_name="google-genai",
            import_style="from google import genai",
        )

    return GeminiSdkLoadResult(
        checked=True,
        sdk_available=False,
        status="missing",
        reason="google.genai imported but did not expose the expected Client factory.",
        error_category="missing_client",
        sdk_name="google-genai",
        import_style="from google import genai",
    )


def _allowed_evidence_values(grounding_context: AgentGroundingContext) -> tuple[object, ...]:
    values: list[object] = []
    for item in grounding_context.evidence_used:
        if isinstance(item, dict) and "value" in item:
            values.append(item["value"])
    return tuple(values)


def _request_summary_line(request: AgentProviderRequest) -> str:
    component_id = request.component_id or "none"
    return (
        "request_summary="
        f"page_id={request.page_id};"
        f"section_id={request.section_id};"
        f"component_id={component_id};"
        "question_sanitized=true"
    )


def _response_summary_line(result: GeminiRealGenerationResult) -> str:
    return (
        "response_summary="
        f"manual_review_visible=true;"
        f"sanitized=true;"
        "raw_response_hidden=true"
    )


def _safe_error_category(result: GeminiRealGenerationResult) -> str:
    status = (result.status or "").strip().lower()
    response = result.provider_response
    if status == "sdk_missing":
        return "sdk_missing"
    if status == "load_error":
        return "sdk_load_error"
    if status == "provider_error":
        return "provider_error"
    if status == "timeout":
        return "timeout"
    if status == "rate_limit":
        return "rate_limited"
    if status == "empty":
        return "empty_response"
    if status == "malformed":
        return "malformed_response"
    if response.safety_status == "blocked":
        return "safety_blocked"
    if response.fallback_used and response.provider_used == "mock":
        return "unavailable"
    return "unknown"


def _safe_safety_diagnostic_lines(result: GeminiRealGenerationResult) -> tuple[str, ...]:
    if _safe_error_category(result) != "safety_blocked" and (result.status or "").strip().lower() != "blocked":
        return ()
    return (
        f"safety_stage={_safe_safety_stage(result)}",
        f"safety_block_reason={_safe_safety_block_reason(result)}",
    )


def _safe_provider_error_diagnostic_lines(result: GeminiRealGenerationResult) -> tuple[str, ...]:
    if _safe_error_category(result) != "provider_error" and (result.status or "").strip().lower() != "provider_error":
        return ()
    return (
        f"provider_error_stage={_safe_provider_error_stage(result)}",
        f"provider_error_reason={_safe_provider_error_reason(result)}",
    )


def _safe_safety_stage(result: GeminiRealGenerationResult) -> str:
    text = _combined_safety_text(result)
    if "prompt was blocked" in text:
        return "pre_generation"
    if "output was blocked" in text:
        return "post_generation"
    return "unknown"


def _safe_safety_block_reason(result: GeminiRealGenerationResult) -> str:
    text = _combined_safety_text(result)
    if "prompt was blocked" in text:
        return "evidence_boundary_violation"
    if "output was blocked" in text:
        return "safety_guard_blocked"
    if "metric" in text or "threshold" in text or "probability" in text:
        return "invented_metric_like_output"
    if "readiness" in text or "production" in text or "deployment" in text:
        return "unsupported_readiness_claim"
    if "secret" in text or "path" in text:
        return "secret_or_path_leak"
    return "unknown"


def _safe_provider_error_stage(result: GeminiRealGenerationResult) -> str:
    text = _combined_provider_error_text(result)
    if "client creation" in text:
        return "client_creation"
    if "normalized" in text or "malformed payload" in text:
        return "response_normalization"
    if "unexpected error" in text or "provider error" in text or "invoked" in text:
        return "client_invocation"
    return "unknown"


def _safe_provider_error_reason(result: GeminiRealGenerationResult) -> str:
    text = _combined_provider_error_text(result)
    if "client creation" in text:
        return "client_creation_failed"
    if "normalized" in text or "malformed payload" in text:
        return "normalized_provider_error"
    if "unexpected error" in text:
        return "invoke_raised_unexpected_error"
    if "provider error" in text or "invoked" in text:
        return "invoke_raised_provider_error"
    return "unknown"


def _combined_safety_text(result: GeminiRealGenerationResult) -> str:
    return " ".join(
        str(value).lower()
        for value in (
            result.provider_error,
            result.fallback_reason,
            result.provider_response.provider_error,
            result.provider_response.fallback_reason,
            result.provider_response.answer,
        )
        if value
    )


def _combined_provider_error_text(result: GeminiRealGenerationResult) -> str:
    return " ".join(
        str(value).lower()
        for value in (
            result.provider_error,
            result.fallback_reason,
            result.provider_response.provider_error,
            result.provider_response.fallback_reason,
        )
        if value
    )


def _is_successful_smoke_result(result: GeminiRealGenerationResult) -> bool:
    response = result.provider_response
    return (
        response.provider_used == "gemini"
        and not response.fallback_used
        and result.safe_to_send
        and result.safe_to_display
        and result.status in {"pass", "success"}
    )


if __name__ == "__main__":
    raise SystemExit(main())

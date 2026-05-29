"""Provider contract and readiness helpers for the Agent/RAG MVP.

This module is intentionally offline-only. It defines typed request/response
containers and readiness checks that future providers can reuse without
calling any external SDK or network service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .context_builder import AgentGroundingContext


ProviderStatus = Literal["available", "disabled", "unavailable"]
ProviderName = Literal["mock", "gemini", "grok", "openai"]


@dataclass(frozen=True, slots=True)
class ProviderRuntimeSettings:
    """Provider-related runtime flags stripped of secret material."""

    enable_llm: bool = False
    enable_real_provider_runtime: bool = False
    default_provider: str = "mock"
    provider_order: tuple[str, ...] = ("mock", "gemini", "grok")
    enable_fallback: bool = True
    gemini_key_present: bool = False
    grok_key_present: bool = False
    openai_key_present: bool = False


@dataclass(frozen=True, slots=True)
class ProviderFallbackPolicy:
    """Fallback policy used by the mock-first MVP."""

    allow_mock_fallback: bool = True
    fallback_provider: str = "mock"
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentProviderStatus:
    """Availability snapshot for one provider."""

    provider_name: ProviderName
    configured: bool
    available: bool
    status: ProviderStatus
    reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


ProviderAvailability = AgentProviderStatus


@dataclass(frozen=True, slots=True)
class ProviderReadinessResult:
    """Provider readiness state and fallback policy."""

    provider_name: ProviderName
    availability: AgentProviderStatus
    fallback_policy: ProviderFallbackPolicy
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AgentProviderRequest:
    """Provider-layer request payload for a future explanation call."""

    provider_name: ProviderName
    question: str
    grounding_context: dict[str, Any]
    sanitized_context: dict[str, Any]
    safety_status: str
    llm_enabled: bool
    page_id: str
    section_id: str
    component_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentProviderResponse:
    """Provider-layer response contract for a future explanation call."""

    answer: str
    provider_used: ProviderName
    fallback_used: bool
    fallback_reason: str | None = None
    provider_error_stage: str | None = None
    provider_error_reason: str | None = None
    safety_block_reason: str | None = None
    grounding_status: str = "insufficient_evidence"
    safety_status: str = "pass"
    limitations: tuple[str, ...] = field(default_factory=tuple)
    evidence_used: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    provider_error: str | None = None
    raw_provider_response_allowed: bool = False


def build_provider_request(
    *,
    provider_name: ProviderName,
    grounding_context: AgentGroundingContext,
    sanitized_context: dict[str, Any],
    safety_status: str,
    llm_enabled: bool,
) -> AgentProviderRequest:
    """Build a provider request without exposing raw secrets."""
    grounding_payload = dict(sanitized_context)
    sanitized_question = grounding_payload.get("question")
    if isinstance(sanitized_question, str) and sanitized_question.strip():
        question = sanitized_question
    else:
        question = grounding_context.question
    grounding_payload.setdefault("page_id", grounding_context.page_id)
    grounding_payload.setdefault("section_id", grounding_context.section_id)
    grounding_payload.setdefault("component_id", grounding_context.component_id)
    grounding_payload["question"] = question
    grounding_payload.setdefault("grounding_status", grounding_context.grounding_status)
    return AgentProviderRequest(
        provider_name=provider_name,
        question=question,
        grounding_context=grounding_payload,
        sanitized_context=sanitized_context,
        safety_status=safety_status,
        llm_enabled=llm_enabled,
        page_id=grounding_context.page_id,
        section_id=grounding_context.section_id,
        component_id=grounding_context.component_id,
    )


def build_provider_response(
    *,
    answer: str,
    provider_used: ProviderName,
    fallback_used: bool,
    fallback_reason: str | None,
    provider_error_stage: str | None = None,
    provider_error_reason: str | None = None,
    safety_block_reason: str | None = None,
    grounding_status: str,
    safety_status: str,
    limitations: list[str],
    evidence_used: list[dict[str, Any]],
    provider_error: str | None = None,
    raw_provider_response_allowed: bool = False,
) -> AgentProviderResponse:
    """Build a provider response contract with safe defaults."""
    deduped_evidence: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in evidence_used:
        source = str(item.get("source", ""))
        if source in seen_sources:
            continue
        seen_sources.add(source)
        deduped_evidence.append(dict(item))
    return AgentProviderResponse(
        answer=answer,
        provider_used=provider_used,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        provider_error_stage=provider_error_stage,
        provider_error_reason=provider_error_reason,
        safety_block_reason=safety_block_reason,
        grounding_status=grounding_status,
        safety_status=safety_status,
        limitations=tuple(dict.fromkeys(limitations)),
        evidence_used=tuple(deduped_evidence),
        provider_error=provider_error,
        raw_provider_response_allowed=raw_provider_response_allowed,
    )


def evaluate_provider_readiness(settings: ProviderRuntimeSettings) -> dict[ProviderName, ProviderReadinessResult]:
    """Return provider readiness results without contacting any provider."""
    results: dict[ProviderName, ProviderReadinessResult] = {}

    results["mock"] = ProviderReadinessResult(
        provider_name="mock",
        availability=AgentProviderStatus(
            provider_name="mock",
            configured=True,
            available=True,
            status="available",
            reason="Mock provider is always available in the MVP.",
            warnings=(),
        ),
        fallback_policy=ProviderFallbackPolicy(
            allow_mock_fallback=True,
            fallback_provider="mock",
            fallback_reason="Mock provider is the MVP fallback.",
        ),
        warnings=("Mock fallback is active in the current MVP.",),
    )

    results["gemini"] = _build_future_provider_readiness(
        provider_name="gemini",
        key_present=settings.gemini_key_present,
        settings=settings,
    )
    results["grok"] = _build_future_provider_readiness(
        provider_name="grok",
        key_present=settings.grok_key_present,
        settings=settings,
    )
    results["openai"] = _build_future_provider_readiness(
        provider_name="openai",
        key_present=settings.openai_key_present,
        settings=settings,
    )

    return results


def _build_future_provider_readiness(
    *,
    provider_name: ProviderName,
    key_present: bool,
    settings: ProviderRuntimeSettings,
) -> ProviderReadinessResult:
    if not settings.enable_llm:
        availability = AgentProviderStatus(
            provider_name=provider_name,
            configured=key_present,
            available=False,
            status="disabled",
            reason="LLM execution is disabled in this MVP slice.",
            warnings=(),
        )
        warning = f"{provider_name.title()} is unavailable while AGENT_ENABLE_LLM is disabled."
    elif not key_present:
        availability = AgentProviderStatus(
            provider_name=provider_name,
            configured=False,
            available=False,
            status="unavailable",
            reason=f"{provider_name.title()} API key is not configured.",
            warnings=(),
        )
        warning = f"{provider_name.title()} is unavailable because its API key is missing."
    else:
        availability = AgentProviderStatus(
            provider_name=provider_name,
            configured=True,
            available=False,
            status="disabled",
            reason="Real provider execution is intentionally disabled in this MVP slice.",
            warnings=(),
        )
        warning = f"{provider_name.title()} is configured but still disabled in this MVP slice."

    return ProviderReadinessResult(
        provider_name=provider_name,
        availability=availability,
        fallback_policy=ProviderFallbackPolicy(
            allow_mock_fallback=True,
            fallback_provider="mock",
            fallback_reason=f"{provider_name.title()} is not available; mock fallback remains the safe path.",
        ),
        warnings=(warning,),
    )

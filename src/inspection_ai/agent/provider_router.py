"""Provider routing for the Agent/RAG MVP.

The MVP is mock-first by design. A disabled-by-default real-provider gate is
present for future explicit activation, but the normal runtime remains mock-
first unless that gate is turned on deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .context_builder import AgentGroundingContext, load_global_context
from .gemini_provider import (
    GeminiG3Readiness,
    GeminiRealProviderConfig,
    GeminiSdkLoadResult,
    _default_gemini_api_key_resolver,
    _classify_safety_block_reason,
    evaluate_gemini_g3_readiness,
    evaluate_gemini_provider_readiness,
    generate_with_real_gemini_provider,
    _normalize_allowed_evidence_value,
    _load_google_genai_module,
)
from .provider_contracts import (
    AgentProviderRequest,
    AgentProviderResponse,
    ProviderRuntimeSettings,
    build_provider_request,
    build_provider_response,
    evaluate_provider_readiness,
)
from .safety_guard import guard_post_generation_text, guard_pre_generation_context
from .schemas import AgentExplainResponse, AgentHealthResponse, AgentEvidenceItem


SUPPORTED_PROVIDERS = ("mock", "gemini", "grok")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


@dataclass(frozen=True, slots=True)
class AgentProviderSettings:
    """Environment-driven provider settings for the MVP."""

    enable_llm: bool = False
    enable_real_provider_runtime: bool = False
    default_provider: str = "mock"
    provider_order: tuple[str, ...] = ("mock", "gemini", "grok")
    enable_fallback: bool = True
    timeout_seconds: int = 20
    max_retries: int = 1
    gemini_api_key: str | None = None
    grok_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "AgentProviderSettings":
        default_provider = os.getenv("AGENT_DEFAULT_PROVIDER", "mock").strip().lower() or "mock"
        if default_provider not in SUPPORTED_PROVIDERS:
            default_provider = "mock"
        provider_order = tuple(
            item for item in _env_csv("LLM_PROVIDER_ORDER", ("mock", "gemini", "grok"))
            if item in SUPPORTED_PROVIDERS
        )
        if not provider_order:
            provider_order = ("mock", "gemini", "grok")
        return cls(
            enable_llm=_env_bool("AGENT_ENABLE_LLM", False),
            enable_real_provider_runtime=_env_bool("AGENT_ENABLE_REAL_PROVIDER_RUNTIME", False),
            default_provider=default_provider,
            provider_order=provider_order,
            enable_fallback=_env_bool("LLM_ENABLE_FALLBACK", True),
            timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 20),
            max_retries=_env_int("LLM_MAX_RETRIES", 1),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            grok_api_key=os.getenv("GROK_API_KEY") or None,
        )


@dataclass(frozen=True, slots=True)
class GeminiRouteDecision:
    """Disabled-by-default Gemini routing decision for the MVP router."""

    requested_provider: str
    selected_provider: str
    should_route_to_gemini: bool
    fallback_used: bool
    fallback_reason: str
    reason: str
    llm_enabled: bool
    provider_allowed: bool
    api_key_present: bool
    sdk_checked: bool
    sdk_status: str
    sdk_available: bool
    real_provider_implemented: bool
    activation_allowed: bool
    safety_ready: bool


class AgentProviderRouter:
    """Resolve explanation responses from the safe MVP provider set."""

    def __init__(self, settings: AgentProviderSettings | None = None) -> None:
        self.settings = settings or AgentProviderSettings.from_env()
        self._global_context = load_global_context()

    @property
    def available_providers(self) -> list[str]:
        """Return providers that are actually available in the MVP runtime."""
        readiness = evaluate_provider_readiness(self._provider_runtime_settings())
        return [name for name, result in readiness.items() if result.availability.available and name == "mock"]

    def gemini_readiness(self) -> GeminiG3Readiness:
        """Return the safe Gemini readiness snapshot without activating Gemini."""
        sdk_loader = _load_runtime_gemini_sdk_status if self.settings.enable_real_provider_runtime else None
        return evaluate_gemini_g3_readiness(
            self._provider_runtime_settings(),
            sdk_loader=sdk_loader,
            real_provider_implemented=self.settings.enable_real_provider_runtime,
        )

    def gemini_route_decision(
        self,
        requested_provider: str | None = None,
        *,
        readiness: GeminiG3Readiness | None = None,
        safety_ready: bool = True,
    ) -> GeminiRouteDecision:
        """Return the disabled-by-default Gemini routing decision without executing Gemini."""
        return evaluate_gemini_route_decision(
            self._provider_runtime_settings(),
            readiness or self.gemini_readiness(),
            requested_provider=requested_provider,
            safety_ready=safety_ready,
        )

    def health(self) -> AgentHealthResponse:
        """Return a safe health snapshot for the agent layer."""
        provider_runtime_settings = self._provider_runtime_settings()
        readiness = evaluate_provider_readiness(provider_runtime_settings)
        gemini_readiness = self.gemini_readiness()
        warnings = list(readiness["mock"].warnings)
        if not self.settings.enable_fallback:
            warnings.append(
                "LLM_ENABLE_FALLBACK was disabled, but mock fallback is mandatory in this MVP slice."
            )
        if self.settings.enable_llm:
            warnings.append(
                "AGENT_ENABLE_LLM was requested, but real provider execution is intentionally disabled in this MVP slice."
            )
        warnings.extend(readiness["gemini"].warnings)
        warnings.extend(readiness["grok"].warnings)
        warnings.extend(evaluate_gemini_provider_readiness(provider_runtime_settings).warnings)
        warnings.extend(gemini_readiness.warnings)
        warnings.append(_format_gemini_readiness_warning(gemini_readiness))
        if self.settings.default_provider != "mock":
            warnings.append("Mock remains the default safe provider for this MVP.")
        return AgentHealthResponse(
            status="ok",
            service="industrial-surface-defect-agent",
            agent_ready=True,
            llm_enabled=False,
            default_provider="mock",
            provider_order=list(self.settings.provider_order),
            available_providers=self.available_providers,
            fallback_available=True,
            grounding_ready=True,
            warnings=warnings,
        )

    def explain(self, grounding_context: AgentGroundingContext) -> AgentExplainResponse:
        """Return a grounded mock explanation response."""
        pre_guard = guard_pre_generation_context(grounding_context)
        gemini_readiness = self.gemini_readiness()
        decision = self.gemini_route_decision(readiness=gemini_readiness, safety_ready=not pre_guard.blocked)
        fallback_reason: str | None = decision.fallback_reason
        provider_error_stage: str | None = None
        provider_error_reason: str | None = None
        safety_block_reason: str | None = None
        safety_status = pre_guard.status
        if pre_guard.blocked:
            provider_request = build_provider_request(
                provider_name="mock",
                grounding_context=grounding_context,
                sanitized_context=pre_guard.sanitized_context,
                safety_status=pre_guard.status,
                llm_enabled=self.settings.enable_llm,
            )
            answer = pre_guard.sanitized_text or (
                "I can’t provide that answer because the requested prompt is unsafe under the Agent safety guard. "
                "Manual review still applies."
            )
            grounding_status = "unsupported"
            fallback_reason = "Request blocked by the Agent safety guard; mock fallback remains the safe path."
            provider_error_stage = "pre_generation"
            provider_error_reason = "safety_blocked"
            safety_block_reason = _classify_safety_block_reason(pre_guard.reasons)
        elif _question_requests_forbidden_claim(grounding_context.question, self._global_context):
            provider_request = build_provider_request(
                provider_name="mock",
                grounding_context=grounding_context,
                sanitized_context=pre_guard.sanitized_context,
                safety_status=pre_guard.status,
                llm_enabled=self.settings.enable_llm,
            )
            answer = (
                "I can’t provide a claim about production readiness, deployment safety, autonomous decisions, "
                "replacing human review, or invented evidence. This assistant is limited to grounded explanations "
                "from governed evidence, and manual review still applies."
            )
            grounding_status = "unsupported"
            fallback_reason = "Request blocked by the Agent safety guard; mock fallback remains the safe path."
            provider_error_stage = "pre_generation"
            provider_error_reason = "safety_blocked"
            safety_block_reason = "unsupported_readiness_claim"
        else:
            gemini_response = self._maybe_route_to_gemini(grounding_context, pre_guard)
            if gemini_response is not None:
                return gemini_response

            provider_request = build_provider_request(
                provider_name="mock",
                grounding_context=grounding_context,
                sanitized_context=pre_guard.sanitized_context,
                safety_status=pre_guard.status,
                llm_enabled=self.settings.enable_llm,
            )
            answer, grounding_status = _build_mock_answer(grounding_context)
            post_guard = guard_post_generation_text(answer, grounding_context=grounding_context)
            if post_guard.blocked:
                answer = post_guard.sanitized_text or (
                    "I can’t provide that answer because the generated text is unsafe under the Agent safety guard. "
                    "Manual review still applies."
                )
                grounding_status = "unsupported"
                safety_status = post_guard.status
                provider_error_stage = "post_generation"
                provider_error_reason = "safety_blocked"
                safety_block_reason = _classify_safety_block_reason(post_guard.reasons)
            elif decision.requested_provider == "gemini" and decision.selected_provider == "mock":
                provider_error_stage, provider_error_reason = _gemini_route_fallback_diagnostics(
                    decision,
                    safety_blocked=False,
                    fallback_enabled=self.settings.enable_fallback,
                )

        limitations = list(dict.fromkeys(grounding_context.limitations + [
            "Mock backend Agent is active in the current MVP path.",
            "External LLM provider integration is not active in this MVP slice.",
            "No real LLM provider call is made in the MVP mock path.",
            "Manual review still applies.",
        ]))

        provider_response = build_provider_response(
            answer=answer,
            provider_used="mock",
            fallback_used=True,
            fallback_reason=fallback_reason or "Mock provider is the MVP fallback.",
            provider_error_stage=provider_error_stage,
            provider_error_reason=provider_error_reason,
            safety_block_reason=safety_block_reason,
            grounding_status=grounding_status,
            safety_status=safety_status,
            limitations=limitations,
            evidence_used=provider_request.grounding_context.get("evidence_used", []),
        )

        return _build_agent_explain_response(grounding_context, provider_response)

    def _provider_runtime_settings(self) -> ProviderRuntimeSettings:
        return ProviderRuntimeSettings(
            enable_llm=self.settings.enable_llm,
            enable_real_provider_runtime=self.settings.enable_real_provider_runtime,
            default_provider=self.settings.default_provider,
            provider_order=self.settings.provider_order,
            enable_fallback=self.settings.enable_fallback,
            gemini_key_present=bool(self.settings.gemini_api_key),
            grok_key_present=bool(self.settings.grok_api_key),
        )

    def _maybe_route_to_gemini(
        self,
        grounding_context: AgentGroundingContext,
        pre_guard,
    ) -> AgentExplainResponse | None:
        gemini_readiness = self.gemini_readiness()
        decision = self.gemini_route_decision(readiness=gemini_readiness, safety_ready=not pre_guard.blocked)
        if not decision.should_route_to_gemini:
            return None

        provider_request = build_provider_request(
            provider_name="gemini",
            grounding_context=grounding_context,
            sanitized_context=pre_guard.sanitized_context,
            safety_status=pre_guard.status,
            llm_enabled=self.settings.enable_llm,
        )
        try:
            gemini_result = generate_with_real_gemini_provider(
                provider_request,
                settings=self._provider_runtime_settings(),
                config=GeminiRealProviderConfig(
                    real_provider_implemented=self.settings.enable_real_provider_runtime,
                    sdk_import_allowed=self.settings.enable_real_provider_runtime,
                    fallback_enabled=self.settings.enable_fallback,
                    max_retries=self.settings.max_retries,
                    api_key_resolver=_default_gemini_api_key_resolver,
                ),
                sdk_loader=_load_runtime_gemini_sdk_status,
                sdk_module_loader=_load_google_genai_module,
                allowed_evidence_values=_allowed_evidence_values_from_provider_request(provider_request),
            )
        except Exception:
            answer, grounding_status = _build_mock_answer(grounding_context)
            safety_status = pre_guard.status
            safety_block_reason = None
            post_guard = guard_post_generation_text(answer, grounding_context=grounding_context)
            if post_guard.blocked:
                answer = post_guard.sanitized_text or (
                    "I can’t provide that answer because the generated text is unsafe under the Agent safety guard. "
                    "Manual review still applies."
                )
                grounding_status = "unsupported"
                safety_status = post_guard.status
                provider_error_stage = "post_generation"
                provider_error_reason = "safety_blocked"
                safety_block_reason = _classify_safety_block_reason(post_guard.reasons)
            else:
                provider_error_stage = "client_invocation"
                provider_error_reason = "provider_error"
            provider_response = build_provider_response(
                answer=answer,
                provider_used="mock",
                fallback_used=True,
                fallback_reason="Gemini real provider raised a provider error; mock fallback remains the safe path.",
                provider_error_stage=provider_error_stage,
                provider_error_reason=provider_error_reason,
                safety_block_reason=safety_block_reason,
                grounding_status=grounding_status,
                safety_status=safety_status,
                limitations=list(
                    dict.fromkeys(
                        grounding_context.limitations
                        + [
                            "Mock backend Agent is active in the current MVP path.",
                            "External LLM provider integration is not active in this MVP slice.",
                            "No real LLM provider call is made in the MVP mock path.",
                            "Manual review still applies.",
                        ]
                    )
                ),
                evidence_used=provider_request.grounding_context.get("evidence_used", []),
            )
            return _build_agent_explain_response(grounding_context, provider_response)

        return _build_agent_explain_response(grounding_context, gemini_result.provider_response)


def _question_requests_forbidden_claim(question: str, global_context: dict[str, Any]) -> bool:
    normalized_question = _normalize_text(question)
    forbidden_claims = [_normalize_text(str(item)) for item in global_context.get("forbidden_claims", [])]
    return any(claim in normalized_question for claim in forbidden_claims)


def _build_agent_explain_response(
    grounding_context: AgentGroundingContext,
    provider_response: AgentProviderResponse,
) -> AgentExplainResponse:
    return AgentExplainResponse(
        answer=provider_response.answer,
        evidence_used=[AgentEvidenceItem(**item) for item in provider_response.evidence_used],
        limitations=list(provider_response.limitations),
        provider_used=provider_response.provider_used,
        fallback_used=provider_response.fallback_used,
        fallback_reason=provider_response.fallback_reason,
        provider_error_stage=provider_response.provider_error_stage,
        provider_error_reason=provider_response.provider_error_reason,
        safety_block_reason=provider_response.safety_block_reason,
        grounding_status=provider_response.grounding_status,
        page_id=grounding_context.page_id,  # type: ignore[arg-type]
        section_id=grounding_context.section_id,
        component_id=grounding_context.component_id,
    )


def _allowed_evidence_values_from_provider_request(request: AgentProviderRequest) -> list[Any]:
    allowed_values: list[Any] = []
    evidence_items = request.grounding_context.get("evidence_used", [])
    if not isinstance(evidence_items, list):
        return allowed_values
    for item in evidence_items:
        if isinstance(item, dict):
            allowed_values.append(_normalize_allowed_evidence_value(item.get("value")))
    return allowed_values


def _gemini_route_fallback_diagnostics(
    decision: GeminiRouteDecision,
    *,
    safety_blocked: bool,
    fallback_enabled: bool,
) -> tuple[str | None, str | None]:
    if decision.selected_provider == "gemini" or decision.requested_provider != "gemini":
        return None, None
    if safety_blocked or not decision.safety_ready:
        return "pre_generation", "safety_blocked"
    if not fallback_enabled:
        return "readiness", "readiness"
    if not decision.llm_enabled or not decision.activation_allowed:
        if not decision.api_key_present or not decision.sdk_available:
            return "readiness", "sdk_missing"
        if not decision.real_provider_implemented:
            return "readiness", "readiness"
        if not decision.provider_allowed:
            return "readiness", "provider_error"
        return "readiness", "unknown"
    return "readiness", "unknown"


def _load_runtime_gemini_sdk_status() -> GeminiSdkLoadResult:
    """Check Gemini SDK availability only when the runtime gate is explicitly enabled."""

    try:
        _load_google_genai_module()
    except ModuleNotFoundError:
        return GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="missing",
            reason="The google-genai SDK is not available in this slice.",
            error_category="missing",
        )
    except Exception:
        return GeminiSdkLoadResult(
            checked=True,
            sdk_available=False,
            status="load_error",
            reason="Gemini SDK import failed in this slice.",
            error_category="import_error",
        )

    return GeminiSdkLoadResult(
        checked=True,
        sdk_available=True,
        status="available",
        reason="google-genai SDK import succeeded.",
    )


def evaluate_gemini_route_decision(
    settings: ProviderRuntimeSettings,
    readiness: GeminiG3Readiness,
    *,
    requested_provider: str | None = None,
    safety_ready: bool = True,
) -> GeminiRouteDecision:
    """Evaluate a disabled-by-default Gemini route decision without activating Gemini."""
    requested = (requested_provider or settings.default_provider or "mock").strip().lower() or "mock"
    if requested not in SUPPORTED_PROVIDERS:
        requested = "mock"

    should_route_to_gemini = (
        requested == "gemini"
        and settings.enable_llm
        and settings.enable_fallback
        and settings.gemini_key_present
        and readiness.gates.sdk_available
        and readiness.gates.provider_allowed
        and readiness.gates.real_provider_implemented
        and readiness.gates.activation_allowed
        and safety_ready
    )
    selected_provider = "gemini" if should_route_to_gemini else "mock"

    if requested == "gemini" and not should_route_to_gemini:
        fallback_reason = (
            "Gemini remains disabled by default; mock fallback remains the safe path."
        )
        reason = (
            "Gemini routing is gated off until all runtime and safety gates pass; mock remains the selected provider."
        )
    elif requested == "gemini":
        fallback_reason = None if selected_provider == "gemini" else (
            "Gemini remains disabled by default; mock fallback remains the safe path."
        )
        reason = "Gemini routing is allowed by the disabled-by-default gate."
    else:
        fallback_reason = "Mock remains the default safe provider."
        reason = "Mock remains the default safe provider."

    return GeminiRouteDecision(
        requested_provider=requested,
        selected_provider=selected_provider,
        should_route_to_gemini=should_route_to_gemini,
        fallback_used=selected_provider != requested,
        fallback_reason=fallback_reason or "Mock remains the default safe provider.",
        reason=reason,
        llm_enabled=settings.enable_llm,
        provider_allowed=readiness.gates.provider_allowed,
        api_key_present=settings.gemini_key_present,
        sdk_checked=readiness.gates.sdk_checked,
        sdk_status=readiness.gates.sdk_status,
        sdk_available=readiness.gates.sdk_available,
        real_provider_implemented=readiness.gates.real_provider_implemented,
        activation_allowed=readiness.gates.activation_allowed,
        safety_ready=safety_ready,
    )


def _build_mock_answer(grounding_context: AgentGroundingContext) -> tuple[str, str]:
    if grounding_context.component_id:
        return _build_component_answer(grounding_context), grounding_context.grounding_status

    page_id = grounding_context.page_id
    section_id = grounding_context.section_id
    inspection_response = grounding_context.inspection_response
    visible_context = grounding_context.visible_context

    if page_id == "image_inspection":
        return _build_image_inspection_answer(section_id, inspection_response), grounding_context.grounding_status
    if page_id == "overview":
        return _build_overview_answer(section_id, visible_context), grounding_context.grounding_status
    if page_id == "classification":
        return _build_classification_answer(section_id, visible_context), grounding_context.grounding_status
    if page_id == "anomaly":
        return _build_anomaly_answer(section_id, visible_context), grounding_context.grounding_status
    if page_id == "detection":
        return _build_detection_answer(section_id, visible_context), grounding_context.grounding_status
    if page_id == "safety":
        return _build_safety_answer(section_id, visible_context, grounding_context.limitations), grounding_context.grounding_status
    if page_id == "ai_assistant":
        return _build_ai_assistant_answer(section_id, visible_context), grounding_context.grounding_status
    return "The requested section is supported, but no page-specific explanation template is available yet.", "insufficient_evidence"


def _build_component_answer(grounding_context: AgentGroundingContext) -> str:
    component_id = grounding_context.component_id or "requested_component"
    label = _evidence_value(grounding_context, "component.user_facing_label") or component_id.replace("_", " ")
    component_type = _evidence_value(grounding_context, "component.component_type") or "component"
    evidence_types = _component_evidence_types(grounding_context)
    evidence_text = ", ".join(evidence_types) if evidence_types else "grounded evidence"

    if grounding_context.page_id == "image_inspection":
        return _build_component_image_inspection_answer(grounding_context, str(label), evidence_text)
    if grounding_context.page_id == "classification":
        return _build_component_classification_answer(grounding_context, str(label), str(component_type), evidence_text)
    if grounding_context.page_id == "anomaly":
        return _build_component_anomaly_answer(grounding_context, str(label), str(component_type), evidence_text)
    if grounding_context.page_id == "detection":
        return _build_component_detection_answer(grounding_context, str(label), str(component_type), evidence_text)

    limitation = _first_relevant_limitation(grounding_context)
    return (
        f"{label} is a {component_type} explained from {evidence_text}. "
        "The mock assistant can summarize the allowlisted evidence for this component without using an external LLM. "
        f"{limitation} "
        "Manual review still applies."
    )


def _build_component_image_inspection_answer(
    grounding_context: AgentGroundingContext,
    label: str,
    evidence_text: str,
) -> str:
    decision = _dict_evidence_value(
        grounding_context,
        "inspection_response#decision",
        "inspection_response.decision",
    )
    classification = _dict_evidence_value(
        grounding_context,
        "inspection_response#classification",
        "inspection_response.classification",
    )
    detection = _dict_evidence_value(
        grounding_context,
        "inspection_response#detection",
        "inspection_response.detection",
    )
    anomaly = _dict_evidence_value(
        grounding_context,
        "inspection_response#anomaly",
        "inspection_response.anomaly",
    )

    final_decision = _safe_value(decision, "final_decision")
    rule_id = _safe_value(decision, "rule_id")
    classification_label = _safe_value(classification, "predicted_label")
    box_count = _safe_value(detection, "predicted_box_count")
    anomaly_status = _safe_value(anomaly, "quality_status")
    limitation = _first_relevant_limitation(grounding_context)

    return (
        f"{label} explains the current inspection result from {evidence_text}. "
        f"The final decision is {final_decision}, using rule {rule_id}. "
        f"The grounded signals include classification ({classification_label}), detection box count ({box_count}), "
        f"and anomaly status ({anomaly_status}). "
        f"{limitation} "
        "This is a mock/offline explanation, and manual review still applies."
    )


def _build_component_classification_answer(
    grounding_context: AgentGroundingContext,
    label: str,
    component_type: str,
    evidence_text: str,
) -> str:
    threshold = _first_nonempty_evidence_value(
        grounding_context,
        "recommended_threshold",
        "selected_threshold",
        "baseline_threshold",
    )
    chart_explanation = _first_nonempty_evidence_value(
        grounding_context,
        "chart_explanation",
        "safe_interpretation",
        "safe_summary",
    )
    threshold_text = f" The key threshold value available here is {_format_value(threshold)}." if threshold is not None else ""
    chart_text = f" The governed note says: {_format_value(chart_explanation)}." if chart_explanation else ""
    return (
        f"{label} is a {component_type} grounded in validation evidence from {evidence_text}."
        f"{threshold_text}{chart_text} "
        "Use it to understand validation tradeoffs, not as an operational approval. "
        "Manual review still applies."
    )


def _build_component_anomaly_answer(
    grounding_context: AgentGroundingContext,
    label: str,
    component_type: str,
    evidence_text: str,
) -> str:
    threshold = _first_nonempty_evidence_value(grounding_context, "selected_threshold", "threshold")
    quality_status = _first_nonempty_evidence_value(grounding_context, "quality_status")
    threshold_text = f" The selected threshold evidence is {_format_value(threshold)}." if threshold is not None else ""
    status_text = f" The quality status evidence is {_format_value(quality_status)}." if quality_status else ""
    return (
        f"{label} is a {component_type} grounded in anomaly review evidence from {evidence_text}."
        f"{threshold_text}{status_text} "
        "The anomaly evidence is weak/review-only when that limitation is present, so it should be treated as supporting review evidence. "
        "Manual review still applies."
    )


def _build_component_detection_answer(
    grounding_context: AgentGroundingContext,
    label: str,
    component_type: str,
    evidence_text: str,
) -> str:
    chart_title = _first_nonempty_evidence_value(grounding_context, "chart_title")
    chart_explanation = _first_nonempty_evidence_value(grounding_context, "chart_explanation")
    confidence_counts = _first_nonempty_evidence_value(grounding_context, "confidence_bins.count")
    title_text = f" It corresponds to {_format_value(chart_title)}." if chart_title else ""
    explanation_text = f" The governed note says: {_format_value(chart_explanation)}." if chart_explanation else ""
    counts_text = f" The confidence-bin counts begin with {_format_value(confidence_counts)}." if confidence_counts else ""
    return (
        f"{label} is a {component_type} grounded in YOLO detection evidence from {evidence_text}."
        f"{title_text}{explanation_text}{counts_text} "
        "Confidence scores summarize detection evidence, but they do not replace review. "
        "Manual review still applies."
    )


def _build_image_inspection_answer(section_id: str, inspection_response: dict[str, Any]) -> str:
    decision = inspection_response.get("decision", {})
    classification = inspection_response.get("classification", {})
    detection = inspection_response.get("detection", {})
    anomaly = inspection_response.get("anomaly", {})

    final_decision = _safe_value(decision, "final_decision")
    predicted_label = _safe_value(classification, "predicted_label")
    box_count = _safe_value(detection, "predicted_box_count")
    anomaly_label = _safe_value(anomaly, "predicted_label")
    quality_status = _safe_value(anomaly, "quality_status")

    if section_id == "final_decision":
        return (
            f"The inspection result is {final_decision}. "
            f"The decision is grounded in the classification result ({predicted_label}), "
            f"the detection box count ({box_count}), and the anomaly signal ({anomaly_label}, {quality_status}). "
            "This is a mock explanation and does not replace manual review."
        )
    if section_id == "detection_overlay":
        return (
            "The overlay shows the detection boxes returned by the governed inspection response. "
            "The assistant can point to confidence, box count, and the best detection when they are present."
        )
    if section_id == "unified_results":
        return (
            "The unified result combines classification, defect localization, anomaly detection, and a rule-based decision. "
            "The assistant should explain the visible summaries using those governed outputs."
        )
    if section_id == "detection_box_details":
        return "The full detection box table can be explained using the box IDs, classes, confidences, and bounding boxes returned by the inspection response."
    if section_id == "inspection_warnings":
        return "Warnings should be explained as review guidance. They do not change the governed response values."
    if section_id == "limitations":
        return "Limitations describe the boundaries of the governed inspection response and should be explained plainly."
    if section_id == "traceability":
        return "Traceability connects the inspection result back to the unified endpoint, runs, and evidence sources."
    if section_id == "technical_evidence":
        return "Technical evidence includes run IDs, contract fields, and the raw response for auditability."
    return "The inspection result can be explained using the governed classification, detection, anomaly, decision, traceability, and limitation fields."


def _build_overview_answer(section_id: str, visible_context: dict[str, Any]) -> str:
    if section_id == "review_path":
        return "This page is a guided review path. It helps a user inspect the classification, anomaly, detection, and Image Inspection pages in a sensible order."
    if section_id == "status_summary":
        return "The overview summarizes what is available now and what is still planned. It should not be read as a production claim."
    if section_id == "technical_evidence":
        return "The technical evidence on Overview exists so reviewers can verify the governed bundle sources and page status."
    return _fallback_page_answer("Overview", visible_context)


def _build_classification_answer(section_id: str, visible_context: dict[str, Any]) -> str:
    if section_id == "detailed_metrics":
        return "The classification page shows threshold behavior, confusion matrix evidence, and class-level validation outputs for the governed classifier."
    if section_id == "hero_summary":
        return "The classification page is a governed validation view for the binary defect classifier."
    if section_id == "technical_evidence":
        return "The classification technical evidence keeps the model metadata, run IDs, and governed bundle references available for review."
    return _fallback_page_answer("Surface Defect Classification", visible_context)


def _build_anomaly_answer(section_id: str, visible_context: dict[str, Any]) -> str:
    if section_id == "visual_evidence":
        return "The anomaly page shows governed anomaly score and reconstruction-loss evidence together with threshold behavior."
    if section_id == "sample_evidence_summary":
        return "The sample evidence summary shows governed validation examples and their review-oriented labels."
    if section_id == "technical_evidence":
        return "The anomaly technical evidence preserves the full table details and governed artifact references for review."
    return _fallback_page_answer("Surface Anomaly Detection", visible_context)


def _build_detection_answer(section_id: str, visible_context: dict[str, Any]) -> str:
    if section_id == "visual_evidence":
        return "The detection page shows confidence distribution, class summary, and governed detection boxes for the validation bundle."
    if section_id == "sample_evidence_summary":
        return "The detection sample evidence summary describes the governed validation samples and their prediction outcomes."
    if section_id == "technical_evidence":
        return "The detection technical evidence keeps the full table rows and run lineage available for reviewers."
    return _fallback_page_answer("Defect Detection & Localization", visible_context)


def _build_safety_answer(section_id: str, visible_context: dict[str, Any], limitations: list[str]) -> str:
    if section_id == "boundaries":
        return "This page explains the dashboard’s boundaries: evidence-driven, not production-ready, not deployment-safe, and still requiring manual review."
    if section_id == "manual_review":
        return "Manual review remains required because the dashboard aggregates governed evidence instead of replacing human judgment."
    if section_id == "ai_assistant_state":
        return (
            "The AI assistant is a mock backend Agent surface for selected evidence-grounded explanations. "
            "External LLM provider integration is not active in this MVP slice."
        )
    if section_id == "docker_release":
        return "Docker and release work are later phases and should not be interpreted as completed deployment readiness."
    return _fallback_page_answer("Safety & Limitations", visible_context, limitations=limitations)


def _build_ai_assistant_answer(section_id: str, visible_context: dict[str, Any]) -> str:
    if section_id == "preview_status":
        return (
            "The AI Explanation Assistant currently uses a mock backend Agent for selected evidence-grounded "
            "explanations. Broader external LLM provider integration remains planned for a later phase."
        )
    if section_id == "future_scope":
        return "Future explanations should stay grounded in governed evidence, page summaries, chart summaries, and current inspection results."
    if section_id == "boundaries":
        return "The assistant must remain secondary to the evidence dashboard and never invent or overclaim."
    if section_id == "design_notes":
        return "The design notes describe a future helper for non-technical users, but the current page is only a placeholder."
    return _fallback_page_answer("AI Explanation Assistant", visible_context)


def _fallback_page_answer(page_title: str, visible_context: dict[str, Any], limitations: list[str] | None = None) -> str:
    limitation_text = ""
    if limitations:
        limitation_text = f" Limitation: {limitations[0]}."
    summary = visible_context.get("summary") or visible_context.get("page_summary") or visible_context.get("visible_summary")
    if summary:
        return f"{page_title} summary: {summary}. {limitation_text}".strip()
    return f"{page_title} can be explained using the governed evidence currently visible on the page. {limitation_text}".strip()


def _safe_value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    return "Unavailable" if value in (None, "") else value


def _evidence_value(grounding_context: AgentGroundingContext, source: str) -> Any:
    for item in grounding_context.evidence_used:
        if item.get("source") == source:
            value = item.get("value")
            if isinstance(value, dict) and "value" in value:
                return value.get("value")
            return value
    return None


def _dict_evidence_value(grounding_context: AgentGroundingContext, *sources: str) -> dict[str, Any]:
    for source in sources:
        value = _evidence_value(grounding_context, source)
        if isinstance(value, dict):
            return value
    return {}


def _first_nonempty_evidence_value(grounding_context: AgentGroundingContext, *field_paths: str) -> Any:
    for field_path in field_paths:
        for item in grounding_context.evidence_used:
            source = str(item.get("source", ""))
            value = item.get("value")
            nested_field_path = value.get("field_path") if isinstance(value, dict) else None
            if source.endswith(f"#{field_path}") or nested_field_path == field_path:
                extracted = value.get("value") if isinstance(value, dict) and "value" in value else value
                if extracted not in (None, "", []):
                    return extracted
    return None


def _component_evidence_types(grounding_context: AgentGroundingContext) -> list[str]:
    labels = {
        "governed_file": "governed file evidence",
        "runtime_inspection": "runtime inspection evidence",
        "global_context": "global context evidence",
    }
    evidence_types: list[str] = []
    for item in grounding_context.evidence_used:
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        evidence_type = value.get("evidence_type")
        label = labels.get(str(evidence_type))
        if label and label not in evidence_types:
            evidence_types.append(label)
    return evidence_types


def _first_relevant_limitation(grounding_context: AgentGroundingContext) -> str:
    for limitation in grounding_context.limitations:
        normalized = limitation.lower()
        if any(token in normalized for token in ("manual review", "review-only", "weak", "mock", "external llm")):
            return str(limitation).rstrip(".") + "."
    return "The explanation is limited to allowlisted governed evidence."


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        preview = value[:5]
        suffix = "..." if len(value) > len(preview) else ""
        return f"{preview}{suffix}"
    if isinstance(value, dict):
        keys = list(value)[:5]
        compact = {key: value[key] for key in keys}
        suffix = "..." if len(value) > len(compact) else ""
        return f"{compact}{suffix}"
    return str(value)


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())


def _format_gemini_readiness_warning(readiness: GeminiG3Readiness) -> str:
    return (
        "Gemini readiness: "
        f"status={readiness.status}, "
        f"available={readiness.available}, "
        f"configured={readiness.availability.configured}, "
        f"llm_enabled={readiness.gates.llm_enabled}, "
        f"api_key_present={readiness.gates.api_key_present}, "
        f"sdk_checked={readiness.gates.sdk_checked}, "
        f"sdk_status={readiness.gates.sdk_status}, "
        f"activation_allowed={readiness.gates.activation_allowed}, "
        f"real_provider_implemented={readiness.gates.real_provider_implemented}."
    )

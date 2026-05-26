"""Provider routing for the Agent/RAG MVP.

The MVP is mock-first by design. Real provider execution is intentionally not
enabled in this slice.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .context_builder import AgentGroundingContext, load_global_context
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
            default_provider=default_provider,
            provider_order=provider_order,
            enable_fallback=_env_bool("LLM_ENABLE_FALLBACK", True),
            timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 20),
            max_retries=_env_int("LLM_MAX_RETRIES", 1),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            grok_api_key=os.getenv("GROK_API_KEY") or None,
        )


class AgentProviderRouter:
    """Resolve explanation responses from the safe MVP provider set."""

    def __init__(self, settings: AgentProviderSettings | None = None) -> None:
        self.settings = settings or AgentProviderSettings.from_env()
        self._global_context = load_global_context()

    @property
    def available_providers(self) -> list[str]:
        """Return providers that are actually available in the MVP runtime."""
        return ["mock"]

    def health(self) -> AgentHealthResponse:
        """Return a safe health snapshot for the agent layer."""
        warnings = []
        warnings.append("Mock fallback is active in the current MVP.")
        if not self.settings.enable_fallback:
            warnings.append(
                "LLM_ENABLE_FALLBACK was disabled, but mock fallback is mandatory in this MVP slice."
            )
        if self.settings.enable_llm:
            warnings.append(
                "AGENT_ENABLE_LLM was requested, but real provider execution is intentionally disabled in this MVP slice."
            )
        if not self.settings.gemini_api_key:
            warnings.append("GEMINI_API_KEY is missing; Gemini is unavailable in this MVP slice.")
        if not self.settings.grok_api_key:
            warnings.append("GROK_API_KEY is missing; Grok is unavailable in this MVP slice.")
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
        if _question_requests_forbidden_claim(grounding_context.question, self._global_context):
            answer = (
                "I can’t provide a claim about production readiness, deployment safety, autonomous decisions, "
                "replacing human review, or invented evidence. This assistant is limited to grounded explanations "
                "from governed evidence, and manual review still applies."
            )
            grounding_status = "unsupported"
        else:
            answer, grounding_status = _build_mock_answer(grounding_context)

        evidence_used = [AgentEvidenceItem(**item) for item in grounding_context.evidence_used]
        limitations = list(dict.fromkeys(grounding_context.limitations + [
            "This assistant is planned / not active as an external provider integration.",
            "No backend agent or LLM call is used in the MVP mock path.",
            "Manual review still applies.",
        ]))

        return AgentExplainResponse(
            answer=answer,
            evidence_used=evidence_used,
            limitations=limitations,
            provider_used="mock",
            fallback_used=True,
            grounding_status=grounding_status,
            page_id=grounding_context.page_id,  # type: ignore[arg-type]
            section_id=grounding_context.section_id,
            component_id=grounding_context.component_id,
        )


def _question_requests_forbidden_claim(question: str, global_context: dict[str, Any]) -> bool:
    normalized_question = _normalize_text(question)
    forbidden_claims = [_normalize_text(str(item)) for item in global_context.get("forbidden_claims", [])]
    return any(claim in normalized_question for claim in forbidden_claims)


def _build_mock_answer(grounding_context: AgentGroundingContext) -> tuple[str, str]:
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
        return "The AI assistant is planned / not active. It is a placeholder for future evidence-grounded explanations."
    if section_id == "docker_release":
        return "Docker and release work are later phases and should not be interpreted as completed deployment readiness."
    return _fallback_page_answer("Safety & Limitations", visible_context, limitations=limitations)


def _build_ai_assistant_answer(section_id: str, visible_context: dict[str, Any]) -> str:
    if section_id == "preview_status":
        return "The AI Explanation Assistant is a planned future capability, not an active agent."
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


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").replace("_", " ").split())

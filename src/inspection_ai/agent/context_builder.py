"""Build grounded explanation context for the Agent/RAG MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .component_registry import ComponentRegistryError, ComponentDefinition, get_component_definition
from .evidence_loader import EvidenceLoadResult, load_component_evidence


ROOT_DIR = Path(__file__).resolve().parents[3]
AGENT_CONFIG_DIR = ROOT_DIR / "configs" / "agent"
GLOBAL_CONTEXT_PATH = AGENT_CONFIG_DIR / "global_context.yaml"
SECTION_REGISTRY_PATH = AGENT_CONFIG_DIR / "section_registry.yaml"


@dataclass(slots=True)
class AgentGroundingContext:
    """Structured grounding material for one agent explanation request."""

    page_id: str
    section_id: str
    component_id: str | None
    question: str
    visible_context: dict[str, Any]
    inspection_response: dict[str, Any]
    global_context: dict[str, Any]
    page_definition: dict[str, Any]
    evidence_used: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    safety_boundaries: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    grounding_status: str = "insufficient_evidence"
    raw_evidence_included: bool = False


@lru_cache(maxsize=1)
def load_global_context() -> dict[str, Any]:
    """Load the shared global grounding context."""
    payload = yaml.safe_load(GLOBAL_CONTEXT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Global context must be a mapping: {GLOBAL_CONTEXT_PATH}")
    return payload


@lru_cache(maxsize=1)
def load_section_registry() -> dict[str, Any]:
    """Load the page/section registry for the agent."""
    payload = yaml.safe_load(SECTION_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Section registry must be a mapping: {SECTION_REGISTRY_PATH}")
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        raise ValueError(f"Section registry missing pages mapping: {SECTION_REGISTRY_PATH}")
    return payload


def get_allowed_page_ids() -> list[str]:
    """Return the supported page identifiers."""
    return list(load_section_registry()["pages"].keys())


def get_page_definition(page_id: str) -> dict[str, Any]:
    """Return the registry definition for one page."""
    pages = load_section_registry()["pages"]
    page_definition = pages.get(page_id)
    if not isinstance(page_definition, dict):
        raise ValueError(f"Unsupported page_id: {page_id}")
    return page_definition


def get_allowed_section_ids(page_id: str) -> list[str]:
    """Return supported sections for a page."""
    page_definition = get_page_definition(page_id)
    allowed_sections = page_definition.get("allowed_sections", [])
    if not isinstance(allowed_sections, list):
        raise ValueError(f"Invalid section registry for page_id={page_id!r}")
    return [str(section_id) for section_id in allowed_sections]


def validate_page_section(page_id: str, section_id: str) -> None:
    """Validate that the requested page/section combination is supported."""
    allowed_pages = get_allowed_page_ids()
    if page_id not in allowed_pages:
        raise ValueError(f"Unsupported page_id: {page_id!r}. Allowed values: {allowed_pages}")

    allowed_sections = get_allowed_section_ids(page_id)
    if section_id not in allowed_sections:
        raise ValueError(
            f"Unsupported section_id: {section_id!r} for page_id={page_id!r}. "
            f"Allowed values: {allowed_sections}"
        )


def validate_page_section_component(page_id: str, section_id: str, component_id: str | None) -> None:
    """Validate page/section and optional component identifiers."""
    validate_page_section(page_id, section_id)
    if component_id is None:
        return
    try:
        get_component_definition(page_id, section_id, component_id)
    except ComponentRegistryError as exc:
        raise ValueError(str(exc)) from exc


def build_grounding_context(
    *,
    page_id: str,
    section_id: str,
    component_id: str | None = None,
    question: str,
    visible_context: dict[str, Any] | None,
    inspection_response: dict[str, Any] | None,
    include_raw_evidence: bool,
) -> AgentGroundingContext:
    """Build an evidence-grounded context object for the provider layer."""
    validate_page_section_component(page_id, section_id, component_id)

    global_context = load_global_context()
    page_definition = get_page_definition(page_id)
    visible_context = visible_context or {}
    inspection_response = inspection_response or {}
    component_definition: ComponentDefinition | None = None
    component_evidence_result: EvidenceLoadResult | None = None
    if component_id is not None:
        component_definition = get_component_definition(page_id, section_id, component_id)
        component_evidence_result = load_component_evidence(
            component_definition,
            inspection_response=inspection_response,
            global_context=global_context,
            include_raw_evidence=include_raw_evidence,
        )

    evidence_used = _build_evidence_used(
        page_id=page_id,
        section_id=section_id,
        component_definition=component_definition,
        component_evidence_result=component_evidence_result,
        visible_context=visible_context,
        inspection_response=inspection_response,
        include_raw_evidence=include_raw_evidence and component_definition is None,
    )
    limitations = _build_limitations(
        page_id=page_id,
        inspection_response=inspection_response,
        global_context=global_context,
        component_evidence_result=component_evidence_result,
    )
    grounding_status = _determine_grounding_status(
        page_id=page_id,
        section_id=section_id,
        component_id=component_id,
        component_evidence_result=component_evidence_result,
        evidence_used=evidence_used,
        inspection_response=inspection_response,
    )

    return AgentGroundingContext(
        page_id=page_id,
        section_id=section_id,
        component_id=component_id,
        question=question,
        visible_context=visible_context,
        inspection_response=inspection_response,
        global_context=global_context,
        page_definition=page_definition,
        evidence_used=evidence_used,
        limitations=limitations,
        safety_boundaries=list(global_context.get("safety_boundaries", [])),
        forbidden_claims=list(global_context.get("forbidden_claims", [])),
        grounding_status=grounding_status,
        raw_evidence_included=bool(component_evidence_result.raw_evidence_included)
        if component_evidence_result is not None
        else include_raw_evidence,
    )


def _build_evidence_used(
    *,
    page_id: str,
    section_id: str,
    component_definition: ComponentDefinition | None,
    component_evidence_result: EvidenceLoadResult | None,
    visible_context: dict[str, Any],
    inspection_response: dict[str, Any],
    include_raw_evidence: bool,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for key in ("page_title", "page_summary", "section_summary", "visible_summary"):
        if key in visible_context and visible_context.get(key) not in (None, ""):
            evidence.append({"source": f"visible_context.{key}", "value": visible_context.get(key)})

    if page_id == "image_inspection":
        evidence.extend(_image_inspection_evidence(inspection_response))
    elif page_id in {"classification", "anomaly", "detection"}:
        evidence.extend(_track_page_evidence(page_id, visible_context))
    elif page_id == "overview":
        evidence.extend(_overview_evidence(visible_context))
    elif page_id == "safety":
        evidence.extend(_safety_evidence(visible_context))
    elif page_id == "ai_assistant":
        evidence.extend(_ai_assistant_evidence(visible_context))

    if include_raw_evidence and inspection_response:
        evidence.append(
            {
                "source": "inspection_response.raw",
                "value": inspection_response,
            }
        )

    # Always include the requested page/section in the grounding surface.
    evidence.append({"source": "request.page_id", "value": page_id})
    evidence.append({"source": "request.section_id", "value": section_id})
    if component_definition is not None and component_evidence_result is not None:
        evidence.append({"source": "request.component_id", "value": component_definition.component_id})
        evidence.append({"source": "component.user_facing_label", "value": component_definition.user_facing_label})
        evidence.append({"source": "component.component_type", "value": component_definition.component_type})
        evidence.extend(_component_evidence(component_evidence_result))
    return evidence


def _component_evidence(component_evidence_result: EvidenceLoadResult) -> list[dict[str, Any]]:
    return [
        {
            "source": item.source,
            "value": {
                "field_path": item.field_path,
                "evidence_type": item.evidence_type,
                "component_id": item.component_id,
                "value": item.value,
            },
        }
        for item in component_evidence_result.evidence_items
    ]


def _image_inspection_evidence(inspection_response: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    decision = inspection_response.get("decision", {})
    classification = inspection_response.get("classification", {})
    detection = inspection_response.get("detection", {})
    anomaly = inspection_response.get("anomaly", {})
    traceability = inspection_response.get("traceability", {})
    warnings = inspection_response.get("warnings", [])
    limitations = inspection_response.get("limitations", [])

    if isinstance(decision, dict):
        for key in ("final_decision", "decision_level", "model_agreement_status", "primary_signal", "rule_id", "rule_summary", "recommended_action"):
            value = decision.get(key)
            if value not in (None, "", []):
                evidence.append({"source": f"inspection_response.decision.{key}", "value": value})
        supporting_signals = decision.get("supporting_signals", [])
        if isinstance(supporting_signals, list) and supporting_signals:
            evidence.append({"source": "inspection_response.decision.supporting_signals", "value": supporting_signals})

    if isinstance(classification, dict):
        for key in ("model_name", "model_version", "run_id", "predicted_label", "decision", "threshold", "probability_defect", "probability_good"):
            value = classification.get(key)
            if value not in (None, "", []):
                evidence.append({"source": f"inspection_response.classification.{key}", "value": value})

    if isinstance(detection, dict):
        for key in ("model_name", "model_version", "run_id", "predicted_box_count", "defect_count", "review_status", "best_detection"):
            value = detection.get(key)
            if value not in (None, "", []):
                evidence.append({"source": f"inspection_response.detection.{key}", "value": value})

    if isinstance(anomaly, dict):
        for key in ("model_name", "model_version", "run_id", "anomaly_score", "reconstruction_loss", "threshold", "predicted_label", "decision", "quality_status"):
            value = anomaly.get(key)
            if value not in (None, "", []):
                evidence.append({"source": f"inspection_response.anomaly.{key}", "value": value})

    if isinstance(traceability, dict):
        source_endpoint = traceability.get("source_endpoint")
        if source_endpoint:
            evidence.append({"source": "inspection_response.traceability.source_endpoint", "value": source_endpoint})
        contract_version = traceability.get("contract_version")
        if contract_version:
            evidence.append({"source": "inspection_response.traceability.contract_version", "value": contract_version})

    if isinstance(warnings, list) and warnings:
        evidence.append({"source": "inspection_response.warnings", "value": warnings})
    if isinstance(limitations, list) and limitations:
        evidence.append({"source": "inspection_response.limitations", "value": limitations})

    return evidence


def _track_page_evidence(page_id: str, visible_context: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for key, source in (
        ("model_name", f"{page_id}.model_name"),
        ("model_version", f"{page_id}.model_version"),
        ("run_id", f"{page_id}.run_id"),
        ("summary", f"{page_id}.summary"),
        ("threshold", f"{page_id}.threshold"),
        ("decision", f"{page_id}.decision"),
    ):
        value = visible_context.get(key)
        if value not in (None, "", []):
            evidence.append({"source": source, "value": value})
    return evidence


def _overview_evidence(visible_context: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for key in ("status_summary", "review_path", "page_summary"):
        value = visible_context.get(key)
        if value not in (None, "", []):
            evidence.append({"source": f"overview.{key}", "value": value})
    return evidence


def _safety_evidence(visible_context: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for key in ("boundaries", "manual_review", "ai_assistant_state", "docker_release"):
        value = visible_context.get(key)
        if value not in (None, "", []):
            evidence.append({"source": f"safety.{key}", "value": value})
    return evidence


def _ai_assistant_evidence(visible_context: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for key in ("preview_status", "future_scope", "boundaries", "design_notes"):
        value = visible_context.get(key)
        if value not in (None, "", []):
            evidence.append({"source": f"ai_assistant.{key}", "value": value})
    return evidence


def _build_limitations(
    *,
    page_id: str,
    inspection_response: dict[str, Any],
    global_context: dict[str, Any],
    component_evidence_result: EvidenceLoadResult | None = None,
) -> list[str]:
    limitations = list(global_context.get("safety_boundaries", []))
    if page_id == "image_inspection":
        response_limitations = inspection_response.get("limitations", [])
        if isinstance(response_limitations, list):
            limitations.extend(str(item) for item in response_limitations)
        if not inspection_response.get("warnings"):
            limitations.append("No inspection warnings were returned.")
    if component_evidence_result is not None:
        limitations.extend(component_evidence_result.limitations)
        if component_evidence_result.missing_files:
            limitations.append(
                "Some component evidence files were unavailable: "
                f"{', '.join(component_evidence_result.missing_files[:3])}."
            )
        if component_evidence_result.missing_fields:
            limitations.append(
                "Some allowlisted component evidence fields were unavailable: "
                f"{', '.join(component_evidence_result.missing_fields[:5])}."
            )
    return list(dict.fromkeys(limitations))


def _determine_grounding_status(
    *,
    page_id: str,
    section_id: str,
    component_id: str | None,
    component_evidence_result: EvidenceLoadResult | None,
    evidence_used: list[dict[str, Any]],
    inspection_response: dict[str, Any],
) -> str:
    if component_id is not None:
        if component_evidence_result is not None and component_evidence_result.evidence_items:
            return "grounded"
        return "insufficient_evidence"

    meaningful_evidence = [
        item for item in evidence_used if not str(item.get("source", "")).startswith("request.")
    ]
    if not meaningful_evidence:
        return "insufficient_evidence"
    if page_id == "image_inspection":
        if inspection_response.get("decision") and inspection_response.get("traceability"):
            return "grounded"
        return "partially_grounded"
    if section_id in {"technical_evidence", "traceability"}:
        return "grounded"
    if len(meaningful_evidence) >= 3:
        return "grounded"
    return "partially_grounded"

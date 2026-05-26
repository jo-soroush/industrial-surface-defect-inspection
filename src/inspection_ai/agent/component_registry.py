"""Component registry contract validation for future component-aware Agent/RAG."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_COMPONENT_REGISTRY_PATH = ROOT_DIR / "configs" / "agent" / "component_registry.yaml"

ALLOWED_COMPONENT_TYPES = {
    "card",
    "chart",
    "table",
    "gallery",
    "warning",
    "limitation",
    "decision",
    "text_summary",
    "expander",
    "upload",
    "result_panel",
    "ai_panel",
}
ALLOWED_READINESS_STATUSES = {
    "READY_FOR_COMPONENT_RAG",
    "PARTIAL_EVIDENCE",
    "MISSING_EVIDENCE",
    "SHOULD_NOT_EXPLAIN_YET",
    "RUNTIME_ONLY",
}
ALLOWED_EXPLANATION_PRIORITIES = {"high", "medium", "low"}
REQUIRED_FIELDS = {
    "page_id",
    "section_id",
    "component_id",
    "user_facing_label",
    "component_type",
    "evidence_files",
    "allowed_fields",
    "traceability_fields",
    "raw_allowed",
    "safe_explanation_scope",
    "limitations",
    "readiness_status",
    "fallback_message",
    "explanation_priority",
}
SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
FORBIDDEN_PATH_MARKERS = (".env", "secret", "secrets", "credential", "credentials")


class ComponentRegistryError(ValueError):
    """Raised when the component registry contract is invalid."""


@dataclass(frozen=True, slots=True)
class ComponentDefinition:
    """One validated dashboard component contract entry."""

    page_id: str
    section_id: str
    component_id: str
    user_facing_label: str
    component_type: str
    evidence_files: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    traceability_fields: tuple[str, ...]
    raw_allowed: bool
    safe_explanation_scope: str
    limitations: tuple[str, ...]
    readiness_status: str
    fallback_message: str
    explanation_priority: str


def load_component_registry(path: Path | None = None) -> list[ComponentDefinition]:
    """Load and validate the component registry."""
    registry_path = path or DEFAULT_COMPONENT_REGISTRY_PATH
    payload = _load_registry_payload(registry_path)
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ComponentRegistryError("Component registry must contain a non-empty 'components' list.")

    definitions = [_parse_component(item, index=index) for index, item in enumerate(components, start=1)]
    _validate_unique_components(definitions)
    return definitions


@lru_cache(maxsize=1)
def _load_default_component_registry() -> tuple[ComponentDefinition, ...]:
    return tuple(load_component_registry(DEFAULT_COMPONENT_REGISTRY_PATH))


def get_component_definition(
    page_id: str,
    section_id: str,
    component_id: str,
) -> ComponentDefinition:
    """Return one component definition from the default registry."""
    for component in _load_default_component_registry():
        if (
            component.page_id == page_id
            and component.section_id == section_id
            and component.component_id == component_id
        ):
            return component
    raise ComponentRegistryError(
        "Unknown component definition: "
        f"page_id={page_id!r}, section_id={section_id!r}, component_id={component_id!r}"
    )


def get_components_for_page(page_id: str) -> list[ComponentDefinition]:
    """Return all component definitions for a page from the default registry."""
    return [component for component in _load_default_component_registry() if component.page_id == page_id]


def _load_registry_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ComponentRegistryError(f"Component registry file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ComponentRegistryError(f"Component registry YAML is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ComponentRegistryError("Component registry top-level structure must be a mapping.")
    return payload


def _parse_component(item: Any, *, index: int) -> ComponentDefinition:
    if not isinstance(item, dict):
        raise ComponentRegistryError(f"Component entry {index} must be a mapping.")

    missing_fields = sorted(REQUIRED_FIELDS - set(item))
    if missing_fields:
        raise ComponentRegistryError(f"Component entry {index} is missing fields: {missing_fields}")

    page_id = _require_nonempty_string(item["page_id"], f"components[{index}].page_id")
    section_id = _require_nonempty_string(item["section_id"], f"components[{index}].section_id")
    component_id = _require_nonempty_string(item["component_id"], f"components[{index}].component_id")
    if not SNAKE_CASE_PATTERN.match(component_id):
        raise ComponentRegistryError(f"component_id must be stable snake_case: {component_id!r}")

    user_facing_label = _require_nonempty_string(
        item["user_facing_label"],
        f"components[{index}].user_facing_label",
    )
    component_type = _require_enum(
        item["component_type"],
        ALLOWED_COMPONENT_TYPES,
        f"components[{index}].component_type",
    )
    evidence_files = _require_string_list(item["evidence_files"], f"components[{index}].evidence_files")
    allowed_fields = _require_string_list(item["allowed_fields"], f"components[{index}].allowed_fields")
    traceability_fields = _require_string_list(
        item["traceability_fields"],
        f"components[{index}].traceability_fields",
    )
    raw_allowed = item["raw_allowed"]
    if not isinstance(raw_allowed, bool):
        raise ComponentRegistryError(f"components[{index}].raw_allowed must be a boolean.")
    safe_explanation_scope = _require_nonempty_string(
        item["safe_explanation_scope"],
        f"components[{index}].safe_explanation_scope",
    )
    limitations = _require_string_list(item["limitations"], f"components[{index}].limitations")
    readiness_status = _require_enum(
        item["readiness_status"],
        ALLOWED_READINESS_STATUSES,
        f"components[{index}].readiness_status",
    )
    fallback_message = _require_nonempty_string(
        item["fallback_message"],
        f"components[{index}].fallback_message",
    )
    explanation_priority = _require_enum(
        item["explanation_priority"],
        ALLOWED_EXPLANATION_PRIORITIES,
        f"components[{index}].explanation_priority",
    )

    _validate_evidence_file_paths(evidence_files, index=index)
    _validate_readiness_contract(
        readiness_status=readiness_status,
        evidence_files=evidence_files,
        allowed_fields=allowed_fields,
        index=index,
    )

    return ComponentDefinition(
        page_id=page_id,
        section_id=section_id,
        component_id=component_id,
        user_facing_label=user_facing_label,
        component_type=component_type,
        evidence_files=tuple(evidence_files),
        allowed_fields=tuple(allowed_fields),
        traceability_fields=tuple(traceability_fields),
        raw_allowed=raw_allowed,
        safe_explanation_scope=safe_explanation_scope,
        limitations=tuple(limitations),
        readiness_status=readiness_status,
        fallback_message=fallback_message,
        explanation_priority=explanation_priority,
    )


def _validate_unique_components(definitions: list[ComponentDefinition]) -> None:
    component_ids: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    for definition in definitions:
        if definition.component_id in component_ids:
            raise ComponentRegistryError(f"Duplicate component_id: {definition.component_id}")
        component_ids.add(definition.component_id)

        triple = (definition.page_id, definition.section_id, definition.component_id)
        if triple in triples:
            raise ComponentRegistryError(f"Duplicate page/section/component triple: {triple}")
        triples.add(triple)


def _validate_evidence_file_paths(paths: list[str], *, index: int) -> None:
    for path_value in paths:
        path = Path(path_value)
        if path.is_absolute():
            raise ComponentRegistryError(
                f"components[{index}].evidence_files must use repo-relative paths: {path_value!r}"
            )
        if any(part == ".." for part in path.parts):
            raise ComponentRegistryError(
                f"components[{index}].evidence_files must not escape the repo: {path_value!r}"
            )
        lowered = path_value.lower()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise ComponentRegistryError(
                f"components[{index}].evidence_files contains a forbidden secret-like path: {path_value!r}"
            )


def _validate_readiness_contract(
    *,
    readiness_status: str,
    evidence_files: list[str],
    allowed_fields: list[str],
    index: int,
) -> None:
    if readiness_status == "RUNTIME_ONLY":
        if not allowed_fields:
            raise ComponentRegistryError(
                f"components[{index}] RUNTIME_ONLY entries must define allowed_fields."
            )
        return

    if readiness_status == "READY_FOR_COMPONENT_RAG":
        if not evidence_files:
            raise ComponentRegistryError(
                f"components[{index}] READY_FOR_COMPONENT_RAG entries must define evidence_files."
            )
        if not allowed_fields:
            raise ComponentRegistryError(
                f"components[{index}] READY_FOR_COMPONENT_RAG entries must define allowed_fields."
            )


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComponentRegistryError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_enum(value: Any, allowed_values: set[str], field_name: str) -> str:
    text = _require_nonempty_string(value, field_name)
    if text not in allowed_values:
        raise ComponentRegistryError(
            f"{field_name} must be one of {sorted(allowed_values)}. Received: {text!r}"
        )
    return text


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ComponentRegistryError(f"{field_name} must be a list.")
    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_require_nonempty_string(item, f"{field_name}[{index}]"))
    return parsed

"""Allowlisted evidence loading for future component-aware Agent/RAG."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .component_registry import ComponentDefinition, ROOT_DIR


MAX_LIST_ITEMS = 10
MAX_DICT_ITEMS = 20
MISSING = object()


class EvidenceLoaderError(ValueError):
    """Raised when evidence loading cannot proceed because of invalid inputs."""


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One compact, traceable evidence value selected for a dashboard component."""

    source: str
    field_path: str
    value: Any
    evidence_type: str
    component_id: str


@dataclass(frozen=True, slots=True)
class EvidenceLoadResult:
    """Result of loading component-scoped evidence."""

    evidence_items: list[EvidenceItem]
    limitations: list[str]
    missing_fields: list[str]
    missing_files: list[str]
    raw_evidence_included: bool


def load_component_evidence(
    component: ComponentDefinition,
    *,
    repo_root: Path | None = None,
    inspection_response: dict[str, Any] | None = None,
    global_context: dict[str, Any] | None = None,
    include_raw_evidence: bool = False,
) -> EvidenceLoadResult:
    """Load compact, allowlisted evidence for one component definition.

    This module is deliberately not wired into the Agent runtime yet. It only
    prepares traceable values that later RAG steps can consume.
    """
    root = repo_root or ROOT_DIR
    limitations = list(component.limitations)
    evidence_items: list[EvidenceItem] = []
    missing_fields: list[str] = []
    missing_files: list[str] = []

    if component.readiness_status == "RUNTIME_ONLY":
        _load_runtime_inspection_evidence(
            component=component,
            inspection_response=inspection_response,
            evidence_items=evidence_items,
            missing_fields=missing_fields,
            limitations=limitations,
        )
    else:
        _load_static_or_partial_evidence(
            component=component,
            repo_root=root,
            global_context=global_context,
            evidence_items=evidence_items,
            missing_fields=missing_fields,
            missing_files=missing_files,
            limitations=limitations,
        )

    raw_evidence_included = False
    if include_raw_evidence:
        if component.raw_allowed:
            limitations.append("Raw evidence inclusion is not implemented for this offline loader slice.")
        else:
            limitations.append(f"Raw evidence is disabled for component {component.component_id}.")

    return EvidenceLoadResult(
        evidence_items=evidence_items,
        limitations=_dedupe_strings(limitations),
        missing_fields=_dedupe_strings(missing_fields),
        missing_files=_dedupe_strings(missing_files),
        raw_evidence_included=raw_evidence_included,
    )


def _load_static_or_partial_evidence(
    *,
    component: ComponentDefinition,
    repo_root: Path,
    global_context: dict[str, Any] | None,
    evidence_items: list[EvidenceItem],
    missing_fields: list[str],
    missing_files: list[str],
    limitations: list[str],
) -> None:
    if component.evidence_files:
        for evidence_file in component.evidence_files:
            resolved = _resolve_evidence_file(repo_root, evidence_file)
            if resolved is None:
                missing_files.append(evidence_file)
                limitations.append(f"Evidence file unavailable: {evidence_file}.")
                continue

            source_path, payload_path = resolved
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                limitations.append(f"Evidence file could not be parsed: {source_path} ({exc.__class__.__name__}).")
                continue

            for field_path in [*component.allowed_fields, *component.traceability_fields]:
                value = _extract_field(payload, field_path)
                if value is MISSING:
                    missing_fields.append(f"{source_path}#{field_path}")
                    continue
                evidence_items.append(
                    EvidenceItem(
                        source=f"{source_path}#{field_path}",
                        field_path=field_path,
                        value=_compact_value(value),
                        evidence_type="governed_file",
                        component_id=component.component_id,
                    )
                )

    if _should_load_global_context(component):
        if not global_context:
            limitations.append(f"Global context unavailable for component {component.component_id}.")
            for field_path in [*component.allowed_fields, *component.traceability_fields]:
                if field_path.startswith("global_context."):
                    missing_fields.append(f"global_context#{field_path.removeprefix('global_context.')}")
            return

        for field_path in [*component.allowed_fields, *component.traceability_fields]:
            if not field_path.startswith("global_context."):
                continue
            local_path = field_path.removeprefix("global_context.")
            value = _extract_field(global_context, local_path)
            if value is MISSING:
                missing_fields.append(f"global_context#{local_path}")
                continue
            evidence_items.append(
                EvidenceItem(
                    source=f"global_context#{local_path}",
                    field_path=field_path,
                    value=_compact_value(value),
                    evidence_type="global_context",
                    component_id=component.component_id,
                )
            )

        dashboard_copy_fields = [
            field_path for field_path in component.allowed_fields if field_path.startswith("dashboard_copy.")
        ]
        if dashboard_copy_fields:
            limitations.append("Dashboard copy evidence was not provided to the loader.")
            missing_fields.extend(f"global_context#{field_path}" for field_path in dashboard_copy_fields)


def _load_runtime_inspection_evidence(
    *,
    component: ComponentDefinition,
    inspection_response: dict[str, Any] | None,
    evidence_items: list[EvidenceItem],
    missing_fields: list[str],
    limitations: list[str],
) -> None:
    if not inspection_response:
        limitations.append(f"Inspection response unavailable for component {component.component_id}.")
        missing_fields.extend(component.allowed_fields)
        return

    for field_path in [*component.allowed_fields, *component.traceability_fields]:
        local_path = field_path.removeprefix("inspection_response.")
        value = _extract_field(inspection_response, local_path)
        if value is MISSING:
            missing_fields.append(f"inspection_response#{local_path}")
            continue
        evidence_items.append(
            EvidenceItem(
                source=f"inspection_response#{local_path}",
                field_path=field_path,
                value=_compact_value(value),
                evidence_type="runtime_inspection",
                component_id=component.component_id,
            )
        )


def _resolve_evidence_file(repo_root: Path, evidence_file: str) -> tuple[str, Path] | None:
    direct_path = repo_root / evidence_file
    if direct_path.is_file():
        return evidence_file, direct_path

    fallback_source = str(Path("runtime_assets") / evidence_file)
    fallback_path = repo_root / fallback_source
    if fallback_path.is_file():
        return fallback_source, fallback_path

    return None


def _extract_field(payload: Any, field_path: str) -> Any:
    if not field_path:
        return MISSING
    return _extract_parts(payload, field_path.split("."))


def _extract_parts(value: Any, parts: list[str]) -> Any:
    if not parts:
        return value

    current_part = parts[0]
    remaining_parts = parts[1:]

    if isinstance(value, dict):
        if current_part not in value:
            return MISSING
        return _extract_parts(value[current_part], remaining_parts)

    if isinstance(value, list):
        extracted_values = []
        for item in value:
            extracted = _extract_parts(item, parts)
            if extracted is not MISSING:
                extracted_values.append(extracted)
            if len(extracted_values) >= MAX_LIST_ITEMS:
                break
        if not extracted_values:
            return MISSING
        return extracted_values

    return MISSING


def _compact_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for index, key in enumerate(sorted(value)):
            if index >= MAX_DICT_ITEMS:
                break
            compacted[key] = _compact_value(value[key])
        return compacted
    return value


def _should_load_global_context(component: ComponentDefinition) -> bool:
    fields = [*component.allowed_fields, *component.traceability_fields]
    return any(field_path.startswith(("global_context.", "dashboard_copy.")) for field_path in fields)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped

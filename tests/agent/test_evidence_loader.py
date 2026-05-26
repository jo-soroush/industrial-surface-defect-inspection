"""Tests for component-scoped evidence loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json

from src.inspection_ai.agent.component_registry import get_component_definition
from src.inspection_ai.agent.evidence_loader import (
    MAX_LIST_ITEMS,
    EvidenceItem,
    load_component_evidence,
)


def test_classification_threshold_curve_chart_loads_compact_evidence() -> None:
    component = get_component_definition(
        "classification",
        "detailed_metrics",
        "classification_threshold_curve_chart",
    )

    result = load_component_evidence(component)
    sources = {item.source for item in result.evidence_items}

    assert (
        "artifacts/frontend/track_a/threshold_curve_chart_data.json#recommended_threshold"
        in sources
    )
    assert "artifacts/frontend/track_a/threshold_curve_chart_data.json#rows.threshold" in sources
    rows_item = _find_item(result.evidence_items, "rows.threshold")
    assert isinstance(rows_item.value, list)
    assert 0 < len(rows_item.value) <= MAX_LIST_ITEMS
    assert result.raw_evidence_included is False


def test_detection_confidence_chart_loads_yolo_confidence_evidence() -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )

    result = load_component_evidence(component)
    sources = {item.source for item in result.evidence_items}

    assert (
        "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json#chart_title"
        in sources
    )
    assert (
        "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json#confidence_bins.count"
        in sources
    )
    confidence_counts = _find_item(result.evidence_items, "confidence_bins.count")
    assert isinstance(confidence_counts.value, list)
    assert confidence_counts.evidence_type == "governed_file"


def test_anomaly_threshold_behavior_preserves_review_only_limitation() -> None:
    component = get_component_definition(
        "anomaly",
        "visual_evidence",
        "anomaly_threshold_behavior_chart",
    )

    result = load_component_evidence(component)
    joined_limitations = " ".join(result.limitations).lower()

    assert any(item.field_path == "selected_threshold" for item in result.evidence_items)
    assert "weak" in joined_limitations
    assert "review-only" in joined_limitations


def test_image_inspection_final_decision_card_loads_runtime_evidence() -> None:
    component = get_component_definition(
        "image_inspection",
        "final_decision",
        "image_inspection_final_decision_card",
    )
    inspection_response = {
        "request_id": "req-123",
        "decision": {
            "final_decision": "defective",
            "decision_level": "review",
            "rule_id": "classification_detection_agree_v0",
            "recommended_action": "manual_review",
        },
        "traceability": {"source_endpoint": "/inspect/image"},
    }

    result = load_component_evidence(component, inspection_response=inspection_response)
    sources = {item.source for item in result.evidence_items}

    assert "inspection_response#decision.final_decision" in sources
    assert "inspection_response#decision.rule_id" in sources
    assert "inspection_response#request_id" in sources
    assert _find_item(result.evidence_items, "inspection_response.decision.final_decision").value == "defective"
    assert all(item.evidence_type == "runtime_inspection" for item in result.evidence_items)


def test_runtime_only_component_without_inspection_response_returns_limitation() -> None:
    component = get_component_definition(
        "image_inspection",
        "final_decision",
        "image_inspection_final_decision_card",
    )

    result = load_component_evidence(component)

    assert result.evidence_items == []
    assert any("Inspection response unavailable" in limitation for limitation in result.limitations)
    assert "inspection_response.decision.final_decision" in result.missing_fields


def test_missing_governed_evidence_file_does_not_crash(tmp_path: Path) -> None:
    component = get_component_definition(
        "classification",
        "detailed_metrics",
        "classification_threshold_curve_chart",
    )

    result = load_component_evidence(component, repo_root=tmp_path)

    assert result.evidence_items == []
    assert "artifacts/frontend/track_a/threshold_curve_chart_data.json" in result.missing_files
    assert any("Evidence file unavailable" in limitation for limitation in result.limitations)


def test_missing_allowed_field_does_not_crash(tmp_path: Path) -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )
    _write_json(
        tmp_path / "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json",
        {"chart_title": "Detection confidence distribution"},
    )

    result = load_component_evidence(component, repo_root=tmp_path)

    assert any(item.field_path == "chart_title" for item in result.evidence_items)
    assert (
        "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json#confidence_bins.count"
        in result.missing_fields
    )


def test_list_of_dict_fields_are_compacted_and_limited(tmp_path: Path) -> None:
    component = get_component_definition(
        "classification",
        "detailed_metrics",
        "classification_threshold_curve_chart",
    )
    _write_json(
        tmp_path / "artifacts/frontend/track_a/threshold_curve_chart_data.json",
        {
            "rows": [{"threshold": index / 100} for index in range(MAX_LIST_ITEMS + 5)],
            "recommended_threshold": 0.65,
        },
    )

    result = load_component_evidence(component, repo_root=tmp_path)
    rows_item = _find_item(result.evidence_items, "rows.threshold")

    assert rows_item.value == [index / 100 for index in range(MAX_LIST_ITEMS)]


def test_include_raw_evidence_does_not_include_raw_when_component_disallows_it() -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )

    result = load_component_evidence(component, include_raw_evidence=True)

    assert result.raw_evidence_included is False
    assert any("Raw evidence is disabled" in limitation for limitation in result.limitations)
    assert all(item.field_path != "raw" for item in result.evidence_items)


def test_evidence_item_source_contains_origin_and_field_path() -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )

    result = load_component_evidence(component)

    assert result.evidence_items
    assert all("#" in item.source for item in result.evidence_items)
    assert any(
        item.source.endswith("#confidence_bins.count")
        and item.field_path == "confidence_bins.count"
        for item in result.evidence_items
    )


def test_no_evidence_item_source_is_absolute() -> None:
    component = get_component_definition(
        "classification",
        "detailed_metrics",
        "classification_threshold_curve_chart",
    )

    result = load_component_evidence(component)

    assert result.evidence_items
    assert all(not Path(item.source.split("#", 1)[0]).is_absolute() for item in result.evidence_items)


def test_partial_evidence_uses_global_context_without_fabricating_dashboard_copy() -> None:
    component = get_component_definition("safety", "boundaries", "safety_boundary_cards")
    global_context = {
        "safety_boundaries": ["manual review required"],
        "forbidden_claims": ["production-ready", "deployment-safe"],
    }

    result = load_component_evidence(component, global_context=global_context)
    sources = {item.source for item in result.evidence_items}

    assert "global_context#safety_boundaries" in sources
    assert "global_context#forbidden_claims" in sources
    assert "global_context#dashboard_copy.manual_review_boundary" in result.missing_fields
    assert any("Dashboard copy evidence was not provided" in limitation for limitation in result.limitations)
    assert all(item.evidence_type == "global_context" for item in result.evidence_items)


def test_loader_does_not_modify_component_registry_entry() -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )
    original = replace(component)

    load_component_evidence(component, include_raw_evidence=True)

    assert component == original


def test_invalid_json_evidence_file_produces_limitation_not_exception(tmp_path: Path) -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )
    evidence_path = tmp_path / "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("{invalid json", encoding="utf-8")

    result = load_component_evidence(component, repo_root=tmp_path)

    assert result.evidence_items == []
    assert result.missing_files == []
    assert any("could not be parsed" in limitation for limitation in result.limitations)


def _find_item(evidence_items: list[EvidenceItem], field_path: str) -> EvidenceItem:
    for item in evidence_items:
        if item.field_path == field_path:
            return item
    raise AssertionError(f"Evidence field not found: {field_path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

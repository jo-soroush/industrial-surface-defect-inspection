"""Tests for the Agent component registry contract."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pytest
import yaml

from src.inspection_ai.agent.component_registry import (
    ComponentRegistryError,
    get_component_definition,
    get_components_for_page,
    load_component_registry,
)


SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

REQUIRED_COMPONENT_IDS = {
    "overview_capability_summary",
    "overview_readiness_summary",
    "overview_review_path",
    "classification_metric_cards",
    "classification_readiness_cards",
    "classification_error_distribution_chart",
    "classification_per_class_chart",
    "classification_threshold_curve_chart",
    "classification_model_comparison_table",
    "classification_confusion_matrix_table",
    "classification_safe_interpretation",
    "anomaly_metric_cards",
    "anomaly_pr_auc_summary",
    "anomaly_reconstruction_loss_chart",
    "anomaly_score_summary_chart",
    "anomaly_threshold_behavior_chart",
    "anomaly_sample_summary",
    "anomaly_safe_interpretation",
    "detection_metric_cards",
    "detection_readiness_cards",
    "detection_confidence_chart",
    "detection_class_summary_chart",
    "detection_sample_gallery_summary",
    "detection_artifact_lineage",
    "detection_safe_interpretation",
    "image_inspection_final_decision_card",
    "image_inspection_classification_result_card",
    "image_inspection_detection_result_card",
    "image_inspection_anomaly_result_card",
    "image_inspection_warning_summary",
    "image_inspection_limitations",
    "image_inspection_traceability_context",
    "image_inspection_ai_explanation_panel",
    "safety_boundary_cards",
    "safety_details",
    "ai_assistant_status_cards",
    "ai_assistant_design_notes",
}


def test_registry_loads_successfully() -> None:
    components = load_component_registry()

    assert len(components) == 37


def test_all_component_ids_are_unique() -> None:
    component_ids = [component.component_id for component in load_component_registry()]

    assert len(component_ids) == len(set(component_ids))


def test_all_component_ids_are_snake_case() -> None:
    component_ids = [component.component_id for component in load_component_registry()]

    assert all(SNAKE_CASE_PATTERN.match(component_id) for component_id in component_ids)


def test_required_high_priority_components_exist() -> None:
    component_ids = {component.component_id for component in load_component_registry()}

    assert REQUIRED_COMPONENT_IDS <= component_ids


def test_classification_threshold_curve_chart_is_ready_for_component_rag() -> None:
    component = get_component_definition(
        "classification",
        "detailed_metrics",
        "classification_threshold_curve_chart",
    )

    assert component.readiness_status == "READY_FOR_COMPONENT_RAG"
    assert "artifacts/frontend/track_a/threshold_curve_chart_data.json" in component.evidence_files
    assert "rows.threshold" in component.allowed_fields
    assert "source_threshold_analysis_path" in component.traceability_fields


def test_anomaly_threshold_behavior_chart_includes_review_only_limitation() -> None:
    component = get_component_definition(
        "anomaly",
        "visual_evidence",
        "anomaly_threshold_behavior_chart",
    )

    joined_limitations = " ".join(component.limitations).lower()
    assert component.readiness_status == "READY_FOR_COMPONENT_RAG"
    assert "weak" in joined_limitations
    assert "review-only" in joined_limitations


def test_detection_confidence_chart_references_yolo_confidence_evidence() -> None:
    component = get_component_definition(
        "detection",
        "visual_evidence",
        "detection_confidence_chart",
    )

    assert component.readiness_status == "READY_FOR_COMPONENT_RAG"
    assert "artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json" in (
        component.evidence_files
    )
    assert "confidence_bins.count" in component.allowed_fields


def test_image_inspection_final_decision_card_is_runtime_only() -> None:
    component = get_component_definition(
        "image_inspection",
        "final_decision",
        "image_inspection_final_decision_card",
    )

    assert component.readiness_status == "RUNTIME_ONLY"
    assert component.evidence_files == ()
    assert "inspection_response.decision.final_decision" in component.allowed_fields
    assert "inspection_response.request_id" in component.traceability_fields


def test_raw_evidence_is_disabled_for_all_initial_components() -> None:
    components = load_component_registry()

    assert all(component.raw_allowed is False for component in components)


def test_evidence_file_paths_are_repo_relative() -> None:
    evidence_paths = [
        evidence_path
        for component in load_component_registry()
        for evidence_path in component.evidence_files
    ]

    assert evidence_paths
    assert all(not Path(evidence_path).is_absolute() for evidence_path in evidence_paths)
    assert all(".." not in Path(evidence_path).parts for evidence_path in evidence_paths)


def test_get_component_definition_returns_expected_component() -> None:
    component = get_component_definition(
        "classification",
        "technical_evidence",
        "classification_model_comparison_table",
    )

    assert component.user_facing_label == "Classification model comparison"
    assert component.component_type == "table"


def test_get_component_definition_rejects_unknown_component_safely() -> None:
    with pytest.raises(ComponentRegistryError, match="Unknown component definition"):
        get_component_definition("classification", "detailed_metrics", "missing_component")


def test_get_components_for_page_returns_expected_page_components() -> None:
    components = get_components_for_page("detection")
    component_ids = {component.component_id for component in components}

    assert len(components) == 7
    assert "detection_metric_cards" in component_ids
    assert "detection_confidence_chart" in component_ids
    assert "detection_safe_interpretation" in component_ids


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda component: component.pop("component_id"), "missing fields"),
        (lambda component: component.update(component_id="Bad-ID"), "snake_case"),
        (lambda component: component.update(component_type="widget"), "component_type"),
        (lambda component: component.update(readiness_status="READY_FOR_COMPONENT_RAG", evidence_files=[]), "evidence_files"),
        (lambda component: component.update(readiness_status="RUNTIME_ONLY", allowed_fields=[]), "allowed_fields"),
        (lambda component: component.update(evidence_files=["/tmp/evidence.json"]), "repo-relative"),
        (lambda component: component.update(evidence_files=["../secret.json"]), "escape"),
        (lambda component: component.update(raw_allowed="false"), "boolean"),
        (lambda component: component.update(fallback_message=""), "fallback_message"),
    ],
)
def test_invalid_registry_examples_fail_validation(
    tmp_path: Path,
    mutation: Any,
    expected_message: str,
) -> None:
    payload = _valid_registry_payload()
    mutation(payload["components"][0])
    registry_path = tmp_path / "component_registry.yaml"
    registry_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ComponentRegistryError, match=expected_message):
        load_component_registry(registry_path)


def test_duplicate_component_id_fails_validation(tmp_path: Path) -> None:
    payload = _valid_registry_payload()
    duplicate = dict(payload["components"][0])
    duplicate["page_id"] = "anomaly"
    duplicate["section_id"] = "visual_evidence"
    payload["components"].append(duplicate)
    registry_path = tmp_path / "component_registry.yaml"
    registry_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ComponentRegistryError, match="Duplicate component_id"):
        load_component_registry(registry_path)


def test_duplicate_page_section_component_triple_fails_validation(tmp_path: Path) -> None:
    payload = _valid_registry_payload()
    payload["components"].append(dict(payload["components"][0]))
    registry_path = tmp_path / "component_registry.yaml"
    registry_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ComponentRegistryError, match="Duplicate component_id"):
        load_component_registry(registry_path)


def _valid_registry_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        "components": [
            {
                "page_id": "classification",
                "section_id": "detailed_metrics",
                "component_id": "valid_component",
                "user_facing_label": "Valid component",
                "component_type": "card",
                "evidence_files": ["artifacts/frontend/track_a/metric_cards.json"],
                "allowed_fields": ["cards.title", "cards.value"],
                "traceability_fields": ["run_id"],
                "raw_allowed": False,
                "safe_explanation_scope": "Explain only governed validation evidence.",
                "limitations": ["No production-ready claim."],
                "readiness_status": "READY_FOR_COMPONENT_RAG",
                "fallback_message": "Evidence is unavailable.",
                "explanation_priority": "high",
            }
        ]
    }

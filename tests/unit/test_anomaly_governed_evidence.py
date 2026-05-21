"""Unit tests for posthoc governed anomaly evidence derivation."""

from __future__ import annotations

import json

import pytest

from scripts.evaluation.generate_anomaly_governed_evidence import (
    average_precision,
    build_governed_evidence_inventory,
    build_quality_decision_artifact,
    build_sample_predictions_artifact,
    build_score_distribution_artifact,
    build_threshold_sweep_artifact,
    precision_recall_curve_points,
    validate_evaluation,
)


def test_pr_auc_and_curve_derivation_from_scores() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]

    assert average_precision(labels, scores) == pytest.approx((1.0 + 2 / 3) / 2)

    points = precision_recall_curve_points(labels, scores)
    assert points[0]["threshold"] == 0.9
    assert points[0]["precision"] == 1.0
    assert points[0]["recall"] == 0.5
    assert points[-1] == {"threshold": None, "precision": 1.0, "recall": 0.0}


def test_threshold_sweep_metrics_from_fixture() -> None:
    evaluation = _evaluation_fixture()
    records = validate_evaluation(evaluation)

    artifact = build_threshold_sweep_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
    )

    selected = artifact["selected_threshold_metrics"]
    assert selected["threshold"] == 0.5
    assert selected["tp"] == 1
    assert selected["fp"] == 1
    assert selected["tn"] == 1
    assert selected["fn"] == 1
    assert selected["precision"] == 0.5
    assert selected["recall"] == 0.5
    assert selected["predicted_anomaly_count"] == 2
    assert any(row["threshold"] == 0.5 for row in artifact["rows"])


def test_score_distribution_binning_and_reconstruction_loss_mapping() -> None:
    evaluation = _evaluation_fixture()
    records = validate_evaluation(evaluation)

    artifact = build_score_distribution_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
        bin_count=2,
    )

    assert artifact["summary"]["all"]["count"] == 4
    assert artifact["summary"]["true_normal"]["count"] == 2
    assert artifact["summary"]["true_anomaly"]["count"] == 2
    assert sum(row["count"] for row in artifact["histograms"]["all"]) == 4
    assert "reconstruction_loss is equal to anomaly_score" in artifact["reconstruction_loss_mapping"]


def test_prediction_consistency_validation_catches_mismatch() -> None:
    evaluation = _evaluation_fixture()
    evaluation["samples"][0]["predicted_label_id"] = 0

    with pytest.raises(ValueError, match="does not match score > threshold"):
        validate_evaluation(evaluation)


def test_quality_decision_returns_review_required_for_weak_metrics() -> None:
    evaluation = _evaluation_fixture()
    records = validate_evaluation(evaluation)
    pr_curve = {"pr_auc": 0.42}
    sweep = build_threshold_sweep_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
    )
    distribution = build_score_distribution_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
    )
    predictions = build_sample_predictions_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
    )

    decision = build_quality_decision_artifact(
        evaluation=evaluation,
        pr_curve=pr_curve,
        threshold_sweep=sweep,
        score_distribution=distribution,
        sample_predictions=predictions,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
    )

    assert decision["quality_status"] == "review_required_weak_evidence"
    assert decision["dashboard_usage_recommendation"] == "review_only_signal"
    assert decision["production_ready"] is False
    assert decision["deployment_safe"] is False
    assert any("Recall is very low" in reason for reason in decision["reasons"])


def test_sample_export_maps_anomaly_score_to_reconstruction_loss() -> None:
    evaluation = _evaluation_fixture()
    records = validate_evaluation(evaluation)

    artifact = build_sample_predictions_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc="2026-05-21T00:00:00Z",
    )

    first = artifact["samples"][0]
    assert first["anomaly_score"] == 0.9
    assert first["reconstruction_loss"] == 0.9
    assert first["threshold"] == 0.5
    assert first["mask_path"] == "mask.png"


def test_governed_evidence_inventory_records_hashes_and_policy(tmp_path) -> None:
    evaluation = _evaluation_fixture()
    records = validate_evaluation(evaluation)
    generated_at = "2026-05-21T00:00:00Z"
    pr_curve = {
        "artifact_type": "anomaly_pr_curve",
        "task_type": "anomaly_detection",
        "run_id": "fixture-run",
        "source_artifact_path": "fixture.json",
        "generated_at_utc": generated_at,
    }
    threshold_sweep = build_threshold_sweep_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc=generated_at,
    )
    distribution = build_score_distribution_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc=generated_at,
    )
    predictions = build_sample_predictions_artifact(
        evaluation=evaluation,
        records=records,
        source_artifact_path="fixture.json",
        generated_at_utc=generated_at,
    )
    decision = build_quality_decision_artifact(
        evaluation=evaluation,
        pr_curve={"pr_auc": 0.42},
        threshold_sweep=threshold_sweep,
        score_distribution=distribution,
        sample_predictions=predictions,
        source_artifact_path="fixture.json",
        generated_at_utc=generated_at,
    )
    payloads = [pr_curve, threshold_sweep, distribution, predictions, decision]
    paths = []
    for payload in payloads:
        path = tmp_path / f"{payload['artifact_type']}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    inventory = build_governed_evidence_inventory(
        run_id="fixture-run",
        source_artifact_path="fixture.json",
        generated_at_utc=generated_at,
        artifact_paths=paths,
    )

    assert inventory["inventory_type"] == "anomaly_governed_evidence_inventory"
    assert inventory["generated_from_existing_evidence"] is True
    assert inventory["no_new_inference"] is True
    assert inventory["no_retraining"] is True
    assert inventory["registry_update_deferred"] is True
    assert set(inventory["artifacts"]) == {
        "anomaly_pr_curve",
        "anomaly_threshold_sweep",
        "anomaly_score_distribution",
        "anomaly_sample_predictions",
        "anomaly_quality_decision",
    }
    for entry in inventory["artifacts"].values():
        assert entry["sha256"]
        assert entry["file_size_bytes"] > 0
        assert entry["generation_script"] == "scripts/evaluation/generate_anomaly_governed_evidence.py"


def _evaluation_fixture() -> dict:
    return {
        "run_id": "fixture-run",
        "threshold": 0.5,
        "threshold_strategy": "fixture",
        "score_definition": "mean_squared_reconstruction_error_per_image",
        "metrics": {
            "roc_auc": 0.48,
            "precision": 0.5,
            "recall": 0.05,
            "f1": 0.07,
        },
        "counts": {
            "test_score_count": 4,
            "normal_test_count": 2,
            "anomaly_test_count": 2,
            "predicted_normal_count": 2,
            "predicted_anomaly_count": 2,
            "correct_count": 2,
            "incorrect_count": 2,
        },
        "samples": [
            {
                "sample_id": 0,
                "image_path": "a.png",
                "true_label": "anomaly",
                "true_label_id": 1,
                "defect_type": "crack",
                "mask_path": "mask.png",
                "anomaly_score": 0.9,
                "predicted_label": "anomaly",
                "predicted_label_id": 1,
                "correct": True,
            },
            {
                "sample_id": 1,
                "image_path": "b.png",
                "true_label": "normal",
                "true_label_id": 0,
                "defect_type": "good",
                "mask_path": None,
                "anomaly_score": 0.7,
                "predicted_label": "anomaly",
                "predicted_label_id": 1,
                "correct": False,
            },
            {
                "sample_id": 2,
                "image_path": "c.png",
                "true_label": "anomaly",
                "true_label_id": 1,
                "defect_type": "crack",
                "mask_path": "mask2.png",
                "anomaly_score": 0.2,
                "predicted_label": "normal",
                "predicted_label_id": 0,
                "correct": False,
            },
            {
                "sample_id": 3,
                "image_path": "d.png",
                "true_label": "normal",
                "true_label_id": 0,
                "defect_type": "good",
                "mask_path": None,
                "anomaly_score": 0.1,
                "predicted_label": "normal",
                "predicted_label_id": 0,
                "correct": True,
            },
        ],
    }

"""Generate frontend-ready YOLO detection bundle JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts/frontend/detection/yolo_train_v0_2_0"
REGISTRY_PATH = REPO_ROOT / "artifacts/models/registry/artifact_registry.yaml"

BBOX_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
)
SUMMARY_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_per_image_summary__yolo_train_v0_2_0__validation.json"
)
CONFIDENCE_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_confidence_distribution__yolo_train_v0_2_0__validation.json"
)
GALLERY_PATH = REPO_ROOT / (
    "artifacts/models/predictions/"
    "detection_sample_gallery__yolo_train_v0_2_0__validation.json"
)

EXPECTED_RUN_ID = "yolo_train_v0_2_0"
EXPECTED_CONFIG_ID = "yolo_train_v0_2_0"
EXPECTED_TRACK_ID = "detection"
EXPECTED_TASK_TYPE = "object_detection"
EXPECTED_MODEL_NAME = "yolo"
EXPECTED_MODEL_TYPE = "yolo"
EXPECTED_MODEL_VERSION = "0.2.0"
EXPECTED_DATASET_ID = "gc10det_detection"
EXPECTED_DATASET_VERSION = "gc10det_1.0"
EXPECTED_SPLIT = "validation"
EXPECTED_IMAGE_COUNT = 345
EXPECTED_TOTAL_BBOX_COUNT = 573
EXPECTED_GALLERY_SAMPLE_COUNT = 30
EXPECTED_REGISTRY_IDS = {
    "bbox_prediction": "track_detection__yolo_train_v0_2_0__bbox_predictions_validation",
    "per_image_summary": "track_detection__yolo_train_v0_2_0__per_image_summary_validation",
    "confidence_distribution": "track_detection__yolo_train_v0_2_0__confidence_distribution_validation",
    "sample_gallery": "track_detection__yolo_train_v0_2_0__sample_gallery_validation",
}
EXPECTED_FILES = [
    "detection_overview.json",
    "detection_model_metadata.json",
    "detection_metric_cards.json",
    "detection_confidence_chart.json",
    "detection_class_summary.json",
    "detection_sample_gallery.json",
    "detection_artifact_lineage.json",
    "detection_quality_decision_summary.json",
    "frontend_detection_recommendation.json",
    "frontend_bundle_manifest.json",
]
SOURCE_PATHS = {
    "bbox_prediction": BBOX_PATH,
    "per_image_summary": SUMMARY_PATH,
    "confidence_distribution": CONFIDENCE_PATH,
    "sample_gallery": GALLERY_PATH,
}


def main() -> int:
    validation_checks: list[dict[str, str]] = []
    source_status = {
        "bbox prediction": "FAIL",
        "per-image summary": "FAIL",
        "confidence distribution": "FAIL",
        "sample gallery": "FAIL",
        "registry entries": "FAIL",
    }
    registry_hash_before = _sha256_file(REGISTRY_PATH) if REGISTRY_PATH.is_file() else ""
    notebook_hash_before = _tree_hash(REPO_ROOT / "notebooks")
    api_hash_before = _combined_tree_hash(
        [REPO_ROOT / "api", REPO_ROOT / "tests/api", REPO_ROOT / "docs/api", REPO_ROOT / "configs/api"]
    )

    try:
        sources = {name: _load_json(path, name) for name, path in SOURCE_PATHS.items()}
        for label in ["bbox prediction", "per-image summary", "confidence distribution", "sample gallery"]:
            source_status[label] = "PASS"

        registry = _load_yaml(REGISTRY_PATH, "artifact registry")
        registry_entries = _load_registry_entries(registry)
        source_status["registry entries"] = "PASS"

        validation_checks.extend(_validate_sources(sources, registry_entries))

        generated_at = _utc_now_iso()
        bundle_payloads = _build_bundle_payloads(sources, registry_entries, generated_at)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        generated_paths = []
        for filename in EXPECTED_FILES:
            path = OUTPUT_DIR / filename
            _atomic_write_json(path, bundle_payloads[filename])
            generated_paths.append(path)

        validation_checks.extend(_validate_outputs(generated_paths, sources))
        validation_checks.extend(
            _validate_no_side_effects(registry_hash_before, notebook_hash_before, api_hash_before)
        )

        if any(check["status"] == "FAIL" for check in validation_checks):
            raise ValueError("frontend bundle validation failed.")

        _print_report(source_status, validation_checks, "PASS")
        return 0
    except Exception as exc:
        validation_checks.append(_check("builder_failure", "FAIL", str(exc)))
        _print_report(source_status, validation_checks, "FAIL")
        print(f"failure_reason={exc}")
        return 1


def _build_bundle_payloads(
    sources: dict[str, dict[str, Any]],
    registry_entries: dict[str, dict[str, Any]],
    generated_at: str,
) -> dict[str, dict[str, Any]]:
    bbox = sources["bbox_prediction"]
    summary = sources["per_image_summary"]
    confidence = sources["confidence_distribution"]
    gallery = sources["sample_gallery"]
    source_artifact_paths = [_repo_relative(path) for path in SOURCE_PATHS.values()]
    registry_artifact_ids = list(EXPECTED_REGISTRY_IDS.values())
    limitations = [
        "This bundle is validation, demo, and dashboard evidence only.",
        "It does not change model quality.",
        "It requires review before operational use.",
    ]

    overview = {
        "artifact_type": "detection_frontend_overview",
        "track_id": EXPECTED_TRACK_ID,
        "task_type": EXPECTED_TASK_TYPE,
        "run_id": EXPECTED_RUN_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "model_type": EXPECTED_MODEL_TYPE,
        "model_version": EXPECTED_MODEL_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split": EXPECTED_SPLIT,
        "image_count": summary["image_count"],
        "image_with_detections_count": summary["image_with_detections_count"],
        "image_without_detections_count": summary["image_without_detections_count"],
        "total_bbox_count": summary["total_bbox_count"],
        "gallery_sample_count": gallery["gallery_sample_count"],
        "generated_at": generated_at,
        "review_status": "review_required",
        "safe_summary": "Governed validation artifacts summarized for dashboard review.",
        "limitations": limitations,
    }

    metadata = {
        "artifact_type": "detection_frontend_model_metadata",
        "run_id": EXPECTED_RUN_ID,
        "run_config_id": EXPECTED_CONFIG_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "model_type": EXPECTED_MODEL_TYPE,
        "model_version": EXPECTED_MODEL_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split": EXPECTED_SPLIT,
        "source_artifact_paths": source_artifact_paths,
        "registry_artifact_ids": registry_artifact_ids,
        "generated_at": generated_at,
        "production_ready": False,
        "deployment_candidate": False,
        "review_required": True,
    }

    cards = {
        "artifact_type": "detection_frontend_metric_cards",
        "run_id": EXPECTED_RUN_ID,
        "cards": [
            _card("validation_images", "Validation images", summary["image_count"], "images", "Images in the governed validation split.", "info"),
            _card("images_with_detections", "Images with detections", summary["image_with_detections_count"], "images", "Images with at least one predicted box.", "info"),
            _card("images_without_detections", "Images without detections", summary["image_without_detections_count"], "images", "Images with no predicted boxes.", "review"),
            _card("predicted_boxes", "Predicted boxes", summary["total_bbox_count"], "boxes", "Total predicted bounding boxes in validation evidence.", "info"),
            _card("sample_gallery_items", "Gallery samples", gallery["gallery_sample_count"], "samples", "Curated frontend gallery samples.", "info"),
        ],
        "generated_at": generated_at,
        "safe_interpretation": "Metrics summarize validation evidence for review and dashboarding only.",
    }

    confidence_bins = confidence["confidence_bins"]
    confidence_chart = {
        "artifact_type": "detection_frontend_confidence_chart",
        "run_id": EXPECTED_RUN_ID,
        "chart_title": "Detection confidence distribution",
        "chart_explanation": "Counts of predicted boxes by confidence band on the validation split.",
        "confidence_bin_edges": confidence["confidence_bin_edges"],
        "confidence_bins": confidence_bins,
        "global_confidence_summary": confidence["global_confidence_summary"],
        "series": [
            {
                "series_id": "bbox_count_by_confidence_bin",
                "label": "Predicted boxes",
                "points": [
                    {"label": item["label"], "count": item["count"], "percentage": item["percentage"]}
                    for item in confidence_bins
                ],
            }
        ],
        "generated_at": generated_at,
    }

    class_rows = [
        {
            "class_id": row["class_id"],
            "class_label": row["class_label"],
            "bbox_count": row["bbox_count"],
            "min_confidence": row["min_confidence"],
            "max_confidence": row["max_confidence"],
            "mean_confidence": row["mean_confidence"],
            "median_confidence": row["median_confidence"],
            "bin_counts": row["bin_counts"],
        }
        for row in confidence["class_confidence_summary"]
    ]
    class_summary = {
        "artifact_type": "detection_frontend_class_summary",
        "run_id": EXPECTED_RUN_ID,
        "total_bbox_count": confidence["total_bbox_count"],
        "class_count": len(class_rows),
        "class_rows": class_rows,
        "generated_at": generated_at,
    }

    category_sample_counts = {category["category_id"]: category["sample_count"] for category in gallery["categories"]}
    sample_gallery = {
        "artifact_type": "detection_frontend_sample_gallery",
        "run_id": EXPECTED_RUN_ID,
        "gallery_explanation": "Curated validation examples grouped for dashboard review.",
        "gallery_sample_count": gallery["gallery_sample_count"],
        "category_ids": [category["category_id"] for category in gallery["categories"]],
        "category_sample_counts": category_sample_counts,
        "categories": gallery["categories"],
        "generated_at": generated_at,
    }

    registry_entries_payload = {
        name: {
            "artifact_id": entry["artifact_id"],
            "artifact_type": entry["artifact_type"],
            "artifact_path": entry["artifact_path"],
            "artifact_hash": entry["artifact_hash"],
            "artifact_size_bytes": entry["artifact_size_bytes"],
            "status": entry["status"],
            "storage_backend": entry["storage_backend"],
        }
        for name, entry in registry_entries.items()
    }
    lineage = {
        "artifact_type": "detection_frontend_artifact_lineage",
        "run_id": EXPECTED_RUN_ID,
        "source_artifacts": source_artifact_paths,
        "registry_entries": registry_entries_payload,
        "artifact_hashes": {name: registry_entries[name]["artifact_hash"] for name in SOURCE_PATHS},
        "artifact_size_bytes": {name: registry_entries[name]["artifact_size_bytes"] for name in SOURCE_PATHS},
        "generated_at": generated_at,
        "limitations": limitations,
    }

    quality = {
        "artifact_type": "detection_frontend_quality_decision_summary",
        "run_id": EXPECTED_RUN_ID,
        "decision": "review_required",
        "review_required": True,
        "production_ready": False,
        "deployment_candidate": False,
        "limitations": limitations,
        "next_recommended_step": "Review the dashboard bundle before adding notebook, API, or UI integrations.",
        "generated_at": generated_at,
    }

    recommendation = {
        "artifact_type": "detection_frontend_recommendation",
        "run_id": EXPECTED_RUN_ID,
        "recommendation_status": "frontend_bundle_ready_for_review",
        "next_step": "Review generated detection dashboard JSON before building UI or API layers.",
        "what_it_can_claim": [
            "Summarizes governed validation artifacts.",
            "Supports dashboard review of counts, confidence bands, class summaries, and sample examples.",
        ],
        "what_it_cannot_claim": [
            "It cannot claim operational readiness.",
            "It cannot claim model quality changed.",
            "It cannot replace reviewer approval.",
        ],
        "source_artifact_paths": source_artifact_paths,
        "generated_at": generated_at,
    }

    manifest = {
        "artifact_type": "detection_frontend_bundle_manifest",
        "bundle_directory": _repo_relative(OUTPUT_DIR),
        "bundle_files": EXPECTED_FILES,
        "generated_file_paths": [_repo_relative(OUTPUT_DIR / filename) for filename in EXPECTED_FILES],
        "source_artifact_paths": source_artifact_paths,
        "source_artifact_count": 4,
        "bundle_artifact_count": len(EXPECTED_FILES),
        "generated_at": generated_at,
        "safe_demo_wording": "Generated dashboard JSON for validation evidence review only.",
    }

    return {
        "detection_overview.json": overview,
        "detection_model_metadata.json": metadata,
        "detection_metric_cards.json": cards,
        "detection_confidence_chart.json": confidence_chart,
        "detection_class_summary.json": class_summary,
        "detection_sample_gallery.json": sample_gallery,
        "detection_artifact_lineage.json": lineage,
        "detection_quality_decision_summary.json": quality,
        "frontend_detection_recommendation.json": recommendation,
        "frontend_bundle_manifest.json": manifest,
    }


def _validate_sources(
    sources: dict[str, dict[str, Any]],
    registry_entries: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    checks = []
    for name, payload in sources.items():
        _validate_common_metadata(payload, name)
    checks.append(_check("source_metadata_consistent", "PASS", "run/model/dataset/split metadata is consistent."))

    bbox = sources["bbox_prediction"]
    summary = sources["per_image_summary"]
    confidence = sources["confidence_distribution"]
    gallery = sources["sample_gallery"]

    if bbox.get("bbox_count") != EXPECTED_TOTAL_BBOX_COUNT:
        raise ValueError("bbox prediction bbox_count mismatch.")
    if summary.get("total_bbox_count") != bbox.get("bbox_count"):
        raise ValueError("per-image summary total_bbox_count mismatch.")
    if confidence.get("total_bbox_count") != bbox.get("bbox_count"):
        raise ValueError("confidence distribution total_bbox_count mismatch.")
    if confidence.get("global_confidence_summary", {}).get("confidence_count") != bbox.get("bbox_count"):
        raise ValueError("confidence_count mismatch.")
    if gallery.get("gallery_sample_count") != EXPECTED_GALLERY_SAMPLE_COUNT:
        raise ValueError("gallery_sample_count mismatch.")
    checks.append(_check("count_consistency", "PASS", "bbox, summary, confidence, and gallery counts are consistent."))

    for name, path in SOURCE_PATHS.items():
        entry = registry_entries[name]
        if entry.get("artifact_path") != _repo_relative(path):
            raise ValueError(f"registry path mismatch for {name}.")
        if entry.get("artifact_hash") != _sha256_file(path):
            raise ValueError(f"registry hash mismatch for {name}.")
        if entry.get("artifact_size_bytes") != path.stat().st_size:
            raise ValueError(f"registry size mismatch for {name}.")
    checks.append(_check("registry_hashes_and_sizes", "PASS", "registry hashes and sizes match local files."))

    _assert_no_forbidden_positive_claims(sources)
    checks.append(_check("source_safe_wording", "PASS", "source artifacts contain no positive production or deployment claim."))
    return checks


def _validate_common_metadata(payload: dict[str, Any], label: str) -> None:
    expected = {
        "track_id": EXPECTED_TRACK_ID,
        "task_type": EXPECTED_TASK_TYPE,
        "run_id": EXPECTED_RUN_ID,
        "run_config_id": EXPECTED_CONFIG_ID,
        "model_name": EXPECTED_MODEL_NAME,
        "model_type": EXPECTED_MODEL_TYPE,
        "model_version": EXPECTED_MODEL_VERSION,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split": EXPECTED_SPLIT,
        "image_count": EXPECTED_IMAGE_COUNT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{label} {key} mismatch.")


def _load_registry_entries(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact registry artifacts must be a list.")
    by_id = {artifact.get("artifact_id"): artifact for artifact in artifacts if isinstance(artifact, dict)}
    entries = {}
    missing = []
    for name, artifact_id in EXPECTED_REGISTRY_IDS.items():
        entry = by_id.get(artifact_id)
        if entry is None:
            missing.append(artifact_id)
        else:
            entries[name] = entry
    if missing:
        raise ValueError(f"missing registry entries: {missing}")
    return entries


def _validate_outputs(generated_paths: list[Path], sources: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    checks = []
    if len(generated_paths) != len(EXPECTED_FILES):
        raise ValueError("generated file count mismatch.")
    payloads = {}
    for path in generated_paths:
        if not path.is_file():
            raise ValueError(f"generated file missing: {_repo_relative(path)}")
        payloads[path.name] = _load_json(path, path.name)
    checks.append(_check("generated_files_parse", "PASS", "all 10 frontend JSON files exist and parse."))

    manifest = payloads["frontend_bundle_manifest.json"]
    if set(manifest.get("bundle_files", [])) != set(EXPECTED_FILES):
        raise ValueError("manifest bundle_files mismatch.")
    checks.append(_check("manifest_lists_generated_files", "PASS", "manifest lists every generated file."))

    overview = payloads["detection_overview.json"]
    if overview.get("image_count") != sources["per_image_summary"].get("image_count"):
        raise ValueError("overview image_count mismatch.")
    if overview.get("total_bbox_count") != sources["per_image_summary"].get("total_bbox_count"):
        raise ValueError("overview total_bbox_count mismatch.")

    cards = payloads["detection_metric_cards.json"]["cards"]
    card_values = {card["card_id"]: card["value"] for card in cards}
    if card_values.get("validation_images") != sources["per_image_summary"].get("image_count"):
        raise ValueError("metric card image count mismatch.")
    if card_values.get("predicted_boxes") != sources["per_image_summary"].get("total_bbox_count"):
        raise ValueError("metric card bbox count mismatch.")
    checks.append(_check("metric_cards_match_source_counts", "PASS", "metric cards match source counts."))

    gallery = payloads["detection_sample_gallery.json"]
    if gallery.get("gallery_sample_count") != sources["sample_gallery"].get("gallery_sample_count"):
        raise ValueError("gallery sample count mismatch.")
    checks.append(_check("gallery_sample_count_matches", "PASS", "gallery sample count matches source gallery."))

    chart = payloads["detection_confidence_chart.json"]
    if sum(item["count"] for item in chart["confidence_bins"]) != sources["confidence_distribution"].get("total_bbox_count"):
        raise ValueError("confidence chart bin count sum mismatch.")
    checks.append(_check("confidence_chart_counts_sum", "PASS", "confidence chart bin counts sum to total_bbox_count."))

    classes = payloads["detection_class_summary.json"]
    if sum(row["bbox_count"] for row in classes["class_rows"]) != sources["confidence_distribution"].get("total_bbox_count"):
        raise ValueError("class summary bbox count sum mismatch.")
    checks.append(_check("class_summary_counts_sum", "PASS", "class summary bbox counts sum to total_bbox_count."))

    for payload in payloads.values():
        _validate_safe_flags(payload)
    _assert_no_forbidden_positive_claims(payloads)
    checks.append(_check("safe_frontend_wording", "PASS", "safe flags and wording validation passed."))
    return checks


def _validate_no_side_effects(
    registry_hash_before: str,
    notebook_hash_before: str,
    api_hash_before: str,
) -> list[dict[str, str]]:
    if (REGISTRY_PATH.is_file() and _sha256_file(REGISTRY_PATH) != registry_hash_before) or (
        not REGISTRY_PATH.is_file() and registry_hash_before
    ):
        raise ValueError("registry files changed unexpectedly.")
    if _tree_hash(REPO_ROOT / "notebooks") != notebook_hash_before:
        raise ValueError("notebook files changed unexpectedly.")
    api_paths = [REPO_ROOT / "api", REPO_ROOT / "tests/api", REPO_ROOT / "docs/api", REPO_ROOT / "configs/api"]
    if _combined_tree_hash(api_paths) != api_hash_before:
        raise ValueError("API files changed unexpectedly.")
    return [
        _check("registry_unchanged", "PASS", "registry update was not performed."),
        _check("notebooks_unchanged", "PASS", "notebook update was not performed."),
        _check("api_unchanged", "PASS", "API update was not performed."),
    ]


def _validate_safe_flags(payload: Any) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"production_ready", "deployment_candidate"} and value is not False:
                raise ValueError(f"{key} must be false wherever present.")
            _validate_safe_flags(value)
    elif isinstance(payload, list):
        for item in payload:
            _validate_safe_flags(item)


def _assert_no_forbidden_positive_claims(payload: Any) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"production_ready", "deployment_candidate"} and value is True:
                raise ValueError(f"forbidden positive readiness flag found: {key}=true")
            _assert_no_forbidden_positive_claims(value)
        return
    if isinstance(payload, list):
        for item in payload:
            _assert_no_forbidden_positive_claims(item)
        return
    if isinstance(payload, str):
        _assert_safe_claim_text(payload)


def _assert_safe_claim_text(value: str) -> None:
    text = " ".join(value.lower().replace("_", " ").split())
    positive_phrases = [
        "production-ready",
        "deployment-safe",
        "ready for production",
        "safe for deployment",
        "deployment ready",
        "production ready",
    ]
    safe_contexts = [
        "not production-ready",
        "not a production-ready claim",
        "no production-ready claim",
        "production-ready claim: not made",
        "does not claim production readiness",
        "does not imply production readiness",
        "not deployment-safe",
        "not a deployment-safe claim",
        "no deployment-safe claim",
        "deployment-safe claim: not made",
        "does not claim deployment safety",
        "does not imply deployment safety",
        "deployment safety is not claimed",
        "operational readiness",
    ]
    for segment in _claim_text_segments(text):
        for phrase in positive_phrases:
            for phrase_index in _find_phrase_indexes(segment, phrase):
                if not _phrase_has_safe_context(segment, phrase, phrase_index, safe_contexts):
                    raise ValueError(f"forbidden positive wording found: {phrase}")


def _claim_text_segments(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"[.;\n\r]+", text) if segment.strip()]


def _find_phrase_indexes(text: str, phrase: str) -> list[int]:
    indexes = []
    start = 0
    while True:
        phrase_index = text.find(phrase, start)
        if phrase_index == -1:
            return indexes
        indexes.append(phrase_index)
        start = phrase_index + len(phrase)


def _phrase_has_safe_context(
    text: str,
    phrase: str,
    phrase_index: int,
    safe_contexts: list[str],
) -> bool:
    context_start = max(0, phrase_index - 32)
    context_end = min(len(text), phrase_index + len(phrase) + 32)
    context = text[context_start:context_end]
    return any(safe_context in context for safe_context in safe_contexts)


def _card(
    card_id: str,
    label: str,
    value: int | float | str,
    unit: str,
    description: str,
    severity_or_status: str,
) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "label": label,
        "value": value,
        "unit": unit,
        "description": description,
        "severity_or_status": severity_or_status,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {_repo_relative(path)}")
    return payload


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {_repo_relative(path)}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a YAML object: {_repo_relative(path)}")
    return payload


def _check(name: str, status: str, details: str) -> dict[str, str]:
    return {"name": name, "status": status, "details": details}


def _print_report(
    source_status: dict[str, str],
    validation_checks: list[dict[str, str]],
    final_verdict: str,
) -> None:
    print("# Detection Frontend Bundle Builder")
    print()
    print("## Source Artifacts")
    print(f"- bbox prediction: {source_status['bbox prediction']}")
    print(f"- per-image summary: {source_status['per-image summary']}")
    print(f"- confidence distribution: {source_status['confidence distribution']}")
    print(f"- sample gallery: {source_status['sample gallery']}")
    print(f"- registry entries: {source_status['registry entries']}")
    print()
    print("## Output Bundle")
    print(f"- output directory: {_repo_relative(OUTPUT_DIR)}")
    print(f"- files generated: {len(EXPECTED_FILES) if final_verdict == 'PASS' else 0}")
    print("- notebook update: NOT PERFORMED")
    print("- API update: NOT PERFORMED")
    print("- registry update: NOT PERFORMED")
    print("- production-ready claim: NOT MADE")
    print("- deployment-safe claim: NOT MADE")
    print()
    print("## Validation")
    for check in validation_checks:
        print(f"- {check['name']}: {check['status']} ({check['details']})")
    print()
    print("## Final Verdict")
    print(final_verdict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    if not path.exists():
        return ""
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(_repo_relative(child).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _combined_tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_repo_relative(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tree_hash(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())

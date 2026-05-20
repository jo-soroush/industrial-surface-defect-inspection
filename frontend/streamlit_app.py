"""Streamlit entrypoint for the initial frontend scaffold.

This scaffold presents the current project status as a read-only demo shell.
It is intentionally limited to local evidence presentation and does not
implement live prediction, API integration, Docker, or production features.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.data_loader import load_all_frontend_bundles


PROJECT_TITLE = "Industrial Surface Defect Inspection Platform"

TRACK_A_EVIDENCE_FILENAMES = (
    "metric_cards.json",
    "confusion_matrix_chart_data.json",
    "per_class_bar_chart_data.json",
    "threshold_curve_chart_data.json",
    "sample_predictions_gallery.json",
    "frontend_model_recommendation.json",
    "quality_decision_summary.json",
    "model_comparison_table.json",
    "artifact_inventory_frontend.json",
    "error_distribution_pie_data.json",
)

TRACK_B_EVIDENCE_FILENAMES = (
    "metric_cards.json",
    "anomaly_score_summary.json",
    "frontend_anomaly_summary.json",
    "reconstruction_loss_summary.json",
    "threshold_behavior.json",
    "sample_anomaly_gallery.json",
    "quality_decision_summary.json",
    "artifact_inventory_frontend.json",
)

DETECTION_EVIDENCE_FILENAMES = (
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
)

STATUS_LINES = [
    ("Track A Classification", "PASS"),
    ("Track B / Autoencoder", "PASS"),
    ("YOLO / Detection evidence layer", "COMPLETE"),
    ("Frontend app", "NOT STARTED / NOT VALIDATED"),
    ("API endpoints", "NOT STARTED"),
    ("Agent layer", "NOT STARTED"),
    ("Production readiness", "NOT CLAIMED"),
    ("Deployment safety", "NOT CLAIMED"),
]


def _safe_text(value: Any, default: str = "Unavailable") -> str:
    """Return a safe, concise display string for dashboard labels."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _render_status_summary() -> None:
    """Render a compact project status summary."""
    cols = st.columns(2)
    for idx, (label, value) in enumerate(STATUS_LINES):
        with cols[idx % 2]:
            st.metric(label, value)


def _render_limitations_banner() -> None:
    """Show the non-production limitations banner."""
    st.warning(
        "Evidence dashboard only. The project is not production-ready and not deployment-safe. "
        "No live prediction and no API upload/predict yet."
    )


def _render_data_load_health(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render a small data-load health section."""
    st.subheader("Data Load Health")
    if bundles is None:
        st.error("Frontend data contracts are not fully loadable.")
        return

    cols = st.columns(3)
    for idx, key in enumerate(("track_a", "track_b", "detection")):
        with cols[idx]:
            bundle = bundles[key]
            st.metric(
                label=f"{key.replace('_', ' ').title()} bundle",
                value="Loaded",
                help=f"{len(bundle)} JSON files loaded from the evidence bundle",
            )
            st.caption(f"{len(bundle)} JSON files")


def _format_value(value: Any) -> str:
    """Format values for compact markdown tables and labels."""
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return str(value)


def _render_markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    """Render a small markdown table from a list of row dictionaries."""
    if not rows:
        st.info("No data available.")
        return

    headers = [header for header, _ in columns]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [_format_value(row.get(key)) for _, key in columns]
        table_lines.append("| " + " | ".join(cells) + " |")

    st.markdown("\n".join(table_lines))


def _render_key_value_grid(items: list[tuple[str, Any]]) -> None:
    """Render a compact key/value grid using Streamlit metrics."""
    if not items:
        st.info("No summary metrics available.")
        return

    cols = st.columns(min(3, len(items)))
    for idx, (label, value) in enumerate(items):
        with cols[idx % len(cols)]:
            st.metric(label, _format_value(value))


def _render_overview(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the overview page content."""
    st.subheader("Overview")
    st.write(
        "This scaffold is a company-grade starting point for a demo dashboard that "
        "consumes the existing JSON data contracts for Track A, Track B, and YOLO."
    )

    if bundles is None:
        st.error("Overview data is unavailable because one or more frontend bundles failed to load.")
        return

    track_a = bundles["track_a"]
    track_b = bundles["track_b"]
    detection = bundles["detection"]

    st.markdown("### Current Project State")
    state_cols = st.columns(3)
    with state_cols[0]:
        st.metric("Track A Classification", "PASS")
    with state_cols[1]:
        st.metric("Track B / Autoencoder", "PASS")
    with state_cols[2]:
        st.metric("YOLO / Detection evidence layer", "COMPLETE")

    st.markdown("### Data Contract Health")
    health_cols = st.columns(3)
    with health_cols[0]:
        st.metric("Track A bundle", f"{len(track_a)} files")
    with health_cols[1]:
        st.metric("Track B bundle", f"{len(track_b)} files")
    with health_cols[2]:
        st.metric("Detection bundle", f"{len(detection)} files")

    st.markdown("### High-Level Evidence Summary")
    summary_cols = st.columns(3)

    track_a_reco = track_a.get("frontend_model_recommendation.json", {})
    track_b_summary = track_b.get("frontend_anomaly_summary.json", {})
    detection_overview = detection.get("detection_overview.json", {})

    with summary_cols[0]:
        st.metric(
            "Track A selected model",
            _safe_text(track_a_reco.get("selected_model_name")),
            help=_safe_text(track_a_reco.get("selected_model_version")),
        )
        st.caption(
            f"Run: {_safe_text(track_a_reco.get('selected_run_id'))} | "
            f"Threshold: {_safe_text(track_a_reco.get('selected_threshold'))}"
        )

    with summary_cols[1]:
        st.metric(
            "Track B model",
            _safe_text(track_b_summary.get("model_type")),
            help=_safe_text(track_b_summary.get("model_version")),
        )
        key_metrics = track_b_summary.get("key_metrics", {})
        st.caption(
            " | ".join(
                [
                    f"Canonical status: {_safe_text(track_b_summary.get('canonical_status'))}",
                    f"ROC AUC: {_safe_text(key_metrics.get('roc_auc'))}",
                    f"Threshold: {_safe_text(key_metrics.get('threshold'))}",
                ]
            )
        )

    with summary_cols[2]:
        st.metric(
            "Detection images",
            _safe_text(detection_overview.get("image_count")),
            help=_safe_text(detection_overview.get("safe_summary")),
        )
        st.caption(
            f"Total bboxes: {_safe_text(detection_overview.get('total_bbox_count'))} | "
            f"Gallery samples: {_safe_text(detection_overview.get('gallery_sample_count'))}"
        )

    st.markdown("### Evidence Paths")
    path_cols = st.columns(3)
    with path_cols[0]:
        st.code("artifacts/frontend/track_a/", language="text")
    with path_cols[1]:
        st.code("artifacts/frontend/track_b/", language="text")
    with path_cols[2]:
        st.code("artifacts/frontend/detection/yolo_train_v0_2_0/", language="text")


def _render_track_a(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the Track A classification page."""
    st.subheader("Track A Classification")
    st.warning(
        "Evidence/dashboard view only. Not production-ready. Not deployment-safe. "
        "No live prediction on this page."
    )
    st.write(
        "This page summarizes the governed Track A classification bundle and keeps the "
        "presentation focused on current evidence, not live inference."
    )

    if bundles is None:
        st.error("Track A evidence is unavailable because the frontend bundles failed to load.")
        return

    track_a = bundles["track_a"]
    metric_cards = track_a.get("metric_cards.json", {})
    recommendation = track_a.get("frontend_model_recommendation.json", {})
    comparison = track_a.get("model_comparison_table.json", {})
    confusion = track_a.get("confusion_matrix_chart_data.json", {})
    per_class = track_a.get("per_class_bar_chart_data.json", {})
    threshold_curve = track_a.get("threshold_curve_chart_data.json", {})
    gallery = track_a.get("sample_predictions_gallery.json", {})
    quality = track_a.get("quality_decision_summary.json", {})
    inventory = track_a.get("artifact_inventory_frontend.json", {})
    error_distribution = track_a.get("error_distribution_pie_data.json", {})

    top_cols = st.columns(3)
    with top_cols[0]:
        st.metric(
            "Selected model",
            _format_value(recommendation.get("selected_model_name") or metric_cards.get("selected_model_name")),
            help=_format_value(recommendation.get("selected_model_version") or metric_cards.get("selected_model_version")),
        )
    with top_cols[1]:
        st.metric(
            "Recommended threshold",
            _format_value(recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
        )
    with top_cols[2]:
        st.metric(
            "Quality target",
            _format_value(quality.get("quality_target_status") or metric_cards.get("quality_target_status")),
            help=_format_value(quality.get("decision") or metric_cards.get("selected_model_quality_status")),
        )

    st.markdown("### Metric Cards")
    cards = metric_cards.get("cards", [])
    if cards:
        for start_idx in range(0, len(cards), 3):
            row_cards = cards[start_idx : start_idx + 3]
            cols = st.columns(len(row_cards))
            for col, card in zip(cols, row_cards):
                with col:
                    st.metric(card.get("title", "Metric"), _format_value(card.get("value")))
                    detail = card.get("detail")
                    if detail:
                        st.caption(str(detail))
    else:
        st.info("No Track A metric cards available.")

    st.markdown("### Comparison and Decision Summary")
    comparison_cols = st.columns([1.4, 1])
    with comparison_cols[0]:
        st.caption("Selected model comparison table")
        _render_markdown_table(
            comparison.get("rows", []),
            [
                ("Rank", "rank"),
                ("Selected", "selected"),
                ("Model", "model_name"),
                ("Version", "model_version"),
                ("Run ID", "run_id"),
                ("Macro F1", "macro_f1"),
                ("Accuracy", "accuracy"),
                ("Recall", "recall"),
                ("Threshold", "threshold_used"),
                ("Status", "short_status"),
            ],
        )
    with comparison_cols[1]:
        st.caption("Quality decision summary")
        _render_key_value_grid(
            [
                ("Decision", quality.get("decision")),
                ("Selected run", quality.get("run_id")),
                ("Selected model", quality.get("model_name")),
                ("Version", quality.get("model_version")),
                ("Production ready", quality.get("production_ready")),
                ("Deployment candidate", quality.get("deployment_candidate")),
                ("Recommended threshold", quality.get("recommended_threshold")),
                ("Next step", quality.get("next_recommended_step")),
            ]
        )

    st.markdown("### Evidence Tables")
    evidence_cols = st.columns(2)
    with evidence_cols[0]:
        st.caption("Confusion matrix data")
        matrix = confusion.get("matrix", [])
        labels = confusion.get("labels", {})
        matrix_rows = []
        row_labels = labels.get("rows", [])
        col_labels = labels.get("columns", [])
        for idx, row in enumerate(matrix):
            matrix_rows.append(
                {
                    "actual": row_labels[idx] if idx < len(row_labels) else f"row_{idx}",
                    **{
                        (col_labels[col_idx] if col_idx < len(col_labels) else f"col_{col_idx}"): value
                        for col_idx, value in enumerate(row)
                    },
                }
            )
        _render_markdown_table(
            matrix_rows,
            [
                ("Actual", "actual"),
                (col_labels[0] if len(col_labels) > 0 else "Column 1", col_labels[0] if len(col_labels) > 0 else "col_0"),
                (col_labels[1] if len(col_labels) > 1 else "Column 2", col_labels[1] if len(col_labels) > 1 else "col_1"),
            ],
        )
    with evidence_cols[1]:
        st.caption("Per-class validation metrics")
        _render_markdown_table(
            per_class.get("classes", []),
            [
                ("Class", "label"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
            ],
        )

    st.markdown("### Threshold And Error Analysis")
    analysis_cols = st.columns(2)
    with analysis_cols[0]:
        st.caption("Threshold sweep summary")
        threshold_rows = threshold_curve.get("rows", [])
        if threshold_rows:
            key_thresholds = [
                row
                for row in threshold_rows
                if row.get("threshold") in {threshold_curve.get("baseline_threshold"), threshold_curve.get("recommended_threshold")}
            ]
            if key_thresholds:
                _render_markdown_table(
                    key_thresholds,
                    [
                        ("Threshold", "threshold"),
                        ("Precision", "precision"),
                        ("Recall", "recall"),
                        ("Macro F1", "macro_f1"),
                        ("Accuracy", "accuracy"),
                        ("False Positives", "false_positive"),
                        ("False Negatives", "false_negative"),
                    ],
                )
            else:
                _render_markdown_table(
                    threshold_rows[:4],
                    [
                        ("Threshold", "threshold"),
                        ("Precision", "precision"),
                        ("Recall", "recall"),
                        ("Macro F1", "macro_f1"),
                        ("Accuracy", "accuracy"),
                    ],
                )
        else:
            st.info("No threshold curve data available.")

    with analysis_cols[1]:
        st.caption("Error distribution")
        error_rows = error_distribution.get("segments", [])
        _render_markdown_table(
            error_rows,
            [
                ("Label", "label"),
                ("Count", "count"),
                ("Share", "percentage"),
            ],
        )

    st.markdown("### Sample Gallery Summary")
    gallery_cols = st.columns(2)
    with gallery_cols[0]:
        st.metric("Gallery samples", _format_value(gallery.get("gallery_sample_count")))
        gallery_counts = gallery.get("counts_by_error_type", {})
        count_rows = [
            {"error_type": label, "count": count}
            for label, count in (
                ("true_positive", gallery_counts.get("true_positive")),
                ("true_negative", gallery_counts.get("true_negative")),
                ("false_positive", gallery_counts.get("false_positive")),
                ("false_negative", gallery_counts.get("false_negative")),
            )
            if count is not None
        ]
        _render_markdown_table(
            count_rows,
            [
                ("Error type", "error_type"),
                ("Count", "count"),
            ],
        )
        st.caption(
            "This page shows summary metadata only; image rendering can be added later without changing the data contract."
        )
    with gallery_cols[1]:
        gallery_counts = gallery.get("selected_counts_by_error_type", {})
        gallery_rows = [
            {"error_type": label, "count": gallery_counts.get(label), "available": gallery.get("counts_by_error_type", {}).get(label)}
            for label in ("true_positive", "true_negative", "false_positive", "false_negative")
            if label in gallery.get("counts_by_error_type", {})
        ]
        _render_markdown_table(
            gallery_rows,
            [
                ("Error type", "error_type"),
                ("Selected count", "count"),
                ("Available count", "available"),
            ],
        )

    st.markdown("### Evidence Inventory")
    inventory_cols = st.columns(3)
    with inventory_cols[0]:
        st.metric("Bundle files", _format_value(inventory.get("bundle_artifact_count")))
    with inventory_cols[1]:
        st.metric("Source artifacts", _format_value(inventory.get("source_artifact_count")))
    with inventory_cols[2]:
        st.metric("Selected threshold", _format_value(recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")))

    with st.expander("Track A evidence file list", expanded=False):
        for filename in TRACK_A_EVIDENCE_FILENAMES:
            st.code(filename, language="text")

    with st.expander("Track A evidence paths", expanded=False):
        st.code("artifacts/frontend/track_a/", language="text")
        st.write("Selected model recommendation, comparison table, quality summary, and evidence inventory remain in the bundle.")

    st.markdown("### Safe Interpretation")
    st.write(
        "ResNet18 v0.4.0 is the strongest governed Track A candidate, but this dashboard page remains an evidence view only."
    )


def _render_track_b(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the Track B anomaly detection page."""
    st.subheader("Track B Anomaly Detection")
    st.warning(
        "Evidence/dashboard view only. Not production-ready. Not deployment-safe. "
        "No live prediction on this page."
    )
    st.write(
        "This page summarizes the governed Track B autoencoder bundle and keeps the "
        "presentation focused on evidence rather than live inference."
    )

    if bundles is None:
        st.error("Track B evidence is unavailable because the frontend bundles failed to load.")
        return

    track_b = bundles["track_b"]
    metric_cards = track_b.get("metric_cards.json", {})
    anomaly_summary = track_b.get("anomaly_score_summary.json", {})
    frontend_summary = track_b.get("frontend_anomaly_summary.json", {})
    reconstruction = track_b.get("reconstruction_loss_summary.json", {})
    threshold_behavior = track_b.get("threshold_behavior.json", {})
    gallery = track_b.get("sample_anomaly_gallery.json", {})
    quality = track_b.get("quality_decision_summary.json", {})
    inventory = track_b.get("artifact_inventory_frontend.json", {})

    top_cols = st.columns(3)
    with top_cols[0]:
        st.metric(
            "Model",
            _format_value(frontend_summary.get("model_type") or metric_cards.get("model_type")),
            help=_format_value(frontend_summary.get("model_version") or metric_cards.get("model_version")),
        )
    with top_cols[1]:
        st.metric(
            "Threshold",
            _format_value(frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")),
        )
    with top_cols[2]:
        st.metric(
            "Canonical status",
            _format_value(frontend_summary.get("canonical_status") or metric_cards.get("canonical_status")),
            help="PR AUC is unavailable in governed evidence and is not fabricated.",
        )

    st.markdown("### Metric Cards")
    cards = metric_cards.get("cards", [])
    if cards:
        for start_idx in range(0, len(cards), 3):
            row_cards = cards[start_idx : start_idx + 3]
            cols = st.columns(len(row_cards))
            for col, card in zip(cols, row_cards):
                with col:
                    st.metric(card.get("title", "Metric"), _format_value(card.get("value")))
                    detail = card.get("detail")
                    if detail:
                        st.caption(str(detail))
    else:
        st.info("No Track B metric cards available.")

    st.markdown("### Anomaly Summary And Decision")
    summary_cols = st.columns([1.3, 1])
    with summary_cols[0]:
        st.caption("Frontend anomaly summary")
        _render_key_value_grid(
            [
                ("Model", frontend_summary.get("model_type")),
                ("Version", frontend_summary.get("model_version")),
                ("Canonical status", frontend_summary.get("canonical_status")),
                ("ROC AUC", frontend_summary.get("key_metrics", {}).get("roc_auc")),
                ("Threshold", frontend_summary.get("key_metrics", {}).get("threshold")),
                ("Next step", frontend_summary.get("next_step")),
            ]
        )
        st.caption("Anomaly score summary")
        _render_key_value_grid(
            [
                ("Score definition", anomaly_summary.get("score_definition")),
                ("Threshold strategy", anomaly_summary.get("threshold_strategy")),
                ("Threshold", anomaly_summary.get("threshold")),
                ("Normal/anomaly separation", anomaly_summary.get("normal_vs_anomaly_score_separation")),
            ]
        )
    with summary_cols[1]:
        st.caption("Quality decision summary")
        _render_key_value_grid(
            [
                ("Decision", quality.get("decision")),
                ("Canonical status", quality.get("canonical_status")),
                ("Model status", quality.get("model_quality_status")),
                ("Production ready", quality.get("production_ready")),
                ("Deployment candidate", quality.get("deployment_candidate")),
                ("Threshold", quality.get("threshold")),
                ("Next step", quality.get("next_recommended_step")),
            ]
        )

    st.markdown("### Reconstruction And Threshold Behavior")
    analysis_cols = st.columns(2)
    with analysis_cols[0]:
        st.caption("Reconstruction loss summary")
        _render_markdown_table(
            reconstruction.get("chart_rows", []),
            [
                ("Epoch", "epoch"),
                ("Reconstruction loss", "reconstruction_loss"),
            ],
        )
    with analysis_cols[1]:
        st.caption("Threshold behavior")
        _render_markdown_table(
            threshold_behavior.get("rows", []),
            [
                ("Threshold", "threshold"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
                ("False Positives", "false_positive"),
                ("False Negatives", "false_negative"),
            ],
        )

    st.markdown("### Sample Gallery Summary")
    gallery_cols = st.columns(2)
    with gallery_cols[0]:
        st.metric("Gallery samples", _format_value(gallery.get("gallery_sample_count")))
        st.caption(
            "This page shows summary metadata only; image rendering can be added later without changing the data contract."
        )
    with gallery_cols[1]:
        count_rows = [
            {"error_type": label, "count": count}
            for label, count in (
                ("true_positive", gallery.get("counts_by_error_type", {}).get("true_positive")),
                ("true_negative", gallery.get("counts_by_error_type", {}).get("true_negative")),
                ("false_positive", gallery.get("counts_by_error_type", {}).get("false_positive")),
                ("false_negative", gallery.get("counts_by_error_type", {}).get("false_negative")),
            )
            if count is not None
        ]
        _render_markdown_table(
            count_rows,
            [
                ("Error type", "error_type"),
                ("Count", "count"),
            ],
        )

    st.markdown("### Evidence Inventory")
    inventory_cols = st.columns(3)
    with inventory_cols[0]:
        st.metric("Bundle files", _format_value(inventory.get("bundle_artifact_count")))
    with inventory_cols[1]:
        st.metric("Source artifacts", _format_value(inventory.get("source_artifact_count")))
    with inventory_cols[2]:
        st.metric("Selected threshold", _format_value(frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")))

    st.caption("PR AUC is unavailable in governed evidence and not fabricated.")

    with st.expander("Track B evidence file list", expanded=False):
        for filename in TRACK_B_EVIDENCE_FILENAMES:
            st.code(filename, language="text")

    with st.expander("Track B evidence paths", expanded=False):
        st.code("artifacts/frontend/track_b/", language="text")
        st.write("Metric cards, anomaly summary, reconstruction loss, threshold behavior, quality summary, and inventory remain in the bundle.")


def _render_yolo(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the YOLO / Detection page."""
    st.subheader("YOLO Detection")
    st.warning(
        "Evidence/dashboard view only. Not production-ready. Not deployment-safe. "
        "No live prediction on this page. No API upload/predict yet."
    )
    st.write(
        "This page summarizes the governed YOLO / Detection bundle and keeps the "
        "presentation focused on current evidence, not live inference."
    )

    if bundles is None:
        st.error("Detection evidence is unavailable because the frontend bundles failed to load.")
        return

    detection = bundles["detection"]
    overview = detection.get("detection_overview.json", {})
    metadata = detection.get("detection_model_metadata.json", {})
    metric_cards = detection.get("detection_metric_cards.json", {})
    confidence_chart = detection.get("detection_confidence_chart.json", {})
    class_summary = detection.get("detection_class_summary.json", {})
    sample_gallery = detection.get("detection_sample_gallery.json", {})
    lineage = detection.get("detection_artifact_lineage.json", {})
    quality = detection.get("detection_quality_decision_summary.json", {})
    recommendation = detection.get("frontend_detection_recommendation.json", {})
    manifest = detection.get("frontend_bundle_manifest.json", {})

    st.markdown("### Detection Overview")
    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.metric("Images", _format_value(overview.get("image_count")))
    with overview_cols[1]:
        st.metric("Images with detections", _format_value(overview.get("image_with_detections_count")))
    with overview_cols[2]:
        st.metric("Total bboxes", _format_value(overview.get("total_bbox_count")))
    with overview_cols[3]:
        st.metric("Gallery samples", _format_value(overview.get("gallery_sample_count")))

    st.caption(
        "Run yolo_train_v0_2_0 | "
        f"Model: {_format_value(metadata.get('model_name'))} | "
        f"Version: {_format_value(metadata.get('model_version'))} | "
        f"Dataset: {_format_value(metadata.get('dataset_id'))}"
    )

    st.markdown("### Metric Cards")
    cards = metric_cards.get("cards", [])
    if cards:
        for start_idx in range(0, len(cards), 3):
            row_cards = cards[start_idx : start_idx + 3]
            cols = st.columns(len(row_cards))
            for col, card in zip(cols, row_cards):
                with col:
                    st.metric(card.get("label", "Metric"), _format_value(card.get("value")))
                    description = card.get("description")
                    if description:
                        st.caption(str(description))
    else:
        st.info("No detection metric cards available.")

    st.markdown("### Confidence Distribution")
    confidence_rows = confidence_chart.get("confidence_bins", [])
    if confidence_rows:
        _render_markdown_table(
            confidence_rows,
            [
                ("Band", "label"),
                ("Count", "count"),
                ("Share", "percentage"),
            ],
        )
    else:
        st.info("No confidence distribution data available.")

    st.markdown("### Class Summary")
    class_rows = class_summary.get("class_rows", [])
    if class_rows:
        _render_markdown_table(
            class_rows,
            [
                ("Class", "class_label"),
                ("BBox count", "bbox_count"),
                ("Min confidence", "min_confidence"),
                ("Mean confidence", "mean_confidence"),
                ("Median confidence", "median_confidence"),
            ],
        )
    else:
        st.info("No class summary data available.")

    st.markdown("### Sample Gallery Summary")
    sample_cols = st.columns(2)
    with sample_cols[0]:
        st.metric("Gallery samples", _format_value(sample_gallery.get("gallery_sample_count")))
        st.caption("This page shows summary metadata only; image rendering can be added later without changing the data contract.")
    with sample_cols[1]:
        category_counts = sample_gallery.get("category_sample_counts", {})
        gallery_rows = [
            {"category": label, "count": count}
            for label, count in category_counts.items()
        ]
        _render_markdown_table(
            gallery_rows,
            [
                ("Category", "category"),
                ("Count", "count"),
            ],
        )

    st.markdown("### Artifact Lineage And Decision")
    lineage_cols = st.columns(2)
    with lineage_cols[0]:
        _render_key_value_grid(
            [
                ("Decision", quality.get("decision")),
                ("Review required", quality.get("review_required")),
                ("Production ready", quality.get("production_ready")),
                ("Deployment candidate", quality.get("deployment_candidate")),
                ("Next step", quality.get("next_recommended_step")),
            ]
        )
        st.caption(
            f"Recommendation: {_format_value(recommendation.get('recommendation_status'))}"
        )
        st.caption(
            f"Run: {_format_value(metadata.get('run_id'))} | "
            f"Version: {_format_value(metadata.get('model_version'))}"
        )
    with lineage_cols[1]:
        source_paths = lineage.get("source_artifacts", [])
        lineage_rows = [
            {
                "artifact": item.get("artifact_type") or item.get("artifact_id"),
                "path": item.get("artifact_path") or item.get("path"),
                "hash": item.get("artifact_hash") or item.get("sha256"),
            }
            for item in source_paths
            if isinstance(item, dict)
        ]
        _render_markdown_table(
            lineage_rows,
            [
                ("Artifact", "artifact"),
                ("Path", "path"),
                ("Hash", "hash"),
            ],
        )

    with st.expander("Detection bundle manifest", expanded=False):
        st.caption("Bundle directory and generated file list")
        st.code("artifacts/frontend/detection/yolo_train_v0_2_0/", language="text")
        _render_markdown_table(
            [
                {"file": name}
                for name in manifest.get("bundle_files", [])
            ],
            [
                ("File", "file"),
            ],
        )

    with st.expander("Detection evidence file list", expanded=False):
        for filename in DETECTION_EVIDENCE_FILENAMES:
            st.code(filename, language="text")

    st.markdown("### Safe Interpretation")
    st.write(
        "YOLO / Detection evidence layer COMPLETE. This page is a review-oriented dashboard view only "
        "and does not claim production readiness or deployment safety."
    )


def _render_upload_predict() -> None:
    """Render the upload/predict page placeholder."""
    st.subheader("Upload / Predict")
    st.write(
        "Planned, not implemented yet. This page will later connect to a simple FastAPI backend "
        "for upload/predict behavior."
    )


def _render_limitations() -> None:
    """Render the limitations and safety page."""
    st.subheader("Limitations / Safety")
    st.write(
        "This scaffold is evidence-focused only. It does not train models, recompute metrics, "
        "create artifacts, update registries, or claim production or deployment readiness."
    )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.title(PROJECT_TITLE)
    _render_limitations_banner()
    _render_status_summary()

    try:
        bundles = load_all_frontend_bundles()
    except (FileNotFoundError, ValueError) as exc:
        bundles = None
        st.error(f"Frontend data contracts are not fully loadable: {exc}")

    _render_data_load_health(bundles)

    pages = {
        "Overview": lambda: _render_overview(bundles),
        "Track A Classification": lambda: _render_track_a(bundles),
        "Track B Anomaly Detection": lambda: _render_track_b(bundles),
        "YOLO Detection": lambda: _render_yolo(bundles),
        "Upload / Predict": _render_upload_predict,
        "Limitations / Safety": _render_limitations,
    }

    choice = st.sidebar.radio("Navigation", list(pages.keys()), index=0)
    st.sidebar.caption("This scaffold is read-only and consumes existing evidence bundles in later phases.")
    pages[choice]()


if __name__ == "__main__":
    main()

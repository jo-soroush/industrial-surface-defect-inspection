"""Streamlit entrypoint for the initial frontend scaffold.

This scaffold presents the current project status as a read-only demo shell.
It is intentionally limited to local evidence presentation and does not
implement live prediction, API integration, Docker, or production features.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

from frontend.data_loader import load_all_frontend_bundles


PROJECT_TITLE = "Industrial Surface Defect Inspection Platform"
API_DEFAULT_BASE_URL = "http://localhost:8000"
UPLOAD_ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "webp")
UPLOAD_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
UPLOAD_MAX_BYTES = 10 * 1024 * 1024

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
        "no live prediction and no API upload/predict yet."
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


def _build_donut_figure(
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
) -> go.Figure:
    """Build a small donut chart for dashboard summaries."""
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                sort=False,
                textinfo="label+percent",
                marker=dict(colors=colors),
                hovertemplate="%{label}<br>%{value}<extra></extra>",
                direction="clockwise",
            )
        ]
    )
    figure.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _build_grouped_bar_figure(
    title: str,
    categories: list[str],
    series_map: dict[str, list[float]],
    colors: dict[str, str],
    yaxis_title: str,
) -> go.Figure:
    """Build a grouped bar chart for dashboard summaries."""
    figure = go.Figure()
    for series_name, series_values in series_map.items():
        figure.add_trace(
            go.Bar(
                name=series_name,
                x=categories,
                y=series_values,
                marker_color=colors.get(series_name),
            )
        )
    figure.update_layout(
        title=title,
        height=320,
        barmode="group",
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Metric",
        yaxis_title=yaxis_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _build_line_figure(
    title: str,
    x_values: list[float],
    series_map: dict[str, list[float]],
    colors: dict[str, str],
    yaxis_title: str,
) -> go.Figure:
    """Build a multi-line chart for dashboard summaries."""
    figure = go.Figure()
    for series_name, series_values in series_map.items():
        figure.add_trace(
            go.Scatter(
                name=series_name,
                x=x_values,
                y=series_values,
                mode="lines+markers",
                line=dict(color=colors.get(series_name)),
                marker=dict(size=7),
            )
        )
    figure.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Metric",
        yaxis_title=yaxis_title,
        xaxis_title="Threshold",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _render_overview_status_chart() -> None:
    """Render a compact plotly overview chart for the current module state."""
    labels = [
        "Evidence complete",
        "Prototype available",
        "Not started",
        "Not claimed",
    ]
    values = [3, 1, 2, 2]
    colors = ["#2ca02c", "#1f77b4", "#9e9e9e", "#ff9800"]

    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                sort=False,
                textinfo="label+percent",
                marker=dict(colors=colors),
                hovertemplate="%{label}<br>%{value}<extra></extra>",
                direction="clockwise",
            )
        ]
    )
    figure.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)


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


def _render_series_chart(
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    chart_kind: str,
    fallback_columns: list[tuple[str, str]],
) -> None:
    """Render a simple Streamlit chart or fall back to a table."""
    numeric_rows = [row for row in rows if isinstance(row, dict)]
    if not numeric_rows:
        st.info("No chart data available.")
        return

    if chart_kind == "bar":
        chart_data = {
            row.get(x_key, f"row_{idx}"): float(row.get(y_keys[0], 0) or 0)
            for idx, row in enumerate(numeric_rows)
        }
        st.bar_chart(chart_data)
    elif chart_kind == "line":
        chart_data = {
            y_key: [float(row.get(y_key, 0) or 0) for row in numeric_rows]
            for y_key in y_keys
        }
        st.line_chart(chart_data)
    else:
        st.info("No chart renderer available.")
        return

    with st.expander("Show table", expanded=False):
        _render_markdown_table(numeric_rows, fallback_columns)


def _render_overview(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the overview page content."""
    st.markdown(f"## {PROJECT_TITLE}")
    st.markdown(
        "A governed evidence dashboard for comparing Track A, Track B, and YOLO summaries "
        "while keeping the local Track A upload prototype clearly separated."
    )
    st.info(
        "Safe status: governed evidence only. not production-ready. not deployment-safe."
    )

    if bundles is None:
        st.error("Overview data is unavailable because one or more frontend bundles failed to load.")
        return

    track_a = bundles["track_a"]
    track_b = bundles["track_b"]
    detection = bundles["detection"]

    track_a_reco = track_a.get("frontend_model_recommendation.json", {})
    track_b_summary = track_b.get("frontend_anomaly_summary.json", {})
    detection_overview = detection.get("detection_overview.json", {})

    hero_cols = st.columns([1.25, 1])
    with hero_cols[0]:
        st.markdown("### Current status")
        st.caption(
            "Green means evidence complete, blue means a local prototype exists, and gray/orange marks work that is not yet a production claim."
        )
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Track A Classification", "PASS", help="Governed evidence and local prototype available")
        with summary_cols[1]:
            st.metric("Track B / Autoencoder", "PASS", help="Governed evidence available")
        with summary_cols[2]:
            st.metric("YOLO Detection", "COMPLETE", help="Governed evidence layer complete")
        with summary_cols[3]:
            st.metric("Upload / Predict", "LOCAL PROTOTYPE", help="Track A only")

        st.markdown("### What this dashboard can do")
        st.write(
            "- View governed evidence from Track A, Track B, and YOLO\n"
            "- Compare the high-level summaries for each track\n"
            "- Run the local Track A upload/predict prototype"
        )

        st.markdown("### What it cannot claim yet")
        st.write(
            "- not production-ready\n"
            "- not deployment-safe\n"
            "- YOLO/anomaly upload prediction not implemented yet"
        )

    with hero_cols[1]:
        st.markdown("### Visual status")
        _render_overview_status_chart()

    st.markdown("### Data contract health")
    health_cols = st.columns(3)
    with health_cols[0]:
        st.metric("Track A bundle", f"{len(track_a)} files")
    with health_cols[1]:
        st.metric("Track B bundle", f"{len(track_b)} files")
    with health_cols[2]:
        st.metric("Detection bundle", f"{len(detection)} files")

    st.markdown("### High-level evidence summary")
    summary_cols = st.columns(3)
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

    with st.expander("Technical evidence", expanded=False):
        tech_tabs = st.tabs(["Data Contracts", "Evidence Paths", "Run Notes"])
        with tech_tabs[0]:
            _render_key_value_grid(
                [
                    ("Track A files", len(track_a)),
                    ("Track B files", len(track_b)),
                    ("Detection files", len(detection)),
                ]
            )
        with tech_tabs[1]:
            st.code("artifacts/frontend/track_a/", language="text")
            st.code("artifacts/frontend/track_b/", language="text")
            st.code("artifacts/frontend/detection/yolo_train_v0_2_0/", language="text")
        with tech_tabs[2]:
            _render_key_value_grid(
                [
                    ("Track A run", track_a_reco.get("selected_run_id")),
                    ("Track A threshold", track_a_reco.get("selected_threshold")),
                    ("Track B threshold", track_b_summary.get("key_metrics", {}).get("threshold")),
                    ("Detection run", detection_overview.get("run_id")),
                ]
            )


def _render_track_a(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the Track A classification page."""
    st.markdown("## Track A Classification")
    st.markdown(
        "A governed binary classifier page for the Track A good-vs-defect task, presented as an evidence dashboard rather than a live deployment view."
    )
    st.warning(
        "Evidence/dashboard view only. not production-ready. not deployment-safe. "
        "No live prediction on this Track A evidence page."
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

    top_cols = st.columns(4)
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
            "Task",
            "good vs defect",
            help="Binary classification task for the governed Track A model.",
        )
    with top_cols[3]:
        st.metric(
            "Quality/status",
            _format_value(quality.get("decision") or metric_cards.get("selected_model_quality_status")),
            help=_format_value(quality.get("quality_target_status") or metric_cards.get("quality_target_status")),
        )

    tabs = st.tabs(["Summary", "Charts", "Evidence"])
    with tabs[0]:
        st.caption("Summary cards and a concise sample gallery view.")
        quality_cols = st.columns(4)
        with quality_cols[0]:
            st.metric("Decision", _format_value(quality.get("decision")))
        with quality_cols[1]:
            st.metric("Production ready", _format_value(quality.get("production_ready")))
        with quality_cols[2]:
            st.metric("Deployment candidate", _format_value(quality.get("deployment_candidate")))
        with quality_cols[3]:
            st.metric(
                "Selected threshold",
                _format_value(recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
            )

        gallery_cols = st.columns(2)
        with gallery_cols[0]:
            st.metric("Sample gallery images", _format_value(gallery.get("gallery_sample_count")))
            st.caption(
                "The gallery remains summary-only here; images can be surfaced later without changing the governed data contract."
            )
        with gallery_cols[1]:
            gallery_counts = gallery.get("counts_by_error_type", {})
            _render_key_value_grid(
                [
                    ("True positive", gallery_counts.get("true_positive")),
                    ("True negative", gallery_counts.get("true_negative")),
                    ("False positive", gallery_counts.get("false_positive")),
                    ("False negative", gallery_counts.get("false_negative")),
                ]
            )

        with st.expander("Metric cards", expanded=False):
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

        st.markdown("### Quality decision")
        _render_key_value_grid(
            [
                ("Selected model", recommendation.get("selected_model_name") or metric_cards.get("selected_model_name")),
                ("Version", recommendation.get("selected_model_version") or metric_cards.get("selected_model_version")),
                ("Run ID", recommendation.get("selected_run_id")),
                ("Threshold", recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
                ("Quality target", quality.get("quality_target_status") or metric_cards.get("quality_target_status")),
            ]
        )
    with tabs[1]:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("Error distribution")
            error_rows = error_distribution.get("segments", [])
            if error_rows:
                labels = [str(row.get("label", f"segment_{idx}")) for idx, row in enumerate(error_rows)]
                values = [float(row.get("count", 0) or 0) for row in error_rows]
                palette = []
                for label in labels:
                    lowered = label.lower()
                    if "false_negative" in lowered or "fn" in lowered:
                        palette.append("#d62728")
                    elif "false_positive" in lowered or "fp" in lowered:
                        palette.append("#ff9800")
                    elif "true_positive" in lowered or "tp" in lowered:
                        palette.append("#2ca02c")
                    else:
                        palette.append("#1f77b4")
                st.plotly_chart(
                    _build_donut_figure(
                        "Track A error distribution",
                        labels,
                        values,
                        palette,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No error distribution data available.")

        with chart_cols[1]:
            st.caption("Per-class performance")
            class_metric_rows = per_class.get("classes", [])
            if class_metric_rows:
                categories = [str(row.get("label", f"class_{idx}")) for idx, row in enumerate(class_metric_rows)]
                series_map = {
                    "Precision": [float(row.get("precision", 0) or 0) for row in class_metric_rows],
                    "Recall": [float(row.get("recall", 0) or 0) for row in class_metric_rows],
                    "F1": [float(row.get("f1", 0) or 0) for row in class_metric_rows],
                }
                st.plotly_chart(
                    _build_grouped_bar_figure(
                        "Track A per-class performance",
                        categories,
                        series_map,
                        {"Precision": "#1f77b4", "Recall": "#ff9800", "F1": "#2ca02c"},
                        "Score",
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No per-class data available for charting.")
            with st.expander("Per-class table", expanded=False):
                _render_markdown_table(
                    class_metric_rows,
                    [
                        ("Class", "label"),
                        ("Precision", "precision"),
                        ("Recall", "recall"),
                        ("F1", "f1"),
                    ],
                )

        st.caption("Threshold behavior")
        threshold_rows = threshold_curve.get("rows", [])
        if threshold_rows:
            threshold_values = [float(row.get("threshold", 0) or 0) for row in threshold_rows]
            series_map = {
                "Precision": [float(row.get("precision", 0) or 0) for row in threshold_rows],
                "Recall": [float(row.get("recall", 0) or 0) for row in threshold_rows],
                "Macro F1": [float(row.get("macro_f1", 0) or 0) for row in threshold_rows],
                "Accuracy": [float(row.get("accuracy", 0) or 0) for row in threshold_rows],
            }
            st.plotly_chart(
                _build_line_figure(
                    "Track A threshold behavior",
                    threshold_values,
                    series_map,
                    {
                        "Precision": "#1f77b4",
                        "Recall": "#ff9800",
                        "Macro F1": "#2ca02c",
                        "Accuracy": "#9467bd",
                    },
                    "Score",
                ),
                use_container_width=True,
            )
            st.caption(
                "Thresholds: "
                + ", ".join(_format_value(row.get("threshold")) for row in threshold_rows)
            )
        else:
            st.info("No threshold sweep data available for charting.")
    with tabs[2]:
        st.caption("Detailed evidence and raw tables are kept inside expanders.")
        st.caption("The model comparison details are kept behind an expander.")
        with st.expander("Model comparison", expanded=False):
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
        with st.expander("Confusion matrix", expanded=False):
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
        with st.expander("Per-class table", expanded=False):
            _render_markdown_table(
                per_class.get("classes", []),
                [
                    ("Class", "label"),
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1", "f1"),
                ],
            )
        with st.expander("Threshold table", expanded=False):
            threshold_rows = threshold_curve.get("rows", [])
            if threshold_rows:
                _render_markdown_table(
                    threshold_rows,
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
                st.info("No threshold curve data available.")
        with st.expander("Artifact inventory", expanded=False):
            _render_key_value_grid(
                [
                    ("Bundle files", inventory.get("bundle_artifact_count")),
                    ("Source artifacts", inventory.get("source_artifact_count")),
                    ("Selected threshold", recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
                ]
            )
            st.caption(
                "The evidence bundle remains governed; raw JSON is intentionally hidden by default."
            )
        with st.expander("Evidence file list", expanded=False):
            for filename in TRACK_A_EVIDENCE_FILENAMES:
                st.code(filename, language="text")
        with st.expander("Evidence paths", expanded=False):
            st.code("artifacts/frontend/track_a/", language="text")
            st.write("Selected model recommendation, comparison table, quality summary, and evidence inventory remain in the bundle.")

    st.markdown("### Safe interpretation")
    st.write(
        "ResNet18 v0.4.0 remains the governed Track A candidate, and this page stays in evidence/dashboard view only."
    )


def _render_track_b(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the Track B anomaly detection page."""
    st.markdown("## Track B Anomaly Detection")
    st.markdown(
        "A governed autoencoder-based anomaly page for reviewing reconstruction behavior, anomaly scores, and quality decision evidence."
    )
    st.warning(
        "Evidence/dashboard view only. not production-ready. not deployment-safe. "
        "No live prediction on this Track B evidence page. Track B upload/predict not implemented yet."
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

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric(
            "Model / autoencoder",
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
            "Quality/status",
            _format_value(quality.get("decision") or frontend_summary.get("canonical_status")),
            help=_format_value(frontend_summary.get("canonical_status") or metric_cards.get("canonical_status")),
        )
    with top_cols[3]:
        st.metric(
            "PR AUC",
            "Unavailable",
            help="PR AUC is unavailable in governed evidence and is not fabricated.",
        )

    tabs = st.tabs(["Summary", "Charts", "Evidence"])
    with tabs[0]:
        st.caption("Summary cards and the key gallery snapshot.")
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Canonical status", _format_value(frontend_summary.get("canonical_status") or metric_cards.get("canonical_status")))
        with summary_cols[1]:
            st.metric("Production ready", _format_value(quality.get("production_ready")))
        with summary_cols[2]:
            st.metric("Deployment candidate", _format_value(quality.get("deployment_candidate")))
        with summary_cols[3]:
            st.metric("Next step", _format_value(quality.get("next_recommended_step")))

        gallery_cols = st.columns(2)
        with gallery_cols[0]:
            st.metric("Gallery samples", _format_value(gallery.get("gallery_sample_count")))
            st.caption(
                "The gallery remains summary-only here; image rendering can be added later without changing the governed data contract."
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

        with st.expander("Metric cards", expanded=False):
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

        st.markdown("### Quality decision")
        _render_key_value_grid(
            [
                ("Model", frontend_summary.get("model_type")),
                ("Version", frontend_summary.get("model_version")),
                ("Threshold", frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")),
                ("Canonical status", frontend_summary.get("canonical_status") or metric_cards.get("canonical_status")),
                ("PR AUC", "Unavailable"),
            ]
        )
    with tabs[1]:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("Reconstruction loss")
            reconstruction_rows = reconstruction.get("chart_rows", [])
            if reconstruction_rows:
                epochs = [float(row.get("epoch", idx + 1) or idx + 1) for idx, row in enumerate(reconstruction_rows)]
                losses = [float(row.get("reconstruction_loss", 0) or 0) for row in reconstruction_rows]
                st.plotly_chart(
                    _build_line_figure(
                        "Track B reconstruction loss",
                        epochs,
                        {"Reconstruction loss": losses},
                        {"Reconstruction loss": "#1f77b4"},
                        "Loss",
                    ),
                    use_container_width=True,
                )
                st.caption("Epochs: " + ", ".join(_format_value(row.get("epoch")) for row in reconstruction_rows))
            else:
                st.info("No reconstruction loss data available for charting.")
            with st.expander("Reconstruction table", expanded=False):
                _render_markdown_table(
                    reconstruction_rows,
                    [
                        ("Epoch", "epoch"),
                        ("Reconstruction loss", "reconstruction_loss"),
                    ],
                )
        with chart_cols[1]:
            st.caption("Anomaly score summary")
            anomaly_rows = anomaly_summary.get("segments", [])
            if anomaly_rows:
                labels = [str(row.get("label", f"segment_{idx}")) for idx, row in enumerate(anomaly_rows)]
                values = [float(row.get("count", 0) or 0) for row in anomaly_rows]
                palette = []
                for label in labels:
                    lowered = label.lower()
                    if "anomaly" in lowered or "high" in lowered:
                        palette.append("#d62728")
                    elif "normal" in lowered or "low" in lowered:
                        palette.append("#2ca02c")
                    else:
                        palette.append("#1f77b4")
                st.plotly_chart(
                    _build_donut_figure(
                        "Track B anomaly score distribution",
                        labels,
                        values,
                        palette,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No anomaly score data available for charting.")

            st.caption("Threshold behavior")
            threshold_rows = threshold_behavior.get("rows", [])
            if threshold_rows:
                thresholds = [float(row.get("threshold", 0) or 0) for row in threshold_rows]
                series_map = {
                    "Precision": [float(row.get("precision", 0) or 0) for row in threshold_rows],
                    "Recall": [float(row.get("recall", 0) or 0) for row in threshold_rows],
                    "F1": [float(row.get("f1", 0) or 0) for row in threshold_rows],
                }
                st.plotly_chart(
                    _build_line_figure(
                        "Track B threshold behavior",
                        thresholds,
                        series_map,
                        {"Precision": "#1f77b4", "Recall": "#ff9800", "F1": "#2ca02c"},
                        "Score",
                    ),
                    use_container_width=True,
                )
                st.caption("Thresholds: " + ", ".join(_format_value(row.get("threshold")) for row in threshold_rows))
            else:
                st.info("No threshold behavior data available for charting.")
            with st.expander("Threshold table", expanded=False):
                _render_markdown_table(
                    threshold_rows,
                    [
                        ("Threshold", "threshold"),
                        ("Precision", "precision"),
                        ("Recall", "recall"),
                        ("F1", "f1"),
                        ("False Positives", "false_positive"),
                        ("False Negatives", "false_negative"),
                    ],
                )
    with tabs[2]:
        st.caption("Detailed evidence and raw tables are kept inside expanders.")
        st.caption("The anomaly score details are kept behind an expander.")
        st.caption("PR AUC is unavailable in governed evidence and not fabricated.")
        with st.expander("Metric cards", expanded=False):
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
        with st.expander("Anomaly score summary", expanded=False):
            anomaly_rows = anomaly_summary.get("segments", [])
            _render_markdown_table(
                anomaly_rows,
                [
                    ("Label", "label"),
                    ("Count", "count"),
                    ("Share", "percentage"),
                ],
            )
        with st.expander("Reconstruction table", expanded=False):
            _render_markdown_table(
                reconstruction.get("chart_rows", []),
                [
                    ("Epoch", "epoch"),
                    ("Reconstruction loss", "reconstruction_loss"),
                ],
            )
        with st.expander("Threshold table", expanded=False):
            _render_markdown_table(
                threshold_rows,
                [
                    ("Threshold", "threshold"),
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1", "f1"),
                    ("False Positives", "false_positive"),
                    ("False Negatives", "false_negative"),
                ],
            )
        with st.expander("Sample anomaly gallery details", expanded=False):
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
        with st.expander("Artifact inventory", expanded=False):
            _render_key_value_grid(
                [
                    ("Bundle files", inventory.get("bundle_artifact_count")),
                    ("Source artifacts", inventory.get("source_artifact_count")),
                    ("Threshold", frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")),
                ]
            )
        with st.expander("Evidence file list", expanded=False):
            for filename in TRACK_B_EVIDENCE_FILENAMES:
                st.code(filename, language="text")
        with st.expander("Evidence paths", expanded=False):
            st.code("artifacts/frontend/track_b/", language="text")
            st.write("Metric cards, anomaly summary, reconstruction loss, threshold behavior, quality summary, and inventory remain in the bundle.")

    st.markdown("### Safe interpretation")
    st.write(
        "The Track B autoencoder remains governed evidence only, with PR AUC unavailable in the current evidence set."
    )


def _render_yolo(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the YOLO / Detection page."""
    st.markdown("## YOLO Detection")
    st.markdown(
        "A governed object-detection evidence page for the validation bundle, designed for review rather than live deployment."
    )
    st.warning(
        "Evidence/dashboard view only. not production-ready. not deployment-safe. "
        "No live prediction on this YOLO evidence page. YOLO upload/predict not implemented yet."
    )
    st.caption("evidence/dashboard view only")

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

    st.caption(
        "Run yolo_train_v0_2_0 | "
        f"Model: {_format_value(metadata.get('model_name'))} | "
        f"Version: {_format_value(metadata.get('model_version'))} | "
        f"Dataset: {_format_value(metadata.get('dataset_id'))} | "
        f"Split: {_format_value(metadata.get('split'))}"
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("Images", _format_value(overview.get("image_count")))
        st.caption("Validation images in the governed bundle")
    with top_cols[1]:
        st.metric("Total bboxes", _format_value(overview.get("total_bbox_count")))
        st.caption("All predicted boxes counted across the split")
    with top_cols[2]:
        st.metric("Review decision", _format_value(quality.get("decision")))
        st.caption(
            f"Review required: {_format_value(quality.get('review_required'))} | "
            f"Production ready: {_format_value(quality.get('production_ready'))}"
        )
    with top_cols[3]:
        st.metric(
            "Model / run",
            f"{_format_value(metadata.get('model_name'))} {_format_value(metadata.get('model_version'))}",
            help=f"Run { _format_value(metadata.get('run_id')) }",
        )
        st.caption("Governed validation evidence only")

    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown("### What the bundle shows")
        st.write(
            "The YOLO bundle summarizes detection counts, confidence bands, class balance, and curated gallery samples for review."
        )
        st.metric(
            "Images with detections",
            _format_value(overview.get("image_with_detections_count")),
            help=_format_value(overview.get("safe_summary")),
        )
    with summary_cols[1]:
        st.markdown("### Sample gallery snapshot")
        gallery_counts = sample_gallery.get("category_sample_counts", {})
        gallery_snapshot_cols = st.columns(4)
        snapshot_items = [
            ("Gallery samples", sample_gallery.get("gallery_sample_count")),
            ("No detections", gallery_counts.get("no_detection_examples")),
            ("Multi detections", gallery_counts.get("multi_detection_examples")),
            ("High confidence", gallery_counts.get("high_confidence_examples")),
        ]
        for col, (label, value) in zip(gallery_snapshot_cols, snapshot_items):
            with col:
                st.metric(label, _format_value(value))

    st.markdown("### Visual evidence")
    visual_cols = st.columns(2)
    with visual_cols[0]:
        confidence_rows = confidence_chart.get("confidence_bins", [])
        if confidence_rows:
            confidence_labels = [row.get("label", f"bin_{idx}") for idx, row in enumerate(confidence_rows)]
            confidence_values = [float(row.get("count", 0) or 0) for row in confidence_rows]
            confidence_colors = ["#1f77b4", "#ff9800", "#2ca02c", "#9e9e9e"][: len(confidence_labels)]
            st.plotly_chart(
                _build_donut_figure(
                    confidence_chart.get("chart_title", "Confidence distribution"),
                    confidence_labels,
                    confidence_values,
                    confidence_colors,
                ),
                use_container_width=True,
            )
            st.caption(
                confidence_chart.get("chart_explanation")
                or "Confidence distribution across predicted boxes."
            )
        else:
            st.info("No confidence distribution data available for charting.")
    with visual_cols[1]:
        class_rows = class_summary.get("class_rows", [])
        if class_rows:
            class_labels = [row.get("class_label", f"class_{idx}") for idx, row in enumerate(class_rows)]
            class_counts = [float(row.get("bbox_count", 0) or 0) for row in class_rows]
            st.plotly_chart(
                _build_grouped_bar_figure(
                    "Class summary by bbox count",
                    class_labels,
                    {"BBox count": class_counts},
                    {"BBox count": "#1f77b4"},
                    "BBox count",
                ),
                use_container_width=True,
            )
            st.caption("Bar heights represent bbox counts per class.")
        else:
            st.info("No class summary data available for charting.")

    st.markdown("### Decision summary")
    decision_cols = st.columns(2)
    with decision_cols[0]:
        st.metric("Review status", _format_value(quality.get("decision")))
        st.caption(
            f"Production ready: {_format_value(quality.get('production_ready'))} | "
            f"Deployment candidate: {_format_value(quality.get('deployment_candidate'))}"
        )
        st.caption(_format_value(quality.get("next_recommended_step")))
    with decision_cols[1]:
        st.metric("Frontend recommendation", _format_value(recommendation.get("recommendation_status")))
        st.caption(_format_value(recommendation.get("next_step")))
        st.caption(
            "What it can claim: "
            + ", ".join(
                _format_value(value)
                for value in recommendation.get("what_it_can_claim", [])
            )
        )

    with st.expander("Technical evidence", expanded=False):
        tech_tabs = st.tabs(["Metric cards", "Tables", "Artifact lineage", "Run notes"])
        with tech_tabs[0]:
            st.caption("Metric cards for the governed YOLO evidence bundle")
            cards = metric_cards.get("cards", [])
            if cards:
                for start_idx in range(0, len(cards), 2):
                    row_cards = cards[start_idx : start_idx + 2]
                    cols = st.columns(len(row_cards))
                    for col, card in zip(cols, row_cards):
                        with col:
                            st.metric(card.get("label", "Metric"), _format_value(card.get("value")))
                            description = card.get("description")
                            if description:
                                st.caption(str(description))
            else:
                st.info("No detection metric cards available.")

        with tech_tabs[1]:
            table_cols = st.columns(2)
            with table_cols[0]:
                st.caption("Confidence distribution table")
                confidence_rows = confidence_chart.get("confidence_bins", [])
                _render_markdown_table(
                    confidence_rows,
                    [
                        ("Band", "label"),
                        ("Count", "count"),
                        ("Share", "percentage"),
                    ],
                )
            with table_cols[1]:
                st.caption("Class summary table")
                class_rows = class_summary.get("class_rows", [])
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

            st.markdown("##### Sample gallery details")
            gallery_rows = [
                {
                    "category": category.get("category_label"),
                    "samples": category.get("sample_count"),
                    "selection_rule": category.get("selection_rule"),
                }
                for category in sample_gallery.get("categories", [])
                if isinstance(category, dict)
            ]
            _render_markdown_table(
                gallery_rows,
                [
                    ("Category", "category"),
                    ("Samples", "samples"),
                    ("Selection rule", "selection_rule"),
                ],
            )

        with tech_tabs[2]:
            st.caption("artifact lineage")
            st.caption("Artifact lineage and bundle files for the detection evidence layer")
            source_rows = [
                {
                    "artifact": item.get("artifact_type") or item.get("artifact_id"),
                    "path": item.get("artifact_path") or item.get("path"),
                    "hash": item.get("artifact_hash") or item.get("sha256"),
                }
                for item in lineage.get("source_artifacts", [])
                if isinstance(item, dict)
            ]
            _render_markdown_table(
                source_rows,
                [
                    ("Artifact", "artifact"),
                    ("Path", "path"),
                    ("Hash", "hash"),
                ],
            )
            st.caption("bundle manifest")
            st.markdown("##### Bundle manifest")
            st.code("artifacts/frontend/detection/yolo_train_v0_2_0/", language="text")
            _render_markdown_table(
                [{"file": name} for name in manifest.get("bundle_files", [])],
                [("File", "file")],
            )
            st.markdown("##### Evidence file list")
            for filename in DETECTION_EVIDENCE_FILENAMES:
                st.code(filename, language="text")

        with tech_tabs[3]:
            _render_key_value_grid(
                [
                    ("Run", metadata.get("run_id")),
                    ("Dataset", metadata.get("dataset_id")),
                    ("Review required", quality.get("review_required")),
                    ("Production ready", quality.get("production_ready")),
                    ("Deployment candidate", quality.get("deployment_candidate")),
                ]
            )
            st.markdown("##### Quality decision")
            st.write(
                f"{_format_value(quality.get('decision'))} | "
                f"{_format_value(quality.get('next_recommended_step'))}"
            )
            st.caption("frontend recommendation")
            st.markdown("##### Frontend recommendation")
            st.write(
                f"{_format_value(recommendation.get('recommendation_status'))} | "
                f"{_format_value(recommendation.get('next_step'))}"
            )
            st.caption(
                "What it cannot claim: "
                + ", ".join(
                    _format_value(value)
                    for value in recommendation.get("what_it_cannot_claim", [])
                )
            )

    st.markdown("### Safe interpretation")
    st.write(
        "This YOLO / Detection evidence layer is review-oriented only and does not claim production readiness or deployment safety."
    )


def _render_upload_predict() -> None:
    """Render the Track A upload/predict page."""
    st.markdown("## Upload / Predict")
    st.markdown(
        "A local Track A classification prototype that sends one uploaded image to the FastAPI endpoint and returns the governed prediction result."
    )
    st.warning(
        "Local prototype endpoint for Track A classification only. not production-ready. "
        "not deployment-safe. YOLO and anomaly upload/predict not implemented yet."
    )
    st.caption("Track A classification only")

    step_cols = st.columns(3)
    with step_cols[0]:
        st.metric("Step 1", "Choose image")
        st.caption("PNG, JPG, JPEG, or WEBP")
    with step_cols[1]:
        st.metric("Step 2", "Confirm API URL")
        st.caption("Default: http://localhost:8000")
    with step_cols[2]:
        st.metric("Step 3", "Run prediction")
        st.caption("Track A classification only")

    controls_cols = st.columns([1.2, 1])
    with controls_cols[0]:
        api_base_url = st.text_input(
            "API base URL",
            value=API_DEFAULT_BASE_URL,
            help="Base URL for POST /predict/classification.",
        )
    with controls_cols[1]:
        st.caption("Connection target")
        st.code(f"POST {api_base_url.rstrip('/')}/predict/classification", language="text")

    uploaded_file = st.file_uploader(
        "Choose an image for Track A classification",
        type=list(UPLOAD_ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        key="track_a_upload_file",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        preview_cols = st.columns([1, 1.1])
        with preview_cols[0]:
            st.markdown("### Image preview")
            st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)
            st.caption(
                f"{uploaded_file.name} | "
                f"{uploaded_file.type or _infer_content_type(uploaded_file.name)} | "
                f"{len(file_bytes)} bytes"
            )
        with preview_cols[1]:
            st.markdown("### Ready to predict")
            _render_key_value_grid(
                [
                    ("Filename", uploaded_file.name),
                    ("Content type", uploaded_file.type or _infer_content_type(uploaded_file.name)),
                    ("File size bytes", len(file_bytes)),
                ]
            )
            st.caption("This preview is local only and does not change the API contract.")
    else:
        st.info("Choose a single image to enable the prediction button.")

    predict_clicked = st.button(
        "Predict Track A classification",
        type="primary",
        disabled=uploaded_file is None,
    )

    if predict_clicked and uploaded_file is not None:
        try:
            payload = _call_classification_api(api_base_url, uploaded_file)
            st.session_state["track_a_upload_prediction"] = payload
            st.session_state["track_a_upload_error"] = None
        except Exception as exc:  # pragma: no cover - UI boundary
            st.session_state["track_a_upload_prediction"] = None
            st.session_state["track_a_upload_error"] = str(exc)

    error_message = st.session_state.get("track_a_upload_error")
    if error_message:
        st.error(error_message)

    payload = st.session_state.get("track_a_upload_prediction")
    if not isinstance(payload, dict):
        st.info("Upload an image and press Predict to call the Track A classification endpoint.")
        return

    st.success("Prediction completed.")

    result_cols = st.columns([1.1, 1])
    with result_cols[0]:
        st.markdown("### Prediction result")
        visual_cols = st.columns(2)
        with visual_cols[0]:
            st.metric("Predicted label", _safe_text(payload.get("predicted_label")))
        with visual_cols[1]:
            st.metric("Decision", _safe_text(payload.get("decision")))
        probability_good = float(payload.get("probability_good") or 0.0)
        probability_defect = float(payload.get("probability_defect") or 0.0)
        st.plotly_chart(
            _build_donut_figure(
                "Prediction confidence",
                ["Good", "Defect"],
                [probability_good, probability_defect],
                ["#2ca02c", "#d62728"],
            ),
            use_container_width=True,
        )
        confidence_cols = st.columns(2)
        with confidence_cols[0]:
            st.metric("Probability good", _format_value(payload.get("probability_good")))
        with confidence_cols[1]:
            st.metric("Probability defect", _format_value(payload.get("probability_defect")))
        st.caption("The chart shows how the model split confidence between good and defect.")

    with result_cols[1]:
        st.markdown("### Model summary")
        _render_key_value_grid(
            [
                ("Model", payload.get("model_name")),
                ("Version", payload.get("model_version")),
                ("Threshold", payload.get("threshold")),
                ("Production ready", payload.get("production_ready")),
                ("Deployment safe", payload.get("deployment_safe")),
            ]
        )
        st.caption(
            "not production-ready | not deployment-safe | local prototype endpoint | Track A classification only"
        )

    with st.expander("Technical details", expanded=False):
        detail_cols = st.columns(2)
        with detail_cols[0]:
            _render_key_value_grid(
                [
                    ("Request ID", payload.get("request_id")),
                    ("Run ID", payload.get("run_id")),
                    ("Model name", payload.get("model_name")),
                    ("Model version", payload.get("model_version")),
                    ("Threshold", payload.get("threshold")),
                ]
            )
        with detail_cols[1]:
            input_meta = payload.get("input", {})
            _render_key_value_grid(
                [
                    ("Filename", input_meta.get("filename")),
                    ("Content type", input_meta.get("content_type")),
                    ("File size bytes", input_meta.get("file_size_bytes")),
                ]
            )
            limitations = payload.get("limitations", [])
            if limitations:
                st.caption("Limitations")
                st.write(" | ".join(str(item) for item in limitations))
        st.markdown("##### Raw API response")
        st.json(payload)

    st.caption(
        "This page is limited to Track A classification. YOLO and anomaly upload/predict are not implemented yet."
    )


def _render_limitations() -> None:
    """Render the limitations and safety page."""
    st.subheader("Limitations / Safety")
    st.write(
        "This scaffold is evidence-focused only. It does not train models, recompute metrics, "
        "create artifacts, update registries, or claim production or deployment readiness."
    )


def _call_classification_api(api_base_url: str, uploaded_file: Any) -> dict[str, Any]:
    """Call the Track A classification API with a local uploaded image."""
    normalized_base_url = api_base_url.strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("API base URL must be a non-empty string.")

    file_name = getattr(uploaded_file, "name", None) or "upload.png"
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("Uploaded image is empty.")
    if len(file_bytes) > UPLOAD_MAX_BYTES:
        raise ValueError(
            f"Uploaded image exceeds the {UPLOAD_MAX_BYTES} byte limit."
        )

    content_type = getattr(uploaded_file, "type", None) or _infer_content_type(file_name)
    if content_type not in UPLOAD_ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type: {content_type!r}. "
            f"Allowed types are: {', '.join(sorted(UPLOAD_ALLOWED_CONTENT_TYPES))}."
        )

    endpoint = f"{normalized_base_url}/predict/classification"
    try:
        response = requests.post(
            endpoint,
            files={"file": (file_name, file_bytes, content_type)},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise ConnectionError(f"Could not reach the API at {normalized_base_url}: {exc}") from exc

    if response.status_code >= 400:
        detail = _extract_api_error_detail(response)
        raise RuntimeError(f"API request failed with status {response.status_code}: {detail}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("API returned a non-JSON response.") from exc

    if not isinstance(payload, dict):
        raise ValueError("API returned an invalid response payload.")

    required_keys = {
        "request_id",
        "model_name",
        "model_version",
        "run_id",
        "threshold",
        "predicted_label",
        "predicted_label_id",
        "probability_good",
        "probability_defect",
        "decision",
        "production_ready",
        "deployment_safe",
        "input",
        "live_prediction_enabled",
        "upload_predict_enabled",
        "limitations",
    }
    missing_keys = sorted(key for key in required_keys if key not in payload)
    if missing_keys:
        raise ValueError(
            "API response is missing required fields: " + ", ".join(missing_keys)
        )

    return payload


def _extract_api_error_detail(response: requests.Response) -> str:
    """Return a readable error message from an API response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "No error details were returned."

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        return str(payload)
    return str(payload)


def _infer_content_type(filename: str) -> str:
    """Infer an accepted image content type from the filename."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


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

"""Streamlit entrypoint for the frontend dashboard shell.

This dashboard presents the current project status as a read-only evidence
shell. It is intentionally limited to local evidence presentation and does
not implement live prediction, API integration, Docker, or production
features.
"""

from __future__ import annotations

import mimetypes
import html
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.data_loader import load_all_frontend_bundles


PROJECT_TITLE = "Industrial Surface Defect Inspection Platform"
API_DEFAULT_BASE_URL = "http://localhost:8000"
UPLOAD_ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "webp")
UPLOAD_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
UPLOAD_MAX_BYTES = 10 * 1024 * 1024

OVERVIEW_PAGE_LABEL = "Overview"
SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL = "Surface Defect Classification"
SURFACE_ANOMALY_DETECTION_PAGE_LABEL = "Surface Anomaly Detection"
DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL = "Defect Detection & Localization"
IMAGE_INSPECTION_PAGE_LABEL = "Image Inspection"
SAFETY_LIMITATIONS_PAGE_LABEL = "Safety & Limitations"
AI_EXPLANATION_ASSISTANT_PAGE_LABEL = "AI Explanation Assistant"
INSPECTION_CAPABILITY_SUMMARY_LABEL = "Recommended review path"

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
    "sample_predictions.json",
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
    ("Surface Defect Classification", "PASS"),
    ("Surface Anomaly Detection", "PASS"),
    ("Defect Detection & Localization", "COMPLETE"),
    ("Image Inspection", "LOCAL WORKFLOW"),
    ("Frontend app", "NOT STARTED / NOT VALIDATED"),
    ("API endpoints", "NOT STARTED"),
    ("AI Explanation Assistant", "NOT STARTED"),
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


def _friendly_status_label(value: Any, default: str = "Unavailable") -> str:
    """Return a user-facing label for raw status values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"

    text = str(value).strip()
    lowered = text.lower()
    status_map = {
        "review_required": "Needs review",
        "review_required_weak_evidence": "Review required: weak evidence",
        "review_only_signal": "Review-only supporting signal",
        "frontend_bundle_ready_for_review": "Frontend evidence bundle ready for review",
    }
    if lowered in status_map:
        return status_map[lowered]
    if "strong_track_a_candidate" in lowered or "track_a_strong_candidate" in lowered:
        return "Strong classification candidate"
    if "production-canonical" in lowered:
        return "Governed review evidence"
    return text.replace("_", " ")


def _friendly_metric_display(value: Any) -> str:
    """Return a user-facing metric value while normalizing raw governance statuses."""
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered == "selected track a candidate, not production-ready.":
            return "Selected governed classification candidate; local review/demo only, not production use."
        if "track_a_strong_candidate" in lowered or "strong_track_a_candidate" in lowered:
            return "Strong classification candidate"
        if "track a" in lowered and "candidate" in lowered and "production" in lowered:
            return "Selected governed classification candidate; local review/demo only, not production use."
        if "track a" in lowered or "track b" in lowered:
            return text.replace("Track A", "classification").replace("Track B", "anomaly")
        return _friendly_status_label(text)
    return _safe_text(value)


def _friendly_decision_label(value: Any, default: str = "Unavailable") -> str:
    """Return a user-facing label for final decision values."""
    if value is None:
        return default
    text = str(value).strip()
    mapping = {
        "good": "Good",
        "defective": "Defective",
        "anomalous": "Anomalous",
        "needs_manual_review": "Needs manual review",
        "inconclusive": "Inconclusive",
    }
    return mapping.get(text.lower(), text.replace("_", " "))


def _format_probability(value: Any) -> str:
    """Format a probability for user-facing display."""
    if value is None:
        return "Unavailable"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric in (0.0, 1.0):
        return f"{numeric:.0f}"
    return f"{numeric:.2%}"


def _lookup_card_value(cards: list[dict[str, Any]], title: str) -> Any:
    """Return the value for a metric card with the requested title."""
    for card in cards:
        if not isinstance(card, dict):
            continue
        if str(card.get("title", "")).strip().lower() == title.strip().lower():
            return card.get("value")
    return None


def _extract_pr_auc(frontend_summary: dict[str, Any], metric_cards: dict[str, Any]) -> Any:
    """Return the governed PR AUC value from the anomaly bundle."""
    pr_auc_value = frontend_summary.get("key_metrics", {}).get("pr_auc")
    if pr_auc_value is not None:
        return pr_auc_value
    return _lookup_card_value(metric_cards.get("cards", []), "PR AUC")


def _extract_histogram_series(histograms: dict[str, Any]) -> tuple[list[str], dict[str, list[float]]]:
    """Return shared histogram labels and series counts from a governed bundle."""
    if not isinstance(histograms, dict):
        return [], {}

    label_source = histograms.get("all")
    if not isinstance(label_source, list) or not label_source:
        return [], {}

    labels = [
        f"{row.get('bin_start', 0):.3f}-{row.get('bin_end', 0):.3f}"
        for row in label_source
        if isinstance(row, dict)
    ]
    series_map: dict[str, list[float]] = {}
    for series_name in ("all", "true_normal", "true_anomaly"):
        rows = histograms.get(series_name, [])
        if isinstance(rows, list) and rows:
            series_map[series_name.replace("_", " ").title()] = [
                float(row.get("count", 0) or 0) for row in rows if isinstance(row, dict)
            ]
    return labels, series_map


def _extract_threshold_rows(threshold_behavior: dict[str, Any]) -> list[dict[str, Any]]:
    """Return threshold sweep rows from the anomaly bundle."""
    rows = threshold_behavior.get("rows", [])
    return rows if isinstance(rows, list) else []


def _extract_sample_prediction_rows(sample_predictions: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sample-level anomaly prediction rows."""
    samples = sample_predictions.get("samples", [])
    return samples if isinstance(samples, list) else []


def _extract_detection_rows(detection_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return detection box rows from a unified inspection response."""
    detections = detection_payload.get("detections", [])
    return detections if isinstance(detections, list) else []


def _annotate_detection_boxes(image: Image.Image, detection_payload: dict[str, Any]) -> Image.Image:
    """Return a copy of the uploaded image annotated with detection boxes."""
    annotated = image.copy().convert("RGB")
    detections = _extract_detection_rows(detection_payload)
    if not detections:
        return annotated

    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    base_width, base_height = annotated.size
    source_width = int(detection_payload.get("image_width") or base_width or 1)
    source_height = int(detection_payload.get("image_height") or base_height or 1)
    scale_x = base_width / source_width if source_width else 1.0
    scale_y = base_height / source_height if source_height else 1.0
    colors = ["#38bdf8", "#f97316", "#22c55e", "#a78bfa", "#ef4444"]

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue
        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        color = colors[index % len(colors)]
        outline_width = max(2, min(base_width, base_height) // 240)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)

        label = str(detection.get("display_label") or detection.get("class_label") or "Detection")
        confidence = detection.get("confidence")
        if confidence is not None:
            label = f"{label} {_format_probability(confidence)}"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = max(0.0, x1)
        text_y = max(0.0, y1 - text_height - 6)
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 8, text_y + text_height + 6],
            fill=color,
        )
        draw.text((text_x + 4, text_y + 3), label, fill="#0f172a", font=font)

    return annotated


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
        "The frontend is still being completed as a governed inspection workflow."
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


def _apply_light_visual_system() -> None:
    """Apply a soft dark navy / petroleum blue visual system for the premium dashboard."""
    st.markdown(
        """
        <style>
        :root {
            --premium-blue: #38bdf8;
            --premium-teal: #14b8a6;
            --premium-green: #22c55e;
            --premium-orange: #fb923c;
            --premium-red: #fb7185;
            --premium-violet: #a78bfa;
            --premium-gray: #94a3b8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(167, 139, 250, 0.10), transparent 20%),
                radial-gradient(circle at center, rgba(20, 184, 166, 0.08), transparent 18%),
                linear-gradient(180deg, #08111d 0%, #0b1728 42%, #0f1f33 100%);
            color: #e2e8f0;
        }
        main .block-container {
            max-width: 1240px;
            padding-top: 1.3rem;
            padding-bottom: 2.2rem;
        }
        header[data-testid="stHeader"] {
            background: linear-gradient(180deg, rgba(6, 12, 22, 0.92), rgba(8, 17, 29, 0.72));
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        }
        div[data-testid="stToolbar"] {
            background: rgba(8, 17, 29, 0.7);
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(9, 17, 30, 0.98) 0%, rgba(11, 23, 40, 0.98) 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-top: 0.25rem;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            display: block;
            margin-bottom: 0.35rem;
            padding: 0.48rem 0.72rem;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.12);
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(30, 41, 59, 0.92);
            border-color: rgba(56, 189, 248, 0.28);
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
            background: rgba(16, 37, 58, 0.98);
            border-color: rgba(56, 189, 248, 0.36);
            box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.18);
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            color: #ffffff !important;
        }
        div[data-testid="stMetric"] {
            background:
                linear-gradient(180deg, rgba(12, 20, 33, 0.96) 0%, rgba(18, 28, 46, 0.94) 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-top: 5px solid var(--premium-blue);
            border-radius: 20px;
            padding: 0.85rem 1rem;
            box-shadow: 0 18px 36px rgba(2, 6, 23, 0.3);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] p,
        div[data-testid="stMetric"] span {
            color: #cbd5e1 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 700;
            font-size: 1.18rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            text-overflow: clip !important;
            overflow: visible !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background: rgba(8, 15, 28, 0.72);
            box-shadow: 0 12px 26px rgba(2, 6, 23, 0.2);
        }
        div[data-testid="stPlotlyChart"] {
            background: rgba(9, 16, 29, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 0.55rem 0.6rem;
            box-shadow: 0 16px 32px rgba(2, 6, 23, 0.24);
        }
        div[data-testid="stRadio"] label {
            color: #e2e8f0 !important;
            font-weight: 600;
        }
        .premium-card {
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-top: 6px solid var(--premium-blue);
            background:
                linear-gradient(180deg, rgba(11, 20, 35, 0.99), rgba(17, 29, 48, 0.96));
            box-shadow: 0 22px 46px rgba(2, 6, 23, 0.34);
            padding: 1rem 1rem 0.95rem;
            margin-bottom: 0.9rem;
        }
        .premium-card--blue { border-top-color: var(--premium-blue); }
        .premium-card--teal { border-top-color: var(--premium-teal); }
        .premium-card--green { border-top-color: var(--premium-green); }
        .premium-card--orange { border-top-color: var(--premium-orange); }
        .premium-card--red { border-top-color: var(--premium-red); }
        .premium-card--violet { border-top-color: var(--premium-violet); }
        .premium-card--gray { border-top-color: var(--premium-gray); }
        .premium-card__eyebrow {
            color: #93c5fd;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .premium-card__title {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }
        .premium-card__title--subtle {
            font-size: 1.0rem;
            font-weight: 700;
        }
        .premium-card__body {
            color: #eef4fb;
            font-size: 0.96rem;
            line-height: 1.55;
        }
        .premium-card__meta {
            color: #cbd5e1;
            font-size: 0.8rem;
            margin-top: 0.55rem;
        }
        .premium-pill {
            display: inline-block;
            background: rgba(56, 189, 248, 0.14);
            color: #ccefff;
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.7rem;
            font-weight: 700;
            margin-right: 0.35rem;
        }
        div[data-testid="stAlert"] {
            background: rgba(8, 15, 28, 0.9);
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
        }
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] div {
            color: #e2e8f0 !important;
        }
        .stCaption, .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: #dbe4f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_health(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render compact bundle health in the sidebar."""
    with st.sidebar.expander("Data Load Health", expanded=False):
        if bundles is None:
            st.error("Frontend data contracts are not fully loadable.")
            return
        st.caption("Loaded JSON files per governed bundle")
        for key, label in (
            ("track_a", "Classification bundle"),
            ("track_b", "Anomaly bundle"),
            ("detection", "Detection bundle"),
        ):
            bundle = bundles[key]
            st.metric(label, f"{len(bundle)} files")
    st.sidebar.caption("Current page is highlighted in the navigation above.")


def _render_hero_card(title: str, subtitle: str, safety: str, accent: str = "blue") -> None:
    """Render a premium hero card for a page."""
    st.markdown(
        f"""
        <div class="premium-card premium-card--{accent}">
            <div class="premium-card__eyebrow">Premium dashboard</div>
            <div class="premium-card__title">{html.escape(title)}</div>
            <div class="premium-card__body">{html.escape(subtitle)}</div>
            <div class="premium-card__meta">{html.escape(safety)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_agent_placeholder(
    action_label: str,
    summary: str,
    note: str,
    *,
    key: str,
    accent: str = "violet",
) -> None:
    """Render a visible but non-functional agent placeholder."""
    st.markdown(
        f"""
        <div class="premium-card premium-card--{accent}">
            <div class="premium-card__eyebrow">Agent layer planned</div>
            <div class="premium-card__title">{html.escape(action_label)}</div>
            <div class="premium-card__body">{html.escape(summary)}</div>
            <div class="premium-card__meta">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(action_label, disabled=True, key=key)


def _render_agent_callout(action_label: str, summary: str, note: str, *, accent: str = "violet") -> None:
    """Render a compact premium agent callout without an action button."""
    st.markdown(
        f"""
        <div class="premium-card premium-card--{accent}" style="margin-top:0.35rem; padding: 1.15rem 1.2rem 1.05rem;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:0.75rem; margin-bottom:0.45rem;">
                <div class="premium-card__eyebrow" style="margin:0;">Future AI explanation</div>
                <span style="display:inline-flex; align-items:center; padding:0.22rem 0.7rem; border-radius:999px; background:rgba(124,58,237,0.16); color:#f5d0fe; border:1px solid rgba(192,132,252,0.28); font-size:0.72rem; font-weight:700; letter-spacing:0.02em; white-space:nowrap;">Planned / not active</span>
            </div>
            <div class="premium-card__title" style="font-size:1.15rem; line-height:1.2; margin-bottom:0.55rem;">{html.escape(action_label)}</div>
            <div class="premium-card__body">{html.escape(summary)}</div>
            <div class="premium-card__meta">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_premium_info_card(title: str, summary: str, note: str = "", accent: str = "blue") -> None:
    """Render a compact premium information card."""
    st.markdown(
        f"""
        <div class="premium-card premium-card--{accent}">
            <div class="premium-card__eyebrow">Dashboard note</div>
            <div class="premium-card__title">{html.escape(title)}</div>
            <div class="premium-card__body">{html.escape(summary)}</div>
            <div class="premium-card__meta">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chart_placeholder(title: str, message: str, accent: str = "gray") -> None:
    """Render a visible placeholder when chart data is unavailable."""
    st.markdown(
        f"""
        <div class="premium-card premium-card--{accent}">
            <div class="premium-card__eyebrow">Visual unavailable</div>
            <div class="premium-card__title">{html.escape(title)}</div>
            <div class="premium-card__body">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        title=dict(text=title, font=dict(color="#f8fafc", size=18)),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
        font=dict(color="#e2e8f0", family="Inter, Segoe UI, Arial"),
        legend=dict(font=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(8, 15, 28, 0.02)",
        plot_bgcolor="rgba(8, 15, 28, 0.18)",
        template="plotly_dark",
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
        title=dict(text=title, font=dict(color="#f8fafc", size=18)),
        height=320,
        barmode="group",
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Metric",
        yaxis_title=yaxis_title,
        font=dict(color="#e2e8f0", family="Inter, Segoe UI, Arial"),
        legend=dict(font=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(8, 15, 28, 0.02)",
        plot_bgcolor="rgba(8, 15, 28, 0.18)",
        template="plotly_dark",
    )
    figure.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.14)",
        tickfont=dict(color="#dbe4f0"),
        zerolinecolor="rgba(148, 163, 184, 0.12)",
    )
    figure.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.14)",
        tickfont=dict(color="#dbe4f0"),
        zerolinecolor="rgba(148, 163, 184, 0.12)",
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
        title=dict(text=title, font=dict(color="#f8fafc", size=18)),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Metric",
        yaxis_title=yaxis_title,
        xaxis_title="Threshold",
        font=dict(color="#e2e8f0", family="Inter, Segoe UI, Arial"),
        legend=dict(font=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(8, 15, 28, 0.02)",
        plot_bgcolor="rgba(8, 15, 28, 0.18)",
        template="plotly_dark",
    )
    figure.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.14)",
        tickfont=dict(color="#dbe4f0"),
        zerolinecolor="rgba(148, 163, 184, 0.12)",
    )
    figure.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.14)",
        tickfont=dict(color="#dbe4f0"),
        zerolinecolor="rgba(148, 163, 184, 0.12)",
    )
    return figure


def _render_overview_review_path() -> None:
    """Render a reviewer-oriented path for the current system."""
    st.caption("A short path for reviewing the current dashboard and its governed evidence.")
    review_steps = [
        (
            "1. Review classification evidence",
            "This page shows the threshold, confusion matrix, false positives, false negatives, and selected classifier evidence.",
            "Use the detailed metrics and technical evidence panels to review the governed classification package.",
            "blue",
        ),
        (
            "2. Review anomaly evidence",
            "This page shows anomaly score, reconstruction loss, PR AUC, threshold behavior, and the review-only anomaly boundary.",
            "Use the anomaly validation metrics and technical evidence panels to review governed anomaly evidence.",
            "orange",
        ),
        (
            "3. Review detection & localization evidence",
            "This page shows confidence distribution, class summary, detection counts, and validation evidence.",
            "Use the detection summary and technical evidence panels to review governed localization evidence.",
            "green",
        ),
        (
            "4. Run Image Inspection",
            "Upload one image to see the final decision, classification output, localization boxes, anomaly signal, warnings, limitations, and traceability.",
            "Use the live inspection page for the unified /inspect/image workflow.",
            "teal",
        ),
        (
            "5. Check Safety & Limitations",
            "This page clarifies production and deployment boundaries, the manual review requirement, and what the dashboard does not claim.",
            "Use the safety page to review the system boundary before any broader use.",
            "gray",
        ),
    ]
    for title, summary, note, accent in review_steps:
        _render_premium_info_card(title, summary, note, accent=accent)


def _render_not_claimed_note(summary: str) -> None:
    """Render a short caption explaining the not-claimed readiness state."""
    st.caption(summary)


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
    _render_hero_card(
        PROJECT_TITLE,
        "A governed evidence dashboard for reviewing surface defect classification, surface anomaly detection, defect detection & localization, unified image inspection, and the rule-based decision layer.",
        "Governed evidence only · not production-ready · not deployment-safe",
        accent="blue",
    )
    with st.expander("Future AI explanation", expanded=False):
        st.caption("Planned / not active.")
        st.write(
            "No backend agent is implemented yet, no LLM call is wired, and future explanations must stay grounded in governed evidence and real inspection responses."
        )

    if bundles is None:
        st.error("Overview data is unavailable because one or more frontend bundles failed to load.")
        return

    track_a = bundles["track_a"]
    track_b = bundles["track_b"]
    detection = bundles["detection"]
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Classification evidence", "Available", help="Governed classification evidence available")
    with summary_cols[1]:
        st.metric("Anomaly evidence", "Available / review-only signal", help="Governed anomaly evidence and quality decision available")
    with summary_cols[2]:
        st.metric("Detection evidence", "Available", help="Governed detection evidence available")
    with summary_cols[3]:
        st.metric("Image Inspection", "Connected", help="Unified local inspection endpoint is registered and wired")

    capability_cols = st.columns(3)
    with capability_cols[0]:
        st.metric("Decision layer", "Available", help="Deterministic rule-based aggregation is implemented")
    with capability_cols[1]:
        st.metric("Production readiness", "Not claimed", help="The dashboard does not claim production readiness")
        _render_not_claimed_note("Not claimed means local review/demo only, not factory production use.")
    with capability_cols[2]:
        st.metric("Deployment readiness", "Not claimed", help="The dashboard does not claim deployment safety")
        _render_not_claimed_note("Not claimed means Docker/release validation is still pending.")

    overview_cols = st.columns([1.15, 0.95])
    with overview_cols[0]:
        st.markdown("### What you can review")
        st.write(
            "- Review governed validation evidence for surface defect classification, surface anomaly detection, and defect detection & localization\n"
            "- Run local image inspection through the unified inspection endpoint\n"
            "- View classification output, localization boxes, anomaly signal, final rule-based decision, warnings, errors, limitations, traceability, and explanation context"
        )
        st.markdown("### What this dashboard does not claim")
        st.write(
            "- It is not production-ready.\n"
            "- It is not deployment-safe.\n"
            "- It does not replace expert/manual inspection.\n"
            "- It does not contain an active AI agent yet.\n"
            "- Docker/release/hardening remain later steps.\n"
            "- Weak anomaly evidence is review-only/supporting signal."
        )

    with overview_cols[1]:
        st.markdown(f"### {INSPECTION_CAPABILITY_SUMMARY_LABEL}")
        _render_overview_review_path()

    with st.expander("Technical evidence", expanded=False):
        evidence_tabs = st.tabs(["Detailed metrics", "Technical evidence", "Artifact and run details"])
        with evidence_tabs[0]:
            _render_key_value_grid(
                [
                    ("Internal metadata: Classification bundle files", len(track_a)),
                    ("Internal metadata: Anomaly bundle files", len(track_b)),
                    ("Internal metadata: Detection bundle files", len(detection)),
                ]
            )
        with evidence_tabs[1]:
            st.caption("Evidence paths")
            st.code("artifacts/frontend/track_a/", language="text")
            st.code("artifacts/frontend/track_b/", language="text")
            st.code("artifacts/frontend/detection/yolo_train_v0_2_0/", language="text")
        with evidence_tabs[2]:
            track_a_reco = track_a.get("frontend_model_recommendation.json", {})
            track_b_summary = track_b.get("frontend_anomaly_summary.json", {})
            detection_overview = detection.get("detection_overview.json", {})
            _render_key_value_grid(
                [
                    ("Internal metadata: Classification run", track_a_reco.get("selected_run_id")),
                    ("Internal metadata: Classification threshold", track_a_reco.get("selected_threshold")),
                    ("Internal metadata: Anomaly threshold", track_b_summary.get("key_metrics", {}).get("threshold")),
                    ("Internal metadata: Detection run", detection_overview.get("run_id")),
                ]
            )


def _render_track_a(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the surface defect classification page."""
    _render_hero_card(
        SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL,
        "This page summarizes governed classification validation evidence for the surface defect good-vs-defect task. Use Image Inspection for live image analysis.",
        "Evidence/dashboard view only · not production-ready · not deployment-safe",
        accent="teal",
    )
    st.warning(
        "Evidence/dashboard view only. Not production-ready. Not deployment-safe. "
        "This page shows validation evidence, not the live inspection workflow."
    )

    if bundles is None:
        st.error("Surface defect evidence is unavailable because the frontend bundles failed to load.")
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
            "Classification status",
            _friendly_status_label(
                quality.get("model_quality_status")
                or metric_cards.get("selected_model_quality_status")
                or quality.get("decision")
            ),
            help=_friendly_metric_display(quality.get("quality_target_status") or metric_cards.get("quality_target_status")),
        )
    with top_cols[3]:
        st.metric(
            "Validation samples",
            _format_value(metric_cards.get("validation_samples")),
            help="Governed validation split sample count used for the classification package.",
        )

    readiness_cols = st.columns(2)
    with readiness_cols[0]:
        st.metric(
            "Production readiness",
            "Not claimed",
            help="Not claimed in this evidence package.",
        )
    with readiness_cols[1]:
        st.metric(
            "Deployment readiness",
            "Not claimed",
            help="Not claimed in this evidence package.",
        )

    st.markdown("### Threshold explanation")
    st.write(
        "The threshold is the decision boundary between good and defect. The selected threshold is validation-derived and is used for evidence review; higher thresholds can reduce false positives but may increase false negatives. Live Image Inspection uses the threshold reported by the backend response."
    )

    st.markdown("### Visual evidence")
    visual_cols = st.columns(3)
    with visual_cols[0]:
        st.caption("Error distribution")
        st.caption("False positives are good parts flagged as defect; false negatives are missed defects.")
        error_rows = error_distribution.get("segments", [])
        if error_rows:
            labels = [str(row.get("label", f"segment_{idx}")) for idx, row in enumerate(error_rows)]
            values = [float(row.get("count", 0) or 0) for row in error_rows]
            palette = []
            for label in labels:
                lowered = label.lower()
                if "false_negative" in lowered or "fn" in lowered:
                    palette.append("#dc2626")
                elif "false_positive" in lowered or "fp" in lowered:
                    palette.append("#d97706")
                elif "true_positive" in lowered or "tp" in lowered:
                    palette.append("#16a34a")
                else:
                    palette.append("#2563eb")
            st.plotly_chart(
                _build_donut_figure(
                    "Surface defect error distribution",
                    labels,
                    values,
                    palette,
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface defect error distribution",
                "No error distribution data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
    with visual_cols[1]:
        st.caption("Per-class performance")
        st.caption("Per-class precision, recall, and F1 show how the classifier performs on each label.")
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
                    "Surface defect per-class performance",
                    categories,
                    series_map,
                    {"Precision": "#2563eb", "Recall": "#d97706", "F1": "#16a34a"},
                    "Score",
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface defect per-class performance",
                "No per-class data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
    with visual_cols[2]:
        st.caption("Threshold behavior")
        st.caption("This curve shows how precision, recall, F1, and accuracy move as the decision threshold changes.")
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
                    "Surface defect threshold behavior",
                    threshold_values,
                    series_map,
                    {
                        "Precision": "#2563eb",
                        "Recall": "#d97706",
                        "Macro F1": "#16a34a",
                        "Accuracy": "#7c3aed",
                    },
                    "Score",
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface defect threshold behavior",
                "No threshold sweep data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )

    _render_agent_callout(
        "Explain these classification charts",
        "Ask for a plain-language summary of the error distribution, per-class performance, and threshold behavior charts.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    st.markdown("### Sample evidence summary")
    gallery_cols = st.columns([0.7, 1.3])
    with gallery_cols[0]:
        st.metric("Sample evidence images", _format_value(gallery.get("gallery_sample_count")))
        st.caption("Summary-only view; individual images stay inside the governed evidence bundle.")
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

    with st.expander("Detailed metrics", expanded=False):
        st.caption("Not claimed means local review/demo only, not factory production use. Docker/release validation is still pending.")
        cards = metric_cards.get("cards", [])
        if cards:
            for start_idx in range(0, len(cards), 3):
                row_cards = cards[start_idx : start_idx + 3]
                cols = st.columns(len(row_cards))
                for col, card in zip(cols, row_cards):
                    with col:
                        st.metric(card.get("title", "Metric"), _friendly_metric_display(card.get("value")))
                        detail = card.get("detail")
                        if detail:
                            st.caption(_friendly_metric_display(detail))
        else:
            st.info("No surface defect metric cards available.")

    with st.expander("Technical evidence", expanded=False):
        st.caption("Model comparison")
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
        st.markdown("##### Confusion matrix")
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
        st.markdown("##### Per-class table")
        _render_markdown_table(
            per_class.get("classes", []),
            [
                ("Class", "label"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
            ],
        )
        st.markdown("##### Threshold table")
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

    with st.expander("Artifact and run details", expanded=False):
        st.caption("Internal metadata: Classification bundle | User-facing module: Surface Defect Classification")
        _render_key_value_grid(
            [
                ("Selected model", recommendation.get("selected_model_name") or metric_cards.get("selected_model_name")),
                ("Version", recommendation.get("selected_model_version") or metric_cards.get("selected_model_version")),
                ("Run ID", recommendation.get("selected_run_id")),
                ("Threshold", recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
                ("Quality target", _friendly_metric_display(quality.get("quality_target_status") or metric_cards.get("quality_target_status"))),
            ]
        )
        _render_key_value_grid(
            [
                ("Bundle files", inventory.get("bundle_artifact_count")),
                ("Source artifacts", inventory.get("source_artifact_count")),
                ("Selected threshold", recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
            ]
        )
        st.caption("Evidence paths")
        st.code("artifacts/frontend/track_a/", language="text")
        st.caption("Evidence file list")
        for filename in TRACK_A_EVIDENCE_FILENAMES:
            st.code(filename, language="text")
        st.caption(
            "The evidence bundle remains governed; raw JSON is intentionally hidden by default."
        )

    _render_premium_info_card(
        "Safe interpretation",
        "ResNet18 v0.4.0 is the selected governed validation model for this page.",
        "Evidence view only — local review/demo, not production use.",
        accent="gray",
    )


def _render_track_b(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the surface anomaly detection page."""
    _render_hero_card(
        SURFACE_ANOMALY_DETECTION_PAGE_LABEL,
        "A governed autoencoder-based anomaly page for reviewing reconstruction behavior, anomaly scores, and quality decision evidence.",
        "Evidence/dashboard view only · not production-ready · not deployment-safe",
        accent="orange",
    )
    st.warning(
        "Evidence view only — local review/demo, not production use. Not production-ready. Not deployment-safe. "
        "This page is evidence-only. Live inspection is available on the Image Inspection page."
    )

    if bundles is None:
        st.error("Surface anomaly evidence is unavailable because the frontend bundles failed to load.")
        return

    track_b = bundles["track_b"]
    metric_cards = track_b.get("metric_cards.json", {})
    anomaly_summary = track_b.get("anomaly_score_summary.json", {})
    frontend_summary = track_b.get("frontend_anomaly_summary.json", {})
    reconstruction = track_b.get("reconstruction_loss_summary.json", {})
    threshold_behavior = track_b.get("threshold_behavior.json", {})
    sample_predictions = track_b.get("sample_predictions.json", {})
    gallery = track_b.get("sample_anomaly_gallery.json", {})
    quality = track_b.get("quality_decision_summary.json", {})
    inventory = track_b.get("artifact_inventory_frontend.json", {})

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric(
            "Model family",
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
            _friendly_status_label(quality.get("quality_status") or quality.get("decision") or frontend_summary.get("quality_status") or frontend_summary.get("canonical_status")),
            help=_format_value(frontend_summary.get("canonical_status") or metric_cards.get("canonical_status")),
        )
    with top_cols[3]:
        st.metric(
            "PR AUC",
            _format_value(_extract_pr_auc(frontend_summary, metric_cards)),
            help="Average precision derived from governed sample-level anomaly scores.",
        )
    _render_premium_info_card(
        "PR AUC is governed evidence",
        "This page reviews reconstruction and anomaly behavior, and the governed evidence now includes PR AUC.",
        "The score is grounded in sample-level anomaly evidence and no production claim is made.",
        accent="orange",
    )
    _render_agent_callout(
        "Explain anomaly behavior",
        "Ask for a plain-language summary of the reconstruction, anomaly score, and threshold charts.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    st.markdown("### Visual evidence")
    visual_cols = st.columns(3)
    with visual_cols[0]:
        st.caption("Reconstruction behavior")
        reconstruction_data = reconstruction.get("sample_level_reconstruction_loss", reconstruction)
        reconstruction_summary = reconstruction_data.get("summary", {}) if isinstance(reconstruction_data, dict) else {}
        reconstruction_histograms = reconstruction_data.get("histograms", {}) if isinstance(reconstruction_data, dict) else {}
        reconstruction_mapping = reconstruction_data.get("mapping") if isinstance(reconstruction_data, dict) else None
        if reconstruction_histograms:
            labels, series_map = _extract_histogram_series(reconstruction_histograms)
            st.plotly_chart(
                _build_grouped_bar_figure(
                    "Surface anomaly reconstruction loss distribution",
                    labels,
                    series_map,
                    {
                        "All": "#2563eb",
                        "True Normal": "#16a34a",
                        "True Anomaly": "#d97706",
                    },
                    "Samples",
                ),
                width="stretch",
            )
            st.caption(
                reconstruction_mapping
                or "Reconstruction loss is shown as governed sample-level loss because the score definition is mean squared reconstruction error per image."
            )
        else:
            _render_chart_placeholder(
                "Surface anomaly reconstruction loss",
                "No reconstruction-loss distribution data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
        if reconstruction_summary:
            _render_key_value_grid(
                [
                    ("Samples", reconstruction_summary.get("all", {}).get("count")),
                    ("Median loss", reconstruction_summary.get("all", {}).get("median")),
                    ("P95 loss", reconstruction_summary.get("all", {}).get("p95")),
                ]
            )
    with visual_cols[1]:
        st.caption("Anomaly score summary")
        score_histograms = anomaly_summary.get("histograms", {})
        if score_histograms:
            labels, series_map = _extract_histogram_series(score_histograms)
            st.plotly_chart(
                _build_grouped_bar_figure(
                    "Surface anomaly score distribution",
                    labels,
                    series_map,
                    {
                        "All": "#2563eb",
                        "True Normal": "#16a34a",
                        "True Anomaly": "#d97706",
                    },
                    "Samples",
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface anomaly score distribution",
                "No anomaly score histogram data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
        anomaly_summary_stats = anomaly_summary.get("summary", {})
        if anomaly_summary_stats:
            _render_key_value_grid(
                [
                    ("Samples", anomaly_summary_stats.get("all", {}).get("count")),
                    ("Median score", anomaly_summary_stats.get("all", {}).get("median")),
                    ("P95 score", anomaly_summary_stats.get("all", {}).get("p95")),
                ]
            )
    with visual_cols[2]:
        st.caption("Threshold behavior")
        threshold_rows = _extract_threshold_rows(threshold_behavior)
        if threshold_rows:
            thresholds = [float(row.get("threshold", 0) or 0) for row in threshold_rows]
            series_map = {
                "Precision": [float(row.get("precision", 0) or 0) for row in threshold_rows],
                "Recall": [float(row.get("recall", 0) or 0) for row in threshold_rows],
                "F1": [float(row.get("f1", 0) or 0) for row in threshold_rows],
                "False Positive Rate": [float(row.get("false_positive_rate", 0) or 0) for row in threshold_rows],
                "False Negative Rate": [float(row.get("false_negative_rate", 0) or 0) for row in threshold_rows],
            }
            st.plotly_chart(
                _build_line_figure(
                    "Surface anomaly threshold behavior",
                    thresholds,
                    series_map,
                    {
                        "Precision": "#2563eb",
                        "Recall": "#d97706",
                        "F1": "#16a34a",
                        "False Positive Rate": "#7c3aed",
                        "False Negative Rate": "#ef4444",
                    },
                    "Score",
                ),
                width="stretch",
            )
            st.caption(
                "Selected threshold: "
                + _format_value(threshold_behavior.get("selected_threshold"))
                + " | "
                + _friendly_status_label("review_required_weak_evidence")
            )
        else:
            _render_chart_placeholder(
                "Surface anomaly threshold behavior",
                "No threshold behavior data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )

    st.markdown("### Sample evidence summary")
    gallery_cols = st.columns([0.7, 1.3])
    with gallery_cols[0]:
        st.metric("Gallery samples", _format_value(gallery.get("gallery_sample_count")))
        st.caption("Summary-only view; images stay inside the governed evidence bundle.")
        if sample_predictions.get("sample_count") is not None:
            st.metric("Sample predictions", _format_value(sample_predictions.get("sample_count")))
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
        sample_preview_rows = []
        for row in _extract_sample_prediction_rows(sample_predictions)[:5]:
            sample_preview_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "true_label": row.get("true_label"),
                    "predicted_label": row.get("predicted_label"),
                    "anomaly_score": row.get("anomaly_score"),
                    "threshold": row.get("threshold"),
                    "correct": row.get("correct"),
                }
            )
        if sample_preview_rows:
            st.caption("Sample-level anomaly evidence preview")
            _render_markdown_table(
                sample_preview_rows,
                [
                    ("Sample ID", "sample_id"),
                    ("True label", "true_label"),
                    ("Predicted label", "predicted_label"),
                    ("Anomaly score", "anomaly_score"),
                    ("Threshold", "threshold"),
                    ("Correct", "correct"),
                ],
            )

    with st.expander("Detailed metrics", expanded=False):
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
            st.info("No surface anomaly metric cards available.")

    with st.expander("Technical evidence", expanded=False):
        st.caption("Anomaly score summary")
        anomaly_rows = []
        for series_name, rows in (anomaly_summary.get("histograms", {}) or {}).items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                anomaly_rows.append(
                    {
                        "series": series_name,
                        "bin_start": row.get("bin_start"),
                        "bin_end": row.get("bin_end"),
                        "count": row.get("count"),
                    }
                )
        _render_markdown_table(
            anomaly_rows,
            [
                ("Series", "series"),
                ("Bin start", "bin_start"),
                ("Bin end", "bin_end"),
                ("Count", "count"),
            ],
        )
        st.markdown("##### Reconstruction table")
        reconstruction_rows = []
        reconstruction_histograms = reconstruction.get("sample_level_reconstruction_loss", {}).get("histograms", {})
        for series_name, rows in reconstruction_histograms.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                reconstruction_rows.append(
                    {
                        "series": series_name,
                        "bin_start": row.get("bin_start"),
                        "bin_end": row.get("bin_end"),
                        "count": row.get("count"),
                    }
                )
        _render_markdown_table(
            reconstruction_rows,
            [
                ("Series", "series"),
                ("Bin start", "bin_start"),
                ("Bin end", "bin_end"),
                ("Count", "count"),
            ],
        )
        st.markdown("##### Threshold table")
        threshold_rows = _extract_threshold_rows(threshold_behavior)
        _render_markdown_table(
            threshold_rows,
            [
                ("Threshold", "threshold"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
                ("False positive rate", "false_positive_rate"),
                ("False negative rate", "false_negative_rate"),
                ("False Positives", "false_positive"),
                ("False Negatives", "false_negative"),
            ],
        )
        st.markdown("##### Sample evidence details")
        _render_markdown_table(
            count_rows,
            [
                ("Error type", "error_type"),
                ("Count", "count"),
            ],
        )
        if sample_predictions.get("samples"):
            st.markdown("##### Sample predictions")
            _render_markdown_table(
                _extract_sample_prediction_rows(sample_predictions),
                [
                    ("Sample ID", "sample_id"),
                    ("True label", "true_label"),
                    ("Predicted label", "predicted_label"),
                    ("Anomaly score", "anomaly_score"),
                    ("Threshold", "threshold"),
                    ("Correct", "correct"),
                ],
            )

    with st.expander("Artifact and run details", expanded=False):
        st.caption("Internal metadata: Anomaly bundle | User-facing module: Surface Anomaly Detection")
        _render_key_value_grid(
            [
                ("Model", frontend_summary.get("model_type")),
                ("Version", frontend_summary.get("model_version")),
                ("Threshold", frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")),
                ("Quality status", _friendly_status_label(quality.get("quality_status") or frontend_summary.get("quality_decision") or quality.get("decision"))),
                ("PR AUC", _format_value(_extract_pr_auc(frontend_summary, metric_cards))),
            ]
        )
        _render_key_value_grid(
            [
                ("Bundle files", inventory.get("bundle_artifact_count")),
                ("Source artifacts", inventory.get("source_artifact_count")),
                ("Threshold", frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")),
            ]
        )
        st.caption("Evidence paths")
        st.code("artifacts/frontend/track_b/", language="text")
        st.caption("Evidence file list")
        for filename in TRACK_B_EVIDENCE_FILENAMES:
            st.code(filename, language="text")

    _render_premium_info_card(
        "Safe interpretation",
        "The surface anomaly detector remains governed evidence only.",
        "Weak-evidence quality status supports review rather than automation.",
        accent="gray",
    )


def _render_yolo(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the defect detection and localization page."""
    _render_hero_card(
        DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL,
        "This page summarizes governed detection and localization validation evidence for the validation bundle. Use Image Inspection for live image analysis and uploaded-image box overlays.",
        "Evidence/dashboard view only · not production-ready · not deployment-safe",
        accent="green",
    )
    st.warning(
        "Evidence/dashboard view only. Not production-ready. Not deployment-safe. "
        "This page shows validation evidence, not the live upload area."
    )
    st.caption("Evidence/dashboard view only")
    _render_agent_callout(
        "Explain detection confidence",
        "Ask for a plain-language summary of the confidence distribution, class balance, and detection review state.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
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

    st.caption(
        "Run yolo_train_v0_2_0 | "
        f"Model: {_format_value(metadata.get('model_name'))} | "
        f"Version: {_format_value(metadata.get('model_version'))} | "
        f"Dataset: {_format_value(metadata.get('dataset_id'))} | "
        f"Split: {_format_value(metadata.get('split'))}"
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("Model", _format_value(metadata.get("model_name") or overview.get("model_name")))
        st.caption(_format_value(metadata.get("model_type") or overview.get("model_type")))
    with top_cols[1]:
        st.metric("Detection evidence", "Available")
        st.caption("Governed validation evidence is loaded and ready for review.")
    with top_cols[2]:
        st.metric("Review status", "Needs review before final use")
        st.caption("Review-oriented evidence package; final use requires manual review.")
    with top_cols[3]:
        st.metric(
            "Validation images",
            _format_value(overview.get("image_count")),
            help=f"Run { _format_value(metadata.get('run_id')) }",
        )
        st.caption("Governed validation images in the bundle")

    status_cols = st.columns(2)
    with status_cols[0]:
        st.metric("Predicted boxes", _format_value(overview.get("total_bbox_count")))
        st.caption("All predicted boxes counted across the split.")
    with status_cols[1]:
        st.metric("Images with detections", _format_value(overview.get("image_with_detections_count")))
        st.caption("Images that produced at least one detection.")

    readiness_cols = st.columns(2)
    with readiness_cols[0]:
        st.metric("Production readiness", "Not claimed", help="Not claimed in this evidence package.")
    with readiness_cols[1]:
        st.metric("Deployment readiness", "Not claimed", help="Not claimed in this evidence package.")

    st.markdown("### Confidence interpretation")
    st.write(
        "Confidence score means how strongly the detector believes a box belongs to a defect class. Medium, high, and low confidence bands are evidence for review, not automatic production decisions. The confidence threshold affects how many boxes are shown."
    )
    st.markdown("### Class summary interpretation")
    st.write(
        "The class summary shows which defect classes were detected in validation evidence. Higher-count classes indicate more detections in evidence, not necessarily higher severity. Low-count classes may have weaker support, so class imbalance should be interpreted carefully."
    )

    st.markdown("### Visual evidence")
    visual_cols = st.columns(2)
    with visual_cols[0]:
        st.caption("Confidence distribution")
        confidence_rows = confidence_chart.get("confidence_bins", [])
        if confidence_rows:
            confidence_labels = [row.get("label", f"bin_{idx}") for idx, row in enumerate(confidence_rows)]
            confidence_values = [float(row.get("count", 0) or 0) for row in confidence_rows]
            confidence_colors = ["#2563eb", "#d97706", "#16a34a", "#64748b"][: len(confidence_labels)]
            st.plotly_chart(
                _build_donut_figure(
                    confidence_chart.get("chart_title", "Confidence distribution"),
                    confidence_labels,
                    confidence_values,
                    confidence_colors,
                ),
                width="stretch",
            )
            st.caption(
                confidence_chart.get("chart_explanation")
                or "Confidence distribution across predicted boxes."
            )
        else:
            _render_chart_placeholder(
                "YOLO confidence distribution",
                "No confidence distribution data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
    with visual_cols[1]:
        st.caption("Class summary")
        class_rows = class_summary.get("class_rows", [])
        if class_rows:
            class_labels = [row.get("class_label", f"class_{idx}") for idx, row in enumerate(class_rows)]
            class_counts = [float(row.get("bbox_count", 0) or 0) for row in class_rows]
            st.plotly_chart(
                _build_grouped_bar_figure(
                    "YOLO class summary by bbox count",
                    class_labels,
                    {"BBox count": class_counts},
                    {"BBox count": "#2563eb"},
                    "BBox count",
                ),
                width="stretch",
            )
            st.caption("Bar heights represent bbox counts per class.")
        else:
            _render_chart_placeholder(
                "YOLO class summary",
                "No class summary data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )

    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown("### What the bundle shows")
        st.write(
            "The detection bundle summarizes detection counts, confidence bands, class balance, and curated gallery samples for review."
        )
        st.metric(
            "Images with detections",
            _format_value(overview.get("image_with_detections_count")),
            help=_format_value(overview.get("safe_summary")),
        )
    with summary_cols[1]:
        st.markdown("### Sample detection summary")
        st.write(
            "This is curated validation evidence, not the current uploaded image. Live uploaded-image boxes are shown in Image Inspection."
        )
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

    _render_agent_callout(
        "Explain detection confidence",
        "Ask for a plain-language summary of the confidence distribution, class balance, and detection review state.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    st.markdown("### Quick take")
    _render_premium_info_card(
        "Review status: Needs review before final use",
        "The detection bundle is review-oriented only and does not claim production readiness or deployment safety.",
        "What it can claim: "
        + ", ".join(_format_value(value) for value in recommendation.get("what_it_can_claim", [])),
        accent="green" if str(quality.get("decision", "")).lower() == "pass" else "orange",
    )

    with st.expander("Detailed metrics", expanded=False):
        st.caption("Not claimed means local review/demo only, not factory production use. Docker/release validation is still pending.")
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

    with st.expander("Technical evidence", expanded=False):
        st.caption("Confidence distribution table")
        _render_markdown_table(
            confidence_rows,
            [
                ("Band", "label"),
                ("Count", "count"),
                ("Share", "percentage"),
            ],
        )
        st.caption("Class summary table")
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
        st.caption("Sample evidence details")
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

    with st.expander("Artifact and run details", expanded=False):
        st.caption("artifact lineage")
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
        st.code("artifacts/frontend/detection/yolo_train_v0_2_0/", language="text")
        _render_markdown_table(
            [{"file": name} for name in manifest.get("bundle_files", [])],
            [("File", "file")],
        )
        st.caption("Evidence file list")
        for filename in DETECTION_EVIDENCE_FILENAMES:
            st.code(filename, language="text")

    with st.expander("Run details", expanded=False):
        _render_key_value_grid(
            [
                ("Run", metadata.get("run_id")),
                ("Dataset", metadata.get("dataset_id")),
                ("Review required", _friendly_status_label(quality.get("review_required"))),
                ("Production ready", "Not claimed"),
                ("Deployment candidate", "Not claimed"),
            ]
        )
        st.markdown("##### Quality decision")
        st.write(
            f"{_friendly_status_label(quality.get('review_required'))} | "
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

    _render_premium_info_card(
        "Safe interpretation",
        "This detection and localization evidence layer is review-oriented only.",
        "It does not claim production readiness or deployment safety.",
        accent="gray",
    )


def _render_upload_predict() -> None:
    """Render the image inspection page."""
    _render_hero_card(
        IMAGE_INSPECTION_PAGE_LABEL,
        "A local inspection workflow that sends one uploaded image to the unified FastAPI endpoint and returns the governed inspection result.",
        "Local inspection workflow · not production-ready · not deployment-safe",
        accent="blue",
    )
    st.warning(
        "Local inspection workflow. not production-ready. not deployment-safe. "
        "The page now calls the unified inspection endpoint and shows the governed inspection response."
    )
    st.caption("Unified inspection workflow")

    step_cols = st.columns(3)
    with step_cols[0]:
        st.metric("Step 1", "Choose image")
        st.caption("PNG, JPG, JPEG, or WEBP")
    with step_cols[1]:
        st.metric("Step 2", "Confirm API URL")
        st.caption("Default: http://localhost:8000")
    with step_cols[2]:
        st.metric("Step 3", "Run inspection")
        st.caption("Unified inspection response")

    controls_cols = st.columns([1.2, 1])
    with controls_cols[0]:
        api_base_url = st.text_input(
            "API base URL",
            value=API_DEFAULT_BASE_URL,
            help="Base URL for POST /inspect/image.",
        )
    with controls_cols[1]:
        st.caption("Connection target")
        st.code(f"POST {api_base_url.rstrip('/')}/inspect/image", language="text")

    uploaded_file = st.file_uploader(
        "Choose an image for inspection",
        type=list(UPLOAD_ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        key="image_inspection_upload_file",
    )
    uploaded_image: Image.Image | None = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        try:
            uploaded_image = Image.open(BytesIO(file_bytes)).convert("RGB")
        except Exception:  # pragma: no cover - preview boundary
            uploaded_image = None
        preview_cols = st.columns([1, 1.1])
        with preview_cols[0]:
            st.markdown("### Image preview")
            st.image(file_bytes, caption=uploaded_file.name, width="stretch")
            st.caption(
                f"{uploaded_file.name} | "
                f"{uploaded_file.type or _infer_content_type(uploaded_file.name)} | "
                f"{len(file_bytes)} bytes"
            )
        with preview_cols[1]:
            st.markdown("### Ready to inspect")
            _render_key_value_grid(
                [
                    ("Filename", uploaded_file.name),
                    ("Content type", uploaded_file.type or _infer_content_type(uploaded_file.name)),
                    ("File size bytes", len(file_bytes)),
                ]
            )
            st.caption("This preview is local only and does not change the API contract.")
    else:
        st.info("Choose a single image to enable the inspection button.")

    predict_clicked = st.button(
        "Run inspection",
        type="primary",
        disabled=uploaded_file is None,
    )

    if predict_clicked and uploaded_file is not None:
        try:
            payload = _call_image_inspection_api(api_base_url, uploaded_file)
            st.session_state["image_inspection_prediction"] = payload
            st.session_state["image_inspection_error"] = None
        except Exception as exc:  # pragma: no cover - UI boundary
            st.session_state["image_inspection_prediction"] = None
            st.session_state["image_inspection_error"] = str(exc)

    error_message = st.session_state.get("image_inspection_error")
    if error_message:
        st.error(error_message)

    payload = st.session_state.get("image_inspection_prediction")
    if not isinstance(payload, dict):
        st.info("Upload an image and press Run inspection to call the unified image inspection endpoint.")
        return

    st.success("Inspection completed.")

    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    classification = payload.get("classification", {}) if isinstance(payload.get("classification"), dict) else {}
    detection = payload.get("detection", {}) if isinstance(payload.get("detection"), dict) else {}
    anomaly = payload.get("anomaly", {}) if isinstance(payload.get("anomaly"), dict) else {}
    traceability = payload.get("traceability", {}) if isinstance(payload.get("traceability"), dict) else {}
    explanation_context = payload.get("explanation_context", {}) if isinstance(payload.get("explanation_context"), dict) else {}
    input_meta = payload.get("input", {}) if isinstance(payload.get("input"), dict) else {}
    errors = payload.get("errors", []) if isinstance(payload.get("errors"), list) else []
    warnings = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []
    limitations = payload.get("limitations", []) if isinstance(payload.get("limitations"), list) else []

    result_cols = st.columns([1.1, 1])
    with result_cols[0]:
        st.markdown("### Final decision")
        accent = "green" if str(decision.get("final_decision", "")).lower() == "good" else "orange"
        _render_premium_info_card(
            "Decision: " + _friendly_decision_label(decision.get("final_decision")),
            "Rule-based aggregation of classification, defect localization, and anomaly signals.",
            _format_value(decision.get("recommended_action")),
            accent=accent,
        )
        _render_key_value_grid(
            [
                ("Decision level", _friendly_status_label(decision.get("decision_level"))),
                ("Model agreement", _friendly_status_label(decision.get("model_agreement_status"))),
                ("Primary signal", _safe_text(decision.get("primary_signal"))),
                ("Rule ID", _safe_text(decision.get("rule_id"))),
            ]
        )
        if decision.get("rule_summary"):
            st.caption(str(decision.get("rule_summary")))
        if decision.get("conflict_reason"):
            st.warning(str(decision.get("conflict_reason")))
        supporting_signals = decision.get("supporting_signals") or []
        if supporting_signals:
            st.caption("Supporting signals")
            st.write(", ".join(_friendly_status_label(item) for item in supporting_signals))
            with st.expander("Supporting signal details", expanded=False):
                st.json(supporting_signals)
    with result_cols[1]:
        st.markdown("### Detection overlay")
        if uploaded_image is not None:
            if _extract_detection_rows(detection):
                st.image(_annotate_detection_boxes(uploaded_image, detection), caption="Uploaded image with detection boxes", width="stretch")
            else:
                st.image(uploaded_image, caption="Uploaded image (no detection boxes returned)", width="stretch")
        else:
            st.info("Uploaded image preview is unavailable.")
        st.caption(
            f"Detection image size: {_format_value(detection.get('image_width'))} x {_format_value(detection.get('image_height'))}"
        )

        st.markdown("### Unified inspection results")
    model_cols = st.columns(3)
    with model_cols[0]:
        st.markdown("#### Classification")
        _render_key_value_grid(
            [
                ("Model", classification.get("model_name")),
                ("Version", classification.get("model_version")),
                ("Run ID", classification.get("run_id")),
                ("Threshold", classification.get("threshold")),
                ("Predicted label", classification.get("predicted_label")),
                ("Decision", _safe_text(classification.get("decision"))),
            ]
        )
        _render_key_value_grid(
            [
                ("Probability good", _format_probability(classification.get("probability_good"))),
                ("Probability defect", _format_probability(classification.get("probability_defect"))),
                ("Production ready", "Not claimed" if classification.get("production_ready") is False else _safe_text(classification.get("production_ready"))),
                ("Deployment safe", "Not claimed" if classification.get("deployment_safe") is False else _safe_text(classification.get("deployment_safe"))),
            ]
        )
        st.caption("Not claimed means local review/demo only, not factory production use. Docker/release validation is still pending.")
        if classification.get("limitations"):
            st.caption("Classification limitations")
            st.write(" | ".join(str(item) for item in classification.get("limitations", [])))

    with model_cols[1]:
        st.markdown("#### Defect detection & localization")
        _render_key_value_grid(
            [
                ("Model", detection.get("model_name")),
                ("Version", detection.get("model_version")),
                ("Run ID", detection.get("run_id")),
                ("Confidence threshold", detection.get("confidence_threshold")),
                ("IoU threshold", detection.get("iou_threshold")),
                ("Predicted boxes", detection.get("predicted_box_count")),
                ("Defect count", detection.get("defect_count")),
                ("Review status", _friendly_status_label(detection.get("review_status"))),
            ]
        )
        detections = _extract_detection_rows(detection)
        if detections:
            detection_rows = []
            for det in detections:
                detection_rows.append(
                    {
                        "box_id": det.get("box_id"),
                        "class_label": det.get("class_label"),
                        "display_label": det.get("display_label"),
                        "confidence": det.get("confidence"),
                        "bbox_xyxy": det.get("bbox_xyxy"),
                        "is_best_prediction": det.get("is_best_prediction"),
                    }
                )
            _render_markdown_table(
                detection_rows[:5],
                [
                    ("Box", "box_id"),
                    ("Class", "class_label"),
                    ("Label", "display_label"),
                    ("Confidence", "confidence"),
                    ("BBox xyxy", "bbox_xyxy"),
                    ("Best", "is_best_prediction"),
                ],
            )
        else:
            st.info("No detection boxes were returned by the unified inspection response.")
        if detection.get("best_detection"):
            best_detection = detection.get("best_detection")
            if isinstance(best_detection, dict):
                best_summary = ", ".join(
                    [
                        f"Class: {_safe_text(best_detection.get('class_label') or best_detection.get('display_label'))}",
                        f"Confidence: {_format_probability(best_detection.get('confidence'))}",
                        f"Box: {_safe_text(best_detection.get('box_id'))}",
                    ]
                )
                st.caption("Best detection summary")
                st.write(best_summary)
            with st.expander("Best detection details", expanded=False):
                st.json(best_detection)

    with model_cols[2]:
        st.markdown("#### Surface anomaly detection")
        _render_key_value_grid(
            [
                ("Model", anomaly.get("model_name")),
                ("Version", anomaly.get("model_version")),
                ("Run ID", anomaly.get("run_id")),
                ("Anomaly score", anomaly.get("anomaly_score")),
                ("Reconstruction loss", anomaly.get("reconstruction_loss")),
                ("Threshold", anomaly.get("threshold")),
                ("Predicted label", anomaly.get("predicted_label")),
                ("Decision", anomaly.get("decision")),
                ("Quality status", _friendly_status_label(anomaly.get("quality_status"))),
                ("Production ready", "Not claimed" if anomaly.get("production_ready") is False else _safe_text(anomaly.get("production_ready"))),
                ("Deployment safe", "Not claimed" if anomaly.get("deployment_safe") is False else _safe_text(anomaly.get("deployment_safe"))),
            ]
        )
        st.caption("Not claimed means local review/demo only, not factory production use. Docker/release validation is still pending.")
        if anomaly.get("optional_reconstruction_artifacts"):
            st.caption("Optional reconstruction artifacts")
            st.json(anomaly.get("optional_reconstruction_artifacts"))

    if warnings:
        st.warning("Inspection warnings were returned by the unified inspection response.")
        _render_markdown_table(
            [{"warning": warning} if isinstance(warning, str) else warning for warning in warnings],
            [("Warning", "warning")],
        )

    if errors:
        st.error("One or more inspection components reported an error.")
        error_rows = []
        for error in errors:
            if isinstance(error, dict):
                error_rows.append(
                    {
                        "component": error.get("component"),
                        "code": error.get("code"),
                        "message": error.get("message"),
                        "recoverable": error.get("recoverable"),
                    }
                )
        _render_markdown_table(
            error_rows,
            [
                ("Component", "component"),
                ("Code", "code"),
                ("Message", "message"),
                ("Recoverable", "recoverable"),
            ],
        )

    with st.expander("Limitations", expanded=False):
        if limitations:
            st.write("\n".join(f"- {item}" for item in limitations))
        else:
            st.info("No additional limitations were returned by the inspection response.")

    with st.expander("Traceability and explanation context", expanded=False):
        st.markdown("##### Input metadata")
        _render_key_value_grid(
            [
                ("Filename", input_meta.get("filename")),
                ("Content type", input_meta.get("content_type")),
                ("File size bytes", input_meta.get("file_size_bytes")),
                ("Image width", input_meta.get("image_width")),
                ("Image height", input_meta.get("image_height")),
                ("Image mode", input_meta.get("image_mode")),
            ]
        )
        st.markdown("##### Traceability")
        st.json(traceability)
        st.markdown("##### Explanation context")
        st.json(explanation_context)

    _render_agent_callout(
        "Explain this prediction",
        "Ask for a plain-language explanation of the unified inspection response, decision rule, and evidence sources.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    with st.expander("Technical evidence", expanded=False):
        detail_tabs = st.tabs(["Detailed metrics", "Artifact and run details", "Raw/API response"])
        with detail_tabs[0]:
            _render_key_value_grid(
                [
                    ("Request ID", payload.get("request_id")),
                    ("Classification run", classification.get("run_id")),
                    ("Detection run", detection.get("run_id")),
                    ("Anomaly run", anomaly.get("run_id")),
                    ("Decision rule", decision.get("rule_id")),
                ]
            )
        with detail_tabs[1]:
            _render_key_value_grid(
                [
                    ("Source endpoint", traceability.get("source_endpoint")),
                    ("Contract version", traceability.get("contract_version")),
                    ("API version", traceability.get("api_version")),
                ]
            )
            st.caption("Frontend evidence sources")
            for source in traceability.get("frontend_evidence_sources", []):
                st.code(str(source), language="text")
        with detail_tabs[2]:
            st.caption("Raw API response")
            st.json(payload)

    st.caption(
        "This page now shows the unified local inspection workflow via /inspect/image."
    )


def _render_ai_assistant() -> None:
    """Render the future AI explanation assistant placeholder page."""
    _render_hero_card(
        AI_EXPLANATION_ASSISTANT_PAGE_LABEL,
        "A planned, evidence-grounded explanation surface for helping non-technical users read governed inspection evidence.",
        "Planned / not active · no backend agent · no LLM call · no fake AI",
        accent="violet",
    )
    st.warning(
        "Planned / not active. No backend agent is implemented yet, no LLM call is wired, and no fake AI behavior exists."
    )

    status_cols = st.columns([1.25, 0.95])
    with status_cols[0]:
        _render_premium_info_card(
            "Future AI explanation assistant",
            "This future assistant will help non-technical users understand governed inspection evidence without changing the underlying model results.",
            "It is designed to sit beside charts and result cards, not replace them.",
            accent="violet",
        )
    with status_cols[1]:
        _render_premium_info_card(
            "Current status",
            "Planned / not active.",
            "No backend agent. No LLM call. No fake AI behavior.",
            accent="orange",
        )

    placement_cols = st.columns(2)
    with placement_cols[0]:
        _render_premium_info_card(
            "What it will explain",
            "Page summaries, chart explanations, inspection result summaries, confidence, warnings, manual review needs, and limitations.",
            "Examples: explain this page, explain this chart, explain this prediction.",
            accent="violet",
        )
    with placement_cols[1]:
        _render_premium_info_card(
            "Evidence it will use",
            "Governed frontend bundles, Image Inspection responses, prediction responses, traceability, and safety docs only.",
            "Future explanations must stay grounded in real evidence and response data.",
            accent="blue",
        )

    placement_cols = st.columns(2)
    with placement_cols[0]:
        _render_premium_info_card(
            "What it will not do",
            "It will not invent metrics or predictions, hide uncertainty, claim production readiness, claim deployment safety, replace reviewer approval, modify artifacts, update registries, or silently recompute evidence.",
            "Assistant not active yet.",
            accent="gray",
        )
    with placement_cols[1]:
        _render_premium_info_card(
            "Visibility note",
            "This is a placeholder for a future product capability.",
            "It should remain visible, but secondary to the model evidence pages.",
            accent="orange",
        )

    with st.expander("Agent design notes", expanded=False):
        st.caption("The future assistant should use governed evidence, prediction outputs, and safety docs.")
        st.write(
            "Future explanations may use governed frontend bundles, Image Inspection response data, classification results, defect localization boxes, anomaly results, final rule-based decisions, warnings, errors, limitations, traceability, explanation_context, safety documentation, model metadata, thresholds, run IDs, and artifact references."
        )
        st.write(
            "Backend agent remains a future phase. This page is a design placeholder only."
        )


def _render_limitations() -> None:
    """Render the limitations and safety page."""
    _render_hero_card(
        SAFETY_LIMITATIONS_PAGE_LABEL,
        "A governed boundary page that explains what the dashboard can review, what it cannot claim, and how local inspection evidence should be interpreted.",
        "not production-ready · not deployment-safe · governed evidence only",
        accent="gray",
    )
    _render_agent_callout(
        "Explain safety boundaries",
        "Ask for a plain-language summary of the local inspection workflow limits, production gaps, and deployment gaps.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence, prediction responses, decision outputs, safety docs, model metadata, thresholds, run IDs, and artifact references",
        accent="violet",
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("Governed evidence", "Available", help="The dashboard presents governed inspection evidence and local image inspection results.")
    with top_cols[1]:
        st.metric("Local image inspection", "Connected", help="The unified /inspect/image flow is available in the frontend.")
    with top_cols[2]:
        st.metric("Multi-model outputs", "Classification + localization + anomaly + decision", help="Local inspection returns the unified inspection response.")
    with top_cols[3]:
        st.metric("Manual review boundary", "Required", help="The dashboard does not replace expert/manual review.")

    st.caption("Not claimed means local review/demo only, not factory production use. Docker/release validation is still pending.")

    second_cols = st.columns(4)
    with second_cols[0]:
        st.metric("Production readiness", "Not claimed", help="No production-ready claim is made.")
    with second_cols[1]:
        st.metric("Deployment readiness", "Not claimed", help="No deployment-safe claim is made.")
    with second_cols[2]:
        st.metric("AI Explanation Assistant", "Planned / not active", help="No backend agent is active yet.")
    with second_cols[3]:
        st.metric("Docker / release", "Pending", help="Docker and release work remain later phases after local smoke tests.")
    st.caption("Current boundaries")

    with st.expander("Safety details", expanded=False):
        st.write(
            "This dashboard is evidence-focused only. It does not train models, silently recompute metrics, create artifacts, or update registries."
        )
        st.write(
            "The dashboard presents governed validation evidence and local image inspection results. The local inspection workflow includes classification, defect localization boxes, anomaly signal, final rule-based decision, warnings, errors, limitations, traceability, and explanation context."
        )
        st.write(
            "The system is not production-ready and not deployment-safe. It does not replace expert/manual review. New evidence files must be created by governed pipeline scripts."
        )
        st.write(
            "The AI explanation assistant remains planned / not active. No backend agent or LLM call is wired yet, and future explanations must stay grounded in governed evidence, prediction responses, decision outputs, safety docs, model metadata, thresholds, run IDs, and artifact references."
        )
        st.write(
            "Docker, release, and hardening remain later phases and stay pending until frontend completion and local smoke tests pass."
        )


def _call_image_inspection_api(api_base_url: str, uploaded_file: Any) -> dict[str, Any]:
    """Call the unified image inspection API with a local uploaded image."""
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

    endpoint = f"{normalized_base_url}/inspect/image"
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
        "input",
        "classification",
        "detection",
        "anomaly",
        "decision",
        "traceability",
        "limitations",
        "errors",
        "warnings",
        "explanation_context",
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
    _apply_light_visual_system()

    try:
        bundles = load_all_frontend_bundles()
    except (FileNotFoundError, ValueError) as exc:
        bundles = None
        st.error(f"Frontend data contracts are not fully loadable: {exc}")

    _render_sidebar_health(bundles)

    pages = {
        OVERVIEW_PAGE_LABEL: lambda: _render_overview(bundles),
        SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL: lambda: _render_track_a(bundles),
        SURFACE_ANOMALY_DETECTION_PAGE_LABEL: lambda: _render_track_b(bundles),
        DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL: lambda: _render_yolo(bundles),
        IMAGE_INSPECTION_PAGE_LABEL: _render_upload_predict,
        SAFETY_LIMITATIONS_PAGE_LABEL: _render_limitations,
        AI_EXPLANATION_ASSISTANT_PAGE_LABEL: _render_ai_assistant,
    }

    st.sidebar.markdown("### Navigation")
    choice = st.sidebar.radio("Navigation", list(pages.keys()), index=0)
    st.sidebar.markdown(
        f"""
        <div class="premium-card premium-card--blue" style="margin-bottom:0.75rem; padding:0.8rem 0.85rem;">
            <div class="premium-card__eyebrow">Current page</div>
            <div class="premium-card__title" style="font-size:1rem;">{html.escape(choice)}</div>
            <div class="premium-card__meta">Premium dashboard navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "This dashboard is read-only and consumes existing evidence bundles. "
        "The future AI explanation assistant is a placeholder only."
    )
    pages[choice]()


if __name__ == "__main__":
    main()

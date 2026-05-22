"""Streamlit entrypoint for the frontend dashboard shell.

This dashboard presents the current project status as a read-only evidence
shell. It is intentionally limited to local evidence presentation and does
not implement live prediction, API integration, Docker, or production
features.
"""

from __future__ import annotations

import mimetypes
import html
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

OVERVIEW_PAGE_LABEL = "Overview"
SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL = "Surface Defect Classification"
SURFACE_ANOMALY_DETECTION_PAGE_LABEL = "Surface Anomaly Detection"
DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL = "Defect Detection & Localization"
IMAGE_INSPECTION_PAGE_LABEL = "Image Inspection"
SAFETY_LIMITATIONS_PAGE_LABEL = "Safety & Limitations"
AI_EXPLANATION_ASSISTANT_PAGE_LABEL = "AI Explanation Assistant"
SYSTEM_CAPABILITY_STATUS_LABEL = "System Capability Status"

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


def _lookup_card_value(cards: list[dict[str, Any]], title: str) -> Any:
    """Return the value for a metric card with the requested title."""
    for card in cards:
        if not isinstance(card, dict):
            continue
        if str(card.get("title", "")).strip().lower() == title.strip().lower():
            return card.get("value")
    return None


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
        for key in ("track_a", "track_b", "detection"):
            bundle = bundles[key]
            st.metric(f"{key.replace('_', ' ').title()}", f"{len(bundle)} files")
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
        <div class="premium-card premium-card--{accent}" style="margin-top:0.35rem;">
            <div class="premium-card__eyebrow">Future AI explanation</div>
            <div class="premium-card__title">{html.escape(action_label)}</div>
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
        font=dict(color="#e2e8f0", family="Inter, Segoe UI, Arial"),
        legend=dict(font=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(8, 15, 28, 0.02)",
        plot_bgcolor="rgba(8, 15, 28, 0.18)",
        template="plotly_dark",
    )
    st.plotly_chart(figure, width="stretch")


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
        "A governed evidence dashboard for comparing surface defect classification, surface anomaly detection, and defect detection & localization while the local image inspection workflow is finalized.",
        "Governed evidence only · not production-ready · not deployment-safe",
        accent="blue",
    )
    _render_agent_callout(
        "Explain this page",
        "Use this placeholder to ask for a plain-language explanation of the dashboard page.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    if bundles is None:
        st.error("Overview data is unavailable because one or more frontend bundles failed to load.")
        return

    track_a = bundles["track_a"]
    track_b = bundles["track_b"]
    detection = bundles["detection"]
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric(SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL, "PASS", help="Governed classification evidence available")
    with summary_cols[1]:
        st.metric(SURFACE_ANOMALY_DETECTION_PAGE_LABEL, "PASS", help="Governed anomaly evidence available")
    with summary_cols[2]:
        st.metric(DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL, "COMPLETE", help="Governed detection evidence layer complete")
    with summary_cols[3]:
        st.metric(IMAGE_INSPECTION_PAGE_LABEL, "LOCAL WORKFLOW", help="Current local inspection flow")

    overview_cols = st.columns([1.15, 0.95])
    with overview_cols[0]:
        st.markdown("### What you can review")
        st.write(
            "- View governed evidence from surface defect classification, surface anomaly detection, and defect detection & localization\n"
            "- Compare the current summaries for each inspection module\n"
            "- Use the current local image inspection workflow"
        )
        st.markdown("### What this dashboard does not claim")
        st.write(
            "- not production-ready\n"
            "- not deployment-safe\n"
            "- the dashboard is still a governed evidence shell for local review"
        )

    with overview_cols[1]:
        st.markdown(f"### {SYSTEM_CAPABILITY_STATUS_LABEL}")
        _render_overview_status_chart()

    with st.expander("Technical evidence", expanded=False):
        evidence_tabs = st.tabs(["Detailed metrics", "Technical evidence", "Artifact and run details"])
        with evidence_tabs[0]:
            _render_key_value_grid(
                [
                    ("Internal track A files", len(track_a)),
                    ("Internal track B files", len(track_b)),
                    ("Internal detection files", len(detection)),
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
                    ("Internal track A run", track_a_reco.get("selected_run_id")),
                    ("Internal track A threshold", track_a_reco.get("selected_threshold")),
                    ("Internal track B threshold", track_b_summary.get("key_metrics", {}).get("threshold")),
                    ("Internal detection run", detection_overview.get("run_id")),
                ]
            )


def _render_track_a(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the surface defect classification page."""
    _render_hero_card(
        SURFACE_DEFECT_CLASSIFICATION_PAGE_LABEL,
        "A governed binary classifier page for the surface defect good-vs-defect task, presented as an evidence dashboard rather than a live deployment view.",
        "Evidence/dashboard view only · not production-ready · not deployment-safe",
        accent="teal",
    )
    st.warning(
        "Evidence/dashboard view only. not production-ready. not deployment-safe. "
        "No live prediction on this evidence page."
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
            "Task",
            "good vs defect",
            help="Binary classification task for the governed surface defect model.",
        )
    with top_cols[3]:
        st.metric(
            "Quality/status",
            _friendly_status_label(quality.get("decision") or metric_cards.get("selected_model_quality_status")),
            help=_format_value(quality.get("quality_target_status") or metric_cards.get("quality_target_status")),
        )

    st.markdown("### Visual evidence")
    visual_cols = st.columns(3)
    with visual_cols[0]:
        st.caption("Error distribution")
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

    st.markdown("### Sample gallery")
    gallery_cols = st.columns([0.7, 1.3])
    with gallery_cols[0]:
        st.metric("Sample gallery images", _format_value(gallery.get("gallery_sample_count")))
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
        st.caption("Internal track: Track A | User-facing module: Surface Defect Classification")
        _render_key_value_grid(
            [
                ("Selected model", recommendation.get("selected_model_name") or metric_cards.get("selected_model_name")),
                ("Version", recommendation.get("selected_model_version") or metric_cards.get("selected_model_version")),
                ("Run ID", recommendation.get("selected_run_id")),
                ("Threshold", recommendation.get("selected_threshold") or metric_cards.get("recommended_threshold")),
                ("Quality target", quality.get("quality_target_status") or metric_cards.get("quality_target_status")),
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

    st.markdown("### Safe interpretation")
    st.write(
        "ResNet18 v0.4.0 remains the governed classification candidate, and this page stays in evidence/dashboard view only."
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
        "Evidence/dashboard view only. not production-ready. not deployment-safe. "
        "No live prediction on this evidence page. Unified image inspection UI is not yet connected here."
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
        pr_auc_value = frontend_summary.get("key_metrics", {}).get("pr_auc")
        if pr_auc_value is None:
            pr_auc_value = _lookup_card_value(metric_cards.get("cards", []), "PR AUC")
        st.metric(
            "PR AUC",
            _format_value(pr_auc_value),
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
        reconstruction_rows = reconstruction.get("chart_rows", [])
        if reconstruction_rows:
            epochs = [float(row.get("epoch", idx + 1) or idx + 1) for idx, row in enumerate(reconstruction_rows)]
            losses = [float(row.get("reconstruction_loss", 0) or 0) for row in reconstruction_rows]
            st.plotly_chart(
                _build_line_figure(
                    "Surface anomaly reconstruction loss",
                    epochs,
                    {"Reconstruction loss": losses},
                    {"Reconstruction loss": "#2563eb"},
                    "Loss",
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface anomaly reconstruction loss",
                "No reconstruction loss data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
    with visual_cols[1]:
        st.caption("Anomaly score summary")
        anomaly_rows = anomaly_summary.get("segments", [])
        if anomaly_rows:
            labels = [str(row.get("label", f"segment_{idx}")) for idx, row in enumerate(anomaly_rows)]
            values = [float(row.get("count", 0) or 0) for row in anomaly_rows]
            palette = []
            for label in labels:
                lowered = label.lower()
                if "anomaly" in lowered or "high" in lowered:
                    palette.append("#dc2626")
                elif "normal" in lowered or "low" in lowered:
                    palette.append("#16a34a")
                else:
                    palette.append("#2563eb")
            st.plotly_chart(
                _build_donut_figure(
                    "Surface anomaly score distribution",
                    labels,
                    values,
                    palette,
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface anomaly score distribution",
                "No anomaly score data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )
    with visual_cols[2]:
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
                    "Surface anomaly threshold behavior",
                    thresholds,
                    series_map,
                    {"Precision": "#2563eb", "Recall": "#d97706", "F1": "#16a34a"},
                    "Score",
                ),
                width="stretch",
            )
        else:
            _render_chart_placeholder(
                "Surface anomaly threshold behavior",
                "No threshold behavior data is available in the governed bundle, so this visual is hidden.",
                accent="gray",
            )

    st.markdown("### Sample gallery")
    gallery_cols = st.columns([0.7, 1.3])
    with gallery_cols[0]:
        st.metric("Gallery samples", _format_value(gallery.get("gallery_sample_count")))
        st.caption("Summary-only view; images stay inside the governed evidence bundle.")
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
        anomaly_rows = anomaly_summary.get("segments", [])
        _render_markdown_table(
            anomaly_rows,
            [
                ("Label", "label"),
                ("Count", "count"),
                ("Share", "percentage"),
            ],
        )
        st.markdown("##### Reconstruction table")
        _render_markdown_table(
            reconstruction.get("chart_rows", []),
            [
                ("Epoch", "epoch"),
                ("Reconstruction loss", "reconstruction_loss"),
            ],
        )
        st.markdown("##### Threshold table")
        threshold_rows = threshold_behavior.get("rows", [])
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
        st.markdown("##### Sample anomaly gallery details")
        _render_markdown_table(
            count_rows,
            [
                ("Error type", "error_type"),
                ("Count", "count"),
            ],
        )

    with st.expander("Artifact and run details", expanded=False):
        st.caption("Internal track: Track B | User-facing module: Surface Anomaly Detection")
        _render_key_value_grid(
            [
                ("Model", frontend_summary.get("model_type")),
                ("Version", frontend_summary.get("model_version")),
                ("Threshold", frontend_summary.get("key_metrics", {}).get("threshold") or metric_cards.get("threshold")),
                ("Canonical status", frontend_summary.get("canonical_status") or metric_cards.get("canonical_status")),
                ("PR AUC", "Unavailable"),
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

    st.markdown("### Safe interpretation")
    st.write(
        "The surface anomaly detector remains governed evidence only, with a weak-evidence quality status that supports review rather than automation."
    )


def _render_yolo(bundles: dict[str, dict[str, Any]] | None) -> None:
    """Render the defect detection and localization page."""
    _render_hero_card(
        DEFECT_DETECTION_LOCALIZATION_PAGE_LABEL,
        "A governed defect detection and localization evidence page for the validation bundle, designed for review rather than live deployment.",
        "Evidence/dashboard view only · not production-ready · not deployment-safe",
        accent="green",
    )
    st.warning(
        "Evidence/dashboard view only. not production-ready. not deployment-safe. "
        "No live prediction on this evidence page. Live detection UI is not yet connected here."
    )
    st.caption("evidence/dashboard view only")
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

    _render_agent_callout(
        "Explain detection confidence",
        "Ask for a plain-language summary of the confidence distribution, class balance, and detection review state.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    st.markdown("### Quick take")
    _render_premium_info_card(
        "Review status: " + _format_value(quality.get("decision")),
        "The detection bundle is review-oriented only and does not claim production readiness or deployment safety.",
        "What it can claim: "
        + ", ".join(_format_value(value) for value in recommendation.get("what_it_can_claim", [])),
        accent="green" if str(quality.get("decision", "")).lower() == "pass" else "orange",
    )

    with st.expander("Detailed metrics", expanded=False):
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
        st.caption("Sample gallery details")
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
        "This detection and localization evidence layer is review-oriented only and does not claim production readiness or deployment safety."
    )


def _render_upload_predict() -> None:
    """Render the image inspection page."""
    _render_hero_card(
        IMAGE_INSPECTION_PAGE_LABEL,
        "A local inspection workflow that sends one uploaded image to the FastAPI endpoint and returns the governed inspection result.",
        "Local inspection workflow · not production-ready · not deployment-safe",
        accent="blue",
    )
    st.warning(
        "Local inspection workflow. not production-ready. not deployment-safe. "
        "The unified inspection UI is not yet connected here."
    )
    st.caption("Current flow: classification only")
    st.caption("Unified inspection UI is not yet connected here")

    step_cols = st.columns(3)
    with step_cols[0]:
        st.metric("Step 1", "Choose image")
        st.caption("PNG, JPG, JPEG, or WEBP")
    with step_cols[1]:
        st.metric("Step 2", "Confirm API URL")
        st.caption("Default: http://localhost:8000")
    with step_cols[2]:
        st.metric("Step 3", "Run inspection")
        st.caption("Current flow: classification only")

    controls_cols = st.columns([1.2, 1])
    with controls_cols[0]:
        api_base_url = st.text_input(
            "API base URL",
            value=API_DEFAULT_BASE_URL,
            help="Base URL for the current local inspection workflow.",
        )
    with controls_cols[1]:
        st.caption("Connection target")
        st.code(f"POST {api_base_url.rstrip('/')}/predict/classification", language="text")

    uploaded_file = st.file_uploader(
        "Choose an image for inspection",
        type=list(UPLOAD_ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        key="image_inspection_upload_file",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
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
        st.info("Choose a single image to enable the prediction button.")

    predict_clicked = st.button(
        "Run inspection",
        type="primary",
        disabled=uploaded_file is None,
    )

    if predict_clicked and uploaded_file is not None:
        try:
            payload = _call_classification_api(api_base_url, uploaded_file)
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
        st.info("Upload an image and press Run inspection to call the current classification endpoint.")
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
            width="stretch",
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
            "not production-ready | not deployment-safe | local inspection workflow | classification only"
        )

    _render_agent_callout(
        "Explain this prediction",
        "Ask for a plain-language explanation of the predicted label, confidence split, and model threshold.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    with st.expander("Technical evidence", expanded=False):
        detail_tabs = st.tabs(["Detailed metrics", "Artifact and run details", "Raw/API response"])
        with detail_tabs[0]:
            _render_key_value_grid(
                [
                    ("Request ID", payload.get("request_id")),
                    ("Run ID", payload.get("run_id")),
                    ("Model name", payload.get("model_name")),
                    ("Model version", payload.get("model_version")),
                    ("Threshold", payload.get("threshold")),
                ]
            )
        with detail_tabs[1]:
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
        with detail_tabs[2]:
            st.caption("Raw API response")
            st.json(payload)

    st.caption(
        "This page currently shows the local classification workflow. The unified inspection UI will expand this in a later phase."
    )


def _render_ai_assistant() -> None:
    """Render the future AI explanation assistant placeholder page."""
    _render_hero_card(
        AI_EXPLANATION_ASSISTANT_PAGE_LABEL,
        "A planned, context-aware explanation surface that will sit beside charts, pages, and predictions.",
        "Agent layer planned · no backend agent implemented yet · no fake AI",
        accent="violet",
    )
    st.info(
        "AI explanation assistant planned. no backend agent implemented yet. no fake AI. "
        "Future explanations should use governed evidence, prediction responses, and safety docs."
    )
    st.write("This is a placeholder only. It does not chat, call an LLM, or generate explanations yet.")

    placement_cols = st.columns(2)
    with placement_cols[0]:
        _render_premium_info_card(
            "What it will explain",
            "Page summaries, chart explanations, prediction result summaries, and safety boundaries.",
            "Explain this page · Explain this chart · Explain this prediction",
            accent="violet",
        )
    with placement_cols[1]:
        _render_premium_info_card(
            "Evidence it will use",
            "Governed evidence bundles, prediction responses, and safety docs only.",
            "Future explanations should stay grounded in tracked bundle data.",
            accent="blue",
        )

    placement_cols = st.columns(2)
    with placement_cols[0]:
        _render_premium_info_card(
            "What it will not do",
            "It will not invent evidence, call an LLM in this phase, or replace governed dashboard content.",
            "no fake AI · no backend agent implemented yet",
            accent="gray",
        )
    with placement_cols[1]:
        _render_premium_info_card(
            "Current status",
            "Placeholder only. The agent layer is planned for a later phase.",
            "Future AI explanation surfaces will be added beside charts and result cards.",
            accent="orange",
        )

    with st.expander("Agent design notes", expanded=False):
        st.caption("The future agent should use governed evidence, prediction outputs, and safety docs.")
        st.write(
            "Backend agent remains a future phase. This page is a design placeholder only."
        )


def _render_limitations() -> None:
    """Render the limitations and safety page."""
    _render_hero_card(
        SAFETY_LIMITATIONS_PAGE_LABEL,
        "A short and explicit safety page that states what the dashboard can claim and what it cannot claim yet.",
        "not production-ready · not deployment-safe · governed evidence only",
        accent="gray",
    )
    _render_agent_callout(
        "Explain safety boundaries",
        "Ask for a plain-language summary of the local inspection workflow limits, production gaps, and deployment gaps.",
        "Agent layer planned · no backend agent implemented yet · no fake AI · future explanations should use governed evidence and prediction responses",
        accent="violet",
    )

    top_cols = st.columns(3)
    with top_cols[0]:
        st.metric("Safe to claim", "Governed evidence", help="Dashboards and evidence summaries only.")
    with top_cols[1]:
        st.metric("Not safe to claim", "Production readiness", help="No production or deployment claim is made.")
    with top_cols[2]:
        st.metric("Current flow", "Local inspection workflow", help="The unified inspection UI is still being finalized.")

    second_cols = st.columns(3)
    with second_cols[0]:
        st.metric("Production gaps", "Not claimed", help="No production claim is made.")
    with second_cols[1]:
        st.metric("Deployment gaps", "Not claimed", help="No deployment-safe claim is made.")
    with second_cols[2]:
        st.metric("Agent limitations", "Placeholder only", help="The AI explanation assistant page is not a backend agent.")
    st.caption("Current boundaries")

    with st.expander("Safety details", expanded=False):
        st.write(
            "This dashboard is evidence-focused only. It does not train models, recompute metrics, create artifacts, update registries, or claim production or deployment readiness."
        )
        st.write(
            "The local inspection workflow is still being finalized. The AI explanation assistant remains a placeholder only."
        )


def _call_classification_api(api_base_url: str, uploaded_file: Any) -> dict[str, Any]:
    """Call the local classification API with a local uploaded image."""
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

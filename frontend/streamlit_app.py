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


def _render_track_a() -> None:
    """Render the Track A page."""
    st.subheader("Track A Classification")
    st.write("Planned: read the Track A JSON data contract and present summary cards, charts, and samples.")
    st.info("Current status: PASS, selected candidate ResNet18 v0.4.0, threshold 0.65.")


def _render_track_b() -> None:
    """Render the Track B page."""
    st.subheader("Track B Anomaly Detection")
    st.write("Planned: read the Track B JSON data contract and present anomaly metrics, thresholds, and examples.")
    st.info("Current status: PASS, PR AUC unavailable in governed evidence and not fabricated.")


def _render_yolo() -> None:
    """Render the YOLO page."""
    st.subheader("YOLO Detection")
    st.write(
        "Planned: read the Detection JSON data contract and present overview, metadata, confidence "
        "distribution, class summary, sample gallery, and lineage."
    )
    st.info("Current status: YOLO / Detection evidence layer COMPLETE.")


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
        "Track A Classification": _render_track_a,
        "Track B Anomaly Detection": _render_track_b,
        "YOLO Detection": _render_yolo,
        "Upload / Predict": _render_upload_predict,
        "Limitations / Safety": _render_limitations,
    }

    choice = st.sidebar.radio("Navigation", list(pages.keys()), index=0)
    st.sidebar.caption("This scaffold is read-only and consumes existing evidence bundles in later phases.")
    pages[choice]()


if __name__ == "__main__":
    main()

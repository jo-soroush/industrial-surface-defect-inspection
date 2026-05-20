"""Streamlit entrypoint for the initial frontend scaffold.

This scaffold presents the current project status as a read-only demo shell.
It is intentionally limited to local evidence presentation and does not
implement live prediction, API integration, Docker, or production features.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.data_loader import (
    DETECTION_BUNDLE_DIR,
    TRACK_A_BUNDLE_DIR,
    TRACK_B_BUNDLE_DIR,
)


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


def _render_status_summary() -> None:
    """Render a compact project status summary."""
    cols = st.columns(2)
    for idx, (label, value) in enumerate(STATUS_LINES):
        with cols[idx % 2]:
            st.metric(label, value)


def _render_limitations_banner() -> None:
    """Show the non-production limitations banner."""
    st.warning(
        "Read-only evidence scaffold only. No production-ready claim, no deployment-safe "
        "claim, no live upload/predict, and no API-backed inference yet."
    )


def _render_overview() -> None:
    """Render the overview page."""
    st.subheader("Overview")
    st.write(
        "This scaffold is a company-grade starting point for a demo dashboard that "
        "will consume the existing JSON data contracts for Track A, Track B, and YOLO."
    )
    st.write("Bundle directories:")
    st.code(
        "\n".join(
            [
                str(TRACK_A_BUNDLE_DIR),
                str(TRACK_B_BUNDLE_DIR),
                str(DETECTION_BUNDLE_DIR),
            ]
        ),
        language="text",
    )


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

    pages = {
        "Overview": _render_overview,
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

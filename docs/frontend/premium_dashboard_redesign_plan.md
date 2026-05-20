# Premium Dashboard Redesign Plan

## 1. Goal

Define a cleaner, more visual, presentation-grade dashboard direction before changing any frontend code. The current dashboard is technically correct, but the UI should be easier to scan, easier to explain, and better suited for non-technical reviewers.

## 2. Current UI Problems

- Too crowded
- Too many tables
- Too much technical text
- Weak visual hierarchy
- Not enough charts
- Technical details are too visible

## 3. Target Dashboard Style

- Clean
- Professional
- Visual-first
- Executive-friendly
- Technical details available but hidden

The dashboard should communicate the most important evidence quickly, with supporting technical detail available on demand.

## 4. Page Redesign Plan

### Overview

- Keep the project title and status as the first visual signal.
- Use compact status cards for Track A, Track B, and YOLO.
- Add a short safety banner with the current limitations.
- Show only the key high-level evidence numbers up front.
- Move the bundle/file detail into an expander.

### Track A Classification

- Show the selected model, version, threshold, and quality outcome as prominent summary cards.
- Replace large visible tables with charts where possible.
- Keep the comparison table and confusion matrix in expanders or secondary tabs.
- Present the key decision visually first, then expose the technical evidence.

### Track B Anomaly Detection

- Show the anomaly model summary, threshold, and status first.
- Use visual summaries for anomaly score and reconstruction behavior.
- Keep PR AUC unavailable messaging clear but not dominant.
- Move threshold detail and evidence inventory into expanders.

### YOLO Detection

- Show detection image count, bbox count, and decision status first.
- Prefer charts for confidence distribution and class summary.
- Keep artifact lineage and manifest detail in expandable technical sections.
- Preserve the clear distinction between evidence dashboard and real deployment.

### Upload / Predict

- Make the upload flow visually simple.
- Show a clean image preview, a single predict action, and a compact result summary.
- Present confidence, decision, and model metadata as cards.
- Keep raw response and debug details hidden by default.
- Preserve the message that this is a local prototype endpoint only.

### Limitations / Safety

- Keep the limitations clear, brief, and visible.
- Use simple status styling rather than a dense block of text.
- Preserve the non-production and non-deployment wording.

## 5. Chart Plan

- Donut/pie chart for Track A error distribution
- Bar chart for Track A class performance
- Line chart for threshold behavior
- Bar/line chart for Track B anomaly/reconstruction behavior
- Bar chart for YOLO class summary
- Confidence distribution chart for YOLO
- Prediction confidence display for Upload / Predict

## 6. Table Strategy

- Only the key tables should remain visible by default.
- Detailed tables should move into expanders.
- Evidence paths should live inside technical expanders.
- Raw JSON should remain hidden by default.
- The user should be able to inspect details without seeing everything at once.

## 7. Color And Status Strategy

- Green: pass / good / successful
- Red: defect / error / failed
- Orange: warning / review required
- Gray: not production-ready / informational caution
- Blue: information / neutral evidence

Use color to reduce reading burden, not to exaggerate confidence.

## 8. Dependency Strategy

Recommendation:
- Use Plotly if a donut/pie chart or richer chart styling is needed.
- Keep Streamlit built-in charts where they are already sufficient.
- Do not add unnecessary dependencies.

Plotly is worth adding only if it clearly improves readability and presentation quality beyond the built-in options.

## 9. Implementation Phases

- Phase 14A: UI/UX redesign plan
- Phase 14B: Add Plotly dependency if approved
- Phase 14C: Redesign Overview page
- Phase 14D: Redesign Track A page
- Phase 14E: Redesign Track B page
- Phase 14F: Redesign YOLO page
- Phase 14G: Redesign Upload/Predict page
- Phase 14H: Full frontend smoke test

## 10. Boundaries

- No API change
- No model change
- No artifact change
- No production-ready claim
- No deployment-safe claim
- No fake YOLO/anomaly upload prediction

## 11. Next Step

The next step after this document is reviewing the plan, then implementing the visual redesign in small commits.

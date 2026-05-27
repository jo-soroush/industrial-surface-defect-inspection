# Component Registry Coverage Audit

## 1. Executive Summary

This audit reviews whether `configs/agent/component_registry.yaml` covers the visible dashboard components in `frontend/streamlit_app.py`.

Result: **PARTIAL**.

What is confirmed:

- The registry covers every visible component group identified in the current static review.
- Four high-priority explainable components are actively wired in the frontend.
- The remaining visible components are either registry-ready but not actively wired, or are documented safety / assistant boundary content that should remain fixed copy for now.
- No missing registry entry was identified in this static review.

Why this is still PARTIAL:

- Registry coverage is present, but the dashboard is not fully active-explainable across all visible groups.
- The AI Assistant and Safety / Limitations pages are intentionally documented-only boundaries, not active explanation surfaces.
- This audit is static and does not replace a future end-to-end coverage verification step.

## 2. Method

Sources reviewed:

- `frontend/streamlit_app.py`
- `configs/agent/component_registry.yaml`
- `src/inspection_ai/agent/component_registry.py`
- `tests/agent/test_component_registry.py`
- `tests/frontend/*`
- `docs/agent/pre_gemini_gap_audit.md`

Method used:

- Identified visible dashboard pages and their major component groups from the frontend render functions.
- Matched each visible component group to a registry `component_id` where possible.
- Distinguished between:
  - active frontend explanation wiring,
  - registry coverage without active wiring,
  - documented boundary content that should remain fixed copy,
  - missing or uncertain coverage.
- Treated a component as explainable only if both registry evidence and a stable explanation path were visible.

## 3. Registry Inventory Summary

Registry totals by page:

| page_id | component count | active explainable count | registry-ready but not active | documented not explainable |
|---|---:|---:|---:|---:|
| overview | 3 | 0 | 3 | 0 |
| classification | 8 | 1 | 7 | 0 |
| anomaly | 7 | 1 | 6 | 0 |
| detection | 7 | 1 | 6 | 0 |
| image_inspection | 8 | 1 | 7 | 0 |
| safety | 2 | 0 | 0 | 2 |
| ai_assistant | 2 | 0 | 0 | 2 |

Totals:

- Registry entries reviewed: 37
- Active explainable components: 4
- Registry-ready but not active components: 26
- Documented not explainable components: 7
- Missing registry entries identified: 0
- Needs-review components identified: 0

## 4. Page-by-Page Coverage Table

| Page | Visible component groups | Frontend location / function | Registry coverage | Active wiring | Readiness category | Note |
|---|---|---|---|---|---|---|
| Overview | Capability summary, readiness summary, review path | `_render_overview` | Yes | No | `REGISTRY_READY_NOT_ACTIVE` | Review-oriented summary cards are registered, but no active Agent panel is attached. |
| Surface Defect Classification | Metric cards, readiness cards, error distribution, per-class performance, threshold behavior, model comparison, confusion matrix, safe interpretation | `_render_track_a` | Yes | Threshold chart only | Mixed: one `ACTIVE_EXPLAINABLE`, rest `REGISTRY_READY_NOT_ACTIVE` | The threshold chart is active; the other governed evidence blocks are registry-backed but not separately wired. |
| Surface Anomaly Detection | Metric cards, PR AUC note, reconstruction loss, anomaly score, threshold behavior, sample summary, safe interpretation | `_render_track_b` | Yes | Threshold chart only | Mixed: one `ACTIVE_EXPLAINABLE`, rest `REGISTRY_READY_NOT_ACTIVE` | The threshold chart is active; the remaining anomaly evidence is registry-backed but not separately wired. |
| Defect Detection & Localization | Metric cards, readiness cards, confidence chart, class summary, sample summary, artifact lineage, safe interpretation | `_render_yolo` | Yes | Confidence chart only | Mixed: one `ACTIVE_EXPLAINABLE`, rest `REGISTRY_READY_NOT_ACTIVE` | The confidence chart is active; the remaining detection evidence is registry-backed but not separately wired. |
| Image Inspection | Final decision, classification result, detection result, anomaly result, warnings, limitations, traceability, AI explanation panel | `_render_upload_predict` | Yes | AI explanation panel only | Mixed: one `ACTIVE_EXPLAINABLE`, rest `REGISTRY_READY_NOT_ACTIVE` | The AI panel is active; the runtime cards are registered and available for grounding, but not separate UI panels. |
| AI Explanation Assistant | Status cards, design notes | `_render_ai_assistant` | Yes | No | `DOCUMENTED_NOT_EXPLAINABLE` | This page describes the current mock boundary and the future LLM assistant; it should remain fixed copy until a formal assistant phase is planned. |
| Safety & Limitations | Safety boundary cards, safety details | `_render_limitations` | Yes | No | `DOCUMENTED_NOT_EXPLAINABLE` | This page is a boundary / policy page; it should stay as fixed safety copy, not an active explanation surface. |

## 5. Active Explainable Components

The following registry-backed components are actively wired to `/agent/explain`:

1. `image_inspection_ai_explanation_panel`
   - Page: Image Inspection
   - Location: `_render_upload_predict`
   - Status: `RUNTIME_ONLY`
   - Evidence: runtime `inspection_response`

2. `detection_confidence_chart`
   - Page: Defect Detection & Localization
   - Location: `_render_yolo`
   - Status: `READY_FOR_COMPONENT_RAG`
   - Evidence: governed detection bundle JSON

3. `classification_threshold_curve_chart`
   - Page: Surface Defect Classification
   - Location: `_render_track_a`
   - Status: `READY_FOR_COMPONENT_RAG`
   - Evidence: governed classification bundle JSON

4. `anomaly_threshold_behavior_chart`
   - Page: Surface Anomaly Detection
   - Location: `_render_track_b`
   - Status: `READY_FOR_COMPONENT_RAG`
   - Evidence: governed anomaly bundle JSON

## 6. Registry-Ready But Not Active Components

These components are present in the registry and have allowlisted evidence, but there is no active explanation panel or button attached to them in the current frontend:

### Overview

- `overview_capability_summary`
- `overview_readiness_summary`
- `overview_review_path`

### Classification

- `classification_metric_cards`
- `classification_readiness_cards`
- `classification_error_distribution_chart`
- `classification_per_class_chart`
- `classification_model_comparison_table`
- `classification_confusion_matrix_table`
- `classification_safe_interpretation`

### Anomaly

- `anomaly_metric_cards`
- `anomaly_pr_auc_summary`
- `anomaly_reconstruction_loss_chart`
- `anomaly_score_summary_chart`
- `anomaly_sample_summary`
- `anomaly_safe_interpretation`

### Detection

- `detection_metric_cards`
- `detection_readiness_cards`
- `detection_class_summary_chart`
- `detection_sample_gallery_summary`
- `detection_artifact_lineage`
- `detection_safe_interpretation`

### Image Inspection

- `image_inspection_classification_result_card`
- `image_inspection_detection_result_card`
- `image_inspection_anomaly_result_card`
- `image_inspection_warning_summary`
- `image_inspection_limitations`
- `image_inspection_traceability_context`
- `image_inspection_final_decision_card` is registry-ready but not a separate frontend control; it is served through the active Image Inspection AI panel.

## 7. Missing or Needs-Review Components

Missing registry entries identified: **none**.

Needs-review components identified: **none**.

Reason:

- Every visible dashboard component group reviewed in the frontend has a corresponding registry entry or is intentionally documented-only boundary content.
- The remaining gap is active wiring coverage, not registry presence.

## 8. Safety and Non-Explainable Components

The following surfaces are intentionally documented rather than exposed as active Agent explanation targets:

- `safety_boundary_cards`
- `safety_details`
- `ai_assistant_status_cards`
- `ai_assistant_design_notes`

Why:

- These pages are boundary / policy surfaces.
- They describe current constraints, not evidence that should be reinterpreted by an assistant.
- They should remain fixed copy until a formal future assistant scope is defined.

## 9. Coverage Decision

**Decision: PARTIAL**

Rationale:

- Registry coverage is strong and no missing entry was identified.
- The active explainability surface is intentionally limited to four priority components.
- Safety and AI assistant surfaces remain documented-only.
- A full dashboard-wide active explainability proof has not yet been established.

## 10. Required Next Actions Before Gemini

1. Confirm whether additional registry-backed components should receive active explanation wiring, or remain registry-ready only.
2. Keep the AI Assistant and Safety / Limitations surfaces as fixed boundary copy unless a formal future assistant scope is approved.
3. Produce a formal safety guard layer that sits ahead of any future LLM integration.
4. Finalize the provider readiness / contract layer before any Gemini work.
5. Produce a requirement-to-test matrix that maps every pre-Gemini blocker to tests and local validations.
6. Run the final LLM-disabled Docker / Compose smoke validation after the remaining pre-Gemini work is complete.

## Coverage Summary for Gemini Planning

Gemini should not begin until:

- Registry coverage is explicitly accepted as complete for the intended explainable surface set.
- The safety guard layer is present and tested.
- Provider readiness is documented and validated.
- The pre-Gemini test matrix exists and is current.
- The final local Docker / Compose smoke validation has passed with LLM disabled.

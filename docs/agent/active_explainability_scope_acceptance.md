# Active Explainability Scope Acceptance

## Executive Summary

For the pre-Gemini Agent/RAG phase, the project accepts a limited active explainability scope.

This decision is intentional and evidence-based:

- It covers the most important user-facing explanation needs.
- It avoids overloading the UI before Gemini.
- It reduces risk before real LLM integration.
- It keeps evidence grounding controlled.
- It preserves manual review and safety boundaries.
- It leaves room for future expansion without blocking Gemini readiness planning.

This is not a decision to make the entire dashboard actively explainable right now.
It is a decision to accept a focused, controlled set of active explanation surfaces for the pre-Gemini phase.

## Decision

Accepted active explainable components for pre-Gemini:

1. `image_inspection_ai_explanation_panel`
2. `detection_confidence_chart`
3. `classification_threshold_curve_chart`
4. `anomaly_threshold_behavior_chart`

Registry-ready but inactive components:

- The remaining 26 registry-ready components stay inactive for now.
- They are not considered defects or missing work.
- They remain available for future controlled expansion.
- They must not be activated silently.

Documented non-explainable components:

- Safety and AI Assistant boundary/status surfaces remain fixed copy for now.
- They are not LLM reinterpretation surfaces before a future formal scope decision.

## Accepted Active Components

The following four surfaces are accepted as the active pre-Gemini explainability scope:

| component_id | page |
|---|---|
| `image_inspection_ai_explanation_panel` | Image Inspection |
| `detection_confidence_chart` | Defect Detection & Localization |
| `classification_threshold_curve_chart` | Surface Defect Classification |
| `anomaly_threshold_behavior_chart` | Surface Anomaly Detection |

## Registry-Ready But Inactive Components

The registry contains 26 additional explainable components that are intentionally inactive for now.

They remain registry-ready and evidence-backed, but they are not part of the accepted pre-Gemini active scope.

Examples include:

- Overview summary cards
- Classification metric and comparison components
- Anomaly summary and evidence components
- Detection metric, summary, and lineage components
- Image Inspection result subcards and traceability blocks

These components are not missing and are not defects. They are reserved for future controlled expansion.

## Documented Non-Explainable Components

The following pages remain fixed boundary content:

- AI Explanation Assistant status / design notes
- Safety & Limitations boundary / policy content

These surfaces should remain static until a separate future scope decision explicitly turns them into explainable surfaces.

## Rationale

This limited scope is accepted because it:

- covers the most important explanation needs for the current product state,
- keeps the UI and mental model manageable before Gemini,
- preserves manual review as the default,
- keeps the evidence grounding path narrow and testable,
- avoids silently expanding scope without explicit approval,
- allows future expansion to happen one component at a time with separate approval, tests, and UI review.

## Future Expansion Rule

The accepted active scope is not a blanket approval for all registry-ready components.

Future expansion must happen only if all of the following are true:

- the component is explicitly approved for active explainability,
- the component has allowlisted evidence and a stable registry mapping,
- the frontend request path is added intentionally,
- tests cover the request contract and safety behavior,
- the UI is reviewed for wording, layout, and manual review boundaries.

## Gemini Readiness Impact

This scope decision removes the active explainability scope acceptance blocker for the pre-Gemini phase.

It does not mean Gemini integration is ready.
Gemini still must wait for the remaining deployment and provider planning items to be completed.

## Remaining Blockers After This Decision

- Final LLM-disabled Docker / Compose smoke validation.
- Gemini provider integration readiness plan.
- Keep the requirement-to-test matrix current if scope changes.


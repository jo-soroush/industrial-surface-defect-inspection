# Premium Dashboard And Agent UI Design Contract

## 1. Purpose

Define the visual and interaction contract for the premium dashboard redesign and the future context-aware AI Agent UI. The goal is a clean, professional, minimal dashboard that is easier to scan, easier to explain, and better suited for review by both technical and non-technical viewers.

## 2. Visual Direction

- Light or soft background only.
- No black full-page background.
- Minimal colorful layout, not crowded.
- Professional dashboard look rather than a dark terminal-style surface.
- Card-based sections for primary summaries and actions.
- Strong spacing, clear hierarchy, and obvious reading order.

## 3. Color System

- Background: soft neutral or very light tinted base.
- Surface / card: white or near-white with subtle separation.
- Primary accent: calm blue or indigo for navigation and key actions.
- Secondary accent: muted teal or violet for supportive highlights.
- Success: green for pass, good, or complete states.
- Warning: orange for review required or caution states.
- Danger / defect: red for defects, failures, or critical issues.
- Info: blue for neutral guidance or evidence context.
- Neutral / not production-ready: gray for safety and limitation messaging.

## 4. Typography And Layout Rules

- Keep text short and direct.
- Use large, readable numbers for key metrics.
- Place charts before detailed tables.
- Use consistent headings across pages.
- Hide technical text by default behind expanders or similar disclosures.
- Keep page copy concise so the dashboard remains presentation-grade.

## 5. Global Layout Rules

- Remove the old global summary block from all pages.
- Each page must begin with its own hero or header section.
- Data load health must be compact or placed inside an expander.
- Evidence paths must live inside technical expanders.
- Technical tables should not dominate the visible page.
- Raw JSON must remain hidden by default.

## 6. Page Design Rules

### Overview

- Show a clear hero/header.
- Show compact high-level cards for the major modules.
- Show charts and status first.
- Keep evidence details inside expanders.

### Track A Classification

- Lead with a clean hero/header and safe boundary message.
- Show the selected model, threshold, and task summary up front.
- Keep the error distribution, per-class performance, and threshold behavior visible near the top.
- Put comparison tables and technical evidence in expanders.

### Track B Anomaly Detection

- Lead with a clean hero/header and safe boundary message.
- Show anomaly summary, threshold behavior, and key quality status up front.
- Keep reconstruction and anomaly charts visible near the top.
- Put detailed tables and evidence metadata in expanders.

### YOLO Detection

- Lead with a clean hero/header and safe boundary message.
- Show image count, bbox count, review status, and model identity up front.
- Keep confidence distribution and class summary visible near the top.
- Put artifact lineage, bundle manifest, and evidence files in expanders.

### Upload / Predict

- Lead with a clean hero/header and safe boundary message.
- Make the three-step flow obvious: choose image, confirm API URL, run prediction.
- Show the confidence result visually near the top.
- Put request IDs, run IDs, input metadata, and raw response details in an expander.

### Limitations / Safety

- Keep this page short and explicit.
- Make safety boundaries easy to scan.
- Avoid clutter and avoid technical overload.

### AI Assistant

- Reserve a future AI Assistant page for context-aware explanations.
- The page should be present in the design, even before the backend exists.

## 7. Chart Strategy

- Track A: error distribution donut, per-class bar chart, threshold line chart.
- Track B: anomaly and reconstruction chart, threshold chart.
- YOLO: confidence distribution chart, class summary chart.
- Upload / Predict: confidence chart for the returned probabilities.
- Charts must be visible near the top of each analytical page, not buried below large tables.

## 8. Table And Evidence Strategy

- Show only important tables directly on the page.
- Place detailed tables inside expanders.
- Keep raw JSON hidden by default.
- Keep artifact paths inside technical evidence sections.
- Use tables for evidence and auditability, not as the main visual language.

## 9. Context-Aware Agent UI Strategy

- Include a separate AI Assistant page for future use.
- Add “Explain this page” controls on each page.
- Add “Explain this chart” controls near key charts.
- Add “Explain this prediction” near the Upload / Predict result panel.
- The Agent UI is planned design only for now.
- No backend agent implementation exists in this phase.
- Future explanations must use governed evidence, prediction responses, and safety docs.

## 10. Safety And Claim Boundaries

- not production-ready
- not deployment-safe
- local prototype endpoint
- Track A upload/predict only
- YOLO/anomaly upload prediction not implemented yet
- no fake AI Agent behavior yet

## 11. Implementation Roadmap

- Phase 14I-0: Design contract
- Phase 14I-1: Global layout cleanup and light visual system
- Phase 14I-2: Overview repair
- Phase 14I-3: Track A chart-first repair
- Phase 14I-4: Track B chart-first repair
- Phase 14I-5: YOLO polish
- Phase 14I-6: Upload/Predict polish
- Phase 14I-7: Limitations/Safety page redesign
- Phase 14I-8: AI Assistant placeholder/page
- Phase 14I-9: Full frontend smoke test

## 12. Next Step

Next step after this document is reviewing and committing the contract, then implementing global layout cleanup and the light visual system.

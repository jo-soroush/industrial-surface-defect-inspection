# Frontend Completion Roadmap Before Docker
## Industrial Surface Defect Inspection Platform

## 1. Core Rule For This Roadmap

- [ ] Complete the frontend before starting Docker work.
- [ ] Treat "frontend complete" as a complete, accurate, extensible, presentation-ready first version, not as a weak prototype.
- [ ] Keep all dashboard outputs evidence-backed by governed files, structured model outputs, or clear deterministic rules.
- [ ] Do not show fake metrics, fake predictions, fake decisions, or fake AI behavior.
- [ ] Do not claim production readiness.
- [ ] Do not claim deployment safety.
- [ ] Do not claim an AI Agent exists until a real backend Agent exists.
- [ ] Do not leave an intended dashboard chart, table, box, or status in a "data unavailable" state unless a deliberate design decision removes or replaces that visual.
- [ ] Create missing PR AUC, anomaly scores, reconstruction losses, threshold sweep data, YOLO boxes, sample-level predictions, and related frontend data only through governed pipeline or evidence-generation scripts.
- [ ] Redesign, replace, or remove visuals that are not useful.
- [ ] Preserve technical honesty: dashboard completeness does not mean production readiness.
- [ ] Record unclear items as open questions inside this roadmap or follow-up documentation, not as assumptions.

## 2. High-Level Execution Order

- [ ] Freeze the current frontend scope and inventory all visible problems.
- [ ] Clean up user-facing names, labels, and page titles before deeper redesign.
- [ ] Clean up the Overview page.
- [ ] Clean up the Surface Defect Classification page.
- [ ] Complete missing Surface Anomaly Detection governed evidence.
- [ ] Redesign the Surface Anomaly Detection page around real evidence.
- [ ] Complete Defect Detection & Localization evidence and interpretation.
- [ ] Implement live YOLO detection with boxes before Docker.
- [ ] Redesign Image Inspection as a multi-model inspection workflow.
- [ ] Add a smart decision/model recommendation layer based on real outputs and explicit rules.
- [ ] Update Safety & Limitations to match the completed frontend.
- [ ] Update the AI Explanation Assistant placeholder honestly.
- [ ] Run a full frontend consistency audit.
- [ ] Run a final local smoke test before Docker.
- [ ] Start Docker readiness work only after all frontend completion criteria pass.

## 3. Phase 1 — Freeze Scope And Current Frontend Problems

- [ ] Capture the current page list, current visible labels, and current frontend navigation.
- [ ] Confirm which frontend implementation is the active one used for the local dashboard.
- [ ] Confirm all current frontend data contracts and evidence folders consumed by the dashboard.
- [ ] Inventory every chart, table, status card, metric, text block, image gallery, and upload result currently visible.
- [ ] Mark each visible element as evidence-backed, structured-output-backed, decision-rule-backed, placeholder, unavailable, confusing, or obsolete.
- [ ] Identify every place where Track A or Track B appears as a main visible user-facing label.
- [ ] Identify every place where Upload / Predict appears as a visible user-facing label.
- [ ] Identify every place where YOLO Detection appears as a visible user-facing label.
- [ ] Identify every place where AI Assistant appears as a visible user-facing label.
- [ ] Identify every place where Limitations / Safety appears as a visible user-facing label.
- [ ] Identify every "data unavailable" or equivalent message and decide whether the feature must be completed, redesigned, replaced, or removed.
- [ ] Identify every frontend claim that might imply production readiness, deployment safety, unsupported automation, unsupported agent behavior, or unsupported model quality.
- [ ] Identify every visible metric that may be missing, stale, ungoverned, or confusing.
- [ ] Identify every visual whose purpose is unclear, including the current Visual Status donut chart.
- [ ] Identify every current Image Inspection limitation, including whether it only shows classification.
- [ ] Identify whether probability displays are rounded to only 0 and 1, and whether those values are true raw values or formatting artifacts.
- [ ] Record open questions for any unclear data source, metric meaning, model output, threshold rule, or visual purpose.

## 4. Phase 2 — Professional Naming And Wording Cleanup

- [ ] Rename Track A Classification to Surface Defect Classification in main user-facing UI.
- [ ] Rename Track B Anomaly Detection to Surface Anomaly Detection in main user-facing UI.
- [ ] Rename YOLO Detection to Defect Detection & Localization in main user-facing UI.
- [ ] Rename Upload / Predict to Image Inspection or Live Image Inspection.
- [ ] Rename AI Assistant to AI Explanation Assistant or Evidence Explanation Assistant.
- [ ] Rename Limitations / Safety to Safety & Limitations.
- [ ] Allow Track A and Track B only inside clearly labeled internal technical metadata.
- [ ] Remove Track A and Track B from page titles, navigation labels, visible status cards, section headings, and primary explanatory copy.
- [ ] Replace internal or truncated status wording with readable professional labels.
- [ ] Rewrite "What this dashboard can do" in user-facing language.
- [ ] Rewrite "What it cannot claim yet" in clearer safety language.
- [ ] Review "production-canonical" wording and replace it with language that users can understand.
- [ ] Keep technical metadata available in technical evidence sections when it is useful for traceability.
- [ ] Ensure no naming cleanup weakens or removes required safety limits.

## 5. Phase 3 — Overview Page Cleanup

- [ ] Make the Overview page professional, accurate, and understandable for non-technical reviewers.
- [ ] Replace Track A, Track B, YOLO Detection, Upload / Predict, AI Assistant, and Limitations / Safety labels with approved professional names.
- [ ] Review whether the Visual Status donut chart is useful.
- [ ] Rename or explain the Visual Status donut chart if it remains.
- [ ] Redesign, replace, or remove the Visual Status donut chart if it does not communicate a meaningful governed status.
- [ ] Ensure every overview status card has a real evidence-backed or rule-backed decision behind it.
- [ ] Ensure Overview does not imply production readiness, deployment safety, or automated agent capability.
- [ ] Rewrite high-level capability text around governed evidence and local inspection results.
- [ ] Rewrite high-level limitations text around current safety boundaries.
- [ ] Show classification, anomaly detection, and detection/localization as complementary inspection capabilities.
- [ ] Make the planned final decision/model recommendation layer visible as planned work until implemented.
- [ ] Keep important technical evidence available without making internal labels the main user-facing language.

## 6. Phase 4 — Surface Defect Classification Page Cleanup

- [ ] Remove visible Track A wording from the classification page.
- [ ] Use Surface Defect Classification as the main page label.
- [ ] Replace truncated or internal quality status with readable status text.
- [ ] Improve evidence wording so users understand what the metrics and charts are based on.
- [ ] Improve live-prediction wording so users understand whether results are local model outputs, API outputs, or placeholders.
- [ ] Keep the threshold explanation.
- [ ] Keep existing useful metrics and charts.
- [ ] Confirm each metric is loaded from correct governed evidence or structured output.
- [ ] Confirm each chart is loaded from correct governed evidence or structured output.
- [ ] Move internal metadata, Track A references, bundle IDs, raw paths, and run details into a technical evidence section.
- [ ] Ensure the page does not claim production readiness or deployment safety.
- [ ] Ensure the page does not imply unsupported generalization beyond the current evidence.
- [ ] Confirm probability display precision is appropriate and does not collapse meaningful probabilities into only 0 and 1 unless those are true raw values.
- [ ] Add open questions for any classification evidence, threshold, or quality status that cannot be verified from current files.

## 7. Phase 5 — Surface Anomaly Detection Evidence Completion

- [ ] Audit existing Surface Anomaly Detection governed evidence files.
- [ ] Audit `anomaly_score_summary.json` and determine whether it contains usable frontend display data.
- [ ] Generate or update anomaly scores through governed pipeline or evidence-generation scripts if required.
- [ ] Generate or update reconstruction loss data through governed pipeline or evidence-generation scripts if required.
- [ ] Generate or update PR AUC evidence through governed pipeline or evidence-generation scripts if the metric remains required.
- [ ] Generate or update threshold sweep data through governed pipeline or evidence-generation scripts if threshold behavior remains required.
- [ ] Generate or update sample-level predictions through governed pipeline or evidence-generation scripts if required.
- [ ] Generate or update clear quality decision data through governed pipeline or evidence-generation scripts if required.
- [ ] Generate or update frontend-ready explanation data through governed pipeline or evidence-generation scripts if required.
- [ ] Investigate and explain very high false negatives.
- [ ] Investigate and decide how to present ROC AUC 0.488.
- [ ] Investigate and decide how to present Recall 0.037.
- [ ] Investigate and decide how to present F1 0.071.
- [ ] Decide whether anomaly detection should remain a dashboard capability, be reframed as experimental evidence, or require additional model work before presentation.
- [ ] Ensure PR AUC does not remain unavailable if the visual or metric is retained as required.
- [ ] Ensure anomaly score distribution is based on real governed data.
- [ ] Ensure reconstruction loss chart is based on meaningful real data.
- [ ] Ensure threshold behavior is based on threshold sweep data.
- [ ] Add open questions for any anomaly evidence gaps that cannot be resolved in this phase.

## 8. Phase 6 — Surface Anomaly Detection Page Redesign

- [ ] Remove visible Track B wording from the anomaly page.
- [ ] Use Surface Anomaly Detection as the main page label.
- [ ] Replace confusing production-canonical wording with clear user-facing wording.
- [ ] Present anomaly model quality honestly, especially if ROC AUC, recall, F1, or false negatives remain weak.
- [ ] Show anomaly score distribution only when real governed data exists.
- [ ] Show reconstruction loss chart only when meaningful real data exists.
- [ ] Show threshold behavior only when threshold sweep data exists.
- [ ] Show PR AUC only when governed PR AUC evidence exists.
- [ ] Show sample-level predictions only when governed sample-level evidence exists.
- [ ] Add clear interpretation for weak anomaly results rather than hiding them.
- [ ] Move internal metadata and raw evidence details into a technical evidence section.
- [ ] Avoid any fake metric, fake prediction, or fake quality claim.
- [ ] Redesign, replace, or remove any anomaly visual that remains unhelpful.
- [ ] Ensure no intended anomaly chart says data unavailable after this phase.

## 9. Phase 7 — Defect Detection & Localization Evidence Completion

- [ ] Rename YOLO Detection to Defect Detection & Localization in all main user-facing contexts.
- [ ] Confirm the current evidence includes images.
- [ ] Confirm the current evidence includes bounding boxes.
- [ ] Confirm the current evidence includes confidence distribution.
- [ ] Confirm the current evidence includes class summary.
- [ ] Confirm the current evidence includes bundle manifest.
- [ ] Confirm the current evidence includes run details.
- [ ] Audit whether review_required is present and how it should be shown in user-facing wording.
- [ ] Add interpretation for confidence bands.
- [ ] Add interpretation for class imbalance.
- [ ] Audit raw class names and map them to readable labels where needed.
- [ ] Confirm whether the gallery can show real images with boxes.
- [ ] Create or update governed frontend-ready box overlays only through proper evidence-generation scripts.
- [ ] Confirm that detection/localization evidence does not imply deployment readiness.
- [ ] Add open questions for any missing detection evidence or unclear class naming.

## 10. Phase 8 — Live YOLO Detection With Boxes

- [ ] Implement live Defect Detection & Localization with boxes before Docker.
- [ ] Ensure live detection uses real model output, not fake boxes or placeholder detections.
- [ ] Ensure the uploaded image can be displayed with detected boxes.
- [ ] Ensure box confidence values are displayed with appropriate precision.
- [ ] Ensure box class labels are readable and professional.
- [ ] Ensure review_required or equivalent review status is shown in user-facing wording.
- [ ] Ensure confidence bands are interpreted consistently with governed evidence.
- [ ] Ensure live detection output can be consumed by Image Inspection.
- [ ] Ensure live detection technical evidence is available for inspection.
- [ ] Ensure live detection does not claim production or deployment readiness.
- [ ] Add open questions for any backend/API requirement that must be resolved before live detection can work.

## 11. Phase 9 — Image Inspection Page Full Redesign

- [ ] Rename Upload / Predict to Image Inspection or Live Image Inspection.
- [ ] Redesign the page as a multi-model inspection workflow.
- [ ] Show the uploaded image clearly.
- [ ] Show Surface Defect Classification result.
- [ ] Show Defect Detection & Localization result with boxes.
- [ ] Show Surface Anomaly Detection result.
- [ ] Show model summaries for all three inspection methods.
- [ ] Show technical evidence for all three inspection methods.
- [ ] Add a final smart decision or recommendation result.
- [ ] Ensure the final decision is based on real model outputs and clear decision rules.
- [ ] Ensure the final decision does not pretend to be a fake AI Agent.
- [ ] Ensure classification probability display uses appropriate precision and does not show only 0 and 1 unless those are true raw values.
- [ ] Ensure YOLO box outputs are visible on the uploaded image.
- [ ] Ensure anomaly result includes score, reconstruction loss, threshold context, and quality caveat where governed evidence supports them.
- [ ] Ensure each result section states whether it is based on live local output, governed evidence, or structured API output.
- [ ] Hide raw debug details by default but keep technical evidence accessible.
- [ ] Ensure no result panel claims production readiness or deployment safety.

## 12. Phase 10 — Smart Decision / Model Recommendation Layer

- [ ] Add a smart decision/model recommendation layer.
- [ ] Compare Surface Defect Classification output, Defect Detection & Localization output, and Surface Anomaly Detection output.
- [ ] Recommend or decide the final inspection result.
- [ ] Base the recommendation on real model outputs and explicit decision rules.
- [ ] Document the decision rules in frontend-accessible technical evidence or documentation.
- [ ] Clearly distinguish deterministic decision logic from an AI Agent.
- [ ] Handle agreement between classification, detection/localization, and anomaly detection.
- [ ] Handle disagreement between classification, detection/localization, and anomaly detection.
- [ ] Handle low-confidence results.
- [ ] Handle missing or failed model output without fabricating a decision.
- [ ] Surface review-required outcomes in professional user-facing wording.
- [ ] Include thresholds and confidence interpretation where relevant.
- [ ] Include safety boundaries and limitations in the final recommendation display.
- [ ] Ensure the recommendation does not claim production readiness or deployment safety.
- [ ] Add open questions for rule conflicts that require domain review.

## 13. Phase 11 — Safety & Limitations Page Update

- [ ] Rename the page to Safety & Limitations.
- [ ] Remove outdated Track A-only wording after Image Inspection is complete.
- [ ] Keep the no production-ready claim.
- [ ] Keep the no deployment-safe claim.
- [ ] Keep the no fake AI claim.
- [ ] Replace weak "scaffold" wording.
- [ ] Explain that the dashboard presents governed evidence and local inspection results.
- [ ] Explain that the dashboard does not train models directly.
- [ ] Explain that the dashboard does not silently recompute metrics.
- [ ] Explain that new evidence files must be created by governed pipeline scripts.
- [ ] Explain that dashboard completeness is not the same as production readiness.
- [ ] Explain that live inspection is local/demo scope unless a later governed deployment path is created.
- [ ] Explain that the decision/recommendation layer is rule-based unless a real Agent is later implemented.
- [ ] Ensure wording matches actual completed frontend behavior.
- [ ] Add open questions for safety claims that require review before Docker.

## 14. Phase 12 — AI Explanation Assistant Placeholder Update

- [ ] Rename AI Assistant to AI Explanation Assistant or Evidence Explanation Assistant.
- [ ] Keep placeholder warning until a backend Agent exists.
- [ ] Clearly state that no real backend Agent exists if that remains true.
- [ ] Update future scope so the assistant can later explain classification results.
- [ ] Update future scope so the assistant can later explain YOLO boxes.
- [ ] Update future scope so the assistant can later explain anomaly results.
- [ ] Update future scope so the assistant can later explain the final decision layer.
- [ ] Update future scope so the assistant can later explain safety boundaries.
- [ ] Ensure the placeholder does not invent metrics.
- [ ] Ensure the placeholder does not claim production readiness.
- [ ] Ensure the placeholder does not claim deployment readiness.
- [ ] Ensure the placeholder does not imply autonomous inspection authority.
- [ ] Keep future scope clear, useful, and bounded by governed evidence.

## 15. Phase 13 — Full Frontend Consistency Audit

- [ ] Verify no internal Track A or Track B names appear as main UI labels.
- [ ] Verify Track A and Track B remain only inside clearly labeled internal technical metadata if present.
- [ ] Verify Surface Defect Classification naming is consistent.
- [ ] Verify Surface Anomaly Detection naming is consistent.
- [ ] Verify Defect Detection & Localization naming is consistent.
- [ ] Verify Image Inspection or Live Image Inspection naming is consistent.
- [ ] Verify AI Explanation Assistant or Evidence Explanation Assistant naming is consistent.
- [ ] Verify Safety & Limitations naming is consistent.
- [ ] Verify every visible chart has real governed data.
- [ ] Verify every visible table comes from correct evidence or structured output.
- [ ] Verify every status box has a real decision behind it.
- [ ] Verify every user-facing label is professional and understandable.
- [ ] Verify no intended chart says data unavailable.
- [ ] Verify no important metric is missing without a deliberate design decision.
- [ ] Verify no fake metric claim exists.
- [ ] Verify no fake prediction claim exists.
- [ ] Verify no fake AI claim exists.
- [ ] Verify no production-ready claim exists.
- [ ] Verify no deployment-safe claim exists.
- [ ] Verify the Image Inspection page includes classification, YOLO boxes, anomaly result, model summaries, technical evidence, and final decision.
- [ ] Verify probability precision is readable and truthful.
- [ ] Verify technical evidence sections preserve traceability without overwhelming primary UI.
- [ ] Verify safety wording matches actual frontend behavior.

## 16. Phase 14 — Final Local Smoke Test Before Docker

- [ ] Run the frontend locally.
- [ ] Confirm the Overview page loads without errors.
- [ ] Confirm the Surface Defect Classification page loads without errors.
- [ ] Confirm the Surface Anomaly Detection page loads without errors.
- [ ] Confirm the Defect Detection & Localization page loads without errors.
- [ ] Confirm the Image Inspection page loads without errors.
- [ ] Confirm the Safety & Limitations page loads without errors.
- [ ] Confirm the AI Explanation Assistant page loads without errors.
- [ ] Upload or select a valid test image for Image Inspection.
- [ ] Confirm Image Inspection shows the uploaded image clearly.
- [ ] Confirm Image Inspection shows classification result.
- [ ] Confirm Image Inspection shows YOLO detection result with boxes.
- [ ] Confirm Image Inspection shows anomaly result.
- [ ] Confirm Image Inspection shows final decision or recommendation.
- [ ] Confirm no frontend page has broken charts.
- [ ] Confirm no frontend page has missing intended data.
- [ ] Confirm no frontend page has misleading production, deployment, AI Agent, metric, or prediction claims.
- [ ] Record smoke test command, date, outcome, and any remaining issues.

## 17. Phase 15 — Docker Readiness After Frontend Completion

- [ ] Confirm every frontend completion criterion is satisfied before starting Docker.
- [ ] Confirm no Phase 1 through Phase 14 blocker remains unresolved.
- [ ] Confirm local smoke test passes.
- [ ] Confirm frontend data dependencies are documented.
- [ ] Confirm API dependencies for live inspection are documented.
- [ ] Confirm evidence-generation dependencies are documented.
- [ ] Confirm safety and limitation wording is final for the first Docker demo.
- [ ] Confirm Docker work will package the completed local frontend/API demo only, not claim production deployment.
- [ ] Start Docker planning only after this checklist is complete.

## 18. Master Grouped Fix List

- [ ] Global: complete frontend before Docker.
- [ ] Global: every visible chart must have real governed data.
- [ ] Global: every table must come from correct evidence or structured output.
- [ ] Global: every box or status must have a real decision behind it.
- [ ] Global: every user-facing label must be professional and understandable.
- [ ] Global: no important visual should say data unavailable if it is part of the intended dashboard.
- [ ] Global: create missing required evidence through governed pipeline/evidence generation.
- [ ] Global: redesign, replace, or remove visuals that are not useful.
- [ ] Global: avoid production-ready, deployment-safe, fake AI Agent, fake metric, and fake prediction claims.
- [ ] Naming: replace Track A Classification with Surface Defect Classification.
- [ ] Naming: replace Track B Anomaly Detection with Surface Anomaly Detection.
- [ ] Naming: replace YOLO Detection with Defect Detection & Localization.
- [ ] Naming: replace Upload / Predict with Image Inspection or Live Image Inspection.
- [ ] Naming: replace AI Assistant with AI Explanation Assistant or Evidence Explanation Assistant.
- [ ] Naming: replace Limitations / Safety with Safety & Limitations.
- [ ] Classification: remove visible Track A wording.
- [ ] Classification: replace truncated/internal quality status with readable status.
- [ ] Classification: improve evidence/live-prediction wording.
- [ ] Classification: keep threshold explanation.
- [ ] Classification: keep metrics and charts.
- [ ] Classification: move internal metadata into technical evidence.
- [ ] Anomaly: remove visible Track B wording.
- [ ] Anomaly: ensure PR AUC is governed or remove/rethink the visual by deliberate design.
- [ ] Anomaly: add real governed anomaly score distribution data.
- [ ] Anomaly: add meaningful reconstruction loss data.
- [ ] Anomaly: add threshold sweep data.
- [ ] Anomaly: investigate high false negatives.
- [ ] Anomaly: investigate ROC AUC 0.488, Recall 0.037, and F1 0.071.
- [ ] Anomaly: fix confusing production-canonical wording.
- [ ] Anomaly: audit `anomaly_score_summary.json`.
- [ ] Anomaly: provide anomaly scores, reconstruction losses, PR AUC, threshold sweep, sample-level predictions, clear quality decision, and frontend-ready explanation data where retained.
- [ ] Detection: rename YOLO Detection to Defect Detection & Localization.
- [ ] Detection: preserve current evidence for images, boxes, confidence distribution, class summary, bundle manifest, and run details.
- [ ] Detection: show review_required in user-facing wording.
- [ ] Detection: interpret confidence bands.
- [ ] Detection: interpret class imbalance.
- [ ] Detection: map raw class names to readable labels where needed.
- [ ] Detection: show real images with boxes in the gallery if possible.
- [ ] Detection: implement live YOLO detection with boxes before Docker.
- [ ] Image Inspection: rename Upload / Predict.
- [ ] Image Inspection: replace classification-only behavior with multi-model inspection.
- [ ] Image Inspection: show uploaded image clearly.
- [ ] Image Inspection: show classification result.
- [ ] Image Inspection: show YOLO detection result with boxes.
- [ ] Image Inspection: show anomaly result.
- [ ] Image Inspection: show model summaries for all three.
- [ ] Image Inspection: show technical evidence for all three.
- [ ] Image Inspection: include final smart decision/recommendation.
- [ ] Image Inspection: fix probability display precision.
- [ ] Safety: remove outdated Track A-only wording after Image Inspection is complete.
- [ ] Safety: keep no production-ready, no deployment-safe, and no fake AI claims.
- [ ] Safety: replace weak scaffold wording.
- [ ] Safety: explain governed evidence and local inspection results.
- [ ] Safety: explain dashboard does not train models directly or silently recompute metrics.
- [ ] Safety: explain new evidence files must be created by governed pipeline scripts.
- [ ] AI Explanation Assistant: keep placeholder warning until backend Agent exists.
- [ ] AI Explanation Assistant: update future scope for classification, YOLO boxes, anomaly result, final decision layer, and safety boundaries.
- [ ] AI Explanation Assistant: do not invent metrics or claim production/deployment readiness.

## 19. Commit Strategy

- [ ] Do not commit this roadmap during Phase 1 unless explicitly requested later.
- [ ] Commit frontend naming and wording cleanup separately from evidence-generation changes.
- [ ] Commit anomaly evidence-generation changes separately from anomaly page redesign.
- [ ] Commit detection evidence-generation changes separately from live detection UI changes.
- [ ] Commit Image Inspection multi-model redesign separately from the smart decision layer if the diff becomes large.
- [ ] Commit Safety & Limitations and AI Explanation Assistant copy updates separately or together if they are small and tightly related.
- [ ] Commit final frontend consistency audit fixes separately from Docker work.
- [ ] Do not mix Docker files into frontend completion commits.
- [ ] Do not commit generated artifacts unless they are intended governed evidence outputs and the repository policy allows them.
- [ ] Keep commit messages explicit about whether a change is documentation, frontend UI, evidence generation, API integration, or governed evidence.

## 20. Final Definition Of Complete Frontend Before Docker

- [ ] Overview is professional and accurate.
- [ ] Surface Defect Classification page is clean and evidence-backed.
- [ ] Surface Anomaly Detection page has complete real evidence and meaningful visuals.
- [ ] Defect Detection & Localization page has complete evidence and clear interpretation.
- [ ] Image Inspection supports classification result, YOLO boxes, anomaly result, and final decision layer.
- [ ] Safety page matches the actual system.
- [ ] AI Explanation Assistant remains honest and clearly planned.
- [ ] No internal Track A or Track B names appear as main UI labels.
- [ ] No intended chart says data unavailable.
- [ ] No important metric is missing without a deliberate design decision.
- [ ] No fake claim exists.
- [ ] No production-ready claim exists.
- [ ] No deployment-safe claim exists.
- [ ] Local smoke test passes.
- [ ] Only then Docker can start.

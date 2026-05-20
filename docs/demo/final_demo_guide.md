# Final Demo Guide

## 1. Purpose

This guide helps present the current evidence-backed project state for teacher/demo review. It does not introduce new evidence.

This guide does not claim production readiness or deployment safety. It also does not require a real frontend, API, or agent layer.

## 2. Recommended Demo Order

1. Start with `docs/demo/teacher_handoff_summary.md`
2. Open `docs/demo/evidence_index.md`
3. Open `docs/status/project_wide_decision_summary.md`
4. Show Track A notebook: `notebooks/track_a_supervised_classification_mvtec.ipynb`
5. Show Track B notebook: `notebooks/track_b_anomaly_detection_mvtec_evidence.ipynb`
6. Show YOLO evidence through either `notebooks/detection_yolo_gc10det_evidence.ipynb` or `artifacts/frontend/detection/yolo_train_v0_2_0/`
7. Optionally show `artifacts/models/registry/artifact_registry.yaml` for governance traceability

## 3. Opening Explanation

Suggested presenter wording:

This project is an industrial surface defect inspection platform. It evaluates three evidence layers: supervised classification, anomaly detection, and object detection. The focus of this demo is governed evidence and honest model status, not a deployed production product.

## 4. Track A Demo Script

Talking points:

- Track A Classification status: PASS
- Selected model: ResNet18 v0.4.0
- Selected run ID: `1bc92561-c5bf-48f2-8246-b8f3d5718ffe`
- Recommended threshold: `0.65`
- MLP role: baseline-only comparison evidence
- CNN role: failed-quality / comparison evidence
- Notebook: `notebooks/track_a_supervised_classification_mvtec.ipynb`
- Frontend data contract folder: `artifacts/frontend/track_a/`

Safe wording:

- Track A is accepted as governed demo evidence.
- Track A is not production-ready.
- Track A is not deployment-safe.

## 5. Track B Demo Script

Talking points:

- Track B / Autoencoder status: PASS
- Autoencoder governed evidence is available.
- Frontend bundle is complete.
- Notebook refresh is complete.
- PR AUC is unavailable in governed evidence and is not fabricated.
- Notebook: `notebooks/track_b_anomaly_detection_mvtec_evidence.ipynb`
- Frontend data contract folder: `artifacts/frontend/track_b/`

Safe wording:

- Track B is accepted as governed demo evidence.
- Track B is not production-ready.
- Track B is not deployment-safe.

## 6. YOLO / Detection Demo Script

Talking points:

- YOLO / Detection evidence layer: COMPLETE
- Structured artifacts are complete:
  - bbox predictions
  - per-image summary
  - confidence distribution
  - sample gallery
- Inventories and registry entries are complete for the governed Detection structured artifacts.
- Detection frontend data-contract generator is complete.
- YOLO notebook: `notebooks/detection_yolo_gc10det_evidence.ipynb`
- Detection frontend data-contract folder: `artifacts/frontend/detection/yolo_train_v0_2_0/`
- YOLO notebook refresh is optional and skipped for now.

Safe wording:

- YOLO evidence is complete as evidence/data-contract work.
- YOLO is not production-ready.
- YOLO is not deployment-safe.

## 7. Governance And Traceability Talking Points

- Claims are supported by artifacts, inventories, registries, and status documents.
- Frontend data-contract folders are JSON evidence bundles, not UI.
- Generated artifacts under `artifacts/` may be ignored by Git under the current repository policy.
- Ignored generated artifacts should not be assumed committed unless explicitly registered or force-added under a later policy decision.
- No unsupported production or deployment claims are made.

## 8. What Not To Say In The Demo

- Do not say the system is production-ready.
- Do not say it is deployment-safe.
- Do not say a real frontend/dashboard app exists.
- Do not say API endpoints exist.
- Do not say an agent layer exists.
- Do not say YOLO notebook has been refreshed to the new detection frontend bundle.
- Do not say additional training is needed now unless a future audit finds a specific gap.

## 9. Expected Questions And Safe Answers

**Is this production-ready?**

No. The project has governed demo evidence, but production readiness is not claimed.

**Is there a real frontend?**

No. The repository has frontend-ready JSON data contracts for future dashboard work, but not a real frontend/dashboard application.

**Why is YOLO notebook not refreshed?**

The YOLO notebook is safe and presents governed YOLO evidence. Refreshing it to consume the new Detection frontend bundle is optional and was skipped to avoid notebook churn before packaging.

**Why is PR AUC unavailable for Track B?**

PR AUC is not available in the governed Track B evidence. It is intentionally not fabricated or inferred.

**What is the strongest Track A model?**

ResNet18 v0.4.0 is the selected Track A candidate, with run ID `1bc92561-c5bf-48f2-8246-b8f3d5718ffe` and recommended threshold `0.65`.

**What is the next step?**

Finish the final packaging docs, then decide whether README entrypoint cleanup is needed. Frontend, API, and agent work should only start if explicitly selected after packaging.

## 10. Recommended Next Step After Demo

Continue with `docs/demo/final_limitations_and_next_steps.md`.

After that, optionally update the README entrypoint if needed. Do not start frontend, API, or agent work unless explicitly selected after packaging.

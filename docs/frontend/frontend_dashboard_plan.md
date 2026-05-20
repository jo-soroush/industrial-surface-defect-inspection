# Frontend Dashboard Plan

## 1. Goal

Define the frontend, API, and Docker direction before implementation starts. The immediate target is a company-grade demo dashboard that presents the current governed evidence clearly and safely.

## 2. Current Project State

- Track A Classification: PASS
- Track B / Autoencoder: PASS
- YOLO / Detection evidence layer: COMPLETE
- Track A notebook: refreshed
- Track B notebook: refreshed
- YOLO notebook: refreshed
- Frontend data contracts exist for Track A, Track B, and Detection
- Production readiness: NOT CLAIMED
- Deployment safety: NOT CLAIMED
- Real frontend/dashboard app: NOT STARTED / NOT VALIDATED
- API endpoints: NOT STARTED
- Agent layer: NOT STARTED
- No retraining is planned now unless a future audit identifies a concrete blocker

## 3. Frontend Strategy

Use Streamlit as the first frontend because the repository already has Python-based evaluation and governance tooling, and the current demo goal is a read-only dashboard that consumes existing JSON data contracts.

The first frontend should:

- read the Track A, Track B, and Detection JSON bundles from disk
- present governed evidence and model status
- avoid live inference until a backend API exists
- avoid production and deployment claims
- avoid agent integration

## 4. Streamlit Dashboard Pages

The first Streamlit app should include the following pages:

- Overview
- Track A Classification
- Track B Anomaly Detection
- YOLO Detection
- Upload / Predict
- Limitations / Safety

Page intent:

- Overview: project snapshot, status cards, and safe limitations summary
- Track A Classification: selected model, metrics, charts, recommendation, and sample evidence
- Track B Anomaly Detection: anomaly summary, threshold behavior, metrics, and sample evidence
- YOLO Detection: overview, model metadata, confidence distribution, class summary, lineage, and sample gallery
- Upload / Predict: placeholder page for future API-backed inference flow
- Limitations / Safety: explicit non-production, non-deployment, and evidence-only boundaries

## 5. Data Sources

The dashboard should read these existing JSON data contracts:

- `artifacts/frontend/track_a/`
- `artifacts/frontend/track_b/`
- `artifacts/frontend/detection/yolo_train_v0_2_0/`

The frontend should treat these folders as JSON evidence/data contracts, not as a UI implementation and not as source-of-truth training state.

## 6. API Strategy

An API is needed later if the dashboard should support live upload/predict behavior.

The first API should be simple:

- one or two upload/predict endpoints
- minimal request/response schemas
- no production-readiness claim
- no deployment-safety claim

The API should not be treated as complete, hardened, or production-ready at the start.

## 7. Docker Strategy

Docker comes after the local frontend and API work are stable.

The Docker goal should be:

- a container for the frontend
- a container for the API
- docker-compose to run both locally

Docker and docker-compose should support demo repeatability, not production deployment claims.

## 8. Implementation Phases

- Phase 1: Plan
- Phase 2: Streamlit scaffold
- Phase 3: Data loader
- Phase 4: Dashboard pages
- Phase 5: Charts and polish
- Phase 6: API upload/predict
- Phase 7: Docker
- Phase 8: Final audit

Phase intent:

- Phase 1 defines structure and boundaries.
- Phase 2 creates the frontend shell only.
- Phase 3 loads the existing JSON bundles.
- Phase 4 adds the dashboard pages.
- Phase 5 adds charts, layout cleanup, and presentation polish.
- Phase 6 introduces a simple API for upload/predict.
- Phase 7 adds Docker and docker-compose.
- Phase 8 checks safety wording and demo readiness.

## 9. Boundaries

- No production-ready claim.
- No deployment-safe claim.
- No fake live prediction.
- No API claim until implemented.
- No agent claim.
- No retraining unless required by a future audit.

## 10. Next Step

The next step after this document is creating the Streamlit scaffold.

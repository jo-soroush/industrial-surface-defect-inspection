# Industrial Surface Defect Inspection Platform - Master Roadmap

## 1. Current Executive Status

- [x] Track A Classification: PASS
- [x] Track B / Autoencoder: PASS
- [x] YOLO / Detection evidence layer: COMPLETE
- [x] Frontend/dashboard evidence surfaces: IMPLEMENTED AS EVIDENCE/DEMO/MOCK-FIRST LAYER
- [x] API routes for inspection, prediction, and agent explainability: IMPLEMENTED AS EVIDENCE/DEMO/MOCK-FIRST LAYER
- [x] Agent/RAG foundation: IMPLEMENTED AS EVIDENCE/DEMO/MOCK-FIRST LAYER
- [ ] Production readiness: NOT CLAIMED
- [ ] Deployment safety: NOT CLAIMED
- [x] Gemini gated frontend integration milestone: validated as an explicit opt-in local path; safe mock fallback remains available

The repository currently has governed evidence for the classification, anomaly detection, and object detection tracks, plus implemented mock-first frontend, API, and agent foundations. This status does not make any production-readiness or deployment-safety claim.
The Gemini gated frontend integration milestone is validated as an explicit opt-in local path. The runtime remains mock-first by default and safe mock fallback remains available.
The latest approved Gemini 2.5 Flash smoke reached `SUCCESS_LIMITED`, used `provider_used=gemini`, used `fallback_used=false`, and kept `grounding_status=grounded` with `safety_status=limited`.
The frontend Image Inspection path now reaches `/agent/explain`, and the gated Gemini provider path is available only when explicit runtime gates are set.
The explicit gated runtime gates are `AGENT_ENABLE_LLM=true`, `AGENT_ENABLE_REAL_PROVIDER_RUNTIME=true`, `AGENT_DEFAULT_PROVIDER=gemini`, `LLM_PROVIDER_ORDER=gemini,mock`, `LLM_ENABLE_FALLBACK=true`, `LLM_MAX_RETRIES=1`, with `GEMINI_API_KEY` provided only at runtime and never committed.
The canonical Gemini provider default is aligned to `gemini-2.5-flash`, matching the validated smoke model.

## 2. Source-of-Truth Rules

- Scripts, configs, governed artifacts, inventories, and registries are the source of truth for project state.
- Notebooks are evidence and presentation layers only. They must not be treated as source-of-truth execution state.
- Frontend bundles are data contracts for later dashboard work. They are not a real frontend UI.
- A model must not be called production-ready without a governed quality decision that explicitly supports that status.
- Unsupported deployment-safety claims are not allowed.
- Generated artifacts under `artifacts/` remain ignored under the current repository policy unless a later explicit policy change says otherwise.

## 3. Track A Status

- Status: PASS
- Selected candidate: ResNet18 v0.4.0
- Selected run ID: `1bc92561-c5bf-48f2-8246-b8f3d5718ffe`
- Recommendation status: `selected`
- Recommended threshold: `0.65`
- `production_ready`: false
- `deployment_candidate`: false
- MLP evidence: baseline-only comparison evidence
- CNN evidence: failed-quality / comparison evidence
- Track A comparison: complete
- Track A frontend-ready bundle: complete
- Track A notebook refresh and quality pass: complete
- Track A demo readiness: PASS

Track A is ready as governed demo evidence. It is not a production-readiness or deployment-safety claim.

## 4. Track B Status

- Status: PASS
- Autoencoder governed/canonical evidence: exists
- Track B frontend-ready bundle: complete
- Track B notebook refresh and quality pass: complete
- Track B demo readiness: PASS
- PR AUC: unavailable in governed evidence and must not be fabricated
- `production_ready`: false unless governed evidence explicitly says otherwise
- `deployment_candidate`: false unless governed evidence explicitly says otherwise

Track B is ready as governed demo evidence. Missing metrics must remain honestly unavailable when they are not present in governed artifacts.

## 5. YOLO / Detection Status

- Status: YOLO / Detection evidence layer COMPLETE
- Bbox prediction artifact: complete
- Bbox prediction inventory: complete
- Bbox prediction registry entry: complete
- Per-image summary artifact: complete
- Per-image summary inventory: complete
- Per-image summary registry entry: complete
- Confidence distribution artifact: complete
- Confidence distribution inventory: complete
- Confidence distribution registry entry: complete
- Sample gallery artifact: complete
- Sample gallery inventory: complete
- Sample gallery registry entry: complete
- Detection frontend-ready data-contract generator: complete
- Generated detection frontend bundle: exists as ignored local evidence/data contract when generated

This is evidence/data-contract completion only.

- It is not real frontend/UI work.
- It is not API work.
- It is not notebook update work.
- It is not production readiness.
- It is not deployment safety.

## 6. Frontend/Data Contract Status

- Track A frontend bundle: complete
- Track B frontend bundle: complete
- Detection frontend data-contract generator: complete
- Detection generated frontend bundle: local ignored data-contract evidence when generated
- Real frontend/dashboard application: not production/public/deployment-ready; only the evidence/demo surfaces are implemented

Frontend bundle JSON files exist to support future dashboard development. They do not constitute a user-facing frontend application.

## 7. Notebook Evidence Status

- Track A notebook: refreshed / PASS
- Track B notebook: refreshed / PASS
- YOLO notebook: refreshed and now consumes `artifacts/frontend/detection/yolo_train_v0_2_0/`
- Notebook role: read-only evidence and presentation layer

Notebook work must not train models, create governed artifacts, update registries, or rewrite source-of-truth state unless a later explicit task defines that behavior.

## 8. Remaining Roadmap Sections

- [x] Roadmap/status canonicalization: current status should be read from the implemented evidence/demo/mock-first layers above.
- [x] Design a gated real-provider runtime activation path and validate it with explicit opt-in only.
- [ ] Decide whether any later public frontend/dashboard hardening is needed.
- [ ] Decide whether any later public API hardening is needed.
- [ ] Prepare final report/demo packaging.

These roadmap sections now reflect that the gated frontend-to-agent path has been validated locally with explicit opt-in only. They are still not a claim that production hardening, public deployment, or always-on live-provider runtime has already occurred.

## 9. Completed Milestones

- [x] Track A comparison
- [x] ResNet18 governed package
- [x] Track A frontend bundle
- [x] Track A notebook pass
- [x] Track B frontend bundle
- [x] Track B notebook pass
- [x] YOLO structured artifacts, inventories, and registry entries
- [x] Detection frontend bundle generator
- [x] Detection evidence layer completion status update

## 10. Current Recommended Next Step

Treat the roadmap/status docs as the source of truth for the current implemented evidence/demo/mock-first layers. The gated frontend-to-agent Gemini path is now validated locally as an explicit opt-in route with safe fallback. If implementation work resumes later, the next technical track is operational documentation for that gated mode or later deployment/env handling, not another default-on Gemini activation step.

Do not run another YOLO structured-output readiness audit as the next step; the YOLO / Detection evidence layer is complete. Do not run another Gemini local smoke as the next step. Do not restart UI, API, notebook, agent, Docker, EC2, or real-LLM implementation from this roadmap update.

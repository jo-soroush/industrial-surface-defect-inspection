# Industrial Surface Defect Inspection Platform - Master Roadmap

## 1. Current Executive Status

- [x] Track A Classification: PASS
- [x] Track B / Autoencoder: PASS
- [x] YOLO / Detection evidence layer: COMPLETE
- [ ] Full frontend/dashboard app: NOT STARTED / NOT VALIDATED
- [ ] API: NOT STARTED
- [ ] Agent layer: NOT STARTED
- [ ] Production readiness: NOT CLAIMED
- [ ] Deployment safety: NOT CLAIMED

The repository currently has governed evidence for the classification, anomaly detection, and object detection tracks. This status does not make any production-readiness or deployment-safety claim.

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
- Real frontend/dashboard application: not started / not validated

Frontend bundle JSON files exist to support future dashboard development. They do not constitute a user-facing frontend application.

## 7. Notebook Evidence Status

- Track A notebook: refreshed / PASS
- Track B notebook: refreshed / PASS
- YOLO notebook: not updated yet unless separately audited
- Notebook role: read-only evidence and presentation layer

Notebook work must not train models, create governed artifacts, update registries, or rewrite source-of-truth state unless a later explicit task defines that behavior.

## 8. Remaining Roadmap Sections

- [ ] Perform a final project-wide status/demo readiness audit.
- [ ] Decide whether to refresh the YOLO notebook.
- [ ] Decide whether to start the full frontend/dashboard application.
- [ ] Decide whether to create API endpoints.
- [ ] Decide whether to create an agent layer.
- [ ] Prepare final report/demo packaging.

These roadmap sections have not started as implementation work in this status update.

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

Perform a project-wide post-YOLO status audit and choose the next roadmap section.

Do not run another YOLO structured-output readiness audit as the next step; the YOLO / Detection evidence layer is complete. Do not start frontend, API, notebook, or agent implementation inside this roadmap update.

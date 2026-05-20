# Evidence Index

## 1. Purpose

This file maps the current project evidence for teacher/demo review. It points reviewers to the status documents, notebooks, governed artifacts, registries, and frontend-ready data contracts that support the current handoff.

This file does not introduce new evidence. It does not claim production readiness or deployment safety.

## 2. Status And Roadmap Documents

- `docs/demo/teacher_handoff_summary.md`
  - Concise reviewer entrypoint for the current demo/handoff state.
- `docs/status/master_roadmap.md`
  - Official repository roadmap and current project direction.
- `docs/status/project_wide_decision_summary.md`
  - Consolidated project-wide decision/status summary after Track A, Track B, and YOLO evidence completion.
- `docs/status/phase3_detection_pause_status.md`
  - Current Detection / YOLO evidence-layer completion status and safety boundaries.

## 3. Track A Evidence

- Status: PASS
- Selected model: ResNet18 v0.4.0
- Selected run ID: `1bc92561-c5bf-48f2-8246-b8f3d5718ffe`
- Notebook: `notebooks/track_a_supervised_classification_mvtec.ipynb`
- Frontend data contract folder: `artifacts/frontend/track_a/`
- `production_ready`: false
- `deployment_candidate`: false

Important Track A frontend-ready JSON files:

- `artifacts/frontend/track_a/artifact_inventory_frontend.json`
- `artifacts/frontend/track_a/confusion_matrix_chart_data.json`
- `artifacts/frontend/track_a/error_distribution_pie_data.json`
- `artifacts/frontend/track_a/frontend_model_recommendation.json`
- `artifacts/frontend/track_a/metric_cards.json`
- `artifacts/frontend/track_a/model_comparison_table.json`
- `artifacts/frontend/track_a/per_class_bar_chart_data.json`
- `artifacts/frontend/track_a/quality_decision_summary.json`
- `artifacts/frontend/track_a/sample_predictions_gallery.json`
- `artifacts/frontend/track_a/threshold_curve_chart_data.json`

The Track A frontend folder is data/evidence for review and dashboard consumption. It is not a real user interface.

## 4. Track B Evidence

- Status: PASS
- Model family: Autoencoder
- Notebook: `notebooks/track_b_anomaly_detection_mvtec_evidence.ipynb`
- Frontend data contract folder: `artifacts/frontend/track_b/`
- PR AUC: unavailable in governed evidence and not fabricated
- Production readiness: not claimed
- Deployment safety: not claimed

Important Track B frontend-ready JSON files:

- `artifacts/frontend/track_b/anomaly_score_summary.json`
- `artifacts/frontend/track_b/artifact_inventory_frontend.json`
- `artifacts/frontend/track_b/frontend_anomaly_summary.json`
- `artifacts/frontend/track_b/metric_cards.json`
- `artifacts/frontend/track_b/quality_decision_summary.json`
- `artifacts/frontend/track_b/reconstruction_loss_summary.json`
- `artifacts/frontend/track_b/sample_anomaly_gallery.json`
- `artifacts/frontend/track_b/threshold_behavior.json`

The Track B frontend folder is data/evidence for review and dashboard consumption. It is not a real user interface.

## 5. YOLO / Detection Evidence

- Status: evidence layer COMPLETE
- Notebook: `notebooks/detection_yolo_gc10det_evidence.ipynb`
- Notebook refresh status: optional and skipped for now

Structured artifacts:

- `artifacts/models/predictions/detection_bbox_predictions__yolo_train_v0_2_0__validation.json`
- `artifacts/models/predictions/detection_per_image_summary__yolo_train_v0_2_0__validation.json`
- `artifacts/models/predictions/detection_confidence_distribution__yolo_train_v0_2_0__validation.json`
- `artifacts/models/predictions/detection_sample_gallery__yolo_train_v0_2_0__validation.json`

Inventories:

- `artifacts/models/inventory/track_detection_bbox_prediction_artifact_inventory__yolo_train_v0_2_0__validation.json`
- `artifacts/models/inventory/track_detection_per_image_summary_artifact_inventory__yolo_train_v0_2_0__validation.json`
- `artifacts/models/inventory/track_detection_confidence_distribution_artifact_inventory__yolo_train_v0_2_0__validation.json`
- `artifacts/models/inventory/track_detection_sample_gallery_artifact_inventory__yolo_train_v0_2_0__validation.json`

Frontend data contract generator:

- `scripts/evaluation/generate_detection_frontend_bundle.py`

Frontend data contract folder:

- `artifacts/frontend/detection/yolo_train_v0_2_0/`

Important Detection frontend-ready JSON files:

- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_artifact_lineage.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_class_summary.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_confidence_chart.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_metric_cards.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_model_metadata.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_overview.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_quality_decision_summary.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/detection_sample_gallery.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/frontend_bundle_manifest.json`
- `artifacts/frontend/detection/yolo_train_v0_2_0/frontend_detection_recommendation.json`

Detection frontend data contracts are not a real frontend/UI implementation. Where present, Detection frontend bundle fields preserve `production_ready=false`, `deployment_candidate=false`, and `review_required=true`.

## 6. Registry And Governance Evidence

- `artifacts/models/registry/artifact_registry.yaml`

The artifact registry links governed artifacts to hashes, sizes, metadata, and active artifact identifiers. Generated artifacts under `artifacts/` may be ignored by Git under the current repository policy. This index does not imply ignored generated artifacts are committed.

## 7. Notebooks

- `notebooks/track_a_supervised_classification_mvtec.ipynb`
- `notebooks/track_b_anomaly_detection_mvtec_evidence.ipynb`
- `notebooks/detection_yolo_gc10det_evidence.ipynb`

Track A and Track B notebooks are refreshed and quality-pass complete. The YOLO notebook is safe but has not been refreshed to consume the new Detection frontend bundle. Notebooks are presentation/evidence layers, not the source of truth.

## 8. Frontend Data Contracts

- `artifacts/frontend/track_a/`
- `artifacts/frontend/track_b/`
- `artifacts/frontend/detection/yolo_train_v0_2_0/`

These folders contain JSON evidence/data-contract bundles for dashboard, report, and demo consumption. They are not a real frontend application.

## 9. What Is Not Included

- No real frontend/dashboard app
- No API service
- No agent layer
- No production deployment
- No deployment-safety approval
- No additional training recommended now

## 10. Recommended Reading Order

1. `docs/demo/teacher_handoff_summary.md`
2. `docs/status/project_wide_decision_summary.md`
3. `docs/status/master_roadmap.md`
4. `notebooks/track_a_supervised_classification_mvtec.ipynb`
5. `notebooks/track_b_anomaly_detection_mvtec_evidence.ipynb`
6. `notebooks/detection_yolo_gc10det_evidence.ipynb` or `artifacts/frontend/detection/yolo_train_v0_2_0/`
7. `artifacts/models/registry/artifact_registry.yaml` if governance traceability is needed

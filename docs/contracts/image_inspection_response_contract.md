# Image Inspection Response Contract
## Industrial Surface Defect Inspection Platform

## 1. Purpose

This document defines the backend and data response contract for the future unified Image Inspection workflow. The contract must be stable before implementing missing backend inference paths, governed evidence updates, the smart decision layer, or final frontend UI cleanup.

The completed workflow must return:

- Surface Defect Classification result.
- Defect Detection & Localization result with bounding boxes.
- Surface Anomaly Detection result.
- Smart decision/model recommendation result.
- Traceability, limitations, errors, warnings, and future AI explanation context.

## 2. Scope

This contract covers the response shape for local image inspection before Docker.

It is based on current repository evidence:

- Classification live endpoint exists at `POST /predict/classification`.
- Classification inference exists in `src/inspection_ai/inference/track_a_classifier.py`.
- Classification response schema exists in `api/app/schemas/prediction.py`.
- Classification checkpoint exists at `artifacts/models/checkpoints/model_checkpoint__1bc92561-c5bf-48f2-8246-b8f3d5718ffe.pt`.
- YOLO weights exist at `artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/best.pt`.
- YOLO validation boxes exist at `artifacts/models/predictions/detection_bbox_predictions__yolo_train_v0_2_0__validation.json`.
- YOLO box extraction logic exists in `scripts/evaluation/generate_detection_bbox_predictions.py`.
- Autoencoder model exists in `src/inspection_ai/models/autoencoder.py`.
- Autoencoder checkpoint exists at `artifacts/models/checkpoints/model_checkpoint__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.pt`.
- Anomaly evaluation utilities exist in `src/inspection_ai/evaluation/anomaly_evaluation.py`.
- Basic decision thresholds exist in `configs/decision/thresholds.yaml`.

## 3. Non-Goals

- Do not implement endpoints in this step.
- Do not implement YOLO live inference in this step.
- Do not implement anomaly live inference in this step.
- Do not implement smart decision logic in this step.
- Do not modify frontend UI in this step.
- Do not modify artifacts, registries, notebooks, or Docker files in this step.
- Do not claim production readiness.
- Do not claim deployment safety.
- Do not invent missing model outputs.
- Do not describe the future AI explanation surface as an implemented backend Agent.

## 4. Endpoint Strategy

Two endpoint strategies are possible:

- Option A: Separate model endpoints plus an aggregation service.
- Option B: One unified `POST /inspect/image` endpoint.

Recommendation for the safest first implementation: use Option A internally and expose Option B as the first frontend-facing contract.

The first implementation should build reusable service functions for classification, detection/localization, anomaly detection, and decision aggregation. The frontend-facing API should then expose one unified `POST /inspect/image` endpoint that returns the full response envelope defined in this contract.

Reason:

- Classification already has `POST /predict/classification`.
- YOLO and anomaly still need real reusable inference helpers and schemas.
- A single frontend-facing response prevents the Image Inspection page from stitching together incompatible response shapes.
- Internal services can still be tested separately before the unified endpoint is enabled.

Future endpoints may include model-specific debug or development endpoints, but the dashboard should consume the unified response once implemented.

## 5. Unified Response Envelope

The unified response must include these top-level fields:

```json
{
  "request_id": "string",
  "timestamp_utc": "string",
  "input": {},
  "classification": {},
  "detection": {},
  "anomaly": {},
  "decision": {},
  "traceability": {},
  "limitations": [],
  "errors": [],
  "warnings": [],
  "explanation_context": {}
}
```

Allowed sub-result `status` values:

- `success`
- `skipped`
- `failed`
- `unavailable`

Rules:

- Every top-level model result must be present even if unavailable.
- Missing outputs must use `status`, `errors`, and `warnings`; they must not be fabricated.
- A failed sub-result must not prevent other successful sub-results from being returned.
- The decision result must handle partial failure.
- All timestamps must use UTC.

## 6. Input Metadata Contract

The `input` object must include:

```json
{
  "filename": "string",
  "content_type": "string",
  "file_size_bytes": 0,
  "image_width": 0,
  "image_height": 0,
  "image_mode": "string",
  "preprocessing_notes": []
}
```

Field rules:

- `filename`: original uploaded filename when available.
- `content_type`: accepted upload content type.
- `file_size_bytes`: uploaded byte size.
- `image_width`: decoded image width in pixels.
- `image_height`: decoded image height in pixels.
- `image_mode`: decoded image mode, for example `RGB`.
- `preprocessing_notes`: optional list of non-secret preprocessing notes, such as resize policy or color conversion.

UNKNOWN:

- Whether the first implementation should store a temporary image hash in `input` or only in `traceability`.

## 7. Classification Result Contract

The `classification` object must include:

```json
{
  "status": "success",
  "model_name": "string",
  "model_version": "string",
  "run_id": "string",
  "threshold": 0.0,
  "predicted_label": "string",
  "predicted_label_id": 0,
  "probability_good": 0.0,
  "probability_defect": 0.0,
  "decision": "string",
  "production_ready": false,
  "deployment_safe": false,
  "limitations": [],
  "traceability": {}
}
```

Current implementation source:

- `api/app/routes/predict.py`
- `api/app/schemas/prediction.py`
- `src/inspection_ai/inference/track_a_classifier.py`

Rules:

- `production_ready` must be `false` unless a future governed policy explicitly changes it.
- `deployment_safe` must be `false` unless a future governed policy explicitly changes it.
- `probability_good` and `probability_defect` must preserve meaningful numeric precision.
- `decision` must be the model-side classification decision, not the unified final decision.

Classification traceability should include:

- `checkpoint_path`
- `run_config_path`
- `model_config_path`
- `preprocessing_config_path`
- `class_mapping_config_path`
- `quality_decision_path`
- `source_endpoint` if result came through an existing endpoint

## 8. Detection / Localization Result Contract

The `detection` object must include:

```json
{
  "status": "success",
  "model_name": "string",
  "model_version": "string",
  "run_id": "string",
  "confidence_threshold": 0.0,
  "iou_threshold": 0.0,
  "image_width": 0,
  "image_height": 0,
  "predicted_box_count": 0,
  "defect_count": 0,
  "detections": [],
  "best_detection": null,
  "review_status": "string",
  "production_ready": false,
  "deployment_safe": false,
  "limitations": [],
  "traceability": {}
}
```

Each item in `detections` must include:

```json
{
  "box_id": 0,
  "class_id": 0,
  "class_label": "string",
  "display_label": "string",
  "confidence": 0.0,
  "bbox_format": "xyxy",
  "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
  "score_rank": 1,
  "is_best_prediction": true,
  "warnings": []
}
```

Current evidence source:

- `artifacts/models/predictions/detection_bbox_predictions__yolo_train_v0_2_0__validation.json`
- `scripts/evaluation/generate_detection_bbox_predictions.py`
- `artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/best.pt`

Rules:

- `bbox_format` must be `xyxy` for the first implementation because governed validation evidence already uses this shape.
- `bbox_xyxy` values must be pixel coordinates in `[x1, y1, x2, y2]` order.
- `display_label` must be frontend-ready and may differ from raw `class_label`.
- `best_detection` must be `null` if no boxes are found.
- `review_status` should use a readable value such as `review_required`, `no_detection`, or `review_not_required`; final allowed values are UNKNOWN.
- `production_ready` must be `false`.
- `deployment_safe` must be `false`.

UNKNOWN:

- Whether live detection should use `confidence_threshold = 0.25`, `iou_threshold = 0.7`, and `imgsz = 640` from the validation bbox artifact as the first API defaults.

## 9. Anomaly Result Contract

The `anomaly` object must include:

```json
{
  "status": "success",
  "model_name": "string",
  "model_version": "string",
  "run_id": "string",
  "anomaly_score": 0.0,
  "reconstruction_loss": 0.0,
  "threshold": 0.0,
  "predicted_label": "string",
  "decision": "string",
  "quality_status": "string",
  "production_ready": false,
  "deployment_safe": false,
  "limitations": [],
  "traceability": {},
  "optional_reconstruction_artifacts": null
}
```

Current evidence source:

- `src/inspection_ai/models/autoencoder.py`
- `src/inspection_ai/evaluation/anomaly_evaluation.py`
- `artifacts/models/checkpoints/model_checkpoint__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.pt`
- `artifacts/models/metrics/anomaly_detection_evaluation__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json`

Rules:

- `anomaly_score` must use the existing score definition: mean squared reconstruction error per image.
- `reconstruction_loss` may equal the single-image reconstruction error if no separate loss definition is introduced.
- `threshold` must come from governed evidence or a documented governed config.
- `predicted_label` should be `normal` or `anomaly` for the anomaly model-side output.
- `decision` must be the model-side anomaly decision, not the unified final decision.
- `quality_status` must honestly reflect the current governed evidence, including known weak metrics where relevant.
- `production_ready` must be `false`.
- `deployment_safe` must be `false`.
- `optional_reconstruction_artifacts` must be `null` until live reconstruction/heatmap artifacts are implemented.

UNKNOWN:

- Whether the first live anomaly endpoint should generate temporary reconstruction images, heatmaps, or overlays.
- Whether current anomaly quality is acceptable for live display before PR AUC and threshold sweep evidence are added.

## 10. Smart Decision Result Contract

The `decision` object must include:

```json
{
  "status": "success",
  "final_decision": "needs_manual_review",
  "decision_level": "string",
  "model_agreement_status": "string",
  "primary_signal": "string",
  "supporting_signals": [],
  "conflict_reason": null,
  "recommended_action": "string",
  "rule_id": "string",
  "rule_summary": "string",
  "limitations": [],
  "traceability": {}
}
```

Allowed `final_decision` values:

- `good`
- `defective`
- `anomalous`
- `needs_manual_review`
- `inconclusive`

Rules:

- The decision layer must be deterministic and rule-based until a real governed Agent exists.
- The decision layer must not invent missing model outputs.
- The decision layer must handle partial failures.
- `final_decision` must be `inconclusive` or `needs_manual_review` when required inputs are missing and no safe deterministic rule applies.
- `rule_id` must identify the applied rule.
- `rule_summary` must be understandable without implying production authority.

UNKNOWN:

- Exact agreement and conflict rules between classification, detection/localization, and anomaly outputs.
- Exact values for `decision_level`, `model_agreement_status`, and `primary_signal`.

## 11. Traceability Contract

The top-level `traceability` object must include:

```json
{
  "contract_version": "image_inspection_response_v0_1",
  "api_version": "UNKNOWN",
  "source_endpoint": "/inspect/image",
  "classification": {},
  "detection": {},
  "anomaly": {},
  "decision": {},
  "frontend_evidence_sources": []
}
```

Traceability must include file paths where available:

- Model checkpoint path.
- Run config path.
- Model config path.
- Preprocessing config path.
- Class mapping path if relevant.
- Quality decision or evaluation artifact path.
- Frontend evidence bundle path if relevant.

Traceability should avoid leaking temporary filesystem paths for uploaded images unless explicitly needed for debugging.

## 12. Limitations And Safety Contract

The top-level `limitations` list must include safety boundaries that apply to the whole response.

Required safety statements:

- The response is not production-ready.
- The response is not deployment-safe.
- The response is local inspection output.
- The response is decision-support evidence, not an authoritative quality-control decision.
- Missing model outputs are not fabricated.
- Future AI explanations must be grounded in this response and governed evidence only.

Each sub-result may also include a `limitations` list.

Rules:

- No response may claim production readiness.
- No response may claim deployment safety.
- No response may claim autonomous AI Agent authority.
- No response may silently recompute governed metrics.

## 13. Future AI Explanation Context Contract

The `explanation_context` object must include:

```json
{
  "status": "available",
  "context_version": "image_inspection_explanation_context_v0_1",
  "allowed_sources": [],
  "summary_inputs": {},
  "safety_boundaries": [],
  "forbidden_claims": []
}
```

Rules:

- This context is for a future AI Explanation Assistant or Evidence Explanation Assistant.
- It must not imply that a backend Agent exists.
- It must contain only governed evidence, model outputs, decision rules, traceability, limitations, and safety boundaries.
- It must not include invented metrics.
- It must not claim production or deployment readiness.

## 14. Error And Partial Failure Contract

The top-level `errors` list must contain structured error objects:

```json
{
  "component": "detection",
  "code": "string",
  "message": "string",
  "recoverable": true
}
```

Rules:

- If classification succeeds and detection fails, return classification with `status = "success"` and detection with `status = "failed"`.
- If one model is unavailable, use `status = "unavailable"` and add an error or warning.
- If a model is intentionally not run, use `status = "skipped"` and explain why.
- Decision must still return a result for partial failures.
- Partial failure should usually lead to `final_decision = "needs_manual_review"` or `final_decision = "inconclusive"` unless a documented rule says otherwise.

## 15. Frontend Consumption Requirements

The frontend must be able to consume this response without guessing:

- Show uploaded image metadata from `input`.
- Show classification result from `classification`.
- Draw detection boxes from `detection.detections[].bbox_xyxy`.
- Show detection status, box count, best detection, and review status.
- Show anomaly score, reconstruction loss, threshold, predicted label, and quality status.
- Show final decision from `decision.final_decision`.
- Show warnings and errors without hiding partial failures.
- Show traceability in technical evidence sections.
- Use `limitations` to preserve safety wording.
- Use `explanation_context` only for future explanation features.

The frontend must not:

- Treat missing sub-results as successful.
- Invent boxes, anomaly scores, thresholds, probabilities, or final decisions.
- Present this response as production-ready or deployment-safe.

## 16. Required Backend Implementation Steps After This Contract

- Add shared schema models for the unified response envelope.
- Preserve or wrap the existing classification endpoint behavior.
- Add a reusable detection inference helper using `artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/best.pt`.
- Add detection schema models using the `xyxy` box contract.
- Add a reusable anomaly single-image inference helper using the autoencoder checkpoint.
- Add anomaly schema models for score, reconstruction loss, threshold, and model-side decision.
- Add deterministic smart decision service.
- Add unified `POST /inspect/image` endpoint.
- Add structured partial-failure handling.
- Add backend tests before frontend UI cleanup.

## 17. Required Evidence/Data Contract Steps After This Contract

- Create or update governed anomaly PR AUC evidence if retained in the dashboard.
- Create or update governed anomaly threshold sweep evidence.
- Create chart-ready anomaly score distribution data.
- Create richer reconstruction loss/distribution evidence or deliberately redesign the visual.
- Create frontend-ready sample-level anomaly prediction contract if needed.
- Create or document live YOLO box overlay contract using the `xyxy` shape.
- Create unified Image Inspection response examples for testing and frontend development.
- Create decision layer evidence/rule documentation.
- Create future AI explanation context examples grounded in governed evidence.

## 18. Open Questions

- UNKNOWN: Should `POST /inspect/image` call the existing classification endpoint internally, or share the same underlying classification service directly?
- UNKNOWN: Should model-specific endpoints remain public after the unified endpoint exists?
- UNKNOWN: Should live YOLO defaults use `confidence_threshold = 0.25`, `iou_threshold = 0.7`, and `imgsz = 640` from validation evidence?
- UNKNOWN: Should the first anomaly implementation create reconstruction image artifacts, or only return numeric output?
- UNKNOWN: What exact rules should resolve conflicts between classification, detection/localization, and anomaly detection?
- UNKNOWN: What exact allowed values should be used for `review_status`, `decision_level`, `model_agreement_status`, and `primary_signal`?
- UNKNOWN: Should uploaded image hashes be included in `input`, `traceability`, or omitted from the first version?

## Illustrative Schema Example

This example is an illustrative schema example only. It is not real model output and must not be used as governed evidence.

```json
{
  "request_id": "example-request-0001",
  "timestamp_utc": "2026-05-21T12:00:00Z",
  "input": {
    "filename": "example_surface.png",
    "content_type": "image/png",
    "file_size_bytes": 245760,
    "image_width": 2048,
    "image_height": 1000,
    "image_mode": "RGB",
    "preprocessing_notes": [
      "classification preprocessing follows configs/data/preprocessing_mvtec.yaml",
      "detection uses original image dimensions for xyxy box display"
    ]
  },
  "classification": {
    "status": "success",
    "model_name": "resnet18",
    "model_version": "0.4.0",
    "run_id": "1bc92561-c5bf-48f2-8246-b8f3d5718ffe",
    "threshold": 0.65,
    "predicted_label": "defect",
    "predicted_label_id": 1,
    "probability_good": 0.214,
    "probability_defect": 0.786,
    "decision": "defect",
    "production_ready": false,
    "deployment_safe": false,
    "limitations": [
      "Classification output is local model output and not production-ready.",
      "Classification output is not deployment-safe."
    ],
    "traceability": {
      "checkpoint_path": "artifacts/models/checkpoints/model_checkpoint__1bc92561-c5bf-48f2-8246-b8f3d5718ffe.pt",
      "run_config_path": "configs/runs/resnet18_train_v0_4_0.yaml",
      "model_config_path": "configs/models/resnet18.yaml",
      "preprocessing_config_path": "configs/data/preprocessing_mvtec.yaml",
      "class_mapping_config_path": "configs/data/class_mapping_mvtec_binary.yaml",
      "quality_decision_path": "artifacts/models/analysis/track_a_resnet18_v0_4_0_quality_decision__1bc92561-c5bf-48f2-8246-b8f3d5718ffe.json"
    }
  },
  "detection": {
    "status": "success",
    "model_name": "yolo",
    "model_version": "0.2.0",
    "run_id": "yolo_train_v0_2_0",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.7,
    "image_width": 2048,
    "image_height": 1000,
    "predicted_box_count": 2,
    "defect_count": 2,
    "detections": [
      {
        "box_id": 0,
        "class_id": 2,
        "class_label": "inclusion",
        "display_label": "Inclusion",
        "confidence": 0.802,
        "bbox_format": "xyxy",
        "bbox_xyxy": [1840.27, 0.0, 1987.1, 1000.0],
        "score_rank": 1,
        "is_best_prediction": true,
        "warnings": []
      },
      {
        "box_id": 1,
        "class_id": 4,
        "class_label": "punching_hole",
        "display_label": "Punching hole",
        "confidence": 0.612,
        "bbox_format": "xyxy",
        "bbox_xyxy": [640.0, 220.0, 790.0, 390.0],
        "score_rank": 2,
        "is_best_prediction": false,
        "warnings": []
      }
    ],
    "best_detection": {
      "box_id": 0,
      "class_id": 2,
      "class_label": "inclusion",
      "display_label": "Inclusion",
      "confidence": 0.802,
      "bbox_format": "xyxy",
      "bbox_xyxy": [1840.27, 0.0, 1987.1, 1000.0],
      "score_rank": 1,
      "is_best_prediction": true,
      "warnings": []
    },
    "review_status": "review_required",
    "production_ready": false,
    "deployment_safe": false,
    "limitations": [
      "Detection output is local model output and not production-ready.",
      "Detection output is not deployment-safe."
    ],
    "traceability": {
      "checkpoint_path": "artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/best.pt",
      "run_config_path": "configs/runs/yolo_train_v0_2_0.yaml",
      "model_config_path": "configs/models/yolo.yaml",
      "source_contract": "artifacts/models/predictions/detection_bbox_predictions__yolo_train_v0_2_0__validation.json"
    }
  },
  "anomaly": {
    "status": "success",
    "model_name": "autoencoder",
    "model_version": "0.1.0",
    "run_id": "b8ca43f5-0d53-4a42-ab37-b5fca9544a36",
    "anomaly_score": 0.219,
    "reconstruction_loss": 0.219,
    "threshold": 0.2043069839477539,
    "predicted_label": "anomaly",
    "decision": "anomaly",
    "quality_status": "weak_governed_evidence_requires_review",
    "production_ready": false,
    "deployment_safe": false,
    "limitations": [
      "Anomaly output is local model output and not production-ready.",
      "Current governed anomaly evidence has weak ROC AUC, recall, and F1.",
      "PR AUC and threshold sweep evidence are not yet complete."
    ],
    "traceability": {
      "checkpoint_path": "artifacts/models/checkpoints/model_checkpoint__b8ca43f5-0d53-4a42-ab37-b5fca9544a36.pt",
      "run_config_path": "configs/runs/autoencoder_train_v0_1_0.yaml",
      "model_config_path": "configs/models/autoencoder.yaml",
      "evaluation_path": "artifacts/models/metrics/anomaly_detection_evaluation__b8ca43f5-0d53-4a42-ab37-b5fca9544a36__test.json"
    },
    "optional_reconstruction_artifacts": null
  },
  "decision": {
    "status": "success",
    "final_decision": "needs_manual_review",
    "decision_level": "review",
    "model_agreement_status": "mixed_signals",
    "primary_signal": "detection",
    "supporting_signals": [
      "classification predicted defect",
      "detection returned two boxes",
      "anomaly predicted anomaly"
    ],
    "conflict_reason": "Detection and anomaly indicate review; final release rules are not production-governed.",
    "recommended_action": "Send image for manual review.",
    "rule_id": "example_review_when_any_model_flags_defect_v0",
    "rule_summary": "If one or more model outputs indicate defect or anomaly and the system is not deployment-governed, require manual review.",
    "limitations": [
      "Decision is rule-based and not an AI Agent decision.",
      "Decision is not production-ready or deployment-safe."
    ],
    "traceability": {
      "rules_config_path": "configs/decision/thresholds.yaml",
      "rule_source": "illustrative example only"
    }
  },
  "traceability": {
    "contract_version": "image_inspection_response_v0_1",
    "api_version": "UNKNOWN",
    "source_endpoint": "/inspect/image",
    "classification": {
      "run_id": "1bc92561-c5bf-48f2-8246-b8f3d5718ffe"
    },
    "detection": {
      "run_id": "yolo_train_v0_2_0"
    },
    "anomaly": {
      "run_id": "b8ca43f5-0d53-4a42-ab37-b5fca9544a36"
    },
    "decision": {
      "rule_id": "example_review_when_any_model_flags_defect_v0"
    },
    "frontend_evidence_sources": [
      "artifacts/frontend/track_a/",
      "artifacts/frontend/detection/yolo_train_v0_2_0/",
      "artifacts/frontend/track_b/"
    ]
  },
  "limitations": [
    "This response is not production-ready.",
    "This response is not deployment-safe.",
    "This response is local inspection output.",
    "This response is decision-support evidence, not an authoritative quality-control decision.",
    "Missing model outputs must not be fabricated.",
    "Future AI explanations must use this response and governed evidence only."
  ],
  "errors": [],
  "warnings": [
    "Illustrative schema example only; values are not governed model output."
  ],
  "explanation_context": {
    "status": "available",
    "context_version": "image_inspection_explanation_context_v0_1",
    "allowed_sources": [
      "classification",
      "detection",
      "anomaly",
      "decision",
      "traceability",
      "limitations"
    ],
    "summary_inputs": {
      "classification_label": "defect",
      "detection_box_count": 2,
      "anomaly_label": "anomaly",
      "final_decision": "needs_manual_review"
    },
    "safety_boundaries": [
      "No production-ready claim.",
      "No deployment-safe claim.",
      "No fake AI Agent claim."
    ],
    "forbidden_claims": [
      "production-ready",
      "deployment-safe",
      "autonomous AI Agent decision"
    ]
  }
}
```

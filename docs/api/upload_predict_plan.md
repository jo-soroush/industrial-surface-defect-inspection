# Upload / Predict API Plan

## 1. Goal

Define the first upload/predict API milestone before implementation begins. The API should support a safe, minimal path for local prediction requests that can later connect to the Streamlit dashboard.

## 2. Current State

- Track A Classification: PASS
- Track B / Autoencoder: PASS
- YOLO / Detection evidence layer: COMPLETE
- Streamlit dashboard exists and uses JSON data contracts
- No API upload/predict implementation exists yet
- No production-ready claim
- No deployment-safe claim
- No fake live prediction claim

## 3. First Prediction Scope

The first prediction path should be Track A classification upload/predict.

Recommended order:
1. Track A classification upload/predict
2. YOLO detection upload/predict
3. Track B anomaly upload/predict only after the model loading path is verified and the response contract is stable

This sequence keeps the first API milestone small and testable.

## 4. API Endpoints

Planned endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict/classification`
- `POST /predict/detection` later
- `POST /predict/anomaly` optional later

The first milestone should implement health and metadata endpoints before any prediction route is enabled.

## 5. Input Contract

The first upload endpoint should accept a single image file.

Input rules:
- accepted image types: `image/png`, `image/jpeg`, `image/webp`
- max file size: 10 MB
- safe decoding rule: decode images using a bounded, fail-fast path that rejects oversized, malformed, or unsupported inputs before model execution

The request should not accept arbitrary binary payloads or folder uploads.

## 6. Output Contract

The prediction response should be JSON and include:
- request identifier
- model name
- model version
- task type
- prediction label or detection summary
- confidence or score fields where applicable
- threshold or decision metadata when used
- timestamps
- safe status or limitation notes

The first version should return a compact, stable schema that is easy for the Streamlit dashboard to consume.

## 7. Model Loading Strategy

The API should load existing governed model artifacts locally and should not retrain models.

Recommended loading order:
- Track A first
- YOLO second
- Track B later or optional until the model loading path is verified end to end

Model loading should be explicit and versioned in the response metadata.

## 8. Error Handling

The API should return clear errors for:
- missing files
- unsupported image types
- oversized uploads
- malformed image data
- missing or unloaded model artifacts
- inference-time failures

Error responses should be JSON and should not leak stack traces or internal paths unless a local debug mode is explicitly enabled.

## 9. Frontend Integration Strategy

Once the API is working locally, the Streamlit dashboard should add a simple Upload / Predict page that:
- uploads a single image
- calls the classification endpoint first
- shows the JSON response in a compact, user-friendly format
- avoids implying real-time or production operation

Detection and anomaly pages should only be wired after the first classification path is stable.

## 10. Security And Safety Boundaries

- no production-ready claim
- no deployment-safe claim
- no fake live prediction claim
- no agent work
- no retraining
- no Docker before local frontend/API integration is verified
- no broad file upload surface
- no unbounded image decoding
- no implicit production deployment assumptions

## 11. Implementation Phases

- Phase 10: Plan
- Phase 11: Health/metadata API
- Phase 12: Classification upload/predict
- Phase 13: Streamlit Upload page integration
- Phase 14: YOLO detection upload/predict
- Phase 15: Docker compose

## 12. Next Step

Next step after this document is implementing basic FastAPI health and metadata endpoints.

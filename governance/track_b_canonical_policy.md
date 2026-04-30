# Track B Canonical Run Policy

## Scope

This policy applies only to Track B anomaly detection artifacts for the Industrial Surface Defect Inspection project. Track A classification artifacts, scripts, and governance rules remain unchanged.

## Current Development-Canonical Run

The current Track B development-canonical run is:

- `run_id`: `858eebad-5bf5-4b88-a20a-c93ce5819516`
- `task_type`: `anomaly_detection`
- `model_type`: `autoencoder`
- `dataset_id`: `mvtec_anomaly`
- `is_experiment`: `true`
- `canonical_status`: `development_canonical`

This run is suitable for development, frontend integration testing, artifact validation, and CI/CD contract development. It is not yet a production-canonical run because the source `TrainingResult` is marked as an experiment.

## Development vs Production Canonical

A development-canonical Track B run is the current best governed artifact set used while Track B is still being validated. It may have `is_experiment=true`, but it must still pass artifact inventory and contract validation.

A production-canonical Track B run must meet stricter governance requirements:

- The source `TrainingResult` must have `is_experiment=false`.
- The model checkpoint must be linked from the `TrainingResult`.
- The anomaly evaluation artifact must be regenerated from the canonical checkpoint.
- The Track B inventory must be regenerated.
- The full Track B artifact contract validator must pass.

## Required Artifact Set

A canonical Track B artifact set includes:

- `TrainingResult`
- Model checkpoint
- Anomaly evaluation JSON
- Track B artifact inventory JSON

The inventory records artifact paths, existence checks, SHA-256 checksums, counts, linkage, and canonical run metadata.

## Paths and Naming Conventions

TrainingResult:

```text
artifacts/models/analysis/training_results/training_result__<run_id>.json
```

Model checkpoint:

```text
artifacts/models/checkpoints/model_checkpoint__<run_id>.pt
```

Anomaly evaluation artifact:

```text
artifacts/models/metrics/anomaly_detection_evaluation__<run_id>__test.json
```

Track B inventory:

```text
artifacts/models/inventory/track_b_artifact_inventory__<run_id>.json
```

## Inventory Generation

Regenerate the Track B inventory after producing or promoting a Track B run:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluation/build_track_b_artifact_inventory.py \
  --training-result artifacts/models/analysis/training_results/training_result__<run_id>.json \
  --evaluation artifacts/models/metrics/anomaly_detection_evaluation__<run_id>__test.json \
  --output-dir artifacts/models/inventory
```

The inventory must include:

- `canonical_run_id`
- `canonical_status`
- `TrainingResult` path and SHA-256
- Model checkpoint path and SHA-256
- Anomaly evaluation JSON path and SHA-256
- Train/test counts
- Normal/anomaly test counts
- Linkage between TrainingResult, checkpoint, and evaluation artifact

## Contract Validation

Validate the complete Track B artifact contract before using a run as canonical:

```bash
PYTHONPATH=src .venv/bin/python scripts/validation/validate_track_b_artifacts.py \
  --training-result artifacts/models/analysis/training_results/training_result__<run_id>.json \
  --evaluation artifacts/models/metrics/anomaly_detection_evaluation__<run_id>__test.json \
  --inventory artifacts/models/inventory/track_b_artifact_inventory__<run_id>.json
```

Expected success output:

```text
track_b_artifact_contract=pass
```

A Track B run must not be considered production-canonical unless this validation passes.

## Current Canonical Status

The current inventory marks:

```text
canonical_run_id=858eebad-5bf5-4b88-a20a-c93ce5819516
canonical_status=development_canonical
```

To promote a future run to production-canonical, generate the run with `is_experiment=false`, regenerate the anomaly evaluation artifact, rebuild the Track B inventory, and rerun the Track B artifact validator.

## CI/CD Usage

CI/CD and frontend pipelines should resolve Track B artifacts through the inventory's `canonical_run_id` instead of hardcoding individual artifact paths. This keeps downstream consumers aligned with the governed canonical artifact set.

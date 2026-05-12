# Detection YOLO v0.2.0 Pre-Training Checklist

## Purpose

This runbook defines the pre-training checklist and execution plan for the stronger governed Detection/YOLO run:

- run_id: `yolo_train_v0_2_0`
- config: `configs/runs/yolo_train_v0_2_0.yaml`
- target output directory: `artifacts/detection/yolo/runs/yolo_train_v0_2_0`

This document is a planning and execution checklist only. It does not claim that `yolo_train_v0_2_0` has been trained, evaluated, registered, or re-audited.

## Pre-Training State

Record these values before training:

- branch: `main`
- expected repo commit before training: `c069261` or a later reviewed commit containing this runbook
- working tree: must be clean before training
- run config path: `configs/runs/yolo_train_v0_2_0.yaml`
- dataset YAML path: `data/processed/gc10det_yolo/dataset.yaml`
- split manifest path: `data/manifests/split_gc10det_detection.yaml`
- planned epochs: `50`
- planned batch size: `16`
- planned learning rate: `0.01`
- planned optimizer: `auto`
- planned device: `auto`
- planned model source: `yolov8n.pt`
- output project: `artifacts/detection/yolo/runs`
- output name: `yolo_train_v0_2_0`
- expected runtime: Colab or cloud GPU

Run these safe preflight checks first:

```bash
git status --short --untracked-files=all
git log --oneline -n 10
python scripts/validation/validate_detection_artifacts.py
python scripts/validation/validate_detection_artifacts.py --run-id yolo_train_v0_1_0
python scripts/validation/validate_detection_artifacts.py --run-id yolo_train_v0_2_0
python scripts/detection/train_yolo_detection.py --run-config configs/runs/yolo_train_v0_2_0.yaml --validate-only
```

Expected preflight results:

- `yolo_train_v0_1_0` validator passes.
- `yolo_train_v0_2_0` validator fails clearly because governed artifacts do not exist yet.
- `yolo_train_v0_2_0` validate-only reports `training_status=not_started`.
- validate-only reports `planned_output_name=yolo_train_v0_2_0`.
- validate-only reports the governed dataset YAML and split manifest.

Do not create placeholder artifacts to make `yolo_train_v0_2_0` pass validation before real training.

## Training Command

After validate-only passes in the selected runtime, run the real training command explicitly:

```bash
python scripts/detection/train_yolo_detection.py \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --run-training
```

The script requires the `--run-training` gate for real training. Without that flag, it runs the safe validation boundary only.

If an execution runtime needs a device override, use the script-supported override and record it:

```bash
python scripts/detection/train_yolo_detection.py \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --device 0 \
  --run-training
```

Do not change the governed run config inside Colab/cloud without bringing the changed config back for review.

## Expected Training Outputs

Required output directory:

- `artifacts/detection/yolo/runs/yolo_train_v0_2_0/`

Required files:

- `artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/best.pt`
- `artifacts/detection/yolo/runs/yolo_train_v0_2_0/weights/last.pt`
- `artifacts/detection/yolo/runs/yolo_train_v0_2_0/results.csv`
- `artifacts/detection/yolo/runs/yolo_train_v0_2_0/args.yaml`

Expected optional YOLO visual outputs if created by Ultralytics:

- `results.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `BoxPR_curve.png`
- `BoxF1_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `labels.jpg`
- `val_batch*_pred.jpg`
- `val_batch*_labels.jpg`
- `train_batch*.jpg`

Preserve the full run directory, not only checkpoints.

## Colab / Cloud Return Checklist

If training runs outside the local machine, copy back:

- full directory: `artifacts/detection/yolo/runs/yolo_train_v0_2_0/`
- the exact `configs/runs/yolo_train_v0_2_0.yaml` used in the runtime
- terminal command output or training log if available
- environment information if available
- package versions if available, especially `ultralytics`, `torch`, Python, CUDA, and GPU name

The remote runtime is not the source of truth. Returned files must be inspected and governed in the repository before use.

## Post-Training Governance Plan

Run these commands only after the real `yolo_train_v0_2_0` run directory exists and required files are present.

1. Build artifact inventory:

```bash
python scripts/evaluation/build_detection_artifact_inventory.py \
  --run-id yolo_train_v0_2_0 \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --run-dir artifacts/detection/yolo/runs/yolo_train_v0_2_0
```

2. Build training result summary:

```bash
python scripts/evaluation/build_detection_training_result_summary.py \
  --run-id yolo_train_v0_2_0 \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --run-dir artifacts/detection/yolo/runs/yolo_train_v0_2_0
```

3. Build metadata summary:

```bash
python scripts/evaluation/build_detection_metadata_summary.py \
  --run-id yolo_train_v0_2_0 \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --run-dir artifacts/detection/yolo/runs/yolo_train_v0_2_0
```

4. Build post-hoc run log:

```bash
python scripts/evaluation/build_detection_posthoc_run_log.py \
  --run-id yolo_train_v0_2_0 \
  --run-dir artifacts/detection/yolo/runs/yolo_train_v0_2_0
```

5. Build validation evaluation summary:

```bash
python scripts/evaluation/build_detection_evaluation_summary.py \
  --run-id yolo_train_v0_2_0 \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --run-dir artifacts/detection/yolo/runs/yolo_train_v0_2_0
```

6. Register run and artifacts:

NEEDS_FOLLOWUP: there is no dedicated Detection registration command identified in this preflight. Register `yolo_train_v0_2_0` in `artifacts/models/registry/run_registry.yaml` and register the governed Detection artifacts in `artifacts/models/registry/artifact_registry.yaml` using the existing registry style and real SHA-256/size values.

7. Validate governed artifacts:

```bash
python scripts/validation/validate_detection_artifacts.py \
  --run-id yolo_train_v0_2_0 \
  --model-version 0.2.0
```

8. Build re-audit report:

```bash
python scripts/validation/build_detection_reaudit_report.py \
  --run-id yolo_train_v0_2_0
```

9. Validate again:

```bash
python scripts/validation/validate_detection_artifacts.py \
  --run-id yolo_train_v0_2_0 \
  --model-version 0.2.0
```

10. Update notebook after validation and re-audit pass:

- `notebooks/detection_yolo_gc10det_evidence.ipynb`

Do not update the notebook before the governed v0.2.0 evidence exists.

## Readiness Interpretation

The readiness policy is implemented in:

- `src/inspection_ai/evaluation/detection_readiness_policy.py`

Baseline metrics from `yolo_train_v0_1_0`:

- precision: `0.00477`
- recall: `0.54003`
- mAP50: `0.04518`
- mAP50_95: `0.01651`

Possible governed statuses:

- `not_ready`
- `improved_baseline`
- `review_required`
- `model_ready_candidate`
- `production_ready`

Minimum success expectations for `yolo_train_v0_2_0`:

- mAP50 improves over `0.04518`.
- mAP50_95 improves over `0.01651`.
- precision improves materially over `0.00477`.
- recall does not collapse relative to `0.54003`.
- visual predictions are reviewed before any model-ready claim.
- production-ready should not be expected from validation metrics alone.

Production-ready requires stronger metric thresholds plus test evaluation, class-level evidence, visual review, and explicit audit approval.

## Commit Plan

Use small commits after each milestone:

1. Preflight runbook:
   - commit message: `[docs] Add Detection YOLO v0.2.0 pretraining checklist`
2. Returned training outputs:
   - commit message: `[artifacts] Add Detection YOLO v0.2.0 run outputs`
   - if large artifacts are handled externally, document the external artifact handling instead
3. Governance summaries:
   - commit message: `[governance] Add Detection YOLO v0.2.0 governed summaries`
4. Registry entries:
   - commit message: `[registry] Register Detection YOLO v0.2.0 artifacts`
5. Validation and re-audit:
   - commit message: `[validation] Add Detection YOLO v0.2.0 re-audit evidence`
6. Notebook update:
   - commit message: `[notebook] Update Detection YOLO evidence notebook for v0.2.0`

Do not combine training outputs, registry edits, and notebook changes into one opaque commit.

## Risks And Known Limitations

- The current local environment reports `ultralytics_available=false`; Colab/cloud GPU is recommended.
- The training command may download `yolov8n.pt` if not already present in the runtime.
- `yolo_train_v0_2_0` may still not become production-ready.
- Validation split metrics alone are not sufficient for production readiness.
- Split imbalance and class-level behavior need review after training.
- Some label files are empty by design or dataset reality; validate-only currently reports them explicitly.
- Post-hoc logs are not original runtime logs unless the real runtime command output is preserved and governed.
- Registry registration for v0.2.0 needs a careful manual/utility-backed governance step before validation can pass.
- Do not make fake readiness claims or fill missing artifacts with placeholders.

## One Next Action

Proceed to Colab/cloud training only after the preflight checks above pass in the target runtime and the working tree is clean.

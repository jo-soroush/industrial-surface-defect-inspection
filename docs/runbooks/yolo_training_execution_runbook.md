# YOLO Training Execution Runbook

## Purpose

This runbook defines how to execute YOLO detection training in a governed and portable way across different runtimes.

- The repository is the canonical source of code, configs, manifests, and governance.
- Local, Colab, server/cloud, and Docker-like environments are execution runtimes only.
- The same repo scripts, configs, and manifests must be used in every runtime.
- Generated outputs must be brought back, inspected, and governed before they are treated as project artifacts.

## Current Readiness State

- The GC10-DET detection split manifest exists.
- The YOLO backend dependency is declared.
- The YOLO dataset exporter exists.
- The YOLO training boundary exists.
- The gated `--run-training` path exists.
- The model source is declared as `yolov8n.pt`.
- Local runtime is currently blocked by missing `ultralytics` due to weak network.
- Colab is the practical first runtime, but not the only supported runtime.

## Shared Governed Execution Flow

The runtime-independent flow is:

1. Prepare a repository checkout.
2. Provide raw GC10-DET data under `data/raw/gc10det/`.
3. Install and verify runtime dependencies.
4. Export the YOLO dataset from the governed split manifest.
5. Run the validate-only boundary.
6. Run `--run-training` only after validate-only passes.
7. Preserve the generated run directory.
8. Return outputs to the governance workflow.

## Required Inputs

The following files and folders must be available in the runtime:

- repository checkout
- `data/raw/gc10det/img/`
- `data/raw/gc10det/ann/`
- `data/manifests/split_gc10det_detection.yaml`
- `scripts/data/export_gc10det_yolo_dataset.py`
- `scripts/detection/train_yolo_detection.py`
- `configs/models/yolo.yaml`
- `configs/runs/yolo_train_v0_1_0.yaml`
- `requirements.txt`

## Local Execution Path

Local execution is valid when the dependencies are installed.

The current local blocker is `ultralytics` installation failure caused by a weak network.
Do not run training locally until readiness passes.

```bash
python scripts/data/export_gc10det_yolo_dataset.py \
  --manifest-path data/manifests/split_gc10det_detection.yaml \
  --output-root data/processed/gc10det_yolo
```

```bash
python scripts/detection/train_yolo_detection.py
```

```bash
python scripts/detection/train_yolo_detection.py --run-training
```

The final command must only be run after validate-only passes.

## Colab Execution Path

Colab is the recommended first runtime because local network conditions blocked dependency installation.
Colab must not become the source of truth.

Use Drive-backed storage so the repo checkout and outputs survive session loss.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
git clone <REPO_URL> /content/drive/MyDrive/industrial-surface-defect-inspection
cd /content/drive/MyDrive/industrial-surface-defect-inspection
```

```bash
mkdir -p data/raw/gc10det
rsync -a /content/drive/MyDrive/<GC10DET_RAW_PATH_IN_DRIVE>/ data/raw/gc10det/
```

```bash
pip install ultralytics
python - <<'PY'
import ultralytics, torch
print("ultralytics=", ultralytics.__version__)
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY
```

```bash
python scripts/data/export_gc10det_yolo_dataset.py \
  --manifest-path data/manifests/split_gc10det_detection.yaml \
  --output-root data/processed/gc10det_yolo
```

```bash
python scripts/detection/train_yolo_detection.py
```

```bash
python scripts/detection/train_yolo_detection.py --run-training
```

The final command must only be run after validate-only passes.

## Server / Cloud Execution Path

A server or cloud VM should follow the same governed flow:

- check out the repository
- place the raw data under the expected repository path
- install dependencies
- run the exporter
- run validate-only
- run gated training only after validate-only passes

Do not add hardcoded environment-specific paths to the repository.

## Dataset Export Step

```bash
python scripts/data/export_gc10det_yolo_dataset.py \
  --manifest-path data/manifests/split_gc10det_detection.yaml \
  --output-root data/processed/gc10det_yolo
```

Expected outputs:

- `data/processed/gc10det_yolo/dataset.yaml`
- `data/processed/gc10det_yolo/export_manifest.yaml`
- train image/label count: `1610 / 1610`
- validation image/label count: `345 / 345`
- test image/label count: `345 / 345`
- empty label files: `8`

## Validate-Only Step

```bash
python scripts/detection/train_yolo_detection.py
```

Expected key output:

- `execution_mode=validate_only`
- `training_status=not_started`
- `run_training_enabled=false`
- `planned_model_source=yolov8n.pt`
- `split_counts=train:1610,validation:345,test:345`
- `validated_split_counts=train:1610,validation:345,test:345`

Training must not be run unless validate-only passes.

## Training Execution Step

```bash
python scripts/detection/train_yolo_detection.py --run-training
```

This is the first command that actually starts training.
It must only be run intentionally after setup, export, and validate-only pass.
It may download `yolov8n.pt` if missing.
It may create files under `artifacts/detection/yolo/runs/yolo_train_v0_1_0`.

## Expected Training Outputs

Expected output directory:

- `artifacts/detection/yolo/runs/yolo_train_v0_1_0/`

Expected possible files:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `args.yaml`
- `results.png`
- `confusion_matrix.png` if created
- train / validation batch images if created
- other Ultralytics generated files

## What to Bring Back From Remote Runtime

Bring back the full run directory, not only `best.pt`.

Required:

- `artifacts/detection/yolo/runs/yolo_train_v0_1_0/`

## Post-Run Governance Workflow

After outputs are returned locally:

1. Inspect the returned run directory.
2. Confirm expected files are present.
3. Create a detection training result summary.
4. Create a detection metadata summary.
5. Create a detection post-hoc log.
6. Create a detection inventory.
7. Register the detection run in `run_registry.yaml`.
8. Register the detection artifacts in `artifact_registry.yaml`.
9. Implement or run the detection evaluation writer.
10. Re-audit Detection.

## Do Not Commit Generated Raw Outputs Directly

- `data/processed/` is ignored.
- `artifacts/` is ignored.
- Generated training files should not be committed blindly.
- Only governed summaries and registry entries should be committed after audit.

## Risks and Mitigations

| risk | severity | mitigation |
|---|---|---|
| local dependency installation failure | high | use a portable runtime such as Colab or a server/cloud VM if local install is blocked |
| Colab session loss | high | use Drive-backed storage and preserve the full run directory |
| raw dataset upload / copy error | medium | verify file counts and split manifest linkage before training |
| Ultralytics output naming differences | medium | preserve the full run directory and inspect generated filenames before promotion |
| pretrained model download | medium | document that `yolov8n.pt` may be fetched by Ultralytics if missing |
| GPU unavailable | medium | verify device availability before training |
| generated artifact size | medium | bring back the full run directory, not only the checkpoint files |
| commands not recorded | high | keep this runbook and the exact executed commands together |
| mismatch between runtime checkout and local repo | high | ensure the runtime uses the intended repo commit before training |
| generated outputs treated as source of truth | high | treat runtime outputs as evidence to be inspected and governed, not canonical truth |

## Pre-Run Checklist

- [ ] runtime is selected and documented
- [ ] repository checkout is at the expected commit
- [ ] raw GC10-DET folders exist
- [ ] dependencies import successfully
- [ ] exporter completed successfully
- [ ] `dataset.yaml` exists
- [ ] validate-only PASS
- [ ] device decision documented
- [ ] output path is understood
- [ ] user intentionally starts `--run-training`

## Post-Run Checklist

- [ ] full run directory preserved
- [ ] `best.pt` exists if training completed
- [ ] `last.pt` exists if training completed
- [ ] `results.csv` exists if generated
- [ ] `args.yaml` exists if generated
- [ ] outputs copied back to local / Drive
- [ ] no generated outputs committed blindly
- [ ] governance artifacts prepared after inspection

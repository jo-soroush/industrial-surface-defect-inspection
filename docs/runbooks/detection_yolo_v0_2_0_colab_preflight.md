# Detection YOLO v0.2.0 Colab / Cloud Preflight

## Purpose

This runbook is the target-runtime preflight guide for the governed Detection/YOLO run:

- run_id: `yolo_train_v0_2_0`
- config: `configs/runs/yolo_train_v0_2_0.yaml`
- planned output directory: `artifacts/detection/yolo/runs/yolo_train_v0_2_0`

This document does not claim that `yolo_train_v0_2_0` has been trained. Colab/cloud preflight status is `NEEDS_RUNTIME_CONFIRMATION` until these commands are run in the actual target runtime.

## Runtime Preparation

Before training, confirm:

- GPU runtime is enabled in Colab/cloud.
- The repository is cloned or mounted.
- The correct branch and commit are checked out.
- The working tree is clean.
- GC10-DET raw/exported dataset files are present.
- Runtime dependencies are installed in the target runtime.
- `ultralytics` imports in the target runtime.
- `torch` sees the GPU if GPU training is intended.
- No `artifacts/detection/yolo/runs/yolo_train_v0_2_0/` directory exists unless intentionally resuming/replacing a reviewed run.

Do not edit governed config files directly in Colab/cloud. If a config change is needed, stop and make the change in the repository review flow.

## Exact Colab / Cloud Preflight Commands

Run from the repository root in the target runtime.

```bash
pwd
git status --short --untracked-files=all
git log --oneline -n 10
python --version
```

GPU and dependency checks:

```bash
nvidia-smi || true
python - <<'PY'
import importlib.util
import platform

print("python=", platform.python_version())
print("torch_available=", importlib.util.find_spec("torch") is not None)
print("ultralytics_available=", importlib.util.find_spec("ultralytics") is not None)

if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch_version=", torch.__version__)
    print("cuda_available=", torch.cuda.is_available())
    print("cuda_device_count=", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device_name=", torch.cuda.get_device_name(0))

if importlib.util.find_spec("ultralytics") is not None:
    import ultralytics
    print("ultralytics_version=", ultralytics.__version__)
PY
```

If `ultralytics_available=false`, install dependencies in Colab/cloud only:

```bash
python -m pip install -r requirements.txt
```

Then rerun the GPU and dependency checks.

Dataset file checks:

```bash
test -f data/manifests/split_gc10det_detection.yaml
test -f data/processed/gc10det_yolo/dataset.yaml
python - <<'PY'
from pathlib import Path
import yaml

dataset_yaml_path = Path("data/processed/gc10det_yolo/dataset.yaml")
manifest_path = Path("data/manifests/split_gc10det_detection.yaml")
dataset = yaml.safe_load(dataset_yaml_path.read_text())
manifest = yaml.safe_load(manifest_path.read_text())

print("dataset_yaml=", dataset_yaml_path)
print("manifest=", manifest_path)
print("dataset_nc=", dataset.get("nc"))
print("dataset_names=", ",".join(dataset.get("names", [])))
print("manifest_split_counts=", manifest.get("split_counts"))

root = dataset_yaml_path.parent
for split_key, split_name in (("train", "train"), ("val", "validation"), ("test", "test")):
    image_dir = root / dataset[split_key]
    label_dir = root / "labels" / split_name
    image_count = len([p for p in image_dir.iterdir() if p.is_file()]) if image_dir.is_dir() else "missing"
    label_count = len([p for p in label_dir.iterdir() if p.is_file()]) if label_dir.is_dir() else "missing"
    print(f"{split_name}_image_dir={image_dir}")
    print(f"{split_name}_label_dir={label_dir}")
    print(f"{split_name}_image_count={image_count}")
    print(f"{split_name}_label_count={label_count}")
PY
```

Existing governed baseline validation, if the v0.1.0 artifacts are present in the runtime:

```bash
python scripts/validation/validate_detection_artifacts.py
python scripts/validation/validate_detection_artifacts.py --run-id yolo_train_v0_1_0
```

Expected v0.2.0 artifact validation before training:

```bash
python scripts/validation/validate_detection_artifacts.py --run-id yolo_train_v0_2_0
```

This should fail before training because governed v0.2.0 artifacts do not exist yet. Do not create placeholder artifacts to make it pass.

Validate-only boundary for the planned v0.2.0 run:

```bash
python scripts/detection/train_yolo_detection.py \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --validate-only
```

Expected validate-only signals:

- `execution_mode=validate_only`
- `training_status=not_started`
- `run_training_enabled=false`
- `run_config_path=configs/runs/yolo_train_v0_2_0.yaml`
- `dataset_yaml_path=data/processed/gc10det_yolo/dataset.yaml`
- `planned_epochs=50`
- `planned_batch_size=16`
- `planned_output_project=artifacts/detection/yolo/runs`
- `planned_output_name=yolo_train_v0_2_0`
- `planned_model_source=yolov8n.pt`
- `validated_split_counts=train:1610,validation:345,test:345`
- `validated_label_counts=train:1610,validation:345,test:345`

Output directory guard:

```bash
if [ -e artifacts/detection/yolo/runs/yolo_train_v0_2_0 ]; then
  echo "NEEDS_REVIEW: v0.2.0 output directory already exists"
  find artifacts/detection/yolo/runs/yolo_train_v0_2_0 -maxdepth 2 -type f | sort
else
  echo "v0.2.0 output directory is clear"
fi
```

## Preflight Pass Criteria

Preflight passes only when:

- correct branch and intended commit are checked out
- working tree is clean or any changes are intentionally reviewed
- `torch` imports
- `ultralytics` imports
- GPU is visible if GPU training is intended
- `data/manifests/split_gc10det_detection.yaml` exists
- `data/processed/gc10det_yolo/dataset.yaml` exists
- train/validation/test image and label paths exist
- validate-only passes
- validate-only reports `training_status=not_started`
- validate-only reports `planned_output_name=yolo_train_v0_2_0`
- v0.2.0 artifact validation fails only because artifacts are not created yet
- no v0.2.0 output directory exists unless an intentional resume/replacement is documented

If any item fails, do not train.

## Failure Handling

- Missing `ultralytics`: install dependencies in Colab/cloud only, then rerun dependency checks.
- Missing dataset YAML: export or copy the governed YOLO dataset before training.
- Missing raw dataset: mount/copy GC10-DET into the expected repository path before export.
- Wrong commit or branch: checkout the intended commit and rerun all preflight checks.
- Dirty worktree: stop and resolve before training.
- Validate-only fails: do not run training; return the full output for review.
- Existing v0.2.0 output directory: stop and decide whether this is a resume, replacement, or stale output.
- GPU unavailable: switch runtime or explicitly accept CPU training only after review.

## Training Command

Do not run this command until every preflight pass criterion above is satisfied:

```bash
python scripts/detection/train_yolo_detection.py \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --run-training
```

If a device override is required and reviewed:

```bash
python scripts/detection/train_yolo_detection.py \
  --run-config configs/runs/yolo_train_v0_2_0.yaml \
  --device 0 \
  --run-training
```

## Return Package After Training

After training, return:

- full directory: `artifacts/detection/yolo/runs/yolo_train_v0_2_0/`
- full terminal log or notebook cell output
- exact commit hash used for training
- `python --version`
- `pip freeze` or equivalent package version output
- GPU/runtime information from `nvidia-smi`
- `torch` and `ultralytics` versions

Returned outputs are not governed evidence until the post-training governance builders, registry updates, validator, and re-audit complete.

## Safety Rules

- Do not change config directly in Colab/cloud.
- Do not rename the output directory.
- Do not overwrite `yolo_train_v0_1_0`.
- Do not create fake artifacts.
- Do not claim model readiness before governance validation and re-audit.
- Do not claim Colab/cloud preflight passed unless the target-runtime output proves it.

"""Governed training entrypoint skeleton for Phase 3.

This script defines the CLI boundary for future model-training execution. In
Phase 3, its role is to provide a clean, governed entrypoint that will
eventually load training configuration and dispatch the appropriate training
flow without embedding training logic directly in the script.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import yaml

from inspection_ai.governance.run_registry_writer import append_run_registry_entry
from inspection_ai.models.factory import create_model
from inspection_ai.training.checkpointing import (
    build_structured_checkpoint_payload,
    extract_model_state_dict,
    load_checkpoint_payload,
    resolve_model_checkpoint_path,
    save_checkpoint,
)
from inspection_ai.training.data_loading import build_data_loaders
from inspection_ai.training.result_persistence import persist_training_result
from inspection_ai.training.result_validation import validate_training_result
from inspection_ai.training.train_loop import (
    build_classification_validation_evaluation_payload,
    evaluate_classification_validation,
    replay_classification_validation,
    write_classification_validation_evaluation_artifact,
    run_training_loop,
)

ALLOWED_TASK_TYPES = {
    "classification",
    "anomaly_detection",
    "object_detection",
}


class _TeeStream:
    """Capture runtime output while preserving console output."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, value: str) -> int:
        self.buffer.write(value)
        return self.stream.write(value)

    def flush(self) -> None:
        self.buffer.flush()
        self.stream.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a governed training config file."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Training config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Training config YAML is invalid: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to read training config: {path}") from exc

    if config is None:
        raise ValueError(f"Training config is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Training config must parse to a dictionary at this stage: {path}"
        )

    return config


def handle_classification(config: dict[str, Any]) -> object:
    """Placeholder handler for classification training dispatch."""
    return _run_training_with_checkpoint(config)


def handle_anomaly_detection(config: dict[str, Any]) -> object:
    """Placeholder handler for anomaly-detection training dispatch."""
    return _run_training_with_checkpoint(config)


def handle_object_detection(config: dict[str, Any]) -> object:
    """Placeholder handler for object-detection training dispatch."""
    return _run_training_with_checkpoint(config)


def _run_training_with_checkpoint(config: dict[str, Any]) -> object:
    """Run training and attach a governed model checkpoint artifact."""
    model = create_model(config)
    data_loaders = build_data_loaders(config)
    result = run_training_loop(config=config, model=model, data_loader=data_loaders)
    _attach_model_checkpoint(result=result, model=model)
    checkpoint_state_dict, validation_evaluation = _materialize_official_validation_evaluation(
        config=config,
        result=result,
    )
    _finalize_model_checkpoint(
        result=result,
        model_state_dict=checkpoint_state_dict,
        validation_evaluation=validation_evaluation,
    )
    return result


def _attach_model_checkpoint(result: Any, model: Any) -> None:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise ValueError("Trained model must provide a callable state_dict method.")

    checkpoint_path = resolve_model_checkpoint_path(result.identity["run_id"])
    save_checkpoint(state_dict(), checkpoint_path)
    result.add_artifact(
        "model_artifact",
        {
            "path": str(checkpoint_path),
            "type": "pytorch_state_dict",
        },
    )


def _materialize_official_validation_evaluation(
    *, config: dict[str, Any], result: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay validation from the saved checkpoint and persist the official artifact."""
    model_artifact = _require_dict(result.artifacts.get("model_artifact"), "model_artifact")
    checkpoint_path = Path(
        _require_string(model_artifact.get("path"), "model_artifact.path")
    )

    replay_config = _build_checkpoint_replay_config(config)
    replay_model = create_model(replay_config)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_model_state_dict(checkpoint_payload)
    replay_model.load_state_dict(state_dict, strict=True)

    data_loaders = build_data_loaders(replay_config)
    official_validation = evaluate_classification_validation(
        config=replay_config,
        model=replay_model,
        data_loader=data_loaders,
    )
    dataset_id = data_loaders.get("dataset_id")
    if dataset_id is None:
        raise ValueError("data_loader is missing required dataset_id.")

    validation_payload = build_classification_validation_evaluation_payload(
        result=result,
        dataset_id=dataset_id,
        evaluation=official_validation,
    )
    validation_path = write_classification_validation_evaluation_artifact(
        result=result,
        dataset_id=dataset_id,
        evaluation=official_validation,
    )
    result.add_metric("val_loss", official_validation["loss"])
    result.add_metric("val_accuracy", official_validation["accuracy"])
    result.add_metric("val_f1", official_validation["f1"])
    result.add_metadata("validation_evaluation_checked", True)
    result.add_metadata(
        "validation_evaluation_batch_count", official_validation["batch_count"]
    )
    result.add_metadata("validation_evaluation_path", str(validation_path))
    return state_dict, validation_payload


def _finalize_model_checkpoint(
    *,
    result: Any,
    model_state_dict: Any,
    validation_evaluation: dict[str, Any],
) -> None:
    """Rewrite the checkpoint as a structured governed payload."""
    checkpoint_path = resolve_model_checkpoint_path(result.identity["run_id"])
    checkpoint_payload = build_structured_checkpoint_payload(
        model_state_dict=model_state_dict,
        result=result,
        checkpoint_kind="final",
        validation_evaluation=validation_evaluation,
    )
    save_checkpoint(checkpoint_payload, checkpoint_path, overwrite=True)


def extract_task_type(config: dict[str, Any]) -> str:
    """Extract and validate task_type from the resolved run config identity block."""
    identity = config.get("identity")

    if identity is None:
        raise ValueError("Training config is missing required section: identity")

    if not isinstance(identity, dict):
        raise ValueError("Training config section identity must be a dictionary")

    task_type = identity.get("task_type")

    if task_type is None:
        raise ValueError(
            "Training config is missing required field: identity.task_type"
        )

    if not isinstance(task_type, str):
        raise ValueError("Training config field identity.task_type must be a string")

    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(
            "Training config has unsupported identity.task_type: "
            f"{task_type}. Allowed values: {sorted(ALLOWED_TASK_TYPES)}"
        )

    return task_type


def dispatch_training(config: dict[str, Any]) -> object:
    """Validate task type and route to the governed training placeholder."""
    task_type = extract_task_type(config)

    if task_type == "classification":
        return handle_classification(config)

    if task_type == "anomaly_detection":
        return handle_anomaly_detection(config)

    if task_type == "object_detection":
        return handle_object_detection(config)

    raise RuntimeError(f"Unhandled task_type dispatch path: {task_type}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the governed training entrypoint."""
    parser = argparse.ArgumentParser(description="Governed model training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to the training config.")
    return parser


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _save_logs_enabled(config: dict[str, Any]) -> bool:
    output_control = config.get("output_control")
    if not isinstance(output_control, dict):
        return False
    return output_control.get("save_logs") is True


def _run_config_id(config: dict[str, Any]) -> str:
    identity = config.get("identity")
    if not isinstance(identity, dict):
        return "unknown_config"
    run_config_id = identity.get("run_config_id")
    if not isinstance(run_config_id, str) or not run_config_id:
        return "unknown_config"
    return run_config_id


def _runtime_log_path(config: dict[str, Any], result: Any | None) -> Path:
    run_id = None
    if result is not None:
        identity = getattr(result, "identity", None)
        if isinstance(identity, dict):
            candidate_run_id = identity.get("run_id")
            if isinstance(candidate_run_id, str) and candidate_run_id:
                run_id = candidate_run_id
    if run_id is None:
        run_id = f"{_run_config_id(config)}__failed__{_utc_now_iso().replace(':', '')}"

    return Path("artifacts/models/logs") / f"training_runtime_log__{run_id}.json"


def _write_runtime_log(
    *,
    config: dict[str, Any],
    config_path: str,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    status: str,
    stdout_text: str,
    stderr_text: str,
    result: Any | None = None,
    result_path: Path | None = None,
    error: BaseException | None = None,
) -> Path | None:
    if not _save_logs_enabled(config):
        return None

    output_path = _runtime_log_path(config, result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact_type": "training_runtime_log",
        "log_generation_mode": "original_runtime_capture",
        "original_runtime_log_available": True,
        "config_path": config_path,
        "run_config_id": _run_config_id(config),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "run_status": status,
        "stdout": stdout_text,
        "stderr": stderr_text,
    }

    if result is not None and isinstance(getattr(result, "identity", None), dict):
        payload["run_id"] = result.identity.get("run_id")
        payload["task_type"] = result.identity.get("task_type")
        payload["model_type"] = result.identity.get("model_type")
        payload["is_experiment"] = result.identity.get("is_experiment")
    if result_path is not None:
        payload["training_result_path"] = str(result_path)
    if error is not None:
        payload["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def main() -> int:
    """Parse CLI arguments and invoke the training dispatch placeholder."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    started_at = _utc_now_iso()
    start_time = time.perf_counter()
    stdout_capture = _TeeStream(sys.stdout)
    stderr_capture = _TeeStream(sys.stderr)
    result = None
    result_path = None
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = dispatch_training(config)
            validate_training_result(result)
            _validate_checkpoint_replay(config, result)
            result_path = persist_training_result(
                result=result,
                output_dir=Path("artifacts/models/analysis/training_results"),
            )
            print(f"Training result created: {type(result).__name__}")
            print(f"Training result saved: {result_path}")
            if not result.identity.get("is_experiment", True):
                append_run_registry_entry(result=result, result_path=result_path)
                print("Training run registered.")
            else:
                print("Experiment run — not registered.")
        ended_at = _utc_now_iso()
        duration_seconds = time.perf_counter() - start_time
        log_path = _write_runtime_log(
            config=config,
            config_path=args.config,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            status="success",
            stdout_text=stdout_capture.getvalue(),
            stderr_text=stderr_capture.getvalue(),
            result=result,
            result_path=result_path,
        )
        if log_path is not None:
            print(f"Runtime log saved: {log_path}")
        return 0
    except Exception as exc:
        ended_at = _utc_now_iso()
        duration_seconds = time.perf_counter() - start_time
        log_path = _write_runtime_log(
            config=config,
            config_path=args.config,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            status="failed",
            stdout_text=stdout_capture.getvalue(),
            stderr_text=stderr_capture.getvalue(),
            result=result,
            result_path=result_path,
            error=exc,
        )
        if log_path is not None:
            print(f"Runtime log saved: {log_path}", file=sys.stderr)
        raise


def _validate_checkpoint_replay(config: dict[str, Any], result: Any) -> None:
    """Ensure the saved checkpoint exactly reproduces the official validation eval."""
    model_artifact = _require_dict(result.artifacts.get("model_artifact"), "model_artifact")
    checkpoint_path = Path(
        _require_string(model_artifact.get("path"), "model_artifact.path")
    )
    validation_evaluation_path = Path(
        _require_string(
            result.metadata.get("validation_evaluation_path"),
            "result.metadata.validation_evaluation_path",
        )
    )
    validation_evaluation = _load_json_file(
        validation_evaluation_path, "validation evaluation"
    )

    replay_config = _build_checkpoint_replay_config(config)
    replay_model = create_model(replay_config)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_model_state_dict(checkpoint_payload)
    replay_model.load_state_dict(state_dict, strict=True)

    data_loaders = build_data_loaders(replay_config)
    validation_loader = data_loaders.get("validation_loader")
    if validation_loader is None:
        raise ValueError("validation_loader must exist for checkpoint verification.")

    class_count = getattr(replay_model, "class_count", None)
    if not isinstance(class_count, int) or class_count < 2:
        raise ValueError("Replay model must expose a valid class_count.")

    replay_metrics = replay_classification_validation(
        model=replay_model,
        data_loader=validation_loader,
        expected_class_count=class_count,
    )

    expected_confusion_matrix = validation_evaluation.get("confusion_matrix")
    if replay_metrics["confusion_matrix"] != expected_confusion_matrix:
        print(f"checkpoint_replay_confusion_matrix={replay_metrics['confusion_matrix']}")
        print(f"official_validation_confusion_matrix={expected_confusion_matrix}")
        raise ValueError(
            "Checkpoint replay confusion matrix does not match the official "
            "validation evaluation."
        )


def _build_checkpoint_replay_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an inference-safe copy for replay validation."""
    model_identity = config.get("model_identity")
    if not isinstance(model_identity, dict):
        return config

    if model_identity.get("model_type") != "resnet18":
        return config

    pretrained_policy = model_identity.get("pretrained_policy")
    if not isinstance(pretrained_policy, dict) or pretrained_policy.get("pretrained") is not True:
        return config

    replay_config = deepcopy(config)
    replay_model_identity = replay_config.get("model_identity")
    if not isinstance(replay_model_identity, dict):
        return config

    replay_pretrained_policy = replay_model_identity.get("pretrained_policy")
    if not isinstance(replay_pretrained_policy, dict):
        return config
    replay_pretrained_policy["pretrained"] = False
    replay_model_identity["pretrained_policy"] = replay_pretrained_policy

    top_level_pretrained_policy = replay_config.get("pretrained_policy")
    if isinstance(top_level_pretrained_policy, dict):
        top_level_pretrained_policy["pretrained"] = False
        replay_config["pretrained_policy"] = top_level_pretrained_policy

    return replay_config


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dictionary.")
    return value


def _load_json_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} JSON not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

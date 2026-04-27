"""Smoke test for the Phase 3 training pipeline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RESULTS_DIR = REPO_ROOT / "artifacts/models/analysis/training_results"


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            "python",
            "scripts/training/train_model.py",
            "--config",
            "configs/runs/mlp_train_v0_1_0.yaml",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    _delete_latest_training_result()

    if completed.returncode == 0 and "Training result created" in completed.stdout:
        print("[PASS] Phase 3 smoke test")
        return 0

    print("[FAIL] Phase 3 smoke test")
    return 1


def _delete_latest_training_result() -> None:
    if not TRAINING_RESULTS_DIR.exists():
        return

    result_files = [
        path for path in TRAINING_RESULTS_DIR.iterdir() if path.is_file()
    ]
    if not result_files:
        return

    latest_result = max(result_files, key=lambda path: path.stat().st_mtime)
    latest_result.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

"""Write a Track A supervised classification comparison artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from inspection_ai.evaluation.comparison import build_track_a_comparison


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Write a Track A supervised classification comparison artifact."
    )
    parser.add_argument(
        "--training-result",
        action="append",
        default=[],
        help="Path to a TrainingResult JSON artifact. Provide at least two.",
    )
    parser.add_argument(
        "--mlp-result",
        help="Legacy path to the MLP TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--cnn-result",
        help="Legacy path to the CNN TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/comparisons",
        help="Directory where the comparison artifact JSON will be written.",
    )
    return parser


def main() -> int:
    """Run the artifact writer."""
    parser = build_parser()
    args = parser.parse_args()

    training_result_paths = _resolve_training_result_paths(args)

    comparison = build_track_a_comparison(training_result_paths)
    comparison.setdefault("created_at", _utc_now_iso())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{comparison['comparison_id']}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    recommended_candidate = _require_recommended_candidate(comparison)
    print(f"comparison_artifact_path={output_path}")
    print(f"recommended_model_type={recommended_candidate.get('model_type')}")
    print(f"recommendation_status={recommended_candidate['recommendation_status']}")
    return 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_training_result_paths(args: Any) -> list[str]:
    training_result_paths = list(args.training_result)
    if args.mlp_result or args.cnn_result:
        if not args.mlp_result or not args.cnn_result:
            raise ValueError(
                "Legacy comparison arguments require both --mlp-result and --cnn-result."
            )
        if training_result_paths:
            raise ValueError(
                "Use either --training-result or legacy --mlp-result/--cnn-result, not both."
            )
        training_result_paths = [args.mlp_result, args.cnn_result]

    if len(training_result_paths) < 2:
        raise ValueError("Track A comparison requires at least 2 TrainingResult paths.")

    for path_value in training_result_paths:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"TrainingResult not found: {path}")

    return training_result_paths


def _require_recommended_candidate(comparison: dict[str, Any]) -> dict[str, Any]:
    recommended_candidate = comparison.get("recommended_candidate")
    if not isinstance(recommended_candidate, dict):
        raise ValueError("Comparison result is missing recommended_candidate.")
    return recommended_candidate


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc

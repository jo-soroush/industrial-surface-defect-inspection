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
        "--mlp-result",
        required=True,
        help="Path to the MLP TrainingResult JSON artifact.",
    )
    parser.add_argument(
        "--cnn-result",
        required=True,
        help="Path to the CNN TrainingResult JSON artifact.",
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

    mlp_result_path = Path(args.mlp_result)
    cnn_result_path = Path(args.cnn_result)
    if not mlp_result_path.is_file():
        raise FileNotFoundError(f"MLP TrainingResult not found: {mlp_result_path}")
    if not cnn_result_path.is_file():
        raise FileNotFoundError(f"CNN TrainingResult not found: {cnn_result_path}")

    comparison = build_track_a_comparison(
        str(mlp_result_path),
        str(cnn_result_path),
    )
    comparison.setdefault("created_at", _utc_now_iso())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{comparison['comparison_id']}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    recommended_candidate = _require_recommended_candidate(comparison)
    print(f"comparison_artifact_path={output_path}")
    print(f"recommended_model_type={recommended_candidate['model_type']}")
    print(f"recommendation_status={recommended_candidate['recommendation_status']}")
    return 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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

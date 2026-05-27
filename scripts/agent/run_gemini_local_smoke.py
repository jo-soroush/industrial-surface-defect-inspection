"""Disabled-by-default local Gemini smoke harness skeleton.

This module contains a planning-only harness for a future manual local smoke.
It does not run a real Gemini call on import, and normal runtime remains
mock-first. The explicit future execution path is intentionally not implemented
in this slice.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disabled-by-default local Gemini smoke harness skeleton."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned checks only. This is the default behavior.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Request the future execution path. Disabled in this slice.",
    )
    parser.add_argument(
        "--i-understand-this-calls-gemini",
        action="store_true",
        help="Required for any future non-dry-run path.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Future sanitized question placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--page-id",
        default="",
        help="Future page identifier placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--section-id",
        default="",
        help="Future section identifier placeholder. Not used in this slice.",
    )
    parser.add_argument(
        "--component-id",
        default="",
        help="Future component identifier placeholder. Not used in this slice.",
    )
    return parser


def build_dry_run_lines() -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=DRY_RUN",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
        "planned_checks=lazy_sdk_import,key_presence,safety_guard,fallback,normal_route_unchanged",
    )


def build_blocked_lines() -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=BLOCKED",
        "reason=missing_confirmation_flag",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
    )


def build_not_implemented_lines() -> tuple[str, ...]:
    return (
        "gemini_local_smoke_status=NOT_IMPLEMENTED",
        "reason=real_smoke_execution_intentionally_not_implemented_in_this_slice",
        "no_real_gemini_api_call_was_made=true",
        "gemini_api_key_read=false",
        "normal_agent_route=mock_first",
        "provider_routing_activation=disabled",
        "future_real_smoke_requires_explicit_user_approval=true",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.execute or args.dry_run:
        for line in build_dry_run_lines():
            print(line)
        return 0

    if not args.i_understand_this_calls_gemini:
        for line in build_blocked_lines():
            print(line)
        return 2

    for line in build_not_implemented_lines():
        print(line)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

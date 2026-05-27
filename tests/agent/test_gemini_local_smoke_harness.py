from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "agent" / "run_gemini_local_smoke.py"


def load_harness_module():
    spec = importlib.util.spec_from_file_location(
        "run_gemini_local_smoke",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_import_does_not_read_gemini_api_key(monkeypatch) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read during import")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    assert module.build_parser is not None


def test_harness_import_does_not_add_google_sdk_modules() -> None:
    before_google_modules = {
        name for name in sys.modules if name == "google" or name.startswith("google.")
    }

    module = load_harness_module()

    after_google_modules = {
        name for name in sys.modules if name == "google" or name.startswith("google.")
    }

    assert after_google_modules == before_google_modules
    assert module.main is not None


def test_default_dry_run_exits_successfully_without_key_access(monkeypatch, capsys) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read in dry-run mode")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    exit_code = module.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "gemini_api_key_read=false" in captured.out
    assert "normal_agent_route=mock_first" in captured.out


def test_dry_run_output_says_no_real_api_call_was_made(capsys) -> None:
    module = load_harness_module()

    exit_code = module.main(["--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "no_real_gemini_api_call_was_made=true" in captured.out
    assert "provider_routing_activation=disabled" in captured.out
    assert "future_real_smoke_requires_explicit_user_approval=true" in captured.out


def test_missing_confirmation_flag_blocks_non_dry_run_before_key_access(monkeypatch, capsys) -> None:
    def _fail_getenv(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("GEMINI_API_KEY should not be read before confirmation")

    monkeypatch.setattr(os, "getenv", _fail_getenv)
    module = load_harness_module()

    exit_code = module.main(["--execute", "--question", "sanitized"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "reason=missing_confirmation_flag" in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out


def test_confirmation_flag_still_does_not_run_real_gemini_in_this_slice(capsys) -> None:
    module = load_harness_module()

    exit_code = module.main(
        [
            "--execute",
            "--i-understand-this-calls-gemini",
            "--question",
            "super-secret-test-question",
            "--page-id",
            "page-1",
            "--section-id",
            "section-2",
            "--component-id",
            "component-3",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 3
    assert "real_smoke_execution_intentionally_not_implemented_in_this_slice" in captured.out
    assert "super-secret-test-question" not in captured.out
    assert "GEMINI_API_KEY" not in captured.out
    assert "no_real_gemini_api_call_was_made=true" in captured.out


def test_harness_source_contains_no_real_sdk_or_network_imports() -> None:
    source = HARNESS_PATH.read_text(encoding="utf-8")

    assert "from google import genai" not in source
    assert "google.genai" not in source
    assert "google.generativeai" not in source
    assert "import google" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "urllib" not in source

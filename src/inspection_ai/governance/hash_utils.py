"""Hashing utility skeleton for Phase 3 governance.

This module defines the minimal source boundary for content hashing used in
artifact and run traceability. In Phase 3 it provides the future governed home
for computing stable file hashes without coupling hashing behavior to model,
training, or evaluation code.

This module will eventually handle file hashing, integrity checks, and
artifact-level identity support.
"""

from __future__ import annotations

from pathlib import Path


def compute_hash(file_path: str) -> str:
    """Return a placeholder hash value for the provided file path."""
    path = Path(file_path)
    return str(path)

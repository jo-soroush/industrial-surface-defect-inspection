"""Hashing utilities for artifact governance and traceability."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_hash(file_path: str) -> str:
    """Return the SHA-256 hex digest for an artifact file.

    The file is read in binary chunks so the function can be used for JSON,
    YAML, model checkpoints, images, CSV files, and other artifact types.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot compute hash; file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Cannot compute hash; path is not a file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

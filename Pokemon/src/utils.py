"""Shared helpers for cg-lib setup and runtime checks."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

from src.config import CG_LIB_DIR, KAGGLE_INPUT_GLOB


def find_cg_lib_path() -> Path | None:
    """Return cg-lib directory if found locally or on Kaggle."""
    if CG_LIB_DIR.is_dir():
        return CG_LIB_DIR

    matches = glob.glob(KAGGLE_INPUT_GLOB, recursive=True)
    if matches:
        return Path(matches[0])

    return None


def setup_cg_lib() -> Path:
    """Add cg-lib to sys.path. Raises FileNotFoundError if missing."""
    path = find_cg_lib_path()
    if path is None:
        raise FileNotFoundError(
            "cg-lib not found. Place it under input/raw/cg-lib or add as Kaggle input."
        )

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

    return path


def cg_lib_status() -> dict:
    """Summary for data-check notebooks."""
    path = find_cg_lib_path()
    return {
        "found": path is not None,
        "path": str(path) if path else None,
        "local_expected": str(CG_LIB_DIR),
        "on_path": str(path) in sys.path if path else False,
    }

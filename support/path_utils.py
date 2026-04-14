from __future__ import annotations

import os
from pathlib import Path


SUPPORT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SUPPORT_DIR.parent
OPENPI_ROOT = SCRIPTS_ROOT.parent
REPO_ROOT = OPENPI_ROOT / "repo"
BUILD_DIR = SCRIPTS_ROOT / ".build"
AUBO_SDK_ROOT = OPENPI_ROOT / "aubo_sdk"
CAPTURES_DIR = OPENPI_ROOT / "captures"


def get_openpi_root() -> Path:
    return OPENPI_ROOT


def get_repo_root() -> Path:
    return REPO_ROOT


def get_build_dir() -> Path:
    return BUILD_DIR


def get_captures_dir() -> Path:
    return CAPTURES_DIR


def get_sdk_root() -> Path:
    override = os.environ.get("OPENPI_SDK_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    if not AUBO_SDK_ROOT.exists():
        raise FileNotFoundError(f"AUBO SDK base directory not found: {AUBO_SDK_ROOT}")

    candidates = sorted(
        path
        for path in AUBO_SDK_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("aubo_sdk-")
    )
    if not candidates:
        raise FileNotFoundError(f"No AUBO SDK found under: {AUBO_SDK_ROOT}")
    return candidates[0]

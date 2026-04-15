from __future__ import annotations

import os
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = UTILS_DIR.parent
OPENPI_ROOT = SCRIPTS_ROOT.parent
SUPPORT_DIR = SCRIPTS_ROOT / "support"
REPO_ROOT = OPENPI_ROOT / "repo"
BUILD_DIR = SCRIPTS_ROOT / ".build"
LOG_DIR = SCRIPTS_ROOT / "log"
AUBO_SDK_ROOT = OPENPI_ROOT / "aubo_sdk"
CAPTURES_DIR = OPENPI_ROOT / "captures"


def get_utils_dir() -> Path:
    return UTILS_DIR


def get_scripts_root() -> Path:
    return SCRIPTS_ROOT


def get_openpi_root() -> Path:
    return OPENPI_ROOT


def get_support_dir() -> Path:
    return SUPPORT_DIR


def get_repo_root() -> Path:
    return REPO_ROOT


def get_build_dir() -> Path:
    return BUILD_DIR


def get_log_dir() -> Path:
    return LOG_DIR


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

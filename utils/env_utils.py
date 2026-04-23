from __future__ import annotations

import os
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = UTILS_DIR.parent
DEFAULT_CONFIG_FILE = SCRIPTS_ROOT / "config"

_LOADED_ENV_FILE: Path | None = None


def _parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[7:].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: str | Path, *, override: bool = False) -> Path | None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return None

    global _LOADED_ENV_FILE
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    _LOADED_ENV_FILE = env_path.resolve()
    return _LOADED_ENV_FILE


def load_default_env(*, override: bool = False) -> Path | None:
    global _LOADED_ENV_FILE
    if _LOADED_ENV_FILE is not None and not override:
        return _LOADED_ENV_FILE

    explicit = os.environ.get("OPENPI_ENV_FILE", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(DEFAULT_CONFIG_FILE)
    for candidate in candidates:
        loaded = load_env_file(candidate, override=override)
        if loaded is not None:
            return loaded
    return None


def get_loaded_env_file() -> Path | None:
    return _LOADED_ENV_FILE

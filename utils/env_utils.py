from __future__ import annotations

import os
from pathlib import Path
from typing import Any


UTILS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = UTILS_DIR.parent
DEFAULT_CONFIG_FILE = SCRIPTS_ROOT / "config.yaml"
LEGACY_CONFIG_FILE = SCRIPTS_ROOT / "config"

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


def _stringify_env_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _iter_yaml_env_items(payload: Any) -> list[tuple[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("YAML config must be a mapping")

    root = payload.get("env", payload)
    if not isinstance(root, dict):
        raise ValueError("YAML config 'env' section must be a mapping")

    items: list[tuple[str, str]] = []
    for raw_key, raw_value in root.items():
        key = str(raw_key).strip()
        if not key or not key.startswith("OPENPI_"):
            continue
        value = _stringify_env_value(raw_value)
        if value is not None:
            items.append((key, value))
    return items


def _parse_yaml_scalar(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _parse_simple_yaml_env(text: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    saw_env_section = False
    in_env_section = False
    env_indent: int | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if stripped == "env:":
            saw_env_section = True
            in_env_section = True
            env_indent = indent
            continue
        if saw_env_section:
            if env_indent is not None and indent <= env_indent:
                in_env_section = False
            if not in_env_section:
                continue
        if ":" not in stripped:
            continue
        key, value_text = stripped.split(":", 1)
        key = key.strip()
        if not key.startswith("OPENPI_"):
            continue
        value = _parse_yaml_scalar(value_text)
        if value is not None:
            items.append((key, value))
    return items


def _load_yaml_env_file(env_path: Path, *, override: bool) -> Path:
    try:
        import yaml
    except ImportError:
        items = _parse_simple_yaml_env(env_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(env_path.read_text(encoding="utf-8"))
        items = _iter_yaml_env_items(payload or {})

    global _LOADED_ENV_FILE
    for key, value in items:
        if override or key not in os.environ:
            os.environ[key] = value
    _LOADED_ENV_FILE = env_path.resolve()
    return _LOADED_ENV_FILE


def _load_legacy_env_file(env_path: Path, *, override: bool) -> Path:
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


def load_env_file(path: str | Path, *, override: bool = False) -> Path | None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return None

    if env_path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml_env_file(env_path, override=override)
    return _load_legacy_env_file(env_path, override=override)


def load_default_env(*, override: bool = False) -> Path | None:
    global _LOADED_ENV_FILE
    if _LOADED_ENV_FILE is not None and not override:
        return _LOADED_ENV_FILE

    explicit = (
        os.environ.get("OPENPI_CONFIG_FILE", "").strip()
        or os.environ.get("OPENPI_ENV_FILE", "").strip()
    )
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(DEFAULT_CONFIG_FILE)
    candidates.append(LEGACY_CONFIG_FILE)
    for candidate in candidates:
        loaded = load_env_file(candidate, override=override)
        if loaded is not None:
            return loaded
    return None


def get_loaded_env_file() -> Path | None:
    return _LOADED_ENV_FILE

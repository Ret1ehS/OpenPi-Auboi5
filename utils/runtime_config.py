from __future__ import annotations

import os
from pathlib import Path

from utils.env_utils import load_default_env
from utils.path_utils import (
    SCRIPTS_ROOT,
    get_openpi_root,
    get_repo_root,
    get_support_dir,
)


load_default_env()


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def _env_path(name: str, default: Path, *, base: Path | None = None) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw).expanduser() if raw else Path(default)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _env_is_set(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw.strip() != ""


DEFAULT_CONFIG_NAME = _env_str("OPENPI_POLICY_CONFIG_NAME", "pi05_aubo_agv_lora")
DEFAULT_POLICY_BACKEND = _env_str("OPENPI_POLICY_BACKEND", "auto").lower()
DEFAULT_CHECKPOINT_DIR = _env_path(
    "OPENPI_CHECKPOINT_DIR",
    Path("checkpoints") / DEFAULT_CONFIG_NAME / "my_eighth_run" / "29999",
    base=get_repo_root(),
)
DEFAULT_PYTORCH_CHECKPOINT_DIR = _env_path(
    "OPENPI_PYTORCH_CHECKPOINT_DIR",
    DEFAULT_CHECKPOINT_DIR,
    base=get_repo_root(),
)
DEFAULT_PYTORCH_DEVICE = _env_str("OPENPI_PYTORCH_DEVICE", "cuda")
KUBECONFIG_PATH = _env_path(
    "OPENPI_KUBECONFIG",
    get_support_dir() / "kubeconfig.yaml",
    base=SCRIPTS_ROOT,
)
DEFAULT_NAMESPACE = _env_str("OPENPI_K8S_NAMESPACE", "wangrui")
DEFAULT_LOCAL_PORT = _env_int("OPENPI_POLICY_LOCAL_PORT", 8000)
DEFAULT_REMOTE_PORT = _env_int("OPENPI_POLICY_REMOTE_PORT", 8000)

DEFAULT_AUBO_RPC_PORT = _env_int("OPENPI_ROBOT_PORT", 30004)
DEFAULT_AUBO_USER = _env_str("OPENPI_ROBOT_USER", "aubo")
DEFAULT_AUBO_PASSWORD = _env_str("OPENPI_ROBOT_PASSWORD", "123456")
DEFAULT_ROBOT_IP = _env_str("OPENPI_ROBOT_IP", "192.168.1.100")
ROBOT_PASSWORD_FROM_ENV = _env_is_set("OPENPI_ROBOT_PASSWORD")
ROBOT_IP_FROM_ENV = _env_is_set("OPENPI_ROBOT_IP")

DEFAULT_GRIPPER_PORT = _env_str(
    "OPENPI_GRIPPER_PORT",
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AB0LIYTU-if00-port0",
)
DEFAULT_GRIPPER_FALLBACK_PORT = _env_str(
    "OPENPI_GRIPPER_FALLBACK_PORT",
    "/dev/ttyUSB3",
)
DEFAULT_FORCE_SENSOR_PORT = _env_str(
    "OPENPI_FORCE_SENSOR_PORT",
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10PHIEG-if00-port0",
)
DEFAULT_FORCE_SENSOR_FALLBACK_PORT = _env_str(
    "OPENPI_FORCE_SENSOR_FALLBACK_PORT",
    "/dev/ttyUSB4",
)

DEFAULT_OBSERVER_PYTHON = _env_path(
    "OPENPI_TASK_OBSERVER_PYTHON",
    get_openpi_root() / "venvs" / "vllm-jp62-clean" / "bin" / "python",
)
DEFAULT_OBSERVER_MODEL = _env_path(
    "OPENPI_TASK_OBSERVER_MODEL",
    get_openpi_root() / "modelscope_models" / "google" / "gemma-4-E2B-it",
)


def get_robot_config_warnings() -> tuple[str, ...]:
    warnings: list[str] = []
    if not ROBOT_IP_FROM_ENV:
        warnings.append(
            "OPENPI_ROBOT_IP is not set; using the built-in default robot address."
        )
    if not ROBOT_PASSWORD_FROM_ENV:
        warnings.append(
            "OPENPI_ROBOT_PASSWORD is not set; using the built-in default robot password."
        )
    return tuple(warnings)


__all__ = [
    "DEFAULT_AUBO_PASSWORD",
    "DEFAULT_AUBO_RPC_PORT",
    "DEFAULT_AUBO_USER",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_POLICY_BACKEND",
    "DEFAULT_PYTORCH_CHECKPOINT_DIR",
    "DEFAULT_PYTORCH_DEVICE",
    "DEFAULT_FORCE_SENSOR_FALLBACK_PORT",
    "DEFAULT_FORCE_SENSOR_PORT",
    "DEFAULT_GRIPPER_FALLBACK_PORT",
    "DEFAULT_GRIPPER_PORT",
    "DEFAULT_LOCAL_PORT",
    "DEFAULT_NAMESPACE",
    "DEFAULT_OBSERVER_MODEL",
    "DEFAULT_OBSERVER_PYTHON",
    "DEFAULT_REMOTE_PORT",
    "DEFAULT_ROBOT_IP",
    "KUBECONFIG_PATH",
    "ROBOT_IP_FROM_ENV",
    "ROBOT_PASSWORD_FROM_ENV",
    "get_robot_config_warnings",
]

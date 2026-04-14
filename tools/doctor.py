#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = TOOLS_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from utils.env_utils import get_loaded_env_file, load_default_env
from utils.path_utils import (
    get_build_dir,
    get_captures_dir,
    get_log_dir,
    get_openpi_root,
    get_repo_root,
    get_scripts_root,
    get_sdk_root,
    get_support_dir,
)
from utils.runtime_config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_FORCE_SENSOR_FALLBACK_PORT,
    DEFAULT_FORCE_SENSOR_PORT,
    DEFAULT_GRIPPER_FALLBACK_PORT,
    DEFAULT_GRIPPER_PORT,
    DEFAULT_OBSERVER_MODEL,
    DEFAULT_OBSERVER_PYTHON,
    DEFAULT_ROBOT_IP,
    KUBECONFIG_PATH,
)


@dataclass
class CheckResult:
    level: str
    label: str
    detail: str


def _ok(label: str, detail: str) -> CheckResult:
    return CheckResult("OK", label, detail)


def _warn(label: str, detail: str) -> CheckResult:
    return CheckResult("WARN", label, detail)


def _fail(label: str, detail: str) -> CheckResult:
    return CheckResult("FAIL", label, detail)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_path(label: str, path: Path, *, required: bool = True, kind: str = "path") -> CheckResult:
    if path.exists():
        return _ok(label, f"{kind} exists: {path}")
    if required:
        return _fail(label, f"{kind} missing: {path}")
    return _warn(label, f"{kind} missing: {path}")


def _runtime_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(_check_path("scripts_root", get_scripts_root(), kind="dir"))
    results.append(_check_path("openpi_root", get_openpi_root(), kind="dir"))
    results.append(_check_path("repo_root", get_repo_root(), kind="dir"))
    results.append(_check_path("support_dir", get_support_dir(), kind="dir"))
    results.append(_check_path("build_dir", get_build_dir(), required=False, kind="dir"))
    results.append(_check_path("log_dir", get_log_dir(), required=False, kind="dir"))
    results.append(_check_path("captures_dir", get_captures_dir(), required=False, kind="dir"))

    try:
        sdk_root = get_sdk_root()
        results.append(_check_path("sdk_root", sdk_root, kind="dir"))
    except Exception as exc:
        results.append(_fail("sdk_root", str(exc)))

    results.append(_check_path("checkpoint_dir", DEFAULT_CHECKPOINT_DIR, required=False, kind="dir"))
    results.append(_check_path("kubeconfig", KUBECONFIG_PATH, required=False, kind="file"))
    results.append(_check_path("gripper_port", Path(DEFAULT_GRIPPER_PORT), required=False, kind="device"))
    results.append(_check_path("gripper_fallback", Path(DEFAULT_GRIPPER_FALLBACK_PORT), required=False, kind="device"))
    results.append(_check_path("force_sensor_port", Path(DEFAULT_FORCE_SENSOR_PORT), required=False, kind="device"))
    results.append(_check_path("force_sensor_fallback", Path(DEFAULT_FORCE_SENSOR_FALLBACK_PORT), required=False, kind="device"))

    helper_bins = [
        get_build_dir() / "tool_io_helper",
        get_build_dir() / "joint_control_helper",
        get_build_dir() / "tcp_control_helper",
    ]
    for helper in helper_bins:
        results.append(_check_path(helper.name, helper, required=False, kind="file"))

    for module_name in ("numpy", "cv2", "serial", "pyorbbecsdk"):
        if _has_module(module_name):
            results.append(_ok(f"module:{module_name}", "importable"))
        else:
            results.append(_warn(f"module:{module_name}", "not importable in current Python"))

    results.append(_ok("robot_ip", DEFAULT_ROBOT_IP))
    return results


def _observer_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(_check_path("observer_python", DEFAULT_OBSERVER_PYTHON, required=False, kind="file"))
    results.append(_check_path("observer_model", DEFAULT_OBSERVER_MODEL, required=False, kind="path"))
    observer_modules = ("torch", "transformers", "PIL")
    for module_name in observer_modules:
        if _has_module(module_name):
            results.append(_ok(f"module:{module_name}", "importable"))
        else:
            results.append(_warn(f"module:{module_name}", "not importable in current Python"))
    return results


def _config_checks() -> list[CheckResult]:
    env_file = load_default_env()
    results: list[CheckResult] = []
    if env_file is None:
        results.append(_warn("env_file", "no config/local.env or config/niic.env found"))
    else:
        results.append(_ok("env_file", str(get_loaded_env_file())))
    return results


def _print_results(results: list[CheckResult]) -> int:
    fail_count = 0
    for item in results:
        print(f"[{item.level}] {item.label}: {item.detail}")
        if item.level == "FAIL":
            fail_count += 1
    return fail_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity checks for the Jetson-side OpenPI scripts repo.")
    parser.add_argument(
        "--section",
        choices=("all", "runtime", "observer", "config"),
        default="all",
    )
    args = parser.parse_args()

    sections: list[list[CheckResult]] = []
    if args.section in {"all", "config"}:
        sections.append(_config_checks())
    if args.section in {"all", "runtime"}:
        sections.append(_runtime_checks())
    if args.section in {"all", "observer"}:
        sections.append(_observer_checks())

    results = [item for section in sections for item in section]
    fail_count = _print_results(results)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

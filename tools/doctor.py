#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
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
    get_robot_config_warnings,
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


def _skip(label: str, detail: str) -> CheckResult:
    return CheckResult("SKIP", label, detail)


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
    for detail in get_robot_config_warnings():
        results.append(_warn("robot_config", detail))
    return results


def _diagnostic_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_diagnostic_prerequisite_checks())
    results.extend(_robot_connectivity_checks())
    results.extend(_policy_connectivity_checks())
    results.extend(_sensor_connectivity_checks())
    return results


def _diagnostic_prerequisite_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    try:
        sdk_root = get_sdk_root()
        results.append(_ok("prereq.robot_sdk", f"sdk root available: {sdk_root}"))
    except Exception as exc:
        results.append(_warn("prereq.robot_sdk", str(exc)))

    tcp_helper = get_build_dir() / "tcp_control_helper"
    if tcp_helper.exists():
        results.append(_ok("prereq.robot_helper", f"helper exists: {tcp_helper}"))
    else:
        results.append(_warn("prereq.robot_helper", f"helper missing: {tcp_helper}"))

    if KUBECONFIG_PATH.exists():
        results.append(_ok("prereq.policy_remote", f"kubeconfig exists: {KUBECONFIG_PATH}"))
    else:
        results.append(_warn("prereq.policy_remote", f"kubeconfig missing: {KUBECONFIG_PATH}"))

    if DEFAULT_CHECKPOINT_DIR.exists():
        results.append(_ok("prereq.policy_local", f"checkpoint path exists: {DEFAULT_CHECKPOINT_DIR}"))
    else:
        results.append(_warn("prereq.policy_local", f"checkpoint path missing: {DEFAULT_CHECKPOINT_DIR}"))

    if _has_module("serial"):
        results.append(_ok("prereq.serial", "pyserial importable"))
    else:
        results.append(_warn("prereq.serial", "pyserial is not importable in current Python"))

    if _has_module("cv2"):
        results.append(_ok("prereq.cv2", "opencv-python importable"))
    else:
        results.append(_warn("prereq.cv2", "cv2 is not importable in current Python"))

    if _has_module("pyorbbecsdk"):
        results.append(_ok("prereq.pyorbbecsdk", "pyorbbecsdk importable"))
    else:
        results.append(_warn("prereq.pyorbbecsdk", "pyorbbecsdk is not importable in current Python"))

    return results


def _robot_connectivity_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        get_sdk_root()
    except Exception as exc:
        results.append(_skip("check.robot_connectivity", f"skipped: robot SDK is unavailable ({exc})"))
        return results
    tcp_helper = get_build_dir() / "tcp_control_helper"
    if not tcp_helper.exists():
        results.append(_skip("check.robot_connectivity", f"skipped: helper missing ({tcp_helper})"))
        return results
    try:
        from support.tcp_control import get_robot_snapshot

        started = time.monotonic()
        snapshot = get_robot_snapshot(use_daemon=False)
        elapsed = time.monotonic() - started
        joint_count = int(getattr(snapshot.joint_q, "size", 0))
        tcp_dim = int(getattr(snapshot.tcp_pose, "size", 0))
        if joint_count == 6 and tcp_dim == 6:
            results.append(
                _ok(
                    "check.robot_connectivity",
                    f"snapshot ok in {elapsed:.2f}s (joint_q={joint_count}, tcp_pose={tcp_dim}, mode={snapshot.robot_mode or 'unknown'})",
                )
            )
        else:
            results.append(
                _warn(
                    "check.robot_connectivity",
                    f"snapshot returned unexpected dimensions in {elapsed:.2f}s (joint_q={joint_count}, tcp_pose={tcp_dim})",
                )
            )
    except Exception as exc:
        results.append(_fail("check.robot_connectivity", f"{type(exc).__name__}: {exc}"))
    return results


def _policy_connectivity_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    if not KUBECONFIG_PATH.exists():
        results.append(_skip("check.policy_remote", f"skipped: kubeconfig missing ({KUBECONFIG_PATH})"))
    else:
        try:
            from support.load_policy import PolicyLoadSpec, RemotePolicyRunner
        except Exception as exc:
            results.append(_warn("check.policy_remote", f"policy loader unavailable: {type(exc).__name__}: {exc}"))
        else:
            runner = RemotePolicyRunner()
            started = time.monotonic()
            try:
                runner.connect(timeout=5.0)
                elapsed = time.monotonic() - started
                metadata_keys = sorted(runner.metadata.keys())
                results.append(
                    _ok(
                        "check.policy_remote",
                        f"remote policy reachable in {elapsed:.2f}s (metadata keys: {metadata_keys[:6]})",
                    )
                )
            except Exception as exc:
                results.append(_fail("check.policy_remote", f"{type(exc).__name__}: {exc}"))
            finally:
                try:
                    runner.close()
                except Exception:
                    pass

    checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR).resolve()
    if not checkpoint_dir.exists():
        results.append(_skip("check.policy_local", f"skipped: checkpoint path missing ({checkpoint_dir})"))
    else:
        results.append(
            _skip(
                "check.policy_local",
                f"skipped: local checkpoint validation is reserved for future implementation ({checkpoint_dir})",
            )
        )
    return results


def _sensor_connectivity_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_force_sensor_checks())
    results.extend(_gripper_checks())
    results.extend(_camera_checks())
    return results


def _force_sensor_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    sensor = None
    if not _has_module("serial"):
        results.append(_skip("check.force_sensor", "skipped: pyserial is not available in current Python"))
        return results
    try:
        from support.force_sensor import ForceSensor

        sensor = ForceSensor()
        started = time.monotonic()
        sensor.start()
        deadline = time.monotonic() + 2.0
        reading = None
        while time.monotonic() < deadline:
            reading = sensor.get()
            if reading is not None:
                break
            time.sleep(0.05)
        elapsed = time.monotonic() - started
        if reading is None:
            results.append(_fail("check.force_sensor", f"no reading received within {elapsed:.2f}s"))
        else:
            results.append(
                _ok(
                    "check.force_sensor",
                    "reading ok "
                    f"in {elapsed:.2f}s (fx={reading.fx:+.2f}N, fy={reading.fy:+.2f}N, fz={reading.fz:+.2f}N)",
                )
            )
    except Exception as exc:
        results.append(_fail("check.force_sensor", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            sensor.stop()
        except Exception:
            pass
    return results


def _gripper_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    if not _has_module("serial"):
        results.append(_skip("check.gripper", "skipped: pyserial is not available in current Python"))
        return results
    try:
        get_sdk_root()
    except Exception as exc:
        results.append(_skip("check.gripper", f"skipped: robot SDK is unavailable ({exc})"))
        return results
    tool_io_helper = get_build_dir() / "tool_io_helper"
    if not tool_io_helper.exists():
        results.append(_skip("check.gripper", f"skipped: tool IO helper missing ({tool_io_helper})"))
        return results
    try:
        from support.gripper_control import resolve_port, GripperController

        port = resolve_port(None)
        started = time.monotonic()
        with GripperController(port=port) as ctrl:
            status = ctrl.read_status()
        elapsed = time.monotonic() - started
        values = (status.position, status.done, status.unhomed)
        if all(value is None for value in values):
            results.append(_fail("check.gripper", f"no valid status register returned from {port}"))
        else:
            results.append(
                _ok(
                    "check.gripper",
                    f"status ok in {elapsed:.2f}s via {port} (position={status.position}, done={status.done}, unhomed={status.unhomed})",
                )
            )
    except Exception as exc:
        results.append(_fail("check.gripper", f"{type(exc).__name__}: {exc}"))
    return results


def _camera_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_modules: list[str] = []
    if not _has_module("cv2"):
        missing_modules.append("cv2")
    if not _has_module("pyorbbecsdk"):
        missing_modules.append("pyorbbecsdk")
    if missing_modules:
        results.append(
            _skip(
                "check.camera_pair",
                f"skipped: missing camera dependencies ({', '.join(missing_modules)})",
            )
        )
        return results
    try:
        from support.get_obs import CameraPair

        started = time.monotonic()
        cameras = CameraPair()
        try:
            main_img, wrist_img = cameras.grab()
        finally:
            cameras.stop()
        elapsed = time.monotonic() - started
        if main_img.size == 0 or wrist_img.size == 0:
            results.append(_fail("camera_pair", f"camera grab returned empty frame in {elapsed:.2f}s"))
        else:
            results.append(
                _ok(
                    "check.camera_pair",
                    f"frames ok in {elapsed:.2f}s (main={tuple(main_img.shape)}, wrist={tuple(wrist_img.shape)})",
                )
            )
    except Exception as exc:
        results.append(_fail("check.camera_pair", f"{type(exc).__name__}: {exc}"))
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
        choices=("all", "runtime", "observer", "config", "diagnostic"),
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
    if args.section in {"all", "diagnostic"}:
        sections.append(_diagnostic_checks())

    results = [item for section in sections for item in section]
    fail_count = _print_results(results)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

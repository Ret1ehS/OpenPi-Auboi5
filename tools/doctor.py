#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
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
    DEFAULT_PYTORCH_CHECKPOINT_DIR,
    DEFAULT_PYTORCH_DEVICE,
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


def _observer_python_path() -> Path:
    raw = os.environ.get("OPENPI_TASK_OBSERVER_PYTHON", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (get_openpi_root() / "venvs" / "vllm-jp62-clean" / "bin" / "python").resolve()


def _observer_model_path() -> Path:
    raw = os.environ.get("OPENPI_TASK_OBSERVER_MODEL", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (get_openpi_root() / "modelscope_models" / "google" / "gemma-4-E2B-it").resolve()


def _robot_config_warnings() -> tuple[str, ...]:
    warnings: list[str] = []
    if not os.environ.get("OPENPI_ROBOT_IP", "").strip():
        warnings.append("OPENPI_ROBOT_IP is not set; using the built-in default robot address.")
    if not os.environ.get("OPENPI_ROBOT_PASSWORD", "").strip():
        warnings.append("OPENPI_ROBOT_PASSWORD is not set; using the built-in default robot password.")
    return tuple(warnings)


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

    results.extend(_pytorch_runtime_checks())

    results.append(_ok("robot_ip", DEFAULT_ROBOT_IP))
    for detail in _robot_config_warnings():
        results.append(_warn("robot_config", detail))
    return results


def _diagnostic_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_diagnostic_prerequisite_checks())
    results.extend(_pytorch_diagnostic_checks())
    results.extend(_robot_connectivity_checks())
    results.extend(_policy_connectivity_checks())
    results.extend(_sensor_connectivity_checks())
    return results


def _select_pytorch_runtime_python() -> Path:
    explicit = os.environ.get("OPENPI_PYTORCH_RUNTIME_PYTHON", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (get_openpi_root() / "miniforge3" / "envs" / "openpi-py310-torch" / "bin" / "python").resolve()


def _pytorch_runtime_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    runtime_python = _select_pytorch_runtime_python()
    results.append(_check_path("pytorch_runtime_python", runtime_python, required=False, kind="file"))
    results.append(_check_path("pytorch_checkpoint_dir", DEFAULT_PYTORCH_CHECKPOINT_DIR, required=False, kind="dir"))
    weight_file = DEFAULT_PYTORCH_CHECKPOINT_DIR / "model.safetensors"
    config_file = DEFAULT_PYTORCH_CHECKPOINT_DIR / "config.json"
    if weight_file.exists() and config_file.exists():
        results.append(_ok("pytorch_checkpoint_export", f"export looks valid: {weight_file.name} + {config_file.name}"))
    else:
        results.append(
            _warn(
                "pytorch_checkpoint_export",
                f"PyTorch export incomplete under {DEFAULT_PYTORCH_CHECKPOINT_DIR} "
                f"(need {weight_file.name} and {config_file.name})",
            )
        )

    backend = os.environ.get("OPENPI_POLICY_BACKEND", "").strip() or "auto"
    attn_backend = os.environ.get("OPENPI_PYTORCH_ATTN_BACKEND", "").strip() or "eager"
    sample_steps = os.environ.get("OPENPI_SAMPLE_NUM_STEPS", "").strip() or "default"
    device = os.environ.get("OPENPI_PYTORCH_DEVICE", "").strip() or DEFAULT_PYTORCH_DEVICE
    trt_vision = os.environ.get("OPENPI_PYTORCH_TRT_VISION", "").strip() or "0"
    trt_denoise = os.environ.get("OPENPI_PYTORCH_TRT_DENOISE", "").strip() or "0"
    results.append(
        _ok(
            "pytorch_env",
            "backend="
            f"{backend} device={device} attn={attn_backend} sample_steps={sample_steps} "
            f"trt_vision={trt_vision} trt_denoise={trt_denoise}",
        )
    )
    return results


def _pytorch_diagnostic_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    runtime_python = _select_pytorch_runtime_python()
    if not runtime_python.exists():
        results.append(_skip("check.pytorch_runtime", f"skipped: runtime python missing ({runtime_python})"))
        return results
    if not DEFAULT_PYTORCH_CHECKPOINT_DIR.exists():
        results.append(
            _skip(
                "check.pytorch_runtime",
                f"skipped: pytorch checkpoint dir missing ({DEFAULT_PYTORCH_CHECKPOINT_DIR})",
            )
        )
        return results
    weight_file = DEFAULT_PYTORCH_CHECKPOINT_DIR / "model.safetensors"
    config_file = DEFAULT_PYTORCH_CHECKPOINT_DIR / "config.json"
    if not (weight_file.exists() and config_file.exists()):
        results.append(
            _skip(
                "check.pytorch_runtime",
                "skipped: pytorch checkpoint export incomplete "
                f"({DEFAULT_PYTORCH_CHECKPOINT_DIR}, need {weight_file.name} and {config_file.name})",
            )
        )
        return results

    probe = r"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent))

out = {
    "python": sys.executable,
    "backend": os.environ.get("OPENPI_POLICY_BACKEND", ""),
    "device": os.environ.get("OPENPI_PYTORCH_DEVICE", ""),
    "attn_backend": os.environ.get("OPENPI_PYTORCH_ATTN_BACKEND", ""),
    "trt_vision_requested": os.environ.get("OPENPI_PYTORCH_TRT_VISION", ""),
    "trt_denoise_requested": os.environ.get("OPENPI_PYTORCH_TRT_DENOISE", ""),
}

import torch
out["torch_version"] = str(torch.__version__)
out["cuda_available"] = bool(torch.cuda.is_available())
if torch.cuda.is_available():
    out["cuda_device_name"] = torch.cuda.get_device_name(0)

for module_name in ("tensorrt", "torch_tensorrt"):
    try:
        __import__(module_name)
        out[f"module_{module_name}"] = True
    except Exception as exc:
        out[f"module_{module_name}"] = f"{type(exc).__name__}: {exc}"

from support.pytorch_support import OpenPIPyTorchPolicy

policy = OpenPIPyTorchPolicy(
    repo_root=Path.cwd().parent / "repo",
    checkpoint_dir=Path(os.environ["OPENPI_PYTORCH_CHECKPOINT_DIR"]),
    pytorch_device=os.environ.get("OPENPI_PYTORCH_DEVICE", "cuda"),
)
try:
    out["runner_type"] = type(policy).__name__
    out["metadata"] = policy.metadata
finally:
    policy.close()

print(json.dumps(out, ensure_ascii=False))
"""

    probe_env = dict(os.environ)
    probe_env["PYTHONNOUSERSITE"] = "1"
    probe_env["OPENPI_POLICY_BACKEND"] = "pytorch"
    probe_env.setdefault("OPENPI_PYTORCH_DEVICE", DEFAULT_PYTORCH_DEVICE)
    probe_env["OPENPI_PYTORCH_CHECKPOINT_DIR"] = str(DEFAULT_PYTORCH_CHECKPOINT_DIR)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(runtime_python), "-c", probe],
            cwd=str(get_scripts_root()),
            env=probe_env,
            capture_output=True,
            text=True,
            timeout=420,
        )
    except subprocess.TimeoutExpired:
        results.append(_fail("check.pytorch_runtime", "timed out after 420s"))
        return results
    except Exception as exc:
        results.append(_fail("check.pytorch_runtime", f"{type(exc).__name__}: {exc}"))
        return results

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"returncode={completed.returncode}"
        results.append(_fail("check.pytorch_runtime", f"{detail}"))
        return results

    stdout = (completed.stdout or "").strip().splitlines()
    payload_line = stdout[-1] if stdout else ""
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError:
        results.append(_fail("check.pytorch_runtime", f"unexpected probe output: {payload_line[:240]}"))
        return results

    metadata = payload.get("metadata", {}) or {}
    detail = (
        f"ok in {elapsed:.2f}s "
        f"(runner={payload.get('runner_type')}, "
        f"cuda={payload.get('cuda_available')}, "
        f"torch={payload.get('torch_version')}, "
        f"attn={metadata.get('attention_backend', 'unknown')}, "
        f"trt_vision={metadata.get('trt_vision_enabled', 'unknown')}, "
        f"trt_denoise={metadata.get('trt_denoise_enabled', 'unknown')}, "
        f"load_s={metadata.get('load_s', 'unknown')})"
    )
    results.append(_ok("check.pytorch_runtime", detail))
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
    results.append(_check_path("observer_python", _observer_python_path(), required=False, kind="file"))
    results.append(_check_path("observer_model", _observer_model_path(), required=False, kind="path"))
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

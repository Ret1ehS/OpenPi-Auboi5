#!/usr/bin/env python3
"""
Jetson-side TCP delta controller for the real AUBO i5.

Primary purpose:
- be imported by the future real-robot OpenPI main program
- take OpenPI-style TCP delta actions
- integrate target TCP pose
- call the robot's own IK SDK and joint motion interfaces
"""

from __future__ import annotations

import argparse
import atexit
import os
import queue
import math
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from support.pose_align import (
    POSE_DIM,
    get_alignment_mode,
    real_pose_to_sim,
    set_alignment_mode,
    sim_pose_to_real,
    wrap_euler_zyx,
)
from utils.path_utils import get_build_dir, get_log_dir, get_sdk_root
from utils.runtime_config import (
    DEFAULT_AUBO_PASSWORD,
    DEFAULT_AUBO_RPC_PORT,
    DEFAULT_AUBO_USER,
    DEFAULT_ROBOT_IP,
)


DEFAULT_SDK_ROOT = str(get_sdk_root())
DEFAULT_HELPER_BIN = str(get_build_dir() / "tcp_control_helper")
DEFAULT_HELPER_LOG_DIR = get_log_dir()
DEFAULT_HELPER_LOG_FILE = DEFAULT_HELPER_LOG_DIR / "tcp_control_helper.log"

DEFAULT_PORT = DEFAULT_AUBO_RPC_PORT
DEFAULT_USER = DEFAULT_AUBO_USER
DEFAULT_PASSWORD = DEFAULT_AUBO_PASSWORD

DEFAULT_SPEED_DEG = 10.0
DEFAULT_ACC_DEG = 20.0
DEFAULT_SPEED_FRACTION = 1.0
DEFAULT_TRACK_CONTROL_DT_S = 0.01
DEFAULT_TCP_LINEAR_SPEED_MPS = 0.05
DEFAULT_TCP_ANGULAR_SPEED_RADPS = 0.60
CONSTANT_SPEED_CORNER_MIN_SCALE = 0.35
CONSTANT_SPEED_CORNER_RAMP_STEPS = 8

RESET_ERR_M = 0.10
DEFAULT_Z_MIN_M = 0.180  # TCP minimum z height (180 mm), clip anything below
FORCE_GUARD_SOFT_FZ_N = 10.0
FORCE_GUARD_SOFT_RELEASE_FZ_N = 12.0
FORCE_GUARD_HARD_FZ_N = 0.0
FORCE_GUARD_HARD_RELEASE_FZ_N = 2.0
FORCE_GUARD_FZ_SIGN = 1.0
FORCE_GUARD_READING_TIMEOUT_S = 0.25
FORCE_GUARD_FILTER_TAU_S = 0.08
FORCE_GUARD_FILTER_RESET_S = 0.5
FORCE_GUARD_READING_HOLD_S = 0.15
FORCE_ADMITTANCE_TARGET_FZ_N = FORCE_GUARD_SOFT_FZ_N
FORCE_ADMITTANCE_VIRTUAL_MASS_KG = 6.0
FORCE_ADMITTANCE_DAMPING_N_S_PER_M = 220.0
FORCE_ADMITTANCE_STIFFNESS_N_PER_M = 2000.0
FORCE_ADMITTANCE_MAX_BLOCK_Z_M = 0.010
FORCE_ADMITTANCE_MIN_DT_S = 0.001
FORCE_ADMITTANCE_MAX_DT_S = 0.05
FORCE_ADMITTANCE_FORCE_DEADBAND_N = 1.5
_FORCE_GUARD_UNSET = object()
DAEMON_STARTUP_TIMEOUT_S = 10.0
DAEMON_COMMAND_TIMEOUT_S = 10.0
DAEMON_STOP_TIMEOUT_S = 3.0
HELPER_RUN_TIMEOUT_S = 20.0
_DAEMON_SENTINEL = object()


def _subprocess_session_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {}
    return {"start_new_session": True}

JOINT_NAMES = [
    "base_link",
    "shoulder_Link",
    "upperArm_Link",
    "foreArm_Link",
    "wrist1_Link",
    "wrist2_Link",
    "wrist3_Link",
]

OPENPI_DELTA_DIM = 6
JOINT_DIM = 6


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.astype(float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass
class RobotSnapshot:
    joint_q: np.ndarray
    tcp_pose: np.ndarray
    tool_pose: np.ndarray
    tcp_offset: np.ndarray
    elbow_pos: np.ndarray
    robot_mode: str
    safety_mode: str
    is_power_on: bool
    collision: bool
    within_safety_limits: bool
    collision_level: int
    collision_stop_type: int


@dataclass
class RetimedTcpStep:
    pose_real: np.ndarray
    pose_sim: np.ndarray
    source_action_index: int


@dataclass
class RetimedTcpChunk:
    start_pose: np.ndarray  # sim frame
    final_pose: np.ndarray  # sim frame
    steps: list[RetimedTcpStep]
    control_dt_s: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    start_pose_real: np.ndarray
    final_pose_real: np.ndarray


@dataclass
class TrackChunkResult:
    ok: bool
    reason: str
    snapshot: RobotSnapshot
    start_pose: np.ndarray  # sim frame
    final_pose: np.ndarray  # sim frame
    sample_count: int
    control_dt_s: float = 0.0
    track_sent: int | None = None
    track_ret: int | None = None
    final_q: np.ndarray | None = None
    tracking_err: float | None = None
    exec_mode: str | None = None
    raw: dict[str, object] | None = None
    start_pose_real: np.ndarray | None = None
    final_pose_real: np.ndarray | None = None


def _companion_cpp_path() -> Path:
    return Path(__file__).resolve().with_name("tcp_control_helper.cpp")


def build_helper(
    sdk_root: str = DEFAULT_SDK_ROOT,
    helper_bin: str = DEFAULT_HELPER_BIN,
) -> Path:
    sdk = Path(sdk_root).resolve()
    src = _companion_cpp_path()
    out = Path(helper_bin).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Skip recompilation if binary exists and is newer than source
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    compile_cmd = [
        "g++",
        "-std=c++17",
        str(src),
        f"-I{sdk / 'include'}",
        f"-L{sdk / 'lib'}",
        f"-Wl,-rpath,{sdk / 'lib'}",
        "-laubo_sdk",
        "-lpthread",
        "-o",
        str(out),
    ]
    subprocess.run(compile_cmd, check=True)
    return out


def _parse_vector(text: str) -> np.ndarray:
    body = text.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return np.array([], dtype=np.float64)
    inner = body[1:-1].strip()
    if not inner:
        return np.array([], dtype=np.float64)
    parts = [p.strip() for p in inner.split(",")]
    return np.array([float(p) for p in parts], dtype=np.float64)


def _parse_helper_output(stdout: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            result[key] = _parse_vector(value)
            continue
        if value in ("true", "false"):
            result[key] = value == "true"
            continue
        if value in ("True", "False"):
            result[key] = value == "True"
            continue
        if re.fullmatch(r"-?\d+", value):
            result[key] = int(value)
            continue
        try:
            result[key] = float(value)
            continue
        except Exception:
            result[key] = value
    return result


_force_sensor = None
_force_sensor_lock = threading.Lock()
_force_sensor_unavailable_reason: str | None = None
_force_guard_state_lock = threading.Lock()
_force_guard_filtered_fz_n: float | None = None
_force_guard_last_ts_s: float | None = None
_force_guard_warning_active = False
_force_guard_hard_active = False


def _reset_force_guard_state() -> None:
    global _force_guard_filtered_fz_n, _force_guard_last_ts_s
    global _force_guard_warning_active, _force_guard_hard_active
    with _force_guard_state_lock:
        _force_guard_filtered_fz_n = None
        _force_guard_last_ts_s = None
        _force_guard_warning_active = False
        _force_guard_hard_active = False


def _stop_force_sensor() -> None:
    global _force_sensor
    sensor = _force_sensor
    if sensor is None:
        return
    try:
        sensor.stop()
    except Exception:
        pass
    _force_sensor = None
    _reset_force_guard_state()


atexit.register(_stop_force_sensor)


def _get_force_sensor():
    global _force_sensor, _force_sensor_unavailable_reason
    if _force_sensor_unavailable_reason is not None:
        return None
    if _force_sensor is not None:
        return _force_sensor
    with _force_sensor_lock:
        if _force_sensor is not None:
            return _force_sensor
        if _force_sensor_unavailable_reason is not None:
            return None
        try:
            from support.force_sensor import ForceSensor

            sensor = ForceSensor()
            sensor.start()
            _force_sensor = sensor
        except Exception as exc:
            _force_sensor_unavailable_reason = str(exc)
            return None
    return _force_sensor


def _get_force_guard_fz_n(timeout_s: float = FORCE_GUARD_READING_TIMEOUT_S) -> float | None:
    sensor = _get_force_sensor()
    if sensor is None:
        return None
    reading = sensor.get()
    if reading is not None:
        return float(reading.fz) * float(FORCE_GUARD_FZ_SIGN)
    remaining_s = max(0.0, float(timeout_s))
    if remaining_s <= 0.0:
        return None
    deadline = time.monotonic() + remaining_s
    while reading is None:
        sleep_s = min(0.01, max(0.0, deadline - time.monotonic()))
        if sleep_s <= 0.0:
            break
        time.sleep(sleep_s)
        reading = sensor.get()
    if reading is None:
        return None
    return float(reading.fz) * float(FORCE_GUARD_FZ_SIGN)


def _force_guard_scale(force_z_n: float | None) -> float | None:
    if force_z_n is None:
        return None
    if force_z_n <= FORCE_GUARD_HARD_FZ_N:
        return 0.0
    if force_z_n >= FORCE_GUARD_SOFT_FZ_N:
        return 1.0
    span = FORCE_GUARD_SOFT_FZ_N - FORCE_GUARD_HARD_FZ_N
    if span <= 1e-9:
        return 0.0
    return float((force_z_n - FORCE_GUARD_HARD_FZ_N) / span)


def _get_force_guard_state(
    timeout_s: float = FORCE_GUARD_READING_TIMEOUT_S,
) -> tuple[float | None, float | None, bool]:
    global _force_guard_filtered_fz_n, _force_guard_last_ts_s
    global _force_guard_warning_active, _force_guard_hard_active

    raw_force_z_n = _get_force_guard_fz_n(timeout_s)
    now = time.monotonic()

    with _force_guard_state_lock:
        if raw_force_z_n is None:
            if (
                _force_guard_filtered_fz_n is None
                or _force_guard_last_ts_s is None
                or (now - _force_guard_last_ts_s) > FORCE_GUARD_READING_HOLD_S
            ):
                return None, None, False
            return (
                _force_guard_filtered_fz_n,
                _force_guard_scale(_force_guard_filtered_fz_n),
                bool(_force_guard_warning_active or _force_guard_hard_active),
            )

        if (
            _force_guard_filtered_fz_n is None
            or _force_guard_last_ts_s is None
            or (now - _force_guard_last_ts_s) > FORCE_GUARD_FILTER_RESET_S
        ):
            filtered = float(raw_force_z_n)
        else:
            dt = max(0.0, now - _force_guard_last_ts_s)
            if FORCE_GUARD_FILTER_TAU_S <= 1e-6:
                filtered = float(raw_force_z_n)
            else:
                alpha = 1.0 - math.exp(-dt / FORCE_GUARD_FILTER_TAU_S)
                filtered = float(
                    _force_guard_filtered_fz_n
                    + alpha * (float(raw_force_z_n) - _force_guard_filtered_fz_n)
                )

        _force_guard_filtered_fz_n = filtered
        _force_guard_last_ts_s = now

        if _force_guard_hard_active:
            if filtered > FORCE_GUARD_HARD_RELEASE_FZ_N:
                _force_guard_hard_active = False
                _force_guard_warning_active = True
        elif filtered <= FORCE_GUARD_HARD_FZ_N:
            _force_guard_hard_active = True
            _force_guard_warning_active = True

        if not _force_guard_hard_active:
            if _force_guard_warning_active:
                if filtered >= FORCE_GUARD_SOFT_RELEASE_FZ_N:
                    _force_guard_warning_active = False
            elif filtered < FORCE_GUARD_SOFT_FZ_N:
                _force_guard_warning_active = True

        return (
            filtered,
            _force_guard_scale(filtered),
            bool(_force_guard_warning_active or _force_guard_hard_active),
        )


def _force_guard_meta(
    force_z_n: float | None,
    scale: float | None,
    adjusted: bool,
    warning_active: bool,
    *,
    blocked_z_m: float | None = None,
    target_fz_n: float | None = None,
) -> dict[str, object]:
    meta = {
        "force_guard_fz_n": force_z_n,
        "force_guard_scale": scale,
        "force_guard_adjusted": bool(adjusted),
        "force_guard_live_mode": bool(warning_active),
    }
    if blocked_z_m is not None:
        meta["force_guard_blocked_z_m"] = float(blocked_z_m)
    if target_fz_n is not None:
        meta["force_guard_target_fz_n"] = float(target_fz_n)
    return meta


def _prepare_force_guard(
    requested_pose: np.ndarray,
    reference_pose: np.ndarray | None,
    *,
    force_z_n: float | None | object = _FORCE_GUARD_UNSET,
    scale: float | None | object = _FORCE_GUARD_UNSET,
    warning_active: bool | object = _FORCE_GUARD_UNSET,
) -> tuple[np.ndarray, np.ndarray | None, float, float | None, float | None, bool]:
    pose = np.asarray(requested_pose, dtype=np.float64).reshape(POSE_DIM).copy()
    if reference_pose is None:
        return pose, None, 0.0, None, None, False
    ref = np.asarray(reference_pose, dtype=np.float64).reshape(POSE_DIM)
    downward_dist = float(ref[2] - pose[2])
    if downward_dist <= 0.0:
        # Preserve the current force-guard state during target hold / upward motion.
        # The collection loop treats a missing force reading during a downward
        # segment hold as a guard failure, so we must not discard a valid live or
        # cached reading just because this particular command no longer decreases z.
        if (
            force_z_n is not _FORCE_GUARD_UNSET
            and scale is not _FORCE_GUARD_UNSET
            and warning_active is not _FORCE_GUARD_UNSET
        ):
            return pose, ref, downward_dist, force_z_n, scale, bool(warning_active)
        warning_val = False if warning_active is _FORCE_GUARD_UNSET else bool(warning_active)
        return pose, ref, downward_dist, None, None, warning_val
    if (
        force_z_n is _FORCE_GUARD_UNSET
        or scale is _FORCE_GUARD_UNSET
        or warning_active is _FORCE_GUARD_UNSET
    ):
        force_val, scale_val, warning_val = _get_force_guard_state()
    else:
        force_val = force_z_n
        scale_val = scale
        warning_val = bool(warning_active)
    return pose, ref, downward_dist, force_val, scale_val, warning_val


def _get_live_tcp_pose_real() -> np.ndarray | None:
    try:
        status = _get_motion_daemon().motion_status()
        tcp_pose = np.asarray(status.get("tcp_pose", []), dtype=np.float64)
        if tcp_pose.size == POSE_DIM:
            return tcp_pose.reshape(POSE_DIM).copy()
    except Exception:
        pass
    try:
        return np.asarray(get_robot_snapshot().tcp_pose, dtype=np.float64).reshape(POSE_DIM).copy()
    except Exception:
        return None


def _apply_servo_force_guard(
    requested_pose: np.ndarray,
    reference_pose: np.ndarray | None,
    *,
    timeout_s: float = 0.0,
) -> tuple[np.ndarray, float | None, float | None, bool, bool]:
    force_z_n, scale, warning_active = _get_force_guard_state(timeout_s=timeout_s)
    return _apply_servo_force_guard_with_scale(
        requested_pose,
        reference_pose,
        force_z_n=force_z_n,
        scale=scale,
        warning_active=warning_active,
    )


def _apply_servo_force_guard_with_scale(
    requested_pose: np.ndarray,
    reference_pose: np.ndarray | None,
    *,
    force_z_n: float | None,
    scale: float | None,
    warning_active: bool,
) -> tuple[np.ndarray, float | None, float | None, bool, bool]:
    pose, ref, downward_dist, force_z_n, scale, warning_active = _prepare_force_guard(
        requested_pose,
        reference_pose,
        force_z_n=force_z_n,
        scale=scale,
        warning_active=warning_active,
    )
    if ref is None or downward_dist <= 0.0 or scale is None or scale >= 1.0:
        return pose, force_z_n, scale, False, warning_active
    pose[2] = float(ref[2] - downward_dist * max(0.0, scale))
    return pose, force_z_n, scale, True, warning_active


def _run_helper(
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    target_pose: np.ndarray | None = None,
    joint_target: np.ndarray | None = None,
    execute: bool = False,
    speed_deg: float = DEFAULT_SPEED_DEG,
    acc_deg: float = DEFAULT_ACC_DEG,
    speed_fraction: float = DEFAULT_SPEED_FRACTION,
) -> dict[str, object]:
    cmd = [
        str(Path(helper_bin).resolve()),
        "--robot-ip",
        robot_ip,
        "--port",
        str(port),
        "--user",
        user,
        "--password",
        password,
        "--speed-deg",
        str(speed_deg),
        "--acc-deg",
        str(acc_deg),
        "--speed-fraction",
        str(speed_fraction),
    ]
    if target_pose is not None:
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(POSE_DIM)
        cmd.extend(["--target-pose", *[f"{v:.12g}" for v in target_pose]])
    if joint_target is not None:
        joint_target = np.asarray(joint_target, dtype=np.float64).reshape(JOINT_DIM)
        cmd.extend(["--joint-target", *[f"{v:.12g}" for v in joint_target]])
    if execute:
        cmd.append("--execute")

    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=HELPER_RUN_TIMEOUT_S)
    parsed = _parse_helper_output(completed.stdout)
    parsed["_returncode"] = completed.returncode
    parsed["_stdout"] = completed.stdout
    parsed["_stderr"] = completed.stderr
    parsed["_cmd"] = cmd
    return parsed


class _DaemonHelper:
    """Persistent subprocess wrapper for the C++ helper in --daemon mode.
    Keeps the RPC connection alive across multiple snapshot requests."""

    def __init__(
        self,
        helper_bin: str = DEFAULT_HELPER_BIN,
        robot_ip: str = DEFAULT_ROBOT_IP,
        port: int = DEFAULT_PORT,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        log_file: str | Path = DEFAULT_HELPER_LOG_FILE,
    ) -> None:
        self._helper_log_file = Path(log_file)
        self._cmd = [
            str(Path(helper_bin).resolve()),
            "--robot-ip", robot_ip,
            "--port", str(port),
            "--user", user,
            "--password", password,
            "--daemon",
            "--log-file", str(self._helper_log_file),
        ]
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stdout_queue: queue.Queue[str | object] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._last_servo_pose_real: np.ndarray | None = None
        self._servo_force_guard_fz_n: float | None = None
        self._servo_force_guard_scale: float | None = None
        self._servo_force_guard_warning_active = False
        self._servo_force_guard_force_live = False
        self._servo_force_guard_live_mode = False
        self._servo_force_guard_block_z_m = 0.0
        self._servo_force_guard_block_v_mps = 0.0
        self._servo_force_guard_last_apply_ts_s: float | None = None
        self._servo_force_target_fz_n: float | None = None

    def _collect_stderr_summary(self) -> str:
        with self._stderr_lock:
            if not self._stderr_lines:
                return ""
            return "".join(self._stderr_lines[-20:]).strip()

    def _start_reader_threads(self, proc: subprocess.Popen) -> None:
        self._stdout_queue = queue.Queue()
        with self._stderr_lock:
            self._stderr_lines.clear()

        def _pump_stdout() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._stdout_queue.put(line)
            finally:
                self._stdout_queue.put(_DAEMON_SENTINEL)

        def _pump_stderr() -> None:
            try:
                assert proc.stderr is not None
                for line in proc.stderr:
                    with self._stderr_lock:
                        self._stderr_lines.append(line)
                        if len(self._stderr_lines) > 200:
                            del self._stderr_lines[:-200]
            except Exception:
                return

        self._stdout_thread = threading.Thread(
            target=_pump_stdout,
            name="tcp-control-helper-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=_pump_stderr,
            name="tcp-control-helper-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _reset_proc_state(self, proc: subprocess.Popen | None = None) -> None:
        if proc is not None:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
        self._proc = None
        self._stdout_queue = None
        self._stdout_thread = None
        self._stderr_thread = None

    def _readline(self, *, timeout_s: float, context: str) -> str:
        if self._stdout_queue is None:
            raise RuntimeError(f"daemon helper is not ready during {context}")
        try:
            item = self._stdout_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(f"daemon helper timed out during {context} after {timeout_s:.1f}s") from exc
        if item is _DAEMON_SENTINEL:
            proc = self._proc
            rc = proc.poll() if proc is not None else None
            stderr_summary = self._collect_stderr_summary()
            self._reset_proc_state(proc)
            detail = f"daemon helper exited during {context}"
            if rc is not None:
                detail += f" (rc={rc})"
            if stderr_summary:
                detail += f": {stderr_summary}"
            raise RuntimeError(detail)
        return str(item)

    def _read_until_end(self, *, timeout_s: float, context: str) -> list[str]:
        deadline = time.monotonic() + timeout_s
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"daemon helper timed out waiting for END during {context}")
            line = self._readline(timeout_s=remaining, context=context)
            stripped = line.strip()
            if stripped == "END":
                return lines
            lines.append(line)

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._helper_log_file.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            **_subprocess_session_kwargs(),
        )
        self._start_reader_threads(self._proc)
        # Wait for DAEMON_READY
        deadline = time.monotonic() + DAEMON_STARTUP_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                proc = self._proc
                stderr_summary = self._collect_stderr_summary()
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=DAEMON_STOP_TIMEOUT_S)
                self._reset_proc_state(proc)
                detail = "daemon helper startup timed out waiting for DAEMON_READY"
                if stderr_summary:
                    detail += f": {stderr_summary}"
                raise TimeoutError(detail)
            line = self._readline(timeout_s=remaining, context="startup")
            if "DAEMON_READY" in line:
                break
        return self._proc

    def _send_cmd(self, cmd: str) -> dict[str, object]:
        """Send a command, read lines until 'END', return parsed dict."""
        proc = self._ensure_started()
        try:
            assert proc.stdin is not None
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except Exception:
            self._reset_proc_state(proc)
            raise
        lines = self._read_until_end(timeout_s=DAEMON_COMMAND_TIMEOUT_S, context=cmd.split()[0])
        return _parse_helper_output("\n".join(lines))

    def snapshot_raw(self) -> str:
        """Send 'snapshot' command and return all output lines until 'END'."""
        with self._lock:
            proc = self._ensure_started()
            try:
                assert proc.stdin is not None
                proc.stdin.write("snapshot\n")
                proc.stdin.flush()
            except Exception:
                self._reset_proc_state(proc)
                raise
            lines = self._read_until_end(timeout_s=DAEMON_COMMAND_TIMEOUT_S, context="snapshot")
            return "\n".join(lines)

    def servo_start(self, track_time_s: float = DEFAULT_TRACK_CONTROL_DT_S) -> dict[str, object]:
        """Enter servo mode on the daemon."""
        with self._lock:
            self._last_servo_pose_real = None
            self._servo_force_guard_fz_n = None
            self._servo_force_guard_scale = None
            self._servo_force_guard_warning_active = False
            self._servo_force_guard_force_live = False
            self._servo_force_guard_live_mode = False
            self._servo_force_guard_block_z_m = 0.0
            self._servo_force_guard_block_v_mps = 0.0
            self._servo_force_guard_last_apply_ts_s = None
            self._servo_force_target_fz_n = None
            return self._send_cmd(f"servo_start {track_time_s:.12g}")

    def servo_begin_chunk(
        self,
        pose6: np.ndarray | None = None,
        *,
        force_live_mode: bool = False,
    ) -> dict[str, object]:
        """Cache chunk-start pose and force. Live force polling only starts inside warning region."""
        with self._lock:
            if pose6 is not None:
                self._last_servo_pose_real = np.asarray(pose6, dtype=np.float64).reshape(POSE_DIM).copy()
            sync_resp = self._send_cmd("servo_sync_seed")
            if int(sync_resp.get("servo_sync_seed_ret", -1)) != 0:
                raise RuntimeError(f"servo_sync_seed failed: {sync_resp}")
            force_z_n, scale, warning_active = _get_force_guard_state()
            self._servo_force_guard_fz_n = force_z_n
            self._servo_force_guard_scale = scale
            self._servo_force_guard_warning_active = warning_active
            self._servo_force_guard_force_live = bool(force_live_mode)
            self._servo_force_guard_live_mode = bool(force_live_mode or warning_active)
            self._servo_force_target_fz_n = (
                max(float(force_z_n), float(FORCE_GUARD_SOFT_FZ_N))
                if force_z_n is not None
                else float(FORCE_GUARD_SOFT_FZ_N)
            )
            return {
                **sync_resp,
                "force_guard_fz_n": force_z_n,
                "force_guard_scale": scale,
                "force_guard_warning_active": warning_active,
                "force_guard_force_live": self._servo_force_guard_force_live,
                "force_guard_live_mode": self._servo_force_guard_live_mode,
            }

    def _current_servo_reference_pose(self) -> np.ndarray | None:
        if self._last_servo_pose_real is not None:
            return self._last_servo_pose_real
        return _get_live_tcp_pose_real()

    def get_servo_reference_pose(self) -> np.ndarray | None:
        with self._lock:
            if self._last_servo_pose_real is None:
                return None
            return np.asarray(self._last_servo_pose_real, dtype=np.float64).reshape(POSE_DIM).copy()

    def _servo_force_guard_dt_s(self) -> float:
        now = time.monotonic()
        if self._servo_force_guard_last_apply_ts_s is None:
            dt = DEFAULT_TRACK_CONTROL_DT_S
        else:
            dt = now - self._servo_force_guard_last_apply_ts_s
        self._servo_force_guard_last_apply_ts_s = now
        return float(np.clip(dt, FORCE_ADMITTANCE_MIN_DT_S, FORCE_ADMITTANCE_MAX_DT_S))

    def _apply_servo_z_admittance(
        self,
        requested_pose: np.ndarray,
        reference_pose: np.ndarray | None,
        *,
        timeout_s: float = 0.0,
        force_z_n: float | None | object = _FORCE_GUARD_UNSET,
        scale: float | None | object = _FORCE_GUARD_UNSET,
        warning_active: bool | object = _FORCE_GUARD_UNSET,
    ) -> tuple[np.ndarray, float | None, float | None, bool, bool]:
        if (
            force_z_n is _FORCE_GUARD_UNSET
            or scale is _FORCE_GUARD_UNSET
            or warning_active is _FORCE_GUARD_UNSET
        ):
            force_z_n, scale, warning_active = _get_force_guard_state(timeout_s=timeout_s)
        pose, ref, downward_dist, force_z_n, scale, warning_active = _prepare_force_guard(
            requested_pose,
            reference_pose,
            force_z_n=force_z_n,
            scale=scale,
            warning_active=warning_active,
        )

        dt_s = self._servo_force_guard_dt_s()
        block_z_m = float(self._servo_force_guard_block_z_m)
        block_v_mps = float(self._servo_force_guard_block_v_mps)
        target_fz_n = float(
            FORCE_GUARD_SOFT_FZ_N if self._servo_force_target_fz_n is None else self._servo_force_target_fz_n
        )
        force_error_n = (
            0.0
            if force_z_n is None
            else max(
                0.0,
                float(target_fz_n) - float(force_z_n) - float(FORCE_ADMITTANCE_FORCE_DEADBAND_N),
            )
        )

        # Second-order outer-loop admittance on the blocked downward distance.
        acc_mps2 = (
            force_error_n
            - float(FORCE_ADMITTANCE_DAMPING_N_S_PER_M) * block_v_mps
            - float(FORCE_ADMITTANCE_STIFFNESS_N_PER_M) * block_z_m
        ) / float(FORCE_ADMITTANCE_VIRTUAL_MASS_KG)
        block_v_mps += float(acc_mps2) * dt_s
        block_z_m += block_v_mps * dt_s

        if block_z_m < 0.0:
            block_z_m = 0.0
            if block_v_mps < 0.0:
                block_v_mps = 0.0
        elif block_z_m > float(FORCE_ADMITTANCE_MAX_BLOCK_Z_M):
            block_z_m = float(FORCE_ADMITTANCE_MAX_BLOCK_Z_M)
            if block_v_mps > 0.0:
                block_v_mps = 0.0

        self._servo_force_guard_block_z_m = block_z_m
        self._servo_force_guard_block_v_mps = block_v_mps

        hard_active = force_z_n is not None and float(force_z_n) <= float(FORCE_GUARD_HARD_FZ_N)
        effective_block_z_m = 0.0
        if downward_dist > 0.0:
            effective_block_z_m = min(float(block_z_m), float(downward_dist))
            if hard_active:
                effective_block_z_m = float(downward_dist)
            if effective_block_z_m > 0.0:
                pose[2] = float(ref[2] - max(0.0, float(downward_dist) - effective_block_z_m))

        report_scale = scale
        if downward_dist > 1e-9:
            report_scale = max(0.0, min(1.0, (float(downward_dist) - effective_block_z_m) / float(downward_dist)))
        elif hard_active:
            report_scale = 0.0

        adjusted = bool(effective_block_z_m > 1e-9 or hard_active)
        return pose, force_z_n, report_scale, adjusted, bool(warning_active)

    def servo_pose(self, pose6: np.ndarray) -> dict[str, object]:
        """Send a single pose in servo mode. Returns parsed response.
        Used by data collection for frame-by-frame execution + capture."""
        requested = np.asarray(pose6, dtype=np.float64).reshape(POSE_DIM)
        reference = self._current_servo_reference_pose()
        guarded, force_z_n, scale, adjusted, warning_active = self._apply_servo_z_admittance(
            requested,
            reference,
            timeout_s=0.0,
        )
        cmd = "servo_pose " + " ".join(f"{v:.12g}" for v in guarded)
        with self._lock:
            resp = self._send_cmd(cmd)
        resp.update(
            _force_guard_meta(
                force_z_n,
                scale,
                adjusted,
                warning_active,
                blocked_z_m=self._servo_force_guard_block_z_m,
                target_fz_n=self._servo_force_target_fz_n,
            )
        )
        resp["requested_pose_real"] = requested.astype(np.float64).reshape(POSE_DIM).tolist()
        resp["guarded_pose_real"] = guarded.astype(np.float64).reshape(POSE_DIM).tolist()
        self._last_servo_pose_real = guarded
        if force_z_n is not None or scale is not None:
            self._servo_force_guard_fz_n = force_z_n
            self._servo_force_guard_scale = scale
        self._servo_force_guard_warning_active = bool(warning_active)
        self._servo_force_guard_live_mode = bool(self._servo_force_guard_force_live or warning_active)
        return resp

    def servo_chunk(self, poses_real: list[np.ndarray]) -> dict[str, object]:
        """Send a batch of poses to execute with C++-side timing.
        All poses are sent at once; C++ handles the precise sleep intervals."""
        guarded_poses: list[np.ndarray] = []
        force_z_n: float | None = None
        scale: float | None = None
        adjusted = False
        warning_active = self._servo_force_guard_warning_active
        reference = self._current_servo_reference_pose()
        for pose in poses_real:
            guarded, step_force_z_n, step_scale, step_adjusted, step_warning_active = self._apply_servo_z_admittance(
                pose,
                reference,
                timeout_s=0.0,
            )
            guarded_poses.append(guarded)
            if step_force_z_n is not None:
                force_z_n = step_force_z_n
                scale = step_scale
            adjusted = adjusted or bool(step_adjusted)
            warning_active = bool(step_warning_active)
            reference = guarded
        with self._lock:
            proc = self._ensure_started()
            # Header: servo_chunk N
            try:
                assert proc.stdin is not None
                proc.stdin.write(f"servo_chunk {len(guarded_poses)}\n")
                # Body: one pose per line
                for p in guarded_poses:
                    vals = np.asarray(p, dtype=np.float64).reshape(POSE_DIM)
                    proc.stdin.write(" ".join(f"{v:.12g}" for v in vals) + "\n")
                proc.stdin.flush()
            except Exception:
                self._reset_proc_state(proc)
                raise
            lines = self._read_until_end(timeout_s=DAEMON_COMMAND_TIMEOUT_S, context="servo_chunk")
        resp = _parse_helper_output("\n".join(lines))
        resp.update(
            _force_guard_meta(
                force_z_n,
                scale,
                adjusted,
                warning_active,
                blocked_z_m=self._servo_force_guard_block_z_m,
                target_fz_n=self._servo_force_target_fz_n,
            )
        )
        if guarded_poses:
            self._last_servo_pose_real = guarded_poses[-1]
        if force_z_n is not None or scale is not None:
            self._servo_force_guard_fz_n = force_z_n
            self._servo_force_guard_scale = scale
        self._servo_force_guard_warning_active = bool(warning_active)
        self._servo_force_guard_live_mode = bool(self._servo_force_guard_force_live or warning_active)
        return resp

    def motion_status(self) -> dict[str, object]:
        """Query the current controller execution state."""
        with self._lock:
            return self._send_cmd("motion_status")

    def stop_motion(self, *, quick: bool = True, all_tasks: bool = True) -> dict[str, object]:
        """Request the controller to stop the current queued motion and wait for steady state."""
        with self._lock:
            return self._send_cmd(f"stop_motion {1 if quick else 0} {1 if all_tasks else 0}")

    def servo_stop(self) -> dict[str, object]:
        """Exit servo mode."""
        with self._lock:
            resp = self._send_cmd("servo_stop")
            self._last_servo_pose_real = None
            self._servo_force_guard_fz_n = None
            self._servo_force_guard_scale = None
            self._servo_force_guard_warning_active = False
            self._servo_force_guard_force_live = False
            self._servo_force_guard_live_mode = False
            self._servo_force_guard_block_z_m = 0.0
            self._servo_force_guard_block_v_mps = 0.0
            self._servo_force_guard_last_apply_ts_s = None
            self._servo_force_target_fz_n = None
            return resp

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write("quit\n")
                        proc.stdin.flush()
                    proc.wait(timeout=DAEMON_STOP_TIMEOUT_S)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=DAEMON_STOP_TIMEOUT_S)
                    except Exception:
                        pass
            self._reset_proc_state(proc)


# Two independent daemon instances:
# - _snapshot_daemon: used by get_robot_snapshot() for state reads
# - _motion_daemon: used by higher-level control for servo / motion-status commands
_snapshot_daemon: _DaemonHelper | None = None
_motion_daemon: _DaemonHelper | None = None
_daemon_lock = threading.Lock()


def _get_snapshot_daemon() -> _DaemonHelper:
    global _snapshot_daemon
    if _snapshot_daemon is None:
        with _daemon_lock:
            if _snapshot_daemon is None:
                _snapshot_daemon = _DaemonHelper(log_file="")
    return _snapshot_daemon


def _get_motion_daemon() -> _DaemonHelper:
    global _motion_daemon
    if _motion_daemon is None:
        with _daemon_lock:
            if _motion_daemon is None:
                _motion_daemon = _DaemonHelper(log_file=DEFAULT_HELPER_LOG_FILE)
    return _motion_daemon


def _get_servo_daemon() -> _DaemonHelper:
    """Backward-compatible alias for older call sites."""
    return _get_motion_daemon()


def _stop_all_daemons() -> None:
    global _snapshot_daemon, _motion_daemon
    with _daemon_lock:
        daemons = (_snapshot_daemon, _motion_daemon)
        _snapshot_daemon = None
        _motion_daemon = None
    for daemon in daemons:
        if daemon is not None:
            try:
                daemon.stop()
            except Exception:
                pass


atexit.register(_stop_all_daemons)


def get_robot_snapshot(
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    use_daemon: bool = True,
) -> RobotSnapshot:
    if use_daemon:
        try:
            raw_text = _get_snapshot_daemon().snapshot_raw()
            raw = _parse_helper_output(raw_text)
        except Exception:
            # Fallback to one-shot if daemon fails
            raw = _run_helper(
                helper_bin=helper_bin,
                robot_ip=robot_ip,
                port=port,
                user=user,
                password=password,
            )
            if int(raw["_returncode"]) != 0:
                raise RuntimeError(f"snapshot helper rc={raw['_returncode']}: {raw.get('_stdout', '')[:200]}")
    else:
        raw = _run_helper(
            helper_bin=helper_bin,
            robot_ip=robot_ip,
            port=port,
            user=user,
            password=password,
        )
        if int(raw["_returncode"]) != 0:
            raise RuntimeError(f"snapshot helper rc={raw['_returncode']}: {raw.get('_stdout', '')[:200]}")

    return RobotSnapshot(
        joint_q=np.asarray(raw.get("joint_q_rad", []), dtype=np.float64),
        tcp_pose=np.asarray(raw.get("tcp_pose", []), dtype=np.float64),
        tool_pose=np.asarray(raw.get("tool_pose", []), dtype=np.float64),
        tcp_offset=np.asarray(raw.get("tcp_offset", []), dtype=np.float64),
        elbow_pos=np.asarray(raw.get("elbow_pos", []), dtype=np.float64),
        robot_mode=str(raw.get("robot_mode", "")),
        safety_mode=str(raw.get("safety_mode", "")),
        is_power_on=bool(raw.get("is_power_on", False)),
        collision=bool(raw.get("collision", False)),
        within_safety_limits=bool(raw.get("within_safety_limits", False)),
        collision_level=int(raw.get("collision_level", -1)),
        collision_stop_type=int(raw.get("collision_stop_type", -1)),
    )


def integrate_delta_tcp_pose(
    pose6_zyx: np.ndarray,
    delta6: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(pose6_zyx, dtype=np.float64).reshape(POSE_DIM)
    delta = np.asarray(delta6, dtype=np.float64).reshape(OPENPI_DELTA_DIM).copy()
    out = pose.copy()
    out[:3] = pose[:3] + delta[:3]
    out[3:] = wrap_euler_zyx(pose[3:] + delta[3:])
    return out


def solve_target_joint_q(
    target_pose: np.ndarray,
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
) -> tuple[RobotSnapshot, np.ndarray | None, int | None, dict[str, object]]:
    raw = _run_helper(
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
        target_pose=target_pose,
        execute=False,
    )
    if int(raw["_returncode"]) not in (0, 20):
        raise RuntimeError(f"IK helper rc={raw['_returncode']}: {raw.get('_stdout', '')[:200]}")

    snapshot = RobotSnapshot(
        joint_q=np.asarray(raw.get("joint_q_rad", []), dtype=np.float64),
        tcp_pose=np.asarray(raw.get("tcp_pose", []), dtype=np.float64),
        tool_pose=np.asarray(raw.get("tool_pose", []), dtype=np.float64),
        tcp_offset=np.asarray(raw.get("tcp_offset", []), dtype=np.float64),
        elbow_pos=np.asarray(raw.get("elbow_pos", []), dtype=np.float64),
        robot_mode=str(raw.get("robot_mode", "")),
        safety_mode=str(raw.get("safety_mode", "")),
        is_power_on=bool(raw.get("is_power_on", False)),
        collision=bool(raw.get("collision", False)),
        within_safety_limits=bool(raw.get("within_safety_limits", False)),
        collision_level=int(raw.get("collision_level", -1)),
        collision_stop_type=int(raw.get("collision_stop_type", -1)),
    )
    target_q = np.asarray(raw.get("target_q_rad", []), dtype=np.float64)
    if target_q.size == 0:
        target_q = None
    ik_ret = raw.get("ik_ret")
    ik_ret = int(ik_ret) if ik_ret is not None else None
    return snapshot, target_q, ik_ret, raw


def execute_joint_target(
    joint_target: np.ndarray,
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    speed_deg: float = DEFAULT_SPEED_DEG,
    acc_deg: float = DEFAULT_ACC_DEG,
    speed_fraction: float = DEFAULT_SPEED_FRACTION,
) -> dict[str, object]:
    return _run_helper(
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
        joint_target=joint_target,
        execute=True,
        speed_deg=speed_deg,
        acc_deg=acc_deg,
        speed_fraction=speed_fraction,
    )


def stop_robot_motion(
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
) -> dict[str, object]:
    return _run_helper(
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
        joint_target=np.asarray(get_robot_snapshot(
            helper_bin=helper_bin,
            robot_ip=robot_ip,
            port=port,
            user=user,
            password=password,
        ).joint_q, dtype=np.float64),
        execute=True,
        speed_deg=0.0,
        acc_deg=0.0,
        speed_fraction=1.0,
    )


def retime_tcp_action_chunk(
    delta_actions: np.ndarray,
    *,
    start_pose_sim: np.ndarray,
    control_dt_s: float = DEFAULT_TRACK_CONTROL_DT_S,
    max_linear_speed_mps: float = DEFAULT_TCP_LINEAR_SPEED_MPS,
    max_angular_speed_radps: float = DEFAULT_TCP_ANGULAR_SPEED_RADPS,
    z_min: float = DEFAULT_Z_MIN_M,
    constant_linear_speed: bool = False,
) -> RetimedTcpChunk:
    """Retime a chunk of TCP delta actions into fine-grained servo steps.

    All integration happens in the **sim frame** (what the policy learned).
    Each step is then converted to the **real frame** for the robot SDK.
    """
    actions = np.asarray(delta_actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, OPENPI_DELTA_DIM)
    if actions.shape[-1] != OPENPI_DELTA_DIM:
        raise ValueError(f"delta_actions must have shape (N, {OPENPI_DELTA_DIM})")
    if control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be > 0")

    current_sim = np.asarray(start_pose_sim, dtype=np.float64).reshape(POSE_DIM).copy()
    start_sim = current_sim.copy()
    start_real = sim_pose_to_real(start_sim)
    current_real = start_real.copy()
    steps: list[RetimedTcpStep] = []

    if constant_linear_speed and np.isfinite(max_linear_speed_mps) and max_linear_speed_mps > 0.0:
        linear_step = float(max_linear_speed_mps) * float(control_dt_s)
        angular_step = (
            float(max_angular_speed_radps) * float(control_dt_s)
            if np.isfinite(max_angular_speed_radps) and max_angular_speed_radps > 0.0
            else float("inf")
        )
        carry_linear_dist = 0.0
        prev_linear_dir: np.ndarray | None = None

        def _interp_pose(start_pose: np.ndarray, end_pose: np.ndarray, frac: float) -> np.ndarray:
            frac = float(np.clip(frac, 0.0, 1.0))
            out = np.asarray(start_pose, dtype=np.float64).reshape(POSE_DIM).copy()
            end_pose = np.asarray(end_pose, dtype=np.float64).reshape(POSE_DIM)
            out[:3] = out[:3] + (end_pose[:3] - out[:3]) * frac
            angle_delta = np.array(
                [
                    math.atan2(math.sin(end_pose[axis] - out[axis]), math.cos(end_pose[axis] - out[axis]))
                    for axis in range(3, 6)
                ],
                dtype=np.float64,
            )
            out[3:] = wrap_euler_zyx(out[3:] + angle_delta * frac)
            return out

        last_emitted_sim = current_sim.copy()
        for idx, raw_delta in enumerate(actions):
            segment_start = current_sim.copy()
            segment_end = integrate_delta_tcp_pose(segment_start, raw_delta)
            segment_linear_dist = float(np.linalg.norm(segment_end[:3] - segment_start[:3]))
            angle_delta = np.array(
                [
                    math.atan2(
                        math.sin(segment_end[axis] - segment_start[axis]),
                        math.cos(segment_end[axis] - segment_start[axis]),
                    )
                    for axis in range(3, 6)
                ],
                dtype=np.float64,
            )
            segment_angular_dist = float(np.linalg.norm(angle_delta))
            segment_linear_dir: np.ndarray | None = None
            if segment_linear_dist > 1e-12:
                segment_linear_dir = (segment_end[:3] - segment_start[:3]) / segment_linear_dist

            start_speed_scale = 1.0
            if prev_linear_dir is not None and segment_linear_dir is not None:
                cos_turn = float(np.clip(np.dot(prev_linear_dir, segment_linear_dir), -1.0, 1.0))
                turn_mag = float(np.sqrt(max(0.0, 0.5 * (1.0 - cos_turn))))
                start_speed_scale = float(
                    1.0 - turn_mag * (1.0 - float(CONSTANT_SPEED_CORNER_MIN_SCALE))
                )
                start_speed_scale = float(
                    np.clip(start_speed_scale, float(CONSTANT_SPEED_CORNER_MIN_SCALE), 1.0)
                )

            linear_fracs: list[float] = []
            if linear_step > 1e-9 and segment_linear_dist > 1e-12:
                consumed = 0.0
                emitted_in_segment = 0
                while True:
                    if emitted_in_segment < int(CONSTANT_SPEED_CORNER_RAMP_STEPS):
                        ramp_alpha = float(
                            emitted_in_segment / max(1, int(CONSTANT_SPEED_CORNER_RAMP_STEPS) - 1)
                        )
                        step_scale = float(
                            start_speed_scale + (1.0 - start_speed_scale) * ramp_alpha
                        )
                    else:
                        step_scale = 1.0
                    target_step = float(linear_step * step_scale)
                    needed = float(target_step - carry_linear_dist) if carry_linear_dist > 1e-12 else target_step
                    if (segment_linear_dist - consumed) + 1e-12 < needed:
                        break
                    consumed += needed
                    linear_fracs.append(float(consumed / segment_linear_dist))
                    carry_linear_dist = 0.0
                    emitted_in_segment += 1
                carry_linear_dist = float(max(0.0, carry_linear_dist + segment_linear_dist - consumed))
                if carry_linear_dist >= linear_step:
                    carry_linear_dist = float(carry_linear_dist % linear_step)
            elif segment_linear_dist <= 1e-12:
                linear_fracs = []
            else:
                linear_fracs = [1.0]
                carry_linear_dist = 0.0

            angular_fracs: list[float] = []
            if np.isfinite(angular_step) and angular_step > 1e-9 and segment_angular_dist > 1e-12:
                angular_count = max(1, int(math.ceil(segment_angular_dist / angular_step)))
                angular_fracs = [float(i / angular_count) for i in range(1, angular_count + 1)]

            fracs = sorted({float(f) for f in (linear_fracs + angular_fracs) if 0.0 < float(f) <= 1.0})
            if not fracs:
                current_sim = segment_end.copy()
                continue

            for frac in fracs:
                current_sim = _interp_pose(segment_start, segment_end, frac)
                current_real = sim_pose_to_real(current_sim)
                if current_real[2] < z_min:
                    current_real[2] = z_min
                current_sim = real_pose_to_sim(current_real)
                last_emitted_sim = current_sim.copy()
                steps.append(
                    RetimedTcpStep(
                        pose_real=current_real.copy(),
                        pose_sim=current_sim.copy(),
                        source_action_index=int(idx),
                    )
                )

            current_sim = segment_end.copy()
            if segment_linear_dir is not None:
                prev_linear_dir = segment_linear_dir.copy()

        final_target_sim = np.asarray(start_pose_sim, dtype=np.float64).reshape(POSE_DIM).copy()
        for raw_delta in actions:
            final_target_sim = integrate_delta_tcp_pose(final_target_sim, raw_delta)
        final_target_real = sim_pose_to_real(final_target_sim)
        if final_target_real[2] < z_min:
            final_target_real[2] = z_min
        final_target_sim = real_pose_to_sim(final_target_real)

        if not steps or not np.allclose(steps[-1].pose_sim, final_target_sim, atol=1e-9, rtol=0.0):
            current_sim = final_target_sim.copy()
            current_real = final_target_real.copy()
            steps.append(
                RetimedTcpStep(
                    pose_real=current_real.copy(),
                    pose_sim=current_sim.copy(),
                    source_action_index=int(max(0, len(actions) - 1)),
                )
            )
        else:
            current_sim = steps[-1].pose_sim.copy()
            current_real = steps[-1].pose_real.copy()

        return RetimedTcpChunk(
            start_pose=start_sim,
            final_pose=current_sim.copy(),
            steps=steps,
            control_dt_s=float(control_dt_s),
            max_linear_speed_mps=float(max_linear_speed_mps),
            max_angular_speed_radps=float(max_angular_speed_radps),
            start_pose_real=start_real,
            final_pose_real=current_real.copy(),
        )

    for idx, raw_delta in enumerate(actions):
        delta = np.asarray(raw_delta, dtype=np.float64).reshape(OPENPI_DELTA_DIM).copy()

        linear_dist = float(np.linalg.norm(delta[:3]))
        angular_dist = float(np.linalg.norm(delta[3:]))
        linear_time = linear_dist / float(max_linear_speed_mps) if max_linear_speed_mps > 0.0 else 0.0
        angular_time = angular_dist / float(max_angular_speed_radps) if max_angular_speed_radps > 0.0 else 0.0
        required_time = max(linear_time, angular_time, float(control_dt_s))
        step_count = max(1, int(math.ceil(required_time / float(control_dt_s))))
        step_delta = delta / float(step_count)

        for _ in range(step_count):
            # Integrate in sim frame
            current_sim = integrate_delta_tcp_pose(current_sim, step_delta)
            # Convert to real frame for robot SDK
            current_real = sim_pose_to_real(current_sim)
            # Clip TCP z in real frame to minimum safe height
            if current_real[2] < z_min:
                current_real[2] = z_min
            # Reflect the clipped real pose back to sim frame
            current_sim = real_pose_to_sim(current_real)
            steps.append(RetimedTcpStep(
                pose_real=current_real.copy(),
                pose_sim=current_sim.copy(),
                source_action_index=int(idx),
            ))

    return RetimedTcpChunk(
        start_pose=start_sim,
        final_pose=current_sim.copy(),
        steps=steps,
        control_dt_s=float(control_dt_s),
        max_linear_speed_mps=float(max_linear_speed_mps),
        max_angular_speed_radps=float(max_angular_speed_radps),
        start_pose_real=start_real,
        final_pose_real=current_real.copy(),
    )


@dataclass
class IntegratedTarget:
    """Lightweight result of integrating delta actions to a single final pose."""
    start_pose_sim: np.ndarray
    start_pose_real: np.ndarray
    final_pose_sim: np.ndarray
    final_pose_real: np.ndarray
    n_actions: int


def integrate_delta_actions_to_target(
    delta_actions: np.ndarray,
    *,
    start_pose_sim: np.ndarray,
) -> IntegratedTarget:
    """Integrate all delta actions and return only the final target pose."""
    actions = np.asarray(delta_actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, OPENPI_DELTA_DIM)
    if actions.shape[-1] != OPENPI_DELTA_DIM:
        raise ValueError(f"delta_actions must have shape (N, {OPENPI_DELTA_DIM})")

    current_sim = np.asarray(start_pose_sim, dtype=np.float64).reshape(POSE_DIM).copy()
    start_sim = current_sim.copy()
    start_real = sim_pose_to_real(start_sim)

    for raw_delta in actions:
        current_sim = integrate_delta_tcp_pose(current_sim, raw_delta)

    current_real = sim_pose_to_real(current_sim)

    return IntegratedTarget(
        start_pose_sim=start_sim,
        start_pose_real=start_real,
        final_pose_sim=current_sim.copy(),
        final_pose_real=current_real.copy(),
        n_actions=int(actions.shape[0]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug entrypoint for the TCP delta control module.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--pose-frame", type=str, choices=("sim", "real"), default="sim")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_alignment_mode(args.pose_frame)
    helper = build_helper()
    if args.compile_only:
        print(f"HELPER_READY={helper}")
        print(f"POSE_FRAME={get_alignment_mode()}")
        return 0
    if args.snapshot:
        snap = get_robot_snapshot(helper_bin=str(helper))
        print(snap)
        return 0
    print(f"HELPER_READY={helper}")
    return 0


__all__ = [
    "RESET_ERR_M",
    "DEFAULT_TRACK_CONTROL_DT_S",
    "DEFAULT_TCP_LINEAR_SPEED_MPS",
    "DEFAULT_TCP_ANGULAR_SPEED_RADPS",
    "RobotSnapshot",
    "RetimedTcpStep",
    "RetimedTcpChunk",
    "TrackChunkResult",
    "build_helper",
    "_get_motion_daemon",
    "get_robot_snapshot",
    "integrate_delta_tcp_pose",
    "solve_target_joint_q",
    "execute_joint_target",
    "stop_robot_motion",
    "retime_tcp_action_chunk",
    "integrate_delta_actions_to_target",
]


if __name__ == "__main__":
    raise SystemExit(main())

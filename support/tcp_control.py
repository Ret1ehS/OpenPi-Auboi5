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
    is_alignment_ready,
    real_pose_to_sim,
    set_alignment_mode,
    set_runtime_alignment,
    sim_pose_to_real,
    wrap_euler_zyx,
)


DEFAULT_SDK_ROOT = "/home/orin/openpi/aubo_sdk/aubo_sdk-0.24.1-rc.3-Linux_aarch64+318754d"
DEFAULT_HELPER_BIN = "/home/orin/openpi/scripts/.build/tcp_control_helper"
DEFAULT_HELPER_LOG_DIR = Path(__file__).resolve().parent.parent / "log"
DEFAULT_HELPER_LOG_FILE = DEFAULT_HELPER_LOG_DIR / "tcp_control_helper.log"

DEFAULT_ROBOT_IP = "192.168.1.100"
DEFAULT_PORT = 30004
DEFAULT_USER = "aubo"
DEFAULT_PASSWORD = "123456"

DEFAULT_SPEED_DEG = 10.0
DEFAULT_ACC_DEG = 20.0
DEFAULT_SPEED_FRACTION = 1.0
DEFAULT_TRACK_CONTROL_DT_S = 0.01
DEFAULT_TRACK_SMOOTH_SCALE = 0.5
DEFAULT_TRACK_DELAY_SCALE = 1.0
DEFAULT_TCP_LINEAR_SPEED_MPS = 0.05
DEFAULT_TCP_ANGULAR_SPEED_RADPS = 0.60
DEFAULT_MOVE_LINE_BLEND_RADIUS_M = 0.01

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
_FORCE_GUARD_UNSET = object()

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
class TcpDeltaResult:
    ok: bool
    reason: str
    snapshot: RobotSnapshot
    start_pose: np.ndarray  # sim frame
    target_pose: np.ndarray  # sim frame
    target_q: np.ndarray | None
    ik_ret: int | None
    move_ret: int | None = None
    wait_ret: int | None = None
    final_q: np.ndarray | None = None
    tracking_err: float | None = None
    start_pose_real: np.ndarray | None = None
    target_pose_real: np.ndarray | None = None


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
) -> dict[str, object]:
    return {
        "force_guard_fz_n": force_z_n,
        "force_guard_scale": scale,
        "force_guard_adjusted": bool(adjusted),
        "force_guard_live_mode": bool(warning_active),
    }


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


def _apply_movel_force_guard(
    requested_pose: np.ndarray,
    reference_pose: np.ndarray | None,
    *,
    speed_frac: float,
    speed_mps: float | None,
) -> tuple[np.ndarray, float, float | None, float | None, float | None, bool, bool]:
    adj_speed_frac = float(speed_frac)
    adj_speed_mps = None if speed_mps is None else float(speed_mps)
    pose, ref, downward_dist, force_z_n, scale, warning_active = _prepare_force_guard(
        requested_pose,
        reference_pose,
    )
    if ref is None or downward_dist <= 0.0 or scale is None or scale >= 1.0:
        return pose, adj_speed_frac, adj_speed_mps, force_z_n, scale, False, warning_active
    if scale <= 0.0:
        pose[2] = float(ref[2])
    elif adj_speed_mps is not None:
        adj_speed_mps = max(1e-4, adj_speed_mps * max(scale, 0.0))
    elif adj_speed_frac > 0.0:
        adj_speed_frac = max(1e-4, adj_speed_frac * max(scale, 0.0))
    return pose, adj_speed_frac, adj_speed_mps, force_z_n, scale, True, warning_active


def _run_helper(
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    target_pose: np.ndarray | None = None,
    joint_target: np.ndarray | None = None,
    track_pose_file: str | None = None,
    track_time_s: float | None = None,
    smooth_scale: float | None = None,
    delay_scale: float | None = None,
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
    if track_pose_file is not None:
        cmd.extend(["--track-pose-file", str(track_pose_file)])
    if track_time_s is not None:
        cmd.extend(["--track-time-s", f"{float(track_time_s):.12g}"])
    if smooth_scale is not None:
        cmd.extend(["--smooth-scale", f"{float(smooth_scale):.12g}"])
    if delay_scale is not None:
        cmd.extend(["--delay-scale", f"{float(delay_scale):.12g}"])
    if execute:
        cmd.append("--execute")

    completed = subprocess.run(cmd, capture_output=True, text=True)
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
    ) -> None:
        self._helper_log_file = Path(DEFAULT_HELPER_LOG_FILE)
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
        self._last_servo_pose_real: np.ndarray | None = None
        self._servo_force_guard_fz_n: float | None = None
        self._servo_force_guard_scale: float | None = None
        self._servo_force_guard_warning_active = False
        self._servo_force_guard_force_live = False
        self._servo_force_guard_live_mode = False

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
        )
        # Wait for DAEMON_READY
        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.wait()
                raise RuntimeError(f"daemon helper exited during startup (rc={rc})")
            if "DAEMON_READY" in line:
                break
        return self._proc

    def _send_cmd(self, cmd: str) -> dict[str, object]:
        """Send a command, read lines until 'END', return parsed dict."""
        proc = self._ensure_started()
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                self._proc = None
                raise RuntimeError(f"daemon helper died during '{cmd.split()[0]}'")
            stripped = line.strip()
            if stripped == "END":
                break
            lines.append(line)
        return _parse_helper_output("\n".join(lines))

    def snapshot_raw(self) -> str:
        """Send 'snapshot' command and return all output lines until 'END'."""
        with self._lock:
            proc = self._ensure_started()
            proc.stdin.write("snapshot\n")
            proc.stdin.flush()
            lines = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    self._proc = None
                    raise RuntimeError("daemon helper died during snapshot")
                stripped = line.strip()
                if stripped == "END":
                    break
                lines.append(line)
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

    def servo_pose(self, pose6: np.ndarray) -> dict[str, object]:
        """Send a single pose in servo mode. Returns parsed response.
        Used by data collection for frame-by-frame execution + capture."""
        requested = np.asarray(pose6, dtype=np.float64).reshape(POSE_DIM)
        reference = self._current_servo_reference_pose()
        if self._servo_force_guard_force_live or self._servo_force_guard_live_mode:
            guarded, force_z_n, scale, adjusted, warning_active = _apply_servo_force_guard(
                requested,
                reference,
                timeout_s=0.0,
            )
        else:
            guarded, force_z_n, scale, adjusted, warning_active = _apply_servo_force_guard_with_scale(
                requested,
                reference,
                force_z_n=self._servo_force_guard_fz_n,
                scale=self._servo_force_guard_scale,
                warning_active=self._servo_force_guard_warning_active,
            )
        cmd = "servo_pose " + " ".join(f"{v:.12g}" for v in guarded)
        with self._lock:
            resp = self._send_cmd(cmd)
        resp.update(_force_guard_meta(force_z_n, scale, adjusted, warning_active))
        self._last_servo_pose_real = guarded
        if force_z_n is not None or scale is not None:
            self._servo_force_guard_fz_n = force_z_n
            self._servo_force_guard_scale = scale
        self._servo_force_guard_warning_active = bool(warning_active)
        self._servo_force_guard_live_mode = bool(self._servo_force_guard_force_live or warning_active)
        return resp

    def servo_pose_j6(self, pose6: np.ndarray, joint6: float) -> dict[str, object]:
        """Send one absolute task target with explicit j6 target in servo mode."""
        requested = np.asarray(pose6, dtype=np.float64).reshape(POSE_DIM)
        reference = self._current_servo_reference_pose()
        if self._servo_force_guard_force_live or self._servo_force_guard_live_mode:
            guarded, force_z_n, scale, adjusted, warning_active = _apply_servo_force_guard(
                requested,
                reference,
                timeout_s=0.0,
            )
        else:
            guarded, force_z_n, scale, adjusted, warning_active = _apply_servo_force_guard_with_scale(
                requested,
                reference,
                force_z_n=self._servo_force_guard_fz_n,
                scale=self._servo_force_guard_scale,
                warning_active=self._servo_force_guard_warning_active,
            )
        cmd = "servo_pose_j6 " + " ".join(f"{v:.12g}" for v in guarded) + f" {float(joint6):.12g}"
        with self._lock:
            resp = self._send_cmd(cmd)
        resp.update(_force_guard_meta(force_z_n, scale, adjusted, warning_active))
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
            if self._servo_force_guard_force_live or self._servo_force_guard_live_mode:
                guarded, step_force_z_n, step_scale, step_adjusted, step_warning_active = _apply_servo_force_guard(
                    pose,
                    reference,
                    timeout_s=0.0,
                )
            else:
                guarded, step_force_z_n, step_scale, step_adjusted, step_warning_active = _apply_servo_force_guard_with_scale(
                    pose,
                    reference,
                    force_z_n=self._servo_force_guard_fz_n,
                    scale=self._servo_force_guard_scale,
                    warning_active=self._servo_force_guard_warning_active,
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
            proc.stdin.write(f"servo_chunk {len(guarded_poses)}\n")
            # Body: one pose per line
            for p in guarded_poses:
                vals = np.asarray(p, dtype=np.float64).reshape(POSE_DIM)
                proc.stdin.write(" ".join(f"{v:.12g}" for v in vals) + "\n")
            proc.stdin.flush()
            # Read response until END
            lines = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    self._proc = None
                    raise RuntimeError("daemon helper died during servo_chunk")
                stripped = line.strip()
                if stripped == "END":
                    break
                lines.append(line)
        resp = _parse_helper_output("\n".join(lines))
        resp.update(_force_guard_meta(force_z_n, scale, adjusted, warning_active))
        if guarded_poses:
            self._last_servo_pose_real = guarded_poses[-1]
        if force_z_n is not None or scale is not None:
            self._servo_force_guard_fz_n = force_z_n
            self._servo_force_guard_scale = scale
        self._servo_force_guard_warning_active = bool(warning_active)
        self._servo_force_guard_live_mode = bool(self._servo_force_guard_force_live or warning_active)
        return resp

    def movel(
        self,
        pose: np.ndarray,
        speed_frac: float = 0.0,
        *,
        speed_mps: float | None = None,
        blocking: bool = True,
    ) -> dict[str, object]:
        """Cartesian line move via moveLine.

        When ``blocking`` is False, the command returns after moveLine is
        accepted by the controller, and the caller can poll ``motion_status()``
        or wait via ``wait_motion_done()``.
        """
        if speed_mps is not None and speed_mps <= 0:
            raise ValueError(f"speed_mps must be positive, got {speed_mps}")
        if speed_mps is not None and speed_frac > 0:
            raise ValueError("Use either speed_frac or speed_mps for movel, not both.")

        requested = np.asarray(pose, dtype=np.float64).reshape(POSE_DIM)
        reference = _get_live_tcp_pose_real()
        guarded, speed_frac, speed_mps, force_z_n, scale, adjusted, warning_active = _apply_movel_force_guard(
            requested,
            reference,
            speed_frac=speed_frac,
            speed_mps=speed_mps,
        )

        if speed_mps is not None:
            cmd_name = "movel_speed" if blocking else "movel_async_speed"
        else:
            cmd_name = "movel" if blocking else "movel_async"
        cmd = cmd_name + " " + " ".join(f"{v:.12g}" for v in guarded)
        if speed_mps is not None:
            cmd += f" {float(speed_mps):.6f}"
        elif speed_frac > 0:
            cmd += f" {speed_frac:.4f}"
        with self._lock:
            resp = self._send_cmd(cmd)
        resp.update(_force_guard_meta(force_z_n, scale, adjusted, warning_active))
        return resp

    def movel_chunk(
        self,
        poses_real: list[np.ndarray],
        speed_frac: float = DEFAULT_SPEED_FRACTION,
        *,
        blend_radius_m: float = DEFAULT_MOVE_LINE_BLEND_RADIUS_M,
    ) -> dict[str, object]:
        """Queue multiple moveLine waypoints as a single blended chunk."""
        if blend_radius_m < 0:
            raise ValueError(f"blend_radius_m must be non-negative, got {blend_radius_m}")
        with self._lock:
            proc = self._ensure_started()
            proc.stdin.write(
                f"movel_chunk {len(poses_real)} {float(speed_frac):.4f} {float(blend_radius_m):.6f}\n"
            )
            for pose in poses_real:
                vals = np.asarray(pose, dtype=np.float64).reshape(POSE_DIM)
                proc.stdin.write(" ".join(f"{v:.12g}" for v in vals) + "\n")
            proc.stdin.flush()
            lines = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    self._proc = None
                    raise RuntimeError("daemon helper died during movel_chunk")
                stripped = line.strip()
                if stripped == "END":
                    break
                lines.append(line)
            return _parse_helper_output("\n".join(lines))

    def motion_status(self) -> dict[str, object]:
        """Query the current controller execution state for async moveLine."""
        with self._lock:
            return self._send_cmd("motion_status")

    def wait_motion_done(self) -> dict[str, object]:
        """Block until the current async motion finishes."""
        with self._lock:
            return self._send_cmd("wait_motion")

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
            return resp

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write("quit\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
                self._proc = None


# Two independent daemon instances:
# - _snapshot_daemon: used by get_robot_snapshot() for state reads
# - _motion_daemon: used by higher-level control for persistent movel commands
_snapshot_daemon: _DaemonHelper | None = None
_motion_daemon: _DaemonHelper | None = None
_daemon_lock = threading.Lock()


def _get_snapshot_daemon() -> _DaemonHelper:
    global _snapshot_daemon
    if _snapshot_daemon is None:
        with _daemon_lock:
            if _snapshot_daemon is None:
                _snapshot_daemon = _DaemonHelper()
    return _snapshot_daemon


def _get_motion_daemon() -> _DaemonHelper:
    global _motion_daemon
    if _motion_daemon is None:
        with _daemon_lock:
            if _motion_daemon is None:
                _motion_daemon = _DaemonHelper()
    return _motion_daemon


def _get_servo_daemon() -> _DaemonHelper:
    """Backward-compatible alias for older call sites."""
    return _get_motion_daemon()


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
    *,
    lock_yaw: bool = False,
) -> np.ndarray:
    pose = np.asarray(pose6_zyx, dtype=np.float64).reshape(POSE_DIM)
    delta = np.asarray(delta6, dtype=np.float64).reshape(OPENPI_DELTA_DIM).copy()
    if lock_yaw:
        delta[5] = 0.0
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


def execute_track_pose_file(
    track_pose_file: str,
    *,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    track_time_s: float = DEFAULT_TRACK_CONTROL_DT_S,
    smooth_scale: float = DEFAULT_TRACK_SMOOTH_SCALE,
    delay_scale: float = DEFAULT_TRACK_DELAY_SCALE,
    speed_fraction: float = DEFAULT_SPEED_FRACTION,
    execute: bool = True,
) -> dict[str, object]:
    return _run_helper(
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
        track_pose_file=track_pose_file,
        track_time_s=track_time_s,
        smooth_scale=smooth_scale,
        delay_scale=delay_scale,
        speed_fraction=speed_fraction,
        execute=execute,
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


def _parse_stop_fields(stop_raw: dict[str, object]) -> tuple[int | None, int | None, np.ndarray | None]:
    """Extract move_ret, wait_ret, final_q from a stop_robot_motion response."""
    move_ret = stop_raw.get("moveJoint_ret")
    move_ret = int(move_ret) if move_ret is not None else None
    wait_ret = stop_raw.get("wait_arrival_ret")
    wait_ret = int(wait_ret) if wait_ret is not None else None
    final_q = np.asarray(stop_raw.get("final_q_rad", []), dtype=np.float64)
    if final_q.size == 0:
        final_q = None
    return move_ret, wait_ret, final_q


def apply_tcp_delta(
    delta6: np.ndarray,
    *,
    expected_pose_sim: np.ndarray | None = None,
    lock_yaw: bool = False,
    execute: bool = True,
    speed_deg: float = DEFAULT_SPEED_DEG,
    acc_deg: float = DEFAULT_ACC_DEG,
    speed_fraction: float = DEFAULT_SPEED_FRACTION,
    reset_err_m: float = RESET_ERR_M,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
) -> TcpDeltaResult:
    """Apply a single TCP delta from the policy.

    ``expected_pose_sim`` and delta integration happen in the **sim frame**.
    IK and execution happen in the **real frame**.
    The returned ``start_pose`` / ``target_pose`` are in sim frame.
    """
    target_input = np.asarray(delta6, dtype=np.float64).reshape(OPENPI_DELTA_DIM)
    snapshot = get_robot_snapshot(
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
    )

    # Start pose in sim frame; track whether we derived it from the snapshot
    # (in which case tracking error is trivially zero).
    if expected_pose_sim is not None:
        start_sim = np.asarray(expected_pose_sim, dtype=np.float64).reshape(POSE_DIM).copy()
        start_real = sim_pose_to_real(start_sim)
        tracking_err = float(np.linalg.norm(snapshot.tcp_pose[:3] - start_real[:3]))
    else:
        start_sim = real_pose_to_sim(snapshot.tcp_pose)
        start_real = snapshot.tcp_pose.copy()
        tracking_err = 0.0

    # Integrate delta in sim frame
    target_sim = integrate_delta_tcp_pose(start_sim, target_input, lock_yaw=lock_yaw)

    # Convert to real frame for IK / execution
    target_real = sim_pose_to_real(target_sim)

    if tracking_err > float(reset_err_m):
        stop_raw = stop_robot_motion(
            helper_bin=helper_bin,
            robot_ip=robot_ip,
            port=port,
            user=user,
            password=password,
        )
        stop_move_ret, stop_wait_ret, stop_final_q = _parse_stop_fields(stop_raw)
        return TcpDeltaResult(
            ok=False,
            reason=f"actual-vs-expected tcp error {tracking_err:.6f} > reset_err_m {reset_err_m:.6f}; protective stop requested",
            snapshot=snapshot,
            start_pose=start_sim,
            target_pose=target_sim,
            target_q=None,
            ik_ret=None,
            move_ret=stop_move_ret,
            wait_ret=stop_wait_ret,
            final_q=stop_final_q,
            tracking_err=tracking_err,
            start_pose_real=start_real,
            target_pose_real=target_real,
        )

    # IK on real-frame target pose (what the robot SDK expects)
    snapshot, target_q, ik_ret, _raw = solve_target_joint_q(
        target_real,
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
    )

    if snapshot.collision or not snapshot.within_safety_limits:
        return TcpDeltaResult(
            ok=False,
            reason="robot is already in collision or outside safety limits",
            snapshot=snapshot,
            start_pose=start_sim,
            target_pose=target_sim,
            target_q=target_q,
            ik_ret=ik_ret,
            tracking_err=tracking_err,
            start_pose_real=start_real,
            target_pose_real=target_real,
        )

    if ik_ret is None or ik_ret != 0 or target_q is None:
        return TcpDeltaResult(
            ok=False,
            reason=f"inverse kinematics failed: ik_ret={ik_ret}",
            snapshot=snapshot,
            start_pose=start_sim,
            target_pose=target_sim,
            target_q=target_q,
            ik_ret=ik_ret,
            tracking_err=tracking_err,
            start_pose_real=start_real,
            target_pose_real=target_real,
        )

    if not execute:
        return TcpDeltaResult(
            ok=True,
            reason="plan only",
            snapshot=snapshot,
            start_pose=start_sim,
            target_pose=target_sim,
            target_q=target_q,
            ik_ret=ik_ret,
            tracking_err=tracking_err,
            start_pose_real=start_real,
            target_pose_real=target_real,
        )

    daemon = _get_motion_daemon()
    exec_raw = daemon.movel(target_real, speed_frac=speed_fraction)
    move_ret = exec_raw.get("movel_ret")
    move_ret = int(move_ret) if move_ret is not None else None
    wait_ret = exec_raw.get("wait_ret")
    wait_ret = int(wait_ret) if wait_ret is not None else None
    final_q = np.asarray(exec_raw.get("final_q_rad", []), dtype=np.float64)
    if final_q.size == 0:
        final_q = None
    rc = 0 if move_ret == 0 and (wait_ret is None or wait_ret == 0) else 1

    return TcpDeltaResult(
        ok=(rc == 0),
        reason="movel executed" if rc == 0 else f"movel failed move_ret={move_ret} wait_ret={wait_ret}",
        snapshot=snapshot,
        start_pose=start_sim,
        target_pose=target_sim,
        target_q=target_q,
        ik_ret=ik_ret,
        move_ret=move_ret,
        wait_ret=wait_ret,
        final_q=final_q,
        tracking_err=tracking_err,
        start_pose_real=start_real,
        target_pose_real=target_real,
    )


def retime_tcp_action_chunk(
    delta_actions: np.ndarray,
    *,
    start_pose_sim: np.ndarray,
    lock_yaw: bool = False,
    control_dt_s: float = DEFAULT_TRACK_CONTROL_DT_S,
    max_linear_speed_mps: float = DEFAULT_TCP_LINEAR_SPEED_MPS,
    max_angular_speed_radps: float = DEFAULT_TCP_ANGULAR_SPEED_RADPS,
    z_min: float = DEFAULT_Z_MIN_M,
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

    for idx, raw_delta in enumerate(actions):
        delta = np.asarray(raw_delta, dtype=np.float64).reshape(OPENPI_DELTA_DIM).copy()
        if lock_yaw:
            delta[5] = 0.0

        linear_dist = float(np.linalg.norm(delta[:3]))
        angular_dist = float(np.linalg.norm(delta[3:]))
        linear_time = linear_dist / float(max_linear_speed_mps) if max_linear_speed_mps > 0.0 else 0.0
        angular_time = angular_dist / float(max_angular_speed_radps) if max_angular_speed_radps > 0.0 else 0.0
        required_time = max(linear_time, angular_time, float(control_dt_s))
        step_count = max(1, int(math.ceil(required_time / float(control_dt_s))))
        step_delta = delta / float(step_count)

        for _ in range(step_count):
            # Integrate in sim frame
            current_sim = integrate_delta_tcp_pose(current_sim, step_delta, lock_yaw=lock_yaw)
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
    lock_yaw: bool = False,
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
        current_sim = integrate_delta_tcp_pose(current_sim, raw_delta, lock_yaw=lock_yaw)

    current_real = sim_pose_to_real(current_sim)

    return IntegratedTarget(
        start_pose_sim=start_sim,
        start_pose_real=start_real,
        final_pose_sim=current_sim.copy(),
        final_pose_real=current_real.copy(),
        n_actions=int(actions.shape[0]),
    )


def execute_tcp_action_chunk(
    delta_actions: np.ndarray,
    *,
    expected_pose_sim: np.ndarray | None = None,
    lock_yaw: bool = False,
    execute: bool = True,
    blocking: bool = True,
    reset_err_m: float = RESET_ERR_M,
    blend_radius_m: float = DEFAULT_MOVE_LINE_BLEND_RADIUS_M,
) -> TrackChunkResult:
    """Retime delta actions into a blended moveLine chunk and queue it.

    ``expected_pose_sim`` is the sim-frame start pose for integration.
    When provided, skips the expensive snapshot helper and uses daemon
    motion_status for a quick tracking-error check.
    """
    # Determine start pose
    snapshot: RobotSnapshot | None = None
    daemon = _get_motion_daemon()
    daemon_status: dict[str, object] | None = None
    if expected_pose_sim is not None:
        start_sim = np.asarray(expected_pose_sim, dtype=np.float64).reshape(POSE_DIM).copy()
        expected_real = sim_pose_to_real(start_sim)
        tracking_err = 0.0
        try:
            daemon_status = daemon.motion_status()
            tcp_pose_raw = np.asarray(daemon_status.get("tcp_pose", []), dtype=np.float64)
            if tcp_pose_raw.size == POSE_DIM:
                tracking_err = float(np.linalg.norm(tcp_pose_raw[:3] - expected_real[:3]))
        except Exception:
            pass
    else:
        snapshot = get_robot_snapshot()
        start_sim = real_pose_to_sim(snapshot.tcp_pose)
        expected_real = snapshot.tcp_pose.copy()
        tracking_err = 0.0

    if tracking_err > float(reset_err_m):
        return TrackChunkResult(
            ok=False,
            reason=f"actual-vs-expected tcp error {tracking_err:.6f} > reset_err_m {reset_err_m:.6f}",
            snapshot=snapshot,
            start_pose=start_sim,
            final_pose=start_sim,
            sample_count=0,
            tracking_err=tracking_err,
            start_pose_real=expected_real,
            final_pose_real=expected_real,
        )

    if snapshot is not None and (snapshot.collision or not snapshot.within_safety_limits):
        return TrackChunkResult(
            ok=False,
            reason="robot is already in collision or outside safety limits",
            snapshot=snapshot,
            start_pose=start_sim,
            final_pose=start_sim,
            sample_count=0,
            tracking_err=tracking_err,
            start_pose_real=expected_real,
            final_pose_real=expected_real,
        )

    # Convert the action horizon into a smooth, speed-limited moveLine chunk.
    target = retime_tcp_action_chunk(
        delta_actions,
        start_pose_sim=start_sim,
        lock_yaw=lock_yaw,
    )

    if not target.steps:
        return TrackChunkResult(
            ok=True,
            reason="empty chunk",
            snapshot=snapshot,
            start_pose=target.start_pose,
            final_pose=target.final_pose,
            sample_count=0,
            control_dt_s=target.control_dt_s,
            tracking_err=tracking_err,
            start_pose_real=target.start_pose_real,
            final_pose_real=target.final_pose_real,
        )

    if not execute:
        return TrackChunkResult(
            ok=True,
            reason="plan only",
            snapshot=snapshot,
            start_pose=target.start_pose,
            final_pose=target.final_pose,
            sample_count=len(target.steps),
            control_dt_s=target.control_dt_s,
            tracking_err=tracking_err,
            start_pose_real=target.start_pose_real,
            final_pose_real=target.final_pose_real,
        )

    if daemon_status is None:
        try:
            daemon_status = daemon.motion_status()
        except Exception:
            daemon_status = None

    if daemon_status is not None:
        collision = bool(daemon_status.get("collision", False))
        within_limits = bool(daemon_status.get("within_safety_limits", True))
        exec_id_raw = daemon_status.get("exec_id")
        queue_size_raw = daemon_status.get("queue_size")
        is_steady_raw = daemon_status.get("is_steady")
        exec_id = int(exec_id_raw) if exec_id_raw is not None else -1
        queue_size = int(queue_size_raw) if queue_size_raw is not None else 0
        is_steady = bool(is_steady_raw) if is_steady_raw is not None else True
        if collision or not within_limits:
            return TrackChunkResult(
                ok=False,
                reason="robot is already in collision or outside safety limits",
                snapshot=snapshot,
                start_pose=target.start_pose,
                final_pose=target.start_pose,
                sample_count=0,
                control_dt_s=target.control_dt_s,
                track_sent=0,
                tracking_err=tracking_err,
                exec_mode="movel_chunk",
                raw={"motion_status": daemon_status},
                start_pose_real=target.start_pose_real,
                final_pose_real=target.start_pose_real,
            )
        if exec_id != -1 or queue_size > 0 or not is_steady:
            return TrackChunkResult(
                ok=True,
                reason="motion in progress; skip requeue",
                snapshot=snapshot,
                start_pose=target.start_pose,
                final_pose=target.start_pose,
                sample_count=0,
                control_dt_s=target.control_dt_s,
                track_sent=0,
                tracking_err=tracking_err,
                exec_mode="movel_chunk",
                raw={"motion_status": daemon_status},
                start_pose_real=target.start_pose_real,
                final_pose_real=target.start_pose_real,
            )

    chunk_raw = daemon.movel_chunk(
        [step.pose_real for step in target.steps],
        blend_radius_m=blend_radius_m,
    )
    chunk_ret_raw = chunk_raw.get("movel_chunk_ret")
    chunk_ret = int(chunk_ret_raw) if chunk_ret_raw is not None else -1
    chunk_queued_raw = chunk_raw.get("chunk_queued")
    chunk_queued = int(chunk_queued_raw) if chunk_queued_raw is not None else 0

    if chunk_ret != 0:
        error_kind = chunk_raw.get("error")
        motion_name = chunk_raw.get("motion_ret_name", "unknown")
        if error_kind:
            reason = f"movel_chunk failed: {error_kind}"
            reason += f" ({motion_name}, ret={chunk_ret})" if motion_name != "unknown" else f" (ret={chunk_ret})"
        else:
            reason = f"movel_chunk failed: {motion_name} (ret={chunk_ret})"
        return TrackChunkResult(
            ok=False,
            reason=reason,
            snapshot=snapshot,
            start_pose=target.start_pose,
            final_pose=target.final_pose,
            sample_count=0,
            control_dt_s=target.control_dt_s,
            track_sent=chunk_queued,
            track_ret=chunk_ret,
            tracking_err=tracking_err,
            exec_mode="movel_chunk",
            raw=chunk_raw,
            start_pose_real=target.start_pose_real,
            final_pose_real=target.final_pose_real,
        )

    # If blocking requested, wait for arrival
    wait_raw: dict[str, object] | None = None
    final_q: np.ndarray | None = None
    actual_final_real = target.final_pose_real.copy()
    actual_final_sim = target.final_pose.copy()
    if blocking:
        wait_raw = daemon.wait_motion_done()
        wait_ret_raw = wait_raw.get("wait_ret")
        wait_ret = int(wait_ret_raw) if wait_ret_raw is not None else None
        final_q_raw = np.asarray(wait_raw.get("final_q_rad", []), dtype=np.float64)
        if final_q_raw.size:
            final_q = final_q_raw
        # Read actual final pose
        final_pose_raw = np.asarray(wait_raw.get("final_pose", []), dtype=np.float64)
        if final_pose_raw.size == POSE_DIM:
            actual_final_real = final_pose_raw.reshape(POSE_DIM).copy()
            actual_final_sim = real_pose_to_sim(actual_final_real)
        if wait_ret is not None and wait_ret != 0:
            return TrackChunkResult(
                ok=False,
                reason=f"movel_chunk wait failed wait_ret={wait_ret}",
                snapshot=snapshot,
                start_pose=target.start_pose,
                final_pose=actual_final_sim,
                sample_count=0,
                control_dt_s=target.control_dt_s,
                track_sent=chunk_queued,
                track_ret=wait_ret,
                final_q=final_q,
                tracking_err=tracking_err,
                exec_mode="movel_chunk",
                raw={"chunk": chunk_raw, "wait": wait_raw},
                start_pose_real=target.start_pose_real,
                final_pose_real=actual_final_real,
            )

    return TrackChunkResult(
        ok=True,
        reason="movel_chunk queued",
        snapshot=snapshot,
        start_pose=target.start_pose,
        final_pose=actual_final_sim,
        sample_count=len(target.steps),
        control_dt_s=target.control_dt_s,
        track_sent=chunk_queued,
        track_ret=0,
        final_q=final_q,
        tracking_err=tracking_err,
        exec_mode="movel_chunk",
        raw={"chunk": chunk_raw, "wait": wait_raw},
        start_pose_real=target.start_pose_real,
        final_pose_real=actual_final_real,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug entrypoint for the TCP delta control module.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--delta", nargs=6, type=float)
    parser.add_argument("--no-execute", action="store_true")
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
    if args.delta is not None:
        if not is_alignment_ready():
            snap = get_robot_snapshot(helper_bin=str(helper))
            set_runtime_alignment(snap.tcp_pose)
            print("POSE_ALIGN_INIT_FROM_CURRENT_TCP=1")
        result = apply_tcp_delta(
            np.asarray(args.delta, dtype=np.float64),
            execute=not args.no_execute,
            helper_bin=str(helper),
        )
        print(result)
        return 0 if result.ok else 1
    print(f"HELPER_READY={helper}")
    return 0


__all__ = [
    "RESET_ERR_M",
    "DEFAULT_TRACK_CONTROL_DT_S",
    "DEFAULT_TRACK_SMOOTH_SCALE",
    "DEFAULT_TRACK_DELAY_SCALE",
    "DEFAULT_TCP_LINEAR_SPEED_MPS",
    "DEFAULT_TCP_ANGULAR_SPEED_RADPS",
    "RobotSnapshot",
    "TcpDeltaResult",
    "RetimedTcpStep",
    "RetimedTcpChunk",
    "TrackChunkResult",
    "build_helper",
    "_get_motion_daemon",
    "get_robot_snapshot",
    "integrate_delta_tcp_pose",
    "solve_target_joint_q",
    "execute_joint_target",
    "execute_track_pose_file",
    "stop_robot_motion",
    "apply_tcp_delta",
    "retime_tcp_action_chunk",
    "plan_tcp_action_chunk_movel",
    "execute_tcp_action_chunk",
]


if __name__ == "__main__":
    raise SystemExit(main())

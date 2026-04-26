#!/usr/bin/env python3
"""
Observation builder for the real-robot OpenPI pipeline.

Responsibilities:
- capture aligned RGB frames from the two Orbbec cameras
- preprocess RGB frames for OpenPI input
- read robot TCP state via tcp_control.py (daemon mode)
- package an OpenPI-compatible observation dict:
    {
      "observation/state": float32[7],  # yaw-mode
      "observation/image": uint8[224,224,3],
      "observation/wrist_image": uint8[224,224,3],
      "prompt": str,
    }

Gripper state is maintained locally (set_gripper_open_scalar).
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBPropertyID

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from support.pose_align import real_pose_to_sim
from support.tcp_control import RobotSnapshot, get_robot_snapshot
from utils.path_utils import get_repo_root
from utils.pyorbbec_utils import frame_to_bgr_image


OPENPI_IMAGE_SIZE = 224
MAIN_CAMERA_KEY = "observation/image"
WRIST_CAMERA_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
PROMPT_KEY = "prompt"
STATE_MODE_YAW = "yaw"

SERIAL_ROLE_HINTS = {
    "335": "main",
    "305": "wrist",
}

CAMERA_COLOR_PROFILE = {
    "auto_exposure": True,
    "ae_max_exposure": 500,
    "brightness": 0,
    "backlight": 1,
    "exposure": 156,
    "gain": 16,
}


def _infer_camera_role(device_name: str) -> str | None:
    for token, hinted_role in SERIAL_ROLE_HINTS.items():
        if token in device_name:
            return hinted_role
    return None


def _apply_camera_profile_to_device(role: str, dev) -> None:
    try:
        dev.set_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            bool(CAMERA_COLOR_PROFILE["auto_exposure"]),
        )
        dev.set_int_property(
            OBPropertyID.OB_PROP_COLOR_AE_MAX_EXPOSURE_INT,
            int(CAMERA_COLOR_PROFILE["ae_max_exposure"]),
        )
        dev.set_int_property(
            OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
            int(CAMERA_COLOR_PROFILE["brightness"]),
        )
        dev.set_int_property(
            OBPropertyID.OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT,
            int(CAMERA_COLOR_PROFILE["backlight"]),
        )
        dev.set_int_property(
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
            int(CAMERA_COLOR_PROFILE["exposure"]),
        )
        dev.set_int_property(
            OBPropertyID.OB_PROP_COLOR_GAIN_INT,
            int(CAMERA_COLOR_PROFILE["gain"]),
        )
    except Exception as exc:
        raise RuntimeError(f"failed to configure {role} camera profile: {exc}") from exc




def _ensure_openpi_client_path() -> None:
    repo_root = get_repo_root()
    candidate = repo_root / "packages" / "openpi-client" / "src"
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


_ensure_openpi_client_path()
from openpi_client import image_tools  # noqa: E402


@dataclass
class TimedFrame:
    role: str
    name: str
    serial: str
    image_bgr: np.ndarray
    timestamp_us: int
    system_timestamp_us: int
    global_timestamp_us: int
    width: int
    height: int


@dataclass
class AlignedObservation:
    obs: dict[str, object]
    robot_snapshot: RobotSnapshot
    aligned_tcp_pose_sim: np.ndarray
    gripper_open_scalar: float
    yaw_state_scalar: float | None
    yaw_readback_scalar: float | None
    main_frame: TimedFrame
    wrist_frame: TimedFrame
    image_pair_system_diff_us: int


def preprocess_image_for_openpi(image_bgr: np.ndarray, *, size: int = OPENPI_IMAGE_SIZE) -> np.ndarray:
    img = np.asarray(image_bgr)
    if img.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image_tools.resize_with_pad(img, size, size)
    img = image_tools.convert_to_uint8(img)
    return img


def _euler_zyx_to_quat_wxyz(e: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = e.astype(np.float64)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / (np.linalg.norm(quat) + 1e-12)


def _quat_to_axis_angle_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    q = quat_wxyz.astype(np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] > 0:
        q = -q
    w, x, y, z = q
    if abs(w) > 0.999999:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half = np.sqrt(max(1.0 - w * w, 1e-12))
    axis = np.array([x, y, z], dtype=np.float64) / sin_half
    return (axis * angle).astype(np.float32)


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    angles = np.asarray(delta, dtype=np.float64).reshape(3)
    return np.arctan2(np.sin(angles), np.cos(angles)).astype(np.float64)


def pose6_to_openpi_state(pose6_zyx: np.ndarray, gripper_open: float) -> np.ndarray:
    pose = np.asarray(pose6_zyx, dtype=np.float64).reshape(6)
    quat = _euler_zyx_to_quat_wxyz(pose[3:6])
    aa = _quat_to_axis_angle_wxyz(quat)
    return np.concatenate(
        [pose[:3].astype(np.float32), aa, [float(gripper_open)]],
        axis=0,
    ).astype(np.float32)


class CameraPair:
    """Shared Orbbec camera stack for both online observation and data collection."""

    def __init__(self) -> None:
        self._ctx = Context()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._latest_frames: dict[str, np.ndarray | None] = {"main": None, "wrist": None}
        self._consecutive_failures: dict[str, int] = {"main": 0, "wrist": 0}
        self._last_restart_time: dict[str, float] = {"main": 0.0, "wrist": 0.0}
        self._restart_threshold = 20
        self._restart_cooldown_s = 2.0
        self._device_serials: dict[str, str] = {}
        self._pipelines: dict[str, Pipeline] = {}

        devs = self._ctx.query_devices()
        for idx in range(devs.get_count()):
            dev = devs.get_device_by_index(idx)
            info = dev.get_device_info()
            role = _infer_camera_role(info.get_name())
            if role is None:
                continue
            self._device_serials[role] = info.get_serial_number()
            _apply_camera_profile_to_device(role, dev)
            self._start_pipeline_for_device(role, dev)

        missing = {"main", "wrist"} - set(self._pipelines.keys())
        if missing:
            raise RuntimeError(f"Missing cameras: {sorted(missing)}")

        for role in self._pipelines.keys():
            thread = threading.Thread(target=self._capture_loop, args=(role,), daemon=True)
            thread.start()
            self._threads.append(thread)

        self._wait_until_ready()
        print(f"  Cameras ready: {sorted(self._pipelines.keys())}")

    def _start_pipeline_for_device(self, role: str, dev) -> None:
        pipe = Pipeline(dev)
        cfg = Config()
        prof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
        cfg.enable_stream(prof)
        pipe.start(cfg)
        with self._lock:
            self._pipelines[role] = pipe

    def _find_device_for_role(self, role: str):
        target_serial = self._device_serials.get(role)
        devs = self._ctx.query_devices()
        for idx in range(devs.get_count()):
            dev = devs.get_device_by_index(idx)
            info = dev.get_device_info()
            if target_serial and info.get_serial_number() == target_serial:
                return dev
        for idx in range(devs.get_count()):
            dev = devs.get_device_by_index(idx)
            info = dev.get_device_info()
            inferred_role = _infer_camera_role(info.get_name())
            if inferred_role == role:
                self._device_serials[role] = info.get_serial_number()
                return dev
        raise RuntimeError(f"camera device not found for role={role}")

    def _restart_pipeline(self, role: str, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_restart_time[role] < self._restart_cooldown_s:
            return
        self._last_restart_time[role] = now
        print(f"  [camera:{role}] restarting pipeline after {reason}")
        old_pipe = None
        with self._lock:
            old_pipe = self._pipelines.get(role)
            self._latest_frames[role] = None
        if old_pipe is not None:
            try:
                old_pipe.stop()
            except Exception:
                pass
        dev = self._find_device_for_role(role)
        _apply_camera_profile_to_device(role, dev)
        self._start_pipeline_for_device(role, dev)
        self._consecutive_failures[role] = 0

    def _capture_loop(self, role: str) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    pipe = self._pipelines.get(role)
                if pipe is None:
                    time.sleep(0.1)
                    continue
                frame_set = pipe.wait_for_frames(100)
            except Exception:
                self._consecutive_failures[role] += 1
                if self._consecutive_failures[role] >= self._restart_threshold:
                    try:
                        self._restart_pipeline(role, "wait_for_frames exception")
                    except Exception as exc:
                        print(f"  [camera:{role}] restart failed: {exc}")
                continue
            if frame_set is None:
                self._consecutive_failures[role] += 1
                if self._consecutive_failures[role] >= self._restart_threshold:
                    try:
                        self._restart_pipeline(role, "wait_for_frames timeout")
                    except Exception as exc:
                        print(f"  [camera:{role}] restart failed: {exc}")
                continue
            color = frame_set.get_color_frame()
            if color is None:
                self._consecutive_failures[role] += 1
                continue
            img = frame_to_bgr_image(color)
            if img is None:
                self._consecutive_failures[role] += 1
                continue
            self._consecutive_failures[role] = 0
            with self._lock:
                self._latest_frames[role] = img

    def _wait_until_ready(self, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if all(frame is not None for frame in self._latest_frames.values()):
                    return
            time.sleep(0.05)
        with self._lock:
            missing = sorted(role for role, frame in self._latest_frames.items() if frame is None)
        raise RuntimeError(f"camera frames not ready: {missing}")

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            main = self._latest_frames["main"]
            wrist = self._latest_frames["wrist"]
            main_img = (self._fallback_frame if main is None else main).copy()
            wrist_img = (self._fallback_frame if wrist is None else wrist).copy()
        return main_img, wrist_img

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for pipe in list(self._pipelines.values()):
            try:
                pipe.stop()
            except Exception:
                pass
        self._pipelines.clear()


class RealRobotOpenPIObservationBuilder:
    def __init__(self, *, state_mode: str = STATE_MODE_YAW) -> None:
        self.ctx = Context()
        self._pipelines: dict[str, tuple[object, Pipeline]] = {}
        self._started = False
        self._pool = ThreadPoolExecutor(max_workers=6)
        # Observation path uses the local cache as the authoritative gripper state.
        self._gripper_open_scalar: float = 1.0  # default: open
        self.set_state_mode(state_mode)

    def _apply_camera_profile(self, role: str, dev) -> None:
        _apply_camera_profile_to_device(role, dev)

    def set_gripper_open_scalar(self, value: float) -> None:
        """Update the locally-cached fallback gripper scalar (1.0=open, 0.0=closed)."""
        self._gripper_open_scalar = float(value)

    def set_semantic_yaw_scalar(self, value: float | None) -> None:
        _ = value

    def set_state_mode(self, mode: str) -> None:
        mode_norm = str(mode).strip().lower()
        if mode_norm != STATE_MODE_YAW:
            mode_norm = STATE_MODE_YAW
        self._state_mode = mode_norm

    def reset_pose_filter(self) -> None:
        pass

    def start(self) -> None:
        if self._started:
            return
        devs = self.ctx.query_devices()
        for i in range(devs.get_count()):
            dev = devs.get_device_by_index(i)
            info = dev.get_device_info()
            name = info.get_name()
            serial = info.get_serial_number()
            role = None
            for token, hinted_role in SERIAL_ROLE_HINTS.items():
                if token in name:
                    role = hinted_role
                    break
            if role is None:
                continue
            self._apply_camera_profile(role, dev)
            pipe = Pipeline(dev)
            cfg = Config()
            prof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
            cfg.enable_stream(prof)
            pipe.start(cfg)
            self._pipelines[role] = (info, pipe)
        missing = {"main", "wrist"} - set(self._pipelines.keys())
        if missing:
            raise RuntimeError(f"missing expected cameras: {sorted(missing)}")
        self._started = True

    def grab_bgr_pair(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Grab one BGR frame from each camera (lightweight, no preprocessing).

        Returns (main_bgr, wrist_bgr).  Either may be None on failure.
        """
        results: dict[str, np.ndarray | None] = {"main": None, "wrist": None}
        for role in ("main", "wrist"):
            entry = self._pipelines.get(role)
            if entry is None:
                continue
            _info, pipe = entry
            try:
                frame_set = pipe.wait_for_frames(100)
                if frame_set is None:
                    continue
                color = frame_set.get_color_frame()
                if color is None:
                    continue
                img = frame_to_bgr_image(color)
                results[role] = img
            except Exception:
                pass
        return results["main"], results["wrist"]

    def stop(self) -> None:
        for _, pipe in self._pipelines.values():
            try:
                pipe.stop()
            except Exception:
                pass
        self._pipelines.clear()
        self._started = False
        self._pool.shutdown(wait=False)

    def __enter__(self) -> "RealRobotOpenPIObservationBuilder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _collect_recent_frames(self, role: str, count: int = 2) -> list[TimedFrame]:
        info, pipe = self._pipelines[role]
        frames: list[TimedFrame] = []
        for _ in range(count):
            frame_set = pipe.wait_for_frames(200)
            if frame_set is None:
                continue
            color = frame_set.get_color_frame()
            if color is None:
                continue
            img = frame_to_bgr_image(color)
            if img is None:
                continue
            frames.append(
                TimedFrame(
                    role=role,
                    name=info.get_name(),
                    serial=info.get_serial_number(),
                    image_bgr=img,
                    timestamp_us=int(color.get_timestamp_us()),
                    system_timestamp_us=int(color.get_system_timestamp_us()),
                    global_timestamp_us=int(color.get_global_timestamp_us()),
                    width=int(color.get_width()),
                    height=int(color.get_height()),
                )
            )
        if not frames:
            raise RuntimeError(f"failed to capture any frames for role={role}")
        return frames

    def _capture_best_pair(self, max_retries: int = 3) -> tuple[TimedFrame, TimedFrame, int]:
        import time as _time
        for attempt in range(max_retries):
            try:
                # Capture both cameras in parallel - each blocks on its own
                # USB pipeline, so concurrent reads cut wall-clock time roughly
                # in half.
                fut_main = self._pool.submit(self._collect_recent_frames, "main")
                fut_wrist = self._pool.submit(self._collect_recent_frames, "wrist")
                main_frames = fut_main.result()
                wrist_frames = fut_wrist.result()
                break
            except RuntimeError:
                if attempt == max_retries - 1:
                    raise
                _time.sleep(0.3)
        # Sort by timestamp and use two-pointer scan: O(n log n) instead of O(n^2)
        main_sorted = sorted(main_frames, key=lambda f: f.system_timestamp_us)
        wrist_sorted = sorted(wrist_frames, key=lambda f: f.system_timestamp_us)
        best_diff, best_m, best_w = abs(main_sorted[0].system_timestamp_us - wrist_sorted[0].system_timestamp_us), main_sorted[0], wrist_sorted[0]
        i = j = 0
        while i < len(main_sorted) and j < len(wrist_sorted):
            diff = abs(main_sorted[i].system_timestamp_us - wrist_sorted[j].system_timestamp_us)
            if diff < best_diff:
                best_diff, best_m, best_w = diff, main_sorted[i], wrist_sorted[j]
            if main_sorted[i].system_timestamp_us < wrist_sorted[j].system_timestamp_us:
                i += 1
            else:
                j += 1
        return best_m, best_w, int(best_diff)

    def build_observation(self, prompt: str, *, image_size: int = OPENPI_IMAGE_SIZE) -> AlignedObservation:
        if not self._started:
            self.start()

        t0 = time.monotonic()

        # Launch robot state poll immediately so it overlaps with camera capture.
        fut_snapshot = self._pool.submit(get_robot_snapshot)

        main_frame, wrist_frame, pair_diff_us = self._capture_best_pair()
        t_capture = time.monotonic()

        # Image preprocessing (fast, ~15ms each).
        fut_main_img = self._pool.submit(preprocess_image_for_openpi, main_frame.image_bgr, size=image_size)
        fut_wrist_img = self._pool.submit(preprocess_image_for_openpi, wrist_frame.image_bgr, size=image_size)
        main_image = fut_main_img.result()
        wrist_image = fut_wrist_img.result()

        gripper_open = self._gripper_open_scalar

        # Snapshot: should already be done (launched before camera capture).
        robot_snapshot = fut_snapshot.result()
        t_done = time.monotonic()

        obs_ms = (t_done - t0) * 1000
        cap_ms = (t_capture - t0) * 1000
        par_ms = (t_done - t_capture) * 1000
        print(f"    [obs] total={obs_ms:.0f}ms  cap={cap_ms:.0f}  post={par_ms:.0f}")

        aligned_tcp_pose_sim = real_pose_to_sim(robot_snapshot.tcp_pose)
        joint_q = np.asarray(robot_snapshot.joint_q, dtype=np.float64).reshape(-1)
        yaw_readback = float(joint_q[5]) if joint_q.size >= 6 else None
        state = pose6_to_openpi_state(aligned_tcp_pose_sim, gripper_open)

        obs = {
            STATE_KEY: state.astype(np.float32),
            MAIN_CAMERA_KEY: main_image,
            WRIST_CAMERA_KEY: wrist_image,
            PROMPT_KEY: prompt,
        }

        return AlignedObservation(
            obs=obs,
            robot_snapshot=robot_snapshot,
            aligned_tcp_pose_sim=aligned_tcp_pose_sim,
            gripper_open_scalar=gripper_open,
            yaw_state_scalar=None,
            yaw_readback_scalar=yaw_readback,
            main_frame=main_frame,
            wrist_frame=wrist_frame,
            image_pair_system_diff_us=pair_diff_us,
        )


__all__ = [
    "OPENPI_IMAGE_SIZE",
    "STATE_MODE_YAW",
    "TimedFrame",
    "AlignedObservation",
    "CameraPair",
    "preprocess_image_for_openpi",
    "pose6_to_openpi_state",
    "RealRobotOpenPIObservationBuilder",
]

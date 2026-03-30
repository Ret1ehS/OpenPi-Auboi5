#!/usr/bin/env python3
"""
Real-robot data collection script for storage-box tasks.

Task: open storage box
  Prompt: "open the storage box"
  Workflow (each episode, RECORDED):
    1. Move to box lid grasp XY position (x=0.65257, y=0.19220)
    2. Rotate j6 by -90 deg while lowering to z=0.27140
    3. Close gripper (snapshot frame)
    4. Move to open position (x=0.41200, y=0.19220), open gripper
    4. Lift 5 cm

Saved format (same as collect_data_cube):
  states.npy     (N, 8) float32  [x, y, z, aa_x, aa_y, aa_z, gripper, j6]
  actions.npy    (N, 7) float32  [dx, dy, dz, droll, dpitch, dj6, gripper_next]
  timestamps.npy (N,)   float32  env_step / 50
  env_steps.npy  (N,)   int64
  images.npz             main_images (N,224,224,3), wrist_images (N,224,224,3)
  metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = OPENPI_ROOT / "repo"
REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUIT_TOKENS = {"q", "quit", "exit"}
STATE_FILE_NAME = ".collect_storage_state.json"

# Data collection parameters
RAW_CAPTURE_FPS = 30
CONTROL_DT = 1.0 / RAW_CAPTURE_FPS
SAVE_FPS = 50
SAVE_CONTROL_DT = 1.0 / SAVE_FPS
IMAGE_SIZE = 224
BASE_FPS = SAVE_FPS

# Motion speed (m/s)
LINEAR_SPEED = 0.10
DEFAULT_ASYNC_MOVE_TIMEOUT_S = 30.0
DEFAULT_J6_SPEED_RADPS = float(np.deg2rad(10.0))
DEFAULT_J6_MOVE_SPEED_DEG = 10.0
DEFAULT_J6_MOVE_ACC_DEG = 20.0

# --- Task-specific waypoints (metres, real-robot base frame) ---
# "open the storage box"
OPEN_BOX_GRASP_XY = (0.69690, 0.18406)
OPEN_BOX_GRASP_Z = 0.26952
OPEN_BOX_RELEASE_XY = (0.49690, 0.18406)
OPEN_BOX_LIFT_CM = 5
OPEN_BOX_GRASP_J6_DELTA_RAD = float(np.deg2rad(-90.0))


def _maybe_reexec_into_repo_venv() -> None:
    target = REPO_VENV_PYTHON
    if not target.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        desired = target.resolve()
        if current == desired:
            return
    except Exception:
        desired = target
    os.execv(str(target), [str(target), *sys.argv])


from support.get_obs import CameraPair, preprocess_image_for_openpi as preprocess_image


# ---------------------------------------------------------------------------
# Pose / state helpers  (identical to collect_data_cube)
# ---------------------------------------------------------------------------

def _euler_zyx_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler.astype(np.float64)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_axis_angle_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4).copy()
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] > 0.0:
        q = -q
    w, x, y, z = q
    if abs(w) > 0.999999:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half = np.sqrt(max(1.0 - w * w, 1e-12))
    axis = np.array([x, y, z], dtype=np.float64) / sin_half
    return (axis * angle).astype(np.float32)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def build_state8(pose6_zyx: np.ndarray, gripper: float, joint6: float) -> np.ndarray:
    """Build one AUBO state row [x, y, z, aa_x, aa_y, aa_z, gripper, j6]."""
    pose = np.asarray(pose6_zyx, dtype=np.float64).reshape(6)
    quat = _euler_zyx_to_quat_wxyz(pose[3:6])
    aa = _quat_to_axis_angle_wxyz(quat)
    return np.array(
        [pose[0], pose[1], pose[2], aa[0], aa[1], aa[2], float(gripper), float(joint6)],
        dtype=np.float32,
    )


def compute_delta_actions(pose6: np.ndarray, gripper: np.ndarray, joint6: np.ndarray) -> np.ndarray:
    """Compute (N, 7) delta actions from resampled pose6 / gripper / j6 sequences."""
    pose6 = np.asarray(pose6, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    joint6 = np.asarray(joint6, dtype=np.float32).reshape(-1)
    num_frames = int(pose6.shape[0])
    actions = np.zeros((num_frames, 7), dtype=np.float32)
    for idx in range(num_frames):
        if idx < num_frames - 1:
            actions[idx, :3] = pose6[idx + 1, :3] - pose6[idx, :3]
            euler_curr = pose6[idx, 3:6]
            euler_next = pose6[idx + 1, 3:6]
            deuler = np.arctan2(np.sin(euler_next - euler_curr), np.cos(euler_next - euler_curr))
            actions[idx, 3:5] = deuler[:2]
            actions[idx, 5] = _wrap_angle(float(joint6[idx + 1] - joint6[idx]))
            actions[idx, 6] = gripper[idx + 1]
        else:
            actions[idx, :6] = 0.0
            actions[idx, 6] = gripper[idx]
    return actions


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _coerce_exec_id(value: object) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def _pose_close(actual_pose: np.ndarray, target_pose: np.ndarray, *, pos_tol: float = 0.003, ang_tol: float = 0.03) -> bool:
    actual = np.asarray(actual_pose, dtype=np.float64).reshape(6)
    target = np.asarray(target_pose, dtype=np.float64).reshape(6)
    pos_err = float(np.linalg.norm(actual[:3] - target[:3]))
    ang_err = np.arctan2(np.sin(actual[3:] - target[3:]), np.cos(actual[3:] - target[3:]))
    return pos_err <= pos_tol and float(np.linalg.norm(ang_err)) <= ang_tol


def ensure_joint_move_ok(result, label: str) -> None:
    if not result.ok:
        raise RuntimeError(f"{label} failed: {result.reason}")


def ensure_movel_ok(resp: dict[str, object], label: str) -> None:
    move_ret = int(resp.get("movel_ret", -999))
    wait_ret_raw = resp.get("wait_ret")
    wait_ret = int(wait_ret_raw) if wait_ret_raw is not None else None
    if move_ret != 0 or (wait_ret is not None and wait_ret != 0):
        raise RuntimeError(f"{label} failed: {resp}")


def ensure_gripper_ok(ok: bool, label: str) -> None:
    if not ok:
        raise RuntimeError(f"{label} failed")


def start_async_movel(daemon, target_real: np.ndarray, label: str, *, speed_mps: float) -> dict[str, object]:
    resp = daemon.movel(target_real, speed_mps=speed_mps, blocking=False)
    ensure_movel_ok(resp, f"{label} start")
    return resp


def execute_movel_and_wait(daemon, target_real: np.ndarray, label: str, *, speed_mps: float) -> dict[str, object]:
    start_async_movel(daemon, target_real, label, speed_mps=speed_mps)
    resp = daemon.wait_motion_done()
    wait_ret_raw = resp.get("wait_ret")
    wait_ret = int(wait_ret_raw) if wait_ret_raw is not None else None
    if wait_ret is not None and wait_ret != 0:
        raise RuntimeError(f"{label} wait failed: {resp}")
    return resp


def stop_motion_and_confirm(daemon, label: str) -> dict[str, object]:
    resp = daemon.stop_motion()
    stop_ret_raw = resp.get("stop_ret")
    stop_ret = int(stop_ret_raw) if stop_ret_raw is not None else None
    wait_ret_raw = resp.get("wait_ret")
    wait_ret = int(wait_ret_raw) if wait_ret_raw is not None else None
    exec_id = _coerce_exec_id(resp.get("exec_id"))
    is_steady = _coerce_bool(resp.get("is_steady"))
    if stop_ret not in (None, 0):
        raise RuntimeError(f"{label} stop failed: {resp}")
    if wait_ret not in (None, 0):
        raise RuntimeError(f"{label} stop wait failed: {resp}")
    if exec_id != -1 or not is_steady:
        raise RuntimeError(f"{label} did not settle after stop: {resp}")
    return resp


def build_pose_at_xy(base_pose: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float64).reshape(6).copy()
    pose[0] = float(x)
    pose[1] = float(y)
    pose[2] = float(z)
    return pose


def interpolate_pose6(start_pose: np.ndarray, end_pose: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(start_pose, dtype=np.float64).reshape(6)
    end = np.asarray(end_pose, dtype=np.float64).reshape(6)
    out = start.copy()
    out[:3] = (1.0 - alpha) * start[:3] + alpha * end[:3]
    ang_delta = np.arctan2(np.sin(end[3:6] - start[3:6]), np.cos(end[3:6] - start[3:6]))
    out[3:6] = start[3:6] + alpha * ang_delta
    return out


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

@dataclass
class RecordedFrame:
    sim_pose6: np.ndarray
    gripper: float
    joint6: float
    main_image: np.ndarray
    wrist_image: np.ndarray
    timestamp: float


def execute_and_record(
    daemon,
    cameras: CameraPair,
    target_real: np.ndarray,
    gripper: float,
    start_frame_idx: int,
    speed_mps: float,
    timeout_s: float = DEFAULT_ASYNC_MOVE_TIMEOUT_S,
) -> list[RecordedFrame]:
    """Execute one async movel segment and record raw state at 30Hz."""
    from support.pose_align import real_pose_to_sim
    from support.tcp_control import get_robot_snapshot

    target_real = np.asarray(target_real, dtype=np.float64).reshape(6).copy()
    frames: list[RecordedFrame] = []
    start_resp = start_async_movel(
        daemon, target_real, f"recorded movel frame {start_frame_idx}", speed_mps=speed_mps,
    )
    saw_active = _coerce_exec_id(start_resp.get("exec_id")) != -1
    deadline = time.monotonic() + max(0.5, timeout_s)
    frame_idx = start_frame_idx

    while True:
        tick_start = time.monotonic()
        status = daemon.motion_status()
        tcp_pose = np.asarray(status.get("tcp_pose", []), dtype=np.float64)
        joint_q = np.asarray(status.get("joint_q_rad", []), dtype=np.float64)
        if tcp_pose.size == 6:
            actual_real = tcp_pose.reshape(6).copy()
        else:
            snap = get_robot_snapshot()
            actual_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
            joint_q = np.asarray(snap.joint_q, dtype=np.float64)
        actual_sim = real_pose_to_sim(actual_real)
        joint6 = float(joint_q[5]) if joint_q.size >= 6 else 0.0
        main_bgr, wrist_bgr = cameras.grab()
        main_img = preprocess_image(main_bgr)
        wrist_img = preprocess_image(wrist_bgr)
        timestamp = frame_idx * CONTROL_DT

        frames.append(RecordedFrame(
            sim_pose6=actual_sim, gripper=gripper, joint6=joint6,
            main_image=main_img, wrist_image=wrist_img, timestamp=timestamp,
        ))

        exec_id = _coerce_exec_id(status.get("exec_id"))
        is_steady = _coerce_bool(status.get("is_steady"))
        if exec_id != -1:
            saw_active = True

        done = (saw_active and exec_id == -1 and is_steady) or (is_steady and _pose_close(actual_real, target_real))
        if done:
            break

        if time.monotonic() > deadline:
            stop_motion_and_confirm(daemon, f"recorded movel frame {start_frame_idx} timeout")
            raise RuntimeError(f"recorded movel timed out after {timeout_s:.1f}s")

        frame_idx += 1
        remaining = CONTROL_DT - (time.monotonic() - tick_start)
        if remaining > 0:
            time.sleep(remaining)

    wait_resp = daemon.wait_motion_done()
    wait_ret_raw = wait_resp.get("wait_ret")
    wait_ret = int(wait_ret_raw) if wait_ret_raw is not None else None
    if wait_ret is not None and wait_ret != 0:
        stop_motion_and_confirm(daemon, f"recorded movel frame {start_frame_idx} wait failure")
        raise RuntimeError(f"recorded movel wait failed: {wait_resp}")
    return frames


def execute_servo_pose_j6_and_record(
    daemon,
    cameras: CameraPair,
    target_real: np.ndarray,
    target_joint6: float,
    gripper: float,
    start_frame_idx: int,
    linear_speed_mps: float,
    *,
    j6_speed_radps: float = DEFAULT_J6_SPEED_RADPS,
    control_dt_s: float = CONTROL_DT,
) -> list[RecordedFrame]:
    """Interpolate TCP pose + absolute j6 target in servo mode and record each step."""
    from support.pose_align import real_pose_to_sim
    from support.tcp_control import get_robot_snapshot

    snap = get_robot_snapshot()
    start_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
    start_joint6 = float(snap.joint_q[5]) if snap.joint_q.size >= 6 else 0.0
    target_real = np.asarray(target_real, dtype=np.float64).reshape(6).copy()
    joint6_delta = _wrap_angle(float(target_joint6) - start_joint6)

    linear_dist = float(np.linalg.norm(target_real[:3] - start_real[:3]))
    linear_time = linear_dist / max(float(linear_speed_mps), 1e-6)
    joint6_time = abs(float(joint6_delta)) / max(float(j6_speed_radps), 1e-6)
    duration_s = max(float(control_dt_s), linear_time, joint6_time)
    # Use ceil so the commanded average j6 speed never exceeds the configured limit.
    step_count = max(1, int(np.ceil(duration_s / float(control_dt_s))))

    frames: list[RecordedFrame] = []
    frame_idx = start_frame_idx
    start_resp = daemon.servo_start(control_dt_s)
    if int(start_resp.get("servo_start_ret", -1)) != 0:
        raise RuntimeError(f"servo_start failed: {start_resp}")

    try:
        for step_idx in range(step_count):
            alpha = float(step_idx + 1) / float(step_count)
            pose_i = interpolate_pose6(start_real, target_real, alpha)
            joint6_i = _wrap_angle(start_joint6 + alpha * joint6_delta)
            resp = daemon.servo_pose_j6(pose_i, joint6_i)
            pose_ret = int(resp.get("servo_pose_ret", -1))
            if pose_ret != 0:
                raise RuntimeError(f"servo_pose_j6 failed: {resp}")

            actual_real = np.asarray(resp.get("target_pose", []), dtype=np.float64)
            if actual_real.size != 6:
                actual_real = pose_i.copy()
            else:
                actual_real = actual_real.reshape(6).copy()

            target_q = np.asarray(resp.get("target_q_rad", []), dtype=np.float64)
            actual_joint6 = float(target_q[5]) if target_q.size >= 6 else float(joint6_i)
            actual_sim = real_pose_to_sim(actual_real)
            main_bgr, wrist_bgr = cameras.grab()
            frames.append(
                RecordedFrame(
                    sim_pose6=actual_sim,
                    gripper=gripper,
                    joint6=actual_joint6,
                    main_image=preprocess_image(main_bgr),
                    wrist_image=preprocess_image(wrist_bgr),
                    timestamp=frame_idx * CONTROL_DT,
                )
            )
            frame_idx += 1
    finally:
        try:
            daemon.servo_stop()
        except Exception:
            pass

    return frames


def execute_joint6_rotation_and_record(
    cameras: CameraPair,
    target_joint6: float,
    gripper: float,
    start_frame_idx: int,
    *,
    speed_deg: float = DEFAULT_J6_MOVE_SPEED_DEG,
    acc_deg: float = DEFAULT_J6_MOVE_ACC_DEG,
    timeout_s: float = DEFAULT_ASYNC_MOVE_TIMEOUT_S,
    joint6_tol_rad: float = float(np.deg2rad(1.0)),
) -> list[RecordedFrame]:
    """Rotate joint 6 in joint space while recording true robot observations at 30Hz."""
    from support.joint_control import move_to_joint_positions
    from support.pose_align import real_pose_to_sim
    from support.tcp_control import get_robot_snapshot

    start_snap = get_robot_snapshot()
    target_q = np.asarray(start_snap.joint_q, dtype=np.float64).reshape(6).copy()
    target_q[5] = float(target_joint6)

    outcome: dict[str, object] = {"result": None, "error": None}

    def _worker() -> None:
        try:
            outcome["result"] = move_to_joint_positions(
                target_q,
                execute=True,
                speed_deg=speed_deg,
                acc_deg=acc_deg,
            )
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    frames: list[RecordedFrame] = []
    frame_idx = start_frame_idx
    deadline = time.monotonic() + max(0.5, timeout_s)

    while True:
        tick_start = time.monotonic()
        snap = get_robot_snapshot()
        actual_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
        joint_q = np.asarray(snap.joint_q, dtype=np.float64).reshape(-1)
        actual_joint6 = float(joint_q[5]) if joint_q.size >= 6 else float(target_joint6)
        actual_sim = real_pose_to_sim(actual_real)
        main_bgr, wrist_bgr = cameras.grab()
        frames.append(
            RecordedFrame(
                sim_pose6=actual_sim,
                gripper=gripper,
                joint6=actual_joint6,
                main_image=preprocess_image(main_bgr),
                wrist_image=preprocess_image(wrist_bgr),
                timestamp=frame_idx * CONTROL_DT,
            )
        )

        done = (not worker.is_alive()) and abs(_wrap_angle(actual_joint6 - float(target_joint6))) <= joint6_tol_rad
        if done:
            break
        if outcome["error"] is not None:
            raise RuntimeError(f"joint6 rotation failed: {outcome['error']}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"joint6 rotation timed out after {timeout_s:.1f}s "
                f"(target_joint6={float(target_joint6):.6f}, actual_joint6={actual_joint6:.6f})"
            )

        frame_idx += 1
        remaining = CONTROL_DT - (time.monotonic() - tick_start)
        if remaining > 0:
            time.sleep(remaining)

    worker.join(timeout=0.1)
    if outcome["error"] is not None:
        raise RuntimeError(f"joint6 rotation failed: {outcome['error']}")
    result = outcome["result"]
    if result is None or not result.ok:
        reason = "unknown"
        if result is not None:
            reason = result.reason
        raise RuntimeError(f"joint6 rotation failed: {reason}")

    return frames


def record_snapshot_frame(
    daemon,
    cameras: CameraPair,
    gripper: float,
    frame_idx: int,
) -> RecordedFrame:
    """Capture a single snapshot frame (used for gripper-only events)."""
    from support.pose_align import real_pose_to_sim
    from support.tcp_control import get_robot_snapshot

    snap = get_robot_snapshot()
    actual_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
    joint_q = np.asarray(snap.joint_q, dtype=np.float64)
    actual_sim = real_pose_to_sim(actual_real)
    joint6 = float(joint_q[5]) if joint_q.size >= 6 else 0.0
    main_bgr, wrist_bgr = cameras.grab()
    main_img = preprocess_image(main_bgr)
    wrist_img = preprocess_image(wrist_bgr)
    return RecordedFrame(
        sim_pose6=actual_sim,
        gripper=gripper,
        joint6=joint6,
        main_image=main_img,
        wrist_image=wrist_img,
        timestamp=frame_idx * CONTROL_DT,
    )


# ---------------------------------------------------------------------------
# Resampling / saving  (same logic as collect_data_cube)
# ---------------------------------------------------------------------------

def resample_episode_to_50hz(
    frames: list[RecordedFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample raw 30Hz episode frames onto a uniform 50Hz grid."""
    if not frames:
        raise RuntimeError("no frames to resample")

    raw_times = np.array([float(f.timestamp) for f in frames], dtype=np.float64)
    raw_pose6 = np.stack([np.asarray(f.sim_pose6, dtype=np.float64).reshape(6) for f in frames], axis=0)
    raw_gripper = np.array([float(f.gripper) for f in frames], dtype=np.float32)
    raw_joint6 = np.array([float(f.joint6) for f in frames], dtype=np.float32)
    raw_main_images = np.stack([f.main_image for f in frames], axis=0)
    raw_wrist_images = np.stack([f.wrist_image for f in frames], axis=0)

    target_count = max(1, int(round((len(frames) - 1) * SAVE_FPS / RAW_CAPTURE_FPS)) + 1)
    env_steps = np.arange(target_count, dtype=np.int64)
    target_times = env_steps.astype(np.float32) * np.float32(SAVE_CONTROL_DT)

    states = np.zeros((target_count, 8), dtype=np.float32)
    pose6_resampled = np.zeros((target_count, 6), dtype=np.float32)
    gripper_resampled = np.zeros((target_count,), dtype=np.float32)
    joint6_resampled = np.zeros((target_count,), dtype=np.float32)
    main_images = np.zeros((target_count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    wrist_images = np.zeros((target_count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for out_idx, target_t in enumerate(target_times.astype(np.float64)):
        right = int(np.searchsorted(raw_times, target_t, side="left"))
        if right <= 0:
            left = 0; right = 0; alpha = 0.0
        elif right >= len(raw_times):
            left = len(raw_times) - 1; right = left; alpha = 0.0
        else:
            left = right - 1
            dt = float(raw_times[right] - raw_times[left])
            alpha = 0.0 if dt <= 1e-9 else float((target_t - raw_times[left]) / dt)
            alpha = float(np.clip(alpha, 0.0, 1.0))

        if left == right:
            interp_pose6 = raw_pose6[left].copy()
            image_idx = left
        else:
            interp_pose6 = (1.0 - alpha) * raw_pose6[left] + alpha * raw_pose6[right]
            angle_delta = np.arctan2(
                np.sin(raw_pose6[right, 3:6] - raw_pose6[left, 3:6]),
                np.cos(raw_pose6[right, 3:6] - raw_pose6[left, 3:6]),
            )
            interp_pose6[3:6] = raw_pose6[left, 3:6] + alpha * angle_delta
            image_idx = left if abs(target_t - raw_times[left]) <= abs(raw_times[right] - target_t) else right

        grip_idx = int(np.searchsorted(raw_times, target_t, side="right") - 1)
        grip_idx = min(max(grip_idx, 0), len(raw_gripper) - 1)
        if left == right:
            interp_joint6 = float(raw_joint6[left])
        else:
            joint6_delta = _wrap_angle(float(raw_joint6[right] - raw_joint6[left]))
            interp_joint6 = float(raw_joint6[left] + alpha * joint6_delta)

        pose6_resampled[out_idx] = interp_pose6.astype(np.float32)
        gripper_resampled[out_idx] = float(raw_gripper[grip_idx])
        joint6_resampled[out_idx] = np.float32(interp_joint6)
        states[out_idx] = build_state8(interp_pose6, gripper_resampled[out_idx], joint6_resampled[out_idx])
        main_images[out_idx] = raw_main_images[image_idx]
        wrist_images[out_idx] = raw_wrist_images[image_idx]

    actions = compute_delta_actions(pose6_resampled, gripper_resampled, joint6_resampled)
    return states, actions, target_times.astype(np.float32), env_steps, main_images, wrist_images


def save_episode(frames: list[RecordedFrame], save_dir: Path, prompt: str) -> Path:
    """Save a recorded episode to disk."""
    from support.pose_align import get_alignment_mode

    existing = list(save_dir.glob("episode_*"))
    ids = []
    for directory in existing:
        try:
            ids.append(int(directory.name.split("_")[1]))
        except Exception:
            pass
    episode_id = 0 if not ids else (max(ids) + 1)
    episode_dir = save_dir / f"episode_{episode_id:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    states, actions, timestamps, env_steps, main_images, wrist_images = resample_episode_to_50hz(frames)

    np.save(episode_dir / "states.npy", states)
    np.save(episode_dir / "actions.npy", actions)
    np.save(episode_dir / "timestamps.npy", timestamps)
    np.save(episode_dir / "env_steps.npy", env_steps)
    np.savez_compressed(
        episode_dir / "images.npz",
        main_images=main_images, wrist_images=wrist_images,
    )

    metadata = {
        "task": prompt,
        "fps": BASE_FPS,
        "nominal_fps": float(BASE_FPS),
        "n_frames": int(states.shape[0]),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "record_every": 1,
        "base_fps": BASE_FPS,
        "image_format": "npz",
        "sampling_mode": "global_env_step_with_event_frames",
        "timestamp_mode": "env_step_times_control_dt",
        "timestamps_file": "timestamps.npy",
        "env_steps_file": "env_steps.npy",
        "state_dim": 8,
        "action_dim": 7,
        "state_schema": ["x", "y", "z", "aa_x", "aa_y", "aa_z", "gripper_open", "j6"],
        "action_schema": ["dx", "dy", "dz", "droll", "dpitch", "dj6", "gripper_next"],
        "pose_frame": get_alignment_mode(),
    }
    with open(episode_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"  Episode {episode_id} saved: {episode_dir} (raw {len(frames)} -> saved {states.shape[0]} frames)")
    return episode_dir


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _state_file_path(data_dir: Path) -> Path:
    return data_dir / STATE_FILE_NAME


def save_collect_state(data_dir: Path, episode_count: int) -> None:
    payload = {"episode_count": int(episode_count)}
    with open(_state_file_path(data_dir), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_collect_state(data_dir: Path) -> dict | None:
    path = _state_file_path(data_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return {"episode_count": int(payload.get("episode_count", 0))}
    except Exception:
        return None


def clear_collect_state(data_dir: Path) -> None:
    path = _state_file_path(data_dir)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-robot data collection for storage-box tasks.")
    parser.add_argument("--data-dir", type=str, default=str(SCRIPT_DIR / "collected_data" / "storage"))
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--speed", type=float, default=LINEAR_SPEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gripper-timeout", type=float, default=10.0)
    parser.add_argument("--pose-frame", type=str, choices=("sim", "real"), default="real")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _maybe_reexec_into_repo_venv()

    from support.gripper_control import command_gripper_state
    from support.joint_control import build_joint_helper, move_to_joint_positions
    from support.pose_align import (
        REAL_INIT_QPOS_RAD,
        get_alignment_mode,
        real_pose_to_sim,
        set_alignment_mode,
        set_runtime_alignment,
    )
    from support.tcp_control import build_helper as build_tcp_helper, get_robot_snapshot, _get_motion_daemon

    args = parse_args()
    set_alignment_mode(args.pose_frame)
    save_dir = Path(args.data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global LINEAR_SPEED
    if args.speed <= 0:
        raise ValueError(f"--speed must be positive, got {args.speed}")
    LINEAR_SPEED = args.speed

    prompt = "open the storage box"

    # --- Interactive mode selection ---
    print(f"\nTask: {prompt}")
    print("\nSelect mode:")
    print("  [m] manual - press ENTER each episode, q to quit")
    print("  [N] auto   - enter a number to auto-collect N episodes")
    mode_choice = input("Enter m or a number (default: m): ").strip().lower()
    auto_episodes = 0
    if mode_choice not in ("", "m", "manual"):
        try:
            auto_episodes = int(mode_choice)
            if auto_episodes <= 0:
                raise ValueError
            print(f"Auto mode: will collect {auto_episodes} episodes without pause.")
        except ValueError:
            print(f"Invalid input '{mode_choice}', using manual mode.")
            auto_episodes = 0
    if auto_episodes == 0:
        print("Manual mode: press ENTER each episode.")

    print(f"\nData dir: {save_dir}")
    print(f"Raw capture FPS: {RAW_CAPTURE_FPS}")
    print(f"Saved dataset FPS: {BASE_FPS}")
    print(f"Speed: {LINEAR_SPEED} m/s")
    print(f"Dry-run: {args.dry_run}")
    print(f"Pose frame: {get_alignment_mode()}")

    print("Compiling C++ helpers...")
    build_joint_helper()
    build_tcp_helper()

    print("Initializing cameras...")
    cameras = CameraPair()
    daemon = _get_motion_daemon()

    print("Moving to initial joint position...")
    ensure_joint_move_ok(
        move_to_joint_positions(REAL_INIT_QPOS_RAD, execute=not args.dry_run),
        "initial joint alignment",
    )
    print("  Joint alignment OK.")

    snap = get_robot_snapshot()
    set_runtime_alignment(snap.tcp_pose, frame_mode=args.pose_frame)
    init_real = snap.tcp_pose.copy()
    home_real = init_real.copy()
    print(f"  Init real TCP: {np.round(init_real, 5).tolist()}")

    if not args.dry_run:
        print("Opening gripper...")
        ensure_gripper_ok(
            command_gripper_state(1, timeout_s=args.gripper_timeout),
            "open gripper before collection",
        )

    # Waypoint heights
    z_grasp = OPEN_BOX_GRASP_Z
    z_home = float(home_real[2])  # home TCP height (safe cruising altitude)
    z_lift = z_grasp + OPEN_BOX_LIFT_CM / 100.0  # 5 cm above grasp for final lift

    episode_count = 0
    saved = load_collect_state(save_dir)
    if saved is not None:
        episode_count = saved["episode_count"]
        print(f"\n  Resuming from saved state: episode_count={episode_count}")

    def refresh_home_pose() -> np.ndarray:
        nonlocal home_real
        snap_local = get_robot_snapshot()
        set_runtime_alignment(snap_local.tcp_pose)
        home_real = np.asarray(snap_local.tcp_pose, dtype=np.float64).reshape(6).copy()
        return home_real.copy()

    def return_home(label: str) -> np.ndarray:
        nonlocal home_real
        if not args.dry_run:
            ensure_joint_move_ok(
                move_to_joint_positions(REAL_INIT_QPOS_RAD, execute=True),
                label,
            )
            return refresh_home_pose()
        return home_real.copy()

    max_ep = auto_episodes if auto_episodes > 0 else args.max_episodes

    print("\n=== Ready For Episodes ===")
    if auto_episodes > 0:
        print(f"Auto mode: {auto_episodes} episodes.")
    else:
        print("Press ENTER to record, q to quit.")
    print()

    try:
        while True:
            if 0 < max_ep <= episode_count:
                print(f"\nReached {episode_count} episodes. Done.")
                break

            episode_label = f"Episode {episode_count} [open storage box]"
            print(f"\n--- {episode_label} ---")

            if auto_episodes > 0:
                print(f"  Auto: recording '{prompt}' ({episode_count + 1}/{auto_episodes})")
            else:
                cmd = input(f"  Press ENTER to record '{prompt}', or q to quit: ").strip().lower()
                if cmd in QUIT_TOKENS:
                    print("Exit requested.")
                    break
                if cmd:
                    print(f"Unsupported input '{cmd}', exiting.")
                    break

            # === NOT recorded: return home + open gripper ===
            return_home("pre-episode return home")
            if False and not args.dry_run:
                ensure_gripper_ok(
                    command_gripper_state(1, timeout_s=args.gripper_timeout),
                    "open gripper before episode",
                )

            all_frames: list[RecordedFrame] = []
            frame_idx = 0

            gx, gy = OPEN_BOX_GRASP_XY
            rx, ry = OPEN_BOX_RELEASE_XY

            if not args.dry_run:
                grasp_xy_at_home_z = build_pose_at_xy(home_real, gx, gy, z_home)

                # 1) home -> grasp XY at home height
                print(f"    [rec] move to grasp XY ({gx}, {gy}) at home z={z_home:.4f}")
                seg = execute_and_record(
                    daemon, cameras, grasp_xy_at_home_z, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # 2) lower to grasp height
                snap_before_lower = get_robot_snapshot()
                grasp_base_real = np.asarray(snap_before_lower.tcp_pose, dtype=np.float64).reshape(6).copy()
                grasp_down_real = build_pose_at_xy(grasp_base_real, gx, gy, z_grasp)
                print(f"    [rec] lower to z={z_grasp}")
                seg = execute_and_record(
                    daemon, cameras, grasp_down_real, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # 3) move to release XY while preserving current orientation
                snap_after_lower = get_robot_snapshot()
                release_base_real = np.asarray(snap_after_lower.tcp_pose, dtype=np.float64).reshape(6).copy()
                release_pose_real = build_pose_at_xy(release_base_real, rx, ry, z_grasp)
                release_lift_real = build_pose_at_xy(release_base_real, rx, ry, z_lift)
                print(f"    [rec] move to release ({rx}, {ry}) at z={z_grasp}")
                seg = execute_and_record(
                    daemon, cameras, release_pose_real, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # 4) lift 5 cm
                print(f"    [rec] lift {OPEN_BOX_LIFT_CM} cm")
                seg = execute_and_record(
                    daemon, cameras, release_lift_real, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

            # waypoints: 水平移动保持 home 高度，垂直下压单独一步
            grasp_xy_at_home_z = build_pose_at_xy(home_real, gx, gy, z_home)
            grasp_down_real    = build_pose_at_xy(home_real, gx, gy, z_grasp)
            release_pose_real  = build_pose_at_xy(home_real, rx, ry, z_grasp)
            release_lift_real  = build_pose_at_xy(home_real, rx, ry, z_lift)

            if False and not args.dry_run:
                # === START RECORDING ===

                # 1) home -> grasp XY at home height (水平移动)
                print(f"    [rec] move to grasp XY ({gx}, {gy}) at home z={z_home:.4f}")
                seg = execute_and_record(
                    daemon, cameras, grasp_xy_at_home_z, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # 2) lower to grasp height (垂直下压)
                print(f"    [rec] lower to z={z_grasp}")
                seg = execute_and_record(
                    daemon, cameras, grasp_down_real, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # 3) close gripper (snapshot 1 frame)
                print("    [rec] close gripper (snapshot)")
                ensure_gripper_ok(
                    command_gripper_state(0, timeout_s=args.gripper_timeout),
                    "close gripper on storage box lid",
                )
                snap_frame = record_snapshot_frame(daemon, cameras, gripper=0.0, frame_idx=frame_idx)
                all_frames.append(snap_frame); frame_idx += 1

                # 4) move to release position (拉开盖子, 保持 z_grasp)
                print(f"    [rec] move to release ({rx}, {ry}) at z={z_grasp}")
                seg = execute_and_record(
                    daemon, cameras, release_pose_real, gripper=0.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # 5) open gripper (snapshot 1 frame)
                print("    [rec] open gripper (release)")
                ensure_gripper_ok(
                    command_gripper_state(1, timeout_s=args.gripper_timeout),
                    "open gripper to release lid",
                )
                snap_frame = record_snapshot_frame(daemon, cameras, gripper=1.0, frame_idx=frame_idx)
                all_frames.append(snap_frame); frame_idx += 1

                # 6) lift 5 cm
                print(f"    [rec] lift {OPEN_BOX_LIFT_CM} cm")
                seg = execute_and_record(
                    daemon, cameras, release_lift_real, gripper=1.0,
                    start_frame_idx=frame_idx, speed_mps=LINEAR_SPEED,
                )
                all_frames.extend(seg); frame_idx += len(seg)

                # === END RECORDING ===
            elif args.dry_run:
                # dry-run: generate placeholder frames
                start_sim = real_pose_to_sim(home_real)
                n_total = 30  # placeholder
                for idx in range(n_total):
                    all_frames.append(RecordedFrame(
                        sim_pose6=start_sim.copy(), gripper=1.0, joint6=0.0,
                        main_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        wrist_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        timestamp=idx * CONTROL_DT,
                    ))

            save_episode(all_frames, save_dir, prompt)
            episode_count += 1

            # return home after recording
            return_home("post-episode return home")

            save_collect_state(save_dir, episode_count)

    except KeyboardInterrupt:
        print(f"\n\nCollection stopped after {episode_count} episodes.")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        save_collect_state(save_dir, episode_count)
        try:
            stop_motion_and_confirm(daemon, "collect_data_storage cleanup")
        except Exception:
            pass
        try:
            daemon.stop()
        except Exception:
            pass
        cameras.stop()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

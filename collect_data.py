#!/usr/bin/env python3
"""
Real-robot data collection script for OpenPI pick-up training.

Workflow (each episode):
  1. [NOT recorded] Preparation phase: seed red/green/blue cubes from origin to random non-overlapping (x, y)
  2. [RECORDED raw @ 30Hz, saved @ 30Hz or optionally resampled to 50Hz] Pick one cube in red->green->blue order:
     move above, lower, grasp, lift
  3. [NOT recorded] Randomly place the held cube back into the workspace, then return home
  4. Save the recorded episode with auto-generated prompt "pick up the <color> cube"

Saved format for the AUBO OpenPI pipeline:
  states.npy     (N, 7|8) float32  yaw: [x, y, z, aa_x, aa_y, aa_z, gripper]
                                   j6 : [x, y, z, aa_x, aa_y, aa_z, gripper, j6]
  actions.npy    (N, 7) float32    yaw: [dx, dy, dz, droll, dpitch, dyaw, gripper_next]
                                   j6 : [dx, dy, dz, droll, dpitch, dj6, gripper_next]
  timestamps.npy (N,)   float32  env_step / saved_fps
  env_steps.npy  (N,)   int64
  images.npz             contains main_images (N,224,224,3) and wrist_images (N,224,224,3)
  metadata.json
"""

from __future__ import annotations

import argparse
import atexit
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

# Workspace bounds (real robot TCP frame)
WORKSPACE_X_MIN = 0.34
WORKSPACE_X_MAX = 0.66
WORKSPACE_Y_MIN = -0.26
WORKSPACE_Y_MAX = 0.26

# Heights
GRASP_HEIGHT_MM = 180.0
MIN_TCP_Z = GRASP_HEIGHT_MM / 1000.0
APPROACH_Z_OFFSET_M = 0.20
PLACE_HEIGHT_OFFSET_M = 0.0

# Object placement
CUBE_HEIGHT_M = 0.05
MIN_CUBE_SPACING_M = 0.10
MAX_XY_SAMPLE_ATTEMPTS = 512
QUIT_TOKENS = {"q", "quit", "exit"}
STATE_FILE_NAME = ".collect_state.json"

# Data collection parameters
RAW_CAPTURE_FPS = 30
CONTROL_DT = 1.0 / RAW_CAPTURE_FPS
IMAGE_SIZE = 224
DEFAULT_SAVE_FPS = RAW_CAPTURE_FPS
UPSAMPLED_SAVE_FPS = 50

# Motion speed (Cartesian linear velocity, m/s)
LINEAR_SPEED = 0.10
DEFAULT_ASYNC_MOVE_TIMEOUT_S = 30.0
DEFAULT_J6_SPEED_RADPS = float(np.deg2rad(10.0))
DEFAULT_J6_MOVE_SPEED_DEG = 10.0
DEFAULT_J6_MOVE_ACC_DEG = 20.0
DEFAULT_J6_HOME_RAD = 0.11434
DEFAULT_J6_EXEC_TOL_RAD = float(np.deg2rad(1.0))
ROTATE_ABS_DEG_MIN = 12.0
ROTATE_ABS_DEG_MAX = 22.5


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


from support.get_obs import (
    CameraPair,
    STATE_MODE_J6,
    STATE_MODE_YAW,
    preprocess_image_for_openpi as preprocess_image,
)
from task.pick_and_place import (
    APPLE_NAME,
    OBJECT_ORDER,
    PlannerConfig,
    TaskStep,
    build_random_episode_plan,
    load_scene_state,
)
from task.open_and_close import (
    MoveStep as OpenCloseMoveStep,
    OpenCloseReference,
    build_open_and_close_episode_plan,
    build_reference_from_tcp_pose,
)


# ---------------------------------------------------------------------------
# Pose / state helpers
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


def _normalize_state_mode(state_mode: str) -> str:
    mode = str(state_mode).strip().lower()
    if mode not in (STATE_MODE_YAW, STATE_MODE_J6):
        raise ValueError(f"invalid state_mode={state_mode!r}, expected '{STATE_MODE_YAW}' or '{STATE_MODE_J6}'")
    return mode


def build_state_row(pose6_zyx: np.ndarray, gripper: float, joint6: float, *, state_mode: str) -> np.ndarray:
    """Build one state row for the selected collect-data mode."""
    mode = _normalize_state_mode(state_mode)
    pose = np.asarray(pose6_zyx, dtype=np.float64).reshape(6)
    quat = _euler_zyx_to_quat_wxyz(pose[3:6])
    aa = _quat_to_axis_angle_wxyz(quat)
    if mode == STATE_MODE_J6:
        return np.array(
            [
                pose[0],
                pose[1],
                pose[2],
                aa[0],
                aa[1],
                aa[2],
                float(gripper),
                float(joint6),
            ],
            dtype=np.float32,
        )
    return np.array(
        [
            pose[0],
            pose[1],
            pose[2],
            aa[0],
            aa[1],
            aa[2],
            float(gripper),
        ],
        dtype=np.float32,
    )


def quat_to_euler_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def compute_delta_actions(
    pose6: np.ndarray,
    gripper: np.ndarray,
    joint6: np.ndarray,
    *,
    state_mode: str,
) -> np.ndarray:
    """Compute (N, 7) delta actions for yaw-mode or j6-mode datasets."""
    mode = _normalize_state_mode(state_mode)
    pose6 = np.asarray(pose6, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    joint6 = np.asarray(joint6, dtype=np.float32).reshape(-1)
    if pose6.ndim != 2 or pose6.shape[1] != 6:
        raise ValueError(f"pose6 must have shape (N, 6), got {pose6.shape}")
    if gripper.shape[0] != pose6.shape[0] or joint6.shape[0] != pose6.shape[0]:
        raise ValueError("pose6, gripper, and joint6 must have the same length")

    num_frames = int(pose6.shape[0])
    actions = np.zeros((num_frames, 7), dtype=np.float32)
    for idx in range(num_frames):
        if idx < num_frames - 1:
            pos_curr = pose6[idx, :3]
            pos_next = pose6[idx + 1, :3]
            actions[idx, :3] = pos_next - pos_curr

            euler_curr = pose6[idx, 3:6]
            euler_next = pose6[idx + 1, 3:6]
            deuler = np.arctan2(np.sin(euler_next - euler_curr), np.cos(euler_next - euler_curr))
            actions[idx, 3:5] = deuler[:2]
            if mode == STATE_MODE_J6:
                actions[idx, 5] = _wrap_angle(float(joint6[idx + 1] - joint6[idx]))
            else:
                actions[idx, 5] = deuler[2]
            actions[idx, 6] = gripper[idx + 1]
        else:
            actions[idx, :6] = 0.0
            actions[idx, 6] = gripper[idx]
    return actions


def state_schema_for_mode(state_mode: str) -> list[str]:
    mode = _normalize_state_mode(state_mode)
    if mode == STATE_MODE_J6:
        return ["x", "y", "z", "aa_x", "aa_y", "aa_z", "gripper_open", "j6"]
    return ["x", "y", "z", "aa_x", "aa_y", "aa_z", "gripper_open"]


def action_schema_for_mode(state_mode: str) -> list[str]:
    mode = _normalize_state_mode(state_mode)
    if mode == STATE_MODE_J6:
        return ["dx", "dy", "dz", "droll", "dpitch", "dj6", "gripper_next"]
    return ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_next"]


def _canonicalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] > 0.0:
        q = -q
    return q


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    qa = _canonicalize_quat_wxyz(q0)
    qb = _canonicalize_quat_wxyz(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot

    if dot > 0.9995:
        out = qa + alpha * (qb - qa)
        out /= max(np.linalg.norm(out), 1e-12)
        return out

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * float(alpha)
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta)) / max(sin_theta_0, 1e-12)
    s1 = sin_theta / max(sin_theta_0, 1e-12)
    out = s0 * qa + s1 * qb
    out /= max(np.linalg.norm(out), 1e-12)
    return out


# ---------------------------------------------------------------------------
# Trajectory planning
# ---------------------------------------------------------------------------

def _state_file_path(data_dir: Path) -> Path:
    return data_dir / STATE_FILE_NAME


def _normalize_xy(xy: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64).reshape(2).copy()


def _normalize_j6(j6: float) -> float:
    return _wrap_angle(float(j6))


def get_object_xy(scene_state: dict[str, dict[str, object]], name: str) -> np.ndarray:
    return _normalize_xy(scene_state[name]["xy"])


def get_object_is_rotate(scene_state: dict[str, dict[str, object]], name: str) -> bool:
    return bool(scene_state[name]["is_rotate"])


def get_object_deg(scene_state: dict[str, dict[str, object]], name: str) -> float:
    return float(scene_state[name]["deg"])


def get_object_standard_j6_rad(scene_state: dict[str, dict[str, object]], name: str) -> float | None:
    raw = scene_state.get(name, {}).get("standard_j6_rad")
    if raw is None:
        return None
    try:
        return _normalize_j6(float(raw))
    except (TypeError, ValueError):
        return None


def set_object_standard_j6_rad(
    scene_state: dict[str, dict[str, object]],
    name: str,
    joint6_rad: float | None,
) -> None:
    state = scene_state.get(name)
    if not isinstance(state, dict):
        return
    if name == APPLE_NAME or not bool(state.get("is_rotate", False)) or joint6_rad is None:
        state["standard_j6_rad"] = None
        return
    state["standard_j6_rad"] = float(_normalize_j6(joint6_rad))


def clone_scene_state(scene_state: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    cloned = load_scene_state(scene_state)
    if cloned is None:
        raise RuntimeError("invalid scene_state payload")
    return cloned


def _normalize_held_object(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in OBJECT_ORDER:
        return text
    return None


def save_collect_state(
    data_dir: Path,
    scene_state: dict[str, dict[str, object]],
    color_index: int,
    episode_count: int,
    *,
    held_object: str | None = None,
    open_close_reference: OpenCloseReference | None = None,
) -> None:
    payload = {
        "object_states": scene_state,
        "color_index": int(color_index),
        "episode_count": int(episode_count),
        "held_object": _normalize_held_object(held_object),
    }
    if open_close_reference is not None:
        payload["open_close_reference"] = {
            "x_start": float(open_close_reference.x_start),
            "y_start": float(open_close_reference.y_start),
            "z_base": float(open_close_reference.z_base),
            "reference_pose6": [
                float(v)
                for v in np.asarray(open_close_reference.reference_pose6, dtype=np.float64).reshape(6).tolist()
            ],
        }
    path = _state_file_path(data_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_state_counter(payload: dict, key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def load_collect_state(data_dir: Path) -> dict | None:
    path = _state_file_path(data_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return None
        scene_payload = payload.get("object_states")
        if scene_payload is None:
            scene_payload = payload
        scene_state = load_scene_state(scene_payload)
        open_close_reference: OpenCloseReference | None = None
        open_close_payload = payload.get("open_close_reference")
        if isinstance(open_close_payload, dict):
            try:
                ref_pose = np.asarray(open_close_payload.get("reference_pose6", []), dtype=np.float64).reshape(6).copy()
                open_close_reference = OpenCloseReference(
                    x_start=float(open_close_payload.get("x_start", ref_pose[0])),
                    y_start=float(open_close_payload.get("y_start", ref_pose[1])),
                    z_base=float(open_close_payload.get("z_base", ref_pose[2])),
                    reference_pose6=ref_pose,
                )
            except Exception:
                open_close_reference = None
        color_index = _load_state_counter(payload, "color_index")
        episode_count = _load_state_counter(payload, "episode_count")
        held_object = _normalize_held_object(payload.get("held_object"))
        has_valid_counters = color_index is not None and episode_count is not None
        if scene_state is None and open_close_reference is None and not has_valid_counters:
            return None
        return {
            "scene_state": scene_state or {},
            "color_index": 0 if color_index is None else color_index,
            "episode_count": 0 if episode_count is None else episode_count,
            "held_object": held_object,
            "has_valid_counters": has_valid_counters,
            "open_close_reference": open_close_reference,
        }
    except Exception:
        return None


def clear_collect_state(data_dir: Path) -> None:
    path = _state_file_path(data_dir)
    if path.exists():
        path.unlink()


def build_pose_at_xy(base_pose: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float64).reshape(6).copy()
    pose[0] = float(x)
    pose[1] = float(y)
    pose[2] = float(z)
    return pose


def sample_non_overlapping_xy(
    occupied_xy: list[np.ndarray],
    *,
    min_dist: float = MIN_CUBE_SPACING_M,
    max_attempts: int = MAX_XY_SAMPLE_ATTEMPTS,
) -> np.ndarray:
    occupied = [np.asarray(xy, dtype=np.float64).reshape(2) for xy in occupied_xy]
    for _ in range(max_attempts):
        candidate = np.array(
            [
                np.random.uniform(WORKSPACE_X_MIN, WORKSPACE_X_MAX),
                np.random.uniform(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
            ],
            dtype=np.float64,
        )
        if all(float(np.linalg.norm(candidate - xy)) >= float(min_dist) for xy in occupied):
            return candidate
    raise RuntimeError(
        f"failed to sample non-overlapping xy after {max_attempts} attempts "
        f"(occupied={np.round(np.asarray(occupied, dtype=np.float64), 4).tolist() if occupied else []})"
    )


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


def start_async_movel(
    daemon,
    target_real: np.ndarray,
    label: str,
    *,
    speed_mps: float,
) -> dict[str, object]:
    resp = daemon.movel(target_real, speed_mps=speed_mps, blocking=False)
    ensure_movel_ok(resp, f"{label} start")
    return resp


def execute_movel_and_wait(
    daemon,
    target_real: np.ndarray,
    label: str,
    *,
    speed_mps: float,
) -> dict[str, object]:
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
    """Execute one async moveLine segment and record raw executed state at 30Hz."""
    from support.pose_align import real_pose_to_sim
    from support.tcp_control import get_robot_snapshot

    target_real = np.asarray(target_real, dtype=np.float64).reshape(6).copy()
    frames: list[RecordedFrame] = []
    start_resp = start_async_movel(
        daemon,
        target_real,
        f"recorded movel frame {start_frame_idx}",
        speed_mps=speed_mps,
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

        frames.append(
            RecordedFrame(
                sim_pose6=actual_sim,
                gripper=gripper,
                joint6=joint6,
                main_image=main_img,
                wrist_image=wrist_img,
                timestamp=timestamp,
            )
        )

        exec_id = _coerce_exec_id(status.get("exec_id"))
        is_steady = _coerce_bool(status.get("is_steady"))
        if exec_id != -1:
            saw_active = True

        done = (saw_active and exec_id == -1 and is_steady) or (is_steady and _pose_close(actual_real, target_real))
        if done:
            break

        if time.monotonic() > deadline:
            stop_resp = stop_motion_and_confirm(daemon, f"recorded movel frame {start_frame_idx} timeout")
            raise RuntimeError(
                f"recorded movel timed out after {timeout_s:.1f}s "
                f"(target={np.round(target_real, 5).tolist()}, stop={stop_resp})"
            )

        frame_idx += 1
        remaining = CONTROL_DT - (time.monotonic() - tick_start)
        if remaining > 0:
            time.sleep(remaining)

    wait_resp = daemon.wait_motion_done()
    wait_ret_raw = wait_resp.get("wait_ret")
    wait_ret = int(wait_ret_raw) if wait_ret_raw is not None else None
    if wait_ret is not None and wait_ret != 0:
        stop_resp = stop_motion_and_confirm(daemon, f"recorded movel frame {start_frame_idx} wait failure")
        raise RuntimeError(f"recorded movel wait failed: {wait_resp}, stop={stop_resp}")
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
) -> list[RecordedFrame]:
    """Rotate joint6 in joint space and record real observations at 30Hz."""
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

        done = not worker.is_alive()
        if done:
            break
        if outcome["error"] is not None:
            raise RuntimeError(f"joint6 rotation failed: {outcome['error']}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"joint6 rotation timed out after {timeout_s:.1f}s "
                f"(target_joint6={float(target_joint6):.6f}, last_joint6={actual_joint6:.6f})"
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


def execute_joint6_rotation(
    target_joint6: float,
    *,
    speed_deg: float = DEFAULT_J6_MOVE_SPEED_DEG,
    acc_deg: float = DEFAULT_J6_MOVE_ACC_DEG,
) -> None:
    """Rotate joint6 without recording."""
    from support.joint_control import move_to_joint_positions
    from support.tcp_control import get_robot_snapshot

    snap = get_robot_snapshot()
    target_q = np.asarray(snap.joint_q, dtype=np.float64).reshape(6).copy()
    target_q[5] = float(target_joint6)
    result = move_to_joint_positions(
        target_q,
        execute=True,
        speed_deg=speed_deg,
        acc_deg=acc_deg,
    )
    ensure_joint_move_ok(result, f"rotate joint6 to {float(target_joint6):.6f}rad")


def prepare_episode_for_save(
    frames: list[RecordedFrame],
    *,
    save_fps: int,
    state_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw episode frames onto the target save grid."""
    if not frames:
        raise RuntimeError("no frames to resample")
    mode = _normalize_state_mode(state_mode)
    if save_fps not in (RAW_CAPTURE_FPS, UPSAMPLED_SAVE_FPS):
        raise ValueError(f"unsupported save_fps={save_fps}, expected {RAW_CAPTURE_FPS} or {UPSAMPLED_SAVE_FPS}")

    raw_times = np.array([float(frame.timestamp) for frame in frames], dtype=np.float64)
    raw_pose6 = np.stack([np.asarray(frame.sim_pose6, dtype=np.float64).reshape(6) for frame in frames], axis=0)
    raw_gripper = np.array([float(frame.gripper) for frame in frames], dtype=np.float32)
    raw_joint6 = np.array([float(frame.joint6) for frame in frames], dtype=np.float32)
    raw_main_images = np.stack([frame.main_image for frame in frames], axis=0)
    raw_wrist_images = np.stack([frame.wrist_image for frame in frames], axis=0)

    if save_fps == RAW_CAPTURE_FPS:
        target_count = len(frames)
        env_steps = np.arange(target_count, dtype=np.int64)
        target_times = env_steps.astype(np.float32) * np.float32(1.0 / RAW_CAPTURE_FPS)
        state_dim = len(state_schema_for_mode(mode))
        states = np.zeros((target_count, state_dim), dtype=np.float32)
        pose6_saved = raw_pose6.astype(np.float32, copy=True)
        gripper_saved = raw_gripper.astype(np.float32, copy=True)
        joint6_saved = raw_joint6.astype(np.float32, copy=True)
        main_images = raw_main_images.copy()
        wrist_images = raw_wrist_images.copy()
        for out_idx in range(target_count):
            states[out_idx] = build_state_row(
                pose6_saved[out_idx],
                gripper_saved[out_idx],
                joint6_saved[out_idx],
                state_mode=mode,
            )
        actions = compute_delta_actions(pose6_saved, gripper_saved, joint6_saved, state_mode=mode)
        return states, actions, target_times.astype(np.float32), env_steps, main_images, wrist_images

    save_control_dt = 1.0 / float(save_fps)
    target_count = max(1, int(round((len(frames) - 1) * save_fps / RAW_CAPTURE_FPS)) + 1)
    env_steps = np.arange(target_count, dtype=np.int64)
    target_times = env_steps.astype(np.float32) * np.float32(save_control_dt)

    state_dim = len(state_schema_for_mode(mode))
    states = np.zeros((target_count, state_dim), dtype=np.float32)
    pose6_resampled = np.zeros((target_count, 6), dtype=np.float32)
    gripper_resampled = np.zeros((target_count,), dtype=np.float32)
    joint6_resampled = np.zeros((target_count,), dtype=np.float32)
    main_images = np.zeros((target_count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    wrist_images = np.zeros((target_count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for out_idx, target_t in enumerate(target_times.astype(np.float64)):
        right = int(np.searchsorted(raw_times, target_t, side="left"))
        if right <= 0:
            left = 0
            right = 0
            alpha = 0.0
        elif right >= len(raw_times):
            left = len(raw_times) - 1
            right = left
            alpha = 0.0
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
        states[out_idx] = build_state_row(
            interp_pose6,
            gripper_resampled[out_idx],
            joint6_resampled[out_idx],
            state_mode=mode,
        )
        main_images[out_idx] = raw_main_images[image_idx]
        wrist_images[out_idx] = raw_wrist_images[image_idx]

    actions = compute_delta_actions(pose6_resampled, gripper_resampled, joint6_resampled, state_mode=mode)
    return states, actions, target_times.astype(np.float32), env_steps, main_images, wrist_images


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_episode(
    frames: list[RecordedFrame],
    save_dir: Path,
    prompt: str,
    *,
    save_fps: int,
    state_mode: str,
) -> Path:
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

    states, actions, timestamps, env_steps, main_images, wrist_images = prepare_episode_for_save(
        frames,
        save_fps=save_fps,
        state_mode=state_mode,
    )
    state_schema = state_schema_for_mode(state_mode)
    action_schema = action_schema_for_mode(state_mode)

    np.save(episode_dir / "states.npy", states)
    np.save(episode_dir / "actions.npy", actions)
    np.save(episode_dir / "timestamps.npy", timestamps)
    np.save(episode_dir / "env_steps.npy", env_steps)
    np.savez_compressed(
        episode_dir / "images.npz",
        main_images=main_images,
        wrist_images=wrist_images,
    )

    metadata = {
        "task": prompt,
        "fps": int(save_fps),
        "nominal_fps": float(save_fps),
        "n_frames": int(states.shape[0]),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "record_every": 1,
        "base_fps": int(save_fps),
        "source_fps": int(RAW_CAPTURE_FPS),
        "image_format": "npz",
        "sampling_mode": "raw_capture_no_resample" if save_fps == RAW_CAPTURE_FPS else "resampled_global_env_step_with_event_frames",
        "timestamp_mode": "env_step_times_control_dt",
        "timestamps_file": "timestamps.npy",
        "env_steps_file": "env_steps.npy",
        "state_dim": len(state_schema),
        "action_dim": 7,
        "state_schema": state_schema,
        "action_schema": action_schema,
        "state_mode": _normalize_state_mode(state_mode),
        "pose_frame": get_alignment_mode(),
    }
    with open(episode_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"  Episode {episode_id} saved: {episode_dir} "
        f"(raw {len(frames)} -> saved {states.shape[0]} frames @ {save_fps}Hz)"
    )
    return episode_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-robot data collection for OpenPI.")
    # task is selected interactively at startup, not via CLI
    parser.add_argument("--prompt", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=str, default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--speed", type=float, default=LINEAR_SPEED)
    parser.add_argument("--save-fps", type=int, choices=(RAW_CAPTURE_FPS, UPSAMPLED_SAVE_FPS), default=DEFAULT_SAVE_FPS)
    parser.add_argument("--state-mode", type=str, choices=(STATE_MODE_YAW, STATE_MODE_J6), default=STATE_MODE_J6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gripper-timeout", type=float, default=10.0)
    parser.add_argument("--pose-frame", type=str, choices=("sim", "real"), default="sim")
    return parser.parse_args()


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
    from support.tui_config import run_collect_tui_config

    args = parse_args()
    set_alignment_mode(args.pose_frame)
    save_dir = Path(args.data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global LINEAR_SPEED
    if args.speed <= 0:
        raise ValueError(f"--speed must be positive, got {args.speed}")
    LINEAR_SPEED = args.speed

    saved_state_preview = load_collect_state(save_dir)
    tui_cfg = run_collect_tui_config(
        default_mode="auto",
        default_auto_episodes=max(1, args.max_episodes) if args.max_episodes > 0 else 10,
        default_resume_mode="continue" if saved_state_preview is not None else "reset",
        default_task="pick_and_place",
        default_save_fps=args.save_fps,
        default_state_mode=args.state_mode,
    )
    if tui_cfg.quit:
        print("Exiting.")
        return 0

    selected_task = str(tui_cfg.task)
    if selected_task == "open_and_close":
        tui_cfg.mode = "manual"
        tui_cfg.auto_episodes = 1
    auto_episodes = 0 if tui_cfg.mode == "manual" else int(tui_cfg.auto_episodes)
    save_fps = int(tui_cfg.save_fps)
    collect_state_mode = _normalize_state_mode(tui_cfg.state_mode)
    resume_mode = str(tui_cfg.resume_mode)

    print("\nConfiguration:")
    print(f"  Mode:       {tui_cfg.mode}")
    if tui_cfg.mode == "auto":
        print(f"  Episodes:   {auto_episodes}")
    print(f"  Resume:     {resume_mode}")
    print(f"  Task:       {selected_task}")
    print(f"  Save FPS:   {save_fps}")
    print(f"  State Mode: {collect_state_mode}")
    print(f"  Pose Frame: {get_alignment_mode()}")
    print(f"  Speed:      {LINEAR_SPEED} m/s")
    print(f"  Dry-run:    {args.dry_run}")
    if selected_task not in {"pick_and_place", "open_and_close"}:
        print(f"Unsupported task '{selected_task}'.")
        return 1

    planner_config = PlannerConfig(
        workspace_x_min=WORKSPACE_X_MIN,
        workspace_x_max=WORKSPACE_X_MAX,
        workspace_y_min=WORKSPACE_Y_MIN,
        workspace_y_max=WORKSPACE_Y_MAX,
        min_spacing_m=MIN_CUBE_SPACING_M,
        object_height_m=CUBE_HEIGHT_M,
        rotate_deg_min=ROTATE_ABS_DEG_MIN,
        rotate_deg_max=ROTATE_ABS_DEG_MAX,
    )

    print(f"\nTask: {selected_task}")
    print(f"Data dir: {save_dir}")
    print(f"Raw capture FPS: {RAW_CAPTURE_FPS}")
    print(f"Saved dataset FPS: {save_fps}")
    print(f"Resample mode: {'off' if save_fps == RAW_CAPTURE_FPS else f'{RAW_CAPTURE_FPS}->{save_fps}'}")
    print(f"Speed: {LINEAR_SPEED} m/s")
    print(f"Dry-run: {args.dry_run}")
    print(f"Pose frame: {get_alignment_mode()}")
    print(f"State mode: {collect_state_mode}")
    if selected_task == "pick_and_place":
        print(f"Objects:      {list(OBJECT_ORDER)}")
        print("Prompt mode:  auto random ('pick up ...' / 'put ... on ...')")
        print("Task mix:     20% pick / 80% place")
    else:
        print("Prompt mode:  alternating ('open the storage box' / 'close the storage box')")
        print("Task mix:     open / close alternating by episode")

    saved_held_object = _normalize_held_object(saved_state_preview.get("held_object")) if saved_state_preview else None
    if selected_task == "pick_and_place" and resume_mode == "continue" and saved_held_object is not None:
        raise RuntimeError(
            f"saved state indicates the gripper is still holding '{saved_held_object}'. "
            "Please manually restore the scene and restart with resume=reset."
        )

    cameras: CameraPair | None = None
    daemon = None
    episode_count = 0
    scene_state: dict[str, dict[str, object]] = {}
    cleanup_scene_state: dict[str, dict[str, object]] | None = None
    runtime_held_object: str | None = None
    open_close_reference: OpenCloseReference | None = None
    task_index = 0
    cleanup_state = {"done": False}

    def cleanup_collection() -> None:
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        latest_scene_state = cleanup_scene_state if cleanup_scene_state is not None else scene_state
        if selected_task == "pick_and_place" and latest_scene_state:
            save_collect_state(
                save_dir,
                latest_scene_state,
                task_index,
                episode_count,
                held_object=runtime_held_object,
            )
            print(f"  State saved to {_state_file_path(save_dir)}")
        elif selected_task == "open_and_close" and open_close_reference is not None:
            save_collect_state(
                save_dir,
                {},
                task_index,
                episode_count,
                open_close_reference=open_close_reference,
            )
            print(f"  State saved to {_state_file_path(save_dir)}")
        if daemon is not None:
            try:
                stop_motion_and_confirm(daemon, "collect_data cleanup")
            except Exception:
                pass
            try:
                daemon.stop()
            except Exception:
                pass
        if cameras is not None:
            cameras.stop()

    atexit.register(cleanup_collection)

    print("Compiling C++ helpers...")
    build_joint_helper()
    build_tcp_helper()

    print("Initializing cameras...")
    cameras = CameraPair()
    daemon = _get_motion_daemon()
    try:
        startup_stop_resp = daemon.stop_motion()
        print(f"  Best-effort startup stop: {startup_stop_resp}")
    except Exception as exc:
        print(f"  Best-effort startup stop skipped: {exc}")
    startup_snap = get_robot_snapshot()
    startup_tcp_pose = np.asarray(startup_snap.tcp_pose, dtype=np.float64).reshape(6).copy()

    print("Moving to initial joint position...")
    ensure_joint_move_ok(
        move_to_joint_positions(REAL_INIT_QPOS_RAD, execute=not args.dry_run),
        "initial joint alignment",
    )
    print("  Joint alignment OK.")

    snap = get_robot_snapshot()
    set_runtime_alignment(snap.tcp_pose, frame_mode=args.pose_frame)
    init_real = snap.tcp_pose.copy()
    init_sim = real_pose_to_sim(init_real)
    home_real = init_real.copy()
    origin_xy = home_real[:2].copy()
    print(f"  Init real TCP: {np.round(init_real, 5).tolist()}")
    print(f"  Init policy TCP:  {np.round(init_sim, 5).tolist()}")

    if not args.dry_run:
        print("Opening gripper...")
        ensure_gripper_ok(
            command_gripper_state(1, timeout_s=args.gripper_timeout),
            "open gripper before collection",
        )

    z_grasp = MIN_TCP_Z
    skip_prep = False
    default_joint6_rad = float(DEFAULT_J6_HOME_RAD)
    local_exec_joint6_rad = float(DEFAULT_J6_HOME_RAD)

    # --- Check for saved state ---
    saved = saved_state_preview
    if selected_task == "pick_and_place":
        saved_scene_state = saved.get("scene_state", {}) if saved is not None else {}
        if saved is not None and saved_scene_state:
            print("\n  Found saved object states from previous run:")
            for name, state in saved_scene_state.items():
                xy = get_object_xy(saved_scene_state, name)
                is_rotate = get_object_is_rotate(saved_scene_state, name)
                deg = get_object_deg(saved_scene_state, name)
                upper = state.get("upper")
                lower = state.get("lower")
                print(
                    f"    {name}: xy=({xy[0]:.4f}, {xy[1]:.4f}), "
                    f"is_rotate={is_rotate}, deg={deg:.1f}, upper={upper}, lower={lower}"
                )
            print(
                f"    task_index={saved['color_index']}, episode_count={saved['episode_count']}, "
                f"held_object={_normalize_held_object(saved.get('held_object'))}"
            )
            if resume_mode == "reset":
                clear_collect_state(save_dir)
                print("  State cleared. Starting fresh.")
            else:
                scene_state = saved_scene_state
                task_index = saved["color_index"]
                episode_count = saved["episode_count"]
                skip_prep = True
                print("  Resuming from saved state.")
        elif resume_mode == "continue":
            print("\n  Resume selected, but no saved state exists. Starting fresh.")
    else:
        if resume_mode == "reset":
            if saved is not None:
                clear_collect_state(save_dir)
                print("\n  State cleared. Open/close reference will be re-captured from current TCP.")
        elif resume_mode == "continue":
            if saved is None:
                print("\n  Resume selected, but no saved state exists. Open/close reference will use startup TCP.")
            else:
                task_index = int(saved.get("color_index", 0))
                episode_count = int(saved.get("episode_count", 0))
                saved_reference = saved.get("open_close_reference")
                if isinstance(saved_reference, OpenCloseReference):
                    open_close_reference = saved_reference
                    print(
                        "\n  Found saved open/close reference: "
                        f"x_start={open_close_reference.x_start:.4f}, "
                        f"y_start={open_close_reference.y_start:.4f}, "
                        f"z_base={open_close_reference.z_base:.4f}"
                    )
                    print(f"  Resuming with saved open/close reference (episode_count={episode_count}).")
                else:
                    if saved.get("has_valid_counters", False):
                        print(
                            "\n  Restored open/close counters from saved state: "
                            f"task_index={task_index}, episode_count={episode_count}."
                        )
                    print(
                        "  Saved state has no reusable open/close reference. "
                        "Open/close reference will use startup TCP."
                    )

    def refresh_home_pose() -> np.ndarray:
        nonlocal home_real, origin_xy
        snap_local = get_robot_snapshot()
        set_runtime_alignment(snap_local.tcp_pose)
        home_real = np.asarray(snap_local.tcp_pose, dtype=np.float64).reshape(6).copy()
        origin_xy = home_real[:2].copy()
        return home_real.copy()

    def build_pose_from_live_orientation(x: float, y: float, z: float) -> np.ndarray:
        snap_local = get_robot_snapshot()
        live_base = np.asarray(snap_local.tcp_pose, dtype=np.float64).reshape(6).copy()
        return build_pose_at_xy(live_base, x, y, z)

    def return_home(label: str) -> np.ndarray:
        nonlocal home_real, origin_xy, local_exec_joint6_rad
        if not args.dry_run:
            ensure_joint_move_ok(
                move_to_joint_positions(REAL_INIT_QPOS_RAD, execute=True),
                label,
            )
            local_exec_joint6_rad = float(DEFAULT_J6_HOME_RAD)
            return refresh_home_pose()
        local_exec_joint6_rad = float(DEFAULT_J6_HOME_RAD)
        origin_xy = home_real[:2].copy()
        return home_real.copy()
    def z_for_pick_level(level: int) -> float:
        return float(z_grasp + CUBE_HEIGHT_M * max(0, int(level)))

    def z_for_place_level(level: int) -> float:
        return float(z_for_pick_level(level) + PLACE_HEIGHT_OFFSET_M)

    def j6_target_from_deg(deg: float) -> float:
        return float(default_joint6_rad + np.deg2rad(float(deg)))

    def resolve_step_target_joint6_rad(
        step: TaskStep,
        lookup_scene_state: dict[str, dict[str, object]],
    ) -> float:
        target_joint6_rad = j6_target_from_deg(step.deg)
        if step.object_name == APPLE_NAME or not step.is_rotate:
            return target_joint6_rad

        reference_name: str | None
        if step.kind == "pick":
            reference_name = step.object_name
        elif step.support_name is not None:
            reference_name = step.support_name
        else:
            reference_name = None

        if reference_name is None:
            return target_joint6_rad

        saved_joint6_rad = get_object_standard_j6_rad(lookup_scene_state, reference_name)
        if saved_joint6_rad is None:
            return target_joint6_rad
        return saved_joint6_rad

    def should_align_joint6_for_step(
        step: TaskStep,
        lookup_scene_state: dict[str, dict[str, object]],
    ) -> tuple[bool, float]:
        target_joint6_rad = resolve_step_target_joint6_rad(step, lookup_scene_state)
        need_align = bool(step.align_j6) and (
            abs(_wrap_angle(local_exec_joint6_rad - target_joint6_rad)) > DEFAULT_J6_EXEC_TOL_RAD
        )
        return need_align, target_joint6_rad

    def sample_initial_object_state(name: str) -> dict[str, object]:
        occupied = [origin_xy.copy()]
        for existing_name in scene_state.keys():
            occupied.append(get_object_xy(scene_state, existing_name))
        xy = sample_non_overlapping_xy(occupied, min_dist=MIN_CUBE_SPACING_M)
        if name == APPLE_NAME:
            is_rotate = False
            deg = 0.0
        elif np.random.random() < float(planner_config.non_rotated_table_place_probability):
            is_rotate = False
            deg = 0.0
        else:
            is_rotate = True
            abs_deg = float(np.random.uniform(planner_config.rotate_deg_min, planner_config.rotate_deg_max))
            deg = -abs_deg if np.random.random() < 0.5 else abs_deg
        return {
            "xy": [float(xy[0]), float(xy[1])],
            "is_rotate": bool(is_rotate),
            "deg": float(deg),
            "standard_j6_rad": None,
            "upper": None,
            "lower": None,
        }

    def scene_top_of(state: dict[str, dict[str, object]], name: str) -> str:
        current = name
        seen: set[str] = set()
        while True:
            upper = state[current].get("upper")
            if upper is None:
                return current
            if current in seen:
                raise RuntimeError(f"cycle detected while following upper chain from {name}")
            seen.add(current)
            current = str(upper)

    def scene_detach_top(state: dict[str, dict[str, object]], name: str) -> None:
        obj = state[name]
        upper = obj.get("upper")
        if upper is not None:
            raise RuntimeError(f"cannot detach {name}: upper={upper} still present")
        lower = obj.get("lower")
        if lower is not None:
            state[str(lower)]["upper"] = None
        obj["lower"] = None

    def scene_place_object(state: dict[str, dict[str, object]], step: TaskStep) -> None:
        obj = state[step.object_name]
        if obj.get("upper") is not None:
            raise RuntimeError(f"cannot place {step.object_name}: upper={obj.get('upper')} still present")
        if step.support_name is None:
            obj["xy"] = [float(step.xy[0]), float(step.xy[1])]
            obj["lower"] = None
        else:
            actual_support = scene_top_of(state, step.support_name)
            if actual_support == APPLE_NAME:
                raise RuntimeError("apple cannot receive an upper object")
            obj["xy"] = list(state[actual_support]["xy"])
            obj["lower"] = actual_support
            state[actual_support]["upper"] = step.object_name
        if step.object_name == APPLE_NAME:
            obj["is_rotate"] = False
            obj["deg"] = 0.0
            obj["standard_j6_rad"] = None
        else:
            obj["is_rotate"] = bool(step.is_rotate)
            obj["deg"] = 0.0 if not step.is_rotate else float(step.deg)

    def commit_place_state(state: dict[str, dict[str, object]], step: TaskStep) -> None:
        nonlocal cleanup_scene_state, runtime_held_object
        scene_place_object(state, step)
        runtime_held_object = None
        normalized_state = load_scene_state(state)
        if normalized_state is not None:
            cleanup_scene_state = normalized_state

    def execute_pick_step(
        step: TaskStep,
        *,
        record: bool,
        frame_idx: int,
        lookup_scene_state: dict[str, dict[str, object]],
    ) -> tuple[list[RecordedFrame], int]:
        nonlocal local_exec_joint6_rad
        target_xy = np.asarray(step.xy, dtype=np.float64).reshape(2)
        target_z = z_for_pick_level(step.level)
        above_z = z_grasp + APPROACH_Z_OFFSET_M
        step_frames: list[RecordedFrame] = []
        should_align_joint6, target_joint6_rad = should_align_joint6_for_step(step, lookup_scene_state)

        if args.dry_run:
            dummy_count = 12 + (3 if should_align_joint6 else 0)
            sim_pose = real_pose_to_sim(home_real)
            joint6 = target_joint6_rad if should_align_joint6 else local_exec_joint6_rad
            for idx in range(dummy_count):
                step_frames.append(
                    RecordedFrame(
                        sim_pose6=sim_pose.copy(),
                        gripper=1.0 if idx < dummy_count - 3 else 0.0,
                        joint6=joint6,
                        main_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        wrist_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        timestamp=(frame_idx + idx) * CONTROL_DT,
                    )
                )
            if should_align_joint6:
                local_exec_joint6_rad = target_joint6_rad
            return step_frames, frame_idx + len(step_frames)

        print(
            f"    [{step.object_name}] pick @ ({target_xy[0]:.4f}, {target_xy[1]:.4f}), "
            f"level={step.level}, rotate={step.is_rotate}, deg={step.deg:.1f}"
        )
        above_pose = build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(above_z))
        seg = execute_and_record(
            daemon,
            cameras,
            above_pose,
            gripper=1.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)

        if should_align_joint6:
            seg = execute_joint6_rotation_and_record(
                cameras,
                target_joint6=target_joint6_rad,
                gripper=1.0,
                start_frame_idx=frame_idx,
            )
            if record:
                step_frames.extend(seg)
            frame_idx += len(seg)
            if seg:
                local_exec_joint6_rad = _normalize_j6(seg[-1].joint6)
            else:
                local_exec_joint6_rad = target_joint6_rad

        down_pose = build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(target_z))
        seg = execute_and_record(
            daemon,
            cameras,
            down_pose,
            gripper=1.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)

        ensure_gripper_ok(
            command_gripper_state(0, timeout_s=args.gripper_timeout),
            f"close gripper on {step.object_name}",
        )

        lift_pose = build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(above_z))
        seg = execute_and_record(
            daemon,
            cameras,
            lift_pose,
            gripper=0.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)
        return step_frames, frame_idx

    def execute_place_step(
        step: TaskStep,
        *,
        record: bool,
        frame_idx: int,
        lookup_scene_state: dict[str, dict[str, object]],
        result_scene_state: dict[str, dict[str, object]],
    ) -> tuple[list[RecordedFrame], int]:
        nonlocal local_exec_joint6_rad
        target_xy = np.asarray(step.xy, dtype=np.float64).reshape(2)
        target_z = z_for_place_level(step.level)
        above_z = z_grasp + APPROACH_Z_OFFSET_M
        step_frames: list[RecordedFrame] = []
        should_align_joint6, target_joint6_rad = should_align_joint6_for_step(step, lookup_scene_state)

        if args.dry_run:
            dummy_count = 10 + (3 if should_align_joint6 else 0)
            sim_pose = real_pose_to_sim(home_real)
            joint6 = target_joint6_rad if should_align_joint6 else local_exec_joint6_rad
            for idx in range(dummy_count):
                step_frames.append(
                    RecordedFrame(
                        sim_pose6=sim_pose.copy(),
                        gripper=0.0 if idx < dummy_count - 3 else 1.0,
                        joint6=joint6,
                        main_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        wrist_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        timestamp=(frame_idx + idx) * CONTROL_DT,
                    )
                )
            if should_align_joint6:
                local_exec_joint6_rad = target_joint6_rad
            set_object_standard_j6_rad(
                result_scene_state,
                step.object_name,
                local_exec_joint6_rad if step.is_rotate else None,
            )
            commit_place_state(result_scene_state, step)
            return step_frames, frame_idx + len(step_frames)

        print(
            f"    [{step.object_name}] place @ ({target_xy[0]:.4f}, {target_xy[1]:.4f}), "
            f"level={step.level}, rotate={step.is_rotate}, deg={step.deg:.1f}"
        )
        above_pose = build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(above_z))
        seg = execute_and_record(
            daemon,
            cameras,
            above_pose,
            gripper=0.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)

        if should_align_joint6:
            seg = execute_joint6_rotation_and_record(
                cameras,
                target_joint6=target_joint6_rad,
                gripper=0.0,
                start_frame_idx=frame_idx,
            )
            if record:
                step_frames.extend(seg)
            frame_idx += len(seg)
            if seg:
                local_exec_joint6_rad = _normalize_j6(seg[-1].joint6)
            else:
                local_exec_joint6_rad = target_joint6_rad

        down_pose = build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(target_z))
        seg = execute_and_record(
            daemon,
            cameras,
            down_pose,
            gripper=0.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)
        actual_standard_j6_rad = _normalize_j6(seg[-1].joint6) if seg else local_exec_joint6_rad
        set_object_standard_j6_rad(
            result_scene_state,
            step.object_name,
            actual_standard_j6_rad if step.is_rotate else None,
        )

        ensure_gripper_ok(
            command_gripper_state(1, timeout_s=args.gripper_timeout),
            f"open gripper for {step.object_name}",
        )
        commit_place_state(result_scene_state, step)

        lift_pose = build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(above_z))
        seg = execute_and_record(
            daemon,
            cameras,
            lift_pose,
            gripper=1.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)
        return step_frames, frame_idx

    def execute_step_sequence(
        steps: list[TaskStep],
        *,
        record: bool,
        initial_held_object: str | None = None,
        lookup_scene_state: dict[str, dict[str, object]] | None = None,
        result_scene_state: dict[str, dict[str, object]] | None = None,
    ) -> tuple[list[RecordedFrame], str | None]:
        nonlocal cleanup_scene_state, runtime_held_object
        execution_state = scene_state if result_scene_state is None else result_scene_state
        lookup_state = execution_state if lookup_scene_state is None else lookup_scene_state
        frames: list[RecordedFrame] = []
        frame_idx = 0
        held_object: str | None = initial_held_object
        for step in steps:
            if step.kind == "pick":
                if held_object is not None:
                    raise RuntimeError(f"cannot pick {step.object_name} while holding {held_object}")
                step_frames, frame_idx = execute_pick_step(
                    step,
                    record=record,
                    frame_idx=frame_idx,
                    lookup_scene_state=lookup_state,
                )
                frames.extend(step_frames)
                scene_detach_top(execution_state, step.object_name)
                held_object = step.object_name
                runtime_held_object = held_object
            elif step.kind == "place":
                if held_object != step.object_name:
                    raise RuntimeError(
                        f"cannot place {step.object_name}: currently holding {held_object!r}"
                    )
                step_frames, frame_idx = execute_place_step(
                    step,
                    record=record,
                    frame_idx=frame_idx,
                    lookup_scene_state=lookup_state,
                    result_scene_state=execution_state,
                )
                frames.extend(step_frames)
                held_object = None
                runtime_held_object = None
            else:
                raise RuntimeError(f"unknown step kind: {step.kind}")
        return frames, held_object

    def build_pose_from_reference_orientation(reference_pose6: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
        return build_pose_at_xy(reference_pose6, x, y, z)

    def execute_move_step(step: OpenCloseMoveStep, *, record: bool, frame_idx: int, base_pose6: np.ndarray) -> tuple[list[RecordedFrame], int]:
        step_frames: list[RecordedFrame] = []
        target_pose = build_pose_from_reference_orientation(base_pose6, float(step.x), float(step.y), float(step.z))
        if args.dry_run:
            dummy_count = 6
            sim_pose = real_pose_to_sim(target_pose)
            for idx in range(dummy_count):
                step_frames.append(
                    RecordedFrame(
                        sim_pose6=sim_pose.copy(),
                        gripper=1.0,
                        joint6=float(DEFAULT_J6_HOME_RAD),
                        main_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        wrist_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                        timestamp=(frame_idx + idx) * CONTROL_DT,
                    )
                )
            return step_frames, frame_idx + len(step_frames)

        print(f"    [move] ({step.x:.4f}, {step.y:.4f}, {step.z:.4f})  {step.note}")
        seg = execute_and_record(
            daemon,
            cameras,
            target_pose,
            gripper=1.0,
            start_frame_idx=frame_idx,
            speed_mps=LINEAR_SPEED,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)
        return step_frames, frame_idx

    def execute_move_step_sequence(
        steps: list[OpenCloseMoveStep],
        *,
        record: bool,
        base_pose6: np.ndarray,
    ) -> list[RecordedFrame]:
        frames: list[RecordedFrame] = []
        frame_idx = 0
        for step in steps:
            step_frames, frame_idx = execute_move_step(step, record=record, frame_idx=frame_idx, base_pose6=base_pose6)
            frames.extend(step_frames)
        return frames

    if selected_task == "pick_and_place":
        if skip_prep:
            print("\n=== Skipping Preparation (resumed from saved state) ===")
        else:
            print("\n=== Preparation Phase ===")
            print(f"Seed order: {list(OBJECT_ORDER)}")
            for object_name in OBJECT_ORDER:
                placed_state = sample_initial_object_state(object_name)
                print(
                    f"[prep] {object_name}: origin -> ({placed_state['xy'][0]:.4f}, {placed_state['xy'][1]:.4f}), "
                    f"is_rotate={placed_state['is_rotate']}, deg={placed_state['deg']:.1f}"
                )
                if not args.dry_run:
                    pick_step = TaskStep(
                        kind="pick",
                        object_name=object_name,
                        xy=(float(origin_xy[0]), float(origin_xy[1])),
                        level=0,
                        is_rotate=False,
                        deg=0.0,
                        align_j6=bool(object_name != APPLE_NAME),
                        note=f"prep pick {object_name}",
                    )
                    place_step = TaskStep(
                        kind="place",
                        object_name=object_name,
                        xy=(float(placed_state["xy"][0]), float(placed_state["xy"][1])),
                        level=0,
                        is_rotate=bool(placed_state["is_rotate"]),
                        deg=float(placed_state["deg"]),
                        align_j6=bool(object_name != APPLE_NAME),
                        note=f"prep place {object_name}",
                    )
                    _, held_after_prep = execute_step_sequence(
                        [pick_step, place_step],
                        record=False,
                        lookup_scene_state=scene_state,
                        result_scene_state={object_name: placed_state},
                    )
                    if held_after_prep is not None:
                        raise RuntimeError(f"prep sequence ended while still holding {held_after_prep}")
                    return_home(f"[prep {object_name}] return home")
                scene_state[object_name] = placed_state
            save_collect_state(save_dir, scene_state, task_index, episode_count)
    else:
        print("\n=== Open/Close Reference Setup ===")
        if open_close_reference is None:
            print("Auto-capturing startup TCP as OPEN start point (x/y source).")
            set_runtime_alignment(startup_tcp_pose)
            open_close_reference = build_reference_from_tcp_pose(startup_tcp_pose)
            print(
                "  Captured open reference: "
                f"x_start={open_close_reference.x_start:.4f}, "
                f"y_start={open_close_reference.y_start:.4f}, "
                f"z_base={open_close_reference.z_base:.4f}"
            )
            save_collect_state(
                save_dir,
                {},
                task_index,
                episode_count,
                open_close_reference=open_close_reference,
            )
            if not args.dry_run:
                return_home("post-reference return home")
        else:
            print(
                "Using saved open reference: "
                f"x_start={open_close_reference.x_start:.4f}, "
                f"y_start={open_close_reference.y_start:.4f}, "
                f"z_base={open_close_reference.z_base:.4f}"
            )

    if selected_task == "open_and_close":
        max_ep = episode_count + 1
    else:
        max_ep = auto_episodes if auto_episodes > 0 else args.max_episodes

    print("\n=== Ready For Episodes ===")
    if auto_episodes > 0:
        print(f"Auto mode: {auto_episodes} episodes, no pause between episodes.")
    else:
        print("Manual mode: press ENTER to collect the next random episode.")
    if selected_task == "pick_and_place":
        print("Task policy: 20% pick / 80% place")
    else:
        print("Task policy: open, close, open, close ...")
        print("Gripper policy: always open (no close/open action during episode)")
    print("Quit with q / quit / exit.\n")

    try:
        while True:
            if 0 < max_ep <= episode_count:
                print(f"\nReached {episode_count} episodes. Done.")
                break

            if selected_task == "pick_and_place":
                if not scene_state:
                    raise RuntimeError("scene_state is empty; cannot plan next episode")

                plan = build_random_episode_plan(scene_state, config=planner_config)
                episode_label = f"Episode {episode_count} [{plan.task_kind}]"

                print(f"\n--- {episode_label} ---")
                print(f"  Prompt: \"{plan.prompt}\"")
                print(f"  Source: {plan.source_name}")
                if plan.target_name is not None:
                    print(f"  Target: {plan.target_name}")
                print(f"  Recorded steps: {len(plan.recorded_steps)}")
                print(f"  Post steps: {len(plan.post_steps)}")
            else:
                if open_close_reference is None:
                    raise RuntimeError("open/close reference is not initialized")
                plan = build_open_and_close_episode_plan(
                    reference=open_close_reference,
                    episode_index=episode_count,
                    workspace_x_min=WORKSPACE_X_MIN,
                    workspace_x_max=WORKSPACE_X_MAX,
                    workspace_y_min=WORKSPACE_Y_MIN,
                    workspace_y_max=WORKSPACE_Y_MAX,
                )
                episode_label = f"Episode {episode_count} [{plan.task_kind}]"
                print(f"\n--- {episode_label} ---")
                print(f"  Prompt: \"{plan.prompt}\"")
                print(f"  Recorded move steps: {len(plan.recorded_steps)}")

            if auto_episodes > 0:
                print(f"  Auto: recording ({episode_count + 1}/{auto_episodes})")
            else:
                cmd = input("  Press ENTER to record this episode, or q to quit: ").strip().lower()
                if cmd in QUIT_TOKENS:
                    print("Exit requested.")
                    break
                if cmd:
                    print(f"Unsupported input '{cmd}', exiting.")
                    break

            if not args.dry_run:
                return_home("pre-episode return home")
                ensure_gripper_ok(
                    command_gripper_state(1, timeout_s=args.gripper_timeout),
                    "open gripper before episode",
                )

            if selected_task == "pick_and_place":
                episode_execution_scene_state = clone_scene_state(scene_state)
                all_frames, held_after_recorded = execute_step_sequence(
                    plan.recorded_steps,
                    record=True,
                    result_scene_state=episode_execution_scene_state,
                )
            else:
                all_frames = execute_move_step_sequence(
                    plan.recorded_steps,
                    record=True,
                    base_pose6=open_close_reference.reference_pose6,
                )
                held_after_recorded = None
            if not all_frames:
                raise RuntimeError("recorded episode produced no frames")

            save_episode(
                all_frames,
                save_dir,
                plan.prompt,
                save_fps=save_fps,
                state_mode=collect_state_mode,
            )
            episode_count += 1
            task_index += 1

            if selected_task == "pick_and_place":
                if plan.post_steps:
                    _, held_after_post = execute_step_sequence(
                        plan.post_steps,
                        record=False,
                        initial_held_object=held_after_recorded,
                        result_scene_state=episode_execution_scene_state,
                    )
                else:
                    held_after_post = held_after_recorded

                if held_after_post is not None:
                    raise RuntimeError(f"episode ended while still holding {held_after_post}")

            if not args.dry_run:
                return_home("post-episode return home")

            if selected_task == "pick_and_place":
                scene_state = clone_scene_state(episode_execution_scene_state)
                save_collect_state(save_dir, scene_state, task_index, episode_count)
                cleanup_scene_state = None
            else:
                save_collect_state(
                    save_dir,
                    {},
                    task_index,
                    episode_count,
                    open_close_reference=open_close_reference,
                )
            continue
    except KeyboardInterrupt:
        print(f"\n\nCollection stopped after {episode_count} episodes.")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        cleanup_collection()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

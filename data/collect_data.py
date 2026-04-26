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
  states.npy     (N, 7) float32    yaw: [x, y, z, aa_x, aa_y, aa_z, gripper]
  actions.npy    (N, 7) float32    yaw: [dx, dy, dz, droll, dpitch, dyaw, gripper_next]
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

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_IMPORT_ROOT = SCRIPT_DIR.parent
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

import numpy as np
from utils.env_utils import load_default_env
from utils.path_utils import get_openpi_root, get_repo_root

load_default_env()

OPENPI_ROOT = get_openpi_root()
REPO_ROOT = get_repo_root()
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
SERVO_CONTROL_DT = 0.01
SERVO_ANGULAR_SPEED_RADPS = 0.60
DEFAULT_YAW_SPEED_RADPS = float(np.deg2rad(10.0))
DEFAULT_YAW_HOME_RAD = 0.11434
DEFAULT_YAW_EXEC_TOL_RAD = float(np.deg2rad(1.0))
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
    STATE_MODE_YAW,
    preprocess_image_for_openpi as preprocess_image,
)
from task.keyboard_teleop import (
    KeyboardTeleopConfig,
    run_session as kt_run_session,
)
from task.pick_and_place import (
    OBJECT_ORDER,
    PlannerConfig,
    load_scene_state,
    PickAndPlaceSession,
    describe_episode as pp_describe_episode,
    finalize_episode as pp_finalize_episode,
    plan_next_episode as pp_plan_next_episode,
    prepare_session as pp_prepare_session,
    record_episode as pp_record_episode,
    restore_session as pp_restore_session,
)
from task.open_and_close import (
    BAND_OBJECT_COUNT_MAX as OC_BAND_COUNT_MAX,
    BAND_OBJECT_COUNT_MIN as OC_BAND_COUNT_MIN,
    CLEAR_MIN_SPACING_M as OC_CLEAR_SPACING,
    ObstacleScene,
    OpenCloseReference,
    OpenCloseSession,
    describe_cycle as oc_describe_cycle,
    finalize_cycle as oc_finalize_cycle,
    plan_cycle as oc_plan_cycle,
    prepare_session as oc_prepare_session,
    record_cycle as oc_record_cycle,
    restore_session as oc_restore_session,
    STACK_PROBABILITY as OC_STACK_PROBABILITY,
)
from task.storage import (
    StorageSession,
    describe_episode as st_describe_episode,
    finalize_episode as st_finalize_episode,
    has_remaining_objects as st_has_remaining_objects,
    plan_next_episode as st_plan_next_episode,
    prepare_session as st_prepare_session,
    record_episode as st_record_episode,
    restore_session as st_restore_session,
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


def _axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / angle
    half = 0.5 * angle
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _require_yaw_readback(joint_q: np.ndarray, *, context: str) -> float:
    joint_q_arr = np.asarray(joint_q, dtype=np.float64).reshape(-1)
    if joint_q_arr.size < 6:
        raise RuntimeError(
            f"{context}: robot snapshot missing yaw readback "
            f"(joint_q size={joint_q_arr.size})"
        )
    return float(joint_q_arr[5])


def build_state_row(pose6_zyx: np.ndarray, gripper: float, yaw: float) -> np.ndarray:
    """Build one yaw-mode state row."""
    _ = yaw
    pose = np.asarray(pose6_zyx, dtype=np.float64).reshape(6)
    quat = _euler_zyx_to_quat_wxyz(pose[3:6])
    aa = _quat_to_axis_angle_wxyz(quat)
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
    yaw: np.ndarray,
) -> np.ndarray:
    """Compute (N, 7) delta actions for yaw-mode datasets."""
    pose6 = np.asarray(pose6, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
    if pose6.ndim != 2 or pose6.shape[1] != 6:
        raise ValueError(f"pose6 must have shape (N, 6), got {pose6.shape}")
    if gripper.shape[0] != pose6.shape[0] or yaw.shape[0] != pose6.shape[0]:
        raise ValueError("pose6, gripper, and yaw must have the same length")

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
            actions[idx, 5] = deuler[2]
            actions[idx, 6] = gripper[idx + 1]
        else:
            actions[idx, :6] = 0.0
            actions[idx, 6] = gripper[idx]
    return actions


def compute_actions_from_saved_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    num_frames = int(states.shape[0])
    actions = np.zeros((num_frames, 7), dtype=np.float32)
    if num_frames <= 0:
        return actions

    pos = states[:, :3].astype(np.float64)
    axis_angle = states[:, 3:6].astype(np.float64)
    grip = states[:, 6].astype(np.float32)
    eulers = np.stack([quat_to_euler_wxyz(_axis_angle_to_quat_wxyz(aa)) for aa in axis_angle], axis=0)

    for idx in range(num_frames):
        if idx < num_frames - 1:
            actions[idx, :3] = (pos[idx + 1] - pos[idx]).astype(np.float32)
            deulers = np.arctan2(np.sin(eulers[idx + 1] - eulers[idx]), np.cos(eulers[idx + 1] - eulers[idx]))
            actions[idx, 3:6] = deulers.astype(np.float32)
            actions[idx, 6] = grip[idx + 1]
        else:
            actions[idx, :6] = 0.0
            actions[idx, 6] = grip[idx]
    return actions


def state_schema() -> list[str]:
    return ["x", "y", "z", "aa_x", "aa_y", "aa_z", "gripper_open"]


def action_schema() -> list[str]:
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


def _normalize_held_object(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in OBJECT_ORDER:
        return text
    return None


def _load_raw_state_payload(data_dir: Path) -> dict:
    path = _state_file_path(data_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _serialize_open_close_reference(reference: OpenCloseReference) -> dict[str, object]:
    return {
        "x_start": float(reference.x_start),
        "y_start": float(reference.y_start),
        "reference_pose6": [
            float(v)
            for v in np.asarray(reference.reference_pose6, dtype=np.float64).reshape(6).tolist()
        ],
    }


def _deserialize_open_close_reference(payload: object) -> OpenCloseReference | None:
    if not isinstance(payload, dict):
        return None
    try:
        ref_pose = np.asarray(payload.get("reference_pose6", []), dtype=np.float64).reshape(6).copy()
        return OpenCloseReference(
            x_start=float(payload.get("x_start", ref_pose[0])),
            y_start=float(payload.get("y_start", ref_pose[1])),
            reference_pose6=ref_pose,
        )
    except Exception:
        return None


def save_collect_state(
    data_dir: Path,
    scene_state: dict[str, dict[str, object]],
    color_index: int,
    episode_count: int,
    *,
    held_object: str | None = None,
    open_close_reference: OpenCloseReference | None = None,
    obstacle_scene: ObstacleScene | None = None,
    storage_state: dict[str, object] | None = None,
) -> None:
    payload = _load_raw_state_payload(data_dir)
    legacy_open_close_reference = _deserialize_open_close_reference(payload.get("open_close_reference"))
    if "open_close_state" not in payload and legacy_open_close_reference is not None:
        legacy_open_close_episode_count = _load_state_counter(payload, "episode_count")
        payload["open_close_state"] = {
            "episode_count": 0 if legacy_open_close_episode_count is None else legacy_open_close_episode_count,
            "reference": _serialize_open_close_reference(legacy_open_close_reference),
        }
    payload.pop("open_close_reference", None)
    if open_close_reference is None and storage_state is None:
        payload["object_states"] = scene_state
        payload["color_index"] = int(color_index)
        payload["episode_count"] = int(episode_count)
        payload["held_object"] = _normalize_held_object(held_object)
    elif open_close_reference is not None:
        oc_state: dict[str, object] = {
            "episode_count": int(episode_count),
            "reference": _serialize_open_close_reference(open_close_reference),
        }
        if obstacle_scene is not None:
            oc_state["obstacles"] = obstacle_scene.to_serializable()
        payload["open_close_state"] = oc_state
    if storage_state is not None:
        payload["storage_state"] = storage_state
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
        payload = _load_raw_state_payload(data_dir)
        if not payload:
            return None
        scene_payload = payload.get("object_states")
        if scene_payload is None:
            scene_payload = payload
        scene_state = load_scene_state(scene_payload)
        open_close_reference: OpenCloseReference | None = None
        open_close_episode_count: int | None = None
        open_close_payload = payload.get("open_close_state")
        obstacle_scene: ObstacleScene | None = None
        if isinstance(open_close_payload, dict):
            open_close_episode_count = _load_state_counter(open_close_payload, "episode_count")
            open_close_reference = _deserialize_open_close_reference(open_close_payload.get("reference"))
            raw_obstacles = open_close_payload.get("obstacles")
            if isinstance(raw_obstacles, dict):
                obstacle_scene = ObstacleScene.from_serialized(raw_obstacles)
        if open_close_reference is None:
            open_close_reference = _deserialize_open_close_reference(payload.get("open_close_reference"))
            if open_close_reference is not None:
                open_close_episode_count = _load_state_counter(payload, "episode_count")
        color_index = _load_state_counter(payload, "color_index")
        episode_count = _load_state_counter(payload, "episode_count")
        held_object = _normalize_held_object(payload.get("held_object"))
        storage_state_payload = payload.get("storage_state")
        storage_state: dict[str, object] | None = None
        if isinstance(storage_state_payload, dict):
            storage_scene_raw = storage_state_payload.get("scene_state")
            storage_scene = load_scene_state(storage_scene_raw) if isinstance(storage_scene_raw, dict) else None
            storage_next_index = _load_state_counter(storage_state_payload, "next_index")
            storage_episode_count = _load_state_counter(storage_state_payload, "episode_count")
            storage_held_object = _normalize_held_object(storage_state_payload.get("held_object"))
            if storage_scene is not None or storage_next_index is not None or storage_episode_count is not None:
                storage_state = {
                    "scene_state": storage_scene or {},
                    "next_index": 0 if storage_next_index is None else storage_next_index,
                    "episode_count": 0 if storage_episode_count is None else storage_episode_count,
                    "held_object": storage_held_object,
                }
        has_valid_counters = color_index is not None and episode_count is not None
        has_valid_open_close_episode_count = open_close_episode_count is not None
        has_valid_storage_state = storage_state is not None
        if (
            scene_state is None
            and open_close_reference is None
            and not has_valid_counters
            and not has_valid_open_close_episode_count
            and not has_valid_storage_state
        ):
            return None
        return {
            "scene_state": scene_state or {},
            "color_index": 0 if color_index is None else color_index,
            "episode_count": 0 if episode_count is None else episode_count,
            "held_object": held_object,
            "has_valid_counters": has_valid_counters,
            "open_close_episode_count": 0 if open_close_episode_count is None else open_close_episode_count,
            "has_valid_open_close_episode_count": has_valid_open_close_episode_count,
            "open_close_reference": open_close_reference,
            "obstacle_scene": obstacle_scene,
            "storage_state": storage_state,
        }
    except Exception:
        return None


def clear_collect_state(data_dir: Path) -> None:
    path = _state_file_path(data_dir)
    if path.exists():
        path.unlink()


def clear_open_close_state(data_dir: Path) -> None:
    path = _state_file_path(data_dir)
    if not path.exists():
        return
    payload = _load_raw_state_payload(data_dir)
    if not payload:
        clear_collect_state(data_dir)
        return
    payload.pop("open_close_state", None)
    payload.pop("open_close_reference", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def clear_storage_state(data_dir: Path) -> None:
    path = _state_file_path(data_dir)
    if not path.exists():
        return
    payload = _load_raw_state_payload(data_dir)
    if not payload:
        clear_collect_state(data_dir)
        return
    payload.pop("storage_state", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_pose_at_xy(base_pose: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float64).reshape(6).copy()
    pose[0] = float(x)
    pose[1] = float(y)
    pose[2] = float(z)
    return pose


def ensure_joint_move_ok(result, label: str) -> None:
    if not result.ok:
        raise RuntimeError(f"{label} failed: {result.reason}")


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


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _build_servo_pose_targets(
    start_real: np.ndarray,
    target_real: np.ndarray,
    *,
    speed_mps: float,
    start_yaw: float | None = None,
    target_yaw: float | None = None,
    angular_speed_radps: float | None = None,
    yaw_speed_radps: float | None = None,
) -> tuple[list[np.ndarray], list[float] | None]:
    start_pose = np.asarray(start_real, dtype=np.float64).reshape(6)
    target_pose = np.asarray(target_real, dtype=np.float64).reshape(6)
    delta = target_pose - start_pose
    delta[3:] = np.array([_wrap_angle(target_pose[i] - start_pose[i]) for i in range(3, 6)], dtype=np.float64)

    linear_dist = float(np.linalg.norm(delta[:3]))
    angular_dist = float(np.linalg.norm(delta[3:]))
    linear_time = linear_dist / max(float(speed_mps), 1e-6)
    pose_angular_speed = float(SERVO_ANGULAR_SPEED_RADPS if angular_speed_radps is None else angular_speed_radps)
    angular_time = angular_dist / max(pose_angular_speed, 1e-6)
    joint_time = 0.0
    joint_targets: list[float] | None = None
    joint_delta = 0.0
    if target_yaw is not None and start_yaw is not None:
        joint_delta = _wrap_angle(float(target_yaw) - float(start_yaw))
        joint_speed = float(DEFAULT_YAW_SPEED_RADPS if yaw_speed_radps is None else yaw_speed_radps)
        joint_time = abs(joint_delta) / max(joint_speed, 1e-6)

    required_time = max(linear_time, angular_time, joint_time, float(SERVO_CONTROL_DT))
    step_count = max(1, int(np.ceil(required_time / float(SERVO_CONTROL_DT))))
    pose_targets = [
        (start_pose + delta * (float(step_idx) / float(step_count))).copy()
        for step_idx in range(1, step_count + 1)
    ]
    if target_yaw is not None and start_yaw is not None:
        joint_targets = [
            float(start_yaw + joint_delta * (float(step_idx) / float(step_count)))
            for step_idx in range(1, step_count + 1)
        ]
    return pose_targets, joint_targets


def _capture_recorded_frame(
    cameras: CameraPair,
    actual_real: np.ndarray,
    semantic_yaw: float,
    readback_yaw: float,
    *,
    gripper: float,
    frame_idx: int,
) -> RecordedFrame:
    from support.pose_align import real_pose_to_sim

    actual_sim = real_pose_to_sim(actual_real)
    main_bgr, wrist_bgr = cameras.grab()
    return RecordedFrame(
        sim_pose6=actual_sim,
        gripper=float(gripper),
        yaw=float(semantic_yaw),
        yaw_readback=float(readback_yaw),
        main_image=preprocess_image(main_bgr),
        wrist_image=preprocess_image(wrist_bgr),
        timestamp=frame_idx * CONTROL_DT,
    )


class _SegmentServoRunner:
    def __init__(
        self,
        daemon,
        *,
        label: str,
        start_real: np.ndarray,
        target_real: np.ndarray,
        pose_targets: list[np.ndarray],
        target_yaw: float | None,
        joint_targets: list[float] | None,
        semantic_yaw: float,
        require_force_guard: bool,
        force_live_mode: bool,
        reuse_servo: bool,
        yaw_speed_radps: float | None,
    ) -> None:
        self.daemon = daemon
        self.label = label
        self.start_real = np.asarray(start_real, dtype=np.float64).reshape(6).copy()
        self.target_real = np.asarray(target_real, dtype=np.float64).reshape(6).copy()
        self.pose_targets = [np.asarray(p, dtype=np.float64).reshape(6).copy() for p in pose_targets]
        self.target_yaw = None if target_yaw is None else float(target_yaw)
        self.joint_targets = None if joint_targets is None else [float(v) for v in joint_targets]
        self.current_semantic_yaw = float(semantic_yaw)
        self.require_force_guard = bool(require_force_guard)
        self.force_live_mode = bool(force_live_mode)
        self.reuse_servo = bool(reuse_servo)
        self.yaw_speed_radps = None if yaw_speed_radps is None else float(yaw_speed_radps)
        self.started_servo = False
        self.stream_finished = False
        self.last_force_guard_adjusted = False
        self.last_force_guard_scale: float | None = None
        self.error: Exception | None = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"collect-servo-{label}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    def _check_servo_response(self, resp: dict[str, object], *, context: str) -> None:
        pose_ret = int(resp.get("servo_pose_ret", -1))
        if pose_ret != 0:
            raise RuntimeError(f"{context} failed: {resp}")
        if self.require_force_guard and resp.get("force_guard_fz_n") is None:
            raise RuntimeError(f"{context} lost force guard")
        self.last_force_guard_adjusted = _coerce_bool(resp.get("force_guard_adjusted"))
        self.last_force_guard_scale = _coerce_optional_float(resp.get("force_guard_scale"))

    def _wait_until(self, target_ts: float) -> bool:
        while not self._stop_event.is_set():
            now = time.monotonic()
            remaining = float(target_ts - now)
            if remaining <= 0.0:
                return True
            self._stop_event.wait(min(remaining, 0.002))
        return False

    def _run(self) -> None:
        next_send_ts = time.monotonic()
        try:
            if not self.reuse_servo:
                start_resp = self.daemon.servo_start(SERVO_CONTROL_DT)
                self.started_servo = True
                if int(start_resp.get("servo_start_ret", -1)) != 0:
                    raise RuntimeError(f"{self.label} servo_start failed: {start_resp}")

            begin_resp = self.daemon.servo_begin_chunk(self.start_real, force_live_mode=self.force_live_mode)
            if self.require_force_guard and begin_resp.get("force_guard_fz_n") is None:
                raise RuntimeError(f"{self.label} aborted: force guard unavailable for downward motion")
            self.last_force_guard_adjusted = _coerce_bool(begin_resp.get("force_guard_adjusted"))
            self.last_force_guard_scale = _coerce_optional_float(begin_resp.get("force_guard_scale"))

            for idx, pose_cmd in enumerate(self.pose_targets):
                if not self._wait_until(next_send_ts):
                    return
                yaw_cmd = None if self.joint_targets is None else self.joint_targets[idx]
                if yaw_cmd is not None:
                    self.current_semantic_yaw = float(yaw_cmd)
                resp = self.daemon.servo_pose(pose_cmd)
                self._check_servo_response(resp, context=f"{self.label} servo pose")
                next_send_ts += float(SERVO_CONTROL_DT)
                late_by = time.monotonic() - next_send_ts
                if late_by > float(SERVO_CONTROL_DT):
                    next_send_ts = time.monotonic()

            self.stream_finished = True
            while not self._stop_event.is_set():
                if self.target_yaw is not None:
                    self.current_semantic_yaw = float(self.target_yaw)
                if not self._wait_until(next_send_ts):
                    return
                resp = self.daemon.servo_pose(self.target_real)
                self._check_servo_response(resp, context=f"{self.label} final servo hold")
                next_send_ts += float(SERVO_CONTROL_DT)
                late_by = time.monotonic() - next_send_ts
                if late_by > float(SERVO_CONTROL_DT):
                    next_send_ts = time.monotonic()
        except Exception as exc:
            self.error = exc
        finally:
            if self.started_servo:
                try:
                    self.daemon.servo_stop()
                except Exception:
                    try:
                        stop_motion_and_confirm(self.daemon, f"{self.label} cleanup")
                    except Exception:
                        pass
            self._done_event.set()


def _execute_servo_segment(
    daemon,
    target_real: np.ndarray,
    *,
    label: str,
    speed_mps: float,
    record: bool,
    cameras: CameraPair | None = None,
    gripper: float = 1.0,
    start_frame_idx: int = 0,
    timeout_s: float = DEFAULT_ASYNC_MOVE_TIMEOUT_S,
    target_yaw: float | None = None,
    semantic_yaw: float | None = None,
    force_live_mode: bool = True,
    reuse_servo: bool = False,
    angular_speed_radps: float | None = None,
    yaw_speed_radps: float | None = None,
) -> list["RecordedFrame"]:
    from support.tcp_control import get_robot_snapshot

    target_real = np.asarray(target_real, dtype=np.float64).reshape(6).copy()
    start_real: np.ndarray | None = None
    if reuse_servo and hasattr(daemon, "get_servo_reference_pose"):
        try:
            ref_pose = daemon.get_servo_reference_pose()
        except Exception:
            ref_pose = None
        if ref_pose is not None:
            start_real = np.asarray(ref_pose, dtype=np.float64).reshape(6).copy()
    if start_real is None:
        start_snap = get_robot_snapshot()
        start_real = np.asarray(start_snap.tcp_pose, dtype=np.float64).reshape(6).copy()
    start_yaw = float(_wrap_angle(start_real[5]))
    pose_targets, joint_targets = _build_servo_pose_targets(
        start_real,
        target_real,
        speed_mps=speed_mps,
        start_yaw=start_yaw if target_yaw is not None else None,
        target_yaw=target_yaw,
        angular_speed_radps=angular_speed_radps,
        yaw_speed_radps=yaw_speed_radps,
    )

    require_force_guard = bool(force_live_mode and float(target_real[2]) < float(start_real[2]) - 1e-6)
    frames: list[RecordedFrame] = []
    frame_idx = int(start_frame_idx)
    next_record_deadline = time.monotonic()
    last_record_ts: float | None = None
    per_step_budget_s = 0.08 if record else 0.04
    stream_budget_s = float(len(pose_targets)) * float(per_step_budget_s) + 5.0
    deadline = time.monotonic() + max(0.5, float(timeout_s), float(stream_budget_s))
    initial_semantic_yaw = float(start_yaw if semantic_yaw is None else semantic_yaw)

    def _record_snapshot(snap, current_semantic_yaw: float, *, force: bool = False) -> None:
        nonlocal frame_idx, next_record_deadline, last_record_ts
        if not record or cameras is None:
            return
        now = time.monotonic()
        if not force and frames and now < next_record_deadline:
            return
        actual_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
        joint_q = np.asarray(snap.joint_q, dtype=np.float64).reshape(-1)
        actual_yaw = _require_yaw_readback(
            joint_q,
            context=f"{label} record snapshot frame_idx={frame_idx}",
        )
        frames.append(
            _capture_recorded_frame(
                cameras,
                actual_real,
                current_semantic_yaw,
                actual_yaw,
                gripper=gripper,
                frame_idx=frame_idx,
            )
        )
        frame_idx += 1
        last_record_ts = now
        next_record_deadline = now + CONTROL_DT
    runner = _SegmentServoRunner(
        daemon,
        label=label,
        start_real=start_real,
        target_real=target_real,
        pose_targets=pose_targets,
        target_yaw=target_yaw,
        joint_targets=joint_targets,
        semantic_yaw=initial_semantic_yaw,
        require_force_guard=require_force_guard,
        force_live_mode=force_live_mode,
        reuse_servo=reuse_servo,
        yaw_speed_radps=yaw_speed_radps,
    )
    runner.start()

    monitor_dt_s = 0.005
    try:
        while True:
            if runner.error is not None:
                raise runner.error

            snap = get_robot_snapshot()
            _record_snapshot(snap, float(runner.current_semantic_yaw), force=not frames)

            actual_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
            joint_q = np.asarray(snap.joint_q, dtype=np.float64).reshape(-1)
            actual_yaw_readback = _require_yaw_readback(
                joint_q,
                context=f"{label} final hold snapshot",
            )
            actual_tcp_yaw = float(_wrap_angle(actual_real[5]))
            yaw_ok = (
                target_yaw is None
                or abs(_wrap_angle(actual_tcp_yaw - float(target_yaw))) <= DEFAULT_YAW_EXEC_TOL_RAD
            )

            if runner.stream_finished and _pose_close(actual_real, target_real) and yaw_ok:
                force_final_record = (
                    record
                    and (last_record_ts is None or (time.monotonic() - last_record_ts) > (0.5 * CONTROL_DT))
                )
                _record_snapshot(snap, float(runner.current_semantic_yaw), force=force_final_record)
                runner.request_stop()
                runner.join(timeout=2.0)
                if runner.error is not None:
                    raise runner.error
                break

            if runner.stream_finished and require_force_guard and runner.last_force_guard_adjusted:
                deadline = max(deadline, time.monotonic() + max(0.5, float(CONTROL_DT) * 4.0))
                if runner.last_force_guard_scale is not None and runner.last_force_guard_scale <= 1e-6:
                    _record_snapshot(snap, float(runner.current_semantic_yaw), force=True)
                    runner.request_stop()
                    runner.join(timeout=2.0)
                    if runner.error is not None:
                        raise runner.error
                    break

            if time.monotonic() > deadline:
                runner.request_stop()
                runner.join(timeout=2.0)
                if runner.error is not None:
                    raise runner.error
                if not runner.stream_finished:
                    raise RuntimeError(f"{label} timed out while streaming to target {np.round(target_real, 5).tolist()}")
                raise RuntimeError(
                    f"{label} timed out waiting for target "
                    f"(target={np.round(target_real, 5).tolist()}, actual={np.round(actual_real, 5).tolist()}, "
                    f"target_yaw={None if target_yaw is None else round(float(target_yaw), 6)}, "
                    f"actual_tcp_yaw={round(actual_tcp_yaw, 6)}, "
                    f"actual_yaw_readback={round(actual_yaw_readback, 6)})"
                )

            time.sleep(monitor_dt_s)
    finally:
        runner.request_stop()
        runner.join(timeout=2.0)

    return frames


def execute_pose_move_and_wait(
    daemon,
    target_real: np.ndarray,
    label: str,
    *,
    speed_mps: float,
    target_yaw: float | None = None,
    timeout_s: float = DEFAULT_ASYNC_MOVE_TIMEOUT_S,
    force_live_mode: bool = True,
    reuse_servo: bool = False,
    angular_speed_radps: float | None = None,
    yaw_speed_radps: float | None = None,
) -> dict[str, object]:
    _execute_servo_segment(
        daemon,
        target_real,
        label=label,
        speed_mps=speed_mps,
        record=False,
        target_yaw=target_yaw,
        timeout_s=timeout_s,
        force_live_mode=force_live_mode,
        reuse_servo=reuse_servo,
        angular_speed_radps=angular_speed_radps,
        yaw_speed_radps=yaw_speed_radps,
    )
    return {"servo_execute": "ok"}


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
    yaw: float
    yaw_readback: float
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
    target_yaw: float | None = None,
    semantic_yaw: float | None = None,
    force_live_mode: bool = True,
    record: bool = True,
    reuse_servo: bool = False,
    angular_speed_radps: float | None = None,
    yaw_speed_radps: float | None = None,
) -> list[RecordedFrame]:
    """Execute one servo segment at 100Hz and record executed state at 30Hz."""
    return _execute_servo_segment(
        daemon,
        target_real,
        label=f"recorded servo frame {start_frame_idx}",
        speed_mps=speed_mps,
        record=record,
        cameras=cameras,
        gripper=gripper,
        start_frame_idx=start_frame_idx,
        timeout_s=timeout_s,
        target_yaw=target_yaw,
        semantic_yaw=semantic_yaw,
        force_live_mode=force_live_mode,
        reuse_servo=reuse_servo,
        angular_speed_radps=angular_speed_radps,
        yaw_speed_radps=yaw_speed_radps,
    )


def capture_manual_snapshot(
    cameras: CameraPair,
    *,
    gripper: float,
    frame_idx: int,
    semantic_yaw: float,
) -> RecordedFrame:
    from support.tcp_control import get_robot_snapshot

    snap = get_robot_snapshot()
    actual_real = np.asarray(snap.tcp_pose, dtype=np.float64).reshape(6).copy()
    joint_q = np.asarray(snap.joint_q, dtype=np.float64).reshape(-1)
    readback_yaw = _require_yaw_readback(
        joint_q,
        context=f"manual snapshot frame_idx={frame_idx}",
    )
    return _capture_recorded_frame(
        cameras,
        actual_real,
        float(semantic_yaw),
        float(readback_yaw),
        gripper=gripper,
        frame_idx=frame_idx,
    )


def prepare_episode_for_save(
    frames: list[RecordedFrame],
    *,
    save_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw episode frames onto the target save grid."""
    if not frames:
        raise RuntimeError("no frames to resample")
    if save_fps not in (RAW_CAPTURE_FPS, UPSAMPLED_SAVE_FPS):
        raise ValueError(f"unsupported save_fps={save_fps}, expected {RAW_CAPTURE_FPS} or {UPSAMPLED_SAVE_FPS}")

    raw_times = np.array([float(frame.timestamp) for frame in frames], dtype=np.float64)
    raw_pose6 = np.stack([np.asarray(frame.sim_pose6, dtype=np.float64).reshape(6) for frame in frames], axis=0)
    raw_gripper = np.array([float(frame.gripper) for frame in frames], dtype=np.float32)
    raw_yaw = np.array([float(frame.yaw) for frame in frames], dtype=np.float32)
    raw_yaw_readback = np.array([float(frame.yaw_readback) for frame in frames], dtype=np.float32)
    raw_main_images = np.stack([frame.main_image for frame in frames], axis=0)
    raw_wrist_images = np.stack([frame.wrist_image for frame in frames], axis=0)

    if save_fps == RAW_CAPTURE_FPS:
        target_count = len(frames)
        env_steps = np.arange(target_count, dtype=np.int64)
        target_times = env_steps.astype(np.float32) * np.float32(1.0 / RAW_CAPTURE_FPS)
        state_dim = len(state_schema())
        states = np.zeros((target_count, state_dim), dtype=np.float32)
        pose6_saved = raw_pose6.astype(np.float32, copy=True)
        gripper_saved = raw_gripper.astype(np.float32, copy=True)
        yaw_saved = raw_yaw.astype(np.float32, copy=True)
        main_images = raw_main_images.copy()
        wrist_images = raw_wrist_images.copy()
        for out_idx in range(target_count):
            states[out_idx] = build_state_row(
                pose6_saved[out_idx],
                gripper_saved[out_idx],
                yaw_saved[out_idx],
            )
        actions = compute_actions_from_saved_states(states)
        return (
            states,
            actions,
            target_times.astype(np.float32),
            env_steps,
            main_images,
            wrist_images,
        )

    save_control_dt = 1.0 / float(save_fps)
    target_count = max(1, int(round((len(frames) - 1) * save_fps / RAW_CAPTURE_FPS)) + 1)
    env_steps = np.arange(target_count, dtype=np.int64)
    target_times = env_steps.astype(np.float32) * np.float32(save_control_dt)

    state_dim = len(state_schema())
    states = np.zeros((target_count, state_dim), dtype=np.float32)
    pose6_resampled = np.zeros((target_count, 6), dtype=np.float32)
    gripper_resampled = np.zeros((target_count,), dtype=np.float32)
    yaw_resampled = np.zeros((target_count,), dtype=np.float32)
    yaw_readback_resampled = np.zeros((target_count,), dtype=np.float32)
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
            interp_yaw = float(raw_yaw[left])
            interp_yaw_readback = float(raw_yaw_readback[left])
        else:
            yaw_delta = _wrap_angle(float(raw_yaw[right] - raw_yaw[left]))
            interp_yaw = float(raw_yaw[left] + alpha * yaw_delta)
            yaw_readback_delta = _wrap_angle(float(raw_yaw_readback[right] - raw_yaw_readback[left]))
            interp_yaw_readback = float(raw_yaw_readback[left] + alpha * yaw_readback_delta)

        pose6_resampled[out_idx] = interp_pose6.astype(np.float32)
        gripper_resampled[out_idx] = float(raw_gripper[grip_idx])
        yaw_resampled[out_idx] = np.float32(interp_yaw)
        yaw_readback_resampled[out_idx] = np.float32(interp_yaw_readback)
        states[out_idx] = build_state_row(
            interp_pose6,
            gripper_resampled[out_idx],
            yaw_resampled[out_idx],
        )
        main_images[out_idx] = raw_main_images[image_idx]
        wrist_images[out_idx] = raw_wrist_images[image_idx]

    actions = compute_actions_from_saved_states(states)
    return (
        states,
        actions,
        target_times.astype(np.float32),
        env_steps,
        main_images,
        wrist_images,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_episode(
    frames: list[RecordedFrame],
    save_dir: Path,
    prompt: str,
    *,
    save_fps: int,
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
    )
    state_names = state_schema()
    action_names = action_schema()

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
        "state_schema": state_names,
        "action_schema": action_names,
        "state_mode": STATE_MODE_YAW,
        "pose_frame": get_alignment_mode(),
    }
    with open(episode_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"  Episode {episode_id} saved: {episode_dir} "
        f"(raw {len(frames)} -> saved {states.shape[0]} frames @ {save_fps}Hz)"
    )
    return episode_dir


@dataclass
class CollectTaskRuntime:
    daemon: object
    cameras: CameraPair
    dry_run: bool
    gripper_timeout: float
    linear_speed: float
    min_tcp_z: float
    approach_z_offset_m: float
    cube_height_m: float
    place_height_offset_m: float
    home_real: np.ndarray
    origin_xy: np.ndarray
    default_yaw_rad: float = DEFAULT_YAW_HOME_RAD
    local_exec_yaw_rad: float = DEFAULT_YAW_HOME_RAD
    default_yaw_exec_tol_rad: float = DEFAULT_YAW_EXEC_TOL_RAD
    cleanup_scene_state: dict[str, dict[str, object]] | None = None
    runtime_held_object: str | None = None
    held_servo_depth: int = 0

    def wrap_angle(self, angle: float) -> float:
        return _wrap_angle(angle)

    def build_pose_at_xy(self, base_pose: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
        return build_pose_at_xy(base_pose, x, y, z)

    def build_pose_at_xy_yaw(self, base_pose: np.ndarray, x: float, y: float, z: float, yaw: float) -> np.ndarray:
        pose = build_pose_at_xy(base_pose, x, y, z)
        pose[5] = self.wrap_angle(float(yaw))
        return pose

    def refresh_home_pose(self) -> np.ndarray:
        from support.pose_align import set_runtime_alignment
        from support.tcp_control import get_robot_snapshot

        snap_local = get_robot_snapshot()
        set_runtime_alignment(snap_local.tcp_pose)
        self.home_real = np.asarray(snap_local.tcp_pose, dtype=np.float64).reshape(6).copy()
        self.origin_xy = self.home_real[:2].copy()
        return self.home_real.copy()

    def build_pose_from_live_orientation(self, x: float, y: float, z: float) -> np.ndarray:
        from support.tcp_control import get_robot_snapshot

        snap_local = get_robot_snapshot()
        live_base = np.asarray(snap_local.tcp_pose, dtype=np.float64).reshape(6).copy()
        return build_pose_at_xy(live_base, x, y, z)

    def build_pose_from_live_orientation_yaw(self, x: float, y: float, z: float, yaw: float) -> np.ndarray:
        pose = self.build_pose_from_live_orientation(x, y, z)
        pose[5] = self.wrap_angle(float(yaw))
        return pose

    def get_live_tcp_pose(self) -> np.ndarray:
        from support.tcp_control import get_robot_snapshot

        snap_local = get_robot_snapshot()
        return np.asarray(snap_local.tcp_pose, dtype=np.float64).reshape(6).copy()

    def get_live_yaw(self) -> float | None:
        from support.tcp_control import get_robot_snapshot

        snap_local = get_robot_snapshot()
        joint_q = np.asarray(snap_local.joint_q, dtype=np.float64).reshape(-1)
        if joint_q.size < 6:
            return None
        return float(joint_q[5])

    def yaw_target_from_deg(self, deg: float) -> float:
        return self.wrap_angle(float(self.home_real[5]) + float(np.deg2rad(float(deg))))

    def begin_stream_servo(self, start_pose_real: np.ndarray) -> None:
        if self.dry_run:
            return
        resp = self.daemon.servo_start(SERVO_CONTROL_DT)
        if int(resp.get("servo_start_ret", -1)) != 0:
            raise RuntimeError(f"servo_start failed: {resp}")
        self.daemon.servo_begin_chunk(np.asarray(start_pose_real, dtype=np.float64).reshape(6).copy(), force_live_mode=False)

    def stop_stream_servo(self) -> None:
        if self.dry_run:
            return
        self.daemon.servo_stop()

    @property
    def hold_servo_active(self) -> bool:
        return bool(self.held_servo_depth > 0)

    def begin_task_servo(self) -> None:
        if self.dry_run:
            return
        if self.held_servo_depth == 0:
            self.begin_stream_servo(self.get_live_tcp_pose())
        self.held_servo_depth += 1

    def end_task_servo(self) -> None:
        if self.dry_run:
            return
        if self.held_servo_depth <= 0:
            return
        self.held_servo_depth -= 1
        if self.held_servo_depth == 0:
            self.stop_stream_servo()

    def send_stream_pose(self, target_real: np.ndarray) -> dict[str, object]:
        pose_cmd = np.asarray(target_real, dtype=np.float64).reshape(6).copy()
        if self.dry_run:
            return {"servo_pose_ret": 0, "dry_run": True}
        return self.daemon.servo_pose(pose_cmd)

    def return_home(self, label: str) -> np.ndarray:
        if not self.dry_run:
            execute_pose_move_and_wait(
                self.daemon,
                self.home_real,
                label,
                speed_mps=self.linear_speed,
                reuse_servo=self.hold_servo_active,
            )
            return self.refresh_home_pose()
        self.origin_xy = self.home_real[:2].copy()
        return self.home_real.copy()

    def move_pose(self, target_real: np.ndarray, label: str) -> None:
        if self.dry_run:
            return
        execute_pose_move_and_wait(
            self.daemon,
            np.asarray(target_real, dtype=np.float64).reshape(6).copy(),
            label,
            speed_mps=self.linear_speed,
            reuse_servo=self.hold_servo_active,
        )

    def z_for_pick_level(self, level: int) -> float:
        return float(self.min_tcp_z + self.cube_height_m * max(0, int(level)))

    def z_for_place_level(self, level: int) -> float:
        return float(self.z_for_pick_level(level) + self.place_height_offset_m)

    def record_pose_move(
        self,
        target_real: np.ndarray,
        *,
        gripper: float,
        start_frame_idx: int,
        record: bool,
        semantic_yaw: float | None = None,
        target_yaw: float | None = None,
        speed_mps: float | None = None,
        timeout_s: float | None = None,
        angular_speed_radps: float | None = None,
        yaw_speed_radps: float | None = None,
    ) -> list[RecordedFrame]:
        effective_semantic_yaw = (
            None
            if target_yaw is not None and semantic_yaw is None
            else (self.local_exec_yaw_rad if semantic_yaw is None else float(semantic_yaw))
        )
        frames = execute_and_record(
            self.daemon,
            self.cameras,
            target_real,
            gripper=gripper,
            start_frame_idx=start_frame_idx,
            speed_mps=self.linear_speed if speed_mps is None else float(speed_mps),
            target_yaw=target_yaw,
            semantic_yaw=effective_semantic_yaw,
            timeout_s=DEFAULT_ASYNC_MOVE_TIMEOUT_S if timeout_s is None else float(timeout_s),
            angular_speed_radps=angular_speed_radps,
            yaw_speed_radps=yaw_speed_radps,
            record=record,
            reuse_servo=self.hold_servo_active,
        )
        if target_yaw is not None:
            self.local_exec_yaw_rad = float(target_yaw)
        elif semantic_yaw is not None:
            self.local_exec_yaw_rad = float(semantic_yaw)
        return frames

    def capture_manual_snapshot(
        self,
        *,
        gripper: float,
        frame_idx: int,
        semantic_yaw: float | None = None,
    ) -> RecordedFrame:
        yaw = self.local_exec_yaw_rad if semantic_yaw is None else semantic_yaw
        return capture_manual_snapshot(
            self.cameras,
            gripper=gripper,
            frame_idx=frame_idx,
            semantic_yaw=float(yaw),
        )

    def make_dummy_frame(self, *, sim_pose: np.ndarray, gripper: float, yaw: float, frame_idx: int) -> RecordedFrame:
        return RecordedFrame(
            sim_pose6=np.asarray(sim_pose, dtype=np.float64).reshape(6).copy(),
            gripper=float(gripper),
            yaw=float(yaw),
            yaw_readback=float(yaw),
            main_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            wrist_image=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            timestamp=frame_idx * CONTROL_DT,
        )

    def command_gripper_state(self, target_state: int) -> bool:
        from support.gripper_control import command_gripper_state as _command_gripper_state

        return bool(_command_gripper_state(int(target_state), timeout_s=self.gripper_timeout))

    def ensure_gripper_ok(self, ok: bool, label: str) -> None:
        ensure_gripper_ok(ok, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-robot data collection for OpenPI.")
    # task is selected interactively at startup, not via CLI
    parser.add_argument("--prompt", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--speed", type=float, default=LINEAR_SPEED)
    parser.add_argument("--save-fps", type=int, choices=(RAW_CAPTURE_FPS, UPSAMPLED_SAVE_FPS), default=DEFAULT_SAVE_FPS)
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
    )
    if tui_cfg.quit:
        print("Exiting.")
        return 0

    selected_task = str(tui_cfg.task)
    if selected_task in {"open_and_close", "keyboard_teleop", "storage"}:
        tui_cfg.mode = "manual"
        tui_cfg.auto_episodes = 1
    auto_episodes = 0 if tui_cfg.mode == "manual" else int(tui_cfg.auto_episodes)
    save_fps = int(tui_cfg.save_fps)
    resume_mode = str(tui_cfg.resume_mode)

    print("\nConfiguration:")
    print(f"  Mode:       {tui_cfg.mode}")
    if tui_cfg.mode == "auto":
        print(f"  Episodes:   {auto_episodes}")
    print(f"  Resume:     {resume_mode}")
    print(f"  Task:       {selected_task}")
    print(f"  Save FPS:   {save_fps}")
    print(f"  Pose Frame: {get_alignment_mode()}")
    print(f"  Speed:      {LINEAR_SPEED} m/s")
    print(f"  Dry-run:    {args.dry_run}")
    if selected_task not in {"pick_and_place", "open_and_close", "keyboard_teleop", "storage"}:
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
    storage_planner_config = PlannerConfig(
        workspace_x_min=WORKSPACE_X_MIN,
        workspace_x_max=WORKSPACE_X_MAX,
        workspace_y_min=WORKSPACE_Y_MIN,
        workspace_y_max=WORKSPACE_Y_MAX,
        min_spacing_m=MIN_CUBE_SPACING_M,
        object_height_m=CUBE_HEIGHT_M,
        non_rotated_table_place_probability=0.5,
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
    if selected_task == "pick_and_place":
        print(f"Objects:      {list(OBJECT_ORDER)}")
        print("Prompt mode:  auto random ('pick up ...' / 'put ... on ...')")
        print("Task mix:     20% pick / 80% place")
    elif selected_task == "storage":
        print(f"Objects:      {list(OBJECT_ORDER)}")
        print("Prompt mode:  fixed order ('put the ... into storage basket')")
        print("Task mix:     one pass, sequential storage drop")
    elif selected_task == "keyboard_teleop":
        print("Prompt mode:  manual text prompt")
        print("Task mix:     operator-controlled keyboard teleop")
    else:
        print("Prompt mode:  fixed pair ('open the storage box' / 'close the storage box')")
        print("Task mix:     each cycle runs open, then close and saves two episodes")

    saved_held_object = _normalize_held_object(saved_state_preview.get("held_object")) if saved_state_preview else None
    if selected_task == "pick_and_place" and resume_mode == "continue" and saved_held_object is not None:
        raise RuntimeError(
            f"saved state indicates the gripper is still holding '{saved_held_object}'. "
            "Please manually restore the scene and restart with resume=reset."
        )
    saved_storage_state = saved_state_preview.get("storage_state") if saved_state_preview else None
    saved_storage_held_object = None
    if isinstance(saved_storage_state, dict):
        saved_storage_held_object = _normalize_held_object(saved_storage_state.get("held_object"))
    if selected_task == "storage" and resume_mode == "continue" and saved_storage_held_object is not None:
        raise RuntimeError(
            f"saved storage state indicates the gripper is still holding '{saved_storage_held_object}'. "
            "Please manually restore the scene and restart with resume=reset."
        )

    cameras: CameraPair | None = None
    daemon = None
    runtime: CollectTaskRuntime | None = None
    pick_session = PickAndPlaceSession(scene_state={})
    open_close_session = OpenCloseSession()
    storage_session = StorageSession(scene_state={})
    cleanup_state = {"done": False}

    def cleanup_collection() -> None:
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        latest_scene_state = pick_session.scene_state
        held_object = None
        if runtime is not None and selected_task == "pick_and_place":
            latest_scene_state = runtime.cleanup_scene_state if runtime.cleanup_scene_state is not None else pick_session.scene_state
            held_object = runtime.runtime_held_object
        elif runtime is not None and selected_task == "storage":
            latest_scene_state = runtime.cleanup_scene_state if runtime.cleanup_scene_state is not None else storage_session.scene_state
            held_object = runtime.runtime_held_object
        if selected_task == "pick_and_place" and latest_scene_state:
            save_collect_state(
                save_dir,
                latest_scene_state,
                pick_session.task_index,
                pick_session.episode_count,
                held_object=held_object,
            )
            print(f"  State saved to {_state_file_path(save_dir)}")
        elif selected_task == "storage" and latest_scene_state:
            save_collect_state(
                save_dir,
                {},
                0,
                0,
                storage_state={
                    "scene_state": latest_scene_state,
                    "next_index": int(storage_session.next_index),
                    "episode_count": int(storage_session.episode_count),
                    "held_object": _normalize_held_object(held_object),
                },
            )
            print(f"  State saved to {_state_file_path(save_dir)}")
        elif selected_task == "open_and_close" and open_close_session.reference is not None:
            save_collect_state(
                save_dir,
                {},
                0,
                open_close_session.episode_count,
                open_close_reference=open_close_session.reference,
                obstacle_scene=open_close_session.obstacle_scene,
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

    # Wait for robot to settle before issuing any motion
    for _retry in range(10):
        snap_check = get_robot_snapshot()
        status = daemon.motion_status()
        if _coerce_bool(status.get("is_steady")):
            break
        time.sleep(0.3)
    else:
        print("  WARNING: robot did not reach steady state after startup stop")

    startup_snap = get_robot_snapshot()
    startup_tcp_pose = np.asarray(startup_snap.tcp_pose, dtype=np.float64).reshape(6).copy()

    # For open_and_close: lift to target_z at startup xy before going to initial joints
    if selected_task == "open_and_close" and not args.dry_run:
        lift_z = float(MIN_TCP_Z + 0.20)
        if startup_tcp_pose[2] < lift_z:
            lift_pose = startup_tcp_pose.copy()
            lift_pose[2] = lift_z
            print(f"  Lifting to z={lift_z:.4f} m at startup xy before homing...")
            execute_pose_move_and_wait(
                daemon,
                lift_pose,
                "lift before homing",
                speed_mps=LINEAR_SPEED,
            )

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
    runtime = CollectTaskRuntime(
        daemon=daemon,
        cameras=cameras,
        dry_run=bool(args.dry_run),
        gripper_timeout=float(args.gripper_timeout),
        linear_speed=float(LINEAR_SPEED),
        min_tcp_z=float(MIN_TCP_Z),
        approach_z_offset_m=float(APPROACH_Z_OFFSET_M),
        cube_height_m=float(CUBE_HEIGHT_M),
        place_height_offset_m=float(PLACE_HEIGHT_OFFSET_M),
        home_real=home_real.copy(),
        origin_xy=origin_xy.copy(),
        default_yaw_rad=float(DEFAULT_YAW_HOME_RAD),
        local_exec_yaw_rad=float(DEFAULT_YAW_HOME_RAD),
        default_yaw_exec_tol_rad=float(DEFAULT_YAW_EXEC_TOL_RAD),
    )
    print(f"  Init real TCP: {np.round(init_real, 5).tolist()}")
    print(f"  Init policy TCP:  {np.round(init_sim, 5).tolist()}")

    if not args.dry_run:
        print("Opening gripper...")
        ensure_gripper_ok(
            command_gripper_state(1, timeout_s=args.gripper_timeout),
            "open gripper before collection",
        )

    saved = saved_state_preview
    if selected_task == "pick_and_place":
        pick_session, should_clear_pick_state = pp_restore_session(saved, resume_mode=resume_mode)
        if should_clear_pick_state:
            clear_collect_state(save_dir)
            pick_session = PickAndPlaceSession(scene_state={})
            print("  State cleared. Starting fresh.")
    elif selected_task == "storage":
        storage_session, should_clear_storage_state = st_restore_session(saved, resume_mode=resume_mode)
        if should_clear_storage_state:
            clear_storage_state(save_dir)
            storage_session = StorageSession(scene_state={})
            print("  Storage state cleared. Starting fresh.")
    else:
        open_close_session, should_clear_open_close_state = oc_restore_session(saved, resume_mode=resume_mode)
        if should_clear_open_close_state:
            clear_open_close_state(save_dir)
            open_close_session = OpenCloseSession()
            print("\n  Open/close state cleared. Reference will be re-captured from current TCP.")

    if selected_task == "keyboard_teleop":
        prompt_text = args.prompt.strip()
        if not prompt_text:
            prompt_text = input("Prompt: ").strip()
        if not prompt_text:
            print("Empty prompt. Exiting.")
            cleanup_collection()
            return 0
        print(f"\nKeyboard prompt: {prompt_text}")
        kt_run_session(
            runtime,
            config=KeyboardTeleopConfig(
                prompt=prompt_text,
                save_fps=save_fps,
                workspace_x_min=WORKSPACE_X_MIN,
                workspace_x_max=WORKSPACE_X_MAX,
                workspace_y_min=WORKSPACE_Y_MIN,
                workspace_y_max=WORKSPACE_Y_MAX,
            ),
            save_dir=save_dir,
            save_episode_fn=save_episode,
        )
        cleanup_collection()
        print("Done.")
        return 0

    if selected_task == "pick_and_place":
        pp_prepare_session(runtime, pick_session, config=planner_config)
        save_collect_state(
            save_dir,
            pick_session.scene_state,
            pick_session.task_index,
            pick_session.episode_count,
        )
    elif selected_task == "storage":
        st_prepare_session(runtime, storage_session, config=storage_planner_config)
        save_collect_state(
            save_dir,
            {},
            0,
            0,
            storage_state={
                "scene_state": storage_session.scene_state,
                "next_index": int(storage_session.next_index),
                "episode_count": int(storage_session.episode_count),
                "held_object": None,
            },
        )
    else:
        if open_close_session.reference is None:
            set_runtime_alignment(startup_tcp_pose)
        oc_prepare_session(
            runtime,
            open_close_session,
            startup_tcp_pose=startup_tcp_pose,
            origin_xy=runtime.origin_xy,
            workspace_x_min=WORKSPACE_X_MIN,
            workspace_x_max=WORKSPACE_X_MAX,
            workspace_y_min=WORKSPACE_Y_MIN,
            workspace_y_max=WORKSPACE_Y_MAX,
            min_spacing=OC_CLEAR_SPACING,
        )
        save_collect_state(
            save_dir,
            {},
            0,
            open_close_session.episode_count,
            open_close_reference=open_close_session.reference,
            obstacle_scene=open_close_session.obstacle_scene,
        )

    max_ep = auto_episodes if auto_episodes > 0 else args.max_episodes

    print("\n=== Ready For Episodes ===")
    if auto_episodes > 0:
        print(f"Auto mode: {auto_episodes} episodes, no pause between episodes.")
    else:
        if selected_task == "storage":
            print("Manual mode: start once, then the task runs through the remaining objects sequentially.")
        else:
            print("Manual mode: press ENTER to collect the next random episode.")
    if selected_task == "pick_and_place":
        print("Task policy: 20% pick / 80% place")
    elif selected_task == "storage":
        print("Task policy: fixed order red -> green -> blue -> apple")
        print("Drop policy: move to basket point while unwinding yaw to baseline")
        print("Stop policy: task exits automatically after all objects are stored")
    else:
        print("Task policy: each cycle performs open, then close")
        print("Save policy: split into one 'open' episode and one 'close' episode")
        print("Gripper policy: always open (no close/open action during episode)")
    print("Quit with q / quit / exit.\n")

    try:
        while True:
            current_episode_count = (
                pick_session.episode_count
                if selected_task == "pick_and_place"
                else (storage_session.episode_count if selected_task == "storage" else open_close_session.episode_count)
            )
            if 0 < max_ep <= current_episode_count:
                print(f"\nReached {current_episode_count} episodes. Done.")
                break
            if selected_task == "open_and_close" and 0 < max_ep and (max_ep - current_episode_count) < 2:
                print(
                    f"\nReached {current_episode_count} episodes. "
                    "Open/close collection saves 2 episodes per cycle, so the remaining budget is insufficient."
                )
                break

            if selected_task == "pick_and_place":
                plan = pp_plan_next_episode(pick_session, config=planner_config)
                pp_describe_episode(plan, episode_count=pick_session.episode_count)
            elif selected_task == "storage":
                if not st_has_remaining_objects(storage_session):
                    print(f"\nAll storage objects completed after {storage_session.episode_count} episodes.")
                    break
                plan = st_plan_next_episode(storage_session)
                remaining_after_save = max(0, len(OBJECT_ORDER) - (storage_session.next_index + 1))
                st_describe_episode(
                    plan,
                    episode_count=storage_session.episode_count,
                    remaining_count=remaining_after_save,
                )
            else:
                cycle_plan = oc_plan_cycle(
                    runtime,
                    open_close_session,
                    workspace_x_min=WORKSPACE_X_MIN,
                    workspace_x_max=WORKSPACE_X_MAX,
                    workspace_y_min=WORKSPACE_Y_MIN,
                    workspace_y_max=WORKSPACE_Y_MAX,
                    target_z=float(MIN_TCP_Z + 0.20),
                    press_z=float(MIN_TCP_Z + 0.05),
                    clear_spacing=OC_CLEAR_SPACING,
                    stack_probability=OC_STACK_PROBABILITY,
                    band_count_min=OC_BAND_COUNT_MIN,
                    band_count_max=OC_BAND_COUNT_MAX,
                )
                oc_describe_cycle(cycle_plan)

            if auto_episodes > 0:
                if selected_task == "pick_and_place":
                    print(f"  Auto: recording ({pick_session.episode_count + 1}/{auto_episodes})")
                elif selected_task == "storage":
                    print(
                        "  Auto: recording storage episode "
                        f"{storage_session.episode_count + 1}/{len(OBJECT_ORDER)}"
                    )
                else:
                    print(
                        "  Auto: recording cycle -> episodes "
                        f"{open_close_session.episode_count + 1}-{open_close_session.episode_count + 2}/{auto_episodes}"
                    )
            else:
                if selected_task == "storage":
                    print(
                        "  Storage: recording next object automatically "
                        f"({storage_session.next_index + 1}/{len(OBJECT_ORDER)})"
                    )
                else:
                    prompt_text = (
                        "  Press ENTER to record this cycle, or q to quit: "
                        if selected_task == "open_and_close"
                        else "  Press ENTER to record this episode, or q to quit: "
                    )
                    cmd = input(prompt_text).strip().lower()
                    if cmd in QUIT_TOKENS:
                        print("Exit requested.")
                        break
                    if cmd:
                        print(f"Unsupported input '{cmd}', exiting.")
                        break

            runtime.begin_task_servo()
            try:
                if not args.dry_run:
                    runtime.return_home("pre-episode return home")
                    ensure_gripper_ok(
                        command_gripper_state(1, timeout_s=args.gripper_timeout),
                        "open gripper before episode",
                    )

                if selected_task == "pick_and_place":
                    recorded = pp_record_episode(runtime, pick_session, plan)
                    save_episode(
                        recorded.frames,
                        save_dir,
                        recorded.plan.prompt,
                        save_fps=save_fps,
                    )
                    pp_finalize_episode(runtime, pick_session, recorded)
                elif selected_task == "storage":
                    recorded = st_record_episode(runtime, storage_session, plan)
                    save_episode(
                        recorded.frames,
                        save_dir,
                        recorded.plan.prompt,
                        save_fps=save_fps,
                    )
                    st_finalize_episode(runtime, storage_session, recorded)
                else:
                    recorded_cycle = oc_record_cycle(runtime, open_close_session, cycle_plan)
                    save_episode(
                        recorded_cycle.combined_open_frames,
                        save_dir,
                        recorded_cycle.plan.open_plan.prompt,
                        save_fps=save_fps,
                    )
                    save_episode(
                        recorded_cycle.close_frames,
                        save_dir,
                        recorded_cycle.plan.close_plan.prompt,
                        save_fps=save_fps,
                    )
                    oc_finalize_cycle(open_close_session, recorded_cycle)

                if not args.dry_run:
                    runtime.return_home("post-episode return home")
            finally:
                runtime.end_task_servo()

            if selected_task == "pick_and_place":
                save_collect_state(
                    save_dir,
                    pick_session.scene_state,
                    pick_session.task_index,
                    pick_session.episode_count,
                )
            elif selected_task == "storage":
                save_collect_state(
                    save_dir,
                    {},
                    0,
                    0,
                    storage_state={
                        "scene_state": storage_session.scene_state,
                        "next_index": int(storage_session.next_index),
                        "episode_count": int(storage_session.episode_count),
                        "held_object": None,
                    },
                )
            else:
                save_collect_state(
                    save_dir,
                    {},
                    0,
                    open_close_session.episode_count,
                    open_close_reference=open_close_session.reference,
                    obstacle_scene=open_close_session.obstacle_scene,
                )
            continue
    except KeyboardInterrupt:
        stopped_episode_count = (
            pick_session.episode_count
            if selected_task == "pick_and_place"
            else (storage_session.episode_count if selected_task == "storage" else open_close_session.episode_count)
        )
        print(f"\n\nCollection stopped after {stopped_episode_count} episodes.")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        cleanup_collection()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

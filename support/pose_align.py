#!/usr/bin/env python3
"""
Decoupled pose alignment helpers for mapping between the real robot frame and
the policy frame.

Two runtime modes are supported:

* ``sim``: keep the existing real->sim alignment behavior.
* ``real``: bypass the alignment and use the real robot frame directly.

The key insight: TCP pose ``[x, y, z, rx, ry, rz]`` uses base-frame
coordinates for *position* (z always points up) and Euler-ZYX angles for
*orientation* (TCP frame relative to the base).  A naïve SE(3) transform
``T_sim × T_real⁻¹`` mixes the orientation convention difference (e.g.
``rx ≈ π`` on the real robot vs ``rx ≈ 0`` in MuJoCo) into the position
mapping, which **flips the z axis** for world-frame displacements.

This module therefore uses a **decoupled** alignment:

* **Position** — a pure yaw rotation ``Rz(θ)`` (extracted from the full
  SE(3)) plus a translation offset.  This preserves the physical z direction.
* **Orientation** — a rotation-matrix composition ``R_sim = R_offset @ R_real``
  that correctly accounts for the full orientation difference including roll.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Shared real/sim pose-alignment configuration
# ---------------------------------------------------------------------------

SIM_INIT_QPOS_RAD = np.array(
    [
        1.57,
        -0.2617993877991494,
        1.7453292519943295,
        0.4363323129985824,
        1.5707963267948966,
        0.0,
    ],
    dtype=np.float64,
)

REAL_INIT_QPOS_RAD = np.array(
    [
        0.0,
        -0.17453292519943295,
        1.3962634015954636,
        0.0,
        1.5707963267948966,  
        0.11434,        
    ],
    dtype=np.float64,
)

# Computed from the MuJoCo grip_site pose under SIM_INIT_QPOS_RAD.
SIM_INIT_TCP_POSE6 = np.array(
    [
        0.5219370412369736,
        0.14877297060308298,
        0.6021932707114368,
        3.329418420910386e-16,
        3.141717494153894e-16,
        -0.0007963267948964843,
    ],
    dtype=np.float64,
)


POSE_DIM = 6


# ---------------------------------------------------------------------------
# Euler / rotation helpers
# ---------------------------------------------------------------------------

def wrap_euler_zyx(euler: np.ndarray) -> np.ndarray:
    angles = np.asarray(euler, dtype=np.float64).reshape(3)
    return np.arctan2(np.sin(angles), np.cos(angles)).astype(np.float64)


def _euler_zyx_to_rot(euler_zyx: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler_zyx, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _rot_to_euler_zyx(rot: np.ndarray) -> np.ndarray:
    R = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    sy = np.hypot(R[0, 0], R[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return wrap_euler_zyx(np.array([roll, pitch, yaw], dtype=np.float64))


# ---------------------------------------------------------------------------
# 4x4 homogeneous transform helpers (kept for FK / general use)
# ---------------------------------------------------------------------------

def pose6_to_T(pose6: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose6, dtype=np.float64).reshape(POSE_DIM)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _euler_zyx_to_rot(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    transform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    pose = np.zeros(POSE_DIM, dtype=np.float64)
    pose[:3] = transform[:3, 3]
    pose[3:] = _rot_to_euler_zyx(transform[:3, :3])
    return pose


def invert_T(T: np.ndarray) -> np.ndarray:
    transform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = transform[:3, :3]
    t = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.float64).reshape(4, 4) @ np.asarray(B, dtype=np.float64).reshape(4, 4)


def build_T_sim_from_real(real_init_pose6: np.ndarray, sim_init_pose6: np.ndarray) -> np.ndarray:
    """Full (naïve) SE(3) — kept for reference / debugging only."""
    return compose_T(pose6_to_T(sim_init_pose6), invert_T(pose6_to_T(real_init_pose6)))


# ---------------------------------------------------------------------------
# Decoupled alignment context
# ---------------------------------------------------------------------------

@dataclass
class PoseAlignmentContext:
    frame_mode: str
    real_init_pose6: np.ndarray
    sim_init_pose6: np.ndarray
    # Position: yaw-only rotation + translation (4x4, z direction preserved)
    T_pos_sim_from_real: np.ndarray
    T_pos_real_from_sim: np.ndarray
    # Orientation: 3x3 rotation offset  (R_sim = R_offset @ R_real)
    R_ori_sim_from_real: np.ndarray
    R_ori_real_from_sim: np.ndarray


_runtime_alignment: PoseAlignmentContext | None = None
ALIGN_FRAME_SIM = "sim"
ALIGN_FRAME_REAL = "real"
ALIGN_FRAME_CHOICES = (ALIGN_FRAME_SIM, ALIGN_FRAME_REAL)
_runtime_alignment_mode = ALIGN_FRAME_SIM


def _normalize_alignment_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in ALIGN_FRAME_CHOICES:
        raise ValueError(f"unsupported pose frame '{mode}', expected one of {ALIGN_FRAME_CHOICES}")
    return normalized


def set_alignment_mode(mode: str) -> str:
    global _runtime_alignment_mode
    _runtime_alignment_mode = _normalize_alignment_mode(mode)
    return _runtime_alignment_mode


def get_alignment_mode() -> str:
    return _runtime_alignment_mode


def clear_runtime_alignment() -> None:
    global _runtime_alignment
    _runtime_alignment = None


def is_alignment_ready() -> bool:
    return get_alignment_mode() == ALIGN_FRAME_REAL or _runtime_alignment is not None


def _build_pos_transform(real_pos: np.ndarray, sim_pos: np.ndarray,
                         full_R: np.ndarray) -> np.ndarray:
    """Build a 4x4 position-mapping transform that uses only the yaw
    component of *full_R* (the rotation from the naïve SE(3)), so that
    the physical z direction is preserved."""
    yaw = float(np.arctan2(full_R[1, 0], full_R[0, 0]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    t = np.asarray(sim_pos, dtype=np.float64) - R_yaw @ np.asarray(real_pos, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_yaw
    T[:3, 3] = t
    return T


def set_runtime_alignment(
    real_init_pose6: np.ndarray,
    sim_init_pose6: np.ndarray = SIM_INIT_TCP_POSE6,
    frame_mode: str | None = None,
) -> PoseAlignmentContext:
    global _runtime_alignment
    if frame_mode is not None:
        set_alignment_mode(frame_mode)
    mode = get_alignment_mode()
    real_pose = np.asarray(real_init_pose6, dtype=np.float64).reshape(POSE_DIM).copy()

    if mode == ALIGN_FRAME_REAL:
        identity = np.eye(4, dtype=np.float64)
        _runtime_alignment = PoseAlignmentContext(
            frame_mode=mode,
            real_init_pose6=real_pose,
            sim_init_pose6=real_pose.copy(),
            T_pos_sim_from_real=identity.copy(),
            T_pos_real_from_sim=identity.copy(),
            R_ori_sim_from_real=np.eye(3, dtype=np.float64),
            R_ori_real_from_sim=np.eye(3, dtype=np.float64),
        )
        return _runtime_alignment

    sim_pose = np.asarray(sim_init_pose6, dtype=np.float64).reshape(POSE_DIM).copy()

    # --- position: yaw-only rotation (no z-flip) ---
    T_full = build_T_sim_from_real(real_pose, sim_pose)
    T_pos = _build_pos_transform(real_pose[:3], sim_pose[:3], T_full[:3, :3])

    # --- orientation: rotation composition ---
    R_real_init = _euler_zyx_to_rot(real_pose[3:])
    R_sim_init = _euler_zyx_to_rot(sim_pose[3:])
    R_ori_offset = R_sim_init @ R_real_init.T  # R_sim = R_offset @ R_real

    _runtime_alignment = PoseAlignmentContext(
        frame_mode=mode,
        real_init_pose6=real_pose,
        sim_init_pose6=sim_pose,
        T_pos_sim_from_real=T_pos,
        T_pos_real_from_sim=invert_T(T_pos),
        R_ori_sim_from_real=R_ori_offset,
        R_ori_real_from_sim=R_ori_offset.T,
    )
    return _runtime_alignment


def get_runtime_alignment() -> PoseAlignmentContext:
    if _runtime_alignment is None:
        raise RuntimeError(
            "pose alignment is not initialized; move to REAL_INIT_QPOS_RAD and call set_runtime_alignment() first"
        )
    return _runtime_alignment


def real_pose_to_sim(real_pose6: np.ndarray) -> np.ndarray:
    rp = np.asarray(real_pose6, dtype=np.float64).reshape(POSE_DIM)
    if get_alignment_mode() == ALIGN_FRAME_REAL:
        return rp.copy()
    ctx = get_runtime_alignment()
    # Position: yaw rotation + translation
    T = ctx.T_pos_sim_from_real
    sim_pos = T[:3, :3] @ rp[:3] + T[:3, 3]
    # Orientation: rotation composition
    R_real = _euler_zyx_to_rot(rp[3:])
    sim_ori = _rot_to_euler_zyx(ctx.R_ori_sim_from_real @ R_real)
    return np.concatenate([sim_pos, sim_ori])


def sim_pose_to_real(sim_pose6: np.ndarray) -> np.ndarray:
    sp = np.asarray(sim_pose6, dtype=np.float64).reshape(POSE_DIM)
    if get_alignment_mode() == ALIGN_FRAME_REAL:
        return sp.copy()
    ctx = get_runtime_alignment()
    # Position: inverse yaw rotation + translation
    T_inv = ctx.T_pos_real_from_sim
    real_pos = T_inv[:3, :3] @ sp[:3] + T_inv[:3, 3]
    # Orientation: inverse rotation composition
    R_sim = _euler_zyx_to_rot(sp[3:])
    real_ori = _rot_to_euler_zyx(ctx.R_ori_real_from_sim @ R_sim)
    return np.concatenate([real_pos, real_ori])


__all__ = [
    "POSE_DIM",
    "PoseAlignmentContext",
    "wrap_euler_zyx",
    "pose6_to_T",
    "T_to_pose6",
    "invert_T",
    "compose_T",
    "build_T_sim_from_real",
    "ALIGN_FRAME_SIM",
    "ALIGN_FRAME_REAL",
    "ALIGN_FRAME_CHOICES",
    "clear_runtime_alignment",
    "set_alignment_mode",
    "get_alignment_mode",
    "is_alignment_ready",
    "set_runtime_alignment",
    "get_runtime_alignment",
    "real_pose_to_sim",
    "sim_pose_to_real",
    "SIM_INIT_QPOS_RAD",
    "REAL_INIT_QPOS_RAD",
    "SIM_INIT_TCP_POSE6",
]

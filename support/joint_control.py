#!/usr/bin/env python3
"""
Generic joint-space control module for the real AUBO i5.

This module is intended to be imported by the future Jetson-side main program.
It exposes a reusable "move to joint target" API instead of baking in a single
"initial angle" concept.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from utils.path_utils import get_build_dir, get_sdk_root
from utils.runtime_config import (
    DEFAULT_AUBO_PASSWORD,
    DEFAULT_AUBO_RPC_PORT,
    DEFAULT_AUBO_USER,
    DEFAULT_ROBOT_IP,
)


DEFAULT_SDK_ROOT = str(get_sdk_root())
DEFAULT_HELPER_CPP = "joint_control_helper.cpp"
DEFAULT_HELPER_BIN = str(get_build_dir() / "joint_control_helper")

DEFAULT_PORT = DEFAULT_AUBO_RPC_PORT
DEFAULT_USER = DEFAULT_AUBO_USER
DEFAULT_PASSWORD = DEFAULT_AUBO_PASSWORD
DEFAULT_SPEED_DEG = 10.0
DEFAULT_ACC_DEG = 20.0
DEFAULT_SPEED_FRACTION = 1.0
JOINT_DIM = 6

# If every joint is within this tolerance of the target, skip the move and
# report success immediately.  AUBO SDK's moveJoint returns error 13 when the
# commanded delta is essentially zero.
ALREADY_AT_TARGET_RAD = 1e-3  # ~0.057 deg


@dataclass
class JointControlResult:
    ok: bool
    reason: str
    current_q_rad: np.ndarray
    target_q_rad: np.ndarray
    delta_q_rad: np.ndarray
    final_q_rad: np.ndarray | None
    max_abs_err_rad: float | None
    collision: bool
    within_safety_limits: bool
    collision_after: bool | None = None
    within_safety_limits_after: bool | None = None
    move_ret: int | None = None
    wait_ret: int | None = None
    raw: dict[str, object] | None = None


def _companion_cpp_path() -> Path:
    return Path(__file__).resolve().with_name(DEFAULT_HELPER_CPP)


def build_joint_helper(
    sdk_root: str = DEFAULT_SDK_ROOT,
    helper_cpp: str | None = None,
    helper_bin: str = DEFAULT_HELPER_BIN,
) -> Path:
    sdk = Path(sdk_root).resolve()
    src = Path(helper_cpp).resolve() if helper_cpp else _companion_cpp_path()
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
    return np.array([float(x.strip()) for x in inner.split(",")], dtype=np.float64)


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


def move_to_joint_positions(
    target_q_rad: np.ndarray,
    *,
    execute: bool = False,
    sdk_root: str = DEFAULT_SDK_ROOT,
    helper_bin: str = DEFAULT_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    speed_deg: float = DEFAULT_SPEED_DEG,
    acc_deg: float = DEFAULT_ACC_DEG,
    speed_fraction: float = DEFAULT_SPEED_FRACTION,
) -> JointControlResult:
    target_q = np.asarray(target_q_rad, dtype=np.float64).reshape(JOINT_DIM)
    helper = build_joint_helper(sdk_root=sdk_root, helper_bin=helper_bin)
    cmd = [
        str(helper),
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
        "--joint-target",
        *[f"{v:.12g}" for v in target_q],
    ]
    if execute:
        cmd.append("--execute")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    raw = _parse_helper_output(completed.stdout)
    raw["_returncode"] = completed.returncode
    raw["_stdout"] = completed.stdout
    raw["_stderr"] = completed.stderr

    current_q = np.asarray(raw.get("current_q_rad", []), dtype=np.float64)
    target_q_out = np.asarray(raw.get("target_q_rad", []), dtype=np.float64)
    delta_q = np.asarray(raw.get("delta_q_rad", []), dtype=np.float64)
    final_q = np.asarray(raw.get("final_q_rad", []), dtype=np.float64)
    if final_q.size == 0:
        final_q = None
    max_abs_err = raw.get("max_abs_err_rad")
    max_abs_err = float(max_abs_err) if max_abs_err is not None else None

    move_ret = raw.get("moveJoint_ret")
    move_ret = int(move_ret) if move_ret is not None else None
    wait_ret = raw.get("wait_arrival_ret")
    wait_ret = int(wait_ret) if wait_ret is not None else None

    # Determine success / reason ----------------------------------------
    #
    # Special case: robot is already at the target position.
    # AUBO SDK moveJoint returns error 13 for near-zero deltas.
    already_there = (
        delta_q.size == JOINT_DIM
        and np.max(np.abs(delta_q)) < ALREADY_AT_TARGET_RAD
    )

    if already_there:
        ok = True
        reason = "already at target"
    elif completed.returncode == 0 and execute:
        ok = True
        reason = "moved"
    elif completed.returncode == 0:
        ok = True
        reason = "dry run only"
    else:
        ok = False
        # Prefer stdout (has structured diagnostics) over stderr (SDK info logs)
        reason = (
            f"helper rc={completed.returncode} moveJoint_ret={move_ret}"
        )

    return JointControlResult(
        ok=ok,
        reason=str(reason).strip(),
        current_q_rad=current_q,
        target_q_rad=target_q_out if target_q_out.size else target_q,
        delta_q_rad=delta_q,
        final_q_rad=final_q if final_q is not None else (current_q if already_there else None),
        max_abs_err_rad=max_abs_err if max_abs_err is not None else (float(np.max(np.abs(delta_q))) if already_there and delta_q.size else None),
        collision=bool(raw.get("collision", False)),
        within_safety_limits=bool(raw.get("within_safety_limits", False)),
        collision_after=bool(raw["collision_after"]) if "collision_after" in raw else None,
        within_safety_limits_after=bool(raw["within_safety_limits_after"]) if "within_safety_limits_after" in raw else None,
        move_ret=move_ret,
        wait_ret=wait_ret,
        raw=raw,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the real robot to a specific joint target.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--joint-target", nargs=6, type=float, help="Target joint angles in radians.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    helper = build_joint_helper()
    if args.compile_only:
        print(f"HELPER_READY={helper}")
        return 0

    if args.joint_target is None:
        raise SystemExit("--joint-target is required unless --compile-only is used")

    result = move_to_joint_positions(np.asarray(args.joint_target, dtype=np.float64), execute=args.execute, helper_bin=str(helper))
    print(result)
    return 0 if result.ok else 1


__all__ = [
    "JointControlResult",
    "build_joint_helper",
    "move_to_joint_positions",
]


if __name__ == "__main__":
    raise SystemExit(main())

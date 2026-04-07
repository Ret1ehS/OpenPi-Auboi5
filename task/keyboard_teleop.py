from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from support.get_obs import STATE_MODE_J6
from support.keyboard_control import (
    KEY_CTRL_C,
    KEY_CTRL_DOWN,
    KEY_CTRL_LEFT,
    KEY_CTRL_RIGHT,
    KEY_CTRL_UP,
    KEY_DOWN,
    KEY_ENTER,
    KEY_LEFT,
    KEY_QUIT,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_UP,
    RawTerminal,
    read_key,
    render_keyboard_ui,
)


DEFAULT_MOVE_STEP_M = 0.005
DEFAULT_ROTATE_STEP_DEG = 5.0


@dataclass(frozen=True)
class KeyboardTeleopConfig:
    prompt: str
    save_fps: int
    state_mode: str
    move_step_m: float = DEFAULT_MOVE_STEP_M
    rotate_step_deg: float = DEFAULT_ROTATE_STEP_DEG


def run_session(
    runtime,
    *,
    config: KeyboardTeleopConfig,
    save_dir: Path,
    save_episode_fn: Callable[..., Path],
) -> int:
    if not runtime.dry_run:
        runtime.return_home("keyboard teleop init home")
        runtime.ensure_gripper_ok(
            runtime.command_gripper_state(1),
            "open gripper before keyboard teleop",
        )

    local_gripper_open = True
    recording = False
    recorded_frames: list[Any] = []
    frame_idx = 0
    saved_episode_count = 0
    move_step_m = float(config.move_step_m)
    rotate_step_deg = float(config.rotate_step_deg)
    status_line = "Ready. Enter starts recording. Idle periods are not recorded."

    def _append_current_snapshot() -> None:
        nonlocal frame_idx
        if not recording:
            return
        recorded_frames.append(
            runtime.capture_manual_snapshot(
                gripper=1.0 if local_gripper_open else 0.0,
                frame_idx=frame_idx,
                semantic_joint6=runtime.local_exec_joint6_rad,
            )
        )
        frame_idx += 1

    def _execute_pose_target(target_pose: np.ndarray) -> None:
        nonlocal frame_idx
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(6).copy()
        if recording:
            seg = runtime.record_pose_move(
                target_pose,
                gripper=1.0 if local_gripper_open else 0.0,
                start_frame_idx=frame_idx,
                semantic_joint6=runtime.local_exec_joint6_rad,
                record=True,
            )
            recorded_frames.extend(seg)
            frame_idx += len(seg)
        else:
            runtime.move_pose(target_pose, "keyboard move")

    def _execute_rotate(delta_deg: float) -> None:
        nonlocal frame_idx
        delta_rad = float(np.deg2rad(delta_deg))
        if config.state_mode == STATE_MODE_J6:
            target_joint6 = runtime.wrap_angle(runtime.local_exec_joint6_rad + delta_rad)
            seg = runtime.record_joint6_rotation(
                target_joint6=target_joint6,
                gripper=1.0 if local_gripper_open else 0.0,
                start_frame_idx=frame_idx,
                record=recording,
            )
            if recording:
                recorded_frames.extend(seg)
                frame_idx += len(seg)
            runtime.local_exec_joint6_rad = target_joint6
            return

        live_pose = runtime.get_live_tcp_pose()
        live_pose[5] = runtime.wrap_angle(float(live_pose[5] + delta_rad))
        _execute_pose_target(live_pose)

    term = RawTerminal.open()
    try:
        while True:
            print(
                render_keyboard_ui(
                    prompt=config.prompt,
                    recording=recording,
                    gripper_open=local_gripper_open,
                    state_mode=config.state_mode,
                    move_step_mm=move_step_m * 1000.0,
                    rotate_step_deg=rotate_step_deg,
                    status_line=status_line,
                ),
                end="",
                flush=True,
            )

            key = read_key(term.fd)
            if key in {KEY_QUIT, KEY_CTRL_C}:
                status_line = "Exiting keyboard teleop."
                break

            if key == KEY_ENTER:
                if not recording:
                    recording = True
                    recorded_frames = []
                    frame_idx = 0
                    _append_current_snapshot()
                    status_line = f"Recording started: {config.prompt}"
                else:
                    recording = False
                    if recorded_frames:
                        save_episode_fn(
                            recorded_frames,
                            save_dir,
                            config.prompt,
                            save_fps=config.save_fps,
                            state_mode=config.state_mode,
                        )
                        saved_episode_count += 1
                        status_line = f"Saved episode {saved_episode_count - 1} for prompt: {config.prompt}"
                    else:
                        status_line = "Recording stopped: no frames captured"
                    recorded_frames = []
                    frame_idx = 0
                continue

            if key == KEY_SPACE:
                target_state = 0 if local_gripper_open else 1
                ok = True if runtime.dry_run else runtime.command_gripper_state(target_state)
                if ok:
                    local_gripper_open = bool(target_state == 1)
                    if recording:
                        _append_current_snapshot()
                    status_line = f"Gripper {'open' if local_gripper_open else 'closed'}"
                else:
                    status_line = "Gripper command failed"
                continue

            move_delta = np.zeros((3,), dtype=np.float64)
            rotate_delta_deg = 0.0
            if key == KEY_UP:
                move_delta[0] = move_step_m
            elif key == KEY_DOWN:
                move_delta[0] = -move_step_m
            elif key == KEY_LEFT:
                move_delta[1] = move_step_m
            elif key == KEY_RIGHT:
                move_delta[1] = -move_step_m
            elif key == KEY_CTRL_UP:
                move_delta[2] = move_step_m
            elif key == KEY_CTRL_DOWN:
                move_delta[2] = -move_step_m
            elif key == KEY_CTRL_LEFT:
                rotate_delta_deg = +rotate_step_deg
            elif key == KEY_CTRL_RIGHT:
                rotate_delta_deg = -rotate_step_deg
            else:
                continue

            try:
                if abs(rotate_delta_deg) > 1e-9:
                    _execute_rotate(rotate_delta_deg)
                    status_line = (
                        f"Rotate {'CCW' if rotate_delta_deg > 0 else 'CW'} "
                        f"{abs(rotate_delta_deg):.1f} deg"
                    )
                else:
                    live_pose = runtime.get_live_tcp_pose()
                    live_pose[:3] = live_pose[:3] + move_delta
                    live_pose[2] = max(float(runtime.min_tcp_z), float(live_pose[2]))
                    _execute_pose_target(live_pose)
                    status_line = (
                        f"Move dx={move_delta[0]*1000.0:+.1f}mm "
                        f"dy={move_delta[1]*1000.0:+.1f}mm "
                        f"dz={move_delta[2]*1000.0:+.1f}mm"
                    )
            except Exception as exc:
                status_line = f"Command failed: {exc}"
    finally:
        term.close()

    return saved_episode_count


__all__ = [
    "DEFAULT_MOVE_STEP_M",
    "DEFAULT_ROTATE_STEP_DEG",
    "KeyboardTeleopConfig",
    "run_session",
]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import time

from support.get_obs import STATE_MODE_J6
from support.keyboard_control import (
    ContinuousKeyState,
    KEY_CTRL_C,
    KEY_ENTER,
    KEY_QUIT,
    KEY_SPACE,
    RawTerminal,
    drain_keys,
    render_keyboard_ui,
)


CONTROL_DT_S = 0.01
RAW_RECORD_DT_S = 1.0 / 30.0
UI_REFRESH_DT_S = 0.05
IDLE_SERVO_STOP_S = 0.20
TARGET_HORIZON_SCALE = 1.0
MAX_LEAD_TICKS = 4.0
DEFAULT_ROTATE_SPEED_DEGPS = 10.0
DEFAULT_MOVE_STEP_M = 0.005
DEFAULT_ROTATE_STEP_DEG = DEFAULT_ROTATE_SPEED_DEGPS * CONTROL_DT_S * TARGET_HORIZON_SCALE


@dataclass(frozen=True)
class KeyboardTeleopConfig:
    prompt: str
    save_fps: int
    state_mode: str
    workspace_x_min: float
    workspace_x_max: float
    workspace_y_min: float
    workspace_y_max: float
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
    move_step_m = float(runtime.linear_speed) * float(CONTROL_DT_S) * float(TARGET_HORIZON_SCALE)
    rotate_step_deg = float(DEFAULT_ROTATE_SPEED_DEGPS) * float(CONTROL_DT_S) * float(TARGET_HORIZON_SCALE)
    max_linear_lead_m = max(move_step_m * float(MAX_LEAD_TICKS), float(runtime.linear_speed) * float(CONTROL_DT_S) * float(MAX_LEAD_TICKS))
    max_yaw_lead_rad = float(np.deg2rad(rotate_step_deg * float(MAX_LEAD_TICKS)))
    status_line = "Ready. Enter starts recording. Idle periods are not recorded."
    next_ui_refresh_ts = 0.0
    next_record_ts = 0.0
    servo_active = False
    command_pose_real = runtime.get_live_tcp_pose()
    last_motion_ts = time.monotonic()
    key_state = ContinuousKeyState()

    def _append_current_snapshot() -> None:
        nonlocal frame_idx, next_record_ts
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
        next_record_ts = time.monotonic() + float(RAW_RECORD_DT_S)

    def _ensure_servo_active() -> None:
        nonlocal servo_active, command_pose_real
        if runtime.dry_run or servo_active:
            return
        command_pose_real = runtime.get_live_tcp_pose()
        resp = runtime.daemon.servo_start(CONTROL_DT_S)
        if int(resp.get("servo_start_ret", -1)) != 0:
            raise RuntimeError(f"servo_start failed: {resp}")
        runtime.daemon.servo_begin_chunk(command_pose_real, force_live_mode=False)
        servo_active = True

    def _stop_servo() -> None:
        nonlocal servo_active, command_pose_real
        if runtime.dry_run:
            servo_active = False
            command_pose_real = runtime.get_live_tcp_pose()
            return
        if servo_active:
            try:
                runtime.daemon.servo_stop()
            except Exception:
                pass
        servo_active = False
        command_pose_real = runtime.get_live_tcp_pose()
        live_joint6 = runtime.get_live_joint6()
        if live_joint6 is not None:
            runtime.local_exec_joint6_rad = float(live_joint6)

    def _maybe_record_active_tick(now_ts: float) -> None:
        if recording and now_ts >= next_record_ts:
            _append_current_snapshot()

    def _clip_command_pose(pose_real: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_real, dtype=np.float64).reshape(6).copy()
        pose[0] = float(np.clip(pose[0], float(config.workspace_x_min), float(config.workspace_x_max)))
        pose[1] = float(np.clip(pose[1], float(config.workspace_y_min), float(config.workspace_y_max)))
        pose[2] = max(float(runtime.min_tcp_z), float(pose[2]))
        pose[5] = runtime.wrap_angle(float(pose[5]))
        return pose

    def _limit_pose_lead(command_pose: np.ndarray, live_pose: np.ndarray) -> np.ndarray:
        limited = np.asarray(command_pose, dtype=np.float64).reshape(6).copy()
        live = np.asarray(live_pose, dtype=np.float64).reshape(6).copy()
        delta_xyz = limited[:3] - live[:3]
        dist = float(np.linalg.norm(delta_xyz))
        if dist > float(max_linear_lead_m) and dist > 1e-9:
            limited[:3] = live[:3] + delta_xyz / dist * float(max_linear_lead_m)
        yaw_err = runtime.wrap_angle(float(limited[5] - live[5]))
        if abs(yaw_err) > float(max_yaw_lead_rad):
            limited[5] = runtime.wrap_angle(float(live[5] + np.sign(yaw_err) * max_yaw_lead_rad))
        return _clip_command_pose(limited)

    def _handle_discrete_key(key: str) -> bool:
        nonlocal recording, recorded_frames, frame_idx, saved_episode_count, status_line, local_gripper_open, next_record_ts
        if key == KEY_ENTER:
            if not recording:
                recording = True
                recorded_frames = []
                frame_idx = 0
                next_record_ts = 0.0
                _append_current_snapshot()
                status_line = f"Recording started: {config.prompt}"
            else:
                recording = False
                if recorded_frames:
                    episode_dir = save_episode_fn(
                        recorded_frames,
                        save_dir,
                        config.prompt,
                        save_fps=config.save_fps,
                        state_mode=config.state_mode,
                    )
                    saved_episode_count += 1
                    status_line = f"Saved {episode_dir.name} for prompt: {config.prompt}"
                else:
                    status_line = "Recording stopped: no frames captured"
                recorded_frames = []
                frame_idx = 0
            return True

        if key == KEY_SPACE:
            _stop_servo()
            target_state = 0 if local_gripper_open else 1
            ok = True if runtime.dry_run else runtime.command_gripper_state(target_state)
            if ok:
                local_gripper_open = bool(target_state == 1)
                if config.state_mode != STATE_MODE_J6:
                    live_joint6 = runtime.get_live_joint6()
                    if live_joint6 is not None:
                        runtime.local_exec_joint6_rad = float(live_joint6)
                if recording:
                    _append_current_snapshot()
                status_line = f"Gripper {'open' if local_gripper_open else 'closed'}"
            else:
                status_line = "Gripper command failed"
            return True

        return False

    term = RawTerminal.open()
    try:
        if not key_state.start(fd=term.fd):
            raise RuntimeError("continuous keyboard teleop backend unavailable")
        while True:
            now_ts = time.monotonic()
            if now_ts >= next_ui_refresh_ts:
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
                next_ui_refresh_ts = now_ts + float(UI_REFRESH_DT_S)

            discrete_keys = key_state.feed_terminal_keys(drain_keys(term.fd), now_ts)
            for key in discrete_keys:
                if key in {KEY_QUIT, KEY_CTRL_C}:
                    status_line = "Exiting keyboard teleop."
                    return saved_episode_count
                if _handle_discrete_key(key):
                    continue
            move_x, move_y, move_z, rotate_axis = key_state.axes(now_ts)
            move_axis = np.array([move_x, move_y, move_z], dtype=np.float64)
            linear_norm = float(np.linalg.norm(move_axis))
            has_linear = linear_norm > 1e-9
            has_rotate = abs(rotate_axis) > 1e-9

            try:
                if has_linear or has_rotate:
                    live_pose = runtime.get_live_tcp_pose()
                    if not runtime.dry_run:
                        _ensure_servo_active()
                    if not servo_active:
                        command_pose_real = live_pose.copy()
                    if has_linear:
                        move_axis = move_axis / linear_norm
                        command_pose_real[:3] = command_pose_real[:3] + move_axis * float(move_step_m)
                    if has_rotate:
                        rotate_delta_rad = float(np.deg2rad(rotate_step_deg * rotate_axis))
                        if config.state_mode == STATE_MODE_J6:
                            runtime.local_exec_joint6_rad = runtime.wrap_angle(runtime.local_exec_joint6_rad + rotate_delta_rad)
                        else:
                            command_pose_real[5] = runtime.wrap_angle(float(command_pose_real[5] + rotate_delta_rad))

                    command_pose_real = _clip_command_pose(command_pose_real)
                    command_pose_real = _limit_pose_lead(command_pose_real, live_pose)

                    if not runtime.dry_run:
                        if config.state_mode == STATE_MODE_J6:
                            resp = runtime.daemon.servo_pose_j6(command_pose_real, runtime.local_exec_joint6_rad)
                        else:
                            resp = runtime.daemon.servo_pose(command_pose_real)
                        pose_ret = int(resp.get("servo_pose_ret", -1))
                        if pose_ret != 0:
                            raise RuntimeError(resp)
                        if config.state_mode != STATE_MODE_J6:
                            live_joint6 = runtime.get_live_joint6()
                            if live_joint6 is not None:
                                runtime.local_exec_joint6_rad = float(live_joint6)

                    last_motion_ts = now_ts
                    _maybe_record_active_tick(now_ts)
                    status_line = (
                        f"Move dx={move_axis[0]*move_step_m*1000.0:+.1f}mm "
                        f"dy={move_axis[1]*move_step_m*1000.0:+.1f}mm "
                        f"dz={move_axis[2]*move_step_m*1000.0:+.1f}mm "
                        f"rot={rotate_axis*rotate_step_deg:+.2f}deg"
                    )
                else:
                    if servo_active and (now_ts - last_motion_ts) >= float(IDLE_SERVO_STOP_S):
                        _stop_servo()
                        status_line = "Idle hold"
                    time.sleep(float(CONTROL_DT_S))
            except Exception as exc:
                status_line = f"Command failed: {exc}"
                _stop_servo()
    finally:
        _stop_servo()
        key_state.stop()
        term.close()

    return saved_episode_count


__all__ = [
    "DEFAULT_MOVE_STEP_M",
    "DEFAULT_ROTATE_STEP_DEG",
    "KeyboardTeleopConfig",
    "run_session",
]

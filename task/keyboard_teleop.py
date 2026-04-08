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
TARGET_HORIZON_SCALE = 1.0
MAX_LEAD_TICKS = 24.0
DEFAULT_ROTATE_SPEED_DEGPS = 30.0
DEFAULT_MOVE_STEP_M = 0.005
DEFAULT_ROTATE_STEP_DEG = DEFAULT_ROTATE_SPEED_DEGPS * CONTROL_DT_S * TARGET_HORIZON_SCALE
LINEAR_FILTER_TAU_S = 0.02
ROTATE_FILTER_TAU_S = 0.02
LEAD_REFERENCE_TAU_S = 0.05
AXIS_EPS = 1e-3
POSE_SEND_DEADBAND_M = 0.0002
POSE_SEND_DEADBAND_RAD = float(np.deg2rad(0.03))


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
    next_control_ts = time.monotonic()
    servo_active = False
    command_pose_real = runtime.get_live_tcp_pose()
    lead_reference_pose_real = command_pose_real.copy()
    last_sent_pose_real: np.ndarray | None = None
    hold_pose_real: np.ndarray | None = None
    hold_joint6_rad: float | None = None
    key_state = ContinuousKeyState()
    linear_axis_cmd = np.zeros(3, dtype=np.float64)
    rotate_axis_cmd = 0.0
    rotate_speed_radps = float(np.deg2rad(DEFAULT_ROTATE_SPEED_DEGPS))

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
        nonlocal servo_active, command_pose_real, lead_reference_pose_real, last_sent_pose_real
        if runtime.dry_run or servo_active:
            return
        command_pose_real = runtime.get_live_tcp_pose()
        lead_reference_pose_real = command_pose_real.copy()
        last_sent_pose_real = None
        runtime.begin_stream_servo(command_pose_real)
        servo_active = True

    def _stop_servo() -> None:
        nonlocal servo_active, command_pose_real, lead_reference_pose_real, last_sent_pose_real, hold_pose_real, hold_joint6_rad
        if runtime.dry_run:
            servo_active = False
            command_pose_real = runtime.get_live_tcp_pose()
            lead_reference_pose_real = command_pose_real.copy()
            last_sent_pose_real = None
            hold_pose_real = None
            hold_joint6_rad = None
            return
        if servo_active:
            try:
                runtime.stop_stream_servo()
            except Exception:
                pass
        servo_active = False
        command_pose_real = runtime.get_live_tcp_pose()
        lead_reference_pose_real = command_pose_real.copy()
        last_sent_pose_real = None
        hold_pose_real = None
        hold_joint6_rad = None
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

    def _apply_pose_deadband(candidate_pose: np.ndarray) -> np.ndarray:
        nonlocal last_sent_pose_real
        pose = np.asarray(candidate_pose, dtype=np.float64).reshape(6).copy()
        if last_sent_pose_real is None:
            last_sent_pose_real = pose.copy()
            return pose
        prev = np.asarray(last_sent_pose_real, dtype=np.float64).reshape(6)
        pos_delta = float(np.linalg.norm(pose[:3] - prev[:3]))
        ang_delta = np.arctan2(np.sin(pose[3:] - prev[3:]), np.cos(pose[3:] - prev[3:]))
        if pos_delta < float(POSE_SEND_DEADBAND_M) and float(np.linalg.norm(ang_delta)) < float(POSE_SEND_DEADBAND_RAD):
            return prev.copy()
        last_sent_pose_real = pose.copy()
        return pose

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
            sleep_s = float(next_control_ts - now_ts)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
                now_ts = time.monotonic()
            else:
                next_control_ts = now_ts
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
            desired_linear_axis = np.array([move_x, move_y, move_z], dtype=np.float64)
            desired_rotate_axis = float(rotate_axis)
            linear_alpha = float(1.0 - np.exp(-CONTROL_DT_S / LINEAR_FILTER_TAU_S))
            rotate_alpha = float(1.0 - np.exp(-CONTROL_DT_S / ROTATE_FILTER_TAU_S))
            linear_axis_cmd = linear_axis_cmd + (desired_linear_axis - linear_axis_cmd) * linear_alpha
            rotate_axis_cmd = float(rotate_axis_cmd + (desired_rotate_axis - rotate_axis_cmd) * rotate_alpha)
            linear_norm = float(np.linalg.norm(linear_axis_cmd))
            has_linear = linear_norm > AXIS_EPS
            has_rotate = abs(rotate_axis_cmd) > AXIS_EPS

            try:
                if has_linear or has_rotate:
                    live_pose = runtime.get_live_tcp_pose()
                    if not servo_active:
                        command_pose_real = live_pose.copy()
                    if not runtime.dry_run:
                        _ensure_servo_active()
                    hold_pose_real = None
                    hold_joint6_rad = None
                    if has_linear:
                        move_dir = linear_axis_cmd / linear_norm
                        command_pose_real[:3] = (
                            command_pose_real[:3]
                            + move_dir * float(runtime.linear_speed) * float(CONTROL_DT_S) * float(linear_norm)
                        )
                        lead_alpha = float(1.0 - np.exp(-CONTROL_DT_S / LEAD_REFERENCE_TAU_S))
                        lead_reference_pose_real[:3] = (
                            lead_reference_pose_real[:3]
                            + (live_pose[:3] - lead_reference_pose_real[:3]) * lead_alpha
                        )
                    else:
                        lead_reference_pose_real[:3] = live_pose[:3]
                    if has_rotate:
                        rotate_delta_rad = float(rotate_speed_radps * rotate_axis_cmd * CONTROL_DT_S)
                        if config.state_mode == STATE_MODE_J6:
                            runtime.local_exec_joint6_rad = runtime.wrap_angle(runtime.local_exec_joint6_rad + rotate_delta_rad)
                        else:
                            command_pose_real[5] = runtime.wrap_angle(float(command_pose_real[5] + rotate_delta_rad))
                    lead_reference_pose_real[3:] = live_pose[3:]

                    command_pose_real = _clip_command_pose(command_pose_real)
                    command_pose_real = _limit_pose_lead(command_pose_real, lead_reference_pose_real)
                    command_pose_real = _apply_pose_deadband(command_pose_real)

                    if not runtime.dry_run:
                        if config.state_mode == STATE_MODE_J6 and has_rotate and not has_linear:
                            resp = runtime.send_stream_joint6(runtime.local_exec_joint6_rad)
                        else:
                            resp = runtime.send_stream_pose(command_pose_real)
                        pose_ret = int(resp.get("servo_pose_ret", -1))
                        if pose_ret != 0:
                            raise RuntimeError(resp)
                        if config.state_mode != STATE_MODE_J6:
                            live_joint6 = runtime.get_live_joint6()
                            if live_joint6 is not None:
                                runtime.local_exec_joint6_rad = float(live_joint6)
                        elif has_linear:
                            live_joint6 = runtime.get_live_joint6()
                            if live_joint6 is not None:
                                runtime.local_exec_joint6_rad = float(live_joint6)

                    _maybe_record_active_tick(now_ts)
                    if has_linear and has_rotate and config.state_mode == STATE_MODE_J6:
                        status_line = (
                            f"Move vx={linear_axis_cmd[0]*float(runtime.linear_speed)*1000.0:+.1f}mm/s "
                            f"vy={linear_axis_cmd[1]*float(runtime.linear_speed)*1000.0:+.1f}mm/s "
                            f"vz={linear_axis_cmd[2]*float(runtime.linear_speed)*1000.0:+.1f}mm/s "
                            f"(j6 rotate deferred)"
                        )
                    else:
                        status_line = (
                            f"Move vx={linear_axis_cmd[0]*float(runtime.linear_speed)*1000.0:+.1f}mm/s "
                            f"vy={linear_axis_cmd[1]*float(runtime.linear_speed)*1000.0:+.1f}mm/s "
                            f"vz={linear_axis_cmd[2]*float(runtime.linear_speed)*1000.0:+.1f}mm/s "
                            f"rot={rotate_axis_cmd*DEFAULT_ROTATE_SPEED_DEGPS:+.2f}deg/s"
                        )
                else:
                    if servo_active:
                        if hold_pose_real is None:
                            hold_pose_real = runtime.get_live_tcp_pose().copy()
                            last_sent_pose_real = hold_pose_real.copy()
                            if config.state_mode == STATE_MODE_J6:
                                live_joint6 = runtime.get_live_joint6()
                                if live_joint6 is not None:
                                    runtime.local_exec_joint6_rad = float(live_joint6)
                                hold_joint6_rad = float(runtime.local_exec_joint6_rad)
                            else:
                                hold_joint6_rad = None
                        if config.state_mode == STATE_MODE_J6:
                            joint6_cmd = runtime.local_exec_joint6_rad if hold_joint6_rad is None else float(hold_joint6_rad)
                            resp = runtime.send_stream_joint6(joint6_cmd, hold_pose_real=hold_pose_real)
                        else:
                            resp = runtime.send_stream_pose(hold_pose_real)
                            live_joint6 = runtime.get_live_joint6()
                            if live_joint6 is not None:
                                runtime.local_exec_joint6_rad = float(live_joint6)
                        pose_ret = int(resp.get("servo_pose_ret", -1))
                        if pose_ret != 0:
                            raise RuntimeError(resp)
                        status_line = "Holding current pose"
                    else:
                        status_line = "Idle. Waiting for input."
            except Exception as exc:
                status_line = f"Command failed: {exc}"
                _stop_servo()
            next_control_ts = float(next_control_ts + float(CONTROL_DT_S))
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

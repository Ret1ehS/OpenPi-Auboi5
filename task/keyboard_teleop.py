from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np
import time

from support.get_obs import STATE_MODE_J6
from support.keyboard_control import (
    ContinuousKeyState,
    KEY_CTRL_C,
    KEY_CTRL_DOWN,
    KEY_CTRL_LEFT,
    KEY_CTRL_RIGHT,
    KEY_CTRL_UP,
    KEY_ENTER,
    KEY_LEFT,
    KEY_QUIT,
    KEY_RIGHT,
    KEY_SHIFT,
    KEY_SPACE,
    KEY_UP,
    KEY_DOWN,
    RawTerminal,
    drain_keys,
    render_keyboard_ui,
)
from support.keyboard_remote import RemoteKeyboardRelay


CONTROL_DT_S = 0.01
RAW_RECORD_DT_S = 1.0 / 30.0
UI_REFRESH_DT_S = 0.05
TARGET_HORIZON_SCALE = 1.0
DEFAULT_ROTATE_SPEED_DEGPS = 31.5
DEFAULT_MOVE_STEP_M = 0.005
DEFAULT_ROTATE_STEP_DEG = DEFAULT_ROTATE_SPEED_DEGPS * CONTROL_DT_S * TARGET_HORIZON_SCALE
AXIS_EPS = 1e-3
INPUT_POLL_DT_S = 0.002
LOOKAHEAD_TIME_S = 0.06
INITIAL_REPEAT_LATCH_S = 0.55
TERMINAL_REPEAT_HOLD_S = 0.06
REMOTE_RELEASE_HOLD_S = 0.03
REMOTE_AXIS_SLEW_PER_TICK = 0.35
ENTER_INPUT_SUPPRESS_S = 0.06
SPACE_INPUT_SUPPRESS_S = 0.15
PROMPT_SWITCH_INPUT_SUPPRESS_S = 0.06
MAX_LINEAR_LEAD_M = 0.010
MAX_YAW_LEAD_RAD = float(np.deg2rad(4.0))
POSE_SEND_DEADBAND_M = 0.00035
POSE_SEND_DEADBAND_RAD = float(np.deg2rad(0.06))


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


class _TerminalInputPump:
    def __init__(self, fd: int, key_state: ContinuousKeyState) -> None:
        self._fd = int(fd)
        self._key_state = key_state
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._discrete_keys: deque[str] = deque()
        self._thread: threading.Thread | None = None
        self._motion_event_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="keyboard-input-pump", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def pop_discrete(self) -> list[str]:
        with self._lock:
            keys = list(self._discrete_keys)
            self._discrete_keys.clear()
            return keys

    def motion_event_count(self) -> int:
        with self._lock:
            return int(self._motion_event_count)

    def clear(self) -> None:
        with self._lock:
            self._discrete_keys.clear()
            self._motion_event_count = 0

    def _run(self) -> None:
        motion_keys = {
            KEY_UP,
            KEY_DOWN,
            KEY_LEFT,
            KEY_RIGHT,
            KEY_CTRL_UP,
            KEY_CTRL_DOWN,
            KEY_CTRL_LEFT,
            KEY_CTRL_RIGHT,
        }
        while not self._stop.is_set():
            now_ts = time.monotonic()
            keys = drain_keys(self._fd)
            discrete = self._key_state.feed_terminal_keys(keys, now_ts)
            motion_count = sum(1 for key in keys if key in motion_keys)
            if discrete or motion_count > 0:
                with self._lock:
                    self._motion_event_count += int(motion_count)
                    self._discrete_keys.extend(discrete)
            time.sleep(float(INPUT_POLL_DT_S))


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
    saving = False
    recorded_frames: list[Any] = []
    frame_idx = 0
    saved_episode_count = 0
    current_prompt_text = str(config.prompt).strip()
    move_step_m = float(runtime.linear_speed) * float(CONTROL_DT_S) * float(TARGET_HORIZON_SCALE)
    rotate_step_deg = float(DEFAULT_ROTATE_SPEED_DEGPS) * float(CONTROL_DT_S) * float(TARGET_HORIZON_SCALE)
    status_line = "Ready. Enter starts recording. Idle periods are not recorded."
    next_ui_refresh_ts = 0.0
    next_record_ts = 0.0
    next_control_ts = time.monotonic()
    servo_active = False
    command_pose_real = runtime.get_live_tcp_pose()
    planned_pose_real = command_pose_real.copy()
    last_sent_pose_real: np.ndarray | None = None
    hold_pose_real: np.ndarray | None = None
    hold_joint6_rad: float | None = None
    key_state = ContinuousKeyState()
    linear_axis_cmd = np.zeros(3, dtype=np.float64)
    rotate_axis_cmd = 0.0
    rotate_speed_radps = float(np.deg2rad(DEFAULT_ROTATE_SPEED_DEGPS))
    raw_motion_active_prev = False
    latched_linear_axis = np.zeros(3, dtype=np.float64)
    latched_rotate_axis = 0.0
    repeat_latch_waiting_repeat = False
    repeat_latch_deadline = 0.0
    repeat_latch_event_count = 0
    remote_relay = RemoteKeyboardRelay.from_ssh_session()
    remote_helper_command = ""
    input_suppress_until_ts = 0.0
    remote_filtered_linear_axis = np.zeros(3, dtype=np.float64)
    remote_filtered_rotate_axis = 0.0
    remote_release_linear_axis = np.zeros(3, dtype=np.float64)
    remote_release_rotate_axis = 0.0
    remote_release_deadline = 0.0
    last_motion_source_remote = False
    input_source = "terminal"
    helper_command = ""

    def _slew_value(current: float, target: float, delta_max: float) -> float:
        delta = float(target) - float(current)
        if delta > float(delta_max):
            return float(current) + float(delta_max)
        if delta < -float(delta_max):
            return float(current) - float(delta_max)
        return float(target)

    def _slew_vector(current: np.ndarray, target: np.ndarray, delta_max: float) -> np.ndarray:
        out = np.asarray(current, dtype=np.float64).reshape(-1).copy()
        tgt = np.asarray(target, dtype=np.float64).reshape(-1)
        for idx in range(out.size):
            out[idx] = _slew_value(out[idx], tgt[idx], delta_max)
        return out

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

    term: RawTerminal | None = None
    input_pump: _TerminalInputPump | None = None

    def _current_prompt() -> str:
        return current_prompt_text

    def _render_ui(force: bool = False) -> None:
        nonlocal next_ui_refresh_ts
        now_ts = time.monotonic()
        if not force and now_ts < next_ui_refresh_ts:
            return
        print(
            render_keyboard_ui(
                prompt=_current_prompt(),
                recording=recording,
                saving=saving,
                gripper_open=local_gripper_open,
                state_mode=config.state_mode,
                move_step_mm=move_step_m * 1000.0,
                rotate_step_deg=rotate_step_deg,
                input_source=input_source,
                helper_command=helper_command,
                status_line=status_line,
            ),
            end="",
            flush=True,
        )
        next_ui_refresh_ts = now_ts + float(UI_REFRESH_DT_S)

    def _clear_pending_inputs() -> None:
        nonlocal raw_motion_active_prev
        nonlocal latched_linear_axis, latched_rotate_axis
        nonlocal repeat_latch_waiting_repeat, repeat_latch_deadline, repeat_latch_event_count
        nonlocal remote_filtered_linear_axis, remote_filtered_rotate_axis
        nonlocal remote_release_linear_axis, remote_release_rotate_axis, remote_release_deadline
        key_state.clear()
        if input_pump is not None:
            input_pump.clear()
        if remote_relay is not None:
            remote_relay.clear()
        try:
            if term is not None:
                drain_keys(term.fd)
        except Exception:
            pass
        raw_motion_active_prev = False
        latched_linear_axis = np.zeros(3, dtype=np.float64)
        latched_rotate_axis = 0.0
        repeat_latch_waiting_repeat = False
        repeat_latch_deadline = 0.0
        repeat_latch_event_count = 0
        remote_filtered_linear_axis = np.zeros(3, dtype=np.float64)
        remote_filtered_rotate_axis = 0.0
        remote_release_linear_axis = np.zeros(3, dtype=np.float64)
        remote_release_rotate_axis = 0.0
        remote_release_deadline = 0.0

    def _edit_prompt() -> bool:
        nonlocal term, input_pump, status_line, input_suppress_until_ts, current_prompt_text
        if recording:
            status_line = "Ignore prompt edit while recording"
            return True
        _clear_pending_inputs()
        new_prompt = ""
        try:
            if input_pump is not None:
                input_pump.stop()
            key_state.stop()
            if term is not None:
                term.close()
            new_prompt = input("\nNew prompt: ").strip()
        finally:
            term = RawTerminal.open()
            input_pump = _TerminalInputPump(term.fd, key_state)
            if not key_state.start(fd=term.fd, repeat_hold_s=TERMINAL_REPEAT_HOLD_S):
                raise RuntimeError("continuous keyboard teleop backend unavailable")
            input_pump.start()
            _clear_pending_inputs()
            input_suppress_until_ts = time.monotonic() + float(PROMPT_SWITCH_INPUT_SUPPRESS_S)
        if new_prompt:
            current_prompt_text = new_prompt
            status_line = f"Prompt updated: {current_prompt_text}"
        else:
            status_line = f"Prompt unchanged: {current_prompt_text}"
        return True

    def _ensure_servo_active() -> None:
        nonlocal servo_active, command_pose_real, planned_pose_real
        nonlocal last_sent_pose_real, hold_pose_real, hold_joint6_rad
        if runtime.dry_run or servo_active:
            return
        command_pose_real = runtime.get_live_tcp_pose()
        planned_pose_real = command_pose_real.copy()
        last_sent_pose_real = None
        hold_pose_real = None
        hold_joint6_rad = None
        runtime.begin_stream_servo(command_pose_real)
        servo_active = True

    def _stop_servo() -> None:
        nonlocal servo_active, command_pose_real, planned_pose_real
        nonlocal last_sent_pose_real, hold_pose_real, hold_joint6_rad
        if runtime.dry_run:
            servo_active = False
            command_pose_real = runtime.get_live_tcp_pose()
            planned_pose_real = command_pose_real.copy()
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
        planned_pose_real = command_pose_real.copy()
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

    def _clamp_command_lead(command_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(command_pose, dtype=np.float64).reshape(6).copy()
        if runtime.dry_run:
            return pose
        live_pose = runtime.get_live_tcp_pose()
        delta_xyz = pose[:3] - live_pose[:3]
        lead_dist = float(np.linalg.norm(delta_xyz))
        if lead_dist > float(MAX_LINEAR_LEAD_M) and lead_dist > 1e-9:
            pose[:3] = live_pose[:3] + delta_xyz / lead_dist * float(MAX_LINEAR_LEAD_M)
        yaw_err = runtime.wrap_angle(float(pose[5] - live_pose[5]))
        if abs(yaw_err) > float(MAX_YAW_LEAD_RAD):
            pose[5] = runtime.wrap_angle(float(live_pose[5] + np.sign(yaw_err) * float(MAX_YAW_LEAD_RAD)))
        return _clip_command_pose(pose)

    def _build_lookahead_pose(
        linear_velocity_xyz: np.ndarray,
        yaw_velocity_radps: float,
    ) -> np.ndarray:
        pose = np.asarray(planned_pose_real, dtype=np.float64).reshape(6).copy()
        pose[:3] = pose[:3] + np.asarray(linear_velocity_xyz, dtype=np.float64).reshape(3) * float(LOOKAHEAD_TIME_S)
        pose[5] = runtime.wrap_angle(float(pose[5] + float(yaw_velocity_radps) * float(LOOKAHEAD_TIME_S)))
        return _clip_command_pose(pose)

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
        nonlocal recording, saving, recorded_frames, frame_idx, saved_episode_count
        nonlocal status_line, local_gripper_open, next_record_ts, input_suppress_until_ts
        if key == KEY_ENTER:
            _clear_pending_inputs()
            input_suppress_until_ts = time.monotonic() + float(ENTER_INPUT_SUPPRESS_S)
            if not recording:
                recording = True
                recorded_frames = []
                frame_idx = 0
                next_record_ts = 0.0
                _append_current_snapshot()
                status_line = f"Recording started: {_current_prompt()}"
            else:
                recording = False
                saving = True
                status_line = f"Saving prompt: {_current_prompt()}"
                _render_ui(force=True)
                if recorded_frames:
                    try:
                        episode_dir = save_episode_fn(
                            recorded_frames,
                            save_dir,
                            _current_prompt(),
                            save_fps=config.save_fps,
                            state_mode=config.state_mode,
                        )
                        saved_episode_count += 1
                        status_line = f"Saved {episode_dir.name} for prompt: {_current_prompt()}"
                    finally:
                        saving = False
                else:
                    saving = False
                    status_line = "Recording stopped: no frames captured"
                recorded_frames = []
                frame_idx = 0
            _clear_pending_inputs()
            input_suppress_until_ts = time.monotonic() + float(ENTER_INPUT_SUPPRESS_S)
            return True

        if key == KEY_SHIFT:
            return _edit_prompt()

        if key == KEY_SPACE:
            _stop_servo()
            _clear_pending_inputs()
            input_suppress_until_ts = time.monotonic() + float(SPACE_INPUT_SUPPRESS_S)
            target_state = 0 if local_gripper_open else 1
            ok = True if runtime.dry_run else runtime.command_gripper_state(target_state)
            _clear_pending_inputs()
            input_suppress_until_ts = time.monotonic() + float(SPACE_INPUT_SUPPRESS_S)
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
    input_pump = _TerminalInputPump(term.fd, key_state)
    try:
        if not key_state.start(fd=term.fd, repeat_hold_s=TERMINAL_REPEAT_HOLD_S):
            raise RuntimeError("continuous keyboard teleop backend unavailable")
        if remote_relay is not None:
            try:
                remote_relay.start()
                remote_helper_command = remote_relay.launcher_command
                if remote_helper_command:
                    status_line = "Run the local helper command to enable raw keyboard streaming."
            except Exception as exc:
                status_line = f"Remote keyboard helper unavailable: {exc}"
                remote_relay = None
        input_pump.start()
        while True:
            now_ts = time.monotonic()
            sleep_s = float(next_control_ts - now_ts)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
                now_ts = time.monotonic()
            else:
                next_control_ts = now_ts
            remote_active = bool(remote_relay is not None and remote_relay.has_active_connection(now_ts))
            if remote_active:
                input_source = "remote helper"
                helper_command = ""
            elif remote_relay is not None:
                input_source = "terminal fallback"
                helper_command = remote_helper_command
            else:
                input_source = key_state.backend or "terminal"
                helper_command = ""
            _render_ui()

            if now_ts < input_suppress_until_ts:
                _clear_pending_inputs()
                next_control_ts = float(next_control_ts + float(CONTROL_DT_S))
                continue

            discrete_keys: list[str] = []
            if remote_relay is not None:
                discrete_keys.extend(remote_relay.pop_discrete())
            discrete_keys.extend(input_pump.pop_discrete())
            for key in discrete_keys:
                if key in {KEY_QUIT, KEY_CTRL_C}:
                    status_line = "Exiting keyboard teleop."
                    return saved_episode_count
                if _handle_discrete_key(key):
                    if time.monotonic() < input_suppress_until_ts:
                        break
                    continue
            if remote_active:
                move_x, move_y, move_z, rotate_axis = remote_relay.axes(now_ts)
                remote_raw_linear_axis = np.array([move_x, move_y, move_z], dtype=np.float64)
                remote_raw_rotate_axis = float(rotate_axis)
                remote_raw_motion_active = (
                    float(np.linalg.norm(remote_raw_linear_axis)) > AXIS_EPS
                    or abs(remote_raw_rotate_axis) > AXIS_EPS
                )
                if remote_raw_motion_active:
                    remote_release_linear_axis = remote_raw_linear_axis.copy()
                    remote_release_rotate_axis = float(remote_raw_rotate_axis)
                    remote_release_deadline = now_ts + float(REMOTE_RELEASE_HOLD_S)
                elif now_ts < remote_release_deadline:
                    remote_raw_linear_axis = remote_release_linear_axis.copy()
                    remote_raw_rotate_axis = float(remote_release_rotate_axis)
                remote_filtered_linear_axis = _slew_vector(
                    remote_filtered_linear_axis,
                    remote_raw_linear_axis,
                    float(REMOTE_AXIS_SLEW_PER_TICK),
                )
                remote_filtered_rotate_axis = _slew_value(
                    remote_filtered_rotate_axis,
                    remote_raw_rotate_axis,
                    float(REMOTE_AXIS_SLEW_PER_TICK),
                )
                raw_linear_axis = remote_filtered_linear_axis.copy()
                raw_rotate_axis = float(remote_filtered_rotate_axis)
            else:
                move_x, move_y, move_z, rotate_axis = key_state.axes(now_ts)
                raw_linear_axis = np.array([move_x, move_y, move_z], dtype=np.float64)
                raw_rotate_axis = float(rotate_axis)
                remote_filtered_linear_axis = np.zeros(3, dtype=np.float64)
                remote_filtered_rotate_axis = 0.0
                remote_release_linear_axis = np.zeros(3, dtype=np.float64)
                remote_release_rotate_axis = 0.0
                remote_release_deadline = 0.0
            raw_motion_active = (
                float(np.linalg.norm(raw_linear_axis)) > AXIS_EPS
                or abs(raw_rotate_axis) > AXIS_EPS
            )
            motion_transition_started = bool(raw_motion_active and not raw_motion_active_prev)
            motion_changed = False
            if (not remote_active) and key_state.backend == "terminal_repeat":
                motion_event_count = input_pump.motion_event_count()
                motion_changed = (
                    not raw_motion_active_prev
                    or float(np.linalg.norm(raw_linear_axis - latched_linear_axis)) > AXIS_EPS
                    or abs(float(raw_rotate_axis) - float(latched_rotate_axis)) > AXIS_EPS
                )
                if raw_motion_active and motion_changed:
                    latched_linear_axis = raw_linear_axis.copy()
                    latched_rotate_axis = float(raw_rotate_axis)
                    repeat_latch_waiting_repeat = True
                    repeat_latch_deadline = now_ts + float(INITIAL_REPEAT_LATCH_S)
                    repeat_latch_event_count = int(motion_event_count)
                elif raw_motion_active:
                    latched_linear_axis = raw_linear_axis.copy()
                    latched_rotate_axis = float(raw_rotate_axis)
                    if int(motion_event_count) > int(repeat_latch_event_count):
                        repeat_latch_waiting_repeat = False
                elif repeat_latch_waiting_repeat and now_ts < repeat_latch_deadline:
                    raw_linear_axis = latched_linear_axis.copy()
                    raw_rotate_axis = float(latched_rotate_axis)
                    raw_motion_active = True
                else:
                    latched_linear_axis = np.zeros(3, dtype=np.float64)
                    latched_rotate_axis = 0.0
                    repeat_latch_waiting_repeat = False
                    repeat_latch_deadline = 0.0
                    repeat_latch_event_count = int(motion_event_count)
            else:
                latched_linear_axis = np.zeros(3, dtype=np.float64)
                latched_rotate_axis = 0.0
                repeat_latch_waiting_repeat = False
                repeat_latch_deadline = 0.0
            raw_motion_active_prev = bool(raw_motion_active)

            linear_axis_cmd = raw_linear_axis
            rotate_axis_cmd = float(raw_rotate_axis)
            linear_norm = float(np.linalg.norm(linear_axis_cmd))
            has_linear = linear_norm > AXIS_EPS
            has_rotate = abs(rotate_axis_cmd) > AXIS_EPS

            try:
                if has_linear or has_rotate:
                    last_motion_source_remote = bool(remote_active)
                    if not servo_active:
                        command_pose_real = runtime.get_live_tcp_pose()
                        planned_pose_real = command_pose_real.copy()
                    if not runtime.dry_run:
                        _ensure_servo_active()
                    hold_pose_real = None
                    hold_joint6_rad = None
                    linear_velocity_xyz = np.zeros(3, dtype=np.float64)
                    yaw_velocity_radps = 0.0
                    if has_linear:
                        move_dir = linear_axis_cmd / linear_norm
                        linear_velocity_xyz = (
                            move_dir * float(runtime.linear_speed) * float(linear_norm)
                        )
                        planned_pose_real[:3] = (
                            planned_pose_real[:3]
                            + linear_velocity_xyz * float(CONTROL_DT_S)
                        )
                    if has_rotate:
                        rotate_delta_rad = float(rotate_speed_radps * rotate_axis_cmd * CONTROL_DT_S)
                        if config.state_mode == STATE_MODE_J6:
                            runtime.local_exec_joint6_rad = runtime.wrap_angle(runtime.local_exec_joint6_rad + rotate_delta_rad)
                        else:
                            yaw_velocity_radps = float(rotate_speed_radps * rotate_axis_cmd)
                            planned_pose_real[5] = runtime.wrap_angle(float(planned_pose_real[5] + rotate_delta_rad))

                    planned_pose_real = _clip_command_pose(planned_pose_real)
                    if remote_active:
                        command_pose_real = planned_pose_real.copy()
                        command_pose_real = _apply_pose_deadband(command_pose_real)
                    else:
                        if motion_transition_started:
                            command_pose_real = planned_pose_real.copy()
                        else:
                            command_pose_real = _build_lookahead_pose(linear_velocity_xyz, yaw_velocity_radps)
                        command_pose_real = _clamp_command_lead(command_pose_real)
                        command_pose_real = _apply_pose_deadband(command_pose_real)

                    if not runtime.dry_run:
                        if config.state_mode == STATE_MODE_J6 and has_rotate and not has_linear:
                            resp = runtime.send_stream_joint6(
                                runtime.local_exec_joint6_rad,
                                hold_pose_real=command_pose_real,
                            )
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
                            if last_motion_source_remote:
                                hold_pose_real = (
                                    last_sent_pose_real.copy()
                                    if last_sent_pose_real is not None
                                    else planned_pose_real.copy()
                                )
                            else:
                                hold_pose_real = (
                                    last_sent_pose_real.copy()
                                    if last_sent_pose_real is not None
                                    else command_pose_real.copy()
                                )
                            planned_pose_real = hold_pose_real.copy()
                            command_pose_real = hold_pose_real.copy()
                            last_sent_pose_real = hold_pose_real.copy()
                            if config.state_mode == STATE_MODE_J6:
                                hold_joint6_rad = float(runtime.local_exec_joint6_rad)
                            else:
                                hold_joint6_rad = None
                            last_motion_source_remote = False
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
        input_pump.stop()
        if remote_relay is not None:
            remote_relay.stop()
        key_state.stop()
        term.close()

    return saved_episode_count


__all__ = [
    "DEFAULT_MOVE_STEP_M",
    "DEFAULT_ROTATE_STEP_DEG",
    "KeyboardTeleopConfig",
    "run_session",
]

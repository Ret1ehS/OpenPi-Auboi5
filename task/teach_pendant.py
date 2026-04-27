from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np


RAW_RECORD_DT_S = 1.0 / 30.0
UI_REFRESH_DT_S = 0.10
STATE_CHANGE_POS_TOL_M = 0.00025
STATE_CHANGE_ANG_TOL_RAD = float(np.deg2rad(0.10))
STATE_CHANGE_GRIP_TOL = 1e-6
GRIPPER_STATUS_POLL_DT_S = 0.10


@dataclass(frozen=True)
class TeachPendantConfig:
    prompt: str
    save_fps: int
    state_mode: str = "yaw"


def _angle_delta_norm(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.arctan2(np.sin(np.asarray(a) - np.asarray(b)), np.cos(np.asarray(a) - np.asarray(b)))
    return float(np.linalg.norm(delta))


def same_recorded_state(
    prev: Any,
    curr: Any,
    *,
    pos_tol_m: float = STATE_CHANGE_POS_TOL_M,
    ang_tol_rad: float = STATE_CHANGE_ANG_TOL_RAD,
    grip_tol: float = STATE_CHANGE_GRIP_TOL,
) -> bool:
    prev_pose = np.asarray(prev.sim_pose6, dtype=np.float64).reshape(6)
    curr_pose = np.asarray(curr.sim_pose6, dtype=np.float64).reshape(6)
    pos_delta = float(np.linalg.norm(curr_pose[:3] - prev_pose[:3]))
    ang_delta = _angle_delta_norm(curr_pose[3:], prev_pose[3:])
    grip_delta = abs(float(curr.gripper) - float(prev.gripper))
    return pos_delta <= float(pos_tol_m) and ang_delta <= float(ang_tol_rad) and grip_delta <= float(grip_tol)


def filter_changed_frames(frames: list[Any]) -> list[Any]:
    """Keep the first frame and later frames whose TCP/gripper state changed."""
    if not frames:
        return []
    kept = [frames[0]]
    for frame in frames[1:]:
        if same_recorded_state(kept[-1], frame):
            continue
        kept.append(frame)
    return [replace(frame, timestamp=float(idx) * float(RAW_RECORD_DT_S)) for idx, frame in enumerate(kept)]


class _GripperStateCache:
    def __init__(self, *, initial: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._value = float(initial)
        self._last_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self.refresh_once()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
        self._thread = threading.Thread(target=self._poll_loop, name="teach-gripper-status", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def value(self) -> float:
        with self._lock:
            return float(self._value)

    def set_local(self, value: float) -> None:
        with self._lock:
            self._value = float(value)
            self._last_error = ""

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def refresh_once(self) -> None:
        from support.gripper_control import get_gripper_status, infer_gripper_observation_state

        status = get_gripper_status()
        observed_state, _contact = infer_gripper_observation_state(status)
        if observed_state is not None:
            with self._lock:
                self._value = float(observed_state)
                self._last_error = ""

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_once()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(float(GRIPPER_STATUS_POLL_DT_S))


def _render_ui(
    *,
    prompt: str,
    recording: bool,
    saving: bool,
    gripper_open: bool,
    raw_frames: int,
    saved_episodes: int,
    status_line: str,
) -> str:
    from support.keyboard_control import BOLD, CLEAR_SCREEN, DIM, FG_CYAN, FG_GREEN, FG_RED, FG_WHITE, FG_YELLOW, RESET

    record_color = FG_YELLOW if saving else (FG_RED if recording else FG_GREEN)
    record_text = "SAVING" if saving else ("ON" if recording else "OFF")
    lines = [
        f"{BOLD}{FG_CYAN}=== OpenPI Teach Pendant Collect ==={RESET}",
        "",
        f"  Prompt:      {BOLD}{FG_WHITE}{prompt}{RESET}",
        f"  Recording:   {BOLD}{record_color}{record_text}{RESET}",
        f"  Gripper:     {BOLD}{FG_WHITE}{'open' if gripper_open else 'closed'}{RESET}",
        f"  Raw Frames:  {BOLD}{FG_WHITE}{raw_frames}{RESET}",
        f"  Episodes:    {BOLD}{FG_WHITE}{saved_episodes}{RESET}",
        "",
        f"  {BOLD}Teach Pendant{RESET}",
        "    Move the robot with the teach pendant while recording.",
        "    The recorder samples at 30Hz and saves only changed TCP/gripper frames.",
        "",
        f"  {BOLD}Session{RESET}",
        "    Enter: start/stop recording and save",
        "    Space: toggle gripper through this program",
        "    q / Ctrl+C: quit",
        "",
    ]
    lines.append(f"  {FG_YELLOW}{status_line}{RESET}" if status_line else f"  {DIM}Idle. Press Enter to record.{RESET}")
    return CLEAR_SCREEN + "\r\n".join(lines)


def run_session(
    runtime,
    *,
    config: TeachPendantConfig,
    save_dir: Path,
    save_episode_fn: Callable[..., Path],
) -> int:
    from support.keyboard_control import HIDE_CURSOR, KEY_CTRL_C, KEY_ENTER, KEY_QUIT, KEY_SPACE, RawTerminal, drain_keys

    prompt = str(config.prompt).strip()
    if not prompt:
        raise ValueError("teach pendant prompt must not be empty")

    recording = False
    saving = False
    recorded_frames: list[Any] = []
    frame_idx = 0
    saved_episode_count = 0
    record_lock = threading.Lock()
    record_thread: threading.Thread | None = None
    record_stop_event: threading.Event | None = None
    status_line = "Ready. Press Enter to start recording."
    next_ui_refresh_ts = 0.0
    gripper_cache = _GripperStateCache(initial=1.0)
    if not runtime.dry_run:
        gripper_cache.start()

    def _append_current_snapshot() -> None:
        nonlocal frame_idx
        live_yaw = runtime.get_live_yaw()
        if live_yaw is not None:
            runtime.local_exec_yaw_rad = float(live_yaw)
        frame = runtime.capture_manual_snapshot(
            gripper=gripper_cache.value(),
            frame_idx=frame_idx,
            semantic_yaw=runtime.local_exec_yaw_rad,
        )
        with record_lock:
            recorded_frames.append(frame)
            frame_idx += 1

    def _start_record_thread() -> None:
        nonlocal record_thread, record_stop_event
        stop_event = threading.Event()
        record_stop_event = stop_event

        def _record_loop() -> None:
            next_ts = time.monotonic()
            while not stop_event.is_set():
                now_ts = time.monotonic()
                if now_ts < next_ts:
                    stop_event.wait(min(float(next_ts - now_ts), 0.002))
                    continue
                _append_current_snapshot()
                next_ts += float(RAW_RECORD_DT_S)
                if (time.monotonic() - next_ts) > float(RAW_RECORD_DT_S):
                    next_ts = time.monotonic()

        record_thread = threading.Thread(target=_record_loop, name="teach-pendant-recorder", daemon=True)
        record_thread.start()

    def _stop_record_thread() -> None:
        nonlocal record_thread, record_stop_event
        stop_event = record_stop_event
        thread = record_thread
        record_stop_event = None
        record_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=1.0)

    def _finalize_frames() -> list[Any]:
        _stop_record_thread()
        if recording:
            _append_current_snapshot()
        with record_lock:
            frames_copy = list(recorded_frames)
        return filter_changed_frames(frames_copy)

    def _render(force: bool = False) -> None:
        nonlocal next_ui_refresh_ts
        now = time.monotonic()
        if not force and now < next_ui_refresh_ts:
            return
        next_ui_refresh_ts = now + float(UI_REFRESH_DT_S)
        with record_lock:
            raw_count = len(recorded_frames)
        err = gripper_cache.last_error()
        suffix = f"  Gripper readback: {err}" if err else ""
        print(
            _render_ui(
                prompt=prompt,
                recording=recording,
                saving=saving,
                gripper_open=gripper_cache.value() >= 0.5,
                raw_frames=raw_count,
                saved_episodes=saved_episode_count,
                status_line=f"{status_line}{suffix}",
            ),
            end="",
            flush=True,
        )

    term: RawTerminal | None = None
    try:
        term = RawTerminal.open()
        print(HIDE_CURSOR, end="", flush=True)
        _render(force=True)
        while True:
            keys = drain_keys(term.fd)
            for key in keys:
                if key in (KEY_QUIT, KEY_CTRL_C):
                    status_line = "Exiting teach pendant collect."
                    _render(force=True)
                    return saved_episode_count
                if key == KEY_ENTER:
                    if not recording:
                        recording = True
                        with record_lock:
                            recorded_frames = []
                            frame_idx = 0
                        _append_current_snapshot()
                        _start_record_thread()
                        status_line = "Recording. Move with teach pendant, Enter to stop/save."
                    else:
                        saving = True
                        status_line = "Saving episode..."
                        _render(force=True)
                        frames_to_save = _finalize_frames()
                        recording = False
                        if frames_to_save:
                            episode_dir = save_episode_fn(
                                frames_to_save,
                                save_dir,
                                prompt,
                                save_fps=config.save_fps,
                            )
                            saved_episode_count += 1
                            status_line = f"Saved {episode_dir.name}: {len(frames_to_save)} changed frames."
                        else:
                            status_line = "Recording stopped: no frames captured."
                        with record_lock:
                            recorded_frames = []
                            frame_idx = 0
                        saving = False
                    _render(force=True)
                elif key == KEY_SPACE:
                    target_state = 0 if gripper_cache.value() >= 0.5 else 1
                    ok = True if runtime.dry_run else runtime.command_gripper_state(target_state)
                    if ok:
                        gripper_cache.set_local(float(target_state))
                        if recording:
                            _append_current_snapshot()
                        status_line = f"Gripper {'open' if target_state == 1 else 'closed'}."
                    else:
                        status_line = "Gripper command failed."
                    _render(force=True)
            _render()
            time.sleep(0.01)
    finally:
        _stop_record_thread()
        gripper_cache.stop()
        if term is not None:
            term.close()


__all__ = [
    "TeachPendantConfig",
    "filter_changed_frames",
    "run_session",
    "same_recorded_state",
]

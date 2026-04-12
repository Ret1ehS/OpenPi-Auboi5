#!/usr/bin/env python3
"""
Main entrypoint for the Jetson-side real-robot OpenPI workflow.

Architecture:
  - Main thread: low-watermark asynchronous chunk inference (get_obs -> infer -> retime -> submit)
  - Executor thread: runs a fixed-rate servo loop against a future action queue keyed by control tick
  - Gripper state change: flush future motion, execute TCP prefix, wait for gripper, re-infer

Inference loop:
  get_obs -> infer -> extract TCP(6D) euler deltas + gripper
    -> take the first action intervals
    -> retime from the observation pose to servo-rate future targets
    -> map each future target to a control tick relative to the observation tick
    -> aggregate only overlapping same-tick targets with buffered future motion
    -> keep observing and inferring while execution continues
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = OPENPI_ROOT / "repo"
REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

GRIPPER_THRESHOLD = 0.6
MAX_EXEC_ACTION_INTERVALS = 20
NATIVE_INTERP_FACTOR = 5  # 20 intervals -> 100 actions in native mode
STATE_MODE_YAW = "yaw"
MAX_CANDIDATES_PER_TICK = 4
EXECUTOR_IDLE_WAIT_S = 0.05
ASYNC_AGGREGATE_OLD_WEIGHT = 0.3
ASYNC_AGGREGATE_NEW_WEIGHT = 0.7
ASYNC_REFILL_THRESHOLD_STEPS = 80


@dataclass
class PoseCandidate:
    pose_sim: np.ndarray
    generation: int
    submit_ts: float


@dataclass
class ChunkSubmission:
    plan: Any
    snapshot: Any
    observation_tick: int
    generation: int
    submit_ts: float


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


class TrajectoryExecutor:
    """Background thread that executes one short servo segment at a time."""

    def __init__(
        self,
        *,
        execute: bool,
        max_speed_mps: float = 0.05,
    ):
        self._execute = execute
        self._max_speed_mps = max_speed_mps
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._sentinel = object()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._expected_pose = None
        self._last_result = None
        self._reset_servo = False
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def expected_pose(self):
        with self._lock:
            if self._expected_pose is not None:
                return self._expected_pose.copy()
            return None

    @property
    def last_result(self):
        with self._lock:
            return self._last_result

    @property
    def is_idle(self) -> bool:
        return self._idle.is_set()

    def submit(self, tcp_deltas, observed_pose) -> None:
        item = (tcp_deltas, observed_pose)
        # Clear idle before enqueue so a caller that immediately waits for idle
        # cannot observe a stale "idle" state from the previous cycle and
        # return before the executor thread has actually started this item.
        self._idle.clear()
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def clear_pending(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout=timeout)

    def reset_state(self) -> None:
        with self._lock:
            self._expected_pose = None
            self._last_result = None
            self._reset_servo = True

    def _loop(self) -> None:
        from support.tcp_control import (
            TrackChunkResult,
            _get_servo_daemon,
            get_robot_snapshot,
            real_pose_to_sim,
            retime_tcp_action_chunk,
            sim_pose_to_real,
        )
        daemon = _get_servo_daemon()
        servo_active = False
        current_plan = None
        current_snapshot = None
        current_index = 0

        def _finish_servo() -> None:
            nonlocal servo_active
            if servo_active:
                try:
                    daemon.servo_stop()
                except Exception:
                    pass
                servo_active = False

        def _drain_latest_pending(*, timeout: float | None = None):
            item = None
            try:
                if timeout is None:
                    item = self._queue.get_nowait()
                else:
                    item = self._queue.get(timeout=timeout)
            except queue.Empty:
                return None

            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
            return item

        def _set_last_result(result) -> None:
            with self._lock:
                self._last_result = result

        def _set_expected_pose(pose) -> None:
            with self._lock:
                self._expected_pose = None if pose is None else pose.copy()

        def _plan_from_item(item):
            nonlocal servo_active
            tcp_deltas, submitted_observed_pose = item
            snapshot = get_robot_snapshot()
            if submitted_observed_pose is not None:
                # Keep chunk planning anchored to the exact pose that produced
                # this observation, so policy input and execution start share
                # the same semantic state.
                start_sim = np.asarray(submitted_observed_pose, dtype=np.float64).reshape(6).copy()
            else:
                start_sim = real_pose_to_sim(snapshot.tcp_pose)

            if snapshot.collision or not snapshot.within_safety_limits:
                result = TrackChunkResult(
                    ok=False,
                    reason="robot is already in collision or outside safety limits",
                    snapshot=snapshot,
                    start_pose=start_sim,
                    final_pose=start_sim,
                    sample_count=0,
                    control_dt_s=0.0,
                    tracking_err=0.0,
                    exec_mode=None,
                    raw=None,
                    start_pose_real=sim_pose_to_real(start_sim),
                    final_pose_real=sim_pose_to_real(start_sim),
                )
                _set_last_result(result)
                _set_expected_pose(None)
                _finish_servo()
                return None

            plan = retime_tcp_action_chunk(
                tcp_deltas,
                start_pose_sim=start_sim,
                max_linear_speed_mps=self._max_speed_mps,
            )

            if not plan.steps:
                result = TrackChunkResult(
                    ok=True,
                    reason="empty chunk",
                    snapshot=snapshot,
                    start_pose=plan.start_pose,
                    final_pose=plan.final_pose,
                    sample_count=0,
                    control_dt_s=plan.control_dt_s,
                    tracking_err=0.0,
                    exec_mode="streaming",
                    raw=None,
                    start_pose_real=plan.start_pose_real,
                    final_pose_real=plan.final_pose_real,
                )
                _set_last_result(result)
                _set_expected_pose(plan.final_pose)
                return None

            if not self._execute:
                result = TrackChunkResult(
                    ok=True,
                    reason="plan only",
                    snapshot=snapshot,
                    start_pose=plan.start_pose,
                    final_pose=plan.final_pose,
                    sample_count=len(plan.steps),
                    control_dt_s=plan.control_dt_s,
                    tracking_err=0.0,
                    exec_mode="streaming",
                    raw=None,
                    start_pose_real=plan.start_pose_real,
                    final_pose_real=plan.final_pose_real,
                )
                _set_last_result(result)
                _set_expected_pose(plan.final_pose)
                return None

            if not servo_active:
                resp = daemon.servo_start(plan.control_dt_s)
                if int(resp.get("servo_start_ret", -1)) != 0:
                    result = TrackChunkResult(
                        ok=False,
                        reason=f"servo_start failed: {resp.get('error', 'unknown')}",
                        snapshot=snapshot,
                        start_pose=start_sim,
                        final_pose=start_sim,
                        sample_count=0,
                        control_dt_s=plan.control_dt_s,
                        tracking_err=0.0,
                        exec_mode="streaming",
                        raw=resp,
                        start_pose_real=sim_pose_to_real(start_sim),
                        final_pose_real=sim_pose_to_real(start_sim),
                    )
                    _set_last_result(result)
                    _set_expected_pose(None)
                    return None
                servo_active = True

            daemon.servo_begin_chunk(plan.start_pose_real)
            return snapshot, plan

        while not self._stop.is_set():
            try:
                # Check if reset was requested (e.g. after Ctrl+C)
                with self._lock:
                    need_reset = self._reset_servo
                    if need_reset:
                        self._reset_servo = False
                if need_reset:
                    _finish_servo()
                    current_plan = None
                    current_snapshot = None
                    current_index = 0

                if current_plan is None:
                    item = _drain_latest_pending(timeout=0.05)
                    if item is None:
                        # No new chunk: helper-side keepalive holds last target.
                        self._idle.set()
                        continue
                    if item is self._sentinel:
                        _finish_servo()
                        break
                    self._idle.clear()
                    planned = _plan_from_item(item)
                    if planned is None:
                        if self._queue.empty():
                            self._idle.set()
                        continue
                    current_snapshot, current_plan = planned
                    current_index = 0

                if current_plan is None or current_index >= len(current_plan.steps):
                    current_snapshot = None
                    current_plan = None
                    current_index = 0
                    if self._queue.empty():
                        # Don't stop servo – just hold position
                        self._idle.set()
                    continue

                step = current_plan.steps[current_index]
                resp = daemon.servo_pose(step.pose_real)
                pose_ret = int(resp.get("servo_pose_ret", -1))
                error = str(resp.get("error", ""))

                if pose_ret == 0:
                    current_index += 1
                    _set_last_result(
                        TrackChunkResult(
                            ok=True,
                            reason=f"streaming executed ({current_index}/{len(current_plan.steps)} steps)",
                            snapshot=current_snapshot,
                            start_pose=current_plan.start_pose,
                            final_pose=step.pose_sim,
                            sample_count=current_index,
                            control_dt_s=current_plan.control_dt_s,
                            tracking_err=0.0,
                            exec_mode="streaming",
                            raw=resp,
                            start_pose_real=current_plan.start_pose_real,
                            final_pose_real=step.pose_real,
                        )
                    )
                    _set_expected_pose(step.pose_sim)
                    if current_index >= len(current_plan.steps):
                        current_snapshot = None
                        current_plan = None
                        current_index = 0
                        if self._queue.empty():
                            self._idle.set()
                else:
                    final_pose = step.pose_sim if current_index > 0 else current_plan.start_pose
                    final_pose_real = step.pose_real if current_index > 0 else current_plan.start_pose_real
                    _set_last_result(
                        TrackChunkResult(
                            ok=False,
                            reason=f"servo_pose failed: {error}",
                            snapshot=current_snapshot,
                            start_pose=current_plan.start_pose,
                            final_pose=final_pose,
                            sample_count=current_index,
                            control_dt_s=current_plan.control_dt_s,
                            tracking_err=0.0,
                            exec_mode="streaming",
                            raw=resp,
                            start_pose_real=current_plan.start_pose_real,
                            final_pose_real=final_pose_real,
                        )
                    )
                    _set_expected_pose(final_pose if current_index > 0 else None)
                    current_snapshot = None
                    current_plan = None
                    current_index = 0
                    if error == "safety":
                        _finish_servo()

            except Exception as exc:
                _set_last_result(None)
                _set_expected_pose(None)
                current_snapshot = None
                current_plan = None
                current_index = 0
                print(f"  [executor] Error: {exc}")
                _finish_servo()

            if current_plan is None and self._queue.empty():
                self._idle.set()


class ParallelTrajectoryExecutor:
    """Background thread that executes a future TCP pose queue at a fixed control rate."""

    def __init__(
        self,
        *,
        execute: bool,
        max_speed_mps: float = 0.05,
    ):
        self._execute = execute
        self._max_speed_mps = max_speed_mps
        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._sentinel = object()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._expected_pose = None
        self._last_result = None
        self._reset_servo = False
        self._current_pose_sim: np.ndarray | None = None
        self._current_tick = 0
        self._control_dt_s = 0.01
        self._latest_snapshot = None
        self._future_candidates: dict[int, list[PoseCandidate]] = {}
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def expected_pose(self):
        with self._lock:
            if self._expected_pose is not None:
                return self._expected_pose.copy()
            return None

    @property
    def last_result(self):
        with self._lock:
            return self._last_result

    @property
    def is_idle(self) -> bool:
        return self._idle.is_set()

    def get_progress(self) -> dict[str, int | float | bool]:
        with self._lock:
            current_tick = int(self._current_tick)
            max_future_tick = max(self._future_candidates.keys(), default=current_tick)
            buffered_steps = max(0, max_future_tick - current_tick)
            control_dt_s = float(self._control_dt_s)
        pending_submissions = max(0, self._queue.qsize())
        return {
            "current_tick": current_tick,
            "max_future_tick": int(max_future_tick),
            "buffered_steps": int(buffered_steps),
            "buffered_time_s": float(buffered_steps * control_dt_s),
            "control_dt_s": control_dt_s,
            "pending_submissions": int(pending_submissions),
            "idle": bool(self._idle.is_set()),
        }

    def submit(self, plan, snapshot, *, observation_tick: int, generation: int) -> None:
        item = ChunkSubmission(
            plan=plan,
            snapshot=snapshot,
            observation_tick=max(0, int(observation_tick)),
            generation=int(generation),
            submit_ts=time.monotonic(),
        )
        self._idle.clear()
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def clear_pending(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._future_candidates.clear()
        self._idle.set()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout=timeout)

    def reset_state(self) -> None:
        with self._lock:
            self._expected_pose = None
            self._last_result = None
            self._reset_servo = True
            self._current_pose_sim = None
            self._current_tick = 0
            self._control_dt_s = 0.01
            self._latest_snapshot = None
            self._future_candidates.clear()

    def _update_idle_flag(self) -> None:
        with self._lock:
            has_future = bool(self._future_candidates)
        if has_future or not self._queue.empty():
            self._idle.clear()
        else:
            self._idle.set()

    @staticmethod
    def _aggregate_pose_candidates(candidates: list[PoseCandidate], fallback_pose: np.ndarray) -> np.ndarray:
        if not candidates:
            return np.asarray(fallback_pose, dtype=np.float64).reshape(6).copy()

        ordered = candidates[-MAX_CANDIDATES_PER_TICK:]
        fused = np.asarray(ordered[0].pose_sim, dtype=np.float64).reshape(6).copy()
        for item in ordered[1:]:
            pose = np.asarray(item.pose_sim, dtype=np.float64).reshape(6)
            fused[:3] = (
                ASYNC_AGGREGATE_OLD_WEIGHT * fused[:3]
                + ASYNC_AGGREGATE_NEW_WEIGHT * pose[:3]
            )
            for axis in range(3, 6):
                fused[axis] = float(
                    np.arctan2(
                        ASYNC_AGGREGATE_OLD_WEIGHT * np.sin(fused[axis])
                        + ASYNC_AGGREGATE_NEW_WEIGHT * np.sin(pose[axis]),
                        ASYNC_AGGREGATE_OLD_WEIGHT * np.cos(fused[axis])
                        + ASYNC_AGGREGATE_NEW_WEIGHT * np.cos(pose[axis]),
                    )
                )
        return fused

    def _ingest_submission(self, submission: ChunkSubmission) -> tuple[int, int]:
        plan = submission.plan
        accepted = 0
        stale = 0
        with self._lock:
            self._latest_snapshot = submission.snapshot
            self._control_dt_s = float(plan.control_dt_s)
            if self._current_pose_sim is None:
                self._current_pose_sim = np.asarray(plan.start_pose, dtype=np.float64).reshape(6).copy()
                self._expected_pose = self._current_pose_sim.copy()

            base_tick = int(submission.observation_tick)
            for offset, step in enumerate(plan.steps, start=1):
                tick = base_tick + offset
                if tick <= self._current_tick:
                    stale += 1
                    continue
                bucket = self._future_candidates.setdefault(tick, [])
                bucket.append(
                    PoseCandidate(
                        pose_sim=np.asarray(step.pose_sim, dtype=np.float64).reshape(6).copy(),
                        generation=int(submission.generation),
                        submit_ts=float(submission.submit_ts),
                    )
                )
                if len(bucket) > MAX_CANDIDATES_PER_TICK:
                    del bucket[:-MAX_CANDIDATES_PER_TICK]
                accepted += 1

            for stale_tick in [tick for tick in self._future_candidates.keys() if tick <= self._current_tick]:
                self._future_candidates.pop(stale_tick, None)

        return accepted, stale

    def _loop(self) -> None:
        from support.tcp_control import (
            TrackChunkResult,
            _get_servo_daemon,
            sim_pose_to_real,
        )

        daemon = _get_servo_daemon()
        servo_active = False
        next_send_ts = time.monotonic()

        def _finish_servo() -> None:
            nonlocal servo_active
            if servo_active:
                try:
                    daemon.servo_stop()
                except Exception:
                    pass
                servo_active = False

        def _drain_pending(*, timeout: float | None = None) -> list[object]:
            items: list[object] = []
            try:
                if timeout is None:
                    items.append(self._queue.get_nowait())
                else:
                    items.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                return []

            while True:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            return items

        def _set_last_result(result) -> None:
            with self._lock:
                self._last_result = result

        def _set_expected_pose(pose) -> None:
            with self._lock:
                self._expected_pose = None if pose is None else pose.copy()

        while not self._stop.is_set():
            try:
                with self._lock:
                    need_reset = self._reset_servo
                    if need_reset:
                        self._reset_servo = False
                if need_reset:
                    _finish_servo()
                    next_send_ts = time.monotonic()

                timeout = 0.0 if self._current_pose_sim is not None else EXECUTOR_IDLE_WAIT_S
                items = _drain_pending(timeout=timeout)
                if items:
                    for item in items:
                        if item is self._sentinel:
                            _finish_servo()
                            return
                        assert isinstance(item, ChunkSubmission)
                        accepted, _stale = self._ingest_submission(item)
                        if accepted > 0:
                            self._idle.clear()
                    if self._current_pose_sim is not None and time.monotonic() > next_send_ts + max(self._control_dt_s, 1e-6):
                        next_send_ts = time.monotonic()

                with self._lock:
                    current_pose_sim = None if self._current_pose_sim is None else self._current_pose_sim.copy()
                    next_tick = int(self._current_tick) + 1
                    candidates = list(self._future_candidates.pop(next_tick, []))
                    snapshot = self._latest_snapshot
                    control_dt_s = float(self._control_dt_s)

                if current_pose_sim is None:
                    self._idle.set()
                    continue

                now = time.monotonic()
                if next_send_ts > now:
                    time.sleep(min(next_send_ts - now, 0.002))
                    self._update_idle_flag()
                    continue

                target_pose_sim = self._aggregate_pose_candidates(candidates, current_pose_sim)
                target_pose_real = sim_pose_to_real(target_pose_sim)

                if self._execute:
                    if not servo_active:
                        resp = daemon.servo_start(control_dt_s)
                        if int(resp.get("servo_start_ret", -1)) != 0:
                            _set_last_result(
                                TrackChunkResult(
                                    ok=False,
                                    reason=f"servo_start failed: {resp.get('error', 'unknown')}",
                                    snapshot=snapshot,
                                    start_pose=current_pose_sim,
                                    final_pose=current_pose_sim,
                                    sample_count=int(next_tick) - 1,
                                    control_dt_s=control_dt_s,
                                    tracking_err=0.0,
                                    exec_mode="streaming",
                                    raw=resp,
                                    start_pose_real=sim_pose_to_real(current_pose_sim),
                                    final_pose_real=sim_pose_to_real(current_pose_sim),
                                )
                            )
                            _set_expected_pose(None)
                            self._idle.set()
                            time.sleep(control_dt_s)
                            continue
                        daemon.servo_begin_chunk(sim_pose_to_real(current_pose_sim))
                        servo_active = True

                    resp = daemon.servo_pose(target_pose_real)
                    pose_ret = int(resp.get("servo_pose_ret", -1))
                    error = str(resp.get("error", ""))
                else:
                    pose_ret = 0
                    error = ""
                    resp = {"servo_pose_ret": 0, "simulated": True}

                if pose_ret == 0:
                    with self._lock:
                        self._current_tick = next_tick
                        self._current_pose_sim = target_pose_sim.copy()
                    _set_expected_pose(target_pose_sim)
                    _set_last_result(
                        TrackChunkResult(
                            ok=True,
                            reason="streaming queued step" if candidates else "holding current pose",
                            snapshot=snapshot,
                            start_pose=current_pose_sim,
                            final_pose=target_pose_sim,
                            sample_count=int(next_tick),
                            control_dt_s=control_dt_s,
                            tracking_err=0.0,
                            exec_mode="streaming",
                            raw=resp,
                            start_pose_real=sim_pose_to_real(current_pose_sim),
                            final_pose_real=target_pose_real,
                        )
                    )
                else:
                    _set_last_result(
                        TrackChunkResult(
                            ok=False,
                            reason=f"servo_pose failed: {error}",
                            snapshot=snapshot,
                            start_pose=current_pose_sim,
                            final_pose=current_pose_sim,
                            sample_count=int(next_tick) - 1,
                            control_dt_s=control_dt_s,
                            tracking_err=0.0,
                            exec_mode="streaming",
                            raw=resp,
                            start_pose_real=sim_pose_to_real(current_pose_sim),
                            final_pose_real=sim_pose_to_real(current_pose_sim),
                        )
                    )
                    _set_expected_pose(None)
                    with self._lock:
                        self._future_candidates.clear()
                    if error == "safety":
                        _finish_servo()

            except Exception as exc:
                _set_last_result(None)
                _set_expected_pose(None)
                with self._lock:
                    self._future_candidates.clear()
                print(f"  [executor] Error: {exc}")
                _finish_servo()

            next_send_ts += max(self._control_dt_s, 1e-6)
            if next_send_ts < time.monotonic() - max(self._control_dt_s, 1e-6):
                next_send_ts = time.monotonic()
            self._update_idle_flag()


def _calibrate_alignment(prefix: str = "", *, pose_frame: str = "sim") -> None:
    import numpy as np

    from support.pose_align import set_runtime_alignment
    from support.tcp_control import get_robot_snapshot

    snap = get_robot_snapshot()
    ctx = set_runtime_alignment(snap.tcp_pose, frame_mode=pose_frame)
    print(f"{prefix}Pose frame calibrated:")
    print(f"{prefix}  frame_mode      = {ctx.frame_mode}")
    print(f"{prefix}  real_init_tcp   = {np.round(ctx.real_init_pose6, 5).tolist()}")
    print(f"{prefix}  policy_init_tcp = {np.round(ctx.sim_init_pose6, 5).tolist()}")


def main() -> int:
    _maybe_reexec_into_repo_venv()

    import numpy as np

    from support.get_obs import RealRobotOpenPIObservationBuilder, STATE_MODE_YAW
    from support.gripper_control import (
        command_gripper_state,
        get_gripper_status,
        gripper_status_to_openpi_state,
    )
    from support.joint_control import build_joint_helper, move_to_joint_positions
    from support.load_policy import PolicyLoadSpec, load_policy
    from support.pose_align import (
        REAL_INIT_QPOS_RAD,
        clear_runtime_alignment,
        get_alignment_mode,
        is_alignment_ready,
        set_alignment_mode,
    )
    from support.tcp_control import build_helper as build_tcp_helper, get_robot_snapshot, retime_tcp_action_chunk
    from support.tui_config import TUIConfig, run_tui_config

    initial_qpos_rad = REAL_INIT_QPOS_RAD.copy()
    initial_qpos_deg = np.degrees(initial_qpos_rad)

    print("Compiling C++ helpers...")
    build_joint_helper()
    build_tcp_helper()

    # --- TUI action callbacks (run before policy is loaded) ---
    def _tui_action(key: str, current_cfg: TUIConfig | None = None) -> str:
        cfg_now = current_cfg or TUIConfig(
            policy_location="remote",
            pose_frame="real",
            obs_state_mode=STATE_MODE_YAW,
            dry_run=False,
            exec_speed_mps=0.05,
        )
        set_alignment_mode(cfg_now.pose_frame)
        execute_now = not cfg_now.dry_run
        if key == "align":
            if not execute_now:
                print("Align skipped (OPENPI_DEBUG_DRY_RUN=1).")
                return "Align: skipped (debug dry-run)"
            print("Moving to initial joint positions...")
            result = move_to_joint_positions(initial_qpos_rad, execute=True)
            print(f"  Target (deg): {[round(v, 3) for v in initial_qpos_deg.tolist()]}")
            print(f"  Result: ok={result.ok}, reason={result.reason}")
            if result.ok:
                _calibrate_alignment(prefix="  ", pose_frame=cfg_now.pose_frame)
            return f"Align: {'ok' if result.ok else result.reason}"
        elif key == "grip_open":
            if not execute_now:
                print("Open gripper skipped (OPENPI_DEBUG_DRY_RUN=1).")
                return "Open gripper: skipped (debug dry-run)"
            print("Opening gripper...")
            ok = command_gripper_state(1, timeout_s=10.0)
            print(f"  Gripper opened: {ok}")
            return f"Open gripper: {'ok' if ok else 'FAIL'}"
        elif key == "grip_close":
            if not execute_now:
                print("Close gripper skipped (OPENPI_DEBUG_DRY_RUN=1).")
                return "Close gripper: skipped (debug dry-run)"
            print("Closing gripper...")
            ok = command_gripper_state(0, timeout_s=10.0)
            print(f"  Gripper closed: {ok}")
            return f"Close gripper: {'ok' if ok else 'FAIL'}"
        elif key == "status":
            try:
                snap = get_robot_snapshot()
                print(f"  Pose frame: {get_alignment_mode()}")
                print(f"  Execute: {execute_now}")
                print(f"  Speed Mode: {cfg_now.speed_mode}  Exec Speed: {cfg_now.exec_speed_mps} m/s")
                print(f"  TCP pose: {np.round(snap.tcp_pose, 5).tolist()}")
                print(f"  Joint q (deg): {np.round(np.degrees(snap.joint_q), 2).tolist()}")
                print(f"  Collision: {snap.collision}")
                print(f"  Safety limits OK: {snap.within_safety_limits}")
                return "Status: ok"
            except Exception as exc:
                print(f"  Error: {exc}")
                return f"Status: error ({exc})"
        return f"Unknown action: {key}"

    # --- Interactive TUI configuration ---
    cfg = run_tui_config(action_callback=_tui_action)
    if cfg.quit:
        print("Exiting.")
        return 0

    # Apply configuration
    set_alignment_mode(cfg.pose_frame)
    execute = not cfg.dry_run

    print(f"\nConfiguration:")
    print(f"  Policy:     {cfg.policy_location}")
    print(f"  Frame:      {cfg.pose_frame}")
    print(f"  State Mode: {cfg.obs_state_mode}")
    print(f"  Speed Mode: {cfg.speed_mode}")
    if cfg.speed_mode == "limited":
        print(f"  Exec Speed: {cfg.exec_speed_mps} m/s")
    else:
        print(f"  Exec Speed: native (no retime)")
    print(f"  Record:     {cfg.record}")
    if cfg.dry_run:
        print(f"  Debug Dry Run: {cfg.dry_run}")

    # --- Load policy ---
    t0 = time.monotonic()
    policy_spec = PolicyLoadSpec(
        remote=(cfg.policy_location == "remote"),
    )
    if policy_spec.remote:
        print("\nConnecting to remote inference server...")
    else:
        from support.load_policy import DEFAULT_CHECKPOINT_DIR
        print(f"\nCHECKPOINT_DIR={DEFAULT_CHECKPOINT_DIR}")
        print("Loading policy from checkpoint...")
    policy = load_policy(policy_spec)
    policy_load_s = time.monotonic() - t0
    print(f"POLICY_READY={type(policy).__name__}  load_time={policy_load_s:.3f}s")
    print(f"POSE_FRAME={get_alignment_mode()}")

    obs_state_mode = STATE_MODE_YAW
    if str(cfg.obs_state_mode).strip().lower() != STATE_MODE_YAW:
        print(f"State Mode '{cfg.obs_state_mode}' overridden to '{STATE_MODE_YAW}' on yaw-refactor branch.")
    print(f"OBS_STATE_MODE={obs_state_mode}")
    obs_builder = RealRobotOpenPIObservationBuilder(state_mode=obs_state_mode)
    cameras_ready = False

    def ensure_cameras_ready() -> None:
        nonlocal cameras_ready
        if cameras_ready:
            return
        print("Initializing cameras...")
        obs_builder.start()
        cameras_ready = True
        print("Cameras ready.")

    effective_speed = float('inf') if cfg.speed_mode == "native" else cfg.exec_speed_mps
    is_native_speed = cfg.speed_mode == "native"
    executor = ParallelTrajectoryExecutor(
        execute=execute,
        max_speed_mps=effective_speed,
    )
    executor.start()
    infer_log_dir = SCRIPT_DIR / "log"
    infer_log_dir.mkdir(parents=True, exist_ok=True)
    infer_log_path = infer_log_dir / "main_inference.log"

    def _append_infer_log(event: dict[str, object]) -> None:
        payload = {
            "ts_ms": int(time.time() * 1000),
            **event,
        }
        with open(infer_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # --- Prompt input loop ---
    print("\n=== Ready. Enter a prompt to start inference. ===")
    print("  Type a prompt and press Enter to begin.")
    print("  Commands: align, grip, close, status, reset, quit")
    print("  During inference: Ctrl+C to stop.\n")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            continue

        cmd_lower = cmd.lower()

        if cmd_lower in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if cmd_lower == "status":
            try:
                snap = get_robot_snapshot()
                print(f"  Policy: {type(policy).__name__}  load_time={policy_load_s:.3f}s")
                print(f"  Pose frame: {get_alignment_mode()}")
                print(f"  Execute: {execute}")
                print(f"  TCP pose: {np.round(snap.tcp_pose, 5).tolist()}")
                print(f"  Joint q (deg): {np.round(np.degrees(snap.joint_q), 2).tolist()}")
                print(f"  Collision: {snap.collision}")
                print(f"  Safety limits OK: {snap.within_safety_limits}")
                print(f"  Executor idle: {executor.is_idle}")
            except Exception as exc:
                print(f"  Error: {exc}")
            continue

        if cmd_lower == "align":
            executor.clear_pending()
            executor.wait_until_idle()
            executor.reset_state()
            print("Moving to initial joint positions...")
            result = move_to_joint_positions(initial_qpos_rad, execute=execute)
            executor.reset_state()
            print(f"  Result: ok={result.ok}, reason={result.reason}")
            if execute:
                if result.ok:
                    _calibrate_alignment(prefix="  ", pose_frame=cfg.pose_frame)
                else:
                    clear_runtime_alignment()
                    print("  Pose alignment cleared because joint alignment failed.")
            continue

        if cmd_lower == "grip":
            if execute:
                print("Opening gripper...")
                ok = command_gripper_state(1, timeout_s=10.0)
                print(f"  Gripper opened: {ok}")
            else:
                print("  Skipped (dry-run)")
            continue

        if cmd_lower == "close":
            if execute:
                print("Closing gripper...")
                ok = command_gripper_state(0, timeout_s=10.0)
                print(f"  Gripper closed: {ok}")
            else:
                print("  Skipped (dry-run)")
            continue

        if cmd_lower == "reset":
            executor.clear_pending()
            executor.wait_until_idle()
            executor.reset_state()
            obs_builder.reset_pose_filter()
            policy.reset()
            print("Policy state reset. Executor state cleared.")
            continue

        # --- Start inference loop ---
        prompt = cmd
        print(f"Starting inference loop with prompt: \"{prompt}\"")
        print(f"  Execute={execute}  PoseFrame={get_alignment_mode()}")
        print("  Press Ctrl+C to stop.\n")
        _append_infer_log(
            {
                "event": "session_start",
                "prompt": prompt,
                "execute": bool(execute),
                "pose_frame": str(get_alignment_mode()),
                "speed_mode": str(cfg.speed_mode),
                "exec_speed_mps": None if cfg.speed_mode == "native" else float(cfg.exec_speed_mps),
            }
        )

        policy.reset()
        executor.reset_state()

        if execute:
            print("Auto-init: aligning joints...")
            align_result = move_to_joint_positions(initial_qpos_rad, execute=True)
            print(f"  Auto-init align: ok={align_result.ok}, reason={align_result.reason}")
            if not align_result.ok:
                clear_runtime_alignment()
                print("  Auto-init aborted: pose alignment cleared because joint alignment failed.")
                continue
            print("Auto-init: opening gripper...")
            command_gripper_state(1, timeout_s=10.0)
            _calibrate_alignment(pose_frame=cfg.pose_frame)

        if not execute and not is_alignment_ready():
            _calibrate_alignment(pose_frame=cfg.pose_frame)

        ensure_cameras_ready()
        obs_builder.reset_pose_filter()
        obs_builder.set_gripper_open_scalar(1.0)
        step = 0
        last_gripper_state: int | None = None

        # --- Optional video recording (background thread) ---
        _rec_stop = threading.Event()
        _rec_thread = None
        video_path = None
        if cfg.record:
            import cv2
            captures_dir = Path("/home/orin/openpi/captures")
            captures_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            video_path = captures_dir / f"inference_{timestamp_str}.mp4"

            def _record_loop():
                writer = None
                while not _rec_stop.is_set():
                    try:
                        main_bgr, wrist_bgr = obs_builder.grab_bgr_pair()
                        if main_bgr is None or wrist_bgr is None:
                            time.sleep(0.05)
                            continue
                        h, w = main_bgr.shape[:2]
                        wh, ww = wrist_bgr.shape[:2]
                        if wh != h:
                            scale = h / wh
                            wrist_bgr = cv2.resize(wrist_bgr, (int(ww * scale), h))
                        concat = np.concatenate([main_bgr, wrist_bgr], axis=1)
                        if writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(str(video_path), fourcc, 15.0,
                                                     (concat.shape[1], concat.shape[0]))
                            print(f"  Recording video to {video_path}")
                        writer.write(concat)
                    except Exception:
                        pass
                    _rec_stop.wait(timeout=1.0 / 15.0)  # ~15 FPS
                if writer is not None:
                    writer.release()

            _rec_thread = threading.Thread(target=_record_loop, daemon=True)
            _rec_thread.start()

        def _apply_gripper_command(step_id: int, target_state: int) -> None:
            nonlocal last_gripper_state
            if cfg.dry_run:
                obs_builder.set_gripper_open_scalar(float(target_state))
                last_gripper_state = target_state
                return

            state_name = "open" if target_state == 1 else "close"
            print(f"  [{step_id:4d}] Gripper {state_name} commanded, waiting...")
            ok = command_gripper_state(
                target_state,
                timeout_s=10.0,
            )
            print(f"  [{step_id:4d}] Gripper {state_name}: {'ok' if ok else 'FAIL'}")
            if ok:
                obs_builder.set_gripper_open_scalar(float(target_state))
                last_gripper_state = target_state
                return

            corrected_state = None
            try:
                corrected_state = gripper_status_to_openpi_state(get_gripper_status())
            except Exception as exc:
                print(f"  [{step_id:4d}] Gripper readback failed: {exc}")
            if corrected_state is not None:
                corrected_name = "open" if corrected_state == 1 else "close"
                obs_builder.set_gripper_open_scalar(float(corrected_state))
                last_gripper_state = corrected_state
                print(f"  [{step_id:4d}] Gripper cache corrected from serial: {corrected_name}")
            else:
                print(f"  [{step_id:4d}] Gripper cache unchanged after failed command")

        try:
            while True:
                queue_state_before_obs = executor.get_progress()
                if queue_state_before_obs["buffered_steps"] > ASYNC_REFILL_THRESHOLD_STEPS:
                    time.sleep(
                        min(
                            max(float(queue_state_before_obs["control_dt_s"]), 0.002),
                            0.01,
                        )
                    )
                    continue

                loop_start = time.monotonic()
                step += 1
                aligned_obs = obs_builder.build_observation(prompt)
                obs = aligned_obs.obs
                obs_queue_state = executor.get_progress()
                observation_tick = int(obs_queue_state["current_tick"])

                t_infer = time.monotonic()
                action_result = policy.infer(obs)
                infer_time = time.monotonic() - t_infer

                raw_actions = np.asarray(action_result["actions"], dtype=np.float64)
                if raw_actions.ndim == 1:
                    raw_actions = raw_actions.reshape(1, -1)

                action_dim = raw_actions.shape[-1]
                if action_dim < 7:
                    raise RuntimeError(
                        f"expected AUBO action_dim >= 7 ([dx,dy,dz,droll,dpitch,dangle,gripper]), got {action_dim}"
                    )
                exec_actions = raw_actions[:MAX_EXEC_ACTION_INTERVALS]
                tcp_deltas = np.zeros((len(exec_actions), 6), dtype=np.float64)
                tcp_deltas[:, :3] = exec_actions[:, :3]
                tcp_deltas[:, 3:5] = exec_actions[:, 3:5]
                tcp_deltas[:, 5] = exec_actions[:, 5]

                # Native mode: linearly interpolate 20 intervals -> 100 actions
                if is_native_speed:
                    interp_deltas = np.repeat(
                        tcp_deltas / float(NATIVE_INTERP_FACTOR),
                        NATIVE_INTERP_FACTOR,
                        axis=0,
                    )
                    tcp_deltas = interp_deltas
                    # Also expand exec_actions for gripper edge detection
                    exec_actions = np.repeat(exec_actions, NATIVE_INTERP_FACTOR, axis=0)

                gripper_action = float(raw_actions[0, 6])
                current_gripper_state = last_gripper_state
                if current_gripper_state is None:
                    current_gripper_state = 1 if aligned_obs.gripper_open_scalar > GRIPPER_THRESHOLD else 0

                tcp_prefix = tcp_deltas
                scheduled_gripper_state: int | None = None
                scheduled_gripper_idx: int | None = None
                gripper_targets = (exec_actions[:, 6] > GRIPPER_THRESHOLD).astype(np.int32)
                edge_indices = np.flatnonzero(gripper_targets != int(current_gripper_state))
                if edge_indices.size > 0:
                    scheduled_gripper_idx = int(edge_indices[0])
                    scheduled_gripper_state = int(gripper_targets[scheduled_gripper_idx])
                    tcp_prefix = tcp_deltas[: scheduled_gripper_idx + 1]

                prefix_integral = np.sum(tcp_prefix, axis=0, dtype=np.float64) if len(tcp_prefix) else np.zeros((6,), dtype=np.float64)
                prefix_abs_integral = np.sum(np.abs(tcp_prefix), axis=0, dtype=np.float64) if len(tcp_prefix) else np.zeros((6,), dtype=np.float64)
                full_integral = np.sum(tcp_deltas, axis=0, dtype=np.float64) if len(tcp_deltas) else np.zeros((6,), dtype=np.float64)
                plan = retime_tcp_action_chunk(
                    tcp_prefix,
                    start_pose_sim=aligned_obs.aligned_tcp_pose_sim,
                    max_linear_speed_mps=effective_speed,
                )
                planning_latency_s = time.monotonic() - loop_start
                submit_queue_state = executor.get_progress()
                stale_steps = max(0, int(submit_queue_state["current_tick"]) - observation_tick)
                stale_steps = min(stale_steps, len(plan.steps))
                remaining_steps = max(0, len(plan.steps) - stale_steps)
                _append_infer_log(
                    {
                        "event": "inference_chunk",
                        "step": int(step),
                        "prompt": prompt,
                        "infer_time_s": float(infer_time),
                        "planning_latency_s": float(planning_latency_s),
                        "raw_action_count": int(len(raw_actions)),
                        "exec_action_count": int(len(exec_actions)),
                        "prefix_action_count": int(len(tcp_prefix)),
                        "full_integral": [float(v) for v in full_integral.tolist()],
                        "prefix_integral": [float(v) for v in prefix_integral.tolist()],
                        "prefix_abs_integral": [float(v) for v in prefix_abs_integral.tolist()],
                        "retimed_step_count": int(len(plan.steps)),
                        "observation_tick": int(observation_tick),
                        "stale_steps_before_submit": int(stale_steps),
                        "remaining_step_count": int(remaining_steps),
                        "queue_refill_threshold_steps": int(ASYNC_REFILL_THRESHOLD_STEPS),
                        "queue_buffered_steps_before_obs": int(queue_state_before_obs["buffered_steps"]),
                        "queue_buffered_steps_at_obs": int(obs_queue_state["buffered_steps"]),
                        "queue_buffered_steps_before_submit": int(submit_queue_state["buffered_steps"]),
                        "gripper_changed": bool(scheduled_gripper_state is not None),
                        "scheduled_gripper_idx": None if scheduled_gripper_idx is None else int(scheduled_gripper_idx),
                        "scheduled_gripper_state": None if scheduled_gripper_state is None else int(scheduled_gripper_state),
                    }
                )

                gripper_changed = scheduled_gripper_state is not None
                if remaining_steps <= 0:
                    print(
                        f"  [{step:4d}] Chunk expired before execution "
                        f"(retimed={len(plan.steps)} stale={stale_steps}), re-inferring..."
                    )
                    continue

                if gripper_changed:
                    # A discrete gripper edge should take precedence over any
                    # previously buffered future motion.
                    executor.clear_pending()

                executor.submit(
                    plan,
                    aligned_obs.robot_snapshot,
                    observation_tick=observation_tick,
                    generation=step,
                )
                post_submit_queue_state = executor.get_progress()
                _append_infer_log(
                    {
                        "event": "chunk_submit",
                        "step": int(step),
                        "prompt": prompt,
                        "observation_tick": int(observation_tick),
                        "submitted_step_count": int(len(plan.steps)),
                        "remaining_step_count": int(remaining_steps),
                        "queue_buffered_steps_after_submit": int(post_submit_queue_state["buffered_steps"]),
                        "queue_max_future_tick_after_submit": int(post_submit_queue_state["max_future_tick"]),
                    }
                )

                if gripper_changed:
                    state_name = "open" if scheduled_gripper_state == 1 else "close"
                    print(
                        f"  [{step:4d}] Gripper {state_name} scheduled at chunk index "
                        f"{scheduled_gripper_idx} within first {len(exec_actions)} intervals, "
                        f"submitting {remaining_steps} queued servo steps and waiting for prefix completion..."
                    )
                    executor.wait_until_idle()
                    last_res = executor.last_result
                    if last_res and not last_res.ok:
                        print(f"    Track error before gripper command: {last_res.reason}")
                        executor.reset_state()
                    else:
                        _apply_gripper_command(step, int(scheduled_gripper_state))
                else:
                    print(
                        f"  [{step:4d}] Async submit: {remaining_steps} queued servo steps "
                        f"(stale {stale_steps}, buffered={post_submit_queue_state['buffered_steps']}), "
                        f"continuing to next observation..."
                    )

                loop_time = time.monotonic() - loop_start
                last_res = executor.last_result
                track_status = "ok" if (last_res and last_res.ok) else ("FAIL" if last_res else "pending")
                samples = last_res.sample_count if last_res else 0
                gripper_str = f"{gripper_action:.2f}" if gripper_action is not None else "N/A"
                print(
                    f"  [{step:4d}] infer={infer_time:.3f}s  "
                    f"track={track_status}  "
                    f"samples={samples}  "
                    f"gripper={gripper_str}  "
                    f"loop={loop_time:.3f}s"
                )

                if last_res and not last_res.ok:
                    print(f"    Track error: {last_res.reason}")
                    executor.reset_state()

                if gripper_changed:
                    continue
                continue

        except KeyboardInterrupt:
            executor.clear_pending()
            print("\n  Waiting for in-flight trajectory to finish...")
            executor.wait_until_idle()
            executor.reset_state()
            _append_infer_log(
                {
                    "event": "session_stop",
                    "prompt": prompt,
                    "steps": int(step),
                    "reason": "keyboard_interrupt",
                }
            )
            if _rec_thread is not None:
                _rec_stop.set()
                _rec_thread.join(timeout=3)
                print(f"  Video saved: {video_path}")
                _rec_thread = None
            print(f"\n  Inference stopped after {step} steps.")

    executor.stop()
    obs_builder.stop()
    print("Shutdown complete. Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

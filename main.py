#!/usr/bin/env python3
"""
Main entrypoint for the Jetson-side real-robot OpenPI workflow.

Architecture:
  - Main thread: serial chunk inference (get_obs -> infer -> execute 10 intervals)
  - Executor thread: executes one 10-interval servo segment at a time
  - Gripper state change: execute TCP prefix, wait for gripper, re-infer

Inference loop:
  get_obs -> infer -> extract TCP(6D) euler deltas + gripper
    -> take the first 10 action intervals
    -> execute the segment to completion
    -> return immediately after the segment finishes
    -> re-observe from the real robot pose and infer again
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = OPENPI_ROOT / "repo"
REPO_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

GRIPPER_THRESHOLD = 0.6
MAX_EXEC_ACTION_INTERVALS = 10
NATIVE_INTERP_FACTOR = 5  # 10 intervals -> 50 actions in native mode
STATE_MODE_YAW = "yaw"
STATE_MODE_J6 = "j6"
DEFAULT_J6_HOME_RAD = 0.11434
def _wrap_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


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
        lock_yaw: bool,
        max_speed_mps: float = 0.05,
        use_joint6: bool = False,
        initial_joint6_rad: float = DEFAULT_J6_HOME_RAD,
    ):
        self._execute = execute
        self._lock_yaw = lock_yaw
        self._max_speed_mps = max_speed_mps
        self._use_joint6 = bool(use_joint6)
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._sentinel = object()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._expected_pose = None
        self._expected_joint6 = float(initial_joint6_rad) if self._use_joint6 else None
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
    def expected_joint6(self):
        with self._lock:
            return None if self._expected_joint6 is None else float(self._expected_joint6)

    @property
    def is_idle(self) -> bool:
        return self._idle.is_set()

    def submit(self, tcp_deltas, observed_pose, *, joint6_deltas=None, semantic_joint6: float | None = None) -> None:
        item = (tcp_deltas, observed_pose, joint6_deltas, semantic_joint6)
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

    def reset_state(self, *, semantic_joint6: float | None = None) -> None:
        with self._lock:
            self._expected_pose = None
            if self._use_joint6 and semantic_joint6 is not None:
                self._expected_joint6 = float(semantic_joint6)
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
        current_joint6_targets = None
        current_joint6_start = None
        current_joint6_final = None

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

        def _set_expected_joint6(value: float | None) -> None:
            if not self._use_joint6:
                return
            with self._lock:
                self._expected_joint6 = None if value is None else float(value)

        def _plan_from_item(item):
            nonlocal servo_active
            tcp_deltas, submitted_observed_pose, joint6_deltas, semantic_joint6 = item
            _ = submitted_observed_pose
            snapshot = get_robot_snapshot()
            # Each new chunk integrates from the current live robot pose.
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
                lock_yaw=self._lock_yaw,
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
                if self._use_joint6:
                    _set_expected_joint6(self.expected_joint6 if semantic_joint6 is None else semantic_joint6)
                return None

            start_joint6 = None
            step_joint6_targets = None
            final_joint6 = None
            if self._use_joint6:
                start_joint6 = float(self.expected_joint6 if semantic_joint6 is None else semantic_joint6)
                joint6_actions = np.zeros((len(tcp_deltas),), dtype=np.float64)
                if joint6_deltas is not None:
                    joint6_actions = np.asarray(joint6_deltas, dtype=np.float64).reshape(-1)
                    if joint6_actions.shape[0] != len(tcp_deltas):
                        raise RuntimeError(
                            f"joint6_deltas length mismatch: expected {len(tcp_deltas)}, got {joint6_actions.shape[0]}"
                        )
                step_counts = [0] * len(joint6_actions)
                for step in plan.steps:
                    src_idx = int(step.source_action_index)
                    if 0 <= src_idx < len(step_counts):
                        step_counts[src_idx] += 1
                step_joint6_targets = []
                current_joint6 = float(start_joint6)
                for action_idx, delta_joint6 in enumerate(joint6_actions):
                    action_steps = step_counts[action_idx]
                    action_delta = _wrap_angle(float(delta_joint6))
                    action_start_joint6 = float(current_joint6)
                    if action_steps <= 0:
                        current_joint6 = _wrap_angle(action_start_joint6 + action_delta)
                        continue
                    for sub_idx in range(action_steps):
                        alpha = float(sub_idx + 1) / float(action_steps)
                        step_joint6_targets.append(_wrap_angle(action_start_joint6 + action_delta * alpha))
                    current_joint6 = _wrap_angle(action_start_joint6 + action_delta)
                final_joint6 = float(current_joint6)
                if len(step_joint6_targets) != len(plan.steps):
                    raise RuntimeError(
                        f"joint6 target planning mismatch: expected {len(plan.steps)} step targets, got {len(step_joint6_targets)}"
                    )

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
                if self._use_joint6:
                    _set_expected_joint6(final_joint6)
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

            daemon.servo_begin_chunk(snapshot.tcp_pose)
            return snapshot, plan, step_joint6_targets, start_joint6, final_joint6

        # hold_pose_real: when a chunk finishes, keep sending this pose to
        # hold the robot in place (servo stays active, no stop/start gap).
        hold_pose_real = None
        hold_joint6 = self.expected_joint6 if self._use_joint6 else None

        while not self._stop.is_set():
            try:
                # Check if reset was requested (e.g. after Ctrl+C)
                with self._lock:
                    need_reset = self._reset_servo
                    if need_reset:
                        self._reset_servo = False
                if need_reset:
                    _finish_servo()
                    hold_pose_real = None
                    hold_joint6 = self.expected_joint6 if self._use_joint6 else None
                    current_plan = None
                    current_snapshot = None
                    current_index = 0
                    current_joint6_targets = None
                    current_joint6_start = None
                    current_joint6_final = None

                if current_plan is None:
                    item = _drain_latest_pending(timeout=0.05)
                    if item is None:
                        # No new chunk – hold position if servo is active
                        if servo_active and hold_pose_real is not None:
                            if self._use_joint6 and hold_joint6 is not None:
                                daemon.servo_pose_j6(hold_pose_real, hold_joint6)
                            else:
                                daemon.servo_pose(hold_pose_real)
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
                    current_snapshot, current_plan, current_joint6_targets, current_joint6_start, current_joint6_final = planned
                    current_index = 0

                if current_plan is None or current_index >= len(current_plan.steps):
                    current_snapshot = None
                    current_plan = None
                    current_joint6_targets = None
                    current_joint6_start = None
                    current_joint6_final = None
                    current_index = 0
                    if self._queue.empty():
                        # Don't stop servo – just hold position
                        self._idle.set()
                    continue

                step = current_plan.steps[current_index]
                step_joint6 = None if current_joint6_targets is None else float(current_joint6_targets[current_index])
                if step_joint6 is None:
                    resp = daemon.servo_pose(step.pose_real)
                else:
                    resp = daemon.servo_pose_j6(step.pose_real, step_joint6)
                pose_ret = int(resp.get("servo_pose_ret", -1))
                error = str(resp.get("error", ""))

                if pose_ret == 0:
                    current_index += 1
                    hold_pose_real = step.pose_real.copy()
                    if step_joint6 is not None:
                        hold_joint6 = float(step_joint6)
                        _set_expected_joint6(hold_joint6)
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
                            raw={**resp, **({"semantic_joint6": float(step_joint6)} if step_joint6 is not None else {})},
                            start_pose_real=current_plan.start_pose_real,
                            final_pose_real=step.pose_real,
                        )
                    )
                    _set_expected_pose(step.pose_sim)
                    if current_index >= len(current_plan.steps):
                        if self._use_joint6 and current_joint6_final is not None:
                            hold_joint6 = float(current_joint6_final)
                            _set_expected_joint6(hold_joint6)
                        current_snapshot = None
                        current_plan = None
                        current_joint6_targets = None
                        current_joint6_start = None
                        current_joint6_final = None
                        current_index = 0
                        if self._queue.empty():
                            self._idle.set()
                else:
                    final_pose = step.pose_sim if current_index > 0 else current_plan.start_pose
                    final_pose_real = step.pose_real if current_index > 0 else current_plan.start_pose_real
                    final_joint6 = None
                    if self._use_joint6:
                        if current_index > 0:
                            final_joint6 = hold_joint6
                        else:
                            final_joint6 = current_joint6_start
                        _set_expected_joint6(final_joint6)
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
                            raw={**resp, **({"semantic_joint6": float(final_joint6)} if final_joint6 is not None else {})},
                            start_pose_real=current_plan.start_pose_real,
                            final_pose_real=final_pose_real,
                        )
                    )
                    _set_expected_pose(final_pose if current_index > 0 else None)
                    current_snapshot = None
                    current_plan = None
                    current_joint6_targets = None
                    current_joint6_start = None
                    current_joint6_final = None
                    current_index = 0
                    if error == "safety":
                        _finish_servo()
                        hold_pose_real = None
                        hold_joint6 = self.expected_joint6 if self._use_joint6 else None

            except Exception as exc:
                _set_last_result(None)
                _set_expected_pose(None)
                current_snapshot = None
                current_plan = None
                current_joint6_targets = None
                current_joint6_start = None
                current_joint6_final = None
                current_index = 0
                hold_pose_real = None
                print(f"  [executor] Error: {exc}")
                _finish_servo()

            if current_plan is None and self._queue.empty():
                self._idle.set()


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

    from support.get_obs import RealRobotOpenPIObservationBuilder, STATE_MODE_J6, STATE_MODE_YAW
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
    from support.tcp_control import build_helper as build_tcp_helper, get_robot_snapshot
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
            lock_yaw=True,
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
                print(f"  Execute: {execute_now}  Lock Yaw: {cfg_now.lock_yaw}")
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
    print(f"  State Mode: {cfg.obs_state_mode} (lock_yaw={'yes' if cfg.lock_yaw else 'no'})")
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

    obs_state_mode = str(cfg.obs_state_mode).strip().lower()
    if obs_state_mode not in (STATE_MODE_YAW, STATE_MODE_J6):
        print(
            f"Invalid TUI State Mode={obs_state_mode!r}, "
            f"fallback to '{STATE_MODE_YAW}' (valid: '{STATE_MODE_YAW}'/'{STATE_MODE_J6}')"
        )
        obs_state_mode = STATE_MODE_YAW
    print(f"OBS_STATE_MODE={obs_state_mode}")
    use_joint6_mode = obs_state_mode == STATE_MODE_J6
    semantic_joint6_rad: float | None = float(DEFAULT_J6_HOME_RAD) if use_joint6_mode else None
    obs_builder = RealRobotOpenPIObservationBuilder(state_mode=obs_state_mode)
    obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad if use_joint6_mode else None)
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
    executor = TrajectoryExecutor(
        execute=execute,
        lock_yaw=cfg.lock_yaw,
        max_speed_mps=effective_speed,
        use_joint6=use_joint6_mode,
        initial_joint6_rad=float(DEFAULT_J6_HOME_RAD),
    )
    executor.start()

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
                print(f"  Execute: {execute}  Lock Yaw: {cfg.lock_yaw}")
                print(f"  TCP pose: {np.round(snap.tcp_pose, 5).tolist()}")
                print(f"  Joint q (deg): {np.round(np.degrees(snap.joint_q), 2).tolist()}")
                if use_joint6_mode and semantic_joint6_rad is not None:
                    print(f"  Semantic j6 (deg): {round(np.degrees(semantic_joint6_rad), 2)}")
                print(f"  Collision: {snap.collision}")
                print(f"  Safety limits OK: {snap.within_safety_limits}")
                print(f"  Executor idle: {executor.is_idle}")
            except Exception as exc:
                print(f"  Error: {exc}")
            continue

        if cmd_lower == "align":
            executor.clear_pending()
            executor.wait_until_idle()
            if use_joint6_mode:
                semantic_joint6_rad = float(DEFAULT_J6_HOME_RAD)
                obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)
                executor.reset_state(semantic_joint6=semantic_joint6_rad)
            else:
                executor.reset_state()
            print("Moving to initial joint positions...")
            result = move_to_joint_positions(initial_qpos_rad, execute=execute)
            if use_joint6_mode:
                semantic_joint6_rad = float(DEFAULT_J6_HOME_RAD)
                obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)
                executor.reset_state(semantic_joint6=semantic_joint6_rad)
            else:
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
            if use_joint6_mode:
                semantic_joint6_rad = float(DEFAULT_J6_HOME_RAD)
                obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)
                executor.reset_state(semantic_joint6=semantic_joint6_rad)
            else:
                executor.reset_state()
            obs_builder.reset_pose_filter()
            policy.reset()
            print("Policy state reset. Executor state cleared.")
            continue

        # --- Start inference loop ---
        prompt = cmd
        print(f"Starting inference loop with prompt: \"{prompt}\"")
        print(f"  Execute={execute}  LockYaw={cfg.lock_yaw}  PoseFrame={get_alignment_mode()}")
        print("  Press Ctrl+C to stop.\n")

        policy.reset()
        if use_joint6_mode:
            semantic_joint6_rad = float(DEFAULT_J6_HOME_RAD)
            obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)
            executor.reset_state(semantic_joint6=semantic_joint6_rad)
        else:
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
        if use_joint6_mode:
            obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)
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

        def _sync_semantic_joint6_from_executor() -> None:
            nonlocal semantic_joint6_rad
            if not use_joint6_mode:
                return
            latest_joint6 = executor.expected_joint6
            if latest_joint6 is not None:
                semantic_joint6_rad = float(latest_joint6)
            obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)

        try:
            while True:
                loop_start = time.monotonic()
                step += 1
                if use_joint6_mode:
                    obs_builder.set_semantic_joint6_scalar(semantic_joint6_rad)

                aligned_obs = obs_builder.build_observation(prompt)
                obs = aligned_obs.obs

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
                joint6_deltas = None
                if use_joint6_mode:
                    tcp_deltas[:, 5] = 0.0
                    joint6_deltas = np.asarray(exec_actions[:, 5], dtype=np.float64).copy()
                else:
                    tcp_deltas[:, 5] = exec_actions[:, 5]

                # Native mode: linearly interpolate 10 intervals -> 50 actions
                if is_native_speed:
                    interp_deltas = np.repeat(
                        tcp_deltas / float(NATIVE_INTERP_FACTOR),
                        NATIVE_INTERP_FACTOR,
                        axis=0,
                    )
                    tcp_deltas = interp_deltas
                    if joint6_deltas is not None:
                        joint6_deltas = np.repeat(
                            joint6_deltas / float(NATIVE_INTERP_FACTOR),
                            NATIVE_INTERP_FACTOR,
                            axis=0,
                        )
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
                    if joint6_deltas is not None:
                        joint6_prefix = joint6_deltas[: scheduled_gripper_idx + 1]
                    else:
                        joint6_prefix = None
                else:
                    joint6_prefix = joint6_deltas

                executor.submit(
                    tcp_prefix,
                    aligned_obs.aligned_tcp_pose_sim,
                    joint6_deltas=joint6_prefix,
                    semantic_joint6=semantic_joint6_rad,
                )

                gripper_changed = scheduled_gripper_state is not None
                if gripper_changed:
                    state_name = "open" if scheduled_gripper_state == 1 else "close"
                    print(
                        f"  [{step:4d}] Gripper {state_name} scheduled at chunk index "
                        f"{scheduled_gripper_idx} within first {len(exec_actions)} intervals, "
                        f"executing TCP prefix ({len(tcp_prefix)} intervals)..."
                    )
                    executor.wait_until_idle()
                    _sync_semantic_joint6_from_executor()
                    last_res = executor.last_result
                    if last_res and not last_res.ok:
                        print(f"    Track error before gripper command: {last_res.reason}")
                        if use_joint6_mode:
                            executor.reset_state(semantic_joint6=semantic_joint6_rad)
                        else:
                            executor.reset_state()
                    else:
                        _apply_gripper_command(step, int(scheduled_gripper_state))
                else:
                    print(
                        f"  [{step:4d}] Executing first {len(tcp_prefix)} intervals, then re-infer..."
                    )
                    executor.wait_until_idle()
                    _sync_semantic_joint6_from_executor()
                    last_res = executor.last_result
                    if last_res and not last_res.ok:
                        print(f"    Track error: {last_res.reason}")
                        if use_joint6_mode:
                            executor.reset_state(semantic_joint6=semantic_joint6_rad)
                        else:
                            executor.reset_state()

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
                    if use_joint6_mode:
                        executor.reset_state(semantic_joint6=semantic_joint6_rad)
                    else:
                        executor.reset_state()

                if gripper_changed:
                    continue
                continue

        except KeyboardInterrupt:
            executor.clear_pending()
            print("\n  Waiting for in-flight trajectory to finish...")
            executor.wait_until_idle()
            _sync_semantic_joint6_from_executor()
            if use_joint6_mode:
                executor.reset_state(semantic_joint6=semantic_joint6_rad)
            else:
                executor.reset_state()
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from support.pose_align import real_pose_to_sim

from task.pick_and_place import TaskStep


OPEN_PROMPT = "open the storage box"
CLOSE_PROMPT = "close the storage box"

# Obstacle constants
NUM_OBSTACLES = 5
OBSTACLE_NAMES = tuple(f"obj{i}" for i in range(1, NUM_OBSTACLES + 1))
CLEAR_BAND_M = 0.12
CLEAR_MIN_SPACING_M = 0.10
BAND_OBJECT_COUNT_MIN = 2
BAND_OBJECT_COUNT_MAX = 4
STACK_PROBABILITY = 0.70
OBSTACLE_LAYOUT_X_MAX_M = 0.60
OBSTACLE_ROTATE_PROB = 0.70
OBSTACLE_ROTATE_DEG_MIN = float(np.degrees(np.pi / 15.0))
OBSTACLE_ROTATE_DEG_MAX = float(np.degrees(np.pi / 8.0))


def _bounded_layout_x_max(workspace_x_min: float, workspace_x_max: float) -> float:
    bounded_x_max = min(float(workspace_x_max), OBSTACLE_LAYOUT_X_MAX_M)
    if bounded_x_max < float(workspace_x_min):
        raise RuntimeError(
            f"invalid layout x bounds: x_min={workspace_x_min}, x_max={bounded_x_max}"
        )
    return bounded_x_max


@dataclass(frozen=True)
class OpenCloseReference:
    x_start: float
    y_start: float
    reference_pose6: np.ndarray


@dataclass(frozen=True)
class MoveStep:
    x: float
    y: float
    z: float
    note: str


@dataclass(frozen=True)
class OpenCloseEpisodePlan:
    task_kind: str  # "open" | "close"
    prompt: str
    recorded_steps: list[MoveStep]


@dataclass
class OpenCloseSession:
    reference: OpenCloseReference | None = None
    obstacle_scene: "ObstacleScene | None" = None
    episode_count: int = 0


@dataclass(frozen=True)
class OpenCloseCyclePlan:
    cycle_index: int
    clearing_steps: list[TaskStep]
    scene_after_clear: "ObstacleScene"
    open_plan: OpenCloseEpisodePlan
    close_plan: OpenCloseEpisodePlan


@dataclass
class OpenCloseRecordedCycle:
    plan: OpenCloseCyclePlan
    combined_open_frames: list[Any]
    close_frames: list[Any]


@dataclass
class ObstacleState:
    name: str
    xy: tuple[float, float]
    is_rotate: bool
    deg: float
    upper: str | None = None
    lower: str | None = None


def build_reference_from_tcp_pose(tcp_pose6: np.ndarray) -> OpenCloseReference:
    pose = np.asarray(tcp_pose6, dtype=np.float64).reshape(6).copy()
    return OpenCloseReference(
        x_start=float(pose[0]),
        y_start=float(pose[1]),
        reference_pose6=pose,
    )


def build_open_episode_plan(
    reference: OpenCloseReference,
    *,
    target_z: float,
    press_z: float,
) -> OpenCloseEpisodePlan:
    x_start = float(reference.x_start)
    y_start = float(reference.y_start)
    return OpenCloseEpisodePlan(
        task_kind="open",
        prompt=OPEN_PROMPT,
        recorded_steps=[
            MoveStep(x=x_start, y=y_start, z=float(target_z), note="open step 1 approach"),
            MoveStep(x=x_start, y=y_start, z=float(press_z), note="open step 2 lower"),
            MoveStep(x=x_start - 0.20, y=y_start, z=float(press_z), note="open step 3 pull"),
            MoveStep(x=x_start - 0.20, y=y_start, z=float(target_z), note="open step 4 lift"),
        ],
    )


def build_close_episode_plan(
    reference: OpenCloseReference,
    *,
    target_z: float,
    press_z: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
) -> OpenCloseEpisodePlan:
    x_rand = float(np.random.uniform(workspace_x_min, workspace_x_max))
    y_rand = float(np.random.uniform(workspace_y_min, workspace_y_max))
    x_start = float(reference.x_start)
    y_start = float(reference.y_start)
    return OpenCloseEpisodePlan(
        task_kind="close",
        prompt=CLOSE_PROMPT,
        recorded_steps=[
            MoveStep(x=x_rand, y=y_rand, z=float(target_z), note="close step 1 random approach"),
            MoveStep(x=x_start - 0.26, y=y_start, z=float(target_z), note="close step 2 align high"),
            MoveStep(x=x_start - 0.26, y=y_start, z=float(press_z - 0.02), note="close step 3 lower"),
            MoveStep(x=x_start + 0.01, y=y_start, z=float(press_z - 0.02), note="close step 4 push"),
        ],
    )


def restore_session(
    saved: dict[str, Any] | None,
    *,
    resume_mode: str,
) -> tuple[OpenCloseSession, bool]:
    session = OpenCloseSession()
    should_clear_saved_state = False

    if resume_mode == "reset":
        if isinstance(saved, dict) and (
            saved.get("open_close_reference") is not None
            or saved.get("obstacle_scene") is not None
            or bool(saved.get("has_valid_open_close_episode_count", False))
        ):
            should_clear_saved_state = True
        return session, should_clear_saved_state

    if resume_mode != "continue":
        return session, should_clear_saved_state

    if saved is None:
        print("\n  Resume selected, but no saved state exists. Open/close reference will use startup TCP.")
        return session, should_clear_saved_state

    session.episode_count = int(saved.get("open_close_episode_count", 0))
    saved_reference = saved.get("open_close_reference")
    saved_obstacle_scene = saved.get("obstacle_scene")
    if isinstance(saved_obstacle_scene, ObstacleScene):
        session.obstacle_scene = saved_obstacle_scene
    if isinstance(saved_reference, OpenCloseReference):
        session.reference = saved_reference
        print(
            "\n  Found saved open/close reference: "
            f"x_start={session.reference.x_start:.4f}, "
            f"y_start={session.reference.y_start:.4f}"
        )
        if session.obstacle_scene is not None:
            print(f"  Loaded obstacle scene with {len(session.obstacle_scene.obstacles)} objects.")
        print(f"  Resuming with saved open/close reference (episode_count={session.episode_count}).")
    else:
        if saved.get("has_valid_open_close_episode_count", False):
            print(
                "\n  Restored open/close counters from saved state: "
                f"episode_count={session.episode_count}."
            )
        print(
            "  Saved state has no reusable open/close reference. "
            "Open/close reference will use startup TCP."
        )

    return session, should_clear_saved_state


class ObstacleScene:
    def __init__(self, obstacles: dict[str, ObstacleState]) -> None:
        self.obstacles = obstacles

    @classmethod
    def empty(cls) -> "ObstacleScene":
        return cls({})

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "ObstacleScene | None":
        if not isinstance(payload, dict):
            return None
        obstacles: dict[str, ObstacleState] = {}
        for name in OBSTACLE_NAMES:
            raw = payload.get(name)
            if raw is None or not isinstance(raw, dict):
                return None
            xy = raw.get("xy")
            if xy is None:
                return None
            arr = np.asarray(xy, dtype=np.float64).reshape(2)
            obstacles[name] = ObstacleState(
                name=name,
                xy=(float(arr[0]), float(arr[1])),
                is_rotate=bool(raw.get("is_rotate", False)),
                deg=float(raw.get("deg", 0.0)),
                upper=_norm_link(raw.get("upper")),
                lower=_norm_link(raw.get("lower")),
            )
        return cls(obstacles)

    def copy(self) -> "ObstacleScene":
        return ObstacleScene(
            {
                n: ObstacleState(
                    name=o.name,
                    xy=tuple(o.xy),
                    is_rotate=o.is_rotate,
                    deg=o.deg,
                    upper=o.upper,
                    lower=o.lower,
                )
                for n, o in self.obstacles.items()
            }
        )

    def to_serializable(self) -> dict[str, dict[str, Any]]:
        return {
            n: {
                "xy": [o.xy[0], o.xy[1]],
                "is_rotate": o.is_rotate,
                "deg": o.deg,
                "upper": o.upper,
                "lower": o.lower,
            }
            for n, o in self.obstacles.items()
        }

    def occupied_positions(self, *, exclude: set[str] | None = None) -> list[np.ndarray]:
        exc = exclude or set()
        return [np.asarray(o.xy) for n, o in self.obstacles.items() if n not in exc]

    def top_of(self, name: str) -> str:
        current = name
        seen: set[str] = set()
        while self.obstacles[current].upper is not None:
            if current in seen:
                raise RuntimeError(f"cycle in upper chain from {name}")
            seen.add(current)
            current = self.obstacles[current].upper  # type: ignore[assignment]
        return current

    def detach_top(self, name: str) -> None:
        obj = self.obstacles[name]
        if obj.upper is not None:
            raise RuntimeError(f"cannot detach {name}: upper={obj.upper}")
        if obj.lower is not None:
            self.obstacles[obj.lower].upper = None
        obj.lower = None

    def place_on_table(self, name: str, xy: tuple[float, float], *, is_rotate: bool, deg: float) -> None:
        obj = self.obstacles[name]
        obj.xy = (float(xy[0]), float(xy[1]))
        obj.is_rotate = bool(is_rotate)
        obj.deg = float(deg) if is_rotate else 0.0
        obj.lower = None

    def place_on_object(self, name: str, target: str) -> None:
        actual = self.top_of(target)
        obj = self.obstacles[name]
        obj.xy = tuple(self.obstacles[actual].xy)
        obj.is_rotate = self.obstacles[actual].is_rotate
        obj.deg = self.obstacles[actual].deg
        obj.lower = actual
        self.obstacles[actual].upper = name


def sample_obstacle_orientation(
    *,
    rotate_prob: float = OBSTACLE_ROTATE_PROB,
    rotate_deg_min: float = OBSTACLE_ROTATE_DEG_MIN,
    rotate_deg_max: float = OBSTACLE_ROTATE_DEG_MAX,
) -> tuple[bool, float]:
    if np.random.random() < rotate_prob:
        abs_deg = float(np.random.uniform(rotate_deg_min, rotate_deg_max))
        sign = -1.0 if np.random.random() < 0.5 else 1.0
        return True, sign * abs_deg
    return False, 0.0


def sample_obstacle_xy(
    scene: ObstacleScene,
    *,
    object_name: str,
    workspace_x_min: float,
    workspace_x_max: float,
    y_min: float,
    y_max: float,
    min_spacing: float = CLEAR_MIN_SPACING_M,
) -> tuple[float, float]:
    if float(y_max) < float(y_min):
        raise RuntimeError(f"invalid layout y bounds: y_min={y_min}, y_max={y_max}")

    layout_x_max = _bounded_layout_x_max(workspace_x_min, workspace_x_max)
    occupied = scene.occupied_positions(exclude={object_name})
    for _ in range(512):
        x = float(np.random.uniform(workspace_x_min, layout_x_max))
        y = float(np.random.uniform(y_min, y_max))
        candidate = np.array([x, y], dtype=np.float64)
        if all(float(np.linalg.norm(candidate - p)) >= min_spacing for p in occupied):
            return float(candidate[0]), float(candidate[1])
    raise RuntimeError(f"failed to sample obstacle xy for {object_name}")


def build_clearing_steps(
    scene: ObstacleScene,
    *,
    y_target: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    min_spacing: float = CLEAR_MIN_SPACING_M,
) -> tuple[list[TaskStep], ObstacleScene]:
    """Build pick/place steps to clear all obstacles inside the y-target band."""
    result = scene.copy()
    band_lo = y_target - CLEAR_BAND_M
    band_hi = y_target + CLEAR_BAND_M
    layout_x_max = _bounded_layout_x_max(workspace_x_min, workspace_x_max)

    can_go_upper = (workspace_y_max - band_hi) >= min_spacing
    can_go_lower = (band_lo - workspace_y_min) >= min_spacing

    def _objects_in_band() -> list[str]:
        in_band: list[str] = []
        for name, obj in result.obstacles.items():
            if band_lo <= obj.xy[1] <= band_hi:
                in_band.append(name)
        return in_band

    def _pick_clear_y(obj_y: float) -> tuple[float, float]:
        wants_upper = obj_y >= y_target
        if wants_upper and can_go_upper:
            return band_hi + min_spacing, workspace_y_max
        if (not wants_upper) and can_go_lower:
            return workspace_y_min, band_lo - min_spacing
        if can_go_upper:
            return band_hi + min_spacing, workspace_y_max
        if can_go_lower:
            return workspace_y_min, band_lo - min_spacing
        return workspace_y_min, workspace_y_max

    def _sample_clear_xy(obj_name: str, obj_y: float) -> np.ndarray:
        y_lo, y_hi = _pick_clear_y(obj_y)
        occupied = result.occupied_positions(exclude={obj_name})
        candidate_ranges = [
            (float(y_lo), float(y_hi)),
            (float(workspace_y_min), float(workspace_y_max)),
        ]
        for y_range_lo, y_range_hi in candidate_ranges:
            for _ in range(512):
                x = float(np.random.uniform(workspace_x_min, layout_x_max))
                y = float(np.random.uniform(y_range_lo, y_range_hi))
                candidate = np.array([x, y], dtype=np.float64)
                if all(float(np.linalg.norm(candidate - p)) >= min_spacing for p in occupied):
                    return candidate
        raise RuntimeError(f"failed to sample clearing xy for {obj_name}")

    steps: list[TaskStep] = []
    processed: set[str] = set()

    def _clear_object(name: str) -> None:
        if name in processed:
            return
        obj = result.obstacles[name]
        if obj.upper is not None:
            _clear_object(obj.upper)

        if name not in _objects_in_band():
            return

        obj = result.obstacles[name]
        level = 1 if obj.lower is not None else 0
        steps.append(
            TaskStep(
                kind="pick",
                object_name=name,
                xy=tuple(obj.xy),
                level=level,
                is_rotate=obj.is_rotate,
                deg=obj.deg,
                note=f"clear pick {name}",
                align_j6=True,
            )
        )
        result.detach_top(name)

        clear_xy = _sample_clear_xy(name, obj.xy[1])
        steps.append(
            TaskStep(
                kind="place",
                object_name=name,
                xy=(float(clear_xy[0]), float(clear_xy[1])),
                level=0,
                is_rotate=False,
                deg=0.0,
                note=f"clear place {name}",
                align_j6=True,
            )
        )
        result.place_on_table(name, (float(clear_xy[0]), float(clear_xy[1])), is_rotate=False, deg=0.0)
        processed.add(name)

    for name in _objects_in_band():
        _clear_object(name)

    return steps, result


def _norm_link(v: Any) -> str | None:
    if v is None:
        return None
    text = str(v).strip()
    return text if text else None


def execute_move_step(
    runtime,
    step: MoveStep,
    *,
    record: bool,
    frame_idx: int,
    base_pose6: np.ndarray,
):
    step_frames: list[Any] = []
    target_pose = runtime.build_pose_at_xy(base_pose6, float(step.x), float(step.y), float(step.z))
    if runtime.dry_run:
        dummy_count = 6
        sim_pose = real_pose_to_sim(target_pose)
        for idx in range(dummy_count):
            step_frames.append(
                runtime.make_dummy_frame(
                    sim_pose=sim_pose,
                    gripper=1.0,
                    joint6=float(runtime.default_joint6_rad),
                    frame_idx=frame_idx + idx,
                )
            )
        return step_frames, frame_idx + len(step_frames)

    print(f"    [move] ({step.x:.4f}, {step.y:.4f}, {step.z:.4f})  {step.note}")
    seg = runtime.record_pose_move(
        target_pose,
        gripper=1.0,
        start_frame_idx=frame_idx,
        record=record,
        semantic_joint6=float(runtime.default_joint6_rad),
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)
    return step_frames, frame_idx


def execute_move_step_sequence(
    runtime,
    steps: list[MoveStep],
    *,
    record: bool,
    base_pose6: np.ndarray,
):
    frames: list[Any] = []
    frame_idx = 0
    runtime.begin_task_servo()
    try:
        for step in steps:
            step_frames, frame_idx = execute_move_step(
                runtime,
                step,
                record=record,
                frame_idx=frame_idx,
                base_pose6=base_pose6,
            )
            frames.extend(step_frames)
    finally:
        runtime.end_task_servo()
    return frames


def prepare_session(
    runtime,
    session: OpenCloseSession,
    *,
    startup_tcp_pose: np.ndarray,
    origin_xy: np.ndarray,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    min_spacing: float,
) -> None:
    print("\n=== Open/Close Reference Setup ===")
    if session.reference is None:
        print("Auto-capturing startup TCP as OPEN start point (x/y source).")
        session.reference = build_reference_from_tcp_pose(startup_tcp_pose)
        print(
            "  Captured open reference: "
            f"x_start={session.reference.x_start:.4f}, "
            f"y_start={session.reference.y_start:.4f}"
        )
    else:
        print(
            "Using saved open reference: "
            f"x_start={session.reference.x_start:.4f}, "
            f"y_start={session.reference.y_start:.4f}"
        )

    if session.obstacle_scene is None:
        print("Place objects one by one at the pickup point. The robot will grab continuously.")
        session.obstacle_scene = initialize_obstacle_scene(
            runtime,
            origin_xy=origin_xy,
            workspace_x_min=workspace_x_min,
            workspace_x_max=workspace_x_max,
            workspace_y_min=workspace_y_min,
            workspace_y_max=workspace_y_max,
            min_spacing=min_spacing,
        )
    else:
        print(f"\n  Using saved obstacle scene ({len(session.obstacle_scene.obstacles)} objects).")


def describe_cycle(plan: OpenCloseCyclePlan) -> None:
    cycle_label = f"Cycle {plan.cycle_index}"
    print(f"\n--- {cycle_label} [clear -> open -> close] ---")
    print(f"  Clearing steps: {len(plan.clearing_steps)}")
    print(f"  Open prompt:  \"{plan.open_plan.prompt}\"")
    print(f"  Open steps:   {len(plan.open_plan.recorded_steps)}")
    print(f"  Close prompt: \"{plan.close_plan.prompt}\"")
    print(f"  Close steps:  {len(plan.close_plan.recorded_steps)}")


def plan_cycle(
    runtime,
    session: OpenCloseSession,
    *,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    target_z: float,
    press_z: float,
    clear_spacing: float,
    stack_probability: float,
    band_count_min: int,
    band_count_max: int,
) -> OpenCloseCyclePlan:
    if session.reference is None:
        raise RuntimeError("open/close reference is not initialized")
    if session.obstacle_scene is None:
        raise RuntimeError("obstacle scene is not initialized")

    is_first_cycle = session.episode_count == 0
    cycle_layout = session.obstacle_scene.copy()
    if is_first_cycle:
        print("  [layout] First cycle: using initial placement, skip rearrange.")
    else:
        print("  [layout] Rearranging obstacles with live sampling...")
        cycle_layout = rearrange_open_close_scene(
            runtime,
            session.obstacle_scene,
            reference_y=session.reference.y_start,
            workspace_x_min=workspace_x_min,
            workspace_x_max=workspace_x_max,
            workspace_y_min=workspace_y_min,
            workspace_y_max=workspace_y_max,
            clear_spacing=clear_spacing,
            stack_probability=stack_probability,
            band_count_min=band_count_min,
            band_count_max=band_count_max,
        )

    session.obstacle_scene = cycle_layout.copy()
    clearing_steps, scene_after_clear = build_clearing_steps(
        session.obstacle_scene,
        y_target=session.reference.y_start,
        workspace_x_min=workspace_x_min,
        workspace_x_max=workspace_x_max,
        workspace_y_min=workspace_y_min,
        workspace_y_max=workspace_y_max,
        min_spacing=clear_spacing,
    )
    return OpenCloseCyclePlan(
        cycle_index=session.episode_count // 2,
        clearing_steps=clearing_steps,
        scene_after_clear=scene_after_clear,
        open_plan=build_open_episode_plan(
            reference=session.reference,
            target_z=float(target_z),
            press_z=float(press_z),
        ),
        close_plan=build_close_episode_plan(
            reference=session.reference,
            target_z=float(target_z),
            press_z=float(press_z),
            workspace_x_min=workspace_x_min,
            workspace_x_max=workspace_x_max,
            workspace_y_min=workspace_y_min,
            workspace_y_max=workspace_y_max,
        ),
    )


def _obstacle_scene_to_pick_state(scene: ObstacleScene) -> dict[str, dict[str, Any]]:
    return {
        oname: {
            "xy": [float(ostate.xy[0]), float(ostate.xy[1])],
            "is_rotate": bool(ostate.is_rotate),
            "deg": float(ostate.deg),
            "standard_j6_rad": None,
            "upper": ostate.upper,
            "lower": ostate.lower,
        }
        for oname, ostate in scene.obstacles.items()
    }


def record_cycle(
    runtime,
    session: OpenCloseSession,
    plan: OpenCloseCyclePlan,
) -> OpenCloseRecordedCycle:
    if session.reference is None:
        raise RuntimeError("open/close reference is not initialized")
    if session.obstacle_scene is None:
        raise RuntimeError("obstacle scene is not initialized")

    from task.pick_and_place import execute_step_sequence

    clearing_frames: list[Any] = []
    if plan.clearing_steps:
        clear_scene_state = _obstacle_scene_to_pick_state(session.obstacle_scene)
        clearing_frames, held_after_clear = execute_step_sequence(
            runtime,
            plan.clearing_steps,
            record=True,
            scene_state=clear_scene_state,
            lookup_scene_state=clear_scene_state,
            result_scene_state=clear_scene_state,
        )
        if held_after_clear is not None:
            raise RuntimeError(f"clearing ended while holding {held_after_clear}")
        if not runtime.dry_run:
            runtime.return_home("post-clear return home")

    open_frames = execute_move_step_sequence(
        runtime,
        plan.open_plan.recorded_steps,
        record=True,
        base_pose6=session.reference.reference_pose6,
    )
    close_frames = execute_move_step_sequence(
        runtime,
        plan.close_plan.recorded_steps,
        record=True,
        base_pose6=session.reference.reference_pose6,
    )

    if not open_frames:
        raise RuntimeError("open episode produced no recorded move frames")
    if not close_frames:
        raise RuntimeError("close episode produced no recorded move frames")

    return OpenCloseRecordedCycle(
        plan=plan,
        combined_open_frames=[*clearing_frames, *open_frames],
        close_frames=close_frames,
    )


def finalize_cycle(
    session: OpenCloseSession,
    recorded: OpenCloseRecordedCycle,
) -> None:
    session.obstacle_scene = recorded.plan.scene_after_clear.copy()
    session.episode_count += 2


def initialize_obstacle_scene(
    runtime,
    *,
    origin_xy: np.ndarray,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    min_spacing: float,
) -> ObstacleScene:
    from task.pick_and_place import execute_step_sequence

    scene = ObstacleScene({})

    def _sample_with_origin_guard(obj_name: str) -> tuple[float, float]:
        for _ in range(512):
            xy = sample_obstacle_xy(
                scene,
                object_name=obj_name,
                workspace_x_min=workspace_x_min,
                workspace_x_max=workspace_x_max,
                y_min=workspace_y_min,
                y_max=workspace_y_max,
                min_spacing=min_spacing,
            )
            if float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64))) >= float(min_spacing):
                return xy
        raise RuntimeError(f"failed to sample obstacle xy for {obj_name} away from origin")

    print(f"\n=== Obstacle Initialization ({NUM_OBSTACLES} objects) ===")
    for idx, obj_name in enumerate(OBSTACLE_NAMES):
        xy = _sample_with_origin_guard(obj_name)
        is_rotate, deg = sample_obstacle_orientation()
        print(
            f"  [{idx + 1}/{NUM_OBSTACLES}] {obj_name}: "
            f"-> ({xy[0]:.4f}, {xy[1]:.4f}), rotate={is_rotate}, deg={deg:.1f}"
        )
        if not runtime.dry_run:
            pick_step = TaskStep(
                kind="pick",
                object_name=obj_name,
                xy=(float(origin_xy[0]), float(origin_xy[1])),
                level=0,
                is_rotate=False,
                deg=0.0,
                align_j6=True,
                note=f"obstacle prep pick {obj_name}",
            )
            place_step = TaskStep(
                kind="place",
                object_name=obj_name,
                xy=(float(xy[0]), float(xy[1])),
                level=0,
                is_rotate=is_rotate,
                deg=deg,
                align_j6=True,
                note=f"obstacle prep place {obj_name}",
            )
            temp_state = {
                obj_name: {
                    "xy": [float(origin_xy[0]), float(origin_xy[1])],
                    "is_rotate": False,
                    "deg": 0.0,
                    "standard_j6_rad": None,
                    "upper": None,
                    "lower": None,
                }
            }
            _, held_after = execute_step_sequence(
                runtime,
                [pick_step, place_step],
                record=False,
                scene_state=temp_state,
                lookup_scene_state=temp_state,
                result_scene_state=temp_state,
            )
            if held_after is not None:
                raise RuntimeError(f"obstacle prep ended while holding {held_after}")
            runtime.return_home(f"[obstacle prep {obj_name}] return home")
        scene.obstacles[obj_name] = ObstacleState(
            name=obj_name,
            xy=(float(xy[0]), float(xy[1])),
            is_rotate=bool(is_rotate),
            deg=float(deg),
        )
    print(f"  All {NUM_OBSTACLES} obstacles placed.")
    return scene


def rearrange_open_close_scene(
    runtime,
    scene: ObstacleScene,
    *,
    reference_y: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    clear_spacing: float,
    stack_probability: float,
    band_count_min: int,
    band_count_max: int,
) -> ObstacleScene:
    from task.pick_and_place import execute_step_sequence

    result_scene = scene.copy()
    band_y_lo = max(float(reference_y - CLEAR_BAND_M), float(workspace_y_min))
    band_y_hi = min(float(reference_y + CLEAR_BAND_M), float(workspace_y_max))

    def _execute_rearrange_move(
        obj_name: str,
        source_xy: tuple[float, float],
        source_is_rotate: bool,
        source_deg: float,
        target_xy: tuple[float, float],
        *,
        level: int,
        target_is_rotate: bool,
        target_deg: float,
        pick_note: str,
        place_note: str,
        return_label: str,
    ) -> None:
        if level == 0:
            print(
                f"    {obj_name}: ({source_xy[0]:.3f},{source_xy[1]:.3f}) -> "
                f"({target_xy[0]:.3f},{target_xy[1]:.3f}) rotate={target_is_rotate}"
            )
        else:
            print(
                f"    {obj_name}: ({source_xy[0]:.3f},{source_xy[1]:.3f}) "
                f"stacking at ({target_xy[0]:.3f},{target_xy[1]:.3f})"
            )

        if runtime.dry_run:
            return

        pick_step = TaskStep(
            kind="pick",
            object_name=obj_name,
            xy=tuple(source_xy),
            level=0,
            is_rotate=source_is_rotate,
            deg=source_deg,
            align_j6=True,
            note=pick_note,
        )
        place_step = TaskStep(
            kind="place",
            object_name=obj_name,
            xy=tuple(target_xy),
            level=level,
            is_rotate=target_is_rotate,
            deg=target_deg,
            align_j6=True,
            note=place_note,
        )
        temp_state: dict[str, dict[str, Any]] = {
            obj_name: {
                "xy": [float(source_xy[0]), float(source_xy[1])],
                "is_rotate": bool(source_is_rotate),
                "deg": float(source_deg),
                "standard_j6_rad": None,
                "upper": None,
                "lower": None,
            }
        }
        _, held = execute_step_sequence(
            runtime,
            [pick_step, place_step],
            record=False,
            scene_state=temp_state,
            lookup_scene_state=temp_state,
            result_scene_state=temp_state,
        )
        if held is not None:
            raise RuntimeError(f"layout rearrange ended while holding {held}")
        runtime.return_home(return_label)

    layout_names = list(OBSTACLE_NAMES)
    np.random.shuffle(layout_names)
    n_in_band = int(np.random.randint(int(band_count_min), int(band_count_max) + 1))
    band_names = layout_names[:n_in_band]
    outside_names = layout_names[n_in_band:]

    stack_bottom: str | None = None
    stack_top: str | None = None
    if len(band_names) >= 2 and np.random.random() < float(stack_probability):
        stack_pair = list(np.random.choice(band_names, size=2, replace=False))
        stack_bottom = str(stack_pair[0])
        stack_top = str(stack_pair[1])

    for obj_name in band_names:
        if obj_name == stack_top:
            continue
        old_obj = result_scene.obstacles[obj_name]
        new_xy = sample_obstacle_xy(
            result_scene,
            object_name=obj_name,
            workspace_x_min=workspace_x_min,
            workspace_x_max=workspace_x_max,
            y_min=band_y_lo,
            y_max=band_y_hi,
            min_spacing=clear_spacing,
        )
        new_is_rotate, new_deg = sample_obstacle_orientation()
        _execute_rearrange_move(
            obj_name,
            tuple(old_obj.xy),
            bool(old_obj.is_rotate),
            float(old_obj.deg),
            tuple(new_xy),
            level=0,
            target_is_rotate=new_is_rotate,
            target_deg=new_deg,
            pick_note=f"layout pick {obj_name}",
            place_note=f"layout place {obj_name}",
            return_label=f"[layout {obj_name}] return home",
        )
        result_scene.place_on_table(obj_name, tuple(new_xy), is_rotate=new_is_rotate, deg=new_deg)

    outside_y_ranges: list[tuple[float, float]] = []
    if workspace_y_min < band_y_lo - clear_spacing:
        outside_y_ranges.append((workspace_y_min, band_y_lo - clear_spacing))
    if band_y_hi + clear_spacing < workspace_y_max:
        outside_y_ranges.append((band_y_hi + clear_spacing, workspace_y_max))
    if not outside_y_ranges:
        outside_y_ranges = [(workspace_y_min, workspace_y_max)]

    for obj_name in outside_names:
        chosen_range = outside_y_ranges[int(np.random.randint(len(outside_y_ranges)))]
        old_obj = result_scene.obstacles[obj_name]
        new_xy = sample_obstacle_xy(
            result_scene,
            object_name=obj_name,
            workspace_x_min=workspace_x_min,
            workspace_x_max=workspace_x_max,
            y_min=float(chosen_range[0]),
            y_max=float(chosen_range[1]),
            min_spacing=clear_spacing,
        )
        new_is_rotate, new_deg = sample_obstacle_orientation()
        _execute_rearrange_move(
            obj_name,
            tuple(old_obj.xy),
            bool(old_obj.is_rotate),
            float(old_obj.deg),
            tuple(new_xy),
            level=0,
            target_is_rotate=new_is_rotate,
            target_deg=new_deg,
            pick_note=f"layout pick {obj_name}",
            place_note=f"layout place {obj_name}",
            return_label=f"[layout {obj_name}] return home",
        )
        result_scene.place_on_table(obj_name, tuple(new_xy), is_rotate=new_is_rotate, deg=new_deg)

    if stack_top is not None and stack_bottom is not None:
        top_obj = result_scene.obstacles[stack_top]
        bottom_obj = result_scene.obstacles[stack_bottom]
        _execute_rearrange_move(
            stack_top,
            tuple(top_obj.xy),
            bool(top_obj.is_rotate),
            float(top_obj.deg),
            tuple(bottom_obj.xy),
            level=1,
            target_is_rotate=bool(bottom_obj.is_rotate),
            target_deg=float(bottom_obj.deg),
            pick_note=f"stack pick {stack_top}",
            place_note=f"stack place {stack_top} on {stack_bottom}",
            return_label=f"[stack {stack_top}] return home",
        )
        result_scene.place_on_object(stack_top, stack_bottom)

    return result_scene

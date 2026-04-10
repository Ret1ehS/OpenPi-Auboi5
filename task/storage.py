from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from task.pick_and_place import (
    APPLE_NAME,
    OBJECT_ORDER,
    PlannerConfig,
    TaskStep,
    execute_pick_step,
    execute_step_sequence,
    load_scene_state,
    sample_random_orientation,
)


FORBIDDEN_X_MIN_M = 0.40
FORBIDDEN_Y_MIN_M = 0.07
STORAGE_DROP_X_M = 0.56
STORAGE_DROP_Y_M = 0.21
STORAGE_DROP_Z_M = 0.270
STORAGE_TRAVEL_SPEED_SCALE = 3.0
STORAGE_LIFT_SPEED_SCALE = 2.0


@dataclass
class StorageSession:
    scene_state: dict[str, dict[str, Any]]
    next_index: int = 0
    episode_count: int = 0
    skip_prep: bool = False


@dataclass(frozen=True)
class StorageEpisodePlan:
    object_name: str
    prompt: str
    pick_step: TaskStep
    drop_xy: tuple[float, float]
    drop_z: float


@dataclass
class StorageRecordedEpisode:
    plan: StorageEpisodePlan
    frames: list[Any]
    scene_after: dict[str, dict[str, Any]]


def object_prompt_name(name: str) -> str:
    if name == APPLE_NAME:
        return "apple"
    return f"{name} cube"


def build_storage_prompt(name: str) -> str:
    return f"put the {object_prompt_name(name)} into storage basket"


def _normalize_xy_tuple(xy: Any) -> tuple[float, float]:
    arr = np.asarray(xy, dtype=np.float64).reshape(2)
    return float(arr[0]), float(arr[1])


def _workspace_axis_bounds(axis_min: float, axis_max: float, min_spacing_m: float) -> tuple[float, float]:
    edge_margin = max(0.0, float(min_spacing_m) * 0.5)
    half_span = max(0.0, (float(axis_max) - float(axis_min)) * 0.5 - 1e-6)
    effective_margin = min(edge_margin, half_span)
    return float(axis_min) + effective_margin, float(axis_max) - effective_margin


def _is_far_enough(candidate: np.ndarray, occupied: list[np.ndarray], min_spacing_m: float) -> bool:
    return all(float(np.linalg.norm(candidate - xy)) >= float(min_spacing_m) for xy in occupied)


def _in_forbidden_seed_region(candidate: np.ndarray) -> bool:
    return float(candidate[0]) > float(FORBIDDEN_X_MIN_M) and float(candidate[1]) > float(FORBIDDEN_Y_MIN_M)


def _sample_initial_storage_state(
    scene_state: dict[str, dict[str, Any]],
    origin_xy: np.ndarray,
    *,
    object_name: str,
    config: PlannerConfig,
) -> dict[str, Any]:
    occupied = [np.asarray(origin_xy, dtype=np.float64).reshape(2).copy()]
    for existing_name, state in scene_state.items():
        if existing_name not in OBJECT_ORDER:
            continue
        occupied.append(np.asarray(state["xy"], dtype=np.float64).reshape(2).copy())
    x_min, x_max = _workspace_axis_bounds(
        float(config.workspace_x_min),
        float(config.workspace_x_max),
        float(config.min_spacing_m),
    )
    y_min, y_max = _workspace_axis_bounds(
        float(config.workspace_y_min),
        float(config.workspace_y_max),
        float(config.min_spacing_m),
    )
    for _ in range(512):
        candidate = np.array(
            [
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
            ],
            dtype=np.float64,
        )
        if _in_forbidden_seed_region(candidate):
            continue
        if _is_far_enough(candidate, occupied, config.min_spacing_m):
            break
    else:
        raise RuntimeError("failed to sample initial storage state")

    if object_name == APPLE_NAME:
        is_rotate = False
        deg = 0.0
    else:
        is_rotate, deg = sample_random_orientation(object_name, config=config)
    return {
        "xy": [float(candidate[0]), float(candidate[1])],
        "is_rotate": bool(is_rotate),
        "deg": 0.0 if not is_rotate else float(deg),
        "upper": None,
        "lower": None,
    }


def restore_session(
    saved: dict[str, Any] | None,
    *,
    resume_mode: str,
) -> tuple[StorageSession, bool]:
    session = StorageSession(scene_state={})
    should_clear_saved_state = False
    raw_storage_state = saved.get("storage_state") if isinstance(saved, dict) else None

    if isinstance(raw_storage_state, dict):
        raw_scene_state = raw_storage_state.get("scene_state", {})
        normalized_state = load_scene_state(raw_scene_state) if isinstance(raw_scene_state, dict) else None
        if normalized_state is None:
            raise RuntimeError("saved storage scene state is invalid")

        next_index = int(raw_storage_state.get("next_index", 0))
        episode_count = int(raw_storage_state.get("episode_count", 0))
        held_object = raw_storage_state.get("held_object")
        print("\n  Found saved storage state:")
        print(f"    next_index={next_index}, episode_count={episode_count}, held_object={held_object}")
        for name in OBJECT_ORDER:
            state = normalized_state[name]
            xy = _normalize_xy_tuple(state["xy"])
            print(
                f"    {name}: xy=({xy[0]:.4f}, {xy[1]:.4f}), "
                f"is_rotate={bool(state['is_rotate'])}, deg={float(state['deg']):.1f}"
            )
        if resume_mode == "reset":
            should_clear_saved_state = True
        else:
            session.scene_state = normalized_state
            session.next_index = max(0, min(next_index, len(OBJECT_ORDER)))
            session.episode_count = max(0, episode_count)
            session.skip_prep = True
            print("  Resuming storage task from saved state.")
    elif resume_mode == "continue":
        print("\n  Resume selected, but no saved storage state exists. Starting fresh.")

    return session, should_clear_saved_state


def prepare_session(
    runtime,
    session: StorageSession,
    *,
    config: PlannerConfig,
) -> None:
    if session.skip_prep:
        print("\n=== Skipping Storage Preparation (resumed from saved state) ===")
        return

    print("\n=== Storage Preparation Phase ===")
    print(f"Seed order: {list(OBJECT_ORDER)}")
    for object_name in OBJECT_ORDER:
        placed_state = _sample_initial_storage_state(
            session.scene_state,
            runtime.origin_xy,
            object_name=object_name,
            config=config,
        )
        print(
            f"[prep] {object_name}: origin -> ({placed_state['xy'][0]:.4f}, {placed_state['xy'][1]:.4f}), "
            f"is_rotate={placed_state['is_rotate']}, deg={placed_state['deg']:.1f}"
        )
        if not runtime.dry_run:
            pick_step = TaskStep(
                kind="pick",
                object_name=object_name,
                xy=(float(runtime.origin_xy[0]), float(runtime.origin_xy[1])),
                level=0,
                is_rotate=False,
                deg=0.0,
                align_yaw=bool(object_name != APPLE_NAME),
                note=f"storage prep pick {object_name}",
            )
            place_step = TaskStep(
                kind="place",
                object_name=object_name,
                xy=(float(placed_state["xy"][0]), float(placed_state["xy"][1])),
                level=0,
                is_rotate=bool(placed_state["is_rotate"]),
                deg=float(placed_state["deg"]),
                align_yaw=bool(object_name != APPLE_NAME),
                note=f"storage prep place {object_name}",
            )
            _, held_after_prep = execute_step_sequence(
                runtime,
                [pick_step, place_step],
                record=False,
                scene_state=session.scene_state,
                lookup_scene_state=session.scene_state,
                result_scene_state={object_name: placed_state},
            )
            if held_after_prep is not None:
                raise RuntimeError(f"storage prep ended while still holding {held_after_prep}")
            runtime.return_home(f"[storage prep {object_name}] return home")
        session.scene_state[object_name] = placed_state

    session.next_index = 0
    session.skip_prep = False


def has_remaining_objects(session: StorageSession) -> bool:
    return int(session.next_index) < len(OBJECT_ORDER)


def describe_episode(plan: StorageEpisodePlan, *, episode_count: int, remaining_count: int) -> None:
    print(f"\n--- Episode {episode_count} [storage] ---")
    print(f"  Prompt: \"{plan.prompt}\"")
    print(f"  Source: {plan.object_name}")
    print(f"  Recorded steps: 1 pick + basket transfer")
    print(f"  Remaining after save: {remaining_count}")


def plan_next_episode(session: StorageSession) -> StorageEpisodePlan:
    if not has_remaining_objects(session):
        raise RuntimeError("storage task is already complete")
    if not session.scene_state:
        raise RuntimeError("storage scene_state is empty; cannot plan next episode")
    object_name = OBJECT_ORDER[int(session.next_index)]
    source_state = session.scene_state.get(object_name)
    if not isinstance(source_state, dict):
        raise RuntimeError(f"missing storage scene state for {object_name}")
    is_rotate = bool(source_state.get("is_rotate", False))
    pick_step = TaskStep(
        kind="pick",
        object_name=object_name,
        xy=_normalize_xy_tuple(source_state["xy"]),
        level=0,
        is_rotate=is_rotate,
        deg=0.0 if not is_rotate else float(source_state.get("deg", 0.0)),
        align_yaw=bool(object_name != APPLE_NAME),
        note=f"pick {object_name} for storage",
    )
    return StorageEpisodePlan(
        object_name=object_name,
        prompt=build_storage_prompt(object_name),
        pick_step=pick_step,
        drop_xy=(float(STORAGE_DROP_X_M), float(STORAGE_DROP_Y_M)),
        drop_z=float(STORAGE_DROP_Z_M),
    )


def _build_scene_after_drop(scene_state: dict[str, dict[str, Any]], object_name: str) -> dict[str, dict[str, Any]]:
    normalized = load_scene_state(scene_state)
    if normalized is None:
        raise RuntimeError("invalid storage scene state payload")
    obj = normalized[object_name]
    obj["xy"] = [float(STORAGE_DROP_X_M), float(STORAGE_DROP_Y_M)]
    obj["is_rotate"] = False
    obj["deg"] = 0.0
    obj["upper"] = None
    obj["lower"] = None
    return normalized


def record_episode(
    runtime,
    session: StorageSession,
    plan: StorageEpisodePlan,
) -> StorageRecordedEpisode:
    if not session.scene_state:
        raise RuntimeError("storage scene_state is empty; cannot record episode")

    execution_scene_state = load_scene_state(session.scene_state)
    if execution_scene_state is None:
        raise RuntimeError("invalid storage scene_state payload")

    frames: list[Any] = []
    frame_idx = 0
    basket_yaw = float(runtime.yaw_target_from_deg(0.0))
    lift_z = float(runtime.min_tcp_z + runtime.approach_z_offset_m)
    basket_above_pose = runtime.build_pose_from_live_orientation_yaw(
        float(plan.drop_xy[0]),
        float(plan.drop_xy[1]),
        float(lift_z),
        float(basket_yaw),
    )
    basket_down_pose = runtime.build_pose_from_live_orientation_yaw(
        float(plan.drop_xy[0]),
        float(plan.drop_xy[1]),
        float(plan.drop_z),
        float(basket_yaw),
    )

    if runtime.dry_run:
        from support.pose_align import real_pose_to_sim

        basket_above_sim = real_pose_to_sim(basket_above_pose)
        basket_down_sim = real_pose_to_sim(basket_down_pose)
        pick_frames, frame_idx = execute_pick_step(
            runtime,
            plan.pick_step,
            record=True,
            frame_idx=frame_idx,
            lookup_scene_state=execution_scene_state,
            lift_speed_mps=float(runtime.linear_speed) * float(STORAGE_LIFT_SPEED_SCALE),
        )
        frames.extend(pick_frames)
        for _ in range(24):
            frames.append(
                runtime.make_dummy_frame(
                    sim_pose=basket_above_sim,
                    gripper=0.0,
                    yaw=float(basket_yaw),
                    frame_idx=frame_idx,
                )
            )
            frame_idx += 1
        for _ in range(10):
            frames.append(
                runtime.make_dummy_frame(
                    sim_pose=basket_down_sim,
                    gripper=0.0,
                    yaw=float(basket_yaw),
                    frame_idx=frame_idx,
                )
            )
            frame_idx += 1
        for _ in range(10):
            frames.append(
                runtime.make_dummy_frame(
                    sim_pose=basket_above_sim,
                    gripper=1.0,
                    yaw=float(basket_yaw),
                    frame_idx=frame_idx,
                )
            )
            frame_idx += 1
        return StorageRecordedEpisode(
            plan=plan,
            frames=frames,
            scene_after=_build_scene_after_drop(session.scene_state, plan.object_name),
        )

    runtime.begin_task_servo()
    try:
        pick_frames, frame_idx = execute_pick_step(
            runtime,
            plan.pick_step,
            record=True,
            frame_idx=frame_idx,
            lookup_scene_state=execution_scene_state,
            lift_speed_mps=float(runtime.linear_speed) * float(STORAGE_LIFT_SPEED_SCALE),
        )
        frames.extend(pick_frames)
        runtime.runtime_held_object = plan.object_name

        seg = runtime.record_pose_move(
            basket_above_pose,
            gripper=0.0,
            start_frame_idx=frame_idx,
            record=True,
            target_yaw=float(basket_yaw),
            speed_mps=float(runtime.linear_speed) * float(STORAGE_TRAVEL_SPEED_SCALE),
        )
        frames.extend(seg)
        frame_idx += len(seg)

        seg = runtime.record_pose_move(
            basket_down_pose,
            gripper=0.0,
            start_frame_idx=frame_idx,
            record=True,
            target_yaw=float(basket_yaw),
        )
        frames.extend(seg)
        frame_idx += len(seg)

        runtime.ensure_gripper_ok(
            runtime.command_gripper_state(1),
            f"open gripper for storage drop {plan.object_name}",
        )
        runtime.runtime_held_object = None

        scene_after = _build_scene_after_drop(session.scene_state, plan.object_name)
        runtime.cleanup_scene_state = scene_after

        seg = runtime.record_pose_move(
            basket_above_pose,
            gripper=1.0,
            start_frame_idx=frame_idx,
            record=True,
            target_yaw=float(basket_yaw),
            speed_mps=float(runtime.linear_speed) * float(STORAGE_LIFT_SPEED_SCALE),
        )
        frames.extend(seg)
    finally:
        runtime.end_task_servo()

    if not frames:
        raise RuntimeError("storage episode produced no frames")

    return StorageRecordedEpisode(
        plan=plan,
        frames=frames,
        scene_after=_build_scene_after_drop(session.scene_state, plan.object_name),
    )


def finalize_episode(
    runtime,
    session: StorageSession,
    recorded: StorageRecordedEpisode,
) -> None:
    session.scene_state = recorded.scene_after
    session.next_index = min(int(session.next_index) + 1, len(OBJECT_ORDER))
    session.episode_count += 1
    runtime.cleanup_scene_state = None


__all__ = [
    "StorageEpisodePlan",
    "StorageRecordedEpisode",
    "StorageSession",
    "build_storage_prompt",
    "describe_episode",
    "finalize_episode",
    "has_remaining_objects",
    "plan_next_episode",
    "prepare_session",
    "record_episode",
    "restore_session",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from support.pose_align import real_pose_to_sim


APPLE_NAME = "apple"
OBJECT_ORDER = ("red", "green", "blue", APPLE_NAME)
PICK_TASK_PROBABILITY = 0.20
NON_ROTATED_TABLE_PLACE_PROBABILITY = 0.0
DEFAULT_ROTATE_DEG_MIN = 12.0
DEFAULT_ROTATE_DEG_MAX = 22.5
DEFAULT_YAW_HOME_RAD = 0.11434
LOCAL_CLEAR_INITIAL_RADIUS_M = 0.15
LOCAL_CLEAR_RADIUS_EXPAND_M = 0.0
LOCAL_CLEAR_POINTS_PER_RETRY = 8
LOCAL_CLEAR_MAX_RETRIES = 1


@dataclass(frozen=True)
class PlannerConfig:
    workspace_x_min: float
    workspace_x_max: float
    workspace_y_min: float
    workspace_y_max: float
    min_spacing_m: float
    object_height_m: float
    pick_task_probability: float = PICK_TASK_PROBABILITY
    non_rotated_table_place_probability: float = NON_ROTATED_TABLE_PLACE_PROBABILITY
    rotate_deg_min: float = DEFAULT_ROTATE_DEG_MIN
    rotate_deg_max: float = DEFAULT_ROTATE_DEG_MAX
    local_clear_initial_radius_m: float = LOCAL_CLEAR_INITIAL_RADIUS_M
    local_clear_radius_expand_m: float = LOCAL_CLEAR_RADIUS_EXPAND_M
    local_clear_points_per_retry: int = LOCAL_CLEAR_POINTS_PER_RETRY
    local_clear_max_retries: int = LOCAL_CLEAR_MAX_RETRIES


@dataclass
class ObjectState:
    name: str
    xy: tuple[float, float]
    is_rotate: bool
    deg: float
    upper: str | None = None
    lower: str | None = None


@dataclass
class TaskStep:
    kind: str  # "pick" | "place"
    object_name: str
    xy: tuple[float, float]
    level: int
    is_rotate: bool
    deg: float
    note: str
    support_name: str | None = None
    align_yaw: bool = False


@dataclass
class EpisodePlan:
    task_kind: str  # "pick" | "place"
    prompt: str
    source_name: str
    target_name: str | None
    recorded_steps: list[TaskStep]
    post_steps: list[TaskStep]
    scene_after: dict[str, dict[str, Any]]


@dataclass
class PickAndPlaceSession:
    scene_state: dict[str, dict[str, Any]]
    task_index: int = 0
    episode_count: int = 0
    skip_prep: bool = False


@dataclass
class PickAndPlaceRecordedEpisode:
    plan: EpisodePlan
    frames: list[Any]
    held_after_recorded: str | None
    execution_scene_state: dict[str, dict[str, Any]]


def object_prompt_name(name: str) -> str:
    if name == APPLE_NAME:
        return "apple"
    return f"{name} cube"


def build_pick_prompt(name: str) -> str:
    return f"pick up the {object_prompt_name(name)}"


def build_place_prompt(source: str, target: str) -> str:
    return f"put the {object_prompt_name(source)} on the {object_prompt_name(target)}"


class SceneState:
    def __init__(self, objects: dict[str, ObjectState]) -> None:
        self.objects = objects

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "SceneState | None":
        if not isinstance(payload, dict):
            return None

        objects: dict[str, ObjectState] = {}
        for name in OBJECT_ORDER:
            raw = payload.get(name)
            if raw is None or not isinstance(raw, dict):
                return None
            xy = raw.get("xy")
            if xy is None:
                return None
            is_rotate = bool(raw.get("is_rotate", False))
            deg = float(raw.get("deg", 0.0))
            if name == APPLE_NAME:
                is_rotate = False
                deg = 0.0
            objects[name] = ObjectState(
                name=name,
                xy=_normalize_xy_tuple(xy),
                is_rotate=is_rotate,
                deg=deg,
                upper=_normalize_link(raw.get("upper")),
                lower=_normalize_link(raw.get("lower")),
            )
        return cls(objects)

    @classmethod
    def from_legacy_collect_state(cls, payload: dict[str, Any]) -> "SceneState | None":
        cube_states = payload.get("cube_states")
        if not isinstance(cube_states, dict):
            return None

        objects: dict[str, ObjectState] = {}
        for name in OBJECT_ORDER:
            raw = cube_states.get(name)
            if not isinstance(raw, dict):
                return None
            xy = raw.get("xy")
            if xy is None:
                return None
            if "j6" in raw:
                saved_yaw = _normalize_optional_yaw(raw.get("j6"))
                deg = float(np.rad2deg(float(saved_yaw or 0.0) - DEFAULT_YAW_HOME_RAD))
                is_rotate = abs(deg) > 1e-6 and name != APPLE_NAME
            else:
                deg = float(raw.get("deg", 0.0))
                is_rotate = bool(raw.get("is_rotate", False))
            if name == APPLE_NAME:
                deg = 0.0
                is_rotate = False
            objects[name] = ObjectState(
                name=name,
                xy=_normalize_xy_tuple(xy),
                is_rotate=is_rotate,
                deg=deg,
                upper=_normalize_link(raw.get("upper")),
                lower=_normalize_link(raw.get("lower")),
            )
        return cls(objects)

    def copy(self) -> "SceneState":
        return SceneState(
            {
                name: ObjectState(
                    name=obj.name,
                    xy=tuple(obj.xy),
                    is_rotate=bool(obj.is_rotate),
                    deg=float(obj.deg),
                    upper=obj.upper,
                    lower=obj.lower,
                )
                for name, obj in self.objects.items()
            }
        )

    def to_serializable(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "xy": [float(obj.xy[0]), float(obj.xy[1])],
                "is_rotate": bool(obj.is_rotate),
                "deg": 0.0 if not obj.is_rotate else float(obj.deg),
                "upper": obj.upper,
                "lower": obj.lower,
            }
            for name, obj in self.objects.items()
        }

    def get(self, name: str) -> ObjectState:
        return self.objects[name]

    def top_of(self, name: str) -> str:
        current = name
        seen: set[str] = set()
        while self.objects[current].upper is not None:
            if current in seen:
                raise RuntimeError(f"cycle detected while following upper chain from {name}")
            seen.add(current)
            current = self.objects[current].upper  # type: ignore[assignment]
        return current

    def lower_chain(self, name: str) -> list[str]:
        chain: list[str] = []
        current = self.objects[name].lower
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise RuntimeError(f"cycle detected while following lower chain from {name}")
            seen.add(current)
            chain.append(current)
            current = self.objects[current].lower
        return chain

    def depth(self, name: str) -> int:
        return len(self.lower_chain(name))

    def occupied_positions(self, *, exclude_names: set[str] | None = None) -> list[np.ndarray]:
        exclude = exclude_names or set()
        out: list[np.ndarray] = []
        for name, obj in self.objects.items():
            if name in exclude:
                continue
            out.append(np.asarray(obj.xy, dtype=np.float64))
        return out

    def detach_top(self, name: str) -> None:
        obj = self.objects[name]
        if obj.upper is not None:
            raise RuntimeError(f"cannot detach {name}: upper={obj.upper} still present")
        if obj.lower is not None:
            self.objects[obj.lower].upper = None
        obj.lower = None

    def place_on_table(self, name: str, xy: tuple[float, float], *, is_rotate: bool, deg: float) -> None:
        obj = self.objects[name]
        if obj.upper is not None:
            raise RuntimeError(f"cannot place {name} on table while upper={obj.upper} exists")
        obj.xy = _normalize_xy_tuple(xy)
        if name == APPLE_NAME:
            obj.is_rotate = False
            obj.deg = 0.0
        else:
            obj.is_rotate = bool(is_rotate)
            obj.deg = 0.0 if not is_rotate else float(deg)
        obj.lower = None

    def place_on_target(self, name: str, target_name: str, *, is_rotate: bool, deg: float) -> str:
        actual_target = self.top_of(target_name)
        if actual_target == APPLE_NAME:
            raise RuntimeError("apple cannot receive an upper object")
        obj = self.objects[name]
        if obj.upper is not None:
            raise RuntimeError(f"cannot place {name}: upper={obj.upper} still present")
        obj.xy = tuple(self.objects[actual_target].xy)
        if name == APPLE_NAME:
            obj.is_rotate = False
            obj.deg = 0.0
        else:
            obj.is_rotate = bool(is_rotate)
            obj.deg = 0.0 if not is_rotate else float(deg)
        obj.lower = actual_target
        self.objects[actual_target].upper = name
        return actual_target


def load_scene_state(payload: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    scene = SceneState.from_serialized(payload)
    if scene is not None:
        return scene.to_serializable()
    legacy = SceneState.from_legacy_collect_state(payload)
    if legacy is not None:
        return legacy.to_serializable()
    return None


def sample_random_table_state(
    scene_payload: dict[str, dict[str, Any]],
    object_name: str,
    *,
    config: PlannerConfig,
) -> tuple[np.ndarray, bool, float]:
    scene = SceneState.from_serialized(scene_payload)
    if scene is None:
        raise RuntimeError("invalid scene payload for sample_random_table_state")
    xy = sample_global_xy(scene, config=config, exclude_names={object_name})
    is_rotate, deg = sample_random_orientation(object_name, config=config)
    return xy, is_rotate, deg


def build_random_episode_plan(
    scene_payload: dict[str, dict[str, Any]],
    *,
    config: PlannerConfig,
) -> EpisodePlan:
    scene = SceneState.from_serialized(scene_payload)
    if scene is None:
        raise RuntimeError("invalid scene payload")

    legal_pick_sources = list(OBJECT_ORDER)
    legal_place_pairs = [
        (source, target)
        for source in OBJECT_ORDER
        for target in OBJECT_ORDER
        if _is_valid_place_pair(scene, source, target)
    ]

    do_pick = not legal_place_pairs or np.random.random() < float(config.pick_task_probability)
    if do_pick:
        source_name = str(np.random.choice(legal_pick_sources))
        return _build_pick_plan(scene, source_name, config=config)

    pair_index = int(np.random.randint(0, len(legal_place_pairs)))
    source_name, target_name = legal_place_pairs[pair_index]
    return _build_place_plan(scene, source_name, target_name, config=config)


def _build_pick_plan(scene: SceneState, source_name: str, *, config: PlannerConfig) -> EpisodePlan:
    recorded_steps: list[TaskStep] = []
    _plan_clear_above(scene, source_name, recorded_steps, config=config)

    recorded_steps.append(_make_pick_step(scene, source_name, note=f"pick {source_name}"))
    scene.detach_top(source_name)

    post_xy = sample_global_xy(scene, config=config, exclude_names={source_name})
    post_is_rotate, post_deg = sample_random_orientation(source_name, config=config)
    post_steps = [
        _make_place_step(
            scene,
            source_name,
            target_name=None,
            xy=tuple(float(v) for v in post_xy.tolist()),
            level=0,
            is_rotate=post_is_rotate,
            deg=post_deg,
            align_yaw=_should_align_yaw(source_name, is_rotate=post_is_rotate),
            note=f"post-place {source_name}",
        )
    ]
    scene.place_on_table(source_name, tuple(float(v) for v in post_xy.tolist()), is_rotate=post_is_rotate, deg=post_deg)

    return EpisodePlan(
        task_kind="pick",
        prompt=build_pick_prompt(source_name),
        source_name=source_name,
        target_name=None,
        recorded_steps=recorded_steps,
        post_steps=post_steps,
        scene_after=scene.to_serializable(),
    )


def _build_place_plan(scene: SceneState, source_name: str, target_name: str, *, config: PlannerConfig) -> EpisodePlan:
    recorded_steps: list[TaskStep] = []
    _plan_clear_above(scene, source_name, recorded_steps, config=config)

    recorded_steps.append(_make_pick_step(scene, source_name, note=f"pick {source_name} for place"))
    scene.detach_top(source_name)

    actual_support = scene.top_of(target_name)
    place_level = scene.depth(actual_support) + 1
    place_is_rotate, place_deg = _resolve_place_orientation(scene, source_name, actual_support)
    recorded_steps.append(
        _make_place_step(
            scene,
            source_name,
            target_name=target_name,
            xy=tuple(scene.get(actual_support).xy),
            level=place_level,
            is_rotate=place_is_rotate,
            deg=place_deg,
            align_yaw=_should_align_yaw(source_name, is_rotate=place_is_rotate),
            note=f"place {source_name} on {target_name}",
            support_name=actual_support,
        )
    )
    scene.place_on_target(source_name, target_name, is_rotate=place_is_rotate, deg=place_deg)

    return EpisodePlan(
        task_kind="place",
        prompt=build_place_prompt(source_name, target_name),
        source_name=source_name,
        target_name=target_name,
        recorded_steps=recorded_steps,
        post_steps=[],
        scene_after=scene.to_serializable(),
    )


def _plan_clear_above(
    scene: SceneState,
    source_name: str,
    recorded_steps: list[TaskStep],
    *,
    config: PlannerConfig,
) -> None:
    upper_name = scene.get(source_name).upper
    if upper_name is None:
        return

    _plan_clear_above(scene, upper_name, recorded_steps, config=config)

    upper_state = scene.get(upper_name)
    temp_xy = sample_local_clear_xy(
        scene,
        center_xy=np.asarray(upper_state.xy, dtype=np.float64),
        config=config,
        exclude_names={upper_name},
    )
    recorded_steps.append(_make_pick_step(scene, upper_name, note=f"clear upper {upper_name}"))
    scene.detach_top(upper_name)
    recorded_steps.append(
        _make_place_step(
            scene,
            upper_name,
            target_name=None,
            xy=tuple(float(v) for v in temp_xy.tolist()),
            level=0,
            is_rotate=False,
            deg=0.0,
            align_yaw=bool(upper_name != APPLE_NAME),
            note=f"place cleared {upper_name} to table",
        )
    )
    scene.place_on_table(upper_name, tuple(float(v) for v in temp_xy.tolist()), is_rotate=False, deg=0.0)


def _make_pick_step(scene: SceneState, object_name: str, *, note: str) -> TaskStep:
    obj = scene.get(object_name)
    should_align_yaw = _should_align_yaw(object_name, is_rotate=obj.is_rotate)
    return TaskStep(
        kind="pick",
        object_name=object_name,
        xy=tuple(obj.xy),
        level=scene.depth(object_name),
        is_rotate=bool(obj.is_rotate),
        deg=0.0 if not obj.is_rotate else float(obj.deg),
        align_yaw=should_align_yaw,
        note=note,
    )


def _make_place_step(
    scene: SceneState,
    object_name: str,
    *,
    target_name: str | None,
    xy: tuple[float, float],
    level: int,
    is_rotate: bool,
    deg: float,
    align_yaw: bool,
    note: str,
    support_name: str | None = None,
) -> TaskStep:
    _ = scene
    return TaskStep(
        kind="place",
        object_name=object_name,
        xy=tuple(float(v) for v in xy),
        level=int(level),
        is_rotate=bool(is_rotate),
        deg=0.0 if not is_rotate else float(deg),
        align_yaw=bool(align_yaw),
        note=note,
        support_name=support_name if target_name is not None else None,
    )


def _should_align_yaw(object_name: str, *, is_rotate: bool) -> bool:
    _ = is_rotate
    # Non-apple objects always align tool yaw to step.deg; deg=0 means return to the default baseline angle.
    return bool(object_name != APPLE_NAME)


def _resolve_place_orientation(scene: SceneState, source_name: str, support_name: str) -> tuple[bool, float]:
    if source_name == APPLE_NAME:
        return False, 0.0
    support = scene.get(support_name)
    if not support.is_rotate:
        return False, 0.0
    return True, float(support.deg)


def _is_valid_place_pair(scene: SceneState, source_name: str, target_name: str) -> bool:
    if source_name == target_name:
        return False
    if target_name == APPLE_NAME:
        return False
    if scene.top_of(target_name) == APPLE_NAME:
        return False
    if target_name in scene.lower_chain(source_name):
        return False
    return True


def sample_random_orientation(object_name: str, *, config: PlannerConfig) -> tuple[bool, float]:
    if object_name == APPLE_NAME:
        return False, 0.0
    if np.random.random() < float(config.non_rotated_table_place_probability):
        return False, 0.0
    abs_deg = float(np.random.uniform(config.rotate_deg_min, config.rotate_deg_max))
    sign = -1.0 if np.random.random() < 0.5 else 1.0
    return True, sign * abs_deg


def sample_global_xy(
    scene: SceneState,
    *,
    config: PlannerConfig,
    exclude_names: set[str] | None = None,
) -> np.ndarray:
    occupied = scene.occupied_positions(exclude_names=exclude_names)
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
        if _is_far_enough(candidate, occupied, config.min_spacing_m):
            return candidate
    raise RuntimeError("failed to sample a global free xy")


def sample_local_clear_xy(
    scene: SceneState,
    *,
    center_xy: np.ndarray,
    config: PlannerConfig,
    exclude_names: set[str] | None = None,
) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    occupied = scene.occupied_positions(exclude_names=exclude_names)
    radius = float(config.local_clear_initial_radius_m)

    for _ in range(int(config.local_clear_max_retries)):
        for _point in range(int(config.local_clear_points_per_retry)):
            angle = float(np.random.uniform(-np.pi, np.pi))
            distance = float(np.random.uniform(0.0, radius))
            candidate = center + np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * distance
            if not _inside_workspace(candidate, config):
                continue
            if _is_far_enough(candidate, occupied, config.min_spacing_m):
                return candidate
        radius += float(config.local_clear_radius_expand_m)

    return sample_global_xy(scene, config=config, exclude_names=exclude_names)


def _is_far_enough(candidate: np.ndarray, occupied: list[np.ndarray], min_spacing_m: float) -> bool:
    return all(float(np.linalg.norm(candidate - xy)) >= float(min_spacing_m) for xy in occupied)


def _inside_workspace(candidate: np.ndarray, config: PlannerConfig) -> bool:
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
    return (
        x_min <= float(candidate[0]) <= x_max
        and y_min <= float(candidate[1]) <= y_max
    )


def _workspace_axis_bounds(axis_min: float, axis_max: float, min_spacing_m: float) -> tuple[float, float]:
    edge_margin = max(0.0, float(min_spacing_m) * 0.5)
    half_span = max(0.0, (float(axis_max) - float(axis_min)) * 0.5 - 1e-6)
    effective_margin = min(edge_margin, half_span)
    return float(axis_min) + effective_margin, float(axis_max) - effective_margin


def _normalize_xy_tuple(xy: Any) -> tuple[float, float]:
    arr = np.asarray(xy, dtype=np.float64).reshape(2)
    return float(arr[0]), float(arr[1])


def _normalize_link(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_optional_yaw(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clone_scene_state(scene_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cloned = load_scene_state(scene_state)
    if cloned is None:
        raise RuntimeError("invalid scene_state payload")
    return cloned


def get_object_xy(scene_state: dict[str, dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray(scene_state[name]["xy"], dtype=np.float64).reshape(2).copy()


def get_object_is_rotate(scene_state: dict[str, dict[str, Any]], name: str) -> bool:
    return bool(scene_state[name]["is_rotate"])


def get_object_deg(scene_state: dict[str, dict[str, Any]], name: str) -> float:
    return float(scene_state[name]["deg"])


def sample_initial_object_state(
    scene_state: dict[str, dict[str, Any]],
    origin_xy: np.ndarray,
    *,
    object_name: str,
    config: PlannerConfig,
) -> dict[str, Any]:
    occupied = [np.asarray(origin_xy, dtype=np.float64).reshape(2).copy()]
    for existing_name in scene_state.keys():
        occupied.append(get_object_xy(scene_state, existing_name))
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
        if _is_far_enough(candidate, occupied, config.min_spacing_m):
            break
    else:
        raise RuntimeError("failed to sample initial object state")

    if object_name == APPLE_NAME:
        is_rotate = False
        deg = 0.0
    else:
        is_rotate, deg = sample_random_orientation(object_name, config=config)
    return {
        "xy": [float(candidate[0]), float(candidate[1])],
        "is_rotate": bool(is_rotate),
        "deg": float(deg),
        "upper": None,
        "lower": None,
    }


def restore_session(
    saved: dict[str, Any] | None,
    *,
    resume_mode: str,
) -> tuple[PickAndPlaceSession, bool]:
    session = PickAndPlaceSession(scene_state={})
    should_clear_saved_state = False
    saved_scene_state = saved.get("scene_state", {}) if isinstance(saved, dict) else {}

    if saved_scene_state:
        print("\n  Found saved object states from previous run:")
        for name, state in saved_scene_state.items():
            xy = get_object_xy(saved_scene_state, name)
            is_rotate = get_object_is_rotate(saved_scene_state, name)
            deg = get_object_deg(saved_scene_state, name)
            upper = state.get("upper")
            lower = state.get("lower")
            print(
                f"    {name}: xy=({xy[0]:.4f}, {xy[1]:.4f}), "
                f"is_rotate={is_rotate}, deg={deg:.1f}, upper={upper}, lower={lower}"
            )
        print(
            f"    task_index={int(saved.get('color_index', 0))}, "
            f"episode_count={int(saved.get('episode_count', 0))}, "
            f"held_object={saved.get('held_object')}"
        )
        if resume_mode == "reset":
            should_clear_saved_state = True
        else:
            normalized_state = load_scene_state(saved_scene_state)
            if normalized_state is None:
                raise RuntimeError("saved pick/place scene state is invalid")
            session.scene_state = normalized_state
            session.task_index = int(saved.get("color_index", 0))
            session.episode_count = int(saved.get("episode_count", 0))
            session.skip_prep = True
            print("  Resuming from saved state.")
    elif resume_mode == "continue":
        print("\n  Resume selected, but no saved state exists. Starting fresh.")

    return session, should_clear_saved_state


def prepare_session(
    runtime,
    session: PickAndPlaceSession,
    *,
    config: PlannerConfig,
) -> None:
    if session.skip_prep:
        print("\n=== Skipping Preparation (resumed from saved state) ===")
        return

    print("\n=== Preparation Phase ===")
    print(f"Seed order: {list(OBJECT_ORDER)}")
    for object_name in OBJECT_ORDER:
        placed_state = sample_initial_object_state(
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
                note=f"prep pick {object_name}",
            )
            place_step = TaskStep(
                kind="place",
                object_name=object_name,
                xy=(float(placed_state["xy"][0]), float(placed_state["xy"][1])),
                level=0,
                is_rotate=bool(placed_state["is_rotate"]),
                deg=float(placed_state["deg"]),
                align_yaw=bool(object_name != APPLE_NAME),
                note=f"prep place {object_name}",
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
                raise RuntimeError(f"prep sequence ended while still holding {held_after_prep}")
            runtime.return_home(f"[prep {object_name}] return home")
        session.scene_state[object_name] = placed_state

    session.skip_prep = False


def describe_episode(plan: EpisodePlan, *, episode_count: int) -> None:
    episode_label = f"Episode {episode_count} [{plan.task_kind}]"
    print(f"\n--- {episode_label} ---")
    print(f"  Prompt: \"{plan.prompt}\"")
    print(f"  Source: {plan.source_name}")
    if plan.target_name is not None:
        print(f"  Target: {plan.target_name}")
    print(f"  Recorded steps: {len(plan.recorded_steps)}")
    print(f"  Post steps: {len(plan.post_steps)}")


def plan_next_episode(
    session: PickAndPlaceSession,
    *,
    config: PlannerConfig,
) -> EpisodePlan:
    if not session.scene_state:
        raise RuntimeError("scene_state is empty; cannot plan next episode")
    return build_random_episode_plan(session.scene_state, config=config)


def record_episode(
    runtime,
    session: PickAndPlaceSession,
    plan: EpisodePlan,
) -> PickAndPlaceRecordedEpisode:
    execution_scene_state = clone_scene_state(session.scene_state)
    frames, held_after_recorded = execute_step_sequence(
        runtime,
        plan.recorded_steps,
        record=True,
        scene_state=session.scene_state,
        result_scene_state=execution_scene_state,
    )
    if not frames:
        raise RuntimeError("recorded episode produced no frames")
    return PickAndPlaceRecordedEpisode(
        plan=plan,
        frames=frames,
        held_after_recorded=held_after_recorded,
        execution_scene_state=execution_scene_state,
    )


def finalize_episode(
    runtime,
    session: PickAndPlaceSession,
    recorded: PickAndPlaceRecordedEpisode,
) -> None:
    if recorded.plan.post_steps:
        _, held_after_post = execute_step_sequence(
            runtime,
            recorded.plan.post_steps,
            record=False,
            initial_held_object=recorded.held_after_recorded,
            scene_state=recorded.execution_scene_state,
            result_scene_state=recorded.execution_scene_state,
        )
    else:
        held_after_post = recorded.held_after_recorded

    if held_after_post is not None:
        raise RuntimeError(f"episode ended while still holding {held_after_post}")

    session.scene_state = clone_scene_state(recorded.execution_scene_state)
    session.task_index += 1
    session.episode_count += 1
    runtime.cleanup_scene_state = None


def scene_top_of(state: dict[str, dict[str, Any]], name: str) -> str:
    current = name
    seen: set[str] = set()
    while True:
        upper = state[current].get("upper")
        if upper is None:
            return current
        if current in seen:
            raise RuntimeError(f"cycle detected while following upper chain from {name}")
        seen.add(current)
        current = str(upper)


def scene_detach_top(state: dict[str, dict[str, Any]], name: str) -> None:
    obj = state[name]
    upper = obj.get("upper")
    if upper is not None:
        raise RuntimeError(f"cannot detach {name}: upper={upper} still present")
    lower = obj.get("lower")
    if lower is not None:
        state[str(lower)]["upper"] = None
    obj["lower"] = None


def scene_place_object(state: dict[str, dict[str, Any]], step: TaskStep) -> None:
    obj = state[step.object_name]
    if obj.get("upper") is not None:
        raise RuntimeError(f"cannot place {step.object_name}: upper={obj.get('upper')} still present")
    if step.support_name is None:
        obj["xy"] = [float(step.xy[0]), float(step.xy[1])]
        obj["lower"] = None
    else:
        actual_support = scene_top_of(state, step.support_name)
        if actual_support == APPLE_NAME:
            raise RuntimeError("apple cannot receive an upper object")
        obj["xy"] = list(state[actual_support]["xy"])
        obj["lower"] = actual_support
        state[actual_support]["upper"] = step.object_name
    if step.object_name == APPLE_NAME:
        obj["is_rotate"] = False
        obj["deg"] = 0.0
    else:
        obj["is_rotate"] = bool(step.is_rotate)
        obj["deg"] = 0.0 if not step.is_rotate else float(step.deg)


def commit_place_state(state: dict[str, dict[str, Any]], step: TaskStep) -> dict[str, dict[str, Any]] | None:
    scene_place_object(state, step)
    normalized_state = load_scene_state(state)
    return normalized_state


def resolve_step_target_yaw_rad(runtime, step: TaskStep, lookup_scene_state: dict[str, dict[str, Any]]) -> float:
    target_yaw_rad = float(runtime.yaw_target_from_deg(step.deg))
    if step.object_name == APPLE_NAME or not step.is_rotate:
        return target_yaw_rad
    if step.kind == "pick":
        reference_name = step.object_name
    elif step.support_name is not None:
        reference_name = step.support_name
    else:
        reference_name = None
    if reference_name is None:
        return target_yaw_rad
    return float(runtime.yaw_target_from_deg(get_object_deg(lookup_scene_state, reference_name)))


def should_align_yaw_for_step(runtime, step: TaskStep, lookup_scene_state: dict[str, dict[str, Any]]) -> tuple[bool, float]:
    target_yaw_rad = resolve_step_target_yaw_rad(runtime, step, lookup_scene_state)
    current_yaw_rad = float(runtime.get_live_tcp_pose()[5]) if not runtime.dry_run else float(runtime.home_real[5])
    need_align = bool(step.align_yaw) and (
        abs(runtime.wrap_angle(float(current_yaw_rad) - float(target_yaw_rad))) > float(runtime.default_yaw_exec_tol_rad)
    )
    return need_align, float(target_yaw_rad)


def execute_pick_step(
    runtime,
    step: TaskStep,
    *,
    record: bool,
    frame_idx: int,
    lookup_scene_state: dict[str, dict[str, Any]],
):
    target_xy = np.asarray(step.xy, dtype=np.float64).reshape(2)
    target_z = float(runtime.z_for_pick_level(step.level))
    above_z = float(runtime.min_tcp_z + runtime.approach_z_offset_m)
    step_frames: list[Any] = []
    should_align_yaw, target_yaw_rad = should_align_yaw_for_step(runtime, step, lookup_scene_state)

    if runtime.dry_run:
        dummy_count = 12 + (3 if should_align_yaw else 0)
        sim_pose = real_pose_to_sim(runtime.home_real)
        yaw = runtime.local_exec_yaw_rad
        for idx in range(dummy_count):
            step_frames.append(
                runtime.make_dummy_frame(
                    sim_pose=sim_pose,
                    gripper=1.0 if idx < dummy_count - 3 else 0.0,
                    yaw=yaw,
                    frame_idx=frame_idx + idx,
                )
            )
        return step_frames, frame_idx + len(step_frames)

    print(
        f"    [{step.object_name}] pick @ ({target_xy[0]:.4f}, {target_xy[1]:.4f}), "
        f"level={step.level}, rotate={step.is_rotate}, deg={step.deg:.1f}"
    )
    above_pose = runtime.build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(above_z))
    seg = runtime.record_pose_move(
        above_pose,
        gripper=1.0,
        start_frame_idx=frame_idx,
        record=record,
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)

    if should_align_yaw:
        align_pose = above_pose.copy()
        align_pose[5] = float(target_yaw_rad)
        seg = runtime.record_pose_move(
            align_pose,
            gripper=1.0,
            start_frame_idx=frame_idx,
            record=record,
            semantic_yaw=runtime.local_exec_yaw_rad,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)

    down_pose = runtime.build_pose_from_live_orientation_yaw(float(target_xy[0]), float(target_xy[1]), float(target_z), float(target_yaw_rad))
    seg = runtime.record_pose_move(
        down_pose,
        gripper=1.0,
        start_frame_idx=frame_idx,
        record=record,
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)

    runtime.ensure_gripper_ok(
        runtime.command_gripper_state(0),
        f"close gripper on {step.object_name}",
    )

    lift_pose = runtime.build_pose_from_live_orientation_yaw(float(target_xy[0]), float(target_xy[1]), float(above_z), float(target_yaw_rad))
    seg = runtime.record_pose_move(
        lift_pose,
        gripper=0.0,
        start_frame_idx=frame_idx,
        record=record,
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)
    return step_frames, frame_idx


def execute_place_step(
    runtime,
    step: TaskStep,
    *,
    record: bool,
    frame_idx: int,
    lookup_scene_state: dict[str, dict[str, Any]],
    result_scene_state: dict[str, dict[str, Any]],
):
    target_xy = np.asarray(step.xy, dtype=np.float64).reshape(2)
    target_z = float(runtime.z_for_place_level(step.level))
    above_z = float(runtime.min_tcp_z + runtime.approach_z_offset_m)
    step_frames: list[Any] = []
    should_align_yaw, target_yaw_rad = should_align_yaw_for_step(runtime, step, lookup_scene_state)

    if runtime.dry_run:
        dummy_count = 10 + (3 if should_align_yaw else 0)
        sim_pose = real_pose_to_sim(runtime.home_real)
        yaw = runtime.local_exec_yaw_rad
        for idx in range(dummy_count):
            step_frames.append(
                runtime.make_dummy_frame(
                    sim_pose=sim_pose,
                    gripper=0.0 if idx < dummy_count - 3 else 1.0,
                    yaw=yaw,
                    frame_idx=frame_idx + idx,
                )
            )
        normalized_state = commit_place_state(result_scene_state, step)
        if normalized_state is not None:
            runtime.cleanup_scene_state = normalized_state
        return step_frames, frame_idx + len(step_frames)

    print(
        f"    [{step.object_name}] place @ ({target_xy[0]:.4f}, {target_xy[1]:.4f}), "
        f"level={step.level}, rotate={step.is_rotate}, deg={step.deg:.1f}"
    )
    above_pose = runtime.build_pose_from_live_orientation(float(target_xy[0]), float(target_xy[1]), float(above_z))
    seg = runtime.record_pose_move(
        above_pose,
        gripper=0.0,
        start_frame_idx=frame_idx,
        record=record,
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)

    if should_align_yaw:
        align_pose = above_pose.copy()
        align_pose[5] = float(target_yaw_rad)
        seg = runtime.record_pose_move(
            align_pose,
            gripper=0.0,
            start_frame_idx=frame_idx,
            record=record,
            semantic_yaw=runtime.local_exec_yaw_rad,
        )
        if record:
            step_frames.extend(seg)
        frame_idx += len(seg)

    down_pose = runtime.build_pose_from_live_orientation_yaw(float(target_xy[0]), float(target_xy[1]), float(target_z), float(target_yaw_rad))
    seg = runtime.record_pose_move(
        down_pose,
        gripper=0.0,
        start_frame_idx=frame_idx,
        record=record,
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)
    runtime.ensure_gripper_ok(
        runtime.command_gripper_state(1),
        f"open gripper for {step.object_name}",
    )
    normalized_state = commit_place_state(result_scene_state, step)
    if normalized_state is not None:
        runtime.cleanup_scene_state = normalized_state

    lift_pose = runtime.build_pose_from_live_orientation_yaw(float(target_xy[0]), float(target_xy[1]), float(above_z), float(target_yaw_rad))
    seg = runtime.record_pose_move(
        lift_pose,
        gripper=1.0,
        start_frame_idx=frame_idx,
        record=record,
    )
    if record:
        step_frames.extend(seg)
    frame_idx += len(seg)
    return step_frames, frame_idx


def execute_step_sequence(
    runtime,
    steps: list[TaskStep],
    *,
    record: bool,
    initial_held_object: str | None = None,
    scene_state: dict[str, dict[str, Any]],
    lookup_scene_state: dict[str, dict[str, Any]] | None = None,
    result_scene_state: dict[str, dict[str, Any]] | None = None,
):
    execution_state = scene_state if result_scene_state is None else result_scene_state
    lookup_state = execution_state if lookup_scene_state is None else lookup_scene_state
    frames: list[Any] = []
    frame_idx = 0
    held_object: str | None = initial_held_object
    runtime.begin_task_servo()
    try:
        for step in steps:
            if step.kind == "pick":
                if held_object is not None:
                    raise RuntimeError(f"cannot pick {step.object_name} while holding {held_object}")
                step_frames, frame_idx = execute_pick_step(
                    runtime,
                    step,
                    record=record,
                    frame_idx=frame_idx,
                    lookup_scene_state=lookup_state,
                )
                frames.extend(step_frames)
                scene_detach_top(execution_state, step.object_name)
                held_object = step.object_name
                runtime.runtime_held_object = held_object
            elif step.kind == "place":
                if held_object != step.object_name:
                    raise RuntimeError(
                        f"cannot place {step.object_name}: currently holding {held_object!r}"
                    )
                step_frames, frame_idx = execute_place_step(
                    runtime,
                    step,
                    record=record,
                    frame_idx=frame_idx,
                    lookup_scene_state=lookup_state,
                    result_scene_state=execution_state,
                )
                frames.extend(step_frames)
                held_object = None
                runtime.runtime_held_object = None
            else:
                raise RuntimeError(f"unknown step kind: {step.kind}")
    finally:
        runtime.end_task_servo()
    return frames, held_object


__all__ = [
    "APPLE_NAME",
    "OBJECT_ORDER",
    "PlannerConfig",
    "ObjectState",
    "TaskStep",
    "EpisodePlan",
    "build_pick_prompt",
    "build_place_prompt",
    "build_random_episode_plan",
    "clone_scene_state",
    "commit_place_state",
    "describe_episode",
    "execute_step_sequence",
    "finalize_episode",
    "get_object_deg",
    "get_object_is_rotate",
    "get_object_xy",
    "load_scene_state",
    "PickAndPlaceRecordedEpisode",
    "PickAndPlaceSession",
    "plan_next_episode",
    "prepare_session",
    "sample_initial_object_state",
    "sample_random_table_state",
    "record_episode",
    "restore_session",
    "scene_detach_top",
    "scene_place_object",
    "scene_top_of",
    "should_align_yaw_for_step",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


APPLE_NAME = "apple"
OBJECT_ORDER = ("red", "green", "blue", APPLE_NAME)
PICK_TASK_PROBABILITY = 0.20
NON_ROTATED_TABLE_PLACE_PROBABILITY = 0.70
DEFAULT_ROTATE_DEG_MIN = 12.0
DEFAULT_ROTATE_DEG_MAX = 22.5
LOCAL_CLEAR_INITIAL_RADIUS_M = 0.10
LOCAL_CLEAR_RADIUS_EXPAND_M = 0.15
LOCAL_CLEAR_POINTS_PER_RETRY = 8
LOCAL_CLEAR_MAX_RETRIES = 5


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


@dataclass
class EpisodePlan:
    task_kind: str  # "pick" | "place"
    prompt: str
    source_name: str
    target_name: str | None
    recorded_steps: list[TaskStep]
    post_steps: list[TaskStep]
    scene_after: dict[str, dict[str, Any]]


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
            objects[name] = ObjectState(
                name=name,
                xy=_normalize_xy_tuple(xy),
                is_rotate=bool(raw.get("is_rotate", False)),
                deg=float(raw.get("deg", 0.0)),
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
                deg = float(np.rad2deg(float(raw.get("j6", 0.0))))
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
            note=f"place cleared {upper_name} to table",
        )
    )
    scene.place_on_table(upper_name, tuple(float(v) for v in temp_xy.tolist()), is_rotate=False, deg=0.0)


def _make_pick_step(scene: SceneState, object_name: str, *, note: str) -> TaskStep:
    obj = scene.get(object_name)
    return TaskStep(
        kind="pick",
        object_name=object_name,
        xy=tuple(obj.xy),
        level=scene.depth(object_name),
        is_rotate=bool(obj.is_rotate),
        deg=0.0 if not obj.is_rotate else float(obj.deg),
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
        note=note,
        support_name=support_name if target_name is not None else None,
    )


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
    "load_scene_state",
    "sample_random_table_state",
]
